"""Continuous edit families. Every family returns full weight dicts (target shapes)
and includes the fixed delta, so lambda=1 reproduces the target model exactly:
W(lam) = W_target + (lam - 1) * (edited term), with W_target = sum_c M_c + delta.
"""

import numpy as np
import torch
from torch import Tensor


def comp_matrix(art: dict, module: str, c: int) -> Tensor:
    """Rank-one matrix of component c in the module's target weight shape."""
    a = art[module]
    M = torch.outer(a["V"][:, c], a["U"][c, :])
    return M if M.shape == a["W_target"].shape else M.T


def comp_cluster_matrix(art: dict, module: str, comp_ids: list[int]) -> Tensor:
    a = art[module]
    M = a["V"][:, comp_ids] @ a["U"][comp_ids, :]
    return M if M.shape == a["W_target"].shape else M.T


def target_weights(art: dict, modules: list[str]) -> dict[str, Tensor]:
    return {m: art[m]["W_target"].clone() for m in modules}


def scaled_component_weights(art: dict, modules: list[str],
                             comp_ids_by_module: dict[str, list[int]], lam: float
                             ) -> dict[str, Tensor]:
    """Scale the selected components by lam, leave everything else (incl. delta) fixed."""
    W = target_weights(art, modules)
    for m, ids in comp_ids_by_module.items():
        if ids:
            W[m] = W[m] + (lam - 1.0) * comp_cluster_matrix(art, m, ids)
    return W


def random_rank1_weights(art: dict, modules: list[str], module: str, lam: float,
                         frob_norm: float, seed: int) -> dict[str, Tensor]:
    """C3: perturb `module` along a fixed random rank-one direction of norm `frob_norm`."""
    g = torch.Generator().manual_seed(seed)
    shape = art[module]["W_target"].shape
    u = torch.randn(shape[0], generator=g)
    v = torch.randn(shape[1], generator=g)
    M = torch.outer(u, v)
    M = M * (frob_norm / M.norm())
    W = target_weights(art, modules)
    W[module] = W[module] + (lam - 1.0) * M
    return W


def fourier_plane_matrix(W_e_weight: Tensor, k: int) -> Tensor:
    """Frequency-k plane of the embedding, as a matrix in W_e's (d, p) shape.

    Token axis of W_e is the INPUT axis (columns); E = W_e.weight.T has token rows.
    """
    E = W_e_weight.T.numpy()
    F = np.fft.rfft(E, axis=0)
    keep = np.zeros_like(F)
    keep[k] = F[k]
    E_k = np.fft.irfft(keep, n=E.shape[0], axis=0)
    return torch.from_numpy(E_k.T.copy()).float()


def oracle_fourier_weights(art: dict, modules: list[str], k: int, lam: float
                           ) -> dict[str, Tensor]:
    """C4: scale frequency k's plane of the embedding directly in the Fourier basis."""
    W = target_weights(art, modules)
    W["W_e"] = W["W_e"] + (lam - 1.0) * fourier_plane_matrix(art["W_e"]["W_target"], k)
    return W


def svd_plane_weights(art: dict, modules: list[str], module: str, sv_ids: list[int],
                      lam: float) -> dict[str, Tensor]:
    """C6: scale the selected singular directions of `module`."""
    Wt = art[module]["W_target"]
    U, S, Vh = torch.linalg.svd(Wt, full_matrices=False)
    M = (U[:, sv_ids] * S[sv_ids]) @ Vh[sv_ids, :]
    W = target_weights(art, modules)
    W[module] = W[module] + (lam - 1.0) * M
    return W


def gauge_rotated_art(art: dict, module: str, pairs: list[tuple[int, int]],
                      seed: int) -> dict:
    """C5: rotate each (c1, c2) pair's read/write vectors within their span.

    V' = V R, U' = R^T U per pair. The pair's SUMMED contribution is exactly unchanged
    (reconstruction identical); the individual rank-one components rotate to a new gauge.
    """
    g = np.random.default_rng(seed)
    new = {k: v for k, v in art.items()}
    a = art[module]
    V, U = a["V"].clone(), a["U"].clone()
    angles = {}
    for c1, c2 in pairs:
        th = float(g.uniform(0, 2 * np.pi))
        angles[f"{c1},{c2}"] = th
        R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).float()
        V[:, [c1, c2]] = V[:, [c1, c2]] @ R
        U[[c1, c2], :] = R.T @ U[[c1, c2], :]
    new[module] = {**a, "V": V, "U": U}
    new["gauge_angles"] = angles
    check = comp_cluster_matrix(new, module, [c for pr in pairs for c in pr]) - \
        comp_cluster_matrix(art, module, [c for pr in pairs for c in pr])
    assert check.norm() < 1e-4 * a["W_target"].norm(), "gauge rotation changed the sum"
    return new
