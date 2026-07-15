"""Symmetry gauge-fixing: pin the W_e gauge via the concept's group action alone.

Method: let S be the cyclic shift of token rows (a -> a+1) — pure CONCEPT-level
structure (for days-of-week in an LM: Monday -> Tuesday), independent of the model.
Solve T = argmin ||V T - S V||_F in the decomposition's coefficient space. If the
components span S-invariant mechanism planes, T's complex eigenpairs with |lambda| ~ 1
ARE those planes, and each plane's frequency is read off the eigenvalue angle:
k = round(angle * p / 2pi). No key-frequency list, no phase convention, no Fourier
targets are used in the construction (they appear only in the post-hoc sanity check).

G columns = Re/Im of selected eigenvectors + residual completion; V' = V G,
U' = G^{-1} U preserves the reconstruction exactly.

Usage: python 05u_symmetry_align.py --art artifacts/spd_seed0_s0
Writes: artifacts/<name>_symaligned/{decomposition.pt, assignments.json, symmetry_report.json}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import key_freqs


def symmetry_align(art: dict, unit_tol: float = 0.05) -> tuple[dict, dict]:
    V = art["W_e"]["V"].numpy().astype(np.float64)   # (p, C)
    U = art["W_e"]["U"].numpy().astype(np.float64)   # (C, d)
    p, C = V.shape

    SV = np.roll(V, 1, axis=0)                       # (S V)[a] = V[a-1]
    T, res, *_ = np.linalg.lstsq(V, SV, rcond=None)
    shift_residual = float(np.linalg.norm(V @ T - SV) / np.linalg.norm(SV))

    eigvals, eigvecs = np.linalg.eig(T)
    planes = []
    for i, lam in enumerate(eigvals):
        if lam.imag <= 1e-9:                          # one per conjugate pair; skip real
            continue
        if abs(abs(lam) - 1.0) > unit_tol:            # mechanism planes rotate, |lam|~1
            continue
        k = int(round(np.angle(lam) * p / (2 * np.pi))) % p
        k = min(k, p - k)
        if k == 0:
            continue
        v = eigvecs[:, i]
        rot_err = float(np.linalg.norm(V @ (T @ v) - lam * (V @ v)) /
                        max(np.linalg.norm(V @ v), 1e-12))
        planes.append({"k": k, "abs_lam": float(abs(lam)),
                       "angle": float(np.angle(lam)), "rot_err": rot_err,
                       "g": np.stack([v.real, v.imag], axis=1)})
    planes.sort(key=lambda d: abs(d["abs_lam"] - 1.0))

    cols, labels = [], []
    for pl in planes:
        for j in range(2):
            g = pl["g"][:, j]
            n = np.linalg.norm(g)
            assert n > 1e-12
            cols.append(g / n)
            labels.append(pl["k"])
    G_partial = np.stack(cols, axis=1)

    basis, _ = np.linalg.qr(G_partial)
    all_cols = list(G_partial.T)
    while len(all_cols) < C:
        R = np.eye(C) - basis @ basis.T
        rn = np.linalg.norm(R, axis=0)
        i = int(rn.argmax())
        new = R[:, i] / rn[i]
        all_cols.append(new)
        basis = np.concatenate([basis, new[:, None]], axis=1)
    G = np.stack(all_cols, axis=1)
    cond = float(np.linalg.cond(G))
    assert cond < 1e8, f"G badly conditioned: {cond:.2e}"

    V_new, U_new = V @ G, np.linalg.solve(G, U)
    recon_err = float(np.abs(V_new @ U_new - V @ U).max())
    assert recon_err < 1e-8, recon_err

    clusters: dict[int, list[int]] = {}
    for i, k in enumerate(labels):
        clusters.setdefault(k, []).append(i)

    E = art["W_e"]["W_target"].T.numpy()
    sanity = {"model_key_freqs_POSTHOC_CHECK_ONLY": key_freqs(E, frac_thresh=0.02),
              "discovered_freqs": sorted(clusters)}

    new_art = {k: v for k, v in art.items()}
    new_art["W_e"] = {**art["W_e"], "V": torch.from_numpy(V_new).float(),
                      "U": torch.from_numpy(U_new).float()}
    new_art["aligned"] = {"module": "W_e", "method": "symmetry",
                          "clusters": clusters}
    report = {"shift_residual": shift_residual, "n_planes": len(planes),
              "planes": [{k2: v for k2, v in pl.items() if k2 != "g"} for pl in planes],
              "G_cond": cond, "recon_err": recon_err, **sanity}
    return new_art, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    args = ap.parse_args()
    art_dir = ROOT / args.art if not Path(args.art).is_absolute() else Path(args.art)
    art = torch.load(art_dir / "decomposition.pt", weights_only=False)
    new_art, report = symmetry_align(art)
    out = art_dir.parent / f"{art_dir.name}_symaligned"
    out.mkdir(exist_ok=True)
    torch.save(new_art, out / "decomposition.pt")
    (out / "symmetry_report.json").write_text(json.dumps(report, indent=2))
    slim = {"clusters": {"W_e": {"clusters": {
        str(k): {"components": ids, "n": len(ids)}
        for k, ids in new_art["aligned"]["clusters"].items()}},
        "W_in": {"clusters": {}}, "W_out": {"clusters": {}}}}
    (out / "assignments.json").write_text(json.dumps(slim, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "planes"}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
