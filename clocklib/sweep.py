"""Lambda-sweep engine: run edit families over the grid, measure frozen-basis metrics.

THE FIXED-BASIS RULE: `FrozenReference` is constructed once from the lambda=1 model;
every projection plane, probe, and reference prediction inside it is reused verbatim at
every lambda. The only current-cloud (refit) metrics are the topology detectors
(monodromy, full-cloud PH), labeled as such in clocklib.geometry.
"""

import json
from pathlib import Path

import numpy as np
import torch

from clocklib.fourier import fourier_plane_basis, freq_power, plane_coords
from clocklib.geometry import (
    FrozenProbe,
    h1_max_lifetime,
    h1_noise_floor,
    participation_ratio,
    plane_stats,
    procrustes_monodromy,
    twonn_id,
)
from clocklib.model import AdderConfig, ModAdder2Hot, all_pairs, two_hot

DEFAULT_LAMBDAS = sorted(
    {-0.5, -0.25} | {round(x, 3) for x in np.linspace(0, 1, 21)} | {1.1, 1.25, 1.5}
)


def fold(k: int, p: int) -> int:
    k = k % p
    return min(k, p - k)


def build_model(cfg: AdderConfig, weights: dict[str, torch.Tensor]) -> ModAdder2Hot:
    model = ModAdder2Hot(cfg)
    with torch.no_grad():
        for m, W in weights.items():
            getattr(model, m).weight.copy_(W)
    return model.eval()


@torch.no_grad()
def clouds(model: ModAdder2Hot, inputs: torch.Tensor, p: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """(E cloud (p,d), hidden token cloud (p,h) marginalized over b, full logits)."""
    E = model.embedding_matrix().numpy()
    H_full = model.hidden(inputs)
    H = H_full.reshape(p, p, -1).mean(dim=1).numpy()
    logits = model.W_out(H_full)
    return E, H, logits


class FrozenReference:
    """Everything fit at lambda=1: plane bases, probes, noise floors, reference logits."""

    def __init__(self, target: ModAdder2Hot, key_freqs: list[int]):
        self.cfg = target.cfg
        p = self.cfg.p
        self.p, self.key_freqs = p, key_freqs
        tokens, self.labels = all_pairs(p)
        self.inputs = two_hot(tokens, p)
        E, H, logits = clouds(target, self.inputs, p)
        self.argmax_ref = logits.argmax(-1)
        top2 = logits.topk(2, dim=-1).values
        self.robust = (top2[:, 0] - top2[:, 1]) > 0.01 * logits.std()
        self.Q_emb = {k: fourier_plane_basis(E, fold(k, p)) for k in key_freqs}
        self.Q_hid = {}
        for k in key_freqs:
            for kk in {fold(k, p), fold(2 * k, p)}:
                if kk > 0 and kk not in self.Q_hid and freq_power(H)[kk] > 1e-10:
                    self.Q_hid[kk] = fourier_plane_basis(H, kk)
        self.probes = {k: FrozenProbe(H, k, p) for k in key_freqs}
        self.h1_floor_emb = {k: h1_noise_floor(plane_coords(E, self.Q_emb[k])) for k in key_freqs}
        self.h1_floor_hid = {kk: h1_noise_floor(plane_coords(H, Q)) for kk, Q in self.Q_hid.items()}
        self.emb_power_ref = freq_power(E)
        self.hid_power_ref = freq_power(H)


@torch.no_grad()
def measure(model: ModAdder2Hot, ref: FrozenReference) -> dict:
    p = ref.p
    E, H, logits = clouds(model, ref.inputs, p)
    preds = logits.argmax(-1)
    met: dict = {
        "acc": float((preds == ref.labels).float().mean()),
        "agree": float((preds == ref.argmax_ref).float().mean()),
        "robust_agree": float((preds == ref.argmax_ref)[ref.robust].float().mean()),
        "twonn_hid": twonn_id(H),
        "pr_hid": participation_ratio(H),
        **procrustes_monodromy(H),
    }
    emb_p, hid_p = freq_power(E), freq_power(H)
    for k in ref.key_freqs:
        kb = fold(k, p)
        met[f"emb_power_k{k}"] = float(emb_p[kb] / max(ref.emb_power_ref[kb], 1e-30))
        xy = plane_coords(E, ref.Q_emb[k])
        for name, v in plane_stats(xy).items():
            met[f"emb_{name}_k{k}"] = v
        met[f"emb_h1_k{k}"] = h1_max_lifetime(xy)
        met[f"emb_h1_floor_k{k}"] = ref.h1_floor_emb[k]
        met[f"probe_err_k{k}"] = ref.probes[k].angular_error(H)
        band = {kb, fold(2 * k, p)} - {0}
        met[f"hid_band_power_k{k}"] = float(
            sum(hid_p[b] for b in band) / max(sum(ref.hid_power_ref[b] for b in band), 1e-30)
        )
    for kk, Q in ref.Q_hid.items():
        xy = plane_coords(H, Q)
        for name, v in plane_stats(xy).items():
            met[f"hid_{name}_b{kk}"] = v
        met[f"hid_h1_b{kk}"] = h1_max_lifetime(xy)
        met[f"hid_h1_floor_b{kk}"] = ref.h1_floor_hid[kk]
    return met, E, H, preds


def run_condition(name: str, weights_fn, ref: FrozenReference, out_dir: Path,
                  lambdas: list[float] | None = None, save_clouds: bool = True) -> list[dict]:
    """weights_fn(lam) -> {module: W}. Writes {name}.json (+ {name}_clouds.npz)."""
    lambdas = lambdas or DEFAULT_LAMBDAS
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cloud_store = [], {}
    for lam in lambdas:
        model = build_model(ref.cfg, weights_fn(lam))
        met, E, H, preds = measure(model, ref)
        rows.append({"lam": lam, **met})
        if save_clouds:
            cloud_store[f"E_{lam}"] = E.astype(np.float16)
            cloud_store[f"H_{lam}"] = H.astype(np.float16)
            cloud_store[f"preds_{lam}"] = preds.numpy().astype(np.int16)
    (out_dir / f"{name}.json").write_text(json.dumps(rows))
    if save_clouds:
        np.savez_compressed(out_dir / f"{name}_clouds.npz", **cloud_store)
    print(f"[sweep] {name}: {len(lambdas)} lambdas done", flush=True)
    return rows
