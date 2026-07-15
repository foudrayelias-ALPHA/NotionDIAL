"""EXPLORATORY METHOD (post-freeze, labeled): gauge-align the W_e decomposition.

Applies an invertible recombination V' = V G, U' = G^{-1} U (reconstruction preserved
EXACTLY: V'U' = VU) chosen so the first 2*n_freq components' read vectors realize the
key-frequency cos/sin modes as least-squares combinations of the EXISTING read vectors.
This is supervised (uses the Fourier basis); its epistemic role is to locate raw SPD's
edit failure in the GAUGE rather than the SPAN: if aligned components achieve
oracle-level selectivity, the mechanism was inside the decomposition all along.

Usage: python 05_gauge_align.py --art artifacts/spd_seed0_s0
Writes: artifacts/<name>_aligned/decomposition.pt (+ alignment_report.json)
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


def align_We(art: dict) -> tuple[dict, dict]:
    V = art["W_e"]["V"].numpy().astype(np.float64)   # (p, C)
    U = art["W_e"]["U"].numpy().astype(np.float64)   # (C, d)
    p, C = V.shape
    E = art["W_e"]["W_target"].T.numpy()             # (p, d)
    kfs = key_freqs(E, frac_thresh=0.02)

    targets, labels = [], []
    for k in kfs:
        t = 2 * np.pi * k * np.arange(p) / p
        for fn, nm in ((np.cos, "cos"), (np.sin, "sin")):
            f = fn(t)
            targets.append(f / np.linalg.norm(f))
            labels.append((k, nm))
    F = np.stack(targets, axis=1)                    # (p, 2*n_freq)

    G_partial, residuals = [], []
    for j in range(F.shape[1]):
        g, res, *_ = np.linalg.lstsq(V, F[:, j], rcond=None)
        G_partial.append(g)
        residuals.append(float(np.linalg.norm(V @ g - F[:, j])))
    G_partial = np.stack(G_partial, axis=1)          # (C, 2*n_freq)

    # Complete to invertible C x C: greedily add identity columns with the largest
    # residual after projection onto the current column span.
    Q, _ = np.linalg.qr(G_partial)
    cols = [G_partial[:, j] for j in range(G_partial.shape[1])]
    basis = Q
    while len(cols) < C:
        R = np.eye(C) - basis @ basis.T
        res_norms = np.linalg.norm(R, axis=0)
        i = int(res_norms.argmax())
        new = R[:, i] / res_norms[i]
        cols.append(new)
        basis = np.concatenate([basis, new[:, None]], axis=1)
    G = np.stack(cols, axis=1)
    cond = float(np.linalg.cond(G))
    assert cond < 1e8, f"G badly conditioned: {cond:.2e}"

    V_new = V @ G
    U_new = np.linalg.solve(G, U)
    recon_err = float(np.abs(V_new @ U_new - V @ U).max())
    assert recon_err < 1e-8, recon_err

    new_art = {k: v for k, v in art.items()}
    new_art["W_e"] = {**art["W_e"],
                      "V": torch.from_numpy(V_new).float(),
                      "U": torch.from_numpy(U_new).float()}
    new_art["aligned"] = {"module": "W_e", "key_freqs": kfs,
                          "component_labels": {i: f"{k}_{nm}" for i, (k, nm) in enumerate(labels)},
                          "clusters": {k: [2 * j, 2 * j + 1] for j, k in enumerate(kfs)}}
    report = {"key_freqs": kfs, "lstsq_residuals": residuals,
              "max_residual": max(residuals), "G_cond": cond, "recon_err": recon_err}
    return new_art, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    args = ap.parse_args()
    art_dir = ROOT / args.art if not Path(args.art).is_absolute() else Path(args.art)
    art = torch.load(art_dir / "decomposition.pt", weights_only=False)
    new_art, report = align_We(art)
    out = art_dir.parent / f"{art_dir.name}_aligned"
    out.mkdir(exist_ok=True)
    torch.save(new_art, out / "decomposition.pt")
    (out / "alignment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # assignments.json in the same schema 04 expects, from the exact aligned clusters
    slim = {"clusters": {"W_e": {"clusters": {
        str(k): {"components": ids, "n": 2}
        for k, ids in new_art["aligned"]["clusters"].items()}},
        "W_in": {"clusters": {}}, "W_out": {"clusters": {}}}}
    (out / "assignments.json").write_text(json.dumps(slim, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
