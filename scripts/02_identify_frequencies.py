"""Phase 2: subcomponent -> frequency map (Result 1a) + distribution + gauge diagnostics.

Frequency signature per component (token-space profile):
  W_e  : read vector V[:, c] (token space directly)
  W_in : token profile E_target @ V[:, c] (component input activation along the a-sweep)
  W_out: write vector U[c, :] (logit token space)

Usage: python 02_identify_frequencies.py --art artifacts/spd_rung0_s0
       python 02_identify_frequencies.py --art artifacts/spd_seed0_s0 --compare artifacts/spd_seed0_s1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.edits import comp_cluster_matrix, comp_matrix, fourier_plane_matrix
from clocklib.spdio import MODULES

FIG = ROOT / "figures"


def signature_vector(art: dict, module: str, c: int) -> np.ndarray:
    a = art[module]
    match module:
        case "W_e":
            return a["V"][:, c].numpy()
        case "W_in":
            E = art["W_e"]["W_target"].T.numpy()  # (p, d)
            return E @ a["V"][:, c].numpy()
        case "W_out":
            return a["U"][c, :].numpy()
        case _:
            raise ValueError(module)


def analyze(art: dict) -> dict:
    p = art["p"]
    out: dict = {"p": p, "modules": {}}
    for m in MODULES:
        C = art[m]["V"].shape[1]
        comps = {}
        for c in range(C):
            s = signature_vector(art, m, c)
            F = np.abs(np.fft.rfft(s)) ** 2
            F[0] = 0.0
            total = float(F.sum())
            k = int(F.argmax()) if total > 0 else 0
            share = float(F[k] / total) if total > 0 else 0.0
            phase = float(np.angle(np.fft.rfft(s)[k])) if k > 0 else 0.0
            comps[c] = {
                "freq": k, "share": share, "phase": phase,
                "norm": float(comp_matrix(art, m, c).norm()),
                "mean_ci": float(art["mean_ci"][m][c]),
                "spectrum": F.tolist(),
            }
        out["modules"][m] = comps
    return out


def cluster_report(analysis: dict, art: dict, norm_frac_thresh: float = 1e-3) -> dict:
    """Frequency-plane clusters per module + quadrature/rank diagnostics.

    Cluster rule (revised 2026-07-01, see decisions.md): dominant-frequency PARTITION —
    every alive component is assigned to its argmax frequency. The originally coded
    share >= 0.5 rule yields empty/singleton clusters on the trained model because SPD
    components mix planes (median dominant share 0.26); the partition is SPD's best
    available plane edit and the sweep quantifies what the mixing costs. Per-cluster
    mean_share records the mixing level.
    """
    report: dict = {}
    for m, comps in analysis["modules"].items():
        total_norm2 = sum(v["norm"] ** 2 for v in comps.values())
        clusters: dict[int, list[int]] = {}
        unassigned: list[int] = []
        for c, v in comps.items():
            if v["norm"] ** 2 < norm_frac_thresh * total_norm2:
                continue  # dead component
            clusters.setdefault(v["freq"], []).append(c)
        cl_stats = {}
        for k, ids in sorted(clusters.items()):
            M = comp_cluster_matrix(art, m, ids)
            sv = torch.linalg.svdvals(M)
            sv2 = (sv**2 / (sv**2).sum()).tolist()
            norms2 = np.array([comps[c]["norm"] ** 2 for c in ids])
            herfindahl = float((norms2 / norms2.sum() ** 1) @ (norms2 / norms2.sum()))
            cl_stats[k] = {
                "components": ids, "n": len(ids),
                "top2_sv_share": float(sum(sv2[:2])),
                "norm_herfindahl": herfindahl,
                "mean_share": float(np.mean([comps[c]["share"] for c in ids])),
                "phases": [round(comps[c]["phase"], 3) for c in ids],
            }
            if m == "W_e":
                P = fourier_plane_matrix(art["W_e"]["W_target"], k)
                cl_stats[k]["fourier_leakage"] = float((M - P).norm() / P.norm())
        report[m] = {"clusters": cl_stats, "unassigned_mixed": unassigned,
                     "n_dead": len(comps) - sum(len(v["components"]) for v in cl_stats.values())
                     - len(unassigned)}
    return report


def heatmap(analysis: dict, name: str) -> None:
    mods = list(analysis["modules"])
    fig, axes = plt.subplots(1, len(mods), figsize=(6 * len(mods), 5))
    for ax, m in zip(np.atleast_1d(axes), mods):
        comps = analysis["modules"][m]
        S = np.array([comps[c]["spectrum"] for c in sorted(comps)])
        S = S / np.maximum(S.sum(axis=1, keepdims=True), 1e-30)
        order = np.argsort([comps[c]["freq"] for c in sorted(comps)])
        ax.imshow(S[order], aspect="auto", cmap="magma")
        ax.set_xlabel("frequency k"); ax.set_ylabel("component (sorted by freq)")
        ax.set_title(m)
    fig.suptitle(f"{name}: component DFT power (row-normalized)")
    fig.savefig(FIG / f"phase2_{name}_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def hungarian_compare(artA: dict, artB: dict) -> dict:
    """Match components across two SPD runs by |cos| of vectorized rank-one matrices."""
    from scipy.optimize import linear_sum_assignment

    out = {}
    for m in MODULES:
        CA = artA[m]["V"].shape[1]
        MA = np.stack([comp_matrix(artA, m, c).numpy().ravel() for c in range(CA)])
        MB = np.stack([comp_matrix(artB, m, c).numpy().ravel() for c in range(CA)])
        MA /= np.maximum(np.linalg.norm(MA, axis=1, keepdims=True), 1e-12)
        MB /= np.maximum(np.linalg.norm(MB, axis=1, keepdims=True), 1e-12)
        cos = np.abs(MA @ MB.T)
        ri, ci = linear_sum_assignment(-cos)
        out[m] = {"mean_matched_abs_cos": float(cos[ri, ci].mean()),
                  "matching": {int(a): int(b) for a, b in zip(ri, ci)}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    art_dir = ROOT / args.art if not Path(args.art).is_absolute() else Path(args.art)
    art = torch.load(art_dir / "decomposition.pt", weights_only=False)
    name = art_dir.name

    analysis = analyze(art)
    report = cluster_report(analysis, art)
    heatmap(analysis, name)

    slim = {"clusters": report,
            "components": {m: {c: {k: v for k, v in d.items() if k != "spectrum"}
                               for c, d in comps.items()}
                           for m, comps in analysis["modules"].items()}}
    if args.compare:
        cmp_dir = ROOT / args.compare if not Path(args.compare).is_absolute() else Path(args.compare)
        artB = torch.load(cmp_dir / "decomposition.pt", weights_only=False)
        slim["seed_stability"] = hungarian_compare(art, artB)

    (art_dir / "assignments.json").write_text(json.dumps(slim, indent=2))
    for m, r in report.items():
        print(f"\n== {m} ==  dead={r['n_dead']} mixed={r['unassigned_mixed']}")
        for k, cs in r["clusters"].items():
            print(f"  freq {k:3d}: n={cs['n']:2d} top2_sv={cs['top2_sv_share']:.3f} "
                  f"herf={cs['norm_herfindahl']:.2f} share={cs['mean_share']:.2f}"
                  + (f" leak={cs['fourier_leakage']:.2f}" if "fourier_leakage" in cs else ""))
    if "seed_stability" in slim:
        for m, s in slim["seed_stability"].items():
            print(f"[stability] {m}: mean |cos| of matched components = "
                  f"{s['mean_matched_abs_cos']:.3f}")


if __name__ == "__main__":
    main()
