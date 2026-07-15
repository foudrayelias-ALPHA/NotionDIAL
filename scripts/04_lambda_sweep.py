"""Phases 4+5: lambda-sweep for the targeted edits and every control condition.

Conditions (see decisions.md and preregistration.md):
  T_emb_plane   scale the W_e frequency-K component cluster (PRIMARY, gauge-invariant)
  T_stack_plane scale frequency-K clusters in W_e + W_in + W_out together
  T_single      scale the single top-norm component of the W_e K-cluster (SECONDARY)
  C1_null_plane scale the W_e cluster of a different frequency K2 (norm-comparable)
  C3_random_rN  random rank-one in W_e, Frobenius-matched to the K cluster (3 seeds)
  C4_oracle     Fourier-basis ablation of plane K in W_e (the Nanda-style edit)
  C5_gauge_gN   gauge-rotate the top-2 K-cluster pair, re-run the single-component edit
  C6_svd        scale the SVD singular-pair of W_e whose right vectors carry frequency K

C2 (untargeted-frequency specificity) is not a separate run: every metric is computed
per frequency in every condition.

Usage: python 04_lambda_sweep.py --model rung0 --art artifacts/spd_rung0_s0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.edits import (
    comp_cluster_matrix,
    gauge_rotated_art,
    oracle_fourier_weights,
    random_rank1_weights,
    scaled_component_weights,
    svd_plane_weights,
)
from clocklib.fourier import dominant_freq, key_freqs
from clocklib.spdio import MODULES
from clocklib.sweep import FrozenReference, fold, run_condition

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module

load_target = import_module("01_decompose_and_faithfulness").load_target


def load_clusters(art_dir: Path) -> dict[str, dict[int, list[int]]]:
    slim = json.loads((art_dir / "assignments.json").read_text())
    return {
        m: {int(k): v["components"] for k, v in slim["clusters"][m]["clusters"].items()}
        for m in MODULES
    }


def cluster_norm2(art: dict, module: str, ids: list[int]) -> float:
    return float(comp_cluster_matrix(art, module, ids).norm() ** 2)


def svd_pair_for_freq(art: dict, K: int, p: int) -> list[int]:
    W = art["W_e"]["W_target"]
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    hits = [i for i in range(Vh.shape[0])
            if S[i] > 1e-6 and dominant_freq(Vh[i].numpy())[0] == fold(K, p)]
    return hits[:2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--art", required=True)
    ap.add_argument("--freq", type=int, default=None)
    ap.add_argument("--conditions", default="all")
    args = ap.parse_args()

    art_dir = ROOT / args.art if not Path(args.art).is_absolute() else Path(args.art)
    art = torch.load(art_dir / "decomposition.pt", weights_only=False)
    clusters = load_clusters(art_dir)
    target = load_target(args.model)
    p = target.cfg.p

    E_t = target.embedding_matrix().detach().numpy()
    kfs = key_freqs(E_t, frac_thresh=0.02)
    ref = FrozenReference(target, kfs)

    emb_clusters = clusters["W_e"]
    assert emb_clusters, "no W_e frequency clusters found"
    by_norm = sorted(emb_clusters, key=lambda k: -cluster_norm2(art, "W_e", emb_clusters[k]))
    K = args.freq if args.freq is not None else by_norm[0]
    assert K in emb_clusters, f"freq {K} has no W_e cluster (have {sorted(emb_clusters)})"
    K_ids = emb_clusters[K]
    K_norm = float(comp_cluster_matrix(art, "W_e", K_ids).norm())
    K2 = next((k for k in by_norm if k != K), None)

    out_dir = ROOT / "artifacts" / f"sweep_{args.model}_{art_dir.name.split('_')[-1]}_K{K}"
    meta = {"model": args.model, "art": str(art_dir), "K": K, "K_ids": K_ids,
            "K2": K2, "key_freqs": kfs, "K_cluster_frob": K_norm}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"target freq K={K} cluster={K_ids} |M|_F={K_norm:.3f}; null K2={K2}")

    conds: dict = {}
    conds["T_emb_plane"] = lambda lam: scaled_component_weights(
        art, MODULES, {"W_e": K_ids}, lam)
    stack_ids = {m: clusters[m].get(K, []) for m in MODULES}
    conds["T_stack_plane"] = lambda lam: scaled_component_weights(art, MODULES, stack_ids, lam)
    top_comp = max(K_ids, key=lambda c: comp_cluster_matrix(art, "W_e", [c]).norm())
    conds["T_single"] = lambda lam: scaled_component_weights(
        art, MODULES, {"W_e": [top_comp]}, lam)

    if K2 is not None:
        K2_ids = emb_clusters[K2]
        conds["C1_null_plane"] = lambda lam: scaled_component_weights(
            art, MODULES, {"W_e": K2_ids}, lam)
    for s in range(3):
        conds[f"C3_random_r{s}"] = (
            lambda lam, s=s: random_rank1_weights(art, MODULES, "W_e", lam, K_norm, seed=s))
    conds["C4_oracle"] = lambda lam: oracle_fourier_weights(art, MODULES, fold(K, p), lam)

    if len(K_ids) >= 2:
        pair = sorted(K_ids, key=lambda c: -comp_cluster_matrix(art, "W_e", [c]).norm())[:2]
        for s in range(3):
            g_art = gauge_rotated_art(art, "W_e", [tuple(pair)], seed=s)
            conds[f"C5_gauge_g{s}"] = (
                lambda lam, g_art=g_art: scaled_component_weights(
                    g_art, MODULES, {"W_e": [pair[0]]}, lam))
    sv_pair = svd_pair_for_freq(art, K, p)
    if len(sv_pair) == 2:
        conds["C6_svd"] = lambda lam: svd_plane_weights(art, MODULES, "W_e", sv_pair, lam)
        meta["svd_pair"] = sv_pair
        (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2))

    selected = list(conds) if args.conditions == "all" else args.conditions.split(",")
    for name in selected:
        run_condition(name, conds[name], ref, out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
