"""Step-3 prerequisite: unsupervised gauge-fixing (preregistration_step3.md, 2245a23).

Part A: prior-free PCA planes on the days ring vs Fourier ground truth — recovery
(principal angles) + functional equivalence (lambda sweeps through RingRef).
Part B: no-group NUMBERS concept — LineRef sweep of discovered subspaces.

Usage:
  python 16_unsupervised.py --model gpt2 --part a --device mps
  python 16_unsupervised.py --model meta-llama/Llama-3.2-1B --part b --device mps
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import fourier_plane_basis
from clocklib.ringlib import DAYS, MONTHS, LM, RingRef
from clocklib.unsup import (NUMBERS, LineRef, fourier_keep_mat, pca_basis,
                            principal_cos, subspace_mat)

ART = ROOT / "artifacts"
LAMBDAS = sorted({-0.5, -0.25} | {round(x, 3) for x in np.linspace(0, 1, 21)}
                 | {1.1, 1.25, 1.5})


def wiki_batch(lm: LM) -> torch.Tensor:
    from datasets import load_dataset

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
    return lm.tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64)


def run_sweep(ref, W1: torch.Tensor, conds: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, M in conds.items():
        rows = [{"lam": lam, **ref.measure(W1 + (lam - 1.0) * M)} for lam in LAMBDAS]
        (out / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name} done", flush=True)


def part_a(lm: LM, out: Path) -> None:
    W0 = lm.model.get_input_embeddings().weight.detach().cpu()
    p_ids = lm.token_ids(DAYS)
    E = W0[p_ids].numpy().astype(np.float64)

    B2, rep2 = pca_basis(W0, p_ids, 2)
    B4, rep4 = pca_basis(W0, p_ids, 4)
    B6, rep6 = pca_basis(W0, p_ids, 6)
    # fourier_plane_basis returns (2, d) with basis vectors as ROWS; principal_cos
    # and pca_basis use column conventions (d, k)
    Q1 = fourier_plane_basis(E, 1).T
    Q12 = np.linalg.qr(np.concatenate(
        [fourier_plane_basis(E, k).T for k in (1, 2)], axis=1))[0]
    Qring = np.linalg.qr(np.concatenate(
        [fourier_plane_basis(E, k).T for k in (1, 2, 3)], axis=1))[0]
    recovery = {
        "pca_spectrum": rep6["spectrum"],
        "top2_share": rep2["share"], "top4_share": rep4["share"],
        "cos_B2_k1": principal_cos(B2, Q1),
        "cos_B4_k12": principal_cos(B4, Q12),
        "cos_B6_ring": principal_cos(B6, Qring),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "recovery.json").write_text(json.dumps(recovery, indent=2))
    print(json.dumps({k: v for k, v in recovery.items() if k != "pca_spectrum"},
                     indent=2), flush=True)

    conds = {
        "T_pca2": subspace_mat(W0, p_ids, B2),
        "T_pca6": subspace_mat(W0, p_ids, B6),
        "R_fourier_k1": fourier_keep_mat(W0, p_ids, [1]),
        "R_fourier_ring": fourier_keep_mat(W0, p_ids, [1, 2, 3]),
    }
    ref = RingRef(lm, DAYS, MONTHS, wiki_batch(lm))
    lm.model = None  # free the base copy; ref holds its own untied model
    gc.collect()
    if lm.device == "mps":
        torch.mps.empty_cache()
    run_sweep(ref, ref.W1, conds, out / "sweepA")


def part_b(lm: LM, out: Path) -> None:
    W0 = lm.model.get_input_embeddings().weight.detach().cpu()
    ids = lm.token_ids(NUMBERS)
    assert ids is not None, "number words not single-token for this model"

    bases = {d: pca_basis(W0, ids, d) for d in (1, 2, 4)}
    conds = {f"T_num_d{d}": subspace_mat(W0, ids, B) for d, (B, _) in bases.items()}
    m1_norm = float(conds["T_num_d1"][ids].norm())
    for s in range(3):
        g = torch.Generator().manual_seed(s)
        u = torch.randn(len(ids), generator=g)
        v = torch.randn(W0.shape[1], generator=g)
        M = torch.zeros_like(W0)
        M[ids] = torch.outer(u, v)
        conds[f"C3_random_r{s}"] = M * (m1_norm / M[ids].norm())

    ref = LineRef(lm, NUMBERS, DAYS, MONTHS, wiki_batch(lm), d=4)
    lm.model = None
    gc.collect()
    if lm.device == "mps":
        torch.mps.empty_cache()
    out.mkdir(parents=True, exist_ok=True)
    structure = {"pca_spectrum": bases[4][1]["spectrum"],
                 "shares": {d: rep["share"] for d, (_, rep) in bases.items()},
                 "ref_metrics": ref.ref}
    (out / "structureB.json").write_text(json.dumps(structure, indent=2))
    print(json.dumps(structure["shares"], indent=2), flush=True)
    print(json.dumps(ref.ref, indent=2), flush=True)
    run_sweep(ref, ref.W1, conds, out / "sweepB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--part", required=True, choices=["a", "b"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(args.model, device=args.device)
    out = ART / f"unsup_{args.model.replace('/', '_')}"
    if args.part == "a":
        part_a(lm, out)
    else:
        part_b(lm, out)


if __name__ == "__main__":
    main()
