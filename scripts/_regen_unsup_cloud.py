"""Targeted re-run of unsup_gpt2 sweepB: regenerate ONLY the primary targeted
condition (T_num_d1) and one matched-norm random control (C3_random_r0) to
attach the new _N_cloud_B2 capture field. Reproduces part_b's exact condition
construction (seeds, m1_norm scaling) so committed scalars reproduce. The other
four sweepB files (T_num_d2/d4, C3_random_r1/r2) are left untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.ringlib import DAYS, MONTHS, LM
from clocklib.unsup import NUMBERS, LineRef, pca_basis, subspace_mat

u16 = __import__("16_unsupervised")
LAMBDAS, wiki_batch = u16.LAMBDAS, u16.wiki_batch

lm = LM("gpt2", device="mps")
out = ROOT / "artifacts" / "unsup_gpt2" / "sweepB"
W0 = lm.model.get_input_embeddings().weight.detach().cpu()
ids = lm.token_ids(NUMBERS)
assert ids is not None

# EXACT construction from part_b: d1 target + m1_norm-scaled random controls
bases = {d: pca_basis(W0, ids, d) for d in (1, 2, 4)}
conds = {f"T_num_d{d}": subspace_mat(W0, ids, B) for d, (B, _) in bases.items()}
m1_norm = float(conds["T_num_d1"][ids].norm())
for s in range(3):
    g = torch.Generator().manual_seed(s)
    uu = torch.randn(len(ids), generator=g)
    v = torch.randn(W0.shape[1], generator=g)
    M = torch.zeros_like(W0)
    M[ids] = torch.outer(uu, v)
    conds[f"C3_random_r{s}"] = M * (m1_norm / M[ids].norm())

ref = LineRef(lm, NUMBERS, DAYS, MONTHS, wiki_batch(lm), d=4)
lm.model = None
import gc
gc.collect()
torch.mps.empty_cache()

regen = ["T_num_d1", "C3_random_r0"]
for name in regen:
    M = conds[name]
    rows = [{"lam": lam, **ref.measure(ref.W1 + (lam - 1.0) * M)} for lam in LAMBDAS]
    (out / f"{name}.json").write_text(json.dumps(rows))
    r0 = next(r for r in rows if r["lam"] == 0.0)
    print(name, "done; lam0 sub_power",
          round(r0["sub_power"], 6), "cloud_pts", len(r0["_N_cloud_B2"]), flush=True)
