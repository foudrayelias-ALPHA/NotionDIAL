"""GPT-2 owed cell: clean-context wiki KL for the months-as-target writer swap
(preregistration_monthstab.md, Panel 1, freeze 5adc074).

Reuses the frozen monthswap conditions (25_monthswap.py: month-plane B_M, writer
set, T_mo_all, matched-norm random C_mo_rand_r0) and splits wiki KL into raw
(all positions) and clean (positions whose causal context contains NO month
token), mirroring clocklib.unsup.LineRef's wiki_kl_clean mask. Evaluates at
lambda=0 (full dose) and lambda=1 (sanity, must be ~0). Scores P-MT-KLC(gpt2):
targeted clean KL <= control clean KL.

Usage:
  python 25b_monthswap_cleankl.py --device mps
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.ringlib import DAYS, MONTHS, LM
from clocklib.unsup import array_pca, cloud_plane_power

w18 = import_module("18_writers_any")
w25 = import_module("25_monthswap")
Writers, positions, wiki_batch, LAMBDAS = (
    w18.Writers, w18.positions, w18.wiki_batch, w18.LAMBDAS)

ART = ROOT / "artifacts"


def kl_split(logp, ref_logp, clean_mask):
    """Per-position KL (log_target); return (raw_mean, clean_mean)."""
    kl = F.kl_div(logp.flatten(0, 1), ref_logp.flatten(0, 1),
                  reduction="none", log_target=True).sum(-1).reshape(logp.shape[:2])
    raw = float(kl.mean())
    clean = float(kl[clean_mask].mean()) if clean_mask.any() else float("nan")
    return raw, clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM("gpt2", device=args.device)
    out = ART / "monthswap_gpt2"

    wr = Writers(lm)
    wiki_ids = wiki_batch(lm)
    month_ids = lm.token_ids(MONTHS)
    day_ids = lm.token_ids(DAYS)
    # clean mask: positions whose causal context (this position + all before,
    # via cummax) contains NO month token -> the edit acts only on off-concept
    # collateral there. Mirror of LineRef.wiki_clean.
    tainted_mo = torch.cummax(torch.isin(wiki_ids, torch.tensor(month_ids)), 1).values
    clean_mo = ~tainted_mo.bool()
    tainted_day = torch.cummax(torch.isin(wiki_ids, torch.tensor(day_ids)), 1).values
    clean_day = ~tainted_day.bool()

    n_tok = int(wiki_ids.numel())
    diag = {"wiki_tokens": n_tok,
            "month_token_positions": int(torch.isin(wiki_ids, torch.tensor(month_ids)).sum()),
            "day_token_positions": int(torch.isin(wiki_ids, torch.tensor(day_ids)).sum()),
            "clean_of_months_positions": int(clean_mo.sum()),
            "clean_of_days_positions": int(clean_day.sum())}

    wiki = wiki_ids.to(lm.device)
    # reference logprobs at lambda=1 (unedited)
    with torch.no_grad():
        ref_logp = F.log_softmax(wr.model(input_ids=wiki).logits.float(), dim=-1).cpu()

    # rebuild the two frozen conditions exactly as 25_monthswap does
    month_tpl = positions(lm, MONTHS)
    Hm = np.zeros((len(MONTHS.words), lm.hidden_size))
    with torch.no_grad():
        for enc, wi, pos in month_tpl:
            hs = wr.model(input_ids=enc.to(lm.device),
                          output_hidden_states=True).hidden_states
            Hm[wi] += hs[lm.probe_layer][0, pos].float().cpu().numpy()
    Hm /= len(MONTHS.templates)
    BM, _ = array_pca(Hm, 2)
    P = torch.from_numpy(BM @ BM.T).float()
    all_names = list(wr.orig.keys())

    def plane_mats(names):
        return {nm: wr.arch.removed(wr.orig[nm], P) for nm in names}

    def rand_mats(names, seed):
        g = torch.Generator().manual_seed(seed)
        res = {}
        for nm in names:
            Q, _ = torch.linalg.qr(torch.randn(lm.hidden_size, 2, generator=g))
            M = wr.arch.removed(wr.orig[nm], Q @ Q.T)
            tgt = float(wr.arch.removed(wr.orig[nm], P).norm())
            res[nm] = M * (tgt / max(float(M.norm()), 1e-12))
        return res

    conds = {"T_mo_all": plane_mats(all_names),
             "C_mo_rand_r0": rand_mats(all_names, 0)}

    result = {"model": "gpt2", "freeze": "5adc074", "diagnostics": diag, "conditions": {}}
    for name, mats in conds.items():
        per_lam = {}
        for lam in (0.0, 1.0):
            wr.set_removals(lam, mats)
            with torch.no_grad():
                logp = F.log_softmax(wr.model(input_ids=wiki).logits.float(), dim=-1).cpu()
            raw, clean_mo_kl = kl_split(logp, ref_logp, clean_mo)
            _, clean_day_kl = kl_split(logp, ref_logp, clean_day)
            per_lam[f"lam{lam}"] = {"wiki_kl_raw": round(raw, 6),
                                    "wiki_kl_clean_of_months": round(clean_mo_kl, 6),
                                    "wiki_kl_clean_of_days": round(clean_day_kl, 6)}
        wr.restore()
        result["conditions"][name] = per_lam

    t0 = result["conditions"]["T_mo_all"]["lam0.0"]
    c0 = result["conditions"]["C_mo_rand_r0"]["lam0.0"]
    verdict = t0["wiki_kl_clean_of_months"] <= c0["wiki_kl_clean_of_months"]
    result["P-MT-KLC"] = {
        "targeted_clean_kl": t0["wiki_kl_clean_of_months"],
        "control_clean_kl": c0["wiki_kl_clean_of_months"],
        "targeted_raw_kl": t0["wiki_kl_raw"],
        "control_raw_kl": c0["wiki_kl_raw"],
        "verdict": "PASS" if verdict else "FAIL",
        "rule": "targeted clean-context wiki KL <= control clean-context wiki KL"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "clean_kl.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
