"""Score preregistration_step3.md predictions P-U1..P-U8 from unsup_* artifacts.

Writes artifacts/unsup_verdict.json and prints a per-model scorecard.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODELS = ["gpt2", "meta-llama_Llama-3.2-1B"]


def load_sweep(d: Path) -> dict:
    return {p.stem: {r["lam"]: r for r in json.loads(p.read_text())}
            for p in sorted(d.glob("*.json"))}


def rms_delta(a: dict, b: dict, key: str) -> float:
    lams = [l for l in a if 0.0 <= l <= 1.0]
    return float(np.sqrt(np.mean([(a[l][key] - b[l][key]) ** 2 for l in lams])))


def score_model(slug: str) -> dict:
    out = ART / f"unsup_{slug}"
    rec = json.loads((out / "recovery.json").read_text())
    A = load_sweep(out / "sweepA")
    st = json.loads((out / "structureB.json").read_text())
    B = load_sweep(out / "sweepB")

    p6, rring = A["T_pca6"], A["R_fourier_ring"]
    p2, rk1 = A["T_pca2"], A["R_fourier_k1"]
    d4 = B["T_num_d4"]
    lams01 = sorted(l for l in d4 if 0.0 <= l <= 1.0)
    c3_at0 = [B[f"C3_random_r{s}"][0.0] for s in range(3)]

    s = {
        "P-U1": {"cos_B2_k1": rec["cos_B2_k1"],
                 "pass": bool(np.mean(rec["cos_B2_k1"]) >= 0.80)},
        "P-U2": {"min_cos_B6_ring": min(rec["cos_B6_ring"]),
                 "pass": min(rec["cos_B6_ring"]) >= 0.999},
        "P-U3a": {"pca6_lam0_k123": [p6[0.0][f"p_k{k}_power"] for k in (1, 2, 3)],
                  "pass": all(p6[0.0][f"p_k{k}_power"] <= 0.01 for k in (1, 2, 3))},
        "P-U3b": {"rms": rms_delta(p6, rring, "p_k1_power"),
                  "pass": rms_delta(p6, rring, "p_k1_power") <= 0.05},
        "P-U3c": {"acc": [p6[0.0]["succ_acc"], rring[0.0]["succ_acc"]],
                  "d_margin": abs(p6[0.0]["succ_margin_mean"]
                                  - rring[0.0]["succ_margin_mean"]),
                  "pass": p6[0.0]["succ_acc"] == rring[0.0]["succ_acc"]
                  and abs(p6[0.0]["succ_margin_mean"]
                          - rring[0.0]["succ_margin_mean"]) <= 0.2},
        "P-U4a": {"pca2_lam0_k1": p2[0.0]["p_k1_power"],
                  "pass": p2[0.0]["p_k1_power"] <= 0.25},
        "P-U4b": {"rms": rms_delta(p2, rk1, "p_k1_power"),
                  "pass": rms_delta(p2, rk1, "p_k1_power") <= 0.10},
        "P-U5": {"days_edit_cross": p6[0.0]["cross_k1_power"],
                 "days_edit_kl": p6[0.0]["wiki_kl"],
                 "num_edit_cross": [d4[0.0]["cross_a_k1_power"],
                                    d4[0.0]["cross_b_k1_power"]],
                 "num_edit_kl": d4[0.0]["wiki_kl"],
                 "num_edit_kl_clean": d4[0.0].get("wiki_kl_clean"),
                 "pass": abs(p6[0.0]["cross_k1_power"] - 1) <= 0.01
                 and p6[0.0]["wiki_kl"] <= 0.01
                 and abs(d4[0.0]["cross_a_k1_power"] - 1) <= 0.01
                 and abs(d4[0.0]["cross_b_k1_power"] - 1) <= 0.01
                 and d4[0.0]["wiki_kl"] <= 0.01},
        "P-U6": {"top4_share": st["shares"]["4"],
                 "emb_spearman_ref": st["ref_metrics"]["emb_spearman"],
                 "pass": st["shares"]["4"] >= 0.50
                 and abs(st["ref_metrics"]["emb_spearman"]) >= 0.80},
        "P-U7a": {"sub_power_lam0": d4[0.0]["sub_power"],
                  "pass": d4[0.0]["sub_power"] <= 0.01},
        "P-U7b": {"rho": float(spearmanr(lams01,
                                         [d4[l]["h_sub_power"] for l in lams01])[0]),
                  "h_sub_lam0": d4[0.0]["h_sub_power"],
                  "pass": float(spearmanr(lams01, [d4[l]["h_sub_power"]
                                                   for l in lams01])[0]) >= 0.9
                  and d4[0.0]["h_sub_power"] <= 0.5},
        "P-U8": {"c3_sub": [r["sub_power"] for r in c3_at0],
                 "c3_h_sub": [r["h_sub_power"] for r in c3_at0],
                 "pass": all(r["sub_power"] >= 0.5 and r["h_sub_power"] >= 0.7
                             for r in c3_at0)},
        "behavioral": {"d_margin": abs(d4[0.0]["succ_margin_mean"]
                                       - d4[1.0]["succ_margin_mean"]),
                       "acc": [d4[1.0]["succ_acc"], d4[0.0]["succ_acc"]],
                       "h_spearman": [d4[1.0]["h_spearman"], d4[0.0]["h_spearman"]]},
    }
    return s


def main() -> None:
    verdict = {}
    for slug in MODELS:
        s = score_model(slug)
        verdict[slug] = s
        print(f"\n=== {slug} ===")
        for k, v in s.items():
            flag = "" if "pass" not in v else ("PASS" if v["pass"] else "FAIL")
            print(f"{k:12s} {flag:4s} "
                  f"{ {a: b for a, b in v.items() if a != 'pass'} }")
    both = all(v["pass"] for slug in MODELS for k, v in verdict[slug].items()
               if "pass" in v)
    margin_hit = any(verdict[s]["behavioral"]["d_margin"] >= 0.3 for s in MODELS)
    verdict["_aggregate"] = {"all_frozen_cells_pass": both,
                             "behavioral_weak_prediction": margin_hit}
    print(f"\nall frozen cells pass: {both} | "
          f"behavioral weak prediction (>=1 model d_margin>=0.3): {margin_hit}")
    (ART / "unsup_verdict.json").write_text(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
