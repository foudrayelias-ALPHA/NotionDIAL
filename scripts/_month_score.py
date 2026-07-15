"""Score the local months-tab answer-manifold sweeps against the frozen criteria
(preregistration_monthstab.md, 5adc074). Reads a sweep_dir of condition JSONs and
prints per-criterion verdicts with numbers. Task keys differ Llama vs Qwen:

  llama: A_mo_B6_power A_day_B6_power month_1hop_acc/margin day_1hop_acc/margin
  qwen : A_mo_B6_power A_day_B6_power month_2hop_acc/margin day_2hop_acc/margin

Usage:
  python _month_score.py --sweep artifacts/monthctx_llama/sweep_out --model llama
  python _month_score.py --sweep artifacts/month2hop_qwen/sweep --model qwen
"""
import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr


def load(sweep: Path, name: str):
    return json.loads((sweep / f"{name}.json").read_text())


def at(rows, lam):
    return [r for r in rows if abs(r["lam"] - lam) < 1e-9][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--model", required=True, choices=["llama", "qwen"])
    args = ap.parse_args()
    sweep = Path(args.sweep)
    macc = "month_1hop_acc" if args.model == "llama" else "month_2hop_acc"
    mmar = "month_1hop_margin" if args.model == "llama" else "month_2hop_margin"
    dacc = "day_1hop_acc" if args.model == "llama" else "day_2hop_acc"
    n_month = 12
    n_day = 7

    T = load(sweep, "T_out_all")
    C0 = load(sweep, "C_out_rand_r0")
    t0, t1 = at(T, 0.0), at(T, 1.0)
    c0 = at(C0, 0.0)

    # dose monotonicity: Spearman(lam, month margin) on lam in [0,1]
    seg = [r for r in T if 0.0 - 1e-9 <= r["lam"] <= 1.0 + 1e-9]
    seg.sort(key=lambda r: r["lam"])
    lams = [r["lam"] for r in seg]
    mmargins = [r[mmar] for r in seg]
    rho = float(spearmanr(lams, mmargins).correlation)

    day_geo = max(t0["A_day_B6_power"], t0["day_tokplane_power"])
    day_base = at(load(sweep, "T_out_top"), 1.0)[dacc]  # unedited day baseline
    month_base = t1[macc]

    res = {}
    res["a_kill"] = {"A_mo_B6_power@lam0(T_out_all)": round(t0["A_mo_B6_power"], 4),
                     "thr": "<=0.05",
                     "verdict": "PASS" if t0["A_mo_B6_power"] <= 0.05 else "FAIL"}
    res["b1_day_geom"] = {"A_day_B6@lam0": round(t0["A_day_B6_power"], 4),
                          "day_tokplane@lam0": round(t0["day_tokplane_power"], 4),
                          "max_used": round(day_geo, 4), "thr": ">=0.90",
                          "verdict": "PASS" if day_geo >= 0.90 else "FAIL"}
    dacc0 = t0[dacc]
    res["b2_day_behav"] = {
        f"{dacc}@lam0": f"{round(dacc0*n_day)}/{n_day}",
        "day_baseline": f"{round(day_base*n_day)}/{n_day}",
        "thr": "within 1 of baseline",
        "verdict": "PASS" if abs(round(dacc0*n_day) - round(day_base*n_day)) <= 1 else "FAIL"}
    macc0n = round(t0[macc]*n_month)
    res["c_month_collapse"] = {
        f"{macc}@lam0": f"{macc0n}/{n_month}", "thr_acc": "<=2/12",
        "spearman_rho(lam,month_margin)": round(rho, 4), "thr_rho": ">=0.95",
        "verdict": "PASS" if (macc0n <= 2 and rho >= 0.95) else "FAIL"}
    tk, ck = t0["wiki_kl_clean"], c0["wiki_kl_clean"]
    res["d_specificity_cleanKL"] = {
        "targeted_clean@lam0": round(tk, 6), "control_clean@lam0": round(ck, 6),
        "targeted_raw@lam0": round(t0["wiki_kl"], 6),
        "control_raw@lam0": round(c0["wiki_kl"], 6),
        "thr": "targeted_clean <= control_clean",
        "verdict": "PASS" if tk <= ck else "FAIL"}
    cA = c0["A_mo_B6_power"]
    cmacc = round(c0[macc]*n_month)
    res["e_random_preserves"] = {
        "control_A_mo_B6@lam0": round(cA, 4), "thr_obj": ">=0.7",
        f"control_{macc}@lam0": f"{cmacc}/{n_month}",
        "month_baseline": f"{round(month_base*n_month)}/{n_month}",
        "thr_task": "within 1 of baseline",
        "verdict": "PASS" if (cA >= 0.7 and abs(cmacc - round(month_base*n_month)) <= 1) else "FAIL"}

    n_pass = sum(1 for v in res.values() if v["verdict"] == "PASS")
    out = {"model": args.model, "sweep": str(sweep),
           "score": f"{n_pass}/{len(res)}", "criteria": res}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
