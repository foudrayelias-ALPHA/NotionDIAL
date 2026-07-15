"""Score the local numbers-tab 2-hop answer-manifold sweeps against the frozen
criteria (preregistration_numberstab.md, freeze 5745ec7, Panels 2 & 3). Reads a
sweep dir of condition JSONs and prints per-criterion verdicts (P-NT-M-a..e).

Number task = 2-hop over 10 NON-WRAPPING items (X in one..ten). Day task = 2-hop
over 7 wrapping items. Keys are identical for Llama and Qwen (both use the 2-hop
frame): A_nu_B6_power A_day_B6_power number_2hop_acc/margin day_2hop_acc/margin
number_tokplane_power day_tokplane_power wiki_kl wiki_kl_clean.

Usage:
  python _number_score.py --sweep artifacts/numberctx_llama/sweep --model llama
  python _number_score.py --sweep artifacts/number2hop_qwen/sweep --model qwen
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
    n_number = 10   # non-wrapping 2-hop items (X in one..ten)
    n_day = 7

    T = load(sweep, "T_out_all")
    C0 = load(sweep, "C_out_rand_r0")
    t0, t1 = at(T, 0.0), at(T, 1.0)
    c0 = at(C0, 0.0)

    seg = [r for r in T if 0.0 - 1e-9 <= r["lam"] <= 1.0 + 1e-9]
    seg.sort(key=lambda r: r["lam"])
    lams = [r["lam"] for r in seg]
    nmargins = [r["number_2hop_margin"] for r in seg]
    rho = float(spearmanr(lams, nmargins).correlation)

    day_geo = max(t0["A_day_B6_power"], t0["day_tokplane_power"])
    day_base = at(load(sweep, "T_out_top"), 1.0)["day_2hop_acc"]
    number_base = t1["number_2hop_acc"]

    res = {}
    res["a_kill"] = {"A_nu_B6_power@lam0(T_out_all)": round(t0["A_nu_B6_power"], 4),
                     "thr": "<=0.05",
                     "verdict": "PASS" if t0["A_nu_B6_power"] <= 0.05 else "FAIL"}
    res["b1_day_geom"] = {"A_day_B6@lam0": round(t0["A_day_B6_power"], 4),
                          "day_tokplane@lam0": round(t0["day_tokplane_power"], 4),
                          "max_used": round(day_geo, 4), "thr": ">=0.90",
                          "verdict": "PASS" if day_geo >= 0.90 else "FAIL"}
    dacc0 = round(t0["day_2hop_acc"] * n_day)
    res["b2_day_behav"] = {
        "day_2hop_acc@lam0": f"{dacc0}/{n_day}",
        "day_baseline": f"{round(day_base*n_day)}/{n_day}",
        "thr": "within 1 of baseline",
        "verdict": "PASS" if abs(dacc0 - round(day_base*n_day)) <= 1 else "FAIL"}
    nacc0 = round(t0["number_2hop_acc"] * n_number)
    res["c_number_collapse"] = {
        "number_2hop_acc@lam0": f"{nacc0}/{n_number}", "thr_acc": "<=2/10",
        "spearman_rho(lam,number_margin)": round(rho, 4), "thr_rho": ">=0.95",
        "verdict": "PASS" if (nacc0 <= 2 and rho >= 0.95) else "FAIL"}
    tk, ck = t0["wiki_kl_clean"], c0["wiki_kl_clean"]
    res["d_specificity_cleanKL"] = {
        "targeted_clean@lam0": round(tk, 6), "control_clean@lam0": round(ck, 6),
        "targeted_raw@lam0": round(t0["wiki_kl"], 6),
        "control_raw@lam0": round(c0["wiki_kl"], 6),
        "thr": "targeted_clean <= control_clean",
        "verdict": "PASS" if tk <= ck else "FAIL"}
    cA = c0["A_nu_B6_power"]
    cnacc = round(c0["number_2hop_acc"] * n_number)
    res["e_random_preserves"] = {
        "control_A_nu_B6@lam0": round(cA, 4), "thr_obj": ">=0.7",
        "control_number_2hop_acc@lam0": f"{cnacc}/{n_number}",
        "number_baseline": f"{round(number_base*n_number)}/{n_number}",
        "thr_task": "within 1 of baseline",
        "verdict": "PASS" if (cA >= 0.7
                              and abs(cnacc - round(number_base*n_number)) <= 1) else "FAIL"}

    n_pass = sum(1 for v in res.values() if v["verdict"] == "PASS")
    print(json.dumps({"model": args.model, "sweep": str(sweep),
                      "score": f"{n_pass}/{len(res)}", "criteria": res}, indent=2))


if __name__ == "__main__":
    main()
