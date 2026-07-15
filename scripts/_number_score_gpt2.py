"""Score the GPT-2 numberswap panel against the frozen criteria
(preregistration_numberstab.md, Panel 1, freeze 5745ec7). Reads the sweep JSONs
(T_nu_all, C_nu_rand_r0) and prints per-criterion verdicts (P-NT-a..e) with
numbers. Behavior column = run-up counting (9 items).

Usage:
  python _number_score_gpt2.py --sweep artifacts/numberswap_gpt2/sweep
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
    args = ap.parse_args()
    sweep = Path(args.sweep)
    n_count = 9   # run-up counting items (i=2..n-2 for 12 numbers)

    T = load(sweep, "T_nu_all")
    C0 = load(sweep, "C_nu_rand_r0")
    t0, t1 = at(T, 0.0), at(T, 1.0)
    c0 = at(C0, 0.0)

    # dose monotonicity: Spearman(lam, counting margin) on lam in [0,1]
    seg = [r for r in T if 0.0 - 1e-9 <= r["lam"] <= 1.0 + 1e-9]
    seg.sort(key=lambda r: r["lam"])
    lams = [r["lam"] for r in seg]
    cmargins = [r["count_margin"] for r in seg]
    rho = float(spearmanr(lams, cmargins).correlation)

    count_base = round(t1["count_acc"] * n_count)
    res = {}
    res["a_number_kill"] = {
        "number_plane_power@lam0(T_nu_all)": round(t0["number_plane_power"], 4),
        "thr": "<=0.05",
        "verdict": "PASS" if t0["number_plane_power"] <= 0.05 else "FAIL"}
    res["b_day_month_spared"] = {
        "day_plane_power@lam0": round(t0["day_plane_power"], 4),
        "month_plane_power@lam0": round(t0["month_plane_power"], 4),
        "thr": "both >=0.90",
        "verdict": "PASS" if (t0["day_plane_power"] >= 0.90
                              and t0["month_plane_power"] >= 0.90) else "FAIL"}
    cnt0 = round(t0["count_acc"] * n_count)
    res["c_counting_collapse"] = {
        "count_acc@lam0": f"{cnt0}/{n_count}", "thr_acc": "<=2/9",
        "count_baseline": f"{count_base}/{n_count}",
        "spearman_rho(lam,count_margin)": round(rho, 4), "thr_rho": ">=0.95",
        "verdict": "PASS" if (cnt0 <= 2 and rho >= 0.95) else "FAIL"}
    tk, ck = t0["wiki_kl_clean"], c0["wiki_kl_clean"]
    res["d_specificity_cleanKL"] = {
        "targeted_clean@lam0": round(tk, 6), "control_clean@lam0": round(ck, 6),
        "targeted_raw@lam0": round(t0["wiki_kl"], 6),
        "control_raw@lam0": round(c0["wiki_kl"], 6),
        "thr": "targeted_clean <= control_clean",
        "verdict": "PASS" if tk <= ck else "FAIL"}
    cnum = round(c0["number_plane_power"], 4)
    ccnt = round(c0["count_acc"] * n_count)
    res["e_random_preserves"] = {
        "control_number_plane@lam0": cnum, "thr_obj": ">=0.7",
        "control_count_acc@lam0": f"{ccnt}/{n_count}",
        "count_baseline": f"{count_base}/{n_count}",
        "thr_task": "within 1 of baseline",
        "verdict": "PASS" if (cnum >= 0.7
                              and abs(ccnt - count_base) <= 1) else "FAIL"}

    n_pass = sum(1 for v in res.values() if v["verdict"] == "PASS")
    print(json.dumps({"model": "gpt2", "panel": "numberswap", "sweep": str(sweep),
                      "score": f"{n_pass}/{len(res)}", "criteria": res}, indent=2))


if __name__ == "__main__":
    main()
