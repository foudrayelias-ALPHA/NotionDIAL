"""Score preregistration_writers.md predictions P-W1..P-W6 from writers_gpt2/."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "writers_gpt2"


def main() -> None:
    at = json.loads((OUT / "attribution.json").read_text())
    sweeps = {p.stem: {r["lam"]: r for r in json.loads(p.read_text())}
              for p in sorted((OUT / "sweep").glob("*.json"))}
    a = at["alphas"]
    wall = sweeps["T_wo_all"]
    emb_kl0 = sweeps["T_emb_pca2"][0.0]["wiki_kl"]
    s = {
        "P-W1": {"alpha_emb": a["emb"], "pass": a["emb"] >= 0.5},
        "P-W2": {"top_writers": {k: round(v, 3) for k, v in list(a.items())[:4]},
                 "pass": any(abs(v) >= 0.05 for k, v in a.items() if k != "emb")},
        "P-W3": {"rho_pred": at["rho_pred"],
                 "measured_lam0": wall[0.0]["day_plane_power"],
                 "deviation": abs(wall[0.0]["day_plane_power"] - at["rho_pred"]),
                 "pass": abs(wall[0.0]["day_plane_power"] - at["rho_pred"]) <= 0.15},
        "P-W4": {"month_plane_lam0": wall[0.0]["month_plane_power"],
                 "emb_day_k1": wall[0.0]["emb_day_k1_power"],
                 "pass": 0.80 <= wall[0.0]["month_plane_power"] <= 1.05
                 and abs(wall[0.0]["emb_day_k1_power"] - 1.0) < 1e-6},
        "P-W5": {"kl_writers": wall[0.0]["wiki_kl"], "kl_emb_edit": emb_kl0,
                 "pass": emb_kl0 < wall[0.0]["wiki_kl"] <= 0.5},
        "P-W6": {"succ_acc_lam0": wall[0.0]["succ_acc"],
                 "margins": [wall[1.0]["succ_margin_mean"],
                             wall[0.0]["succ_margin_mean"]],
                 "pass": wall[0.0]["succ_acc"] > 1 / 7 + 1e-9},
        "conditions_lam0": {
            name: {k: round(sw[0.0][k], 4)
                   for k in ("day_plane_power", "month_plane_power", "succ_acc",
                             "succ_margin_mean", "wiki_kl")}
            for name, sw in sweeps.items()},
    }
    for k, v in s.items():
        if "pass" in v:
            print(f"{k:6s} {'PASS' if v['pass'] else 'FAIL'} "
                  f"{ {a_: b for a_, b in v.items() if a_ != 'pass'} }")
    (OUT / "writers_verdict.json").write_text(json.dumps(s, indent=2))
    print("\nconditions at lam=0:")
    for name, row in s["conditions_lam0"].items():
        print(f"  {name:16s} {row}")


if __name__ == "__main__":
    main()
