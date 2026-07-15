"""Qwen2.5-1.5B month 2-hop answer manifold mirror (preregistration_monthstab.md,
Panel 3, freeze 5adc074).

Full mirror of 20_qwen2hop.py's few-shot 2-hop output-coordinate B6 pipeline with
MONTHS as target and DAYS spared. Answer cloud A2_mo = final pre-norm residual of
the frozen few-shot 2-hop MONTH frame (want X+2); B6_mo = top-6 prior-free PCA of
the centered month rows of the OUTPUT embedding (tied -> input rows). The DAY B6
2-hop answer + layer-19 day-token plane + the DAY few-shot 2-hop task are the
frozen SPARED readouts. Clean-context wiki-KL split (no month token in context)
for the specificity gate. Frozen basis lambda=1, 26-point grid.

Usage:
  python 27_month2hop.py --stage attrib --device mps
  python 27_month2hop.py --stage sweep  --device mps
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
w19 = import_module("19_contextual")
w20 = import_module("20_qwen2hop")
w26 = import_module("26_monthctx")

MODEL = "Qwen/Qwen2.5-1.5B"
# month analogs of 20_qwen2hop's day frames (same wording, months substituted)
FEWSHOT_MO = ("Let's do some month math. Two months after January is March. "
              "Two months after October is December. Two months after {} is")
FEWSHOT_DAY = ("Let's do some day of the week math. Two days after Monday is "
               "Wednesday. Two days after Friday is Sunday. Two days after {} is")
OUT = ROOT / "artifacts" / "month2hop_qwen"
LAMBDAS = w18.LAMBDAS
SHIFT = 2


def out_rows(lm: LM) -> tuple[torch.Tensor, bool]:
    return w20.out_rows(lm)


def month_b6(lm: LM) -> np.ndarray:
    ids = lm.token_ids(MONTHS)
    Wout, _ = out_rows(lm)
    X = Wout[ids].numpy().astype(np.float64)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:6].T


def day_b6(lm: LM) -> np.ndarray:
    ids = lm.token_ids(DAYS)
    Wout, _ = out_rows(lm)
    X = Wout[ids].numpy().astype(np.float64)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:6].T


def stage_attrib(lm: LM) -> None:
    Wout, tied = out_rows(lm)
    B6 = month_b6(lm)
    P = B6 @ B6.T
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    few = w26.ctx_positions(lm, FEWSHOT_MO, MONTHS.words)
    clouds = wr.clouds(few, len(MONTHS.words))
    A = clouds.pop("_total")
    ssum = clouds["emb"] + sum(v for k, v in clouds.items() if k != "emb")
    rel_err = float(np.abs(ssum - A).max() / np.abs(A).max())
    assert rel_err < 1e-3, rel_err
    Ac = A - A.mean(0)
    denom = float(((Ac @ B6) ** 2).sum())
    alphas = {name: float(np.sum(((C - C.mean(0)) @ P) * (Ac @ P)) / denom)
              for name, C in clouds.items()}
    late = sum(abs(v) for k, v in alphas.items()
               if k != "emb" and int(k.split("_")[1]) >= 21)
    early = sum(abs(v) for k, v in alphas.items()
                if k != "emb" and int(k.split("_")[1]) < 21)
    top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
    rep = {"model": lm.name, "tied": tied, "target": "months", "spared": "days",
           "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
           "alpha_emb": alphas["emb"], "late_share_21_27": late,
           "early_share_0_20": early, "top_writer": top,
           "additivity_rel_err": rel_err}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(rep, indent=2))
    np.save(OUT / "refs.npy", {"B6": B6}, allow_pickle=True)
    print(json.dumps({"top8": dict(list(rep["alphas"].items())[:8]),
                      "late": late, "early": early, "top": top, "tied": tied},
                     indent=2), flush=True)


class MonthQRef:
    """Month 2-hop battery: MONTHS target, DAYS spared. Mirror of QRef + clean-KL."""

    def __init__(self, wr: w18.Writers, wiki_ids: torch.Tensor):
        self.B6 = np.load(OUT / "refs.npy", allow_pickle=True).item()["B6"]
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.month_ids = lm.token_ids(MONTHS)
        self.day_ids = lm.token_ids(DAYS)
        self.B6_day = day_b6(lm)
        self.day_tpl = w18.positions(lm, DAYS)
        self.month_tpl = w18.positions(lm, MONTHS)
        self.few_mo = w26.ctx_positions(lm, FEWSHOT_MO, MONTHS.words)   # target
        self.few_day = w26.ctx_positions(lm, FEWSHOT_DAY, DAYS.words)   # spared
        Hm = self._cloud(self.month_tpl, len(MONTHS.words), lm.probe_layer)
        self.BHm, _ = array_pca(Hm, 2)
        self.month_tokplane_ref = cloud_plane_power(Hm, self.BHm)
        Hd = self._cloud(self.day_tpl, len(DAYS.words), lm.probe_layer)
        self.BHd, _ = array_pca(Hd, 2)
        self.day_tokplane_ref = cloud_plane_power(Hd, self.BHd)
        self.A_ref = cloud_plane_power(
            self._cloud(self.few_mo, len(MONTHS.words), lm.n_layers), self.B6)
        self.Aday_ref = cloud_plane_power(
            self._cloud(self.few_day, len(DAYS.words), lm.n_layers), self.B6_day)
        self.wiki = wiki_ids.to(lm.device)
        tainted_mo = torch.cummax(
            torch.isin(wiki_ids, torch.tensor(self.month_ids)), 1).values
        self.clean_mo = (~tainted_mo.bool())
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        self.wiki_logp_ref = F.log_softmax(logits.float(), dim=-1).cpu()

    _cloud = w19.OutRef._cloud
    _restricted = w26.MonthOutRef._restricted

    def measure_state(self) -> dict:
        m: dict = {}
        Acl = self._cloud(self.few_mo, len(MONTHS.words), self.lm.n_layers)
        m["A_mo_B6_power"] = cloud_plane_power(Acl, self.B6) / max(self.A_ref, 1e-12)
        m["_A_cloud_B6"] = [[round(float(v), 4) for v in row]
                            for row in ((Acl - Acl.mean(0)) @ self.B6)[:, :2]]
        Aday = self._cloud(self.few_day, len(DAYS.words), self.lm.n_layers)
        m["A_day_B6_power"] = cloud_plane_power(Aday, self.B6_day) \
            / max(self.Aday_ref, 1e-12)
        m["month_tokplane_power"] = cloud_plane_power(
            self._cloud(self.month_tpl, len(MONTHS.words), self.lm.probe_layer),
            self.BHm) / max(self.month_tokplane_ref, 1e-12)
        m["day_tokplane_power"] = cloud_plane_power(
            self._cloud(self.day_tpl, len(DAYS.words), self.lm.probe_layer),
            self.BHd) / max(self.day_tokplane_ref, 1e-12)
        m["month_2hop_acc"], m["month_2hop_margin"] = self._restricted(
            self.few_mo, self.month_ids, SHIFT)     # TARGET (should collapse)
        m["day_2hop_acc"], m["day_2hop_margin"] = self._restricted(
            self.few_day, self.day_ids, SHIFT)      # SPARED (should hold)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        logp = F.log_softmax(logits.float(), dim=-1).cpu()
        kl = F.kl_div(logp.flatten(0, 1), self.wiki_logp_ref.flatten(0, 1),
                      reduction="none", log_target=True).sum(-1).reshape(logp.shape[:2])
        m["wiki_kl"] = float(kl.mean())
        cm = self.clean_mo
        m["wiki_kl_clean"] = float(kl[cm].mean()) if cm.any() else float("nan")
        return m


def stage_sweep(lm: LM) -> None:
    attrib = json.loads((OUT / "attribution.json").read_text())
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    ref = MonthQRef(wr, w18.wiki_batch(lm))
    P = torch.from_numpy(ref.B6 @ ref.B6.T).float()
    all_names = list(wr.orig.keys())
    attn_names = [k for k in all_names if k.startswith("attn")]
    mlp_names = [k for k in all_names if k.startswith("mlp")]
    top = attrib["top_writer"]

    def plane_mats(names):
        return {nm: wr.arch.removed(wr.orig[nm], P) for nm in names}

    def rand_mats(names, seed):
        g = torch.Generator().manual_seed(seed)
        res = {}
        for nm in names:
            Q, _ = torch.linalg.qr(torch.randn(lm.hidden_size, 6, generator=g))
            M = wr.arch.removed(wr.orig[nm], Q @ Q.T)
            tgt = float(wr.arch.removed(wr.orig[nm], P).norm())
            res[nm] = M * (tgt / max(float(M.norm()), 1e-12))
        return res

    conds = {
        "T_out_top": plane_mats([top]),
        "T_out_allattn": plane_mats(attn_names),
        "T_out_allmlp": plane_mats(mlp_names),
        "T_out_all": plane_mats(all_names),
        **{f"C_out_rand_r{s}": rand_mats(all_names, s) for s in range(3)},
    }
    sweep_dir = OUT / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for name, mats in conds.items():
        if (sweep_dir / f"{name}.json").exists():
            print(f"{name} exists, skipping", flush=True)
            continue
        rows = []
        for lam in LAMBDAS:
            wr.set_removals(lam, mats)
            rows.append({"lam": lam, **ref.measure_state()})
        wr.restore()
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        r0 = [r for r in rows if r["lam"] == 0.0][0]
        print(f"{name} done; lam0:",
              json.dumps({k: round(v, 4) for k, v in r0.items()
                          if not k.startswith("_")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["attrib", "sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(MODEL, device=args.device)
    if args.stage == "attrib":
        stage_attrib(lm)
    else:
        stage_sweep(lm)


if __name__ == "__main__":
    main()
