"""Llama-3.2-1B month-answer manifold mirror (preregistration_monthstab.md,
Panel 2, freeze 5adc074).

Full mirror of 19_contextual.py's output-coordinate B6 pipeline (attrib-out /
sweep-out, addendum 529dbd4) with MONTHS as the target and DAYS as the spared
neighbor. Answer cloud A_mo = final pre-norm residual of "The month after {X} is"
(12 months, want X+1); B6_mo = top-6 prior-free PCA of the centered month rows of
the input embedding; writers = all attn o_proj + MLP down-proj, layers
0..n_layers-1. The DAY B6 answer + layer-probe day-token plane + the DAY 1-hop
task are the frozen SPARED readouts (they ride the month-edited weights). Adds a
clean-context wiki-KL split (positions with no month token in context) for the
specificity gate. Frozen basis at lambda=1, 26-point lambda grid.

Usage:
  python 26_monthctx.py --stage attrib --device mps
  python 26_monthctx.py --stage sweep  --device mps
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.ringlib import DAYS, MONTHS, LM
from clocklib.unsup import array_pca, cloud_plane_power, pca_basis

w18 = import_module("18_writers_any")
w19 = import_module("19_contextual")

MODEL = "meta-llama/Llama-3.2-1B"
CTX_MO = "The month after {} is"     # month 1-hop target, want X+1
CTX_DAY = "The day after {} is"      # day 1-hop spared readout, want X+1
OUT = ROOT / "artifacts" / "monthctx_llama"
LAMBDAS = w18.LAMBDAS


def ctx_positions(lm: LM, tpl: str, words) -> list:
    out = []
    for wi, w in enumerate(words):
        enc = lm.tok(tpl.format(w), return_tensors="pt").input_ids
        out.append((enc, wi, enc.shape[1] - 1))
    return out


def stage_attrib(lm: LM) -> None:
    """Output-coordinate attribution: plane = B6 embedding MONTH span; cloud =
    final-position hs[n_layers] of the month-answer prompt; writers = all modules.
    Mirror of 19_contextual.stage_attrib_out with months."""
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    ids = lm.token_ids(MONTHS)
    B6, _ = pca_basis(wr.W1, ids, 6)
    P = B6 @ B6.T
    ctx = ctx_positions(lm, CTX_MO, MONTHS.words)
    clouds = wr.clouds(ctx, len(MONTHS.words))
    A = clouds.pop("_total")
    ssum = clouds["emb"] + sum(v for k, v in clouds.items() if k != "emb")
    rel_err = float(np.abs(ssum - A).max() / np.abs(A).max())
    assert rel_err < 1e-3, rel_err
    Ac = A - A.mean(0)
    denom = float(((Ac @ B6) ** 2).sum())
    alphas = {name: float(np.sum(((C - C.mean(0)) @ P) * (Ac @ P)) / denom)
              for name, C in clouds.items()}
    late = sum(abs(v) for k, v in alphas.items()
               if k != "emb" and int(k.split("_")[1]) >= 12)
    early = sum(abs(v) for k, v in alphas.items()
                if k != "emb" and int(k.split("_")[1]) < 12)
    top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
    rep_out = {"model": lm.name, "target": "months", "spared": "days",
               "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
               "alpha_emb": alphas["emb"], "late_share_12_15": late,
               "early_share_0_11": early, "top_writer": top,
               "additivity_rel_err": rel_err}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(rep_out, indent=2))
    np.save(OUT / "refs.npy", {"B6": B6}, allow_pickle=True)
    show = dict(list(rep_out["alphas"].items())[:8])
    print(json.dumps({"top8": show, "alpha_emb": alphas["emb"], "late": late,
                      "early": early, "top": top}, indent=2), flush=True)


class MonthOutRef:
    """Battery for the month-answer output-coordinate sweep. MONTHS target, DAYS
    spared. Mirror of 19_contextual.OutRef with the roles swapped + clean-KL."""

    def __init__(self, wr: w18.Writers, wiki_ids: torch.Tensor):
        self.B6 = np.load(OUT / "refs.npy", allow_pickle=True).item()["B6"]
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.month_ids = lm.token_ids(MONTHS)
        self.day_ids = lm.token_ids(DAYS)
        self.day_tpl = w18.positions(lm, DAYS)
        self.month_tpl = w18.positions(lm, MONTHS)
        self.ctx_mo = ctx_positions(lm, CTX_MO, MONTHS.words)   # target task
        self.ctx_day = ctx_positions(lm, CTX_DAY, DAYS.words)   # spared task
        # DAY answer B6 (spared answer object): top-6 PCA of DAY embedding rows
        self.B6_day, _ = pca_basis(wr.W1, self.day_ids, 6)
        # frozen probe-layer token planes
        Hm11 = self._cloud(self.month_tpl, len(MONTHS.words), lm.probe_layer)
        self.BHm11, _ = array_pca(Hm11, 2)
        self.month_tokplane_ref = cloud_plane_power(Hm11, self.BHm11)
        Hd11 = self._cloud(self.day_tpl, len(DAYS.words), lm.probe_layer)
        self.BHd11, _ = array_pca(Hd11, 2)
        self.day_tokplane_ref = cloud_plane_power(Hd11, self.BHd11)
        # frozen answer-cloud references at lambda=1
        self.A_ref = cloud_plane_power(
            self._cloud(self.ctx_mo, len(MONTHS.words), lm.n_layers), self.B6)
        self.Aday_ref = cloud_plane_power(
            self._cloud(self.ctx_day, len(DAYS.words), lm.n_layers), self.B6_day)
        # wiki + clean masks
        self.wiki = wiki_ids.to(lm.device)
        tainted_mo = torch.cummax(
            torch.isin(wiki_ids, torch.tensor(self.month_ids)), 1).values
        self.clean_mo = (~tainted_mo.bool()).to(lm.device)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        self.wiki_logp_ref = F.log_softmax(logits.float(), dim=-1).cpu()

    _cloud = w19.OutRef._cloud

    @torch.no_grad()
    def _restricted(self, encs, ids, shift: int) -> tuple[float, float]:
        n = len(ids)
        margins, correct = [], 0
        for i, item in enumerate(encs):
            enc = item[0] if isinstance(item, tuple) else item
            dl = self.wr.model(input_ids=enc.to(self.lm.device)).logits[0, -1][ids]
            want = (i + shift) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / n, float(np.mean(margins))

    def measure_state(self) -> dict:
        m: dict = {}
        Acl = self._cloud(self.ctx_mo, len(MONTHS.words), self.lm.n_layers)
        m["A_mo_B6_power"] = cloud_plane_power(Acl, self.B6) / max(self.A_ref, 1e-12)
        m["_A_cloud_B6"] = [[round(float(v), 4) for v in row]
                            for row in ((Acl - Acl.mean(0)) @ self.B6)[:, :2]]
        Aday = self._cloud(self.ctx_day, len(DAYS.words), self.lm.n_layers)
        m["A_day_B6_power"] = cloud_plane_power(Aday, self.B6_day) \
            / max(self.Aday_ref, 1e-12)
        m["month_tokplane_power"] = cloud_plane_power(
            self._cloud(self.month_tpl, len(MONTHS.words), self.lm.probe_layer),
            self.BHm11) / max(self.month_tokplane_ref, 1e-12)
        m["day_tokplane_power"] = cloud_plane_power(
            self._cloud(self.day_tpl, len(DAYS.words), self.lm.probe_layer),
            self.BHd11) / max(self.day_tokplane_ref, 1e-12)
        m["month_1hop_acc"], m["month_1hop_margin"] = self._restricted(
            self.ctx_mo, self.month_ids, 1)     # TARGET task (should collapse)
        m["day_1hop_acc"], m["day_1hop_margin"] = self._restricted(
            self.ctx_day, self.day_ids, 1)      # SPARED task (should hold)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        logp = F.log_softmax(logits.float(), dim=-1).cpu()
        kl = F.kl_div(logp.flatten(0, 1), self.wiki_logp_ref.flatten(0, 1),
                      reduction="none", log_target=True).sum(-1).reshape(logp.shape[:2])
        m["wiki_kl"] = float(kl.mean())
        cm = self.clean_mo.cpu()
        m["wiki_kl_clean"] = float(kl[cm].mean()) if cm.any() else float("nan")
        return m


def stage_sweep(lm: LM) -> None:
    attrib = json.loads((OUT / "attribution.json").read_text())
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    ref = MonthOutRef(wr, w18.wiki_batch(lm))
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
    sweep_dir = OUT / "sweep_out"
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
