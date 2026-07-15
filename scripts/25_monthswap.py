"""Months-as-target writer swap (preregistration_monthswap.md, freeze 349a99f).

The symmetric control to the day writer experiment (17/18_writers): MONTHS is the
edit target, DAYS is the spared neighbor. Same writer set (all attn c_proj + MLP
down-proj below the GPT-2 probe layer), same lambda grid, same prior-free centered
PCA discovery — roles reversed.

Imports the writer-edit machinery (Writers / Arch / positions / LAMBDAS /
wiki_batch) from 18_writers_any rather than modifying it.

Usage:
  python 25_monthswap.py --model gpt2 --stage attrib --device mps
  python 25_monthswap.py --model gpt2 --stage sweep  --device mps
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
Writers, positions, wiki_batch, LAMBDAS = (
    w18.Writers, w18.positions, w18.wiki_batch, w18.LAMBDAS)

ART = ROOT / "artifacts"


class MonthSwapRef:
    """Frozen lambda=1 reference: MONTHS target, DAYS spared neighbor.

    Mirror of WriterRef (18_writers_any) with the target/spared concepts swapped.
    The frozen month activation plane B_M is both the discovery plane (readout of
    the target) AND the plane whose projector drives the writer edit.
    """

    def __init__(self, wr: Writers, wiki_ids: torch.Tensor):
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.p_ids = lm.token_ids(MONTHS)   # target = months
        self.x_ids = lm.token_ids(DAYS)     # spared neighbor = days
        assert self.p_ids and self.x_ids
        self.month_tpl = positions(lm, MONTHS)
        self.day_tpl = positions(lm, DAYS)
        # spared behavioral readout: day successor (months succ is ineligible, 2/12)
        self.succ_enc = [lm.tok(DAYS.successor_prompt.format(w), return_tensors="pt"
                                ).input_ids.to(lm.device) for w in DAYS.words]
        self.d_ids = lm.token_ids(DAYS)
        self.wiki = wiki_ids.to(lm.device)

        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        self.BM, _ = array_pca(Hm, 2)       # frozen top-2 MONTH plane (target)
        self.BH, _ = array_pca(Hd, 2)       # frozen top-2 DAY plane (spared)
        self.month_ref = cloud_plane_power(Hm, self.BM)
        self.day_ref = cloud_plane_power(Hd, self.BH)
        m = self._model_metrics()
        self.wiki_logp_ref = m.pop("_wiki_logp")

    @torch.no_grad()
    def _cloud(self, tpl, n_words: int) -> np.ndarray:
        H = np.zeros((n_words, self.lm.hidden_size))
        for enc, wi, pos in tpl:
            hs = self.wr.model(input_ids=enc.to(self.lm.device),
                               output_hidden_states=True).hidden_states
            H[wi] += hs[self.lm.probe_layer][0, pos].float().cpu().numpy()
        return H / (len(tpl) // n_words)

    @torch.no_grad()
    def _restricted(self, encs, ids) -> tuple[float, float]:
        n = len(ids)
        margins, correct = [], 0
        for i, enc in enumerate(encs):
            dl = self.wr.model(input_ids=enc).logits[0, -1][ids]
            want = (i + 1) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / n, float(np.mean(margins))

    @torch.no_grad()
    def _model_metrics(self) -> dict:
        out: dict = {}
        # spared behavioral readout: DAY successor over the day token set
        out["day_succ_acc"], out["day_succ_margin_mean"] = self._restricted(
            self.succ_enc, self.d_ids)
        logits = self.wr.model(input_ids=self.wiki).logits
        out["_wiki_logp"] = F.log_softmax(logits.float(), dim=-1).cpu()
        return out

    def measure_state(self) -> dict:
        m = self._model_metrics()
        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        # target cloud on the frozen month plane, centered, 4-decimal [x,y] pairs
        m["_H_cloud_B2"] = [[round(float(v), 4) for v in row]
                            for row in ((Hm - Hm.mean(0)) @ self.BM)[:, :2]]
        m["month_plane_power"] = cloud_plane_power(Hm, self.BM) \
            / max(self.month_ref, 1e-12)   # TARGET (should collapse)
        m["day_plane_power"] = cloud_plane_power(Hd, self.BH) \
            / max(self.day_ref, 1e-12)     # SPARED neighbor (should hold)
        logp = m.pop("_wiki_logp")
        m["wiki_kl"] = float(F.kl_div(logp.flatten(0, 1),
                                      self.wiki_logp_ref.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
        return m


def stage_attrib(lm: LM, out: Path) -> None:
    """Record the frozen month/day planes + additive attribution of the month
    plane to the writers (informative; the criteria live in the sweep)."""
    wr = Writers(lm)
    tpl = positions(lm, MONTHS)
    n = len(MONTHS.words)
    clouds = wr.clouds(tpl, n)
    total = clouds.pop("_total")
    ssum = clouds["emb"] + sum(v for k, v in clouds.items() if k != "emb")
    add_err = float(np.abs(ssum - total).max())
    rel_err = add_err / float(np.abs(total).max())
    assert rel_err < 1e-3, f"residual additivity violated: {add_err} (rel {rel_err})"

    BM, rep = array_pca(total, 2)
    P = BM @ BM.T
    Tc = total - total.mean(0)
    denom = float(((Tc @ BM) ** 2).sum())
    alphas = {name: float(np.sum(((C - C.mean(0)) @ P) * (Tc @ P)) / denom)
              for name, C in clouds.items()}
    rho_pred = float(cloud_plane_power(clouds["emb"], BM) / denom)
    top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
    rep_out = {"model": lm.name, "probe_layer": lm.probe_layer, "target": "months",
               "spared": "days",
               "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
               "rho_pred": rho_pred, "top_writer": top,
               "additivity_rel_err": rel_err, "plane_top2_share": rep["share"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution.json").write_text(json.dumps(rep_out, indent=2))
    top8 = dict(list(rep_out["alphas"].items())[:8])
    print(json.dumps({"top8_alphas": top8, "alpha_emb": alphas["emb"],
                      "rho_pred": rho_pred}, indent=2), flush=True)


def stage_sweep(lm: LM, out: Path) -> None:
    wr = Writers(lm)
    ref = MonthSwapRef(wr, wiki_batch(lm))

    P = torch.from_numpy(ref.BM @ ref.BM.T).float()   # month-plane projector
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

    conds = {
        "T_mo_all": plane_mats(all_names),
        "C_mo_rand_r0": rand_mats(all_names, 0),
    }
    sweep_dir = out / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    for name, mats in conds.items():
        rows = []
        for lam in LAMBDAS:
            wr.set_removals(lam, mats)
            rows.append({"lam": lam, **ref.measure_state()})
        wr.restore()
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        r0 = rows[LAMBDAS.index(0.0)]
        print(f"{name} done; lam0:",
              json.dumps({k: round(v, 4) for k, v in r0.items()
                          if not k.startswith("_")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--stage", required=True, choices=["attrib", "sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(args.model, device=args.device)
    out = ART / f"monthswap_{args.model.replace('/', '_')}"
    if args.stage == "attrib":
        stage_attrib(lm, out)
    else:
        stage_sweep(lm, out)


if __name__ == "__main__":
    main()
