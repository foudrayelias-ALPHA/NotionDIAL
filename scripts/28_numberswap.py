"""GPT-2 numbers-as-target writer swap (preregistration_numberstab.md, Panel 1,
freeze 5745ec7).

The numbers analog of 25_monthswap.py: NUMBERS is the edit target, DAYS and MONTHS
are the spared neighbor concepts. Same writer set (all attn c_proj + MLP down-proj
below the GPT-2 probe layer), same 26-point lambda grid, same prior-free centered
PCA discovery on the number token rows. Behavior column = the run-up counting task
(9/9 at lambda=1; days/months had no eligible GPT-2 behavior task). Number words
occur in wikitext, so the wiki KL is split into raw and clean-of-numbers inline
(mirror of clocklib.unsup.LineRef.wiki_clean).

Imports the writer-edit machinery (Writers / Arch / positions / LAMBDAS /
wiki_batch) from 18_writers_any rather than modifying it.

Usage:
  python 28_numberswap.py --stage attrib --device mps
  python 28_numberswap.py --stage sweep  --device mps
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
from clocklib.unsup import NUMBERS, array_pca, cloud_plane_power

w18 = import_module("18_writers_any")
Writers, positions, wiki_batch, LAMBDAS = (
    w18.Writers, w18.positions, w18.wiki_batch, w18.LAMBDAS)

ART = ROOT / "artifacts"


class NumberSwapRef:
    """Frozen lambda=1 reference: NUMBERS target, DAYS + MONTHS spared neighbors.

    Mirror of MonthSwapRef (25_monthswap) with numbers as target and TWO spared
    neighbors. The frozen number activation plane B_N (top-2 PCA of the number
    token rows at the probe layer) is both the target readout AND the plane whose
    projector drives the writer edit. Behavior readout = run-up counting (no wrap).
    """

    def __init__(self, wr: Writers, wiki_ids: torch.Tensor):
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.n_ids = lm.token_ids(NUMBERS)   # target = numbers
        self.d_ids = lm.token_ids(DAYS)      # spared neighbor 1 = days
        self.m_ids = lm.token_ids(MONTHS)    # spared neighbor 2 = months
        assert self.n_ids and self.d_ids and self.m_ids
        self.num_tpl = positions(lm, NUMBERS)
        self.day_tpl = positions(lm, DAYS)
        self.month_tpl = positions(lm, MONTHS)
        # spared behavioral readout: run-up counting (mirror of unsup.LineRef).
        words = NUMBERS.words
        self.count_enc = [
            lm.tok(f"Count: {words[i-2]}, {words[i-1]}, {words[i]},",
                   return_tensors="pt").input_ids.to(lm.device)
            for i in range(2, len(words) - 1)]
        self.wiki = wiki_ids.to(lm.device)
        # clean mask: positions whose causal context contains NO number token.
        tainted = torch.cummax(
            torch.isin(wiki_ids, torch.tensor(self.n_ids)), 1).values
        self.wiki_clean = ~tainted.bool()

        Hn = self._cloud(self.num_tpl, len(NUMBERS.words))
        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        self.BN, _ = array_pca(Hn, 2)        # frozen top-2 NUMBER plane (target)
        self.BH, _ = array_pca(Hd, 2)        # frozen top-2 DAY plane (spared)
        self.BM, _ = array_pca(Hm, 2)        # frozen top-2 MONTH plane (spared)
        self.num_ref = cloud_plane_power(Hn, self.BN)
        self.day_ref = cloud_plane_power(Hd, self.BH)
        self.month_ref = cloud_plane_power(Hm, self.BM)
        self.wiki_logp_ref = self._wiki_logp()

    @torch.no_grad()
    def _cloud(self, tpl, n_words: int) -> np.ndarray:
        H = np.zeros((n_words, self.lm.hidden_size))
        for enc, wi, pos in tpl:
            hs = self.wr.model(input_ids=enc.to(self.lm.device),
                               output_hidden_states=True).hidden_states
            H[wi] += hs[self.lm.probe_layer][0, pos].float().cpu().numpy()
        return H / (len(tpl) // n_words)

    @torch.no_grad()
    def _wiki_logp(self):
        logits = self.wr.model(input_ids=self.wiki).logits
        return F.log_softmax(logits.float(), dim=-1).cpu()

    @torch.no_grad()
    def _counting(self) -> tuple[float, float]:
        """Run-up counting, non-wrapping. items i=2..n-2, want i+1. argmax over
        the 12 number tokens."""
        n = len(self.n_ids)
        idx = torch.tensor(self.n_ids)
        margins, correct = [], 0
        for j, enc in enumerate(self.count_enc):
            dl = self.wr.model(input_ids=enc).logits[0, -1][idx]
            want = j + 3   # item i=2 predicts word 3, etc.
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / len(self.count_enc), float(np.mean(margins))

    def measure_state(self) -> dict:
        m: dict = {}
        Hn = self._cloud(self.num_tpl, len(NUMBERS.words))
        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        # target cloud on the frozen number plane, centered, 4-decimal [x,y] pairs
        m["_H_cloud_B2"] = [[round(float(v), 4) for v in row]
                            for row in ((Hn - Hn.mean(0)) @ self.BN)[:, :2]]
        m["number_plane_power"] = cloud_plane_power(Hn, self.BN) \
            / max(self.num_ref, 1e-12)     # TARGET (should collapse)
        m["day_plane_power"] = cloud_plane_power(Hd, self.BH) \
            / max(self.day_ref, 1e-12)     # SPARED neighbor 1 (should hold)
        m["month_plane_power"] = cloud_plane_power(Hm, self.BM) \
            / max(self.month_ref, 1e-12)   # SPARED neighbor 2 (should hold)
        m["count_acc"], m["count_margin"] = self._counting()   # behavior column
        logp = self._wiki_logp()
        kl = F.kl_div(logp.flatten(0, 1), self.wiki_logp_ref.flatten(0, 1),
                      reduction="none", log_target=True).sum(-1).reshape(logp.shape[:2])
        m["wiki_kl"] = float(kl.mean())
        cm = self.wiki_clean
        m["wiki_kl_clean"] = float(kl[cm].mean()) if cm.any() else float("nan")
        return m


def stage_attrib(lm: LM, out: Path) -> None:
    """Record the frozen number plane + additive attribution of the number plane
    to the writers (informative; the criteria live in the sweep)."""
    wr = Writers(lm)
    tpl = positions(lm, NUMBERS)
    n = len(NUMBERS.words)
    clouds = wr.clouds(tpl, n)
    total = clouds.pop("_total")
    ssum = clouds["emb"] + sum(v for k, v in clouds.items() if k != "emb")
    rel_err = float(np.abs(ssum - total).max() / np.abs(total).max())
    assert rel_err < 1e-3, f"residual additivity violated: rel {rel_err}"

    BN, rep = array_pca(total, 2)
    P = BN @ BN.T
    Tc = total - total.mean(0)
    denom = float(((Tc @ BN) ** 2).sum())
    alphas = {name: float(np.sum(((C - C.mean(0)) @ P) * (Tc @ P)) / denom)
              for name, C in clouds.items()}
    rho_pred = float(cloud_plane_power(clouds["emb"], BN) / denom)
    top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
    rep_out = {"model": lm.name, "probe_layer": lm.probe_layer, "target": "numbers",
               "spared": "days+months",
               "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
               "rho_pred": rho_pred, "top_writer": top,
               "additivity_rel_err": rel_err, "plane_top2_share": rep["share"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution.json").write_text(json.dumps(rep_out, indent=2))
    print(json.dumps({"top8_alphas": dict(list(rep_out["alphas"].items())[:8]),
                      "alpha_emb": alphas["emb"], "rho_pred": rho_pred,
                      "top": top}, indent=2), flush=True)


def stage_sweep(lm: LM, out: Path) -> None:
    wr = Writers(lm)
    ref = NumberSwapRef(wr, wiki_batch(lm))

    P = torch.from_numpy(ref.BN @ ref.BN.T).float()   # number-plane projector
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
        "T_nu_all": plane_mats(all_names),
        "C_nu_rand_r0": rand_mats(all_names, 0),
    }
    sweep_dir = out / "sweep"
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
        r0 = rows[LAMBDAS.index(0.0)]
        print(f"{name} done; lam0:",
              json.dumps({k: round(v, 4) for k, v in r0.items()
                          if not k.startswith("_")}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["attrib", "sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM("gpt2", device=args.device)
    out = ART / "numberswap_gpt2"
    if args.stage == "attrib":
        stage_attrib(lm, out)
    else:
        stage_sweep(lm, out)


if __name__ == "__main__":
    main()
