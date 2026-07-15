"""Llama-3.2-1B number 2-hop answer manifold mirror
(preregistration_numberstab.md, Panel 2, freeze 5745ec7).

Full mirror of 27_month2hop.py's few-shot 2-hop output-coordinate B6 pipeline with
NUMBERS as target and DAYS as the spared neighbor. DEVIATION disclosed in the
prereg: the numbers panel uses the 2-HOP frame (not 1-hop) because the 1-hop
"number after" task is dead for the small models (Llama 0/11) while the 2-hop
few-shot clears the bar (Llama 8/10). Answer cloud A2_nu = final pre-norm residual
(hidden_states after the last block) of the frozen few-shot 2-hop NUMBER frame
(want X+2, X in one..ten, 10 non-wrapping items); B6n = top-6 prior-free PCA of the
centered number rows of the OUTPUT embedding (tied -> input rows). The DAY B6 2-hop
answer + probe-layer day-token plane + the DAY few-shot 2-hop task are the frozen
SPARED readouts. Clean-context wiki-KL split (no number token in context) for the
specificity gate. Frozen basis lambda=1, 26-point grid.

CRITICAL: the number line does not wrap. The number task is scored over the 10
items X in one..ten with NON-MODULAR want = X_index+2; the day task wraps as in the
day tab. B6n is discovered from ALL 12 number rows (prior-free discovery set), the
answer cloud is the 10-prompt A2_nu.

Usage:
  python 29_numberctx.py --stage attrib --device mps
  python 29_numberctx.py --stage sweep  --device mps
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

from clocklib.ringlib import DAYS, LM
from clocklib.unsup import NUMBERS, array_pca, cloud_plane_power

w18 = import_module("18_writers_any")
w19 = import_module("19_contextual")
w20 = import_module("20_qwen2hop")
w26 = import_module("26_monthctx")

MODEL = "meta-llama/Llama-3.2-1B"
# number analog of the day 2-hop frame (same wording form, numbers substituted).
# Few-shot exemplars use in-range successors (three, seven) so no answer leaks.
FEWSHOT_NU = ("Let's do some number math. Two after one is three. "
              "Two after five is seven. Two after {} is")
FEWSHOT_DAY = ("Let's do some day of the week math. Two days after Monday is "
               "Wednesday. Two days after Friday is Sunday. Two days after {} is")
OUT = ROOT / "artifacts" / "numberctx_llama"
LAMBDAS = w18.LAMBDAS
SHIFT = 2
# non-wrapping number 2-hop item set: X in one..ten (indices 0..9), want X+2.
NU_ITEMS = list(range(len(NUMBERS.words) - SHIFT))   # 0..9


def number_b6(lm: LM) -> np.ndarray:
    """top-6 PCA of the centered OUTPUT-embedding rows of ALL 12 number tokens."""
    ids = lm.token_ids(NUMBERS)
    Wout, _ = w20.out_rows(lm)
    X = Wout[ids].numpy().astype(np.float64)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:6].T


def stage_attrib(lm: LM) -> None:
    Wout, tied = w20.out_rows(lm)
    B6 = number_b6(lm)
    P = B6 @ B6.T
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    # answer cloud discovered over the 10 non-wrapping number prompts
    few_words = [NUMBERS.words[i] for i in NU_ITEMS]
    few = w26.ctx_positions(lm, FEWSHOT_NU, few_words)
    clouds = wr.clouds(few, len(few_words))
    A = clouds.pop("_total")
    ssum = clouds["emb"] + sum(v for k, v in clouds.items() if k != "emb")
    rel_err = float(np.abs(ssum - A).max() / np.abs(A).max())
    assert rel_err < 1e-3, rel_err
    Ac = A - A.mean(0)
    denom = float(((Ac @ B6) ** 2).sum())
    alphas = {name: float(np.sum(((C - C.mean(0)) @ P) * (Ac @ P)) / denom)
              for name, C in clouds.items()}
    late = sum(abs(v) for k, v in alphas.items()
               if k != "emb" and int(k.split("_")[1]) >= 2 * lm.n_layers // 3)
    early = sum(abs(v) for k, v in alphas.items()
                if k != "emb" and int(k.split("_")[1]) < 2 * lm.n_layers // 3)
    top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
    rep = {"model": lm.name, "tied": tied, "target": "numbers", "spared": "days",
           "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
           "alpha_emb": alphas["emb"], "late_share": late, "early_share": early,
           "top_writer": top, "additivity_rel_err": rel_err}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(rep, indent=2))
    np.save(OUT / "refs.npy", {"B6": B6}, allow_pickle=True)
    print(json.dumps({"top8": dict(list(rep["alphas"].items())[:8]),
                      "alpha_emb": alphas["emb"], "late": late, "early": early,
                      "top": top, "tied": tied}, indent=2), flush=True)


class NumberQRef:
    """Number 2-hop battery: NUMBERS target (10 non-wrapping items), DAYS spared.
    Mirror of MonthQRef (27_month2hop) with numbers + a non-wrapping number task."""

    def __init__(self, wr: w18.Writers, wiki_ids: torch.Tensor):
        self.B6 = np.load(OUT / "refs.npy", allow_pickle=True).item()["B6"]
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.number_ids = lm.token_ids(NUMBERS)
        self.day_ids = lm.token_ids(DAYS)
        # DAY answer B6 (spared answer object): top-6 PCA of DAY output rows
        Wout, _ = w20.out_rows(lm)
        Xd = Wout[self.day_ids].numpy().astype(np.float64)
        Xd = Xd - Xd.mean(0)
        _, _, Vtd = np.linalg.svd(Xd, full_matrices=False)
        self.B6_day = Vtd[:6].T
        self.number_tpl = w18.positions(lm, NUMBERS)
        self.day_tpl = w18.positions(lm, DAYS)
        few_words = [NUMBERS.words[i] for i in NU_ITEMS]
        self.few_nu = w26.ctx_positions(lm, FEWSHOT_NU, few_words)      # target (10)
        self.few_day = w26.ctx_positions(lm, FEWSHOT_DAY, DAYS.words)   # spared (7)
        Hn = self._cloud(self.number_tpl, len(NUMBERS.words), lm.probe_layer)
        self.BHn, _ = array_pca(Hn, 2)
        self.number_tokplane_ref = cloud_plane_power(Hn, self.BHn)
        Hd = self._cloud(self.day_tpl, len(DAYS.words), lm.probe_layer)
        self.BHd, _ = array_pca(Hd, 2)
        self.day_tokplane_ref = cloud_plane_power(Hd, self.BHd)
        self.A_ref = cloud_plane_power(
            self._cloud(self.few_nu, len(few_words), lm.n_layers), self.B6)
        self.Aday_ref = cloud_plane_power(
            self._cloud(self.few_day, len(DAYS.words), lm.n_layers), self.B6_day)
        self.wiki = wiki_ids.to(lm.device)
        tainted = torch.cummax(
            torch.isin(wiki_ids, torch.tensor(self.number_ids)), 1).values
        self.clean = ~tainted.bool()
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        self.wiki_logp_ref = F.log_softmax(logits.float(), dim=-1).cpu()

    _cloud = w19.OutRef._cloud

    @torch.no_grad()
    def _restricted_nowrap(self, encs, ids, shift: int) -> tuple[float, float]:
        """Non-wrapping number task: item i predicts word i+shift (no modulo).
        argmax restricted over the full number token set."""
        n = len(ids)
        idx = torch.tensor(ids)
        margins, correct = [], 0
        for i, item in enumerate(encs):
            enc = item[0] if isinstance(item, tuple) else item
            dl = self.wr.model(input_ids=enc.to(self.lm.device)).logits[0, -1][idx]
            want = i + shift
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / len(encs), float(np.mean(margins))

    @torch.no_grad()
    def _restricted_wrap(self, encs, ids, shift: int) -> tuple[float, float]:
        """Wrapping day task (as in the day tab): want = (i+shift) % n."""
        n = len(ids)
        idx = torch.tensor(ids)
        margins, correct = [], 0
        for i, item in enumerate(encs):
            enc = item[0] if isinstance(item, tuple) else item
            dl = self.wr.model(input_ids=enc.to(self.lm.device)).logits[0, -1][idx]
            want = (i + shift) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / len(encs), float(np.mean(margins))

    def measure_state(self) -> dict:
        m: dict = {}
        few_words = [NUMBERS.words[i] for i in NU_ITEMS]
        Acl = self._cloud(self.few_nu, len(few_words), self.lm.n_layers)
        m["A_nu_B6_power"] = cloud_plane_power(Acl, self.B6) / max(self.A_ref, 1e-12)
        m["_A_cloud_B6"] = [[round(float(v), 4) for v in row]
                            for row in ((Acl - Acl.mean(0)) @ self.B6)[:, :2]]
        Aday = self._cloud(self.few_day, len(DAYS.words), self.lm.n_layers)
        m["A_day_B6_power"] = cloud_plane_power(Aday, self.B6_day) \
            / max(self.Aday_ref, 1e-12)
        m["number_tokplane_power"] = cloud_plane_power(
            self._cloud(self.number_tpl, len(NUMBERS.words), self.lm.probe_layer),
            self.BHn) / max(self.number_tokplane_ref, 1e-12)
        m["day_tokplane_power"] = cloud_plane_power(
            self._cloud(self.day_tpl, len(DAYS.words), self.lm.probe_layer),
            self.BHd) / max(self.day_tokplane_ref, 1e-12)
        m["number_2hop_acc"], m["number_2hop_margin"] = self._restricted_nowrap(
            self.few_nu, self.number_ids, SHIFT)    # TARGET (should collapse)
        m["day_2hop_acc"], m["day_2hop_margin"] = self._restricted_wrap(
            self.few_day, self.day_ids, SHIFT)      # SPARED (should hold)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        logp = F.log_softmax(logits.float(), dim=-1).cpu()
        kl = F.kl_div(logp.flatten(0, 1), self.wiki_logp_ref.flatten(0, 1),
                      reduction="none", log_target=True).sum(-1).reshape(logp.shape[:2])
        m["wiki_kl"] = float(kl.mean())
        cm = self.clean
        m["wiki_kl_clean"] = float(kl[cm].mean()) if cm.any() else float("nan")
        return m


def stage_sweep(lm: LM) -> None:
    attrib = json.loads((OUT / "attribution.json").read_text())
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    ref = NumberQRef(wr, w18.wiki_batch(lm))
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
