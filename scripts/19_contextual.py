"""Contextual placement experiment (preregistration_contextual.md, 80b0793).

The answer cloud A: final-token states of "The day after {X} is" — a day-manifold
point computed in-flight (X+1 appears nowhere in the input). Stage attrib measures
placement (probe transfer), subspace sharing (P_A vs day-token plane), and writer
attribution of A. Stage sweep doses A's plane at the writers.

Usage:
  python 19_contextual.py --stage attrib --device cpu
  python 19_contextual.py --stage sweep --device mps
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
from clocklib.unsup import array_pca, cloud_plane_power, principal_cos

w18 = import_module("18_writers_any")

MODEL = "meta-llama/Llama-3.2-1B"
CTX_TPL = "The day after {} is"
FEWSHOT_TPL = ("Let's do some day of the week math. Two days after Monday is "
               "Wednesday. Two days after Friday is Sunday. Two days after {} is")
OUT = ROOT / "artifacts" / "contextual_llama"
LAMBDAS = w18.LAMBDAS


def ctx_positions(lm: LM, tpl: str) -> list[tuple[torch.Tensor, int, int]]:
    out = []
    for wi, w in enumerate(DAYS.words):
        enc = lm.tok(tpl.format(w), return_tensors="pt").input_ids
        out.append((enc, wi, enc.shape[1] - 1))  # final position
    return out


def fit_probe(H: np.ndarray):
    from sklearn.linear_model import Ridge

    n = H.shape[0]
    t = 2 * np.pi * np.arange(n) / n
    return Ridge(alpha=1e-3).fit(H, np.stack([np.cos(t), np.sin(t)], 1))


def probe_decode(probe, H: np.ndarray) -> list[int]:
    pred = probe.predict(H)
    ang = np.arctan2(pred[:, 1], pred[:, 0]) % (2 * np.pi)
    return [int(round(a / (2 * np.pi / 7))) % 7 for a in ang]


def stage_attrib(lm: LM) -> None:
    wr = w18.Writers(lm)
    day_tpl = w18.positions(lm, DAYS)
    Hd = np.zeros((7, lm.hidden_size))
    with torch.no_grad():
        for enc, wi, pos in day_tpl:
            hs = wr.model(input_ids=enc.to(lm.device),
                          output_hidden_states=True).hidden_states
            Hd[wi] += hs[lm.probe_layer][0, pos].float().cpu().numpy()
    Hd /= len(DAYS.templates)
    BH, _ = array_pca(Hd, 2)
    probe = fit_probe(Hd)
    assert probe_decode(probe, Hd) == list(range(7)), "probe fails on its own data"

    ctx = ctx_positions(lm, CTX_TPL)
    clouds = wr.clouds(ctx, 7)
    A = clouds.pop("_total")
    decoded = probe_decode(probe, A)
    want = [(i + 1) % 7 for i in range(7)]
    transfer_acc = sum(d == w for d, w in zip(decoded, want))
    PA_B, rep = array_pca(A, 2)
    cos_PA_BH = principal_cos(PA_B, BH)

    P = PA_B @ PA_B.T
    Ac = A - A.mean(0)
    denom = float(((Ac @ PA_B) ** 2).sum())
    alphas = {name: float(np.sum(((C - C.mean(0)) @ P) * (Ac @ P)) / denom)
              for name, C in clouds.items()}
    attn_share = sum(abs(v) for k, v in alphas.items() if k.startswith("attn"))
    mlp_share = sum(abs(v) for k, v in alphas.items() if k.startswith("mlp"))
    top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))
    attn = [k for k in clouds if k.startswith("attn")]
    mlp = [k for k in clouds if k.startswith("mlp")]

    def lin_pred(uncut):
        return float(cloud_plane_power(sum(clouds[c] for c in uncut), PA_B) / denom)

    rep_out = {
        "probe_transfer": {"decoded": decoded, "want": want, "acc": transfer_acc},
        "cos_PA_vs_daytoken_plane": cos_PA_BH,
        "PA_top2_share": rep["share"],
        "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
        "alpha_emb": alphas["emb"], "attn_share": attn_share,
        "mlp_share": mlp_share, "top_writer": top,
        "linear_pred": {
            "T_ctx_top": lin_pred(["emb"] + [c for c in attn + mlp if c != top]),
            "T_ctx_allattn": lin_pred(["emb"] + mlp),
            "T_ctx_allmlp": lin_pred(["emb"] + attn),
            "T_ctx_all": lin_pred(["emb"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ctx_attribution.json").write_text(json.dumps(rep_out, indent=2))
    np.save(OUT / "ctx_refs.npy", {"BH": BH, "PA": PA_B, "Hd": Hd, "A": A},
            allow_pickle=True)
    show = {k: v for k, v in rep_out.items() if k not in ("alphas",)}
    show["top8_alphas"] = dict(list(rep_out["alphas"].items())[:8])
    print(json.dumps(show, indent=2), flush=True)


class CtxRef:
    """Frozen battery for the contextual sweep."""

    def __init__(self, wr: w18.Writers, wiki_ids: torch.Tensor):
        refs = np.load(OUT / "ctx_refs.npy", allow_pickle=True).item()
        self.BH, self.PA = refs["BH"], refs["PA"]
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.p_ids = lm.token_ids(DAYS)
        self.day_tpl = w18.positions(lm, DAYS)
        self.month_tpl = w18.positions(lm, MONTHS)
        self.ctx = ctx_positions(lm, CTX_TPL)
        self.few = ctx_positions(lm, FEWSHOT_TPL)
        self.succ_enc = [lm.tok(DAYS.successor_prompt.format(w), return_tensors="pt"
                                ).input_ids.to(lm.device) for w in DAYS.words]
        Hm = self._cloud(self.month_tpl, 12)
        self.BHm, _ = array_pca(Hm, 2)
        self.month_ref = cloud_plane_power(Hm, self.BHm)
        self.day_ref = cloud_plane_power(self._cloud(self.day_tpl, 7), self.BH)
        self.A_ref = cloud_plane_power(self._cloud(self.ctx, 7), self.PA)
        self.wiki = wiki_ids.to(lm.device)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        self.wiki_logp_ref = F.log_softmax(logits.float(), dim=-1).cpu()

    @torch.no_grad()
    def _cloud(self, tpl, n_words: int) -> np.ndarray:
        H = np.zeros((n_words, self.lm.hidden_size))
        for enc, wi, pos in tpl:
            hs = self.wr.model(input_ids=enc.to(self.lm.device),
                               output_hidden_states=True).hidden_states
            H[wi] += hs[self.lm.probe_layer][0, pos].float().cpu().numpy()
        return H / (len(tpl) // n_words)

    @torch.no_grad()
    def _restricted(self, encs, shift: int) -> tuple[float, float]:
        n = len(self.p_ids)
        margins, correct = [], 0
        for i, (enc, _, _) in enumerate(encs):
            dl = self.wr.model(input_ids=enc.to(self.lm.device)).logits[0, -1][self.p_ids]
            want = (i + shift) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / n, float(np.mean(margins))

    def measure_state(self) -> dict:
        m: dict = {}
        m["A_plane_power"] = cloud_plane_power(self._cloud(self.ctx, 7), self.PA) \
            / max(self.A_ref, 1e-12)
        m["day_plane_power"] = cloud_plane_power(self._cloud(self.day_tpl, 7), self.BH) \
            / max(self.day_ref, 1e-12)
        m["month_plane_power"] = cloud_plane_power(self._cloud(self.month_tpl, 12),
                                                   self.BHm) / max(self.month_ref, 1e-12)
        m["ctx_acc"], m["ctx_margin_mean"] = self._restricted(self.ctx, 1)
        m["few2_acc"], m["few2_margin_mean"] = self._restricted(self.few, 2)
        m["succ_acc"], m["succ_margin_mean"] = self._restricted(
            [(e, 0, 0) for e in self.succ_enc], 1)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        logp = F.log_softmax(logits.float(), dim=-1).cpu()
        m["wiki_kl"] = float(F.kl_div(logp.flatten(0, 1),
                                      self.wiki_logp_ref.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
        return m


def stage_sweep(lm: LM) -> None:
    attrib = json.loads((OUT / "ctx_attribution.json").read_text())
    wr = w18.Writers(lm)
    ref = CtxRef(wr, w18.wiki_batch(lm))
    P = torch.from_numpy(ref.PA @ ref.PA.T).float()
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
            Q, _ = torch.linalg.qr(torch.randn(lm.hidden_size, 2, generator=g))
            M = wr.arch.removed(wr.orig[nm], Q @ Q.T)
            tgt = float(wr.arch.removed(wr.orig[nm], P).norm())
            res[nm] = M * (tgt / max(float(M.norm()), 1e-12))
        return res

    conds = {
        "T_ctx_top": plane_mats([top]),
        "T_ctx_allattn": plane_mats(attn_names),
        "T_ctx_allmlp": plane_mats(mlp_names),
        "T_ctx_all": plane_mats(all_names),
        **{f"C_ctx_rand_r{s}": rand_mats(all_names, s) for s in range(3)},
    }
    sweep_dir = OUT / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    for name, mats in conds.items():
        rows = []
        for lam in LAMBDAS:
            wr.set_removals(lam, mats)
            rows.append({"lam": lam, **ref.measure_state()})
        wr.restore()
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name} done", flush=True)


def stage_attrib_out(lm: LM) -> None:
    """Output-coordinate attribution (addendum, 529dbd4): plane = B6 embedding
    day span; cloud = final-position hs[n_layers]; writers = all modules."""
    from clocklib.unsup import pca_basis

    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    ids = lm.token_ids(DAYS)
    B6, _ = pca_basis(wr.W1, ids, 6)
    P = B6 @ B6.T
    ctx = ctx_positions(lm, CTX_TPL)
    clouds = wr.clouds(ctx, 7)
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
    rep_out = {"alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
               "alpha_emb": alphas["emb"], "late_share_12_15": late,
               "early_share_0_11": early, "top_writer": top,
               "additivity_rel_err": rel_err}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "out_attribution.json").write_text(json.dumps(rep_out, indent=2))
    np.save(OUT / "out_refs.npy", {"B6": B6}, allow_pickle=True)
    show = dict(list(rep_out["alphas"].items())[:8])
    print(json.dumps({"top8": show, "alpha_emb": alphas["emb"], "late": late,
                      "early": early, "top": top}, indent=2), flush=True)


class OutRef:
    """Battery for the output-coordinate sweep (addendum, 529dbd4)."""

    def __init__(self, wr: w18.Writers, wiki_ids: torch.Tensor):
        self.B6 = np.load(OUT / "out_refs.npy", allow_pickle=True).item()["B6"]
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.p_ids = lm.token_ids(DAYS)
        self.day_tpl = w18.positions(lm, DAYS)
        self.month_tpl = w18.positions(lm, MONTHS)
        self.ctx = ctx_positions(lm, CTX_TPL)
        self.few = ctx_positions(lm, FEWSHOT_TPL)
        self.succ_enc = [lm.tok(DAYS.successor_prompt.format(w), return_tensors="pt"
                                ).input_ids.to(lm.device) for w in DAYS.words]
        Hd11 = self._cloud(self.day_tpl, 7, lm.probe_layer)
        self.BH11, _ = array_pca(Hd11, 2)
        self.day_ref = cloud_plane_power(Hd11, self.BH11)
        Hm11 = self._cloud(self.month_tpl, 12, lm.probe_layer)
        self.BHm11, _ = array_pca(Hm11, 2)
        self.month_ref = cloud_plane_power(Hm11, self.BHm11)
        self.A_ref = cloud_plane_power(self._cloud(self.ctx, 7, lm.n_layers), self.B6)
        self.wiki = wiki_ids.to(lm.device)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        self.wiki_logp_ref = F.log_softmax(logits.float(), dim=-1).cpu()

    @torch.no_grad()
    def _cloud(self, tpl, n_words: int, layer: int) -> np.ndarray:
        H = np.zeros((n_words, self.lm.hidden_size))
        grabbed = {}
        handle = None
        if layer == self.lm.n_layers:  # raw pre-norm residual, not HF's norm(h)
            def fn(_m, _i, out):
                grabbed["h"] = out[0] if isinstance(out, tuple) else out
            handle = self.wr.arch.blocks[layer - 1].register_forward_hook(fn)
        try:
            for enc, wi, pos in tpl:
                hs = self.wr.model(input_ids=enc.to(self.lm.device),
                                   output_hidden_states=True).hidden_states
                src = grabbed["h"] if handle is not None else hs[layer]
                H[wi] += src[0, pos].float().cpu().numpy()
        finally:
            if handle is not None:
                handle.remove()
        return H / (len(tpl) // n_words)

    @torch.no_grad()
    def _restricted(self, encs, shift: int) -> tuple[float, float]:
        n = len(self.p_ids)
        margins, correct = [], 0
        for i, item in enumerate(encs):
            enc = item[0] if isinstance(item, tuple) else item
            dl = self.wr.model(input_ids=enc.to(self.lm.device)).logits[0, -1][self.p_ids]
            want = (i + shift) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / n, float(np.mean(margins))

    def measure_state(self) -> dict:
        m: dict = {}
        Acl = self._cloud(self.ctx, 7, self.lm.n_layers)
        m["A_B6_power"] = cloud_plane_power(Acl, self.B6) / max(self.A_ref, 1e-12)
        m["_A_cloud_B6"] = [[round(float(v), 4) for v in row]
                            for row in ((Acl - Acl.mean(0)) @ self.B6)[:, :2]]
        m["day_plane_power"] = cloud_plane_power(
            self._cloud(self.day_tpl, 7, self.lm.probe_layer), self.BH11) \
            / max(self.day_ref, 1e-12)
        m["month_plane_power"] = cloud_plane_power(
            self._cloud(self.month_tpl, 12, self.lm.probe_layer), self.BHm11) \
            / max(self.month_ref, 1e-12)
        m["ctx_acc"], m["ctx_margin_mean"] = self._restricted(self.ctx, 1)
        m["succ_acc"], m["succ_margin_mean"] = self._restricted(self.succ_enc, 1)
        m["few2_acc"], m["few2_margin_mean"] = self._restricted(self.few, 2)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        logp = F.log_softmax(logits.float(), dim=-1).cpu()
        m["wiki_kl"] = float(F.kl_div(logp.flatten(0, 1),
                                      self.wiki_logp_ref.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
        return m


def stage_sweep_out(lm: LM) -> None:
    attrib = json.loads((OUT / "out_attribution.json").read_text())
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    ref = OutRef(wr, w18.wiki_batch(lm))
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
    sweep_dir.mkdir(exist_ok=True)
    for name, mats in conds.items():
        rows = []
        for lam in LAMBDAS:
            wr.set_removals(lam, mats)
            rows.append({"lam": lam, **ref.measure_state()})
        wr.restore()
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["attrib", "sweep", "attrib-out", "sweep-out"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(MODEL, device=args.device)
    match args.stage:
        case "attrib":
            stage_attrib(lm)
        case "sweep":
            stage_sweep(lm)
        case "attrib-out":
            stage_attrib_out(lm)
        case "sweep-out":
            stage_sweep_out(lm)


if __name__ == "__main__":
    main()
