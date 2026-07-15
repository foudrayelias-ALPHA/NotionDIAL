"""2-hop in-context day arithmetic under dose control (preregistration_qwen2hop.md,
193f444). Qwen2.5-1.5B, few-shot frame, output-coordinate day span B6.

Usage:
  python 20_qwen2hop.py --stage survey --device cpu     # P-Q1 (logit-lens + probe)
  python 20_qwen2hop.py --stage attrib --device cpu     # P-Q2 (writer attribution)
  python 20_qwen2hop.py --stage sweep  --device mps     # P-Q3..P-Q5
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

MODEL = "Qwen/Qwen2.5-1.5B"
FEWSHOT = ("Let's do some day of the week math. Two days after Monday is "
           "Wednesday. Two days after Friday is Sunday. Two days after {} is")
AFTER1 = "The day after {} is"
OUT = ROOT / "artifacts" / "qwen2hop"
LAMBDAS = w18.LAMBDAS
SHIFT = 2


def out_rows(lm: LM) -> tuple[torch.Tensor, bool]:
    """Output-embedding weight (tied -> input rows); returns (W, tied)."""
    oe = lm.model.get_output_embeddings()
    ie = lm.model.get_input_embeddings()
    tied = oe is None or oe.weight is ie.weight
    W = (ie if tied else oe).weight.detach().cpu()
    return W, tied


def stage_survey(lm: LM) -> None:
    ids = lm.token_ids(DAYS)
    Wout, tied = out_rows(lm)
    norm = lm.model.model.norm
    day_tpl = w18.positions(lm, DAYS)
    few = w19.ctx_positions(lm, FEWSHOT)
    L = lm.n_layers
    # pre-norm residuals per layer at final position; index L = after the LAST
    # block, captured by hook (hidden_states[L] is post-final-norm in HF)
    A = np.zeros((L + 1, 7, lm.hidden_size))
    Hd = np.zeros((L + 1, 7, lm.hidden_size))
    grabbed = {}

    def fn(_m, _i, out):
        grabbed["h"] = out[0] if isinstance(out, tuple) else out

    arch = w18.Arch(lm.model)
    handle = arch.blocks[L - 1].register_forward_hook(fn)
    try:
        with torch.no_grad():
            for enc, wi, pos in few:
                hs = lm.model(input_ids=enc, output_hidden_states=True).hidden_states
                for l in range(L):
                    A[l, wi] = hs[l][0, -1].float().numpy()
                A[L, wi] = grabbed["h"][0, -1].float().numpy()
            for enc, wi, pos in day_tpl:
                hs = lm.model(input_ids=enc, output_hidden_states=True).hidden_states
                for l in range(L):
                    Hd[l, wi] += hs[l][0, pos].float().numpy()
                Hd[L, wi] += grabbed["h"][0, pos].float().numpy()
    finally:
        handle.remove()
    Hd /= len(DAYS.templates)
    want = [(i + SHIFT) % 7 for i in range(7)]
    rows = []
    with torch.no_grad():
        for l in range(L + 1):
            h = norm(torch.from_numpy(A[l]).float())
            dl = h @ Wout[ids].T
            lens_acc = int(sum(int(dl[i].argmax()) == want[i] for i in range(7)))
            probe = w19.fit_probe(Hd[l])
            dec = w19.probe_decode(probe, A[l])
            probe_acc = int(sum(d == w for d, w in zip(dec, want)))
            rows.append({"layer": l, "lens_acc": lens_acc, "probe_acc": probe_acc})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "survey.json").write_text(json.dumps({"tied": tied, "rows": rows}, indent=2))
    best_lens = max(r["lens_acc"] for r in rows)
    max_probe = max(r["probe_acc"] for r in rows)
    first5 = next((r["layer"] for r in rows if r["lens_acc"] >= 5), None)
    print(json.dumps({"tied": tied, "best_lens": best_lens,
                      "first_layer_lens>=5": first5, "max_probe": max_probe,
                      "lens_by_layer": [r["lens_acc"] for r in rows]}, indent=2),
          flush=True)


def stage_attrib(lm: LM) -> None:
    from clocklib.unsup import array_pca as _pca

    ids = lm.token_ids(DAYS)
    Wout, tied = out_rows(lm)
    X = Wout[ids].numpy().astype(np.float64)
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    B6 = Vt[:6].T
    P = B6 @ B6.T
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    few = w19.ctx_positions(lm, FEWSHOT)
    clouds = wr.clouds(few, 7)
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
    rep = {"tied": tied, "alphas": dict(sorted(alphas.items(),
                                               key=lambda kv: -abs(kv[1]))),
           "alpha_emb": alphas["emb"], "late_share_21_27": late,
           "early_share_0_20": early, "top_writer": top,
           "additivity_rel_err": rel_err}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(rep, indent=2))
    np.save(OUT / "refs.npy", {"B6": B6}, allow_pickle=True)
    print(json.dumps({"top8": dict(list(rep["alphas"].items())[:8]),
                      "late": late, "early": early, "top": top}, indent=2),
          flush=True)


class QRef:
    def __init__(self, wr: w18.Writers, wiki_ids: torch.Tensor):
        self.B6 = np.load(OUT / "refs.npy", allow_pickle=True).item()["B6"]
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.p_ids = lm.token_ids(DAYS)
        self.day_tpl = w18.positions(lm, DAYS)
        self.month_tpl = w18.positions(lm, MONTHS)
        self.few = w19.ctx_positions(lm, FEWSHOT)
        self.after1 = w19.ctx_positions(lm, AFTER1)
        self.succ_enc = [lm.tok(DAYS.successor_prompt.format(w), return_tensors="pt"
                                ).input_ids.to(lm.device) for w in DAYS.words]
        Hd = self._cloud(self.day_tpl, 7, lm.probe_layer)
        self.BH, _ = array_pca(Hd, 2)
        self.day_ref = cloud_plane_power(Hd, self.BH)
        Hm = self._cloud(self.month_tpl, 12, lm.probe_layer)
        self.BHm, _ = array_pca(Hm, 2)
        self.month_ref = cloud_plane_power(Hm, self.BHm)
        self.A_ref = cloud_plane_power(self._cloud(self.few, 7, lm.n_layers), self.B6)
        self.wiki = wiki_ids.to(lm.device)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        self.wiki_logp_ref = F.log_softmax(logits.float(), dim=-1).cpu()

    _cloud = w19.OutRef._cloud
    _restricted = w19.OutRef._restricted

    def measure_state(self) -> dict:
        m: dict = {}
        Acl = self._cloud(self.few, 7, self.lm.n_layers)
        m["A_B6_power"] = cloud_plane_power(Acl, self.B6) / max(self.A_ref, 1e-12)
        m["_A_cloud_B6"] = [[round(float(v), 4) for v in row]
                            for row in ((Acl - Acl.mean(0)) @ self.B6)[:, :2]]
        m["day_plane_power"] = cloud_plane_power(
            self._cloud(self.day_tpl, 7, self.lm.probe_layer), self.BH) \
            / max(self.day_ref, 1e-12)
        m["month_plane_power"] = cloud_plane_power(
            self._cloud(self.month_tpl, 12, self.lm.probe_layer), self.BHm) \
            / max(self.month_ref, 1e-12)
        m["few2_acc"], m["few2_margin_mean"] = self._restricted(self.few, SHIFT)
        m["succ_acc"], m["succ_margin_mean"] = self._restricted(self.succ_enc, 1)
        m["after1_acc"], m["after1_margin_mean"] = self._restricted(self.after1, 1)
        with torch.no_grad():
            logits = self.wr.model(input_ids=self.wiki).logits
        logp = F.log_softmax(logits.float(), dim=-1).cpu()
        m["wiki_kl"] = float(F.kl_div(logp.flatten(0, 1),
                                      self.wiki_logp_ref.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
        return m


def stage_sweep(lm: LM) -> None:
    attrib = json.loads((OUT / "attribution.json").read_text())
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    ref = QRef(wr, w18.wiki_batch(lm))
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
    sweep_dir.mkdir(exist_ok=True)
    for name, mats in conds.items():
        if (sweep_dir / f"{name}.json").exists():  # resume after a killed run
            print(f"{name} exists, skipping", flush=True)
            continue
        rows = []
        for lam in LAMBDAS:
            wr.set_removals(lam, mats)
            rows.append({"lam": lam, **ref.measure_state()})
        wr.restore()
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["survey", "attrib", "sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(MODEL, device=args.device)
    match args.stage:
        case "survey":
            stage_survey(lm)
        case "attrib":
            stage_attrib(lm)
        case "sweep":
            stage_sweep(lm)


if __name__ == "__main__":
    main()
