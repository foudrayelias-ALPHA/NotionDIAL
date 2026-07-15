"""Model-agnostic writer attribution + writer-space lambda edits.

Generalizes 17_writers.py across weight conventions (preregistration_writers.md
addendum, dc715f8):
  - GPT-2 family (Conv1D, out = x W + b):    W(lam) = W - (1-lam) * (W @ P)
  - Llama family (nn.Linear, out = x W^T):   W(lam) = W - (1-lam) * (P @ W)

Writers = every block's attn out-projection and MLP down-projection below the
probe layer. Battery adds an optional contextual 1-hop cell when the model
passes the measured precondition.

Usage:
  python 18_writers_any.py --model meta-llama/Llama-3.2-1B --stage attrib --device mps
  python 18_writers_any.py --model meta-llama/Llama-3.2-1B --stage sweep --device mps
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import fourier_plane_basis, freq_power
from clocklib.ringlib import DAYS, MONTHS, LM
from clocklib.unsup import (array_pca, cloud_plane_power, pca_basis,
                            principal_cos, subspace_mat)

ART = ROOT / "artifacts"
LAMBDAS = sorted({-0.5, -0.25} | {round(x, 3) for x in np.linspace(0, 1, 21)}
                 | {1.1, 1.25, 1.5})
CONTEXT_TPL = {"meta-llama/Llama-3.2-1B": "The day after {} is"}  # passed precondition


def wiki_batch(lm: LM) -> torch.Tensor:
    from datasets import load_dataset

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
    return lm.tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64)


def positions(lm: LM, concept) -> list[tuple[torch.Tensor, int, int]]:
    ids = lm.token_ids(concept)
    tpl = []
    for t in concept.templates:
        for wi, w in enumerate(concept.words):
            enc = lm.tok(t.format(w), return_tensors="pt").input_ids
            tpl.append((enc, wi, enc[0].tolist().index(ids[wi])))
    return tpl


class Arch:
    """Weight-convention adapter: block list, writer projections, edit side."""

    def __init__(self, model):
        if hasattr(model, "transformer"):
            self.blocks, self.left = model.transformer.h, False
            self.attn_mod = lambda b: b.attn
            self.mlp_mod = lambda b: b.mlp
            self.attn_proj = lambda b: b.attn.c_proj
            self.mlp_proj = lambda b: b.mlp.c_proj
        else:
            self.blocks, self.left = model.model.layers, True
            self.attn_mod = lambda b: b.self_attn
            self.mlp_mod = lambda b: b.mlp
            self.attn_proj = lambda b: b.self_attn.o_proj
            self.mlp_proj = lambda b: b.mlp.down_proj

    def apply_removal(self, orig: torch.Tensor, P: torch.Tensor, lam: float
                      ) -> torch.Tensor:
        M = (P @ orig) if self.left else (orig @ P)
        return orig - (1.0 - lam) * M

    def removed(self, orig: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        return (P @ orig) if self.left else (orig @ P)


class Writers:
    def __init__(self, lm: LM, n_below: int | None = None,
                 capture_layer: int | None = None):
        self.lm = lm
        m = lm.untied_copy()
        self.W1 = m.get_input_embeddings().weight.detach().clone()  # CPU snapshot
        assert float(self.W1.norm()) > 0
        self.model = m.to(lm.device).eval()
        back = self.model.get_input_embeddings().weight.detach().cpu()
        assert torch.allclose(back, self.W1, atol=1e-5), "weights corrupted by device move"
        self.arch = Arch(self.model)
        self.n_below = n_below if n_below is not None else lm.probe_layer
        self.capture_layer = capture_layer if capture_layer is not None \
            else lm.probe_layer
        self.orig = {}
        for l in range(self.n_below):
            b = self.arch.blocks[l]
            self.orig[f"attn_{l}"] = self.arch.attn_proj(b).weight.detach().cpu().clone()
            self.orig[f"mlp_{l}"] = self.arch.mlp_proj(b).weight.detach().cpu().clone()

    def module_weight(self, name: str) -> torch.nn.Parameter:
        l = int(name.split("_")[1])
        b = self.arch.blocks[l]
        proj = self.arch.attn_proj(b) if name.startswith("attn") else \
            self.arch.mlp_proj(b)
        return proj.weight

    def set_removals(self, lam: float, mats: dict[str, torch.Tensor]) -> None:
        for name, M in mats.items():
            W = self.orig[name] - (1.0 - lam) * M
            self.module_weight(name).data.copy_(W.to(self.lm.device))

    def restore(self) -> None:
        for name, W in self.orig.items():
            self.module_weight(name).data.copy_(W.to(self.lm.device))
        self.model.get_input_embeddings().weight.data.copy_(self.W1.to(self.lm.device))

    @torch.no_grad()
    def clouds(self, tpl, n_words: int) -> dict[str, np.ndarray]:
        d = self.lm.hidden_size
        caps = {name: np.zeros((n_words, d))
                for l in range(self.n_below) for name in (f"attn_{l}", f"mlp_{l}")}
        caps["emb"] = np.zeros((n_words, d))
        caps["_total"] = np.zeros((n_words, d))
        grabbed = {}

        def hook(name):
            def fn(_m, _i, out):
                grabbed[name] = out[0] if isinstance(out, tuple) else out
            return fn

        handles = []
        for l in range(self.n_below):
            b = self.arch.blocks[l]
            handles.append(self.arch.attn_mod(b).register_forward_hook(hook(f"attn_{l}")))
            handles.append(self.arch.mlp_mod(b).register_forward_hook(hook(f"mlp_{l}")))
        # capture the RAW residual after block capture_layer-1 via a block hook:
        # identical to hidden_states[capture_layer] mid-stack, but avoids the
        # post-final-norm entry at the top of the stack (HF puts norm(h) there)
        handles.append(self.arch.blocks[self.capture_layer - 1]
                       .register_forward_hook(hook("_total")))
        try:
            for enc, wi, pos in tpl:
                hs = self.model(input_ids=enc.to(self.lm.device),
                                output_hidden_states=True).hidden_states
                caps["emb"][wi] += hs[0][0, pos].float().cpu().numpy()
                caps["_total"][wi] += grabbed["_total"][0, pos].float().cpu().numpy()
                for l in range(self.n_below):
                    for name in (f"attn_{l}", f"mlp_{l}"):
                        caps[name][wi] += grabbed[name][0, pos].float().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        n_tpl = len(tpl) // n_words
        return {k: v / n_tpl for k, v in caps.items()}


class WriterRef:
    def __init__(self, wr: Writers, wiki_ids: torch.Tensor):
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.p_ids = lm.token_ids(DAYS)
        self.x_ids = lm.token_ids(MONTHS)
        self.day_tpl = positions(lm, DAYS)
        self.month_tpl = positions(lm, MONTHS)
        self.succ_enc = [lm.tok(DAYS.successor_prompt.format(w), return_tensors="pt"
                                ).input_ids.to(lm.device) for w in DAYS.words]
        ctx = CONTEXT_TPL.get(lm.name)
        self.ctx_enc = None if ctx is None else \
            [lm.tok(ctx.format(w), return_tensors="pt").input_ids.to(lm.device)
             for w in DAYS.words]
        self.wiki = wiki_ids.to(lm.device)
        self.pw_ref = freq_power(wr.W1[self.p_ids].numpy())

        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        self.BH, _ = array_pca(Hd, 2)
        self.BHm, _ = array_pca(Hm, 2)
        self.day_ref = cloud_plane_power(Hd, self.BH)
        self.month_ref = cloud_plane_power(Hm, self.BHm)
        Q1 = fourier_plane_basis(Hd - Hd.mean(0), 1).T
        self.plane_vs_fourier = principal_cos(self.BH, Q1)
        m = self._model_metrics()
        self.wiki_logp_ref = m.pop("_wiki_logp")
        self.ref = {**m, "day_plane_power": 1.0, "month_plane_power": 1.0}

    @torch.no_grad()
    def _cloud(self, tpl, n_words: int) -> np.ndarray:
        H = np.zeros((n_words, self.lm.hidden_size))
        for enc, wi, pos in tpl:
            hs = self.wr.model(input_ids=enc.to(self.lm.device),
                               output_hidden_states=True).hidden_states
            H[wi] += hs[self.lm.probe_layer][0, pos].float().cpu().numpy()
        return H / (len(tpl) // n_words)

    @torch.no_grad()
    def _restricted(self, encs) -> tuple[float, float]:
        n = len(self.p_ids)
        margins, correct = [], 0
        for i, enc in enumerate(encs):
            dl = self.wr.model(input_ids=enc).logits[0, -1][self.p_ids]
            want = (i + 1) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        return correct / n, float(np.mean(margins))

    @torch.no_grad()
    def _model_metrics(self) -> dict:
        out: dict = {}
        We = self.wr.model.get_input_embeddings().weight.detach().cpu()
        out["emb_day_k1_power"] = float(
            freq_power(We[self.p_ids].numpy())[1] / max(self.pw_ref[1], 1e-12))
        out["succ_acc"], out["succ_margin_mean"] = self._restricted(self.succ_enc)
        if self.ctx_enc is not None:
            out["ctx_acc"], out["ctx_margin_mean"] = self._restricted(self.ctx_enc)
        logits = self.wr.model(input_ids=self.wiki).logits
        out["_wiki_logp"] = F.log_softmax(logits.float(), dim=-1).cpu()
        return out

    def measure_state(self) -> dict:
        m = self._model_metrics()
        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        m["_H_cloud_B2"] = [[round(float(v), 4) for v in row]
                            for row in ((Hd - Hd.mean(0)) @ self.BH)[:, :2]]
        m["day_plane_power"] = cloud_plane_power(Hd, self.BH) / max(self.day_ref, 1e-12)
        m["month_plane_power"] = cloud_plane_power(Hm, self.BHm) / max(self.month_ref, 1e-12)
        logp = m.pop("_wiki_logp")
        m["wiki_kl"] = float(F.kl_div(logp.flatten(0, 1),
                                      self.wiki_logp_ref.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
        return m


def stage_attrib(lm: LM, out: Path) -> None:
    wr = Writers(lm)
    tpl = positions(lm, DAYS)
    n = len(DAYS.words)
    clouds = wr.clouds(tpl, n)
    total = clouds.pop("_total")
    ssum = clouds["emb"] + sum(v for k, v in clouds.items() if k != "emb")
    add_err = float(np.abs(ssum - total).max())
    rel_err = add_err / float(np.abs(total).max())
    assert rel_err < 1e-3, f"residual additivity violated: {add_err} (rel {rel_err})"

    BH, rep = array_pca(total, 2)
    P = BH @ BH.T
    Tc = total - total.mean(0)
    denom = float(((Tc @ BH) ** 2).sum())
    alphas = {name: float(np.sum(((C - C.mean(0)) @ P) * (Tc @ P)) / denom)
              for name, C in clouds.items()}
    rho_pred = float(cloud_plane_power(clouds["emb"], BH) / denom)
    # linear-field predictions for the partial conditions (coupling baseline)
    attn = [k for k in clouds if k.startswith("attn")]
    mlp = [k for k in clouds if k.startswith("mlp")]
    top = max((k for k in alphas if k != "emb"), key=lambda k: abs(alphas[k]))

    def lin_pred(uncut: list[str]) -> float:
        S = sum(clouds[c] for c in uncut)
        return float(cloud_plane_power(S, BH) / denom)

    linear_pred = {
        "T_wo_top": lin_pred(["emb"] + [c for c in attn + mlp if c != top]),
        "T_wo_allattn": lin_pred(["emb"] + mlp),
        "T_wo_allmlp": lin_pred(["emb"] + attn),
        "T_wo_all": rho_pred,
    }
    rep_out = {"model": lm.name, "probe_layer": lm.probe_layer,
               "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
               "rho_pred": rho_pred, "top_writer": top,
               "linear_pred": linear_pred,
               "additivity_rel_err": rel_err, "plane_top2_share": rep["share"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "attribution.json").write_text(json.dumps(rep_out, indent=2))
    top8 = dict(list(rep_out["alphas"].items())[:8])
    print(json.dumps({"top8_alphas": top8, "alpha_emb": alphas["emb"],
                      "rho_pred": rho_pred, "linear_pred": linear_pred}, indent=2),
          flush=True)


def stage_sweep(lm: LM, out: Path) -> None:
    attrib = json.loads((out / "attribution.json").read_text())
    wr = Writers(lm)
    ref = WriterRef(wr, wiki_batch(lm))
    print("activation plane vs Fourier k1:", ref.plane_vs_fourier, flush=True)

    P = torch.from_numpy(ref.BH @ ref.BH.T).float()
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

    B2, _ = pca_basis(wr.W1, ref.p_ids, 2)
    M_emb = subspace_mat(wr.W1, ref.p_ids, B2)

    conds = {
        "T_wo_top": plane_mats([top]),
        "T_wo_allattn": plane_mats(attn_names),
        "T_wo_allmlp": plane_mats(mlp_names),
        "T_wo_all": plane_mats(all_names),
        **{f"C_wo_rand_r{s}": rand_mats(all_names, s) for s in range(3)},
    }
    sweep_dir = out / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    for name, mats in conds.items():
        rows = []
        for lam in LAMBDAS:
            wr.set_removals(lam, mats)
            rows.append({"lam": lam, **ref.measure_state()})
        wr.restore()
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name} done", flush=True)
    rows = []
    for lam in LAMBDAS:
        We = wr.W1 + (lam - 1.0) * M_emb
        wr.model.get_input_embeddings().weight.data.copy_(We.to(lm.device))
        rows.append({"lam": lam, **ref.measure_state()})
    wr.restore()
    (sweep_dir / "T_emb_pca2.json").write_text(json.dumps(rows))
    print("T_emb_pca2 done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True, choices=["attrib", "sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(args.model, device=args.device)
    out = ART / f"writers_{args.model.replace('/', '_')}"
    if args.stage == "attrib":
        stage_attrib(lm, out)
    else:
        stage_sweep(lm, out)


if __name__ == "__main__":
    main()
