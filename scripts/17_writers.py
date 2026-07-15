"""Writer attribution + writer-space lambda edits (preregistration_writers.md, abb07c0).

Stage attrib: decompose the layer-8 day-plane content into the additive residual
writers (embedding vs each block's attn/mlp output); per-head split for the top
attention layers. Stage sweep: rank-2 output-side projection removal on the
writing weights, same lambda grid and battery discipline as all prior sweeps.

GPT-2 only (Conv1D convention out = x @ W + b; projection removal W(lam) =
W - (1-lam) * (W @ P)).

Usage:
  python 17_writers.py --stage attrib --device mps
  python 17_writers.py --stage sweep --device mps
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
from clocklib.unsup import pca_basis, principal_cos, subspace_mat

ART = ROOT / "artifacts"
OUT = ART / "writers_gpt2"
LAMBDAS = sorted({-0.5, -0.25} | {round(x, 3) for x in np.linspace(0, 1, 21)}
                 | {1.1, 1.25, 1.5})
N_LAYERS_BELOW = 8  # writers below the probe layer (hidden_states[8])


def wiki_batch(lm: LM) -> torch.Tensor:
    from datasets import load_dataset

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
    return lm.tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64)


def day_positions(lm: LM, concept) -> list[tuple[torch.Tensor, int, int]]:
    ids = lm.token_ids(concept)
    tpl = []
    for t in concept.templates:
        for wi, w in enumerate(concept.words):
            enc = lm.tok(t.format(w), return_tensors="pt").input_ids
            tpl.append((enc, wi, enc[0].tolist().index(ids[wi])))
    return tpl


class Writers:
    """Untied GPT-2 on device with per-writer capture and stateless weight edits."""

    def __init__(self, lm: LM):
        self.lm = lm
        m = lm.untied_copy()
        self.W1 = m.get_input_embeddings().weight.detach().clone()  # CPU snapshot
        assert float(self.W1.norm()) > 0
        self.model = m.to(lm.device).eval()
        back = self.model.get_input_embeddings().weight.detach().cpu()
        assert torch.allclose(back, self.W1, atol=1e-5), "weights corrupted by device move"
        self.blocks = self.model.transformer.h
        # originals for every editable writer weight (CPU clones)
        self.orig = {}
        for l in range(N_LAYERS_BELOW):
            self.orig[f"attn_{l}"] = self.blocks[l].attn.c_proj.weight.detach().cpu().clone()
            self.orig[f"mlp_{l}"] = self.blocks[l].mlp.c_proj.weight.detach().cpu().clone()

    def module_weight(self, name: str) -> torch.nn.Parameter:
        l = int(name.split("_")[1])
        mod = self.blocks[l].attn.c_proj if name.startswith("attn") else \
            self.blocks[l].mlp.c_proj
        return mod.weight

    def set_writers(self, lam: float, mats: dict[str, torch.Tensor]) -> None:
        """mats[name] = the removed component M (CPU); weight = orig - (1-lam) M."""
        for name, M in mats.items():
            W = self.orig[name] - (1.0 - lam) * M
            self.module_weight(name).data.copy_(W.to(self.lm.device))

    def restore(self) -> None:
        for name, W in self.orig.items():
            self.module_weight(name).data.copy_(W.to(self.lm.device))
        self.model.get_input_embeddings().weight.data.copy_(self.W1.to(self.lm.device))

    @torch.no_grad()
    def clouds(self, tpl, n_words: int, probe_layer: int) -> dict[str, np.ndarray]:
        """Template-averaged contribution clouds at the concept-token position."""
        caps: dict[str, np.ndarray] = {
            name: np.zeros((n_words, self.lm.hidden_size))
            for l in range(N_LAYERS_BELOW) for name in (f"attn_{l}", f"mlp_{l}")}
        caps["emb"] = np.zeros((n_words, self.lm.hidden_size))
        caps["_total"] = np.zeros((n_words, self.lm.hidden_size))
        grabbed: dict[str, torch.Tensor] = {}

        def hook(name):
            def fn(_mod, _inp, out):
                grabbed[name] = out[0] if isinstance(out, tuple) else out
            return fn

        handles = []
        for l in range(N_LAYERS_BELOW):
            handles.append(self.blocks[l].attn.register_forward_hook(hook(f"attn_{l}")))
            handles.append(self.blocks[l].mlp.register_forward_hook(hook(f"mlp_{l}")))
        try:
            for enc, wi, pos in tpl:
                hs = self.model(input_ids=enc.to(self.lm.device),
                                output_hidden_states=True).hidden_states
                caps["emb"][wi] += hs[0][0, pos].float().cpu().numpy()
                caps["_total"][wi] += hs[probe_layer][0, pos].float().cpu().numpy()
                for l in range(N_LAYERS_BELOW):
                    for name in (f"attn_{l}", f"mlp_{l}"):
                        caps[name][wi] += grabbed[name][0, pos].float().cpu().numpy()
        finally:
            for h in handles:
                h.remove()
        n_tpl = len(tpl) // n_words
        return {k: v / n_tpl for k, v in caps.items()}


class WriterRef:
    """Frozen lambda=1 battery: layer-8 day/month plane power, emb k1, succ, KL."""

    def __init__(self, wr: Writers, wiki_ids: torch.Tensor):
        self.wr, lm = wr, wr.lm
        self.lm = lm
        self.p_ids = lm.token_ids(DAYS)
        self.x_ids = lm.token_ids(MONTHS)
        self.day_tpl = day_positions(lm, DAYS)
        self.month_tpl = day_positions(lm, MONTHS)
        self.succ_enc = [lm.tok(DAYS.successor_prompt.format(w), return_tensors="pt"
                                ).input_ids.to(lm.device) for w in DAYS.words]
        self.wiki = wiki_ids.to(lm.device)
        self.pw_ref = freq_power(wr.W1[self.p_ids].numpy())

        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        self.BH, _ = _cloud_pca(Hd, 2)
        self.BHm, _ = _cloud_pca(Hm, 2)
        self.day_ref = _plane_power(Hd, self.BH)
        self.month_ref = _plane_power(Hm, self.BHm)
        # ground-truth comparison of the discovered activation plane (recorded)
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
    def _model_metrics(self) -> dict:
        lm, n = self.lm, len(self.p_ids)
        out: dict = {}
        We = self.wr.model.get_input_embeddings().weight.detach().cpu()
        out["emb_day_k1_power"] = float(
            freq_power(We[self.p_ids].numpy())[1] / max(self.pw_ref[1], 1e-12))
        margins, correct = [], 0
        for i, enc in enumerate(self.succ_enc):
            dl = self.wr.model(input_ids=enc).logits[0, -1][self.p_ids]
            want = (i + 1) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        out["succ_margin_mean"] = float(np.mean(margins))
        out["succ_margin_wrap"] = margins[-1]
        out["succ_acc"] = correct / n
        logits = self.wr.model(input_ids=self.wiki).logits
        out["_wiki_logp"] = F.log_softmax(logits.float(), dim=-1).cpu()
        return out

    def measure_state(self) -> dict:
        m = self._model_metrics()
        Hd = self._cloud(self.day_tpl, len(DAYS.words))
        Hm = self._cloud(self.month_tpl, len(MONTHS.words))
        m["day_plane_power"] = _plane_power(Hd, self.BH) / max(self.day_ref, 1e-12)
        m["month_plane_power"] = _plane_power(Hm, self.BHm) / max(self.month_ref, 1e-12)
        logp = m.pop("_wiki_logp")
        m["wiki_kl"] = float(F.kl_div(logp.flatten(0, 1),
                                      self.wiki_logp_ref.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
        return m


def _cloud_pca(H: np.ndarray, d: int) -> tuple[np.ndarray, dict]:
    Hc = H - H.mean(0)
    _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    return Vt[:d].T, {"share": float((S[:d] ** 2).sum() / (S**2).sum())}


def _plane_power(H: np.ndarray, B: np.ndarray) -> float:
    Hc = H - H.mean(0)
    return float(((Hc @ B) ** 2).sum())


def stage_attrib(lm: LM) -> None:
    wr = Writers(lm)
    tpl = day_positions(lm, DAYS)
    n = len(DAYS.words)
    clouds = wr.clouds(tpl, n, lm.probe_layer)
    total = clouds.pop("_total")
    ssum = clouds["emb"] + sum(v for k, v in clouds.items() if k != "emb")
    add_err = float(np.abs(ssum - total).max())
    assert add_err < 1e-2, f"residual additivity violated: {add_err}"

    BH, rep = _cloud_pca(total, 2)
    P = BH @ BH.T
    Tc = total - total.mean(0)
    denom = float(((Tc @ BH) ** 2).sum())
    alphas = {}
    for name, C in clouds.items():
        Cc = C - C.mean(0)
        alphas[name] = float(np.sum((Cc @ P) * (Tc @ P)) / denom)
    rho_pred = float(((clouds["emb"] - clouds["emb"].mean(0)) @ BH).__pow__(2).sum()
                     / denom)
    # per-head split for the top-2 |alpha| attention layers
    top_attn = sorted((k for k in alphas if k.startswith("attn")),
                      key=lambda k: -abs(alphas[k]))[:2]
    heads = {}
    for name in top_attn:
        l = int(name.split("_")[1])
        Wc = wr.orig[name]  # (768, 768) Conv1D
        grabbed = {}

        def pre(mod, inp):
            grabbed["x"] = inp[0]

        h = wr.blocks[l].attn.c_proj.register_forward_pre_hook(pre)
        Hh = np.zeros((12, n, lm.hidden_size))
        try:
            with torch.no_grad():
                for enc, wi, pos in tpl:
                    wr.model(input_ids=enc.to(lm.device))
                    x = grabbed["x"][0, pos].cpu()
                    for hh in range(12):
                        seg = torch.zeros(768)
                        seg[hh * 64:(hh + 1) * 64] = x[hh * 64:(hh + 1) * 64]
                        Hh[hh, wi] += (seg @ Wc).numpy()
        finally:
            h.remove()
        Hh /= len(tpl) // n
        heads[name] = {f"h{hh}": float(np.sum(((Hh[hh] - Hh[hh].mean(0)) @ P)
                                              * (Tc @ P)) / denom)
                       for hh in range(12)}
    rep_out = {
        "alphas": dict(sorted(alphas.items(), key=lambda kv: -abs(kv[1]))),
        "rho_pred": rho_pred, "additivity_max_err": add_err,
        "plane_top2_share": rep["share"], "per_head_top_attn": heads,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(rep_out, indent=2))
    print(json.dumps({k: v for k, v in rep_out.items() if k != "per_head_top_attn"},
                     indent=2), flush=True)


def stage_sweep(lm: LM) -> None:
    attrib = json.loads((OUT / "attribution.json").read_text())
    wr = Writers(lm)
    ref = WriterRef(wr, wiki_batch(lm))
    print("activation plane vs Fourier k1:", ref.plane_vs_fourier, flush=True)

    P = torch.from_numpy(ref.BH @ ref.BH.T).float()
    all_names = list(wr.orig.keys())
    attn_names = [k for k in all_names if k.startswith("attn")]
    mlp_names = [k for k in all_names if k.startswith("mlp")]
    top = next(k for k in attrib["alphas"] if k != "emb")

    def plane_mats(names: list[str]) -> dict[str, torch.Tensor]:
        return {nm: wr.orig[nm] @ P for nm in names}

    def rand_mats(names: list[str], seed: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        out = {}
        for nm in names:
            Q, _ = torch.linalg.qr(torch.randn(768, 2, generator=g))
            M = wr.orig[nm] @ (Q @ Q.T)
            tgt = float((wr.orig[nm] @ P).norm())
            out[nm] = M * (tgt / max(float(M.norm()), 1e-12))
        return out

    B2, _ = pca_basis(wr.W1, ref.p_ids, 2)
    M_emb = subspace_mat(wr.W1, ref.p_ids, B2)

    conds: dict[str, dict] = {
        "T_wo_top": plane_mats([top]),
        "T_wo_allattn": plane_mats(attn_names),
        "T_wo_allmlp": plane_mats(mlp_names),
        "T_wo_all": plane_mats(all_names),
        **{f"C_wo_rand_r{s}": rand_mats(all_names, s) for s in range(3)},
    }
    sweep_dir = OUT / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    for name, mats in conds.items():
        rows = []
        for lam in LAMBDAS:
            wr.set_writers(lam, mats)
            rows.append({"lam": lam, **ref.measure_state()})
        wr.restore()
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name} done", flush=True)
    # embedding anchor with the same layer-8 readout
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
    ap.add_argument("--stage", required=True, choices=["attrib", "sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM("gpt2", device=args.device)
    if args.stage == "attrib":
        stage_attrib(lm)
    else:
        stage_sweep(lm)


if __name__ == "__main__":
    main()
