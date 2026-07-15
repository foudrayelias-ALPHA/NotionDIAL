"""Localization test: FULL-recipe SPD (stochastic losses + CI) on reduced-vocab GPT-2.

Predictions and registered quantities: preregistration_localization.md (frozen e7937c3).

Usage: python 15_reduced_spd.py [--steps 10000] [--device mps] [--analyze-only]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from param_decomp.configs import Cadence, RuntimeConfig
from param_decomp.optimize import Trainer

from clocklib.fourier import freq_power
from clocklib.spdio import LocalSink, build_pd_config

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module

lmspd = import_module("10_lm_spd")

RED = ROOT / "artifacts" / "reduced_gpt2"
OUT = ROOT / "artifacts" / "spd_reduced_gpt2_full"


def load_reduced() -> tuple[nn.Module, dict, Tensor]:
    from transformers import AutoModelForCausalLM

    meta = json.loads((RED / "meta.json").read_text())
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    d = model.config.hidden_size
    model.transformer.wte = nn.Embedding(meta["n_keep"], d)
    model.lm_head = nn.Linear(d, meta["n_keep"], bias=False)
    model.config.vocab_size = meta["n_keep"]
    model.load_state_dict(torch.load(RED / "model.pt", weights_only=False))
    model.eval().requires_grad_(False)
    corpus = torch.load(RED / "corpus_ids.pt", weights_only=False)
    return model, meta, corpus


class CorpusStream(IterableDataset[Tensor]):
    def __init__(self, corpus: Tensor, batch: int, seq: int, seed: int):
        self.corpus, self.batch, self.seq, self.seed = corpus, batch, seq, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            starts = torch.randint(0, len(self.corpus) - self.seq - 1, (self.batch,),
                                   generator=g)
            yield torch.stack([self.corpus[s : s + self.seq] for s in starts])


def analyze(model: nn.Module, meta: dict, corpus: Tensor, cm=None) -> dict:
    W = model.get_input_embeddings().weight.detach().cpu()
    day_ids = meta["day_ids"]
    art = torch.load(OUT / "decomposition.pt", weights_only=False)
    V, U = art["V"], art["U"]
    recon = V @ U

    def k1_capture(M: Tensor) -> float:
        return float(freq_power(M[day_ids].numpy())[1] / freq_power(W[day_ids].numpy())[1])

    Uf, Sf, Vf = torch.svd_lowrank(W, q=min(W.shape[1], 256 + 32))
    W_svd = (Uf[:, :256] * Sf[:256]) @ Vf[:, :256].T
    report = {
        "ring_capture": k1_capture(recon),
        "svd_floor": k1_capture(W_svd),
        "delta_frac": float((W - recon).norm() / W.norm()),
        "svd_floor_frac": float((W - W_svd).norm() / W.norm()),
        "day_rows_recon_rel_err": float(
            (recon[day_ids] - W[day_ids]).norm() / W[day_ids].norm()),
    }
    if cm is not None:
        seq = 64
        day_set = set(day_ids)
        day_batches, free_batches = [], []
        g = torch.Generator().manual_seed(123)
        while len(day_batches) < 8 or len(free_batches) < 8:
            s = int(torch.randint(0, len(corpus) - seq - 1, (), generator=g))
            chunk = corpus[s : s + seq]
            has = bool(set(chunk.tolist()) & day_set)
            (day_batches if has else free_batches).append(chunk)
            if len(day_batches) > 32 and len(free_batches) < 1:
                break
        device = next(cm.parameters()).device
        ci_means = {}
        for name, batches in (("day", day_batches[:8]), ("free", free_batches[:8])):
            X = torch.stack(batches).to(device)
            out = cm(X, cache_type="input")
            ci = cm.calc_causal_importances(out.cache, sampling="continuous")
            v = ci.upper_leaky["transformer.wte"]
            ci_means[name] = v.reshape(-1, v.shape[-1]).mean(0).detach().cpu().numpy()
        a = (V[day_ids].norm(dim=0) ** 2 / V.norm(dim=0) ** 2).numpy()
        r = ci_means["day"] / np.maximum(ci_means["free"], 1e-9)
        alive = V.norm(dim=0).numpy() > 1e-4
        report["ci_corr_a_r"] = float(np.corrcoef(a[alive], r[alive])[0, 1])
        top = np.argsort(a)[-max(1, alive.sum() // 10):]
        report["ci_top_decile_mean_r"] = float(r[top].mean())
        report["mean_ci_day"] = float(ci_means["day"].mean())
        report["mean_ci_free"] = float(ci_means["free"].mean())
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    model, meta, corpus = load_reduced()
    if args.analyze_only:
        print(json.dumps(analyze(model, meta, corpus), indent=2))
        return

    model.to(args.device)
    pd_config = build_pd_config({"transformer.wte": args.c}, steps=args.steps,
                                batch_size=args.batch, seed=0, lr=1e-3, grad_clip=1.0,
                                stoch_coeff=1.0, warmup_steps=2000)
    runtime = RuntimeConfig(autocast_bf16=False, device=args.device, dp=None)
    trainer = Trainer(target_model=model, run_batch=lmspd.run_lm_batch,
                      reconstruction_loss=lmspd.recon_kl, pd_config=pd_config,
                      runtime_config=runtime)
    loader = DataLoader(CorpusStream(corpus, args.batch, args.seq, 0), batch_size=None)
    sink = LocalSink(OUT)
    t0 = time.time()
    try:
        trainer.run(loader, sink, Cadence(train_log_every=200), eval_loop=None)
    finally:
        sink.finish()
    cm = trainer.component_model
    comp = cm.components["transformer.wte"]
    torch.save({"V": comp.V.detach().cpu().clone(), "U": comp.U.detach().cpu().clone(),
                "delta": cm.calc_weight_deltas()["transformer.wte"].detach().cpu().clone(),
                "wall_seconds": round(time.time() - t0, 1)},
               OUT / "decomposition.pt")
    report = analyze(model, meta, corpus, cm=cm)
    report["wall_seconds"] = round(time.time() - t0, 1)
    (OUT / "localization_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
