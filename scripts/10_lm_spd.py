"""Step-2 phase A: SPD-decompose GPT-2 small's token embedding (transformer.wte).

Data: mixed stream — wikitext-2 chunks (70%) + day/month template sentences (30%,
differential use for the ring mechanism). Reconstruction loss: KL on logits (the LM
convention in param_decomp_lab). Only wte is decomposed. Device: mps (patched).

Note on weight tying: GPT-2 ties wte and lm_head. The decomposition's forward hooks
replace the INPUT-embedding use only, so the artifact's semantics — and all later
edits — cover the representation-building side; the unembedding stays fixed.

Usage: python 10_lm_spd.py [--steps 8000] [--c 256] [--batch 8] [--seq 64]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from param_decomp.configs import Cadence, RuntimeConfig
from param_decomp.optimize import Trainer

from clocklib.spdio import LocalSink, build_pd_config

ART = ROOT / "artifacts"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def template_text() -> str:
    tpls = [
        "The meeting is scheduled for {w}.", "She will arrive on {w}.",
        "Everything closed last {w} evening.", "I always go swimming on {w}.",
        "The deadline is next {w}.", "It happened one {w} morning.",
        "We usually rest on {w}.", "The store reopens on {w}.",
    ]
    succ = ["If today is {a}, then tomorrow is {b}.",
            "The day after {a} is {b}.", "{a} comes right before {b}."]
    parts = []
    for t in tpls:
        parts += [t.format(w=w) for w in DAYS + MONTHS]
    for t in succ:
        parts += [t.format(a=DAYS[i], b=DAYS[(i + 1) % 7]) for i in range(7)]
        parts += [t.format(a=MONTHS[i], b=MONTHS[(i + 1) % 12]) for i in range(12)]
    return " ".join(parts)


class MixedStream(IterableDataset[Tensor]):
    """Batches of (B, seq) token ids: wikitext chunks with template chunks mixed in."""

    def __init__(self, tok, batch: int, seq: int, seed: int, template_frac: float = 0.3):
        from datasets import load_dataset

        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in wiki["text"] if t.strip())
        self.wiki_ids = tok(text, return_tensors="pt").input_ids[0]
        self.tpl_ids = tok(template_text(), return_tensors="pt").input_ids[0]
        self.batch, self.seq, self.seed = batch, seq, seed
        self.template_frac = template_frac

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            rows = []
            for _ in range(self.batch):
                src = self.tpl_ids if torch.rand((), generator=g) < self.template_frac \
                    else self.wiki_ids
                start = int(torch.randint(0, len(src) - self.seq - 1, (), generator=g))
                rows.append(src[start : start + self.seq])
            yield torch.stack(rows)


def run_lm_batch(model: nn.Module, batch: Tensor) -> Tensor:
    return model(input_ids=batch).logits


def recon_kl(pred: Tensor, target: Tensor) -> tuple[Tensor, int]:
    log_q = F.log_softmax(pred.flatten(0, -2), dim=-1)
    p = F.softmax(target.flatten(0, -2), dim=-1)
    return F.kl_div(log_q, p, reduction="sum"), log_q.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--spd-seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=6000)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    model.requires_grad_(False).to(args.device)

    # REDUCED VARIANT (documented in decisions.md): stochastic-mask losses are
    # unsatisfiable at C=256 against the rank-768, 50k-row embedding and drag the
    # factorization away from the target (observed: faithfulness 0.007 -> 0.15
    # monotone). Step-2 needs exact reconstruction semantics + a ring-carrying span,
    # not full-SPD masking pressure; faithfulness + minimality only.
    pd_config = build_pd_config({"transformer.wte": args.c}, steps=args.steps,
                                batch_size=args.batch, seed=args.spd_seed,
                                lr=2e-3, grad_clip=1.0, stoch_coeff=0.0,
                                warmup_steps=args.warmup, warmup_lr=0.02)
    runtime = RuntimeConfig(autocast_bf16=False, device=args.device, dp=None)
    out_dir = ART / f"spd_gpt2_wte_s{args.spd_seed}{args.out_suffix}"
    trainer = Trainer(target_model=model, run_batch=run_lm_batch,
                      reconstruction_loss=recon_kl, pd_config=pd_config,
                      runtime_config=runtime)
    loader = DataLoader(MixedStream(tok, args.batch, args.seq, args.spd_seed),
                        batch_size=None)
    sink = LocalSink(out_dir)
    t0 = time.time()
    try:
        trainer.run(loader, sink, Cadence(train_log_every=200), eval_loop=None)
    finally:
        sink.finish()

    cm = trainer.component_model
    comp = cm.components["transformer.wte"]
    delta = cm.calc_weight_deltas()["transformer.wte"]
    day_ids = [tok.encode(" " + d)[0] for d in DAYS]
    month_ids = [tok.encode(" " + m)[0] for m in MONTHS]
    art = {
        "model": "gpt2", "module": "transformer.wte",
        "pd_config": pd_config.model_dump(),
        "U": comp.U.detach().cpu().clone(),          # (C, 768)
        "V": comp.V.detach().cpu().clone(),          # (50257, C)
        "delta": delta.detach().cpu().half(),
        "day_ids": day_ids, "month_ids": month_ids,
        "wall_seconds": round(time.time() - t0, 1),
    }
    torch.save(art, out_dir / "decomposition.pt")

    W = model.get_input_embeddings().weight.detach().cpu()
    recon = art["V"] @ art["U"]
    day_rows_err = float((recon[day_ids] - W[day_ids]).norm() / W[day_ids].norm())
    Uf, Sf, Vf = torch.svd_lowrank(W, q=min(args.c + 32, 768))
    W_floor = (Uf[:, : args.c] * Sf[: args.c]) @ Vf[:, : args.c].T
    report = {
        "wall_seconds": art["wall_seconds"],
        "delta_frac_overall": float(delta.norm().cpu() / W.norm()),
        "rankC_svd_floor_overall": float((W - W_floor).norm() / W.norm()),
        "day_rows_recon_rel_err": day_rows_err,
        "day_rows_rankC_floor": float(
            (W[day_ids] - W_floor[day_ids]).norm() / W[day_ids].norm()),
        "month_rows_recon_rel_err": float(
            (recon[month_ids] - W[month_ids]).norm() / W[month_ids].norm()),
    }
    (out_dir / "lm_faithfulness.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
