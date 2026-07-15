"""Model-agnostic ring pipeline CLI: survey / decompose / sweep any HF causal LM.

Artifacts land in artifacts/ring_<model-slug>/. GPT-2 serves as the regression case
(same numbers as scripts 09-11 expected up to decomposition seed).

Usage:
  python 12_ring_pipeline.py --model EleutherAI/pythia-410m --stage survey
  python 12_ring_pipeline.py --model EleutherAI/pythia-410m --stage decompose --device mps
  python 12_ring_pipeline.py --model EleutherAI/pythia-410m --stage sweep --device mps
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.ringlib import DAYS, MONTHS, LM, RingRef, align, survey

ART = ROOT / "artifacts"
LAMBDAS = sorted({-0.5, -0.25} | {round(x, 3) for x in np.linspace(0, 1, 21)} | {1.1, 1.25, 1.5})


def slug(name: str) -> str:
    return name.replace("/", "_")


def template_text() -> str:
    parts = []
    for c in (DAYS, MONTHS):
        for t in c.templates:
            parts += [t.format(w) for w in c.words]
        n = len(c.words)
        parts += [c.successor_prompt.format(c.words[i]) + f" {c.words[(i + 1) % n]}."
                  for i in range(n)]
    return " ".join(parts)


class MixedStream(IterableDataset[Tensor]):
    def __init__(self, tok, batch: int, seq: int, seed: int, template_frac: float = 0.3):
        from datasets import load_dataset

        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(t for t in wiki["text"] if t.strip())
        self.wiki_ids = tok(text, return_tensors="pt").input_ids[0]
        self.tpl_ids = tok(template_text(), return_tensors="pt").input_ids[0]
        self.batch, self.seq, self.seed, self.template_frac = batch, seq, seed, template_frac

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


def stage_survey(lm: LM, out: Path) -> None:
    res = {"model": lm.name, "emb_path": lm.emb_path, "n_layers": lm.n_layers,
           "hidden_size": lm.hidden_size, "probe_layer": lm.probe_layer,
           "concepts": {c.name: survey(lm, c) for c in (DAYS, MONTHS)}}
    (out / "survey.json").write_text(json.dumps(res, indent=2))
    for cname, c in res["concepts"].items():
        if not c.get("single_token"):
            print(f"{cname}: NOT single-token — model unusable for this concept")
            continue
        for lname in ("embedding", f"resid_{lm.probe_layer}"):
            s = c["layers"][lname]
            print(f"{cname:>7} {lname:>12}: top2={s['top2_pca_share']:.2f} "
                  f"k1={s['k1_power_frac']:.2f} wind={s['winding_top2']:+.1f}")


def stage_decompose(lm: LM, out: Path, steps: int, warmup: int, batch: int, seq: int,
                    c: int | None, seed: int) -> None:
    from param_decomp.configs import Cadence, RuntimeConfig
    from param_decomp.optimize import Trainer

    from clocklib.spdio import LocalSink, build_pd_config

    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module

    lmspd = import_module("10_lm_spd")

    C = c or lm.hidden_size
    pd_config = build_pd_config({lm.emb_path: C}, steps=steps, batch_size=batch,
                                seed=seed, lr=2e-3, grad_clip=1.0, stoch_coeff=0.0,
                                warmup_steps=warmup, warmup_lr=0.02)
    runtime = RuntimeConfig(autocast_bf16=False, device=lm.device, dp=None)
    trainer = Trainer(target_model=lm.model, run_batch=lmspd.run_lm_batch,
                      reconstruction_loss=lmspd.recon_kl, pd_config=pd_config,
                      runtime_config=runtime)
    loader = DataLoader(MixedStream(lm.tok, batch, seq, seed), batch_size=None)
    sink = LocalSink(out)
    t0 = time.time()
    try:
        trainer.run(loader, sink, Cadence(train_log_every=500), eval_loop=None)
    finally:
        sink.finish()
    cm = trainer.component_model
    comp = cm.components[lm.emb_path]
    delta = cm.calc_weight_deltas()[lm.emb_path]
    p_ids, x_ids = lm.token_ids(DAYS), lm.token_ids(MONTHS)
    art = {"model": lm.name, "module": lm.emb_path, "C": C,
           "U": comp.U.detach().cpu().clone(), "V": comp.V.detach().cpu().clone(),
           "delta": delta.detach().cpu().half(), "day_ids": p_ids, "month_ids": x_ids,
           "wall_seconds": round(time.time() - t0, 1)}
    torch.save(art, out / "decomposition.pt")
    W = lm.model.get_input_embeddings().weight.detach().cpu()
    recon = art["V"] @ art["U"]
    report = {
        "C": C, "wall_seconds": art["wall_seconds"],
        "delta_frac_overall": float(delta.norm().cpu() / W.norm()),
        "day_rows_recon_rel_err": float(
            (recon[p_ids] - W[p_ids]).norm() / W[p_ids].norm()),
        "month_rows_recon_rel_err": float(
            (recon[x_ids] - W[x_ids]).norm() / W[x_ids].norm()),
    }
    (out / "faithfulness.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def stage_sweep(lm: LM, out: Path, direct: bool = False) -> None:
    """direct=True: instrument-only arm — identity factorization V=W, U=I (per the
    audit, at full capacity the span equals col(W), so no decomposition is needed;
    the same alignment/edit construction runs on the weights directly)."""
    if direct:
        W0 = lm.model.get_input_embeddings().weight.detach().cpu()
        art = {"V": W0.clone(), "U": torch.eye(W0.shape[1]),
               "day_ids": lm.token_ids(DAYS), "month_ids": lm.token_ids(MONTHS)}
    else:
        art = torch.load(out / "decomposition.pt", weights_only=False)
    p_ids, x_ids = art["day_ids"], art["month_ids"]
    V = art["V"].numpy().astype(np.float64)
    U = art["U"].numpy().astype(np.float64)
    Vp, Up, clusters, rep = align(V, U, [("days", p_ids, DAYS.freqs),
                                         ("months", x_ids, MONTHS.freqs)])
    (out / "align_report.json").write_text(json.dumps(rep, indent=2))
    Vp_t, Up_t = torch.from_numpy(Vp).float(), torch.from_numpy(Up).float()

    from datasets import load_dataset

    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
    wiki_ids = lm.tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64)

    ref = RingRef(lm, DAYS, MONTHS, wiki_ids)
    W1 = ref.W1
    import gc

    lm.model = None  # free the base copy; ref holds its own untied model
    gc.collect()
    if lm.device == "mps":
        torch.mps.empty_cache()

    def plane_mat(names: list[str]) -> Tensor:
        ids = [c for n in names for c in clusters[n]]
        return Vp_t[:, ids] @ Up_t[ids, :]

    k1_frob = float(plane_mat(["days_k1"])[p_ids].norm())

    def oracle_mat() -> Tensor:
        E = W1[p_ids].numpy()
        Fk = np.fft.rfft(E, axis=0)
        keep = np.zeros_like(Fk)
        keep[1] = Fk[1]
        P = np.zeros((W1.shape[0], W1.shape[1]))
        P[p_ids] = np.fft.irfft(keep, n=len(p_ids), axis=0)
        return torch.from_numpy(P).float()

    def rand_mat(seed: int) -> Tensor:
        g = torch.Generator().manual_seed(seed)
        u = torch.randn(len(p_ids), generator=g)
        v = torch.randn(W1.shape[1], generator=g)
        M = torch.zeros_like(W1)
        M[p_ids] = torch.outer(u, v)
        return M * (k1_frob / M[p_ids].norm())

    conds = {
        "T_sym_k1": plane_mat(["days_k1"]),
        "T_sym_ring": plane_mat([f"days_k{k}" for k in DAYS.freqs]),
        "C1_cross_month": plane_mat(["months_k1"]),
        "C4_oracle": oracle_mat(),
        **{f"C3_random_r{s}": rand_mat(s) for s in range(3)},
    }
    sweep_dir = out / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    for name, M in conds.items():
        rows = [{"lam": lam, **ref.measure(W1 + (lam - 1.0) * M)} for lam in LAMBDAS]
        (sweep_dir / f"{name}.json").write_text(json.dumps(rows))
        print(f"[{lm.name}] {name} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["survey", "decompose", "sweep", "sweep-direct", "all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--c", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lm = LM(args.model, device=args.device)
    out = ART / f"ring_{slug(args.model)}"
    out.mkdir(exist_ok=True)
    stages = ["survey", "decompose", "sweep"] if args.stage == "all" else [args.stage]
    for s in stages:
        match s:
            case "survey":
                stage_survey(lm, out)
            case "decompose":
                stage_decompose(lm, out, args.steps, args.warmup, args.batch, args.seq,
                                args.c, args.seed)
            case "sweep":
                stage_sweep(lm, out)
            case "sweep-direct":
                stage_sweep(lm, out, direct=True)


if __name__ == "__main__":
    main()
