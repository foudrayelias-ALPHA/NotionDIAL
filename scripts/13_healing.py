"""Healing experiment: ablate GPT-2's day ring (lambda=0, aligned k=1 plane edit),
fine-tune briefly, and watch whether the ring regrows in wte or behavior reroutes.

Default: wte-only trainable (the ablated matrix itself); --full trains everything.
Usage: python 13_healing.py [--steps 300] [--device cpu] [--full]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import freq_power
from clocklib.ringlib import DAYS, LM, align

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module

pipeline = import_module("12_ring_pipeline")

ART = ROOT / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    art = torch.load(ART / "spd_gpt2_wte_s0_c768" / "decomposition.pt", weights_only=False)
    p_ids, x_ids = art["day_ids"], art["month_ids"]
    V = art["V"].numpy().astype(np.float64)
    U = art["U"].numpy().astype(np.float64)
    Vp, Up, clusters, _ = align(V, U, [("days", p_ids, DAYS.freqs),
                                       ("months", x_ids, [1])])
    ids = clusters["days_k1"]
    M = (torch.from_numpy(Vp[:, ids]).float() @ torch.from_numpy(Up[ids, :]).float())

    lm = LM("gpt2", device=args.device)
    model = lm.untied_copy().to(args.device)
    wte = model.get_input_embeddings()
    W_ref = wte.weight.detach().cpu().clone()
    pw_ref = freq_power(W_ref[p_ids].numpy())

    with torch.no_grad():
        wte.weight.data.copy_((W_ref - M).to(args.device))   # lambda = 0

    model.requires_grad_(args.full)
    wte.weight.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    loader = iter(DataLoader(pipeline.MixedStream(lm.tok, 8, 64, seed=0), batch_size=None))

    succ_enc = [lm.tok(DAYS.successor_prompt.format(w), return_tensors="pt"
                       ).input_ids.to(args.device) for w in DAYS.words]

    @torch.no_grad()
    def snapshot(step: int, loss: float | None) -> dict:
        W = wte.weight.detach().cpu()
        pw = freq_power(W[p_ids].numpy())
        margins, correct = [], 0
        for i, enc in enumerate(succ_enc):
            dl = model(input_ids=enc).logits[0, -1][p_ids]
            want = (i + 1) % 7
            margins.append(float(dl[want] - dl[torch.arange(7) != want].max()))
            correct += int(dl.argmax()) == want
        return {"step": step, "loss": loss,
                "day_k1_power": float(pw[1] / pw_ref[1]),
                "succ_acc": correct / 7, "succ_margin_mean": float(np.mean(margins)),
                "wrap_margin": margins[6]}

    traj = [snapshot(0, None)]
    model.train()
    for step in range(1, args.steps + 1):
        batch = next(loader).to(args.device)
        out = model(input_ids=batch, labels=batch)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        if step % 25 == 0:
            model.eval()
            traj.append(snapshot(step, float(out.loss)))
            model.train()
            print(traj[-1], flush=True)

    tag = "full" if args.full else "wte_only"
    (ART / f"healing_gpt2_{tag}.json").write_text(json.dumps(traj, indent=2))
    print(f"-> healing_gpt2_{tag}.json")


if __name__ == "__main__":
    main()
