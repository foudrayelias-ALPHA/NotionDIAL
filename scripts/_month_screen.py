"""Pre-freeze eligibility screens for the months tab (STEP 1, disclosed in prereg).

Mirrors the day-task prompt/few-shot structure of 19_contextual.py (Llama 1-hop)
and 20_qwen2hop.py (Qwen few-shot 2-hop) EXACTLY, substituting months for days.
Measures baseline accuracy /12 at unedited weights (lambda=1). No edits, no sweep.

Usage:
  python _month_screen.py --model meta-llama/Llama-3.2-1B --device mps
  python _month_screen.py --model Qwen/Qwen2.5-1.5B --device mps
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.ringlib import MONTHS, LM

# Month analogs of the day templates in 19/20 (same wording, months substituted).
CTX_MO = "The month after {} is"                       # 1-hop, want = X+1
FEWSHOT_MO = ("Let's do some month math. Two months after January is March. "
              "Two months after October is December. Two months after {} is")  # 2-hop, X+2
SUCC_MO = MONTHS.successor_prompt                       # "The month after {} is"


@torch.no_grad()
def restricted(lm: LM, prompts: list[str], ids: list[int], shift: int) -> tuple[int, float, list]:
    n = len(ids)
    correct, margins, decoded = 0, [], []
    for i, p in enumerate(prompts):
        enc = lm.tok(p, return_tensors="pt").input_ids.to(lm.device)
        dl = lm.model(input_ids=enc).logits[0, -1][ids]
        want = (i + shift) % n
        pred = int(dl.argmax())
        decoded.append(pred)
        correct += pred == want
        margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
    return correct, float(np.mean(margins)), decoded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(args.model, device=args.device)
    m = lm.model.to(lm.device).eval()  # base is small; screen only, no edits
    lm.model = m
    ids = lm.token_ids(MONTHS)
    single = ids is not None
    words = MONTHS.words

    out = {"model": args.model, "month_single_token": single,
           "n_months": len(words)}
    if single:
        # confirm leading-space single tokens explicitly
        lead_space = all(
            len(lm.tok.encode(" " + w, add_special_tokens=False)) == 1 for w in words)
        out["month_leading_space_single"] = lead_space

        ctx_p = [CTX_MO.format(w) for w in words]
        succ_p = [SUCC_MO.format(w) for w in words]
        few_p = [FEWSHOT_MO.format(w) for w in words]

        c1, m1, d1 = restricted(lm, ctx_p, ids, 1)
        cs, ms, ds = restricted(lm, succ_p, ids, 1)
        c2, m2, d2 = restricted(lm, few_p, ids, 2)
        out["month_after_1hop"] = {"acc": f"{c1}/12", "n": c1, "margin_mean": round(m1, 4),
                                   "decoded": d1, "want": [(i + 1) % 12 for i in range(12)]}
        out["month_successor"] = {"acc": f"{cs}/12", "n": cs, "margin_mean": round(ms, 4),
                                  "decoded": ds, "want": [(i + 1) % 12 for i in range(12)]}
        out["month_2hop_fewshot"] = {"acc": f"{c2}/12", "n": c2, "margin_mean": round(m2, 4),
                                     "decoded": d2, "want": [(i + 2) % 12 for i in range(12)]}
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
