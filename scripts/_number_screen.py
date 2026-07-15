"""Pre-freeze eligibility screens for the numbers tab (STEP 1, disclosed in prereg).

Mirrors the day/month-task prompt/few-shot structure of 19_contextual.py (1-hop)
and 20_qwen2hop.py (few-shot 2-hop), substituting NUMBER WORDS one..twelve for
months. CRITICAL DIFFERENCE FROM DAYS/MONTHS: the number line does NOT wrap.
Successor of twelve leaves the set, so the task ranges are truncated to avoid
overflow and the `want` index is NON-modular:
  - 1-hop "The number after X is": X in one..eleven (11 items), want = X_index+1.
  - 2-hop "Two after X is":        X in one..ten   (10 items), want = X_index+2.
The argmax is restricted over all 12 number tokens; predicting the out-of-range
successor is impossible only because those items are excluded from the prompt set,
not by wrapping. Measures baseline accuracy at unedited weights (lambda=1). No
edits, no sweep.

Usage:
  python _number_screen.py --model meta-llama/Llama-3.2-1B --device mps
  python _number_screen.py --model Qwen/Qwen2.5-1.5B --device mps
  python _number_screen.py --model gpt2 --device mps   # counting run-up column
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

from clocklib.ringlib import LM
from clocklib.unsup import NUMBERS

# Number analogs of the day/month templates (same wording, numbers substituted).
CTX_NU = "The number after {} is"                    # 1-hop, want = X+1, X in one..eleven
FEWSHOT_NU = ("Let's do some number math. Two after one is three. "
              "Two after five is seven. Two after {} is")  # 2-hop, X+2, X in one..ten


@torch.no_grad()
def restricted_nowrap(lm: LM, words: list[str], ids: list[int], tpl: str,
                      shift: int) -> tuple[int, int, float, list]:
    """Non-wrapping successor screen. Items = words[: n - shift]; want = i + shift.
    Argmax restricted over all n number tokens. Returns (n_correct, n_items,
    mean margin, decoded)."""
    n = len(ids)
    idx = torch.tensor(ids)
    n_items = n - shift
    correct, margins, decoded = 0, [], []
    for i in range(n_items):
        enc = lm.tok(tpl.format(words[i]), return_tensors="pt").input_ids.to(lm.device)
        dl = lm.model(input_ids=enc).logits[0, -1][idx]
        want = i + shift
        pred = int(dl.argmax())
        decoded.append(pred)
        correct += pred == want
        margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
    return correct, n_items, float(np.mean(margins)), decoded


@torch.no_grad()
def counting_runup(lm: LM, words: list[str], ids: list[int]) -> tuple[int, int, float, list]:
    """GPT-2 behavior column: run-up counting `Count: {i-2}, {i-1}, {i},` -> want i+1.
    Items i=2..n-2 (9 for 12 numbers). Argmax restricted over the 12 number tokens.
    Mirror of clocklib.unsup.LineRef.succ_enc / _raw."""
    n = len(ids)
    idx = torch.tensor(ids)
    correct, margins, decoded = 0, [], []
    for i in range(2, n - 1):
        enc = lm.tok(f"Count: {words[i-2]}, {words[i-1]}, {words[i]},",
                     return_tensors="pt").input_ids.to(lm.device)
        dl = lm.model(input_ids=enc).logits[0, -1][idx]
        want = i + 1
        pred = int(dl.argmax())
        decoded.append(pred)
        correct += pred == want
        margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
    return correct, n - 3, float(np.mean(margins)), decoded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(args.model, device=args.device)
    m = lm.model.to(lm.device).eval()   # base is small; screen only, no edits
    lm.model = m
    ids = lm.token_ids(NUMBERS)
    single = ids is not None
    words = NUMBERS.words

    out = {"model": args.model, "number_single_token": single, "n_numbers": len(words)}
    if single:
        lead_space = all(
            len(lm.tok.encode(" " + w, add_special_tokens=False)) == 1 for w in words)
        out["number_leading_space_single"] = lead_space

        c1, n1, m1, d1 = restricted_nowrap(lm, words, ids, CTX_NU, 1)
        c2, n2, m2, d2 = restricted_nowrap(lm, words, ids, FEWSHOT_NU, 2)
        cc, nc, mc, dc = counting_runup(lm, words, ids)
        out["number_after_1hop"] = {
            "acc": f"{c1}/{n1}", "n": c1, "n_items": n1, "margin_mean": round(m1, 4),
            "decoded": d1, "want": list(range(1, n1 + 1)),
            "note": "X in one..eleven, want X+1, no wrap"}
        out["number_2hop_fewshot"] = {
            "acc": f"{c2}/{n2}", "n": c2, "n_items": n2, "margin_mean": round(m2, 4),
            "decoded": d2, "want": list(range(2, n2 + 2)),
            "note": "X in one..ten, want X+2, no wrap"}
        out["counting_runup"] = {
            "acc": f"{cc}/{nc}", "n": cc, "n_items": nc, "margin_mean": round(mc, 4),
            "decoded": dc, "want": list(range(3, len(words))),
            "note": "Count: i-2, i-1, i, -> want i+1, items i=2..n-2"}
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
