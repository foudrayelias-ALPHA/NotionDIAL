"""Qwen2.5-1.5B number 2-hop answer manifold mirror
(preregistration_numberstab.md, Panel 3, freeze 5745ec7).

Same 2-hop output-coordinate B6 pipeline as 29_numberctx.py (Llama Panel 2) with
NUMBERS target and DAYS spared, on Qwen2.5-1.5B. Qwen's embeddings are untied, so
B6n / B6_day are taken from the OUTPUT embedding rows (w20.out_rows handles this).
The number line does not wrap: number 2-hop scored over the 10 items X in one..ten
with NON-MODULAR want; day 2-hop wraps. Reuses the NumberQRef battery and stage
functions from 29_numberctx (only MODEL and OUT change). Frozen basis lambda=1,
26-point grid.

Usage:
  python 30_number2hop.py --stage attrib --device mps
  python 30_number2hop.py --stage sweep  --device mps
"""

import argparse
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.ringlib import LM

w29 = import_module("29_numberctx")

MODEL = "Qwen/Qwen2.5-1.5B"
OUT = ROOT / "artifacts" / "number2hop_qwen"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["attrib", "sweep"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    # retarget the shared 29_numberctx module's output dir to the Qwen artifacts
    w29.OUT = OUT
    lm = LM(MODEL, device=args.device)
    if args.stage == "attrib":
        w29.stage_attrib(lm)
    else:
        w29.stage_sweep(lm)


if __name__ == "__main__":
    main()
