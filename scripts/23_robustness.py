"""Robustness hardening R1-R4 (preregistration_robustness.md, bf03695).

Endpoint re-tests (lambda in {0, 0.5, 1}) of claim-critical cells under fresh
template sets and phrasings. R5 (Mistral echo) runs on Kaggle.

Usage: python 23_robustness.py --model {gpt2|llama|qwen} --device mps
"""

import argparse
import dataclasses
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.ringlib import DAYS, MONTHS, LM
from clocklib.unsup import array_pca, cloud_plane_power, principal_cos

w18 = import_module("18_writers_any")

SET_B = ["My appointment falls on {}.", "They left early on {} afternoon.",
         "Classes resume on {}.", "The concert is this {}.",
         "He was born on a {}.", "Trash pickup happens every {}.",
         "The flight departs {} night.", "Payday lands on {}."]
SET_C = ["See you on {}!", "The report is due {} at noon.",
         "It rained all day {}.", "Her shift starts {} morning.",
         "The bakery closes on {}.", "We celebrate every {}.",
         "Auditions are held on {}.", "The market opens {}."]
SUCC = ["If today is {}, then tomorrow is", "The day that comes after {} is",
        "Yesterday was {}, so today is", "{} is always followed by"]
FRAMES = [
    ("Let's do some day of the week math. Two days after Monday is Wednesday. "
     "Two days after Friday is Sunday. Two days after {} is"),
    ("Day math: two days after Tuesday is Thursday. Two days after Saturday is "
     "Monday. Two days after {} is"),
    ("If today is Monday, in two days it will be Wednesday. If today is Friday, "
     "in two days it will be Sunday. If today is {}, in two days it will be"),
]
LAMS = [0.0, 0.5, 1.0]
MODELS = {"gpt2": "gpt2", "llama": "meta-llama/Llama-3.2-1B",
          "qwen": "Qwen/Qwen2.5-1.5B"}


def concept_with(templates, name):
    return dataclasses.replace(DAYS, templates=templates, name=name)


@torch.no_grad()
def layer_cloud(wr, tpl, n_words, layer):
    H = np.zeros((n_words, wr.lm.hidden_size))
    for enc, wi, pos in tpl:
        hs = wr.model(input_ids=enc.to(wr.lm.device),
                      output_hidden_states=True).hidden_states
        H[wi] += hs[layer][0, pos].float().cpu().numpy()
    return H / (len(tpl) // n_words)


@torch.no_grad()
def task_cells(wr, lm, tpl_str, shift):
    ids = lm.token_ids(DAYS)
    margins, correct = [], 0
    for i, w in enumerate(DAYS.words):
        enc = lm.tok(tpl_str.format(w), return_tensors="pt").input_ids.to(lm.device)
        dl = wr.model(input_ids=enc).logits[0, -1][ids].float()
        want = (i + shift) % 7
        margins.append(float(dl[want] - dl[torch.arange(7) != want].max()))
        correct += int(dl.argmax()) == want
    return correct / 7, float(np.mean(margins))


def run_writers_robustness(key: str, device: str) -> dict:
    """R1 + R2 (gpt2) / R1 + R3 (llama): plane stability + T_wo_all endpoints."""
    lm = LM(MODELS[key], device=device)
    wr = w18.Writers(lm)
    L = lm.probe_layer
    tplA = w18.positions(lm, DAYS)
    tplB = w18.positions(lm, concept_with(SET_B, "days_b"))
    tplC = w18.positions(lm, concept_with(SET_C, "days_c"))
    HA, HB, HC = (layer_cloud(wr, t, 7, L) for t in (tplA, tplB, tplC))
    BA, _ = array_pca(HA, 2)
    BB, _ = array_pca(HB, 2)
    BC, _ = array_pca(HC, 2)
    res = {"R1_cos_A_B": principal_cos(BA, BB), "R1_cos_A_C": principal_cos(BA, BC)}

    base = {}
    for i, s in enumerate(SUCC):
        acc, mar = task_cells(wr, lm, s, 1)
        base[f"s{i+1}"] = {"acc": acc, "margin": mar, "eligible": acc >= 5 / 7 - 1e-9}
    res["succ_baselines"] = base

    P = torch.from_numpy(BA @ BA.T).float()
    mats = {nm: wr.arch.removed(wr.orig[nm], P) for nm in wr.orig}
    refB = cloud_plane_power(HB, BB)
    refC = cloud_plane_power(HC, BC)
    tplM = w18.positions(lm, MONTHS)
    HM = layer_cloud(wr, tplM, 12, L)
    BM, _ = array_pca(HM, 2)
    refM = cloud_plane_power(HM, BM)
    rows = {}
    for lam in LAMS:
        wr.set_removals(lam, mats)
        row = {"day_plane_setB": cloud_plane_power(layer_cloud(wr, tplB, 7, L), BB) / refB,
               "day_plane_setC": cloud_plane_power(layer_cloud(wr, tplC, 7, L), BC) / refC,
               "month_plane": cloud_plane_power(layer_cloud(wr, tplM, 12, L), BM) / refM}
        for i, s in enumerate(SUCC):
            if base[f"s{i+1}"]["eligible"]:
                acc, mar = task_cells(wr, lm, s, 1)
                row[f"s{i+1}"] = {"acc": acc, "margin": mar}
        rows[lam] = row
    wr.restore()
    res["T_wo_all_endpoints"] = rows
    return res


def run_qwen_robustness(device: str) -> dict:
    """R4: 2-hop dose control endpoints under alternative frames."""
    lm = LM(MODELS["qwen"], device=device)
    wr = w18.Writers(lm, n_below=lm.n_layers, capture_layer=lm.n_layers)
    refs = np.load(ROOT / "artifacts" / "qwen2hop" / "refs.npy",
                   allow_pickle=True).item()
    B6 = refs["B6"]
    P = torch.from_numpy(B6 @ B6.T).float()
    mats = {nm: wr.arch.removed(wr.orig[nm], P) for nm in wr.orig}
    base = {}
    for i, f in enumerate(FRAMES):
        acc, mar = task_cells(wr, lm, f, 2)
        base[f"f{i+1}"] = {"acc": acc, "margin": mar, "eligible": acc >= 5 / 7 - 1e-9}
    rows = {}
    for lam in LAMS:
        wr.set_removals(lam, mats)
        row = {}
        for i, f in enumerate(FRAMES):
            if base[f"f{i+1}"]["eligible"]:
                acc, mar = task_cells(wr, lm, f, 2)
                row[f"f{i+1}"] = {"acc": acc, "margin": mar}
        rows[lam] = row
    wr.restore()
    return {"frame_baselines": base, "T_out_all_endpoints": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.model == "qwen":
        res = run_qwen_robustness(args.device)
    else:
        res = run_writers_robustness(args.model, args.device)
    out = ROOT / "artifacts" / f"robustness_{args.model}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
