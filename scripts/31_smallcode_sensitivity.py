"""Small-model token-manifold answer-code sensitivity cell
(preregistration_smallcode.md, e0aed98). The last owed experiment.

Two-sided test of whether the day-token-manifold answer code is truly ABSENT at
<=1.5B or merely below the day-token-trained transfer probe's floor:

  A) answer-trained leave-one-out probe on the frozen day-token plane (removes
     the transfer handicap -> upper bound on sensitivity), per layer;
  B) day-plane amplitude vs R matched-norm random rank-2 planes (removes the
     probe entirely), percentile per layer.

Frozen decision rule (two-sided): FAINT if any late layer LOO acc >= 4/7 OR
amplitude percentile >= 99 on >= 2 contiguous late layers; else
ABSENT-AT-SENSITIVITY. Late = layer >= probe_layer.

Usage:
  python 31_smallcode_sensitivity.py --model llama  --device mps
  python 31_smallcode_sensitivity.py --model qwen15 --device mps
"""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clocklib.ringlib import DAYS, LM
from clocklib.unsup import array_pca, cloud_plane_power

w18 = import_module("18_writers_any")
w19 = import_module("19_contextual")

R_PLANES = 300
NULL_BASE_SEED = 12345

SPECS = {
    "llama": {
        "model": "meta-llama/Llama-3.2-1B",
        "tpl": "The day after {} is",
        "shift": 1,
        "ref": "artifacts/mistral7b/survey.json",  # 7B same-lineage-ish signature
        "out": "smallcode_llama",
    },
    "qwen15": {
        "model": "Qwen/Qwen2.5-1.5B",
        "tpl": ("Let's do some day of the week math. Two days after Monday is "
                "Wednesday. Two days after Friday is Sunday. Two days after {} is"),
        "shift": 2,
        "ref": "artifacts/qwen7b/survey.json",  # within-family 7B signature
        "out": "smallcode_qwen15",
    },
}


def loo_probe_hits(Z: np.ndarray, want: list[int]) -> tuple[int, list[int]]:
    """Answer-trained leave-one-out probe on the 2D day-plane projections Z.
    For each held-out day j, fit the cos/sin ridge probe on the other 6 answer
    points at their answer-day angles, decode Z_j. Returns (hits, decoded)."""
    n = Z.shape[0]
    decoded = [None] * n
    for j in range(n):
        idx = [i for i in range(n) if i != j]
        # angle targets are the answer-day angles want_i
        t = 2 * np.pi * np.asarray([want[i] for i in idx]) / n
        from sklearn.linear_model import Ridge
        probe = Ridge(alpha=1e-3).fit(Z[idx], np.stack([np.cos(t), np.sin(t)], 1))
        pred = probe.predict(Z[j:j + 1])
        ang = np.arctan2(pred[0, 1], pred[0, 0]) % (2 * np.pi)
        decoded[j] = int(round(ang / (2 * np.pi / n))) % n
    hits = sum(int(decoded[j] == want[j]) for j in range(n))
    return hits, decoded


def null_percentile(A_l: np.ndarray, B_H: np.ndarray, dim: int) -> tuple[float, float, float]:
    """Measurement B: true day-plane amplitude vs R matched-norm random rank-2
    planes. Returns (true_amp, percentile_in_percent, null_median)."""
    a_true = cloud_plane_power(A_l, B_H)
    nulls = np.empty(R_PLANES)
    for r in range(R_PLANES):
        g = torch.Generator().manual_seed(NULL_BASE_SEED + r)
        Q, _ = torch.linalg.qr(torch.randn(dim, 2, generator=g))
        nulls[r] = cloud_plane_power(A_l, Q.numpy())
    pct = 100.0 * float((nulls < a_true).sum()) / R_PLANES
    return float(a_true), pct, float(np.median(nulls))


def run(spec: dict, device: str) -> None:
    lm = LM(spec["model"], device=device)
    L = lm.n_layers
    dim = lm.hidden_size
    shift = spec["shift"]
    want = [(i + shift) % 7 for i in range(7)]

    # --- collect answer states A_l (0..L) and day-token cloud Hd at probe_layer
    day_tpl = w18.positions(lm, DAYS)
    ans_tpl = w19.ctx_positions(lm, spec["tpl"])  # (enc, wi, final_pos)
    model = lm.model.to(device)
    A = np.zeros((L + 1, 7, dim))
    Hd = np.zeros((7, dim))
    grabbed = {}

    def fn(_m, _i, out):
        grabbed["h"] = out[0] if isinstance(out, tuple) else out

    arch = w18.Arch(model)
    handle = arch.blocks[L - 1].register_forward_hook(fn)
    try:
        with torch.no_grad():
            for enc, wi, _pos in ans_tpl:
                hs = model(input_ids=enc.to(device),
                           output_hidden_states=True).hidden_states
                for l in range(L):
                    A[l, wi] = hs[l][0, -1].float().cpu().numpy()
                A[L, wi] = grabbed["h"][0, -1].float().cpu().numpy()
            for enc, wi, pos in day_tpl:
                hs = model(input_ids=enc.to(device),
                           output_hidden_states=True).hidden_states
                Hd[wi] += hs[lm.probe_layer][0, pos].float().cpu().numpy()
    finally:
        handle.remove()
    Hd /= len(DAYS.templates)

    # --- frozen day-token plane
    B_H, rep = array_pca(Hd, 2)

    # --- per-layer measurements
    rows = []
    for l in range(L + 1):
        Z = (A[l] - A[l].mean(0)) @ B_H  # (7, 2)
        hits, decoded = loo_probe_hits(Z, want)
        amp, pct, null_med = null_percentile(A[l], B_H, dim)
        rows.append({"layer": l, "loo_probe_hits": hits, "loo_decoded": decoded,
                     "amp": round(amp, 6), "null_median": round(null_med, 6),
                     "null_pct": round(pct, 3)})

    # --- frozen decision rule
    late = [r for r in rows if r["layer"] >= lm.probe_layer]
    a_hit = [r for r in late if r["loo_probe_hits"] >= 4]
    # contiguous late layers with pct >= 99
    late_sorted = sorted(late, key=lambda r: r["layer"])
    best_run = cur = 0
    b_layers = []
    run_layers = []
    for r in late_sorted:
        if r["null_pct"] >= 99.0:
            cur += 1
            run_layers.append(r["layer"])
            if cur > best_run:
                best_run, b_layers = cur, list(run_layers)
        else:
            cur = 0
            run_layers = []
    branch_a = len(a_hit) > 0
    branch_b = best_run >= 2
    branch = "FAINT" if (branch_a or branch_b) else "ABSENT-AT-SENSITIVITY"

    # reference signature echo (read-only 7B survey.json)
    ref = json.loads((ROOT / spec["ref"]).read_text())
    ref_rows = ref["rows"]
    ref_peak = max(r["probe"] for r in ref_rows)
    ref_peakL = [r["layer"] for r in ref_rows if r["probe"] == ref_peak]

    peak_loo = max(r["loo_probe_hits"] for r in late)
    peak_loo_L = [r["layer"] for r in late if r["loo_probe_hits"] == peak_loo]
    peak_pct = max(r["null_pct"] for r in late)
    peak_pct_L = [r["layer"] for r in late if r["null_pct"] == peak_pct]

    out = {
        "model": spec["model"],
        "prereg": "preregistration_smallcode.md e0aed98",
        "frozen": {"n_layers": L, "probe_layer": lm.probe_layer, "shift": shift,
                   "R_planes": R_PLANES, "null_base_seed": NULL_BASE_SEED,
                   "day_plane_share": round(rep["share"], 4),
                   "thresholds": {"loo_acc_faint": "4/7",
                                  "amp_pct_faint": 99.0,
                                  "contiguous_late_layers": 2},
                   "late_layers": [r["layer"] for r in late]},
        "want": want,
        "rows": rows,
        "measurement_A_answer_trained_loo": {
            "peak_late_loo_hits": peak_loo, "peak_late_loo_layers": peak_loo_L,
            "branch_a_faint_fired": branch_a,
            "late_layers_ge4": [r["layer"] for r in a_hit]},
        "measurement_B_amplitude_null": {
            "peak_late_pct": peak_pct, "peak_late_pct_layers": peak_pct_L,
            "best_contiguous_late_run_ge99": best_run,
            "contiguous_run_layers": b_layers,
            "branch_b_faint_fired": branch_b},
        "reference_signature_7B": {
            "source": spec["ref"], "measure": "day-token transfer probe (read-only)",
            "peak_probe": ref_peak, "peak_layers": ref_peakL,
            "note": "7B day-token TRANSFER probe peaks here; not recomputed on 7B"},
        "branch": branch,
    }
    out_dir = ROOT / "artifacts" / spec["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("model", "frozen", "measurement_A_answer_trained_loo",
                       "measurement_B_amplitude_null", "reference_signature_7B",
                       "branch")}, indent=2), flush=True)
    print("LOO by late layer:",
          [(r["layer"], r["loo_probe_hits"]) for r in late], flush=True)
    print("null_pct by late layer:",
          [(r["layer"], r["null_pct"]) for r in late], flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(SPECS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(SPECS[args.model], args.device)


if __name__ == "__main__":
    main()
