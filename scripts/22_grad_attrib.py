"""Gradient vs additive attribution (preregistration_gradattrib.md, 7aa9751).

g_c = dG/ds_c with s_c a scalar on module c's output (evaluated at s=1), G the
frozen-plane power of the day-token cloud at the probe layer. Two-pass exact
decomposition: dG/dH computed in closed form after a no-grad pass, then one
backward per template item (graphs never coexist).

Usage: python 22_grad_attrib.py --model gpt2 --device cpu
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
from clocklib.unsup import array_pca

w18 = import_module("18_writers_any")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    lm = LM(args.model, device=args.device)
    wr = w18.Writers(lm)  # working copy on device; n_below = probe_layer
    tpl = w18.positions(lm, DAYS)
    n_tpl = len(DAYS.templates)

    # pass 1 (no grad): frozen basis + closed-form dG/dH
    with torch.no_grad():
        H = np.zeros((7, lm.hidden_size))
        for enc, wi, pos in tpl:
            hs = wr.model(input_ids=enc.to(lm.device),
                          output_hidden_states=True).hidden_states
            H[wi] += hs[lm.probe_layer][0, pos].float().cpu().numpy()
        H /= n_tpl
    B, _ = array_pca(H, 2)
    Hc = H - H.mean(0)
    denom = float(((Hc @ B) ** 2).sum())
    # G = ||C (H) B||^2 with C the row-centering projector: dG/dH = 2 C^T (C H B) B^T
    G_grad_H = 2.0 * (Hc @ B) @ B.T
    G_grad_H = G_grad_H - G_grad_H.mean(0)  # C^T application (C symmetric idempotent)
    V = torch.from_numpy(G_grad_H / n_tpl).float().to(lm.device)  # per-item weight

    # scalar multipliers on writer outputs
    names = list(wr.orig.keys())
    scalars = {nm: torch.ones((), device=lm.device, requires_grad=True)
               for nm in names}

    def hook(nm):
        def fn(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] * scalars[nm], *out[1:])
            return out * scalars[nm]
        return fn

    handles = []
    for l in range(wr.n_below):
        b = wr.arch.blocks[l]
        handles.append(wr.arch.attn_mod(b).register_forward_hook(hook(f"attn_{l}")))
        handles.append(wr.arch.mlp_mod(b).register_forward_hook(hook(f"mlp_{l}")))
    g = {nm: 0.0 for nm in names}
    try:
        for enc, wi, pos in tpl:
            hs = wr.model(input_ids=enc.to(lm.device),
                          output_hidden_states=True).hidden_states
            loss = (V[wi] * hs[lm.probe_layer][0, pos].float()).sum()
            loss.backward()
            for nm in names:
                if scalars[nm].grad is not None:
                    g[nm] += float(scalars[nm].grad)
                    scalars[nm].grad = None
    finally:
        for h in handles:
            h.remove()

    # additive alphas from the committed attribution artifact (writer modules only)
    slug = args.model.replace("/", "_")
    art = ROOT / "artifacts" / (f"writers_{slug}" if slug != "gpt2" else "writers_gpt2")
    at = json.loads((art / "attribution.json").read_text())
    alphas = {k: v for k, v in at["alphas"].items() if k != "emb"}
    grad_attr = {nm: g[nm] / (2.0 * denom) for nm in names}  # alpha-normalized units

    from scipy.stats import spearmanr

    common = [nm for nm in names if nm in alphas]
    rho = float(spearmanr([grad_attr[nm] for nm in common],
                          [alphas[nm] for nm in common])[0])
    a_tot = sum(abs(v) for v in alphas.values())
    g_tot = sum(abs(v) for v in grad_attr.values())
    top = max(alphas, key=lambda k: abs(alphas[k]))
    ratio_top = (abs(grad_attr[top]) / g_tot) / (abs(alphas[top]) / a_tot)
    res = {"model": args.model, "spearman_grad_vs_additive": rho,
           "top_writer": top,
           "grad_share_over_additive_share_top": ratio_top,
           "grad_attr": {k: round(v, 4) for k, v in
                         sorted(grad_attr.items(), key=lambda kv: -abs(kv[1]))},
           "additive": {k: round(v, 4) for k, v in alphas.items()}}
    out = ROOT / "artifacts" / f"gradattrib_{slug}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in
                      ("grad_attr", "additive")}, indent=2))
    print("top8 grad:", dict(list(res["grad_attr"].items())[:8]))


if __name__ == "__main__":
    main()
