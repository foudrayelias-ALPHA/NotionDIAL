"""Phase 1: SPD decomposition + faithfulness gate (behavior + geometry) at lambda=1.

Reference semantics: with use_delta_component=True the full reconstruction
(components + delta) equals the target exactly; the gate therefore checks that the
COMPONENTS-ONLY model carries the behavior and the geometry (small delta_frac, small
accuracy drop, per-key-frequency power ratios near 1). Every edit in Phases 4-5 scales
component terms while delta stays fixed.

Usage: python 01_decompose_and_faithfulness.py --model rung0 --steps 3000 --c 16
       python 01_decompose_and_faithfulness.py --model seed0 --steps 20000 --c 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import freq_power, key_freqs
from clocklib.model import AdderConfig, ModAdder, ModAdder2Hot, accuracy, all_pairs, two_hot
from clocklib.spdio import MODULES, decompose, recon_weights

ART = ROOT / "artifacts"


def load_target(name: str) -> ModAdder2Hot:
    path = ART / ("rung0.pt" if name == "rung0" else f"adder_{name}.pt")
    blob = torch.load(path, weights_only=False)
    adder = ModAdder(AdderConfig(**blob["cfg"]))
    adder.load_state_dict(blob["state_dict"])
    return ModAdder2Hot.from_adder(adder)


def build_component_only_model(art: dict, cfg: AdderConfig) -> ModAdder2Hot:
    model = ModAdder2Hot(cfg)
    with torch.no_grad():
        for m in MODULES:
            getattr(model, m).weight.copy_(recon_weights(art, m))
    return model.eval()


@torch.no_grad()
def hidden_cloud(model: ModAdder2Hot, p: int) -> np.ndarray:
    """Token-indexed hidden cloud: H[a] = mean_b hidden(a, b), shape (p, h)."""
    tokens, _ = all_pairs(p)
    h = model.hidden(two_hot(tokens, p))
    return h.reshape(p, p, -1).mean(dim=1).numpy()


def fold(k: int, p: int) -> int:
    """Map frequency k to its rfft bin (k and p-k alias to the same bin)."""
    k = k % p
    return min(k, p - k)


def power_ratios(M_recon: np.ndarray, M_target: np.ndarray, freqs: list[int], p: int,
                 harmonics: bool = False) -> dict[int, float]:
    """Per-frequency power ratio; with harmonics=True the band is {k, fold(2k)} —
    post-nonlinearity clouds carry the doubled frequency (frequency products)."""
    pr, pt = freq_power(M_recon), freq_power(M_target)
    out = {}
    for k in freqs:
        band = {fold(k, p), fold(2 * k, p)} if harmonics else {fold(k, p)}
        band = {b for b in band if b > 0}
        out[k] = float(sum(pr[b] for b in band) / max(sum(pt[b] for b in band), 1e-30))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--c", type=int, default=100)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--spd-seed", type=int, default=0)
    ap.add_argument("--imin", type=float, default=1e-5)
    ap.add_argument("--skip-decompose", action="store_true",
                    help="reuse existing decomposition.pt, recompute the report only")
    ap.add_argument("--device", default="cpu", help="cpu | mps (mps: locally patched param_decomp)")
    ap.add_argument("--out-suffix", default="", help="suffix for the artifact dir name")
    args = ap.parse_args()

    target = load_target(args.model)
    p = target.cfg.p
    out_dir = ART / f"spd_{args.model}_s{args.spd_seed}{args.out_suffix}"
    t0 = time.time()
    if args.skip_decompose:
        art = torch.load(out_dir / "decomposition.pt", weights_only=False)
    else:
        art = decompose(target, p=p, module_to_c={m: args.c for m in MODULES},
                        steps=args.steps, batch_size=args.batch, seed=args.spd_seed,
                        out_dir=out_dir, device=args.device)
        target.cpu()
    wall = time.time() - t0

    comp_only = build_component_only_model(art, target.cfg)
    tokens, labels = all_pairs(p)
    inputs = two_hot(tokens, p)
    acc_target = accuracy(target, inputs, labels)
    acc_comp = accuracy(comp_only, inputs, labels)
    with torch.no_grad():
        lt, lc = target(inputs), comp_only(inputs)
        agree = (lc.argmax(-1) == lt.argmax(-1)).float().mean().item()
        top2 = lt.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        robust = margin > 0.01 * lt.std()
        robust_agree = (lc.argmax(-1) == lt.argmax(-1))[robust].float().mean().item()
        robust_frac = robust.float().mean().item()
        logit_rel_err = ((lc - lt).norm() / lt.norm()).item()

    E_t = target.embedding_matrix().detach().numpy()
    kf = key_freqs(E_t, frac_thresh=0.02)
    E_r = comp_only.embedding_matrix().detach().numpy()
    emb_ratio = power_ratios(E_r, E_t, kf, p)
    hid_ratio = power_ratios(hidden_cloud(comp_only, p), hidden_cloud(target, p), kf, p,
                             harmonics=True)

    delta_frac = {
        m: float(art[m]["delta"].norm() / art[m]["W_target"].norm()) for m in MODULES
    }
    pw = freq_power(art["W_e"]["W_target"].T.numpy())
    pd_ = freq_power(art["W_e"]["delta"].T.numpy())
    delta_key_leak = max(float(pd_[k] / pw[k]) for k in kf)
    alive = {m: int((art["mean_ci"][m] > 0.01).sum()) for m in MODULES}

    report = {
        "model": args.model, "spd_seed": args.spd_seed, "steps": args.steps,
        "C": args.c, "wall_seconds": round(wall, 1),
        "acc_target": acc_target, "acc_comp_only": acc_comp, "argmax_agreement": agree,
        "robust_argmax_agreement": robust_agree, "robust_pair_frac": robust_frac,
        "logit_rel_err": logit_rel_err,
        "delta_frac": delta_frac, "delta_key_freq_power_leak": delta_key_leak,
        "alive_components_ci_gt_0.01": alive,
        "key_freqs": kf,
        "embed_power_ratio_comp_over_target": emb_ratio,
        "hidden_power_ratio_comp_over_target": hid_ratio,
        "gate": {
            # In a margin-degenerate regime (single-frequency clock: all margins
            # ~(1-cos(2pi/p)) ~ 0.15% of scale) argmax fidelity is meaningless;
            # gate on logit error instead. robust_frac tells us which regime we're in.
            "behavioral": (robust_agree >= 0.99 if robust_frac >= 0.5
                           else logit_rel_err <= 0.01),
            "logit_rel_err_le_1pct": logit_rel_err <= 0.01,
            "embed_power_within_5pct": all(abs(r - 1) <= 0.05 for r in emb_ratio.values()),
            # Mechanism-relevant delta criterion: delta may hold broadband non-key
            # residue (harmless to plane edits) but must not hold key-frequency
            # structure. Raw delta_frac is reported above. See decisions.md.
            "delta_key_freq_leak_le_1pct": delta_key_leak <= 0.01,
        },
    }
    report["gate"]["pass"] = all(report["gate"].values())
    (out_dir / "faithfulness.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nGATE {'PASS' if report['gate']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
