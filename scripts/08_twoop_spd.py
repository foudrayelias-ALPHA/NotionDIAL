"""Two-op experiment, phase B: SPD + the differential-use question.

Does op-conditional (differential) use pin SPD's gauge? Measures per component:
  - op-selectivity of causal importance: sel = (ci_add - ci_sub) / (ci_add + ci_sub)
  - frequency mixing (dominant share) per embedding block
Hypothesis (from Step-1 report): op-differentiated modules (W_in/W_out) get pinned
along the op axis; the shared embedding planes stay gauge-mixed. If so, mechanisms
factor into manifold-BUILDING parts (dense use, gauge-free) and manifold-CONSUMING
parts (sparse use, pinned) — the regime map for what SPD will pin in a real LM.

Usage: python 08_twoop_spd.py [--steps 15000] [--c 96] [--device mps]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.edits import comp_matrix
from clocklib.model import AdderConfig, TwoOpAdder, twoop_all, twoop_encode
from clocklib.spdio import MODULES, decompose

ART, FIG = ROOT / "artifacts", ROOT / "figures"
P = 113


class TwoOpDataset(IterableDataset):
    def __init__(self, p: int, batch_size: int, seed: int):
        self.p, self.batch_size, self.seed = p, batch_size, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            a = torch.randint(0, self.p, (self.batch_size,), generator=g)
            b = torch.randint(0, self.p, (self.batch_size,), generator=g)
            op = torch.randint(0, 2, (self.batch_size,), generator=g)
            yield twoop_encode(a, b, op, self.p)


def signature_share(art: dict, module: str, c: int, E_blocks: dict) -> tuple[int, float]:
    """Dominant frequency + share of the component's token profile (best block)."""
    a = art[module]
    best = (0, 0.0)
    match module:
        case "W_e":
            profiles = [a["V"][:P, c].numpy(), a["V"][P : 2 * P, c].numpy()]
        case "W_in":
            profiles = [E_blocks[b] @ a["V"][:, c].numpy() for b in ("a", "b")]
        case "W_out":
            profiles = [a["U"][c, :].numpy()]
    for s in profiles:
        F = np.abs(np.fft.rfft(s)) ** 2
        F[0] = 0.0
        tot = F.sum()
        if tot <= 0:
            continue
        k, share = int(F.argmax()), float(F.max() / tot)
        if share > best[1]:
            best = (k, share)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--c", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-decompose", action="store_true")
    ap.add_argument("--imin", type=float, default=1e-5)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    blob = torch.load(ART / "twoop_seed0.pt", weights_only=False)
    model = TwoOpAdder(AdderConfig(**blob["cfg"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()

    a, b, op, labels = twoop_all(P)
    X = twoop_encode(a, b, op, P)
    out_dir = ART / f"spd_twoop_s0{args.suffix}"
    if args.skip_decompose:
        art = torch.load(out_dir / "decomposition.pt", weights_only=False)
    else:
        art = decompose(
            model, p=P, module_to_c={m: args.c for m in MODULES},
            steps=args.steps, batch_size=args.batch, seed=0, out_dir=out_dir,
            device=args.device, imin_coeff=args.imin,
            loader=DataLoader(TwoOpDataset(P, args.batch, 0), batch_size=None),
            ci_input_groups={"add": X[op == 0], "sub": X[op == 1]})
        model.cpu()

    comp_only = TwoOpAdder(model.cfg)
    with torch.no_grad():
        for m in MODULES:
            a_ = art[m]
            W = torch.einsum("ic,co->io", a_["V"], a_["U"])
            W = W if W.shape == a_["W_target"].shape else W.T
            getattr(comp_only, m).weight.copy_(W)
    comp_only.eval()
    with torch.no_grad():
        lt, lc = model(X), comp_only(X)
        agree = (lc.argmax(-1) == lt.argmax(-1)).float()
        report = {
            "acc_target_add": float((lt.argmax(-1) == labels)[op == 0].float().mean()),
            "acc_target_sub": float((lt.argmax(-1) == labels)[op == 1].float().mean()),
            "agree_add": float(agree[op == 0].mean()),
            "agree_sub": float(agree[op == 1].mean()),
            "logit_rel_err": float((lc - lt).norm() / lt.norm()),
            "delta_frac": {m: float(art[m]["delta"].norm() / art[m]["W_target"].norm())
                           for m in MODULES},
        }

    E_blocks = {b_: model.block(b_).detach().numpy() for b_ in ("a", "b")}
    rows = []
    for m in MODULES:
        ci_add, ci_sub = art["mean_ci_add"][m], art["mean_ci_sub"][m]
        for c in range(art[m]["V"].shape[1]):
            k, share = signature_share(art, m, c, E_blocks)
            ca, cs = float(ci_add[c]), float(ci_sub[c])
            sel = (ca - cs) / (ca + cs) if (ca + cs) > 1e-9 else 0.0
            rows.append({"module": m, "c": c, "freq": k, "share": share,
                         "ci_add": ca, "ci_sub": cs, "op_sel": sel,
                         "norm": float(comp_matrix(art, m, c).norm())})
    (out_dir / "twoop_components.json").write_text(json.dumps(rows))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, m in enumerate(MODULES):
        sub = [r for r in rows if r["module"] == m and r["norm"] > 1e-3]
        sels = [r["op_sel"] for r in sub]
        shares = [r["share"] for r in sub]
        axes[0, j].hist(sels, bins=30, range=(-1, 1))
        axes[0, j].set_title(f"{m}: op-selectivity of CI")
        axes[1, j].scatter(np.abs(sels), shares, s=14)
        axes[1, j].set_xlabel("|op selectivity|"); axes[1, j].set_ylabel("dominant freq share")
        axes[1, j].set_title(f"{m}: pinning vs mixing")
    fig.savefig(FIG / f"phase8_twoop_selectivity{args.suffix}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    for m in MODULES:
        sub = [r for r in rows if r["module"] == m and r["norm"] > 1e-3]
        sels = np.array([abs(r["op_sel"]) for r in sub])
        shares = np.array([r["share"] for r in sub])
        report[m] = {
            "n_alive": len(sub),
            "mean_abs_op_sel": float(sels.mean()),
            "frac_op_selective_gt_0.5": float((sels > 0.5).mean()),
            "mean_share": float(shares.mean()),
            "corr_sel_share": float(np.corrcoef(sels, shares)[0, 1]) if len(sub) > 2 else None,
        }
    (out_dir / "twoop_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
