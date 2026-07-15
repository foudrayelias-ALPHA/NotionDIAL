"""Phase 0: analytic rung-0 model + grokked ReLU adders + baseline circles + gate.

Outputs: artifacts/rung0.pt, artifacts/adder_seed{S}.pt, artifacts/phase0_gate.json,
figures/phase0_*.png
"""

import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import fourier_plane_basis, freq_power, key_freqs, plane_coords
from clocklib.model import AdderConfig, ModAdder, accuracy, all_pairs, build_analytic_rung0

ART, FIG = ROOT / "artifacts", ROOT / "figures"
P, K0, SEEDS = 113, 7, [0, 1, 2]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def verify_analytic() -> dict:
    model = build_analytic_rung0(P, K0)
    tokens, labels = all_pairs(P)
    acc = accuracy(model, tokens, labels)
    with torch.no_grad():
        logits = model(tokens).double().numpy()
    w = 2 * math.pi * K0 / P
    a = tokens[:, 0].numpy().astype(float)
    b = tokens[:, 1].numpy().astype(float)
    c = np.arange(P).astype(float)
    pred = (1 + np.cos(w * (a - b)))[:, None] * np.cos(w * (a[:, None] + b[:, None] - c[None, :]))
    resid = float(np.abs(logits - pred).max())
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg.__dict__, "k0": K0}, ART / "rung0.pt")
    return {"acc": acc, "logit_law_max_resid": resid}


def train_adder(seed: int, train_frac: float = 0.3, lr: float = 1e-3, wd: float = 1.0,
                min_epochs: int = 100_000, max_epochs: int = 100_000) -> tuple[ModAdder, list]:
    """Train past generalization: weight decay's cleanup phase prunes the embedding
    spectrum to sparse Fourier modes only well after test accuracy saturates."""
    torch.manual_seed(seed)
    cfg = AdderConfig()
    model = ModAdder(cfg).to(DEVICE)
    tokens, labels = all_pairs(P, DEVICE)
    perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(seed))
    n_train = int(train_frac * P * P)
    tr, te = perm[:n_train].to(DEVICE), perm[n_train:].to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    hist, streak = [], 0
    for epoch in range(max_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(tokens[tr]), labels[tr])
        loss.backward()
        opt.step()
        if epoch % 200 == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                tr_acc = accuracy(model, tokens[tr], labels[tr])
                te_acc = accuracy(model, tokens[te], labels[te])
            hist.append((epoch, loss.item(), tr_acc, te_acc))
            streak = streak + 1 if te_acc >= 0.999 else 0
            if streak >= 5 and epoch >= min_epochs:
                break
        if epoch % 10_000 == 0:
            E = model.embed.weight.detach().cpu().numpy()
            power = freq_power(E)
            kf = key_freqs(E, frac_thresh=0.02)
            print(f"  [seed {seed} epoch {epoch}] n_key={len(kf)} "
                  f"key_power={power[kf].sum() / power.sum():.1%}", flush=True)
    model = model.cpu().eval()
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "seed": seed,
                "train_frac": train_frac, "lr": lr, "wd": wd, "hist": hist,
                "train_idx": tr.cpu(), "test_idx": te.cpu()}, ART / f"adder_seed{seed}.pt")
    return model, hist


def circle_figure(E: np.ndarray, freqs: list[int], title: str, path: Path) -> None:
    n = len(freqs)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(4 * max(n, 1), 4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, freqs):
        Q = fourier_plane_basis(E, k)
        xy = plane_coords(E, Q)
        ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="hsv", s=12)
        ax.set_title(f"freq {k}")
        ax.set_aspect("equal")
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ART.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    t0 = time.time()

    rung0 = verify_analytic()
    print(f"[rung0] acc={rung0['acc']:.4f} logit-law max resid={rung0['logit_law_max_resid']:.2e}")
    E0 = torch.load(ART / "rung0.pt", weights_only=False)["state_dict"]["embed.weight"].numpy()
    circle_figure(E0, [K0], "rung-0 analytic embedding circle", FIG / "phase0_rung0_circle.png")

    gate = {"rung0": rung0, "seeds": {}}
    for seed in SEEDS:
        model, hist = train_adder(seed)
        E = model.embed.weight.detach().numpy()
        power = freq_power(E)
        kf = key_freqs(E, frac_thresh=0.02)
        top8 = float(np.sort(power)[::-1][:8].sum() / power.sum())
        epochs, losses, tr_accs, te_accs = zip(*hist)
        gate["seeds"][seed] = {
            "final_train_acc": tr_accs[-1], "final_test_acc": te_accs[-1],
            "epochs": epochs[-1], "key_freqs": kf, "top8_power_frac": top8,
            "key_freq_power_frac": float(power[kf].sum() / power.sum()) if kf else 0.0,
        }
        print(f"[seed {seed}] test_acc={te_accs[-1]:.4f} @ epoch {epochs[-1]} "
              f"key_freqs={kf} ({gate['seeds'][seed]['key_freq_power_frac']:.1%} of power, "
              f"top8={top8:.1%})")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(epochs, tr_accs, label="train")
        ax1.plot(epochs, te_accs, label="test")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy"); ax1.legend()
        ax1.set_title(f"seed {seed} grokking")
        ax2.bar(np.arange(len(power)), power / power.sum())
        ax2.set_xlabel("frequency k"); ax2.set_ylabel("power frac"); ax2.set_title("DFT(E)")
        fig.savefig(FIG / f"phase0_seed{seed}_train_dft.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        circle_figure(E, kf[:6], f"seed {seed} embedding circles", FIG / f"phase0_seed{seed}_circles.png")

    gate["pass"] = (
        rung0["acc"] == 1.0 and rung0["logit_law_max_resid"] < 1e-3
        and all(s["final_test_acc"] >= 0.99 and 1 <= len(s["key_freqs"]) <= 16
                and s["key_freq_power_frac"] >= 0.85
                for s in gate["seeds"].values())
    )
    gate["wall_seconds"] = round(time.time() - t0, 1)
    (ART / "phase0_gate.json").write_text(json.dumps(gate, indent=2))
    print(f"\nGATE {'PASS' if gate['pass'] else 'FAIL'} ({gate['wall_seconds']}s)")


if __name__ == "__main__":
    main()
