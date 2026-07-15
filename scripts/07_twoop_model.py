"""Two-op experiment, phase A: train (a, b, op) -> a±b mod p to grokking.

Gate: per-op test accuracy >= 99%; key frequencies in BOTH embedding blocks with
>= 85% of non-DC power. Output: artifacts/twoop_seed{S}.pt + figures.
"""

import json
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

from clocklib.fourier import freq_power, key_freqs
from clocklib.model import AdderConfig, TwoOpAdder, twoop_all, twoop_encode

ART, FIG = ROOT / "artifacts", ROOT / "figures"
P = 113
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def train(seed: int, train_frac: float = 0.3, lr: float = 1e-3, wd: float = 1.0,
          min_epochs: int = 100_000, max_epochs: int = 100_000) -> None:
    torch.manual_seed(seed)
    cfg = AdderConfig(p=P)
    model = TwoOpAdder(cfg).to(DEVICE)
    a, b, op, labels = twoop_all(P, DEVICE)
    X = twoop_encode(a, b, op, P)
    n = X.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).to(DEVICE)
    n_train = int(train_frac * n)
    tr, te = perm[:n_train], perm[n_train:]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    hist, streak = [], 0
    t0 = time.time()
    for epoch in range(max_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(X[tr]), labels[tr])
        loss.backward()
        opt.step()
        if epoch % 200 == 0 or epoch == max_epochs - 1:
            with torch.no_grad():
                pred_te = model(X[te]).argmax(-1)
                te_acc = (pred_te == labels[te]).float().mean().item()
                acc_add = (pred_te == labels[te])[op[te] == 0].float().mean().item()
                acc_sub = (pred_te == labels[te])[op[te] == 1].float().mean().item()
            hist.append((epoch, loss.item(), te_acc, acc_add, acc_sub))
            streak = streak + 1 if te_acc >= 0.999 else 0
            if streak >= 5 and epoch >= min_epochs:
                break
        if epoch % 10_000 == 0:
            Ea = model.block("a").detach().cpu().numpy()
            pw = freq_power(Ea)
            kf = key_freqs(Ea, frac_thresh=0.02)
            print(f"[epoch {epoch}] te={hist[-1][2]:.4f} add={hist[-1][3]:.4f} "
                  f"sub={hist[-1][4]:.4f} n_key(Ea)={len(kf)} "
                  f"key_pw={pw[kf].sum() / pw.sum():.1%}", flush=True)

    model = model.cpu().eval()
    Ea, Eb = model.block("a").detach().numpy(), model.block("b").detach().numpy()
    gate = {"seed": seed, "epochs": hist[-1][0], "test_acc": hist[-1][2],
            "acc_add": hist[-1][3], "acc_sub": hist[-1][4],
            "wall_seconds": round(time.time() - t0, 1)}
    for name, E in (("a", Ea), ("b", Eb)):
        pw = freq_power(E)
        kf = key_freqs(E, frac_thresh=0.02)
        gate[f"key_freqs_{name}"] = kf
        gate[f"key_power_frac_{name}"] = float(pw[kf].sum() / pw.sum()) if kf else 0.0
    gate["pass"] = (gate["acc_add"] >= 0.99 and gate["acc_sub"] >= 0.99
                    and gate["key_power_frac_a"] >= 0.85 and gate["key_power_frac_b"] >= 0.85)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__, "seed": seed,
                "train_frac": train_frac, "lr": lr, "wd": wd, "hist": hist,
                "train_idx": tr.cpu(), "test_idx": te.cpu()}, ART / f"twoop_seed{seed}.pt")
    (ART / f"twoop_seed{seed}_gate.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps(gate, indent=2))
    print(f"GATE {'PASS' if gate['pass'] else 'FAIL'}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ep, ls, te_, ad, su = zip(*hist)
    axes[0].plot(ep, ad, label="add"); axes[0].plot(ep, su, label="sub")
    axes[0].set_title("per-op test accuracy"); axes[0].legend()
    for ax, (nm, E) in zip(axes[1:], (("E_a", Ea), ("E_b", Eb))):
        pw = freq_power(E)
        ax.bar(np.arange(len(pw)), pw / pw.sum())
        ax.set_title(f"DFT({nm})")
    fig.savefig(FIG / f"phase7_twoop_seed{seed}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    train(seed=0)
