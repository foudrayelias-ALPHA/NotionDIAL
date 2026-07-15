"""Phase 6: curve panels, selectivity indices, verdict vs preregistration, animation.

Usage: python 06_analysis_and_verdict.py --sweep artifacts/sweep_seed0_s0_K42 [--behavioral]
       (--behavioral includes accuracy-based criteria; omit for margin-degenerate rung-0)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"

STYLE = {
    "T_emb_plane": dict(color="tab:red", lw=2.5, zorder=5),
    "T_stack_plane": dict(color="tab:orange", lw=2),
    "T_single": dict(color="tab:pink", lw=1.5, ls="--"),
    "C1_null_plane": dict(color="tab:green", lw=1.5),
    "C3_random_r0": dict(color="gray", lw=1, ls=":"),
    "C3_random_r1": dict(color="gray", lw=1, ls=":"),
    "C3_random_r2": dict(color="gray", lw=1, ls=":"),
    "C4_oracle": dict(color="tab:blue", lw=2, ls="-."),
    "C5_gauge_g0": dict(color="tab:purple", lw=1, ls="--"),
    "C5_gauge_g1": dict(color="tab:purple", lw=1, ls="--"),
    "C5_gauge_g2": dict(color="tab:purple", lw=1, ls="--"),
    "C6_svd": dict(color="tab:cyan", lw=2, ls="-."),
}


def load_sweep(d: Path) -> tuple[dict, dict]:
    meta = json.loads((d / "sweep_meta.json").read_text())
    conds = {}
    for f in sorted(d.glob("*.json")):
        rows = json.loads(f.read_text())
        if isinstance(rows, list) and rows and "lam" in rows[0]:
            conds[f.stem] = rows
    return meta, conds


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in rows])


def lams(rows: list[dict]) -> np.ndarray:
    return np.array([r["lam"] for r in rows])


def classify_curve(lam: np.ndarray, y: np.ndarray) -> str:
    """Plan §6 taxonomy over λ ∈ [0, 1], curve read 1 -> 0."""
    m = (lam >= 0) & (lam <= 1)
    x, v = lam[m], y[m]
    order = np.argsort(x)[::-1]  # λ: 1 -> 0
    v = v[order]
    rng = v.max() - v.min()
    if rng < 0.05 * max(abs(v).max(), 1e-9):
        return "flat"
    dv = np.diff(v)
    sign_changes = int((np.abs(np.diff(np.sign(dv[np.abs(dv) > 0.02 * rng]))) > 0).sum())
    if sign_changes >= 1:
        return "non-monotone"
    d2 = np.abs(np.diff(v, 2))
    if d2.max() > 0.35 * rng:
        return "threshold"
    return "graceful-monotone"


def h1_death_lambda(lam: np.ndarray, h1: np.ndarray, floor: np.ndarray) -> float:
    m = (lam >= 0) & (lam <= 1)
    dead = lam[m][(h1 < floor)[m]]
    return float(dead.max()) if len(dead) else float("nan")


def panels(meta: dict, conds: dict, tag: str) -> None:
    K, kfs = meta["K"], meta["key_freqs"]
    others = [j for j in kfs if j != K]

    def overlay(ax, key, title, ref_curve=None):
        for name, rows in conds.items():
            st = STYLE.get(name, dict(color="k", lw=1))
            ax.plot(lams(rows), col(rows, key), label=name, **st)
        if ref_curve is not None:
            lam = lams(next(iter(conds.values())))
            ax.plot(lam, ref_curve(lam), "k:", lw=1, label="λ²")
        ax.set_title(title)
        ax.set_xlabel("λ")
        ax.axvline(1.0, color="k", alpha=0.15)

    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    overlay(axes[0, 0], f"emb_power_k{K}", f"targeted emb power (K={K})",
            ref_curve=lambda l: l**2)
    if others:
        for name, rows in conds.items():
            st = STYLE.get(name, dict(color="k", lw=1))
            Y = np.nanmean([col(rows, f"emb_power_k{j}") for j in others], axis=0)
            axes[0, 1].plot(lams(rows), Y, **st)
        axes[0, 1].set_title(f"untargeted emb power (mean over {len(others)} freqs)")
        axes[0, 1].set_xlabel("λ"); axes[0, 1].axvline(1.0, color="k", alpha=0.15)
    overlay(axes[0, 2], f"emb_radius_mean_k{K}", "targeted plane radius")
    overlay(axes[1, 0], f"emb_axis_ratio_k{K}", "targeted plane axis ratio")
    overlay(axes[1, 1], f"emb_winding_k{K}", "targeted plane winding")
    overlay(axes[1, 2], f"emb_h1_k{K}", "targeted plane H1 max lifetime")
    fl = col(conds["T_emb_plane"], f"emb_h1_floor_k{K}")
    axes[1, 2].axhline(fl[0], color="r", alpha=0.4, ls=":", label="noise floor")
    overlay(axes[2, 0], f"probe_err_k{K}", "frozen probe angular error")
    overlay(axes[2, 1], "pr_hid", "participation ratio (hidden)")
    overlay(axes[2, 2], "acc", "accuracy (all p² pairs)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{tag}: metric-vs-λ, all conditions (frozen λ=1 basis)")
    fig.savefig(FIG / f"phase6_{tag}_panels.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    overlay(ax[0], "monodromy_winding", "sheaf monodromy winding (hidden)")
    overlay(ax[1], "twonn_hid", "TwoNN intrinsic dim (hidden)")
    fig2.savefig(FIG / f"phase6_{tag}_sheaf_id.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)


def animation(meta: dict, sweep_dir: Path, tag: str) -> None:
    import matplotlib.animation as anim

    K = meta["K"]
    npz = np.load(sweep_dir / "T_emb_plane_clouds.npz")
    lam_keys = sorted({float(k.split("_", 1)[1]) for k in npz.files if k.startswith("E_")})
    ref = json.loads((sweep_dir / "T_emb_plane.json").read_text())
    E1 = npz[f"E_{1.0}"].astype(np.float64)
    from clocklib.fourier import fourier_plane_basis, plane_coords
    from clocklib.sweep import fold

    p = E1.shape[0]
    Q = fourier_plane_basis(E1, fold(K, p))
    fig, ax = plt.subplots(figsize=(5, 5))
    frames = [l for l in lam_keys if 0 <= l <= 1][::-1]
    lim = np.abs(plane_coords(E1, Q)).max() * 1.2

    def draw(i):
        ax.clear()
        lam = frames[i]
        xy = plane_coords(npz[f"E_{lam}"].astype(np.float64), Q)
        ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(p), cmap="hsv", s=14)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"freq-{K} plane (frozen basis), λ={lam:.2f}")

    a = anim.FuncAnimation(fig, draw, frames=len(frames), interval=180)
    a.save(FIG / f"phase6_{tag}_collapse.gif", writer="pillow", dpi=90)
    plt.close(fig)


def verdict(meta: dict, conds: dict, behavioral: bool) -> dict:
    K, kfs = meta["K"], meta["key_freqs"]
    others = [j for j in kfs if j != K]
    T = conds["T_emb_plane"]
    lam = lams(T)
    m01 = (lam >= 0) & (lam <= 1)

    pK = col(T, f"emb_power_k{K}")
    a_rms = float(np.sqrt(np.mean((pK[m01] - lam[m01] ** 2) ** 2)))
    crit_a = bool(a_rms <= 0.10)

    if others:
        dev = np.max([np.abs(col(T, f"emb_power_k{j}") - 1)[m01].max() for j in others])
        crit_b = bool(dev <= 0.10)
    else:
        dev, crit_b = float("nan"), None

    def S(rows):
        i0 = int(np.argmin(np.abs(lams(rows))))
        s_t = 1 - col(rows, f"emb_power_k{K}")[i0]
        if not others:
            return float(s_t)
        s_o = np.mean([abs(1 - col(rows, f"emb_power_k{j}")[i0]) for j in others])
        return float(s_t - s_o)

    S_T = S(T)
    S_C3 = float(np.mean([S(conds[c]) for c in conds if c.startswith("C3")]))
    crit_c = bool((S_T - S_C3) >= 0.4)

    C4 = conds.get("C4_oracle")
    if C4:
        d_rms = float(np.sqrt(np.mean((pK[m01] - col(C4, f"emb_power_k{K}")[m01]) ** 2)))
        lstar_T = h1_death_lambda(lam, col(T, f"emb_h1_k{K}"), col(T, f"emb_h1_floor_k{K}"))
        lstar_C4 = h1_death_lambda(lams(C4), col(C4, f"emb_h1_k{K}"),
                                   col(C4, f"emb_h1_floor_k{K}"))
        crit_d = bool(d_rms <= 0.15 and (np.isnan(lstar_T) == np.isnan(lstar_C4)
                                         or abs(lstar_T - lstar_C4) <= 0.1))
    else:
        d_rms = lstar_T = lstar_C4 = float("nan")
        crit_d = None

    gauge = [c for c in conds if c.startswith("C5_gauge")]
    if gauge and "T_single" in conds:
        effects = []
        for c in gauge + ["T_single"]:
            rows = conds[c]
            i0 = int(np.argmin(np.abs(lams(rows))))
            effects.append(abs(1 - col(rows, f"emb_power_k{K}")[i0]))
        gauge_rel_var = float(np.std(effects) / max(np.mean(effects), 1e-9))
    else:
        gauge_rel_var = float("nan")

    taxonomy = {}
    for key in [f"emb_power_k{K}", f"emb_radius_mean_k{K}", f"emb_h1_k{K}",
                f"probe_err_k{K}", "pr_hid", "acc"]:
        taxonomy[key] = classify_curve(lam, col(T, key))

    checks = {"a_power_tracks_lam2": crit_a, "b_specificity": crit_b,
              "c_beats_random": crit_c, "d_matches_oracle": crit_d}
    core = [v for v in checks.values() if v is not None]
    return {
        "criteria": checks, "all_pass": all(core),
        "details": {"a_rms": a_rms, "b_max_untargeted_dev": dev,
                    "S_T": S_T, "S_C3": S_C3, "d_rms_vs_oracle": d_rms,
                    "h1_death_lambda_T": lstar_T, "h1_death_lambda_C4": lstar_C4,
                    "gauge_rel_variation": gauge_rel_var},
        "gauge_artifact_finding": bool(gauge_rel_var >= 0.25) if not np.isnan(gauge_rel_var) else None,
        "curve_taxonomy_T_emb_plane": taxonomy,
        "behavioral_included": behavioral,
    }


def error_set_jaccard(sweep_dir: Path, cond_a: str, cond_b: str, p: int) -> dict[str, float]:
    """EXPLORATORY (not preregistered): per-lambda Jaccard overlap of the two
    conditions' misclassification sets. High overlap = the SPD edit breaks the same
    specific (a,b) pairs the oracle breaks, not just the same number of them."""
    A = np.load(sweep_dir / f"{cond_a}_clouds.npz")
    B = np.load(sweep_dir / f"{cond_b}_clouds.npz")
    a_idx = np.arange(p)
    labels = ((a_idx[:, None] + a_idx[None, :]) % p).reshape(-1)
    out = {}
    for key in A.files:
        if not key.startswith("preds_"):
            continue
        lam = key.split("_", 1)[1]
        if f"preds_{lam}" not in B.files:
            continue
        ea = set(np.nonzero(A[key] != labels)[0].tolist())
        eb = set(np.nonzero(B[key] != labels)[0].tolist())
        union = ea | eb
        out[lam] = float(len(ea & eb) / len(union)) if union else 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--behavioral", action="store_true")
    ap.add_argument("--no-anim", action="store_true")
    args = ap.parse_args()
    d = ROOT / args.sweep if not Path(args.sweep).is_absolute() else Path(args.sweep)
    tag = d.name
    meta, conds = load_sweep(d)
    panels(meta, conds, tag)
    if not args.no_anim:
        animation(meta, d, tag)
    v = verdict(meta, conds, args.behavioral)
    if "C4_oracle" in conds and (d / "T_emb_plane_clouds.npz").exists():
        p = np.load(d / "T_emb_plane_clouds.npz")["E_1.0"].shape[0]
        v["exploratory_error_set_jaccard_T_vs_oracle"] = error_set_jaccard(
            d, "T_emb_plane", "C4_oracle", int(p))
    (d / "verdict.json").write_text(json.dumps(v, indent=2))
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()
