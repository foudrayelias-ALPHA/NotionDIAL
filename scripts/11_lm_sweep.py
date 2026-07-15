"""Step-2 phase B: lambda-sweep of GPT-2's day ring via the wte decomposition.

Conditions and predictions: preregistration_step2.md (frozen before this ran).
Alignment: lstsq of full-vocab targets (Fourier modes on concept rows, zero off-rows)
against the decomposition's read vectors — minimizes vocab leakage within the span;
reconstruction preserved exactly (V' U' = V U).

Usage: python 11_lm_sweep.py [--art artifacts/spd_gpt2_wte_s0] [--device mps]
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clocklib.fourier import fourier_plane_basis, freq_power, plane_coords
from clocklib.geometry import winding_number
from clocklib.sweep import DEFAULT_LAMBDAS

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPLATES = [
    "The meeting is scheduled for {}.", "She will arrive on {}.",
    "Everything closed last {} evening.", "I always go swimming on {}.",
    "The deadline is next {}.", "It happened one {} morning.",
    "We usually rest on {}.", "The store reopens on {}.",
]
SUCC = "If today is {}, then tomorrow is"
L_PROBE = 8


def row_fourier_targets(n: int, freqs: list[int]) -> tuple[np.ndarray, list[str]]:
    cols, labels = [], []
    for k in freqs:
        t = 2 * np.pi * k * np.arange(n) / n
        for fn, nm in ((np.cos, "cos"), (np.sin, "sin")):
            f = fn(t)
            cols.append(f / np.linalg.norm(f))
            labels.append(f"k{k}_{nm}")
    return np.stack(cols, axis=1), labels


def constrained_solve(V: np.ndarray, ids: list[int], f: np.ndarray) -> tuple[np.ndarray, float, float]:
    """min ||V_offrows g|| s.t. V[ids] g = f. Returns (g, on-rows residual, off-row leakage).

    (Plain lstsq against zero-padded full-vocab targets is degenerate: 50k zero
    equations dominate the 7 signal equations and the solution collapses to g ~ 0.)
    """
    A = V[ids]                                       # (n, C)
    BtB = V.T @ V - A.T @ A                          # (C, C) = off-rows gram
    M = BtB + 1e-9 * np.trace(BtB) / BtB.shape[0] * np.eye(BtB.shape[0])
    MinvAt = np.linalg.solve(M, A.T)                 # (C, n)
    S = A @ MinvAt                                   # (n, n)
    g = MinvAt @ np.linalg.solve(S, f)
    on_res = float(np.linalg.norm(A @ g - f))
    leak = float(np.sqrt(max(g @ (BtB @ g), 0.0)))
    return g, on_res, leak


def align_lm(art: dict) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """Returns V', U', clusters {name: [comp ids]}, report."""
    V = art["V"].numpy().astype(np.float64)          # (vocab, C)
    U = art["U"].numpy().astype(np.float64)          # (C, 768)
    vocab, C = V.shape
    Fd, lab_d = row_fourier_targets(len(art["day_ids"]), [1, 2, 3])
    Fm, lab_m = row_fourier_targets(len(art["month_ids"]), [1])
    jobs = [(art["day_ids"], Fd[:, j], f"day_{lab_d[j]}") for j in range(Fd.shape[1])] + \
           [(art["month_ids"], Fm[:, j], f"month_{lab_m[j]}") for j in range(Fm.shape[1])]

    G_cols, residuals = [], {}
    for ids, f, label in jobs:
        g, on_res, leak = constrained_solve(V, ids, f)
        residuals[label] = {"on_rows_residual": on_res, "offrow_leakage": leak}
        G_cols.append(g / np.linalg.norm(g))
    G_partial = np.stack(G_cols, axis=1)
    # Zero-plane-content completion (see clocklib/ringlib.py align): completion columns
    # in null(A^T), A columns = V[ids]^T f per target, so the aligned pairs carry ALL
    # concept-plane content and plane ablation is exact at full capacity.
    A = np.stack([V[ids].T @ f for ids, f, _ in jobs], axis=1)
    Ua, _, _ = np.linalg.svd(A, full_matrices=True)
    N = Ua[:, A.shape[1]:]                           # orthonormal basis of null(A^T)
    G = np.concatenate([G_partial, N], axis=1)
    cond = float(np.linalg.cond(G))
    assert cond < 1e8, cond
    V_new, U_new = V @ G, np.linalg.solve(G, U)
    recon_err = float(np.abs(V_new @ U_new - V @ U).max())
    assert recon_err < 1e-6, recon_err
    clusters = {"day_k1": [0, 1], "day_k2": [2, 3], "day_k3": [4, 5], "month_k1": [6, 7]}
    report = {"lstsq_residuals": residuals, "G_cond": cond, "recon_err": recon_err}
    return (torch.from_numpy(V_new).float(), torch.from_numpy(U_new).float(),
            clusters, report)


def untied_copy(base: nn.Module) -> nn.Module:
    m = copy.deepcopy(base)
    m.transformer.wte.weight = nn.Parameter(m.transformer.wte.weight.detach().clone())
    assert m.lm_head.weight is not m.transformer.wte.weight
    return m


class LMRef:
    """Everything frozen at lambda=1."""

    def __init__(self, base, tok, art, device):
        self.tok, self.device = tok, device
        self.day_ids, self.month_ids = art["day_ids"], art["month_ids"]
        self.model = untied_copy(base).to(device).eval()
        self.W1 = self.model.transformer.wte.weight.detach().cpu().clone()

        E = self.W1[self.day_ids].numpy()
        self.pw_day_ref = freq_power(E)
        self.Q_day = {k: fourier_plane_basis(E, k) for k in (1, 2, 3)}
        M = self.W1[self.month_ids].numpy()
        self.pw_month_ref = freq_power(M)

        self.succ_enc = [tok(SUCC.format(d), return_tensors="pt").input_ids.to(device)
                         for d in DAYS]
        self.tpl_enc, self.tpl_pos = [], []
        for t in TEMPLATES:
            for wi, d in enumerate(DAYS):
                enc = tok(t.format(d), return_tensors="pt").input_ids
                self.tpl_enc.append(enc.to(device))
                self.tpl_pos.append((wi, enc[0].tolist().index(self.day_ids[wi])))

        from datasets import load_dataset

        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n".join(t for t in wiki["text"] if t.strip())[:20000]
        ids = tok(text, return_tensors="pt").input_ids[0][: 8 * 64].reshape(8, 64)
        self.wiki = ids.to(device)

        m = self.measure_raw(self.W1)
        self.H8_ref = m.pop("_H8")
        self.wiki_logp_ref = m.pop("_wiki_logp")
        self.Q_h8 = fourier_plane_basis(self.H8_ref, 1)
        from sklearn.linear_model import Ridge

        t7 = 2 * np.pi * np.arange(7) / 7
        self.probe = Ridge(alpha=1e-3).fit(self.H8_ref, np.stack([np.cos(t7), np.sin(t7)], 1))
        self.ref_metrics = m

    @torch.no_grad()
    def measure_raw(self, W: torch.Tensor) -> dict:
        self.model.transformer.wte.weight.data.copy_(W.to(self.device))
        out: dict = {}
        E = W[self.day_ids].numpy()
        pw = freq_power(E)
        for k in (1, 2, 3):
            out[f"day_k{k}_power"] = float(pw[k] / max(self.pw_day_ref[k], 1e-12))
        Mpw = freq_power(W[self.month_ids].numpy())
        out["month_k1_power"] = float(Mpw[1] / max(self.pw_month_ref[1], 1e-12))
        xy = plane_coords(E, self.Q_day[1])
        c = xy - xy.mean(0)
        out["day_k1_radius"] = float(np.linalg.norm(c, axis=1).mean())
        out["day_k1_winding"] = winding_number(xy)

        margins, correct = [], 0
        for i, enc in enumerate(self.succ_enc):
            logits = self.model(input_ids=enc).logits[0, -1]
            dl = logits[self.day_ids]
            want = (i + 1) % 7
            m = float(dl[want] - dl[torch.arange(7) != want].max())
            margins.append(m)
            correct += int(dl.argmax()) == want
        out["succ_margin_mean"] = float(np.mean(margins))
        out["succ_margin_sun"] = margins[6]
        out["succ_acc"] = correct / 7

        H8 = np.zeros((7, 768))
        for enc, (wi, pos) in zip(self.tpl_enc, self.tpl_pos):
            hs = self.model(input_ids=enc, output_hidden_states=True).hidden_states
            H8[wi] += hs[L_PROBE][0, pos].float().cpu().numpy()
        H8 /= len(TEMPLATES)
        out["_H8"] = H8
        pw8 = freq_power(H8)
        out["h8_k1_power_raw"] = float(pw8[1])

        logits = self.model(input_ids=self.wiki).logits
        logp = F.log_softmax(logits.float(), dim=-1)
        out["_wiki_logp"] = logp.cpu()
        return out

    def measure(self, W: torch.Tensor) -> dict:
        m = self.measure_raw(W)
        H8 = m.pop("_H8")
        m["h8_k1_power"] = m.pop("h8_k1_power_raw") / max(self.ref_metrics["h8_k1_power_raw"], 1e-12)
        pred = self.probe.predict(H8)
        t7 = 2 * np.pi * np.arange(7) / 7
        d = np.arctan2(pred[:, 1], pred[:, 0]) - t7
        m["h8_probe_err"] = float(np.abs((d + np.pi) % (2 * np.pi) - np.pi).mean())
        logp = m.pop("_wiki_logp")
        m["wiki_kl"] = float(
            F.kl_div(logp.flatten(0, 1), self.wiki_logp_ref.flatten(0, 1),
                     reduction="batchmean", log_target=True))
        return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts/spd_gpt2_wte_s0")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="sweep_gpt2_day")
    args = ap.parse_args()
    art_dir = ROOT / args.art if not Path(args.art).is_absolute() else Path(args.art)
    art = torch.load(art_dir / "decomposition.pt", weights_only=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    base = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    Vp, Up, clusters, align_report = align_lm(art)
    out_dir = ROOT / "artifacts" / args.out
    out_dir.mkdir(exist_ok=True)
    (out_dir / "align_report.json").write_text(json.dumps(align_report, indent=2))
    print(json.dumps(align_report, indent=2))

    ref = LMRef(base, tok, art, args.device)
    W1 = ref.W1

    def plane_mat(names: list[str]) -> torch.Tensor:
        ids = [c for n in names for c in clusters[n]]
        return Vp[:, ids] @ Up[ids, :]

    day_ids = art["day_ids"]
    # C3 norm matching uses the DAY-ROW restriction of the k1 plane matrix: the
    # full-vocab Frobenius norm is dominated by off-day leakage (~40x) and would make
    # the random control a vastly larger perturbation than the targeted edit's
    # concept-relevant content.
    k1_frob = float(plane_mat(["day_k1"])[day_ids].norm())

    def oracle_mat() -> torch.Tensor:
        E = W1[day_ids].numpy()
        Fk = np.fft.rfft(E, axis=0)
        keep = np.zeros_like(Fk)
        keep[1] = Fk[1]
        P = np.zeros((W1.shape[0], W1.shape[1]))
        P[day_ids] = np.fft.irfft(keep, n=7, axis=0)
        return torch.from_numpy(P).float()

    def rand_mat(seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        u7 = torch.randn(7, generator=g)
        v = torch.randn(W1.shape[1], generator=g)
        M = torch.zeros_like(W1)
        M[day_ids] = torch.outer(u7, v)
        return M * (k1_frob / M.norm())

    conds = {
        "T_sym_k1": plane_mat(["day_k1"]),
        "T_sym_ring": plane_mat(["day_k1", "day_k2", "day_k3"]),
        "C1_cross_month": plane_mat(["month_k1"]),
        "C4_oracle": oracle_mat(),
        **{f"C3_random_r{s}": rand_mat(s) for s in range(3)},
    }
    for name, M in conds.items():
        rows = []
        for lam in DEFAULT_LAMBDAS:
            W = W1 + (lam - 1.0) * M
            rows.append({"lam": lam, **ref.measure(W)})
        (out_dir / f"{name}.json").write_text(json.dumps(rows))
        print(f"[lm-sweep] {name} done", flush=True)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
