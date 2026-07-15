"""Model-agnostic ring pipeline: survey -> decompose -> align -> sweep.

Works on any HF causal LM whose concept words are single tokens. Model-specific
bits are auto-resolved: the embedding module via get_input_embeddings() (and its
module path via identity search), weight untying via parameter replacement, the
downstream probe layer at 2/3 depth. All formulas are ports of the versions
validated on GPT-2 (scripts 09-11) and on the clock (Step 1).
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from clocklib.fourier import fourier_plane_basis, freq_power, plane_coords
from clocklib.geometry import winding_number


@dataclass(frozen=True)
class Concept:
    name: str
    words: list[str]
    freqs: list[int]
    templates: list[str]
    successor_prompt: str
    successor_shift: int = 1


SHARED_TEMPLATES = [
    "The meeting is scheduled for {}.", "She will arrive on {}.",
    "Everything closed last {} evening.", "I always go swimming on {}.",
    "The deadline is next {}.", "It happened one {} morning.",
    "We usually rest on {}.", "The store reopens on {}.",
]

DAYS = Concept(
    name="days",
    words=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    freqs=[1, 2, 3],
    templates=SHARED_TEMPLATES,
    successor_prompt="If today is {}, then tomorrow is",
)
MONTHS = Concept(
    name="months",
    words=["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
    freqs=[1],
    templates=SHARED_TEMPLATES,
    successor_prompt="The month after {} is",
)


class LM:
    """A loaded causal LM with the ring-pipeline's model-specific bits resolved."""

    def __init__(self, name: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.device = name, device
        self.tok = AutoTokenizer.from_pretrained(name)
        # The base model STAYS ON CPU. Only working copies move to `device` (and are
        # verified after the move): on MPS under multi-GB pressure we observed a
        # freshly deep-copied 1B model's embedding read back as ALL ZEROS through the
        # device round-trip while being provably nonzero moments earlier in the same
        # process. CPU-side copies + post-move readback asserts make this impossible
        # to miss. See decisions.md 2026-07-02.
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        emb = self.model.get_input_embeddings()
        assert isinstance(emb, nn.Embedding), type(emb)
        paths = [n for n, m in self.model.named_modules() if m is emb]
        assert len(paths) == 1, paths
        self.emb_path = paths[0]
        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.hidden_size = cfg.hidden_size
        self.probe_layer = round(2 * self.n_layers / 3)

    def token_ids(self, concept: Concept) -> list[int] | None:
        """Single leading-space token per word, or None if any word is multi-token."""
        ids = []
        for w in concept.words:
            t = self.tok.encode(" " + w, add_special_tokens=False)
            if len(t) != 1:
                return None
            ids.append(t[0])
        return ids

    def untied_copy(self) -> nn.Module:
        assert next(self.model.parameters()).device.type == "cpu"
        m = copy.deepcopy(self.model)
        e = m.get_input_embeddings()
        e.weight = nn.Parameter(e.weight.detach().clone())
        assert float(e.weight.detach().abs().sum()) > 0, "untied copy has zero embedding"
        out = m.get_output_embeddings()
        if out is not None:
            assert out.weight is not e.weight
        return m


def ring_scores(X: np.ndarray) -> dict:
    """Ring-ness of an ordered concept cloud (n, d). Port of the survey metric."""
    n = X.shape[0]
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    xy = Xc @ Vt[:2].T
    pw = freq_power(Xc)
    T, *_ = np.linalg.lstsq(Xc, np.roll(Xc, 1, axis=0), rcond=None)
    eig = np.linalg.eigvals(T)
    eig = eig[np.abs(np.abs(eig) - 1) < 0.35]
    angles = sorted({round(abs(float(np.angle(l))) * n / (2 * np.pi), 1)
                     for l in eig if abs(l.imag) > 1e-6})
    return {"top2_pca_share": float((S[:2] ** 2).sum() / (S**2).sum()),
            "k1_power_frac": float(pw[1] / pw.sum()) if pw.sum() > 0 else 0.0,
            "winding_top2": winding_number(xy), "shift_eigen_freqs": angles}


@torch.no_grad()
def survey(lm: LM, concept: Concept) -> dict:
    ids = lm.token_ids(concept)
    if ids is None:
        return {"single_token": False}
    out: dict = {"single_token": True, "token_ids": ids, "layers": {}}
    emb = lm.model.get_input_embeddings().weight[ids].float().cpu().numpy()
    out["layers"]["embedding"] = ring_scores(emb)
    acc = [np.zeros((len(ids), lm.hidden_size)) for _ in range(lm.n_layers + 1)]
    for tpl in concept.templates:
        for wi, w in enumerate(concept.words):
            enc = lm.tok(tpl.format(w), return_tensors="pt").to(lm.device)
            pos = enc.input_ids[0].tolist().index(ids[wi])
            hs = lm.model(**enc, output_hidden_states=True).hidden_states
            for li in range(lm.n_layers + 1):
                acc[li][wi] += hs[li][0, pos].float().cpu().numpy()
    for li in range(lm.n_layers + 1):
        out["layers"][f"resid_{li}"] = ring_scores(acc[li] / len(concept.templates))
    return out


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
    """min ||V_offrows g|| s.t. V[ids] g = f (validated formulation; plain lstsq
    against zero-padded targets is degenerate)."""
    A = V[ids]
    BtB = V.T @ V - A.T @ A
    M = BtB + 1e-9 * np.trace(BtB) / BtB.shape[0] * np.eye(BtB.shape[0])
    MinvAt = np.linalg.solve(M, A.T)
    S = A @ MinvAt
    g = MinvAt @ np.linalg.solve(S, f)
    return g, float(np.linalg.norm(A @ g - f)), float(np.sqrt(max(g @ (BtB @ g), 0.0)))


def align(V: np.ndarray, U: np.ndarray, concept_jobs: list[tuple[str, list[int], list[int]]]
          ) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], dict]:
    """concept_jobs: (concept_name, row_ids, freqs). Returns V', U', clusters, report."""
    C = V.shape[1]
    G_cols, residuals, labels_all = [], {}, []
    for cname, ids, freqs in concept_jobs:
        Fm, labels = row_fourier_targets(len(ids), freqs)
        for j, lab in enumerate(labels):
            g, on_res, leak = constrained_solve(V, ids, Fm[:, j])
            residuals[f"{cname}_{lab}"] = {"on_rows_residual": on_res, "offrow_leakage": leak}
            G_cols.append(g / np.linalg.norm(g))
            labels_all.append(f"{cname}_{lab.rsplit('_', 1)[0]}")
    G_partial = np.stack(G_cols, axis=1)
    # Completion columns must carry ZERO concept-plane content on the concept rows,
    # else the un-edited components retain ring power and plane ablation is partial
    # (observed: 24% residual at exact faithfulness). Constraint vectors a = V[ids]^T f
    # per target; completion lives in null(A^T).
    A_cols = []
    for cname, ids, freqs in concept_jobs:
        Fm, _ = row_fourier_targets(len(ids), freqs)
        for j in range(Fm.shape[1]):
            A_cols.append(V[ids].T @ Fm[:, j])
    A = np.stack(A_cols, axis=1)                     # (C, n_targets)
    Ua, _, _ = np.linalg.svd(A, full_matrices=True)
    N = Ua[:, A.shape[1]:]                           # orthonormal basis of null(A^T)
    G = np.concatenate([G_partial, N], axis=1)
    cond = float(np.linalg.cond(G))
    assert cond < 1e8, cond
    V_new, U_new = V @ G, np.linalg.solve(G, U)
    assert np.abs(V_new @ U_new - V @ U).max() < 1e-6
    clusters: dict[str, list[int]] = {}
    for i, lab in enumerate(labels_all):
        clusters.setdefault(lab, []).append(i)
    report = {"residuals": residuals, "G_cond": cond}
    return V_new, U_new, clusters, report


class RingRef:
    """Frozen lambda=1 reference for one primary concept + one cross concept."""

    def __init__(self, lm: LM, primary: Concept, cross: Concept, wiki_ids: Tensor):
        self.lm = lm
        self.p_ids = lm.token_ids(primary)
        self.x_ids = lm.token_ids(cross)
        assert self.p_ids and self.x_ids
        self.primary, self.cross = primary, cross
        m = lm.untied_copy()
        self.W1 = m.get_input_embeddings().weight.detach().clone()  # CPU snapshot first
        assert float(self.W1.norm()) > 0
        self.model = m.to(lm.device).eval()
        back = self.model.get_input_embeddings().weight.detach().cpu()
        assert torch.allclose(back, self.W1, atol=1e-5), "weights corrupted by device move"
        E = self.W1[self.p_ids].numpy()
        self.pw_ref = freq_power(E)
        self.Q1 = fourier_plane_basis(E, 1)
        self.xw_ref = freq_power(self.W1[self.x_ids].numpy())
        n = len(self.p_ids)
        self.succ_enc = [lm.tok(primary.successor_prompt.format(w), return_tensors="pt"
                                ).input_ids.to(lm.device) for w in primary.words]
        self.tpl = []
        for t in primary.templates:
            for wi, w in enumerate(primary.words):
                enc = lm.tok(t.format(w), return_tensors="pt").input_ids
                self.tpl.append((enc.to(lm.device), wi,
                                 enc[0].tolist().index(self.p_ids[wi])))
        self.wiki = wiki_ids.to(lm.device)
        m = self._raw(self.W1)
        self.H_ref = m.pop("_H")
        self.wiki_logp_ref = m.pop("_wiki_logp")
        self.h_pw_ref = m.pop("_h_pw_raw")
        from sklearn.linear_model import Ridge

        t7 = 2 * np.pi * np.arange(n) / n
        self.probe = Ridge(alpha=1e-3).fit(self.H_ref, np.stack([np.cos(t7), np.sin(t7)], 1))
        self.ref = m

    @torch.no_grad()
    def _raw(self, W: Tensor) -> dict:
        lm, n = self.lm, len(self.p_ids)
        self.model.get_input_embeddings().weight.data.copy_(W.to(lm.device))
        out: dict = {}
        pw = freq_power(W[self.p_ids].numpy())
        for k in self.primary.freqs:
            out[f"p_k{k}_power"] = float(pw[k] / max(self.pw_ref[k], 1e-12))
        xw = freq_power(W[self.x_ids].numpy())
        out["cross_k1_power"] = float(xw[1] / max(self.xw_ref[1], 1e-12))
        xy = plane_coords(W[self.p_ids].numpy(), self.Q1)
        out["p_k1_radius"] = float(np.linalg.norm(xy - xy.mean(0), axis=1).mean())
        out["p_k1_winding"] = winding_number(xy)
        margins, correct = [], 0
        for i, enc in enumerate(self.succ_enc):
            logits = self.model(input_ids=enc).logits[0, -1]
            dl = logits[self.p_ids]
            want = (i + self.primary.successor_shift) % n
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        out["succ_margin_mean"] = float(np.mean(margins))
        out["succ_margin_wrap"] = margins[-1]
        out["succ_acc"] = correct / n
        H = np.zeros((n, lm.hidden_size))
        for enc, wi, pos in self.tpl:
            hs = self.model(input_ids=enc, output_hidden_states=True).hidden_states
            H[wi] += hs[lm.probe_layer][0, pos].float().cpu().numpy()
        H /= len(self.primary.templates)
        out["_H"] = H
        out["_h_pw_raw"] = float(freq_power(H)[1])
        logits = self.model(input_ids=self.wiki).logits
        out["_wiki_logp"] = F.log_softmax(logits.float(), dim=-1).cpu()
        return out

    def measure(self, W: Tensor) -> dict:
        m = self._raw(W)
        H = m.pop("_H")
        m["h_k1_power"] = m.pop("_h_pw_raw") / max(self.h_pw_ref, 1e-12)
        n = len(self.p_ids)
        pred = self.probe.predict(H)
        t7 = 2 * np.pi * np.arange(n) / n
        d = np.arctan2(pred[:, 1], pred[:, 0]) - t7
        m["h_probe_err"] = float(np.abs((d + np.pi) % (2 * np.pi) - np.pi).mean())
        logp = m.pop("_wiki_logp")
        m["wiki_kl"] = float(F.kl_div(logp.flatten(0, 1),
                                      self.wiki_logp_ref.flatten(0, 1),
                                      reduction="batchmean", log_target=True))
        return m
