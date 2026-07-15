"""Prior-free subspace discovery + line-manifold reference (Step 3 prerequisite).

Discovery uses ONLY the concept token set: centered PCA of the concept rows.
No word ordering, no group action, no frequencies enter discovery or edit
construction; ordering is used strictly as a readout (preregistration_step3.md).
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import Tensor

from clocklib.fourier import freq_power
from clocklib.ringlib import LM, Concept

NUMBERS = Concept(
    name="numbers",
    words=["one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten", "eleven", "twelve"],
    freqs=[],  # no group prior — that is the point
    templates=[
        "She counted to {} slowly.", "There are {} apples in the basket.",
        "He finished the race in {} minutes.", "The recipe calls for {} eggs.",
        "They waited for {} hours.", "Chapter {} was the longest.",
        "We planted {} trees in the yard.", "The team scored {} points.",
    ],
    successor_prompt="",  # unused: run-up counting task instead (see LineRef)
)


def array_pca(H: np.ndarray, d: int) -> tuple[np.ndarray, dict]:
    """Top-d right singular vectors of a centered cloud (n, dim) -> (dim, d)."""
    Hc = H - H.mean(0)
    _, S, Vt = np.linalg.svd(Hc, full_matrices=False)
    return Vt[:d].T, {"spectrum": [float(s) for s in S],
                      "share": float((S[:d] ** 2).sum() / (S**2).sum())}


def cloud_plane_power(H: np.ndarray, B: np.ndarray) -> float:
    """Power of the centered cloud inside span(B)."""
    Hc = H - H.mean(0)
    return float(((Hc @ B) ** 2).sum())


def pca_basis(W: Tensor, ids: list[int], d: int) -> tuple[np.ndarray, dict]:
    """Top-d right singular vectors of the centered concept rows (d_model, d)."""
    X = W[ids].numpy().astype(np.float64)
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:d].T, {"spectrum": [float(s) for s in S],
                      "share": float((S[:d] ** 2).sum() / (S**2).sum())}


def principal_cos(B1: np.ndarray, B2: np.ndarray) -> list[float]:
    """Principal-angle cosines between span(B1) and span(B2)."""
    Q1, _ = np.linalg.qr(B1)
    Q2, _ = np.linalg.qr(B2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return [float(x) for x in s]


def subspace_mat(W: Tensor, ids: list[int], B: np.ndarray) -> Tensor:
    """Edit matrix: concept rows' centered component inside span(B); zero elsewhere.
    Row mean is untouched (the projection of centered rows has zero row-mean),
    mirroring the Fourier constructions' preservation of bin 0."""
    X = W[ids].numpy().astype(np.float64)
    Xc = X - X.mean(0)
    M = np.zeros((W.shape[0], W.shape[1]))
    M[ids] = Xc @ B @ B.T
    return torch.from_numpy(M).float()


def fourier_keep_mat(W: Tensor, ids: list[int], bins: list[int]) -> Tensor:
    """Prior-based reference: DFT over the ordered concept rows, keep given bins."""
    E = W[ids].numpy().astype(np.float64)
    Fk = np.fft.rfft(E, axis=0)
    keep = np.zeros_like(Fk)
    for b in bins:
        keep[b] = Fk[b]
    M = np.zeros((W.shape[0], W.shape[1]))
    M[ids] = np.fft.irfft(keep, n=len(ids), axis=0)
    return torch.from_numpy(M).float()


class LineRef:
    """Frozen lambda=1 reference for an ordinal (no-group) concept + two cross rings.

    Mirrors RingRef's hardening: base model stays on CPU; the working copy is
    snapshotted on CPU and verified after the device move (MPS zero-corruption,
    decisions.md 2026-07-02).
    """

    def __init__(self, lm: LM, concept: Concept, cross_a: Concept, cross_b: Concept,
                 wiki_ids: Tensor, d: int = 4):
        self.lm, self.concept, self.d = lm, concept, d
        self.ids = lm.token_ids(concept)
        self.ca_ids = lm.token_ids(cross_a)
        self.cb_ids = lm.token_ids(cross_b)
        assert self.ids and self.ca_ids and self.cb_ids
        m = lm.untied_copy()
        self.W1 = m.get_input_embeddings().weight.detach().clone()  # CPU snapshot first
        assert float(self.W1.norm()) > 0
        self.model = m.to(lm.device).eval()
        back = self.model.get_input_embeddings().weight.detach().cpu()
        assert torch.allclose(back, self.W1, atol=1e-5), "weights corrupted by device move"

        self.B, self.pca_rep = pca_basis(self.W1, self.ids, d)
        Xc = self.W1[self.ids].numpy().astype(np.float64)
        Xc = Xc - Xc.mean(0)
        self.sub_ref = float(((Xc @ self.B) ** 2).sum())
        self.ca_ref = freq_power(self.W1[self.ca_ids].numpy())
        self.cb_ref = freq_power(self.W1[self.cb_ids].numpy())

        n, words = len(self.ids), concept.words
        self.succ_enc = [lm.tok(f"Count: {words[i-2]}, {words[i-1]}, {words[i]},",
                                return_tensors="pt").input_ids.to(lm.device)
                         for i in range(2, n - 1)]
        self.tpl = []
        for t in concept.templates:
            for wi, w in enumerate(words):
                enc = lm.tok(t.format(w), return_tensors="pt").input_ids
                self.tpl.append((enc.to(lm.device), wi,
                                 enc[0].tolist().index(self.ids[wi])))
        self.wiki = wiki_ids.to(lm.device)
        # Positions whose causal context contains an edited concept token: KL there
        # reflects the edit acting on genuine occurrences, not off-concept collateral.
        # High-frequency concept words (numbers) appear in wikitext; days do not.
        tainted = torch.cummax(torch.isin(wiki_ids, torch.tensor(self.ids)), 1).values
        self.wiki_clean = ~tainted.bool()

        m0 = self._raw(self.W1)
        self.H_ref = m0.pop("_H")
        self.wiki_logp_ref = m0.pop("_wiki_logp")
        Hc = self.H_ref - self.H_ref.mean(0)
        _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
        self.BH = Vt[:d].T
        self.h_sub_ref = float(((Hc @ self.BH) ** 2).sum())
        from sklearn.linear_model import Ridge

        t01 = np.arange(n) / (n - 1)
        self.probe = Ridge(alpha=1e-3).fit(self.H_ref, t01)
        self.ref = self.measure(self.W1)

    @torch.no_grad()
    def _raw(self, W: Tensor) -> dict:
        lm, n = self.lm, len(self.ids)
        self.model.get_input_embeddings().weight.data.copy_(W.to(lm.device))
        out: dict = {}
        Xc = W[self.ids].numpy().astype(np.float64)
        Xc = Xc - Xc.mean(0)
        out["sub_power"] = float(((Xc @ self.B) ** 2).sum() / max(self.sub_ref, 1e-12))
        out["emb_spearman"] = float(spearmanr(np.arange(n), Xc @ self.B[:, 0])[0])
        ca = freq_power(W[self.ca_ids].numpy())
        cb = freq_power(W[self.cb_ids].numpy())
        out["cross_a_k1_power"] = float(ca[1] / max(self.ca_ref[1], 1e-12))
        out["cross_b_k1_power"] = float(cb[1] / max(self.cb_ref[1], 1e-12))
        margins, correct = [], 0
        for j, enc in enumerate(self.succ_enc):
            logits = self.model(input_ids=enc).logits[0, -1]
            dl = logits[self.ids]
            want = j + 3  # items i=2..n-2 predict word i+1
            margins.append(float(dl[want] - dl[torch.arange(n) != want].max()))
            correct += int(dl.argmax()) == want
        out["succ_margin_mean"] = float(np.mean(margins))
        out["succ_acc"] = correct / len(self.succ_enc)
        H = np.zeros((n, lm.hidden_size))
        for enc, wi, pos in self.tpl:
            hs = self.model(input_ids=enc, output_hidden_states=True).hidden_states
            H[wi] += hs[lm.probe_layer][0, pos].float().cpu().numpy()
        H /= len(self.concept.templates)
        out["_H"] = H
        logits = self.model(input_ids=self.wiki).logits
        out["_wiki_logp"] = F.log_softmax(logits.float(), dim=-1).cpu()
        return out

    def measure(self, W: Tensor) -> dict:
        m = self._raw(W)
        H = m.pop("_H")
        n = len(self.ids)
        Hc = H - H.mean(0)
        # capture: the hidden-layer number-token cloud on its own frozen top-2 PCA
        # basis (BH[:, :2], computed once at lambda=1 from H_ref), centered, 4-dp
        m["_N_cloud_B2"] = [[round(float(v), 4) for v in row]
                            for row in (Hc @ self.BH[:, :2])]
        m["h_sub_power"] = float(((Hc @ self.BH) ** 2).sum() / max(self.h_sub_ref, 1e-12))
        m["h_spearman"] = float(spearmanr(np.arange(n), Hc @ self.BH[:, 0])[0])
        t01 = np.arange(n) / (n - 1)
        m["h_probe_err"] = float(np.abs(self.probe.predict(H) - t01).mean())
        logp = m.pop("_wiki_logp")
        kl = F.kl_div(logp.flatten(0, 1), self.wiki_logp_ref.flatten(0, 1),
                      reduction="none", log_target=True).sum(-1).reshape(logp.shape[:2])
        m["wiki_kl"] = float(kl.mean())
        m["wiki_kl_clean"] = float(kl[self.wiki_clean].mean())
        return m
