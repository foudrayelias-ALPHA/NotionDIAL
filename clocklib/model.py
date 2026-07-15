"""Modular-addition target models: trainable adder + analytic single-frequency rung-0."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class AdderConfig:
    p: int = 113
    d: int = 128
    h: int = 512
    act: str = "relu"  # "relu" | "square"


class ModAdder(nn.Module):
    """(a, b) -> logits over c for c = (a + b) mod p. No biases anywhere."""

    def __init__(self, cfg: AdderConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.p, cfg.d)
        self.W_in = nn.Linear(cfg.d, cfg.h, bias=False)
        self.W_out = nn.Linear(cfg.h, cfg.p, bias=False)

    def hidden(self, tokens: Tensor) -> Tensor:
        assert tokens.ndim == 2 and tokens.shape[1] == 2, tokens.shape
        e = self.embed(tokens).sum(dim=1)
        z = self.W_in(e)
        match self.cfg.act:
            case "relu":
                return F.relu(z)
            case "square":
                return z * z
            case _:
                raise ValueError(self.cfg.act)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.W_out(self.hidden(tokens))


def all_pairs(p: int, device: str = "cpu") -> tuple[Tensor, Tensor]:
    """All p^2 (a,b) token pairs and their labels (a+b) mod p."""
    a = torch.arange(p, device=device)
    aa, bb = torch.meshgrid(a, a, indexing="ij")
    tokens = torch.stack([aa.reshape(-1), bb.reshape(-1)], dim=1)
    labels = (tokens[:, 0] + tokens[:, 1]) % p
    return tokens, labels


@torch.no_grad()
def accuracy(model: nn.Module, tokens: Tensor, labels: Tensor) -> float:
    return (model(tokens).argmax(dim=-1) == labels).float().mean().item()


class ModAdder2Hot(nn.Module):
    """Same function as ModAdder, but the embedding is a Linear over two-hot inputs.

    x = W_e @ (onehot_a + onehot_b) == E[a] + E[b] with W_e.weight = E.T. Exists because
    param_decomp's stochastic masks assume every decomposed module shares the same
    leading batch dims; nn.Embedding sees (B, 2) tokens while the MLP sees (B,).
    Two-hot inputs make all three modules (B,)-leading. Weights are interchangeable
    with ModAdder by transpose.
    """

    def __init__(self, cfg: AdderConfig):
        super().__init__()
        self.cfg = cfg
        self.W_e = nn.Linear(cfg.p, cfg.d, bias=False)
        self.W_in = nn.Linear(cfg.d, cfg.h, bias=False)
        self.W_out = nn.Linear(cfg.h, cfg.p, bias=False)

    @classmethod
    def from_adder(cls, adder: ModAdder) -> "ModAdder2Hot":
        model = cls(adder.cfg)
        with torch.no_grad():
            model.W_e.weight.copy_(adder.embed.weight.T)
            model.W_in.weight.copy_(adder.W_in.weight)
            model.W_out.weight.copy_(adder.W_out.weight)
        return model.eval()

    def hidden(self, x: Tensor) -> Tensor:
        assert x.ndim == 2 and x.shape[1] == self.cfg.p, x.shape
        z = self.W_in(self.W_e(x))
        match self.cfg.act:
            case "relu":
                return F.relu(z)
            case "square":
                return z * z
            case _:
                raise ValueError(self.cfg.act)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_out(self.hidden(x))

    def embedding_matrix(self) -> Tensor:
        """E as (p, d) — rows are token embeddings (the canonical circle)."""
        return self.W_e.weight.T


class TwoOpAdder(nn.Module):
    """(a, b, op) -> logits over c = (a + b) mod p [op=0] or (a - b) mod p [op=1].

    Input is [onehot_a ; onehot_b ; onehot_op] (2p+2 dims): a and b must be
    distinguishable because subtraction is antisymmetric, so embeddings are untied.
    Same module names as ModAdder2Hot (W_e, W_in, W_out), all with (B,) leading dims.
    Purpose: DIFFERENTIAL USE — the op-conditional mechanisms are used on half the
    inputs each, restoring the sparsity SPD's causal-importance machinery keys on.
    """

    def __init__(self, cfg: AdderConfig):
        super().__init__()
        self.cfg = cfg
        self.W_e = nn.Linear(2 * cfg.p + 2, cfg.d, bias=False)
        self.W_in = nn.Linear(cfg.d, cfg.h, bias=False)
        self.W_out = nn.Linear(cfg.h, cfg.p, bias=False)

    def hidden(self, x: Tensor) -> Tensor:
        assert x.ndim == 2 and x.shape[1] == 2 * self.cfg.p + 2, x.shape
        z = self.W_in(self.W_e(x))
        assert self.cfg.act == "relu"
        return F.relu(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_out(self.hidden(x))

    def block(self, which: str) -> Tensor:
        """Embedding block as (p, d) token rows: 'a', 'b', or (2, d) for 'op'."""
        W = self.W_e.weight  # (d, 2p+2)
        p = self.cfg.p
        match which:
            case "a":
                return W[:, :p].T
            case "b":
                return W[:, p : 2 * p].T
            case "op":
                return W[:, 2 * p :].T
            case _:
                raise ValueError(which)


def twoop_encode(a: Tensor, b: Tensor, op: Tensor, p: int) -> Tensor:
    """(B,) index tensors -> (B, 2p+2) concatenated one-hots."""
    B = a.shape[0]
    x = torch.zeros(B, 2 * p + 2, device=a.device)
    x[torch.arange(B), a] = 1.0
    x[torch.arange(B), p + b] = 1.0
    x[torch.arange(B), 2 * p + op] = 1.0
    return x


def twoop_all(p: int, device: str = "cpu") -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """All 2*p^2 (a, b, op) triples and labels."""
    a = torch.arange(p, device=device)
    aa, bb, oo = torch.meshgrid(a, a, torch.arange(2, device=device), indexing="ij")
    aa, bb, oo = aa.reshape(-1), bb.reshape(-1), oo.reshape(-1)
    labels = torch.where(oo == 0, (aa + bb) % p, (aa - bb) % p)
    return aa, bb, oo, labels


def two_hot(tokens: Tensor, p: int) -> Tensor:
    """(B, 2) token pairs -> (B, p) float two-hot vectors (2.0 at a when a == b)."""
    assert tokens.ndim == 2 and tokens.shape[1] == 2
    x = torch.zeros(tokens.shape[0], p, dtype=torch.float32, device=tokens.device)
    x.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float32))
    return x


def build_analytic_rung0(p: int, k0: int, n_hidden: int = 32) -> ModAdder:
    """Single-frequency quadratic clock, exact by construction.

    logits(a,b,c) = [1 + cos(w(a-b))] * cos(w(a+b-c)) + c-independent constant,
    w = 2*pi*k0/p. Amplitude > 0 for odd p => exactly 100% accuracy.
    """
    assert p % 2 == 1 and math.gcd(k0, p) == 1 and n_hidden > 4
    cfg = AdderConfig(p=p, d=2, h=n_hidden, act="square")
    model = ModAdder(cfg)
    w = 2 * math.pi * k0 / p
    a = torch.arange(p, dtype=torch.float64)
    phases = math.pi * torch.arange(n_hidden, dtype=torch.float64) / n_hidden
    with torch.no_grad():
        model.embed.weight.copy_(
            torch.stack([torch.cos(w * a), torch.sin(w * a)], dim=1).float()
        )
        model.W_in.weight.copy_(
            torch.stack([torch.cos(phases), torch.sin(phases)], dim=1).float()
        )
        c = torch.arange(p, dtype=torch.float64)
        model.W_out.weight.copy_(
            ((2.0 / n_hidden) * torch.cos(w * c[:, None] - 2 * phases[None, :])).float()
        )
    return model
