"""Token-space Fourier analysis of matrices and point clouds indexed by token 0..p-1."""

import numpy as np


def freq_power(M: np.ndarray, exclude_dc: bool = True) -> np.ndarray:
    """Per-frequency power of M (p x d) along the token axis.

    Returns power[k] for k = 0..p//2 (rfft bins), summed over columns.
    Frequencies k and p-k are conjugate bins and land in the same k <= p//2 entry.
    """
    assert M.ndim == 2
    F = np.fft.rfft(M, axis=0)
    power = (np.abs(F) ** 2).sum(axis=1)
    if exclude_dc:
        power = power.copy()
        power[0] = 0.0
    return power


def key_freqs(M: np.ndarray, frac_thresh: float = 0.02) -> list[int]:
    """Frequencies carrying more than `frac_thresh` of total non-DC power."""
    power = freq_power(M)
    total = power.sum()
    assert total > 0
    return [int(k) for k in np.nonzero(power / total > frac_thresh)[0] if k > 0]


def fourier_plane_basis(M: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal basis (2 x d) of frequency k's plane in the column space of M (p x d).

    Rows of the returned Q approximate the cos/sin directions: least-squares Fourier
    coefficient vectors, orthonormalized by QR. Fit ONCE at lambda=1 and freeze.
    """
    p, _ = M.shape
    assert 0 < k <= p // 2
    t = 2 * np.pi * k * np.arange(p) / p
    u_cos = (2.0 / p) * (np.cos(t) @ M)
    u_sin = (2.0 / p) * (np.sin(t) @ M)
    B = np.stack([u_cos, u_sin])
    Q, R = np.linalg.qr(B.T)
    assert np.linalg.matrix_rank(R) == 2, f"frequency {k} plane is degenerate"
    return Q.T


def plane_coords(M: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Project rows of M (p x d) into a frozen plane basis Q (2 x d) -> (p x 2)."""
    return M @ Q.T


def plane_power(M: np.ndarray, k: int) -> float:
    """Power of M in frequency band k (rfft bin), summed over columns."""
    return float(freq_power(M)[k])


def dominant_freq(v: np.ndarray) -> tuple[int, float]:
    """Dominant non-DC frequency of a length-p vector and its share of non-DC power."""
    assert v.ndim == 1
    F = np.abs(np.fft.rfft(v)) ** 2
    F[0] = 0.0
    total = F.sum()
    if total == 0:
        return 0, 0.0
    k = int(F.argmax())
    return k, float(F[k] / total)
