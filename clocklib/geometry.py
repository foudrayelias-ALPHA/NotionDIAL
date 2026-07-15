"""Manifold metrics. Projection bases and probes are fit ONCE at lambda=1 and frozen.

Exception (documented): topology detectors that operate on the current cloud
(persistent homology on the full cloud, Procrustes monodromy) legitimately look at the
lambda-dependent cloud — they measure what the manifold IS, not where it moved to.
Anything that compares coordinates across lambda uses frozen bases only.
"""

import numpy as np
from numpy.linalg import det, norm, svd
from ripser import ripser


def h1_max_lifetime(cloud: np.ndarray) -> float:
    """Max persistence (death - birth) of any H1 feature; 0.0 if none."""
    dgm = ripser(cloud, maxdim=1)["dgms"][1]
    if len(dgm) == 0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    return float((finite[:, 1] - finite[:, 0]).max()) if len(finite) else 0.0


def h1_noise_floor(cloud: np.ndarray, n_boot: int = 20, seed: int = 0) -> float:
    """95th percentile of max H1 lifetime over Gaussian nulls matched to cloud covariance."""
    rng = np.random.default_rng(seed)
    centered = cloud - cloud.mean(0)
    cov = np.cov(centered.T) + 1e-12 * np.eye(cloud.shape[1])
    L = np.linalg.cholesky(cov)
    vals = [
        h1_max_lifetime(rng.standard_normal(cloud.shape) @ L.T) for _ in range(n_boot)
    ]
    return float(np.quantile(vals, 0.95))


def twonn_id(cloud: np.ndarray, discard_frac: float = 0.1) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017), linear-fit variant."""
    n = cloud.shape[0]
    d2 = ((cloud[:, None, :] - cloud[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    part = np.partition(d2, 1, axis=1)[:, :2]
    r1, r2 = np.sqrt(part[:, 0]), np.sqrt(part[:, 1])
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    n_keep = int(len(mu) * (1 - discard_frac))
    mu = mu[:n_keep]
    F = np.arange(1, n_keep + 1) / (len(np.nonzero(ok)[0]) + 1)
    x, y = np.log(mu), -np.log(1 - F)
    return float((x @ y) / (x @ x)) if (x @ x) > 0 else 0.0


def participation_ratio(cloud: np.ndarray) -> float:
    """(sum eig)^2 / sum eig^2 of the covariance spectrum."""
    ev = np.linalg.eigvalsh(np.cov((cloud - cloud.mean(0)).T))
    ev = np.clip(ev, 0, None)
    s = ev.sum()
    return float(s**2 / (ev**2).sum()) if s > 0 else 0.0


def plane_stats(xy: np.ndarray) -> dict[str, float]:
    """Radius / axis ratio / winding of a token-ordered 2D loop (rows = tokens 0..p-1)."""
    c = xy - xy.mean(0)
    r = norm(c, axis=1)
    ev = np.linalg.eigvalsh(np.cov(c.T))
    ev = np.clip(ev, 0, None)
    axis_ratio = float(np.sqrt(ev[0] / ev[1])) if ev[1] > 0 else 0.0
    return {
        "radius_mean": float(r.mean()),
        "radius_cv": float(r.std() / r.mean()) if r.mean() > 0 else 0.0,
        "axis_ratio": axis_ratio,  # 1 = circle, 0 = segment
        "winding": winding_number(xy),
    }


def winding_number(xy: np.ndarray, r_floor_frac: float = 0.05) -> float:
    """Signed winding of the closed token loop around the centroid.

    Returns 0.0 when the loop has collapsed (mean radius below r_floor_frac of the
    cloud's RMS scale) — angles are meaningless there.
    """
    c = xy - xy.mean(0)
    r = norm(c, axis=1)
    if r.mean() < r_floor_frac * max(np.sqrt((c**2).sum(1).mean()), 1e-12) or r.mean() == 0:
        return 0.0
    ang = np.arctan2(c[:, 1], c[:, 0])
    d = np.diff(np.concatenate([ang, ang[:1]]))
    d = (d + np.pi) % (2 * np.pi) - np.pi
    return float(d.sum() / (2 * np.pi))


def procrustes_monodromy(cloud: np.ndarray, window: int = 5, dim: int = 2) -> dict[str, float]:
    """Cellular-sheaf readout on the token cycle: local PCA stalks, Procrustes transport.

    Fits a local `dim`-frame at each token from its cyclic neighborhood, transports
    around the cycle via orthogonal Procrustes, and accumulates rotation angle.
    Returns accumulated winding and the holonomy determinant (-1 = orientation flip).
    Operates on the CURRENT cloud by design (topology detector; see module docstring).
    """
    p = cloud.shape[0]
    frames = []
    for a in range(p):
        idx = [(a + o) % p for o in range(-window, window + 1)]
        local = cloud[idx] - cloud[idx].mean(0)
        _, _, Vt = svd(local, full_matrices=False)
        frames.append(Vt[:dim].T)  # (d, dim)
    total_angle, det_prod = 0.0, 1.0
    for a in range(p):
        F0, F1 = frames[a], frames[(a + 1) % p]
        M = F0.T @ F1
        U, _, Vt = svd(M)
        R = U @ Vt
        det_prod *= float(np.sign(det(R)))
        if det(R) > 0:
            total_angle += float(np.arctan2(R[1, 0], R[0, 0]))
    return {"monodromy_winding": total_angle / (2 * np.pi), "holonomy_det": det_prod}


class FrozenProbe:
    """Ridge probe acts -> (cos, sin) of token angle at frequency k. Fit once, freeze."""

    def __init__(self, acts: np.ndarray, k: int, p: int, alpha: float = 1e-3):
        from sklearn.linear_model import Ridge

        t = 2 * np.pi * k * np.arange(p) / p
        self.target = np.stack([np.cos(t), np.sin(t)], axis=1)
        self.k, self.p = k, p
        self.model = Ridge(alpha=alpha).fit(acts, self.target)

    def angular_error(self, acts: np.ndarray) -> float:
        """Mean |angle error| in radians of the decoded token angle."""
        pred = self.model.predict(acts)
        ang_pred = np.arctan2(pred[:, 1], pred[:, 0])
        ang_true = np.arctan2(self.target[:, 1], self.target[:, 0])
        d = ang_pred - ang_true
        d = (d + np.pi) % (2 * np.pi) - np.pi
        return float(np.abs(d).mean())
