"""SAME latent geometry, whitening, distance, and transport utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .latent_math import as_time_major, latent_summary
from .schema import LatentItem


@dataclass(frozen=True, slots=True)
class LatentGeometry:
    """Dataset geometry over SAME-style latent frames."""

    dim: int
    frame_count: int
    mean: np.ndarray
    components: np.ndarray
    variances: np.ndarray
    total_variance: float | None = None
    eps: float = 1e-5

    def __post_init__(self) -> None:
        _check_vector(self.mean, self.dim, "mean")
        components = np.asarray(self.components, dtype=np.float32)
        if components.ndim != 2 or components.shape[1] != self.dim:
            raise ValueError(f"components must have shape K x {self.dim}, got {components.shape}")
        variances = np.asarray(self.variances, dtype=np.float32).reshape(-1)
        if variances.shape[0] != components.shape[0]:
            raise ValueError("variances must match number of components")
        if not np.isfinite(components).all() or not np.isfinite(variances).all():
            raise ValueError("geometry contains NaN or infinite values")
        if self.total_variance is None:
            total_variance = float(np.sum(np.maximum(variances, 0.0)))
        else:
            total_variance = float(self.total_variance)
        if not np.isfinite(total_variance) or total_variance <= 0:
            total_variance = float(max(np.sum(np.maximum(variances, 0.0)), self.eps))
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "variances", np.maximum(variances, self.eps))
        object.__setattr__(self, "total_variance", total_variance)


def latent_frame_matrix(latents: Sequence[LatentItem | np.ndarray] | LatentItem | np.ndarray) -> np.ndarray:
    """Concatenate one or many latent sequences into a frame matrix ``N x D``."""

    if isinstance(latents, (LatentItem, np.ndarray)):
        return as_time_major(latents)
    frames = [as_time_major(latent) for latent in latents]
    if not frames:
        raise ValueError("at least one latent is required")
    dim = frames[0].shape[1]
    if any(frame.shape[1] != dim for frame in frames):
        raise ValueError("all latents must have the same feature dimension")
    return np.concatenate(frames, axis=0).astype(np.float32, copy=False)


def fit_latent_geometry(
    latents: Sequence[LatentItem | np.ndarray] | LatentItem | np.ndarray,
    *,
    n_components: int | None = None,
    eps: float = 1e-5,
) -> LatentGeometry:
    """Fit PCA geometry over latent frames."""

    x = latent_frame_matrix(latents).astype(np.float32, copy=False)
    frame_count, dim = x.shape
    if frame_count < 2:
        raise ValueError("at least two latent frames are required")
    keep = dim if n_components is None else max(1, min(int(n_components), dim))
    mean = x.mean(axis=0)
    centered = x - mean
    cov = (centered.T @ centered) / max(frame_count - 1, 1)
    values, vectors = np.linalg.eigh(cov.astype(np.float64))
    full_variances = np.maximum(values, 0.0)
    order = np.argsort(values)[::-1][:keep]
    variances = full_variances[order].astype(np.float32)
    components = vectors[:, order].T.astype(np.float32)
    return LatentGeometry(
        dim=dim,
        frame_count=frame_count,
        mean=mean.astype(np.float32),
        components=components,
        variances=variances,
        total_variance=float(np.sum(full_variances)),
        eps=eps,
    )


def pca_project(latent: LatentItem | np.ndarray, geometry: LatentGeometry, *, whiten: bool = False) -> np.ndarray:
    """Project latent frames into the fitted PCA geometry."""

    z = _check_latent_dim(as_time_major(latent), geometry.dim)
    coeffs = (z - geometry.mean) @ geometry.components.T
    if whiten:
        coeffs = coeffs / np.sqrt(geometry.variances + geometry.eps)
    return coeffs.astype(np.float32, copy=False)


def pca_reconstruct(coeffs: np.ndarray, geometry: LatentGeometry, *, whitened: bool = False) -> np.ndarray:
    """Reconstruct latent frames from PCA coefficients."""

    c = np.asarray(coeffs, dtype=np.float32)
    if c.ndim != 2 or c.shape[1] != geometry.components.shape[0]:
        raise ValueError(
            f"coeffs must have shape T x {geometry.components.shape[0]}, got {c.shape}"
        )
    if whitened:
        c = c * np.sqrt(geometry.variances + geometry.eps)
    return (c @ geometry.components + geometry.mean).astype(np.float32, copy=False)


def whiten_latent(latent: LatentItem | np.ndarray, geometry: LatentGeometry) -> np.ndarray:
    """Return PCA-whitened frame coefficients."""

    return pca_project(latent, geometry, whiten=True)


def mahalanobis_summary_distance(
    a: LatentItem | np.ndarray,
    b: LatentItem | np.ndarray,
    geometry: LatentGeometry,
    *,
    components: int | None = None,
) -> float:
    """Mahalanobis distance between whole-clip latent means under dataset PCA."""

    za = _check_latent_dim(as_time_major(a), geometry.dim).mean(axis=0)
    zb = _check_latent_dim(as_time_major(b), geometry.dim).mean(axis=0)
    delta = (za - zb) @ geometry.components.T
    if components is not None:
        delta = delta[:components]
        variances = geometry.variances[:components]
    else:
        variances = geometry.variances
    return float(np.sqrt(np.sum((delta * delta) / (variances + geometry.eps))))


def covariance_transport(
    latent: LatentItem | np.ndarray,
    reference: Sequence[LatentItem | np.ndarray] | LatentItem | np.ndarray,
    *,
    alpha: float = 1.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """Transport latent frame statistics toward a reference Gaussian.

    This is the full-covariance analogue of AdaIN-style mean/std matching:

        z_target = (z - mu_s) C_s^{-1/2} C_r^{1/2} + mu_r

    ``alpha`` interpolates between original and transported latents.
    """

    z = as_time_major(latent).astype(np.float32, copy=False)
    ref = latent_frame_matrix(reference)
    if ref.shape[1] != z.shape[1]:
        raise ValueError("reference latent dimension must match source")
    mu_s, cov_s = _mean_cov(z, eps=eps)
    mu_r, cov_r = _mean_cov(ref, eps=eps)
    transform = _matrix_invsqrt(cov_s, eps=eps) @ _matrix_sqrt(cov_r, eps=eps)
    transported = (z - mu_s) @ transform + mu_r
    return ((1.0 - alpha) * z + alpha * transported).astype(np.float32, copy=False)


def latent_barycenter(latents: Sequence[LatentItem | np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Framewise barycenter for equal-length latent sequences."""

    arrays = [as_time_major(latent) for latent in latents]
    if not arrays:
        raise ValueError("at least one latent is required")
    shape = arrays[0].shape
    if any(arr.shape != shape for arr in arrays):
        raise ValueError("latent_barycenter requires equal-length latent arrays")
    if weights is None:
        w = np.full(len(arrays), 1.0 / len(arrays), dtype=np.float32)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.shape[0] != len(arrays):
            raise ValueError("weights must match number of latents")
        total = float(w.sum())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / total
    stacked = np.stack(arrays, axis=0)
    return np.tensordot(w, stacked, axes=(0, 0)).astype(np.float32, copy=False)


def geometry_report(latents: Sequence[LatentItem | np.ndarray], *, n_components: int = 8) -> dict[str, float | list[float]]:
    """Small diagnostic report for a latent set."""

    geometry = fit_latent_geometry(latents, n_components=n_components)
    total = float(max(geometry.total_variance or 0.0, geometry.eps))
    explained = (geometry.variances / total).astype(float).tolist()
    kept_fraction = float(np.sum(geometry.variances) / total)
    summaries = np.stack([latent_summary(latent) for latent in latents])
    return {
        "frame_count": float(geometry.frame_count),
        "dim": float(geometry.dim),
        "explained_variance": explained,
        "kept_variance_fraction": kept_fraction,
        "summary_mean_norm": float(np.linalg.norm(summaries.mean(axis=0))),
        "summary_std_mean": float(summaries.std(axis=0).mean()),
    }


def _mean_cov(x: np.ndarray, *, eps: float) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    centered = x - mean
    cov = (centered.T @ centered) / max(x.shape[0] - 1, 1)
    cov = cov + np.eye(x.shape[1], dtype=np.float32) * eps
    return mean.astype(np.float32), cov.astype(np.float32)


def _matrix_sqrt(matrix: np.ndarray, *, eps: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix.astype(np.float64))
    values = np.maximum(values, eps)
    return (vectors @ np.diag(np.sqrt(values)) @ vectors.T).astype(np.float32)


def _matrix_invsqrt(matrix: np.ndarray, *, eps: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix.astype(np.float64))
    values = np.maximum(values, eps)
    return (vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T).astype(np.float32)


def _check_vector(value: np.ndarray, dim: int, name: str) -> None:
    arr = np.asarray(value)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def _check_latent_dim(z: np.ndarray, dim: int) -> np.ndarray:
    if z.shape[1] != dim:
        raise ValueError(f"latent dimension {z.shape[1]} does not match geometry dimension {dim}")
    return z
