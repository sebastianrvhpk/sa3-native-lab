from __future__ import annotations

import numpy as np

from .schema import LatentItem


def as_time_major(latent: LatentItem | np.ndarray, *, channel_first: bool = False) -> np.ndarray:
    """Return a finite float32 latent array with shape ``(time, dim)``."""

    if isinstance(latent, LatentItem):
        arr = latent.latent
    else:
        arr = np.asarray(latent, dtype=np.float32)
        if channel_first:
            arr = arr.T
    if arr.ndim != 2:
        raise ValueError(f"latent must be 2D, got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"latent must have non-empty axes, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("latent contains NaN or infinite values")
    return np.ascontiguousarray(arr, dtype=np.float32)


def latent_velocity(latent: LatentItem | np.ndarray) -> np.ndarray:
    """Return first-order temporal differences with shape ``(T-1, D)``."""

    z = as_time_major(latent)
    if z.shape[0] < 2:
        return np.zeros((0, z.shape[1]), dtype=np.float32)
    return np.diff(z, axis=0)


def latent_summary(
    latent: LatentItem | np.ndarray,
    *,
    include_std: bool = True,
    include_mean_abs_velocity: bool = True,
) -> np.ndarray:
    """Summarize a latent sequence for simple nearest-neighbor retrieval.

    The default vector is ``concat(mean, std, mean(abs(diff)))``. It is a
    deliberately plain baseline: future learned embeddings should prove that
    they beat this before becoming part of the memory kernel.
    """

    z = as_time_major(latent)
    parts = [z.mean(axis=0)]
    if include_std:
        parts.append(z.std(axis=0))
    if include_mean_abs_velocity:
        velocity = latent_velocity(z)
        if velocity.shape[0] == 0:
            parts.append(np.zeros(z.shape[1], dtype=np.float32))
        else:
            parts.append(np.abs(velocity).mean(axis=0))
    return np.concatenate(parts).astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-8) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError(f"vectors must have same shape, got {va.shape} and {vb.shape}")
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= eps:
        return 0.0
    return float(np.dot(va, vb) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.shape != vb.shape:
        raise ValueError(f"vectors must have same shape, got {va.shape} and {vb.shape}")
    return float(np.linalg.norm(va - vb))


def _window(latent: LatentItem | np.ndarray, side: str, k: int) -> np.ndarray:
    z = as_time_major(latent)
    if k < 1:
        raise ValueError("k must be at least 1")
    width = min(k, z.shape[0])
    if side == "start":
        return z[:width]
    if side == "end":
        return z[-width:]
    raise ValueError("side must be 'start' or 'end'")


def boundary_summary(latent: LatentItem | np.ndarray, side: str, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(state, velocity)`` for a start or end boundary window."""

    w = _window(latent, side, k)
    state = w.mean(axis=0)
    if w.shape[0] < 2:
        velocity = np.zeros(w.shape[1], dtype=np.float32)
    else:
        velocity = np.diff(w, axis=0).mean(axis=0)
    return state.astype(np.float32, copy=False), velocity.astype(np.float32, copy=False)
