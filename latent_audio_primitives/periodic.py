from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .latent_math import as_time_major
from .schema import LatentItem


@dataclass(frozen=True, slots=True)
class PeriodicityReport:
    best_lag: int
    best_score: float
    boundary_l2: float
    velocity_l2: float
    spectral_centroid: float


def latent_autocorrelation(latent: LatentItem | np.ndarray, *, max_lag: int | None = None) -> np.ndarray:
    """Normalized autocorrelation of a latent sequence over time."""

    z = as_time_major(latent).astype(np.float32)
    z = z - z.mean(axis=0, keepdims=True)
    frames = z.shape[0]
    if frames < 2:
        return np.ones(1, dtype=np.float32)
    max_lag = frames - 1 if max_lag is None else max(1, min(int(max_lag), frames - 1))
    denom = float(np.sum(z * z))
    if denom <= 1e-12:
        return np.ones(max_lag + 1, dtype=np.float32)
    scores = [1.0]
    for lag in range(1, max_lag + 1):
        left = z[:-lag]
        right = z[lag:]
        scores.append(float(np.sum(left * right) / denom))
    return np.asarray(scores, dtype=np.float32)


def best_period_lag(
    latent: LatentItem | np.ndarray,
    *,
    min_lag: int = 2,
    max_lag: int | None = None,
) -> tuple[int, float]:
    """Return the strongest autocorrelation lag in a search range."""

    ac = latent_autocorrelation(latent, max_lag=max_lag)
    if ac.shape[0] <= min_lag:
        return 0, 0.0
    start = max(1, int(min_lag))
    index = int(np.argmax(ac[start:]) + start)
    return index, float(ac[index])


def latent_fft_energy(latent: LatentItem | np.ndarray, *, remove_dc: bool = True) -> np.ndarray:
    """Mean latent-time FFT energy per frequency bin."""

    z = as_time_major(latent).astype(np.float32)
    if remove_dc:
        z = z - z.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(z, axis=0)
    energy = np.mean(np.abs(spectrum) ** 2, axis=1)
    return energy.astype(np.float32, copy=False)


def latent_spectral_centroid(latent: LatentItem | np.ndarray) -> float:
    """Normalized centroid of latent-time energy, 0=slow/DC, 1=Nyquist."""

    energy = latent_fft_energy(latent, remove_dc=True)
    if energy.shape[0] <= 1:
        return 0.0
    freqs = np.linspace(0.0, 1.0, energy.shape[0], dtype=np.float32)
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    return float(np.sum(freqs * energy) / total)


def loop_boundary_loss(latent: LatentItem | np.ndarray, *, window: int = 8, velocity_weight: float = 1.0) -> tuple[float, float, float]:
    """Return total/state/velocity loop boundary mismatch."""

    z = as_time_major(latent).astype(np.float32)
    frames = z.shape[0]
    if frames < 2:
        return 0.0, 0.0, 0.0
    k = max(1, min(int(window), frames // 2))
    start = z[:k].mean(axis=0)
    end = z[-k:].mean(axis=0)
    state = float(np.linalg.norm(start - end))
    if k < 2:
        velocity = 0.0
    else:
        start_v = np.diff(z[:k], axis=0).mean(axis=0)
        end_v = np.diff(z[-k:], axis=0).mean(axis=0)
        velocity = float(np.linalg.norm(start_v - end_v))
    total = state + float(velocity_weight) * velocity
    return float(total), state, velocity


def periodicity_report(
    latent: LatentItem | np.ndarray,
    *,
    min_lag: int = 2,
    max_lag: int | None = None,
    boundary_window: int = 8,
) -> PeriodicityReport:
    lag, score = best_period_lag(latent, min_lag=min_lag, max_lag=max_lag)
    _total, boundary, velocity = loop_boundary_loss(latent, window=boundary_window)
    centroid = latent_spectral_centroid(latent)
    return PeriodicityReport(
        best_lag=lag,
        best_score=score,
        boundary_l2=boundary,
        velocity_l2=velocity,
        spectral_centroid=centroid,
    )
