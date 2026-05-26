from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class AudioDescriptorConfig:
    """Lightweight MIR descriptor settings."""

    n_fft: int = 2048
    hop_length: int = 512
    rolloff_percent: float = 0.85
    eps: float = 1e-10


def audio_descriptor_report(
    audio: Any,
    sample_rate: int,
    *,
    config: AudioDescriptorConfig | None = None,
) -> dict[str, float]:
    """Return dependency-light audio descriptors for experiment auditing.

    These are observability metrics, not ground truth musical labels. They are
    useful for comparing latent-DSP variants after SAME decode or SA3 polish.
    """

    cfg = config or AudioDescriptorConfig()
    x = _as_audio_array(audio)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    mono = x.mean(axis=0)
    duration = mono.shape[-1] / float(sample_rate)
    rms = float(np.sqrt(np.mean(mono * mono) + cfg.eps))
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    crest = peak / max(rms, cfg.eps)
    zcr = _zero_crossing_rate(mono)
    stft = _magnitude_stft(mono, cfg.n_fft, cfg.hop_length)
    freqs = np.fft.rfftfreq(cfg.n_fft, d=1.0 / sample_rate).astype(np.float32)
    spectrum = stft + cfg.eps
    energy = spectrum.sum(axis=0)
    centroid = _safe_mean((freqs[:, None] * spectrum).sum(axis=0) / energy)
    bandwidth = _safe_mean(np.sqrt(((freqs[:, None] - centroid) ** 2 * spectrum).sum(axis=0) / energy))
    rolloff = _rolloff_hz(spectrum, freqs, cfg.rolloff_percent)
    flatness = _safe_mean(np.exp(np.mean(np.log(spectrum), axis=0)) / np.mean(spectrum, axis=0))
    flux = _spectral_flux(spectrum)
    low, mid, high = _band_energy_ratios(spectrum, freqs)
    width, correlation = _stereo_metrics(x, cfg.eps)

    return {
        "duration_seconds": float(duration),
        "rms": rms,
        "rms_dbfs": float(20.0 * np.log10(max(rms, cfg.eps))),
        "peak": peak,
        "crest_factor_db": float(20.0 * np.log10(max(crest, cfg.eps))),
        "zero_crossing_rate": float(zcr),
        "spectral_centroid_hz": float(centroid),
        "spectral_bandwidth_hz": float(bandwidth),
        "spectral_rolloff_hz": float(rolloff),
        "spectral_flatness": float(flatness),
        "spectral_flux": float(flux),
        "low_energy_ratio": float(low),
        "mid_energy_ratio": float(mid),
        "high_energy_ratio": float(high),
        "stereo_width": float(width),
        "stereo_correlation": float(correlation),
    }


def descriptor_delta(
    before: dict[str, float],
    after: dict[str, float],
    *,
    keys: tuple[str, ...] = (
        "rms_dbfs",
        "spectral_centroid_hz",
        "spectral_flux",
        "spectral_flatness",
        "stereo_width",
    ),
) -> dict[str, float]:
    """Compute ``after - before`` for common descriptor keys."""

    return {key: float(after[key] - before[key]) for key in keys if key in before and key in after}


def _as_audio_array(audio: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(audio, torch.Tensor):
        arr = audio.detach().float().cpu().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"audio must have shape C x N or N, got {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    if not np.isfinite(arr).all():
        raise ValueError("audio contains NaN or infinite values")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _magnitude_stft(mono: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    n_fft = max(16, int(n_fft))
    hop_length = max(1, int(hop_length))
    if mono.shape[0] < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.shape[0]))
    frame_count = 1 + max(0, (mono.shape[0] - n_fft) // hop_length)
    frames = np.empty((frame_count, n_fft), dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_length
        frames[index] = mono[start : start + n_fft]
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    return np.abs(spectrum).T.astype(np.float32, copy=False)


def _zero_crossing_rate(mono: np.ndarray) -> float:
    if mono.shape[0] < 2:
        return 0.0
    signs = np.signbit(mono)
    return float(np.mean(signs[1:] != signs[:-1]))


def _rolloff_hz(spectrum: np.ndarray, freqs: np.ndarray, percent: float) -> float:
    percent = min(max(float(percent), 0.0), 1.0)
    cumulative = np.cumsum(spectrum, axis=0)
    threshold = cumulative[-1:, :] * percent
    indices = np.argmax(cumulative >= threshold, axis=0)
    return _safe_mean(freqs[indices])


def _spectral_flux(spectrum: np.ndarray) -> float:
    if spectrum.shape[1] < 2:
        return 0.0
    normalized = spectrum / np.maximum(np.linalg.norm(spectrum, axis=0, keepdims=True), 1e-10)
    diff = np.diff(normalized, axis=1)
    return float(np.sqrt(np.mean(diff * diff)))


def _band_energy_ratios(spectrum: np.ndarray, freqs: np.ndarray) -> tuple[float, float, float]:
    power = spectrum * spectrum
    total = float(power.sum()) + 1e-10
    low = float(power[freqs < 250.0].sum() / total)
    mid = float(power[(freqs >= 250.0) & (freqs < 4000.0)].sum() / total)
    high = float(power[freqs >= 4000.0].sum() / total)
    return low, mid, high


def _stereo_metrics(audio: np.ndarray, eps: float) -> tuple[float, float]:
    if audio.shape[0] < 2:
        return 0.0, 1.0
    left = audio[0]
    right = audio[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    width = float(np.sqrt(np.mean(side * side) + eps) / np.sqrt(np.mean(mid * mid) + eps))
    left_z = left - left.mean()
    right_z = right - right.mean()
    denom = float(np.linalg.norm(left_z) * np.linalg.norm(right_z))
    corr = 0.0 if denom <= eps else float(np.dot(left_z, right_z) / denom)
    return width, corr


def _safe_mean(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))
