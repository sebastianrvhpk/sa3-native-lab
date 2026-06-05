"""Time-varying evidence lanes for audio and SAME latent trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .latent_math import as_time_major


@dataclass(frozen=True, slots=True)
class ControlLane:
    """A time-varying control signal aligned to audio or latent time."""

    name: str
    values: np.ndarray
    rate_hz: float
    confidence: np.ndarray | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if values.size < 1:
            raise ValueError("control lane values must be non-empty")
        if not np.isfinite(values).all():
            raise ValueError("control lane values contain NaN or infinite values")
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        confidence = None
        if self.confidence is not None:
            confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
            if confidence.shape != values.shape:
                raise ValueError("confidence must have the same shape as values")
            if not np.isfinite(confidence).all():
                raise ValueError("confidence contains NaN or infinite values")
            confidence = np.clip(confidence, 0.0, 1.0)
        object.__setattr__(self, "values", np.ascontiguousarray(values))
        object.__setattr__(self, "confidence", confidence)

    @property
    def frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(self.frames / self.rate_hz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": self.values.astype(float).tolist(),
            "rate_hz": float(self.rate_hz),
            "confidence": None if self.confidence is None else self.confidence.astype(float).tolist(),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ControlLane":
        return cls(
            name=str(payload["name"]),
            values=np.asarray(payload["values"], dtype=np.float32),
            rate_hz=float(payload["rate_hz"]),
            confidence=None if payload.get("confidence") is None else np.asarray(payload["confidence"], dtype=np.float32),
            source=payload.get("source"),
            metadata=dict(payload.get("metadata", {})),
        )


def latent_motion_lane(
    latent: Any,
    *,
    latent_rate: float,
    name: str = "latent_motion_energy",
    source: str | None = None,
) -> ControlLane:
    """Return per-frame latent motion energy ``||z_t - z_{t-1}||``."""

    z = as_time_major(latent)
    diffs = np.diff(z, axis=0, prepend=z[:1])
    values = np.sqrt(np.mean(diffs * diffs, axis=1)).astype(np.float32)
    return ControlLane(name=name, values=values, rate_hz=latent_rate, source=source)


def latent_channel_energy_lane(
    latent: Any,
    *,
    latent_rate: float,
    channels: Sequence[int] | None = None,
    name: str = "latent_channel_energy",
    source: str | None = None,
) -> ControlLane:
    """Return RMS energy over selected latent channels per frame."""

    z = as_time_major(latent)
    if channels is not None:
        z = z[:, [int(channel) for channel in channels]]
    values = np.sqrt(np.mean(z * z, axis=1)).astype(np.float32)
    return ControlLane(name=name, values=values, rate_hz=latent_rate, source=source)


def audio_envelope_lane(
    audio: Any,
    sample_rate: int,
    *,
    frame_seconds: float = 0.05,
    hop_seconds: float | None = None,
    name: str = "rms_envelope",
    source: str | None = None,
) -> ControlLane:
    """Return a framewise audio control lane.

    Supported names: ``rms_envelope``, ``zero_crossing_rate``,
    ``spectral_centroid_hz``.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    x = _as_audio_array(audio).mean(axis=0)
    frame = max(16, int(round(frame_seconds * sample_rate)))
    hop = max(1, int(round((hop_seconds if hop_seconds is not None else frame_seconds) * sample_rate)))
    frames = _frame_audio(x, frame, hop)
    if name == "rms_envelope":
        values = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    elif name == "zero_crossing_rate":
        values = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
    elif name == "spectral_centroid_hz":
        window = np.hanning(frame).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32) + 1e-10
        freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate).astype(np.float32)
        values = (spectrum * freqs[None, :]).sum(axis=1) / spectrum.sum(axis=1)
    else:
        raise ValueError("name must be 'rms_envelope', 'zero_crossing_rate', or 'spectral_centroid_hz'")
    return ControlLane(name=name, values=values.astype(np.float32), rate_hz=sample_rate / hop, source=source)


def normalize_control_lane(lane: ControlLane, *, mode: str = "minmax", eps: float = 1e-8) -> ControlLane:
    """Normalize lane values while preserving alignment metadata."""

    values = lane.values.astype(np.float32)
    if mode == "minmax":
        lo = float(values.min())
        hi = float(values.max())
        normalized = (values - lo) / max(hi - lo, eps)
    elif mode == "zscore":
        normalized = (values - float(values.mean())) / max(float(values.std()), eps)
    elif mode == "unit_peak":
        normalized = values / max(float(np.max(np.abs(values))), eps)
    elif mode == "none":
        normalized = values.copy()
    else:
        raise ValueError("mode must be 'minmax', 'zscore', 'unit_peak', or 'none'")
    return ControlLane(
        name=lane.name,
        values=normalized,
        rate_hz=lane.rate_hz,
        confidence=lane.confidence,
        source=lane.source,
        metadata={**lane.metadata, "normalization": mode},
    )


def resample_control_lane(lane: ControlLane, target_frames: int, *, target_rate_hz: float | None = None) -> ControlLane:
    """Interpolate a lane to a target frame count for comparison or display."""

    target_frames = int(target_frames)
    if target_frames < 1:
        raise ValueError("target_frames must be at least 1")
    if lane.frames == target_frames:
        values = lane.values.copy()
        confidence = None if lane.confidence is None else lane.confidence.copy()
    else:
        source_x = np.linspace(0.0, 1.0, lane.frames, dtype=np.float32)
        target_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
        values = np.interp(target_x, source_x, lane.values).astype(np.float32)
        confidence = None if lane.confidence is None else np.interp(target_x, source_x, lane.confidence).astype(np.float32)
    return ControlLane(
        name=lane.name,
        values=values,
        rate_hz=float(target_rate_hz if target_rate_hz is not None else lane.rate_hz),
        confidence=confidence,
        source=lane.source,
        metadata=dict(lane.metadata),
    )


def control_lane_distance(a: ControlLane, b: ControlLane, *, normalize: bool = True) -> float:
    """Return RMS distance between two lanes after frame-count alignment."""

    frames = max(a.frames, b.frames)
    av = resample_control_lane(a, frames).values
    bv = resample_control_lane(b, frames).values
    if normalize:
        av = normalize_control_lane(ControlLane(a.name, av, a.rate_hz), mode="zscore").values
        bv = normalize_control_lane(ControlLane(b.name, bv, b.rate_hz), mode="zscore").values
    return float(np.sqrt(np.mean((av - bv) ** 2)))


def control_lane_similarity(a: ControlLane, b: ControlLane) -> float:
    """Return cosine similarity between z-scored, frame-aligned lane values."""

    frames = max(a.frames, b.frames)
    av = normalize_control_lane(ControlLane(a.name, resample_control_lane(a, frames).values, a.rate_hz), mode="zscore").values
    bv = normalize_control_lane(ControlLane(b.name, resample_control_lane(b, frames).values, b.rate_hz), mode="zscore").values
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return 0.0 if denom <= 1e-8 else float(np.dot(av, bv) / denom)


def save_control_lanes(lanes: Sequence[ControlLane], path: str | Path) -> Path:
    """Save notebook control lanes as a portable JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([lane.to_dict() for lane in lanes], indent=2), encoding="utf-8")
    return path


def load_control_lanes(path: str | Path) -> list[ControlLane]:
    """Load notebook control lanes from a JSON artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("control lane file must contain a list")
    return [ControlLane.from_dict(item) for item in payload]


def control_lane_svg(lanes: Sequence[ControlLane], *, width: int = 720, lane_height: int = 56) -> str:
    """Return a compact SVG lane view for notebook HTML display."""

    lanes = list(lanes)
    if not lanes:
        return "<div>No control lanes.</div>"
    height = max(1, len(lanes)) * lane_height
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="control lanes">']
    for lane_index, lane in enumerate(lanes):
        y0 = lane_index * lane_height
        values = normalize_control_lane(lane, mode="minmax").values
        if values.size == 1:
            points = f"0,{y0 + lane_height / 2:.1f} {width},{y0 + lane_height / 2:.1f}"
        else:
            points = " ".join(
                f"{x:.1f},{y0 + 10 + (1.0 - float(value)) * (lane_height - 20):.1f}"
                for x, value in zip(np.linspace(0, width, values.size), values)
            )
        parts.append(f'<rect x="0" y="{y0}" width="{width}" height="{lane_height}" fill="#10131a"/>')
        parts.append(f'<text x="8" y="{y0 + 14}" fill="#dce3ee" font-size="11">{_escape_xml(lane.name)}</text>')
        parts.append(f'<polyline points="{points}" fill="none" stroke="#6dd6b0" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


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
    return np.ascontiguousarray(arr, dtype=np.float32)


def _frame_audio(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if audio.shape[0] < frame:
        audio = np.pad(audio, (0, frame - audio.shape[0]))
    count = 1 + max(0, (audio.shape[0] - frame) // hop)
    frames = np.empty((count, frame), dtype=np.float32)
    for index in range(count):
        start = index * hop
        frames[index] = audio[start : start + frame]
    return frames


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
