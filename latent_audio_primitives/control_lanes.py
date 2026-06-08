"""Time-varying evidence lanes for audio and SAME latent trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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


@dataclass(frozen=True, slots=True)
class ControlLaneComparisonRow:
    """One matched-lane comparison between a reference and candidate artifact."""

    name: str
    distance: float
    similarity: float
    mean_abs_delta: float
    signed_mean_delta: float
    max_abs_delta: float
    confidence_mean: float
    frames: int
    reference_source: str | None = None
    candidate_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "distance": float(self.distance),
            "similarity": float(self.similarity),
            "mean_abs_delta": float(self.mean_abs_delta),
            "signed_mean_delta": float(self.signed_mean_delta),
            "max_abs_delta": float(self.max_abs_delta),
            "confidence_mean": float(self.confidence_mean),
            "frames": int(self.frames),
            "reference_source": self.reference_source,
            "candidate_source": self.candidate_source,
        }


@dataclass(frozen=True, slots=True)
class LaneRegion:
    """A contiguous selected time region from a control lane."""

    lane_name: str
    start_frame: int
    end_frame: int
    start_seconds: float
    end_seconds: float
    score: float
    label: str = "region"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_name": self.lane_name,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "start_seconds": float(self.start_seconds),
            "end_seconds": float(self.end_seconds),
            "score": float(self.score),
            "label": self.label,
            "metadata": dict(self.metadata),
        }


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
    silence_floor_db: float = -60.0,
    silence_full_db: float = -35.0,
) -> ControlLane:
    """Return a framewise audio control lane.

    Supported names: ``rms_envelope``, ``zero_crossing_rate``,
    ``spectral_centroid_hz``, ``spectral_flux``, ``spectral_bandwidth_hz``,
    ``spectral_entropy``, ``spectral_flatness``, ``spectral_contrast``,
    ``spectral_density_low``, ``spectral_density_mid``,
    ``spectral_density_high``, and ``onset_density``.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    x = _as_audio_array(audio).mean(axis=0)
    frame = max(16, int(round(frame_seconds * sample_rate)))
    hop = max(1, int(round((hop_seconds if hop_seconds is not None else frame_seconds) * sample_rate)))
    frames = _frame_audio(x, frame, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10).astype(np.float32)
    confidence = _rms_to_confidence(rms, floor_db=silence_floor_db, full_db=silence_full_db)
    if name == "rms_envelope":
        values = rms
        lane_confidence = None
    elif name == "zero_crossing_rate":
        values = np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
        lane_confidence = confidence
    elif name == "spectral_centroid_hz":
        spectrum, power, freqs = _spectral_frame_features(frames, frame, sample_rate)
        values = (power * freqs[None, :]).sum(axis=1) / np.maximum(power.sum(axis=1), 1e-10)
        lane_confidence = confidence
    elif name == "spectral_flux":
        spectrum, _power, _freqs = _spectral_frame_features(frames, frame, sample_rate)
        norm = np.linalg.norm(spectrum, axis=1, keepdims=True)
        spectrum = spectrum / np.maximum(norm, 1e-10)
        diff = np.diff(spectrum, axis=0, prepend=spectrum[:1])
        values = np.sqrt(np.mean(diff * diff, axis=1))
        lane_confidence = confidence
    elif name == "spectral_bandwidth_hz":
        _spectrum, power, freqs = _spectral_frame_features(frames, frame, sample_rate)
        total = np.maximum(power.sum(axis=1), 1e-10)
        centroid = (power * freqs[None, :]).sum(axis=1) / total
        values = np.sqrt((power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / total)
        lane_confidence = confidence
    elif name == "spectral_entropy":
        _spectrum, power, _freqs = _spectral_frame_features(frames, frame, sample_rate)
        prob = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-10)
        values = -np.sum(prob * np.log(np.maximum(prob, 1e-10)), axis=1) / np.log(max(prob.shape[1], 2))
        lane_confidence = confidence
    elif name == "spectral_flatness":
        _spectrum, power, _freqs = _spectral_frame_features(frames, frame, sample_rate)
        values = np.exp(np.mean(np.log(np.maximum(power, 1e-10)), axis=1)) / np.maximum(np.mean(power, axis=1), 1e-10)
        lane_confidence = confidence
    elif name == "spectral_contrast":
        spectrum, _power, _freqs = _spectral_frame_features(frames, frame, sample_rate)
        db = 20.0 * np.log10(np.maximum(spectrum, 1e-10))
        values = np.percentile(db, 90.0, axis=1) - np.percentile(db, 10.0, axis=1)
        lane_confidence = confidence
    elif name in {"spectral_density_low", "spectral_density_mid", "spectral_density_high"}:
        _spectrum, power, freqs = _spectral_frame_features(frames, frame, sample_rate)
        nyquist = float(sample_rate) / 2.0
        bands = {
            "spectral_density_low": (20.0, min(250.0, nyquist)),
            "spectral_density_mid": (250.0, min(4000.0, nyquist)),
            "spectral_density_high": (4000.0, nyquist),
        }
        low_hz, high_hz = bands[name]
        values = _spectral_band_fraction(power, freqs, low_hz, high_hz)
        lane_confidence = confidence
    elif name == "onset_density":
        spectrum, _power, _freqs = _spectral_frame_features(frames, frame, sample_rate)
        norm = np.linalg.norm(spectrum, axis=1, keepdims=True)
        spectrum = spectrum / np.maximum(norm, 1e-10)
        positive_flux = np.maximum(np.diff(spectrum, axis=0, prepend=spectrum[:1]), 0.0).sum(axis=1)
        smooth_frames = max(1, int(round(0.25 / (hop / sample_rate))))
        values = _moving_average(normalize_array(positive_flux), smooth_frames)
        lane_confidence = confidence
    else:
        raise ValueError(
            "unsupported audio lane name: "
            f"{name!r}. Use core envelope/spectral lanes or MIR/DSP lane-bank names."
        )
    return ControlLane(
        name=name,
        values=values.astype(np.float32),
        rate_hz=sample_rate / hop,
        confidence=lane_confidence,
        source=source,
        metadata={
            "frame_seconds": float(frame_seconds),
            "hop_seconds": float(hop / sample_rate),
            "silence_floor_db": float(silence_floor_db),
            "silence_full_db": float(silence_full_db),
        },
    )


def audio_loudness_confidence_lane(
    audio: Any,
    sample_rate: int,
    *,
    frame_seconds: float = 0.05,
    hop_seconds: float | None = None,
    floor_db: float = -60.0,
    full_db: float = -35.0,
    name: str = "audio_confidence",
    source: str | None = None,
) -> ControlLane:
    """Return a silence-aware confidence lane from framewise audio RMS."""

    rms_lane = audio_envelope_lane(
        audio,
        sample_rate,
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
        name="rms_envelope",
        source=source,
        silence_floor_db=floor_db,
        silence_full_db=full_db,
    )
    confidence = _rms_to_confidence(rms_lane.values, floor_db=floor_db, full_db=full_db)
    return ControlLane(
        name=name,
        values=confidence,
        rate_hz=rms_lane.rate_hz,
        confidence=confidence,
        source=source,
        metadata={"floor_db": float(floor_db), "full_db": float(full_db)},
    )


def audio_mir_control_lanes(
    audio: Any,
    sample_rate: int,
    *,
    frame_seconds: float = 0.05,
    hop_seconds: float | None = None,
    names: Sequence[str] | None = None,
    source: str | None = None,
    normalize: bool = True,
) -> list[ControlLane]:
    """Return deterministic MIR/DSP lanes from framed audio spectra.

    These are evidence lanes, not semantic labels. They are useful for asking
    whether SAME latents or SA3 residual activations expose measurable spectral
    density, entropy, flatness, contrast, band-energy, or onset structure.
    """

    resolved_names = list(names) if names is not None else [
        "spectral_bandwidth_hz",
        "spectral_entropy",
        "spectral_flatness",
        "spectral_contrast",
        "spectral_density_low",
        "spectral_density_mid",
        "spectral_density_high",
        "onset_density",
    ]
    lanes = [
        audio_envelope_lane(
            audio,
            sample_rate,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            name=name,
            source=source,
        )
        for name in resolved_names
    ]
    return [normalize_control_lane(lane, mode="minmax") for lane in lanes] if normalize else lanes


def audio_same_control_lanes(
    *,
    audio: Any | None = None,
    sample_rate: int | None = None,
    latent: Any | None = None,
    latent_rate: float | None = None,
    frame_seconds: float = 0.05,
    source: str | None = None,
    normalize: bool = True,
    include_mir: bool = False,
    mir_names: Sequence[str] | None = None,
) -> list[ControlLane]:
    """Build the standard notebook lane set from audio and/or SAME latents."""

    lanes: list[ControlLane] = []
    if audio is not None:
        if sample_rate is None:
            raise ValueError("sample_rate is required when audio is provided")
        lanes.extend(
            [
                audio_envelope_lane(audio, sample_rate, frame_seconds=frame_seconds, name="rms_envelope", source=source),
                audio_envelope_lane(audio, sample_rate, frame_seconds=frame_seconds, name="spectral_centroid_hz", source=source),
                audio_envelope_lane(audio, sample_rate, frame_seconds=frame_seconds, name="spectral_flux", source=source),
                audio_loudness_confidence_lane(audio, sample_rate, frame_seconds=frame_seconds, source=source),
            ]
        )
        if include_mir:
            lanes.extend(
                audio_mir_control_lanes(
                    audio,
                    sample_rate,
                    frame_seconds=frame_seconds,
                    names=mir_names,
                    source=source,
                    normalize=False,
                )
            )
    if latent is not None:
        if latent_rate is None:
            raise ValueError("latent_rate is required when latent is provided")
        lanes.extend(
            [
                latent_motion_lane(latent, latent_rate=latent_rate, source=source),
                latent_channel_energy_lane(latent, latent_rate=latent_rate, source=source),
            ]
        )
    if not lanes:
        raise ValueError("audio or latent must be provided")
    if normalize:
        return [normalize_control_lane(lane, mode="minmax") for lane in lanes]
    return lanes


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


def compare_control_lane_sets(
    reference: Sequence[ControlLane],
    candidate: Sequence[ControlLane],
    *,
    names: Sequence[str] | None = None,
    normalize: bool = True,
    use_confidence: bool = True,
) -> list[ControlLaneComparisonRow]:
    """Compare matching lanes between a reference and candidate artifact."""

    reference_map = _lane_map(reference)
    candidate_map = _lane_map(candidate)
    resolved_names = list(names) if names is not None else sorted(set(reference_map) & set(candidate_map))
    rows: list[ControlLaneComparisonRow] = []
    for name in resolved_names:
        if name not in reference_map or name not in candidate_map:
            continue
        a = reference_map[name]
        b = candidate_map[name]
        frames = max(a.frames, b.frames)
        av = resample_control_lane(a, frames).values
        bv = resample_control_lane(b, frames).values
        confidence = _combined_confidence(a, b, frames) if use_confidence else np.ones(frames, dtype=np.float32)
        weights = _safe_weights(confidence)
        if normalize:
            av = _weighted_zscore(av, weights)
            bv = _weighted_zscore(bv, weights)
        delta = bv - av
        distance = float(np.sqrt(np.average(delta * delta, weights=weights)))
        mean_abs_delta = float(np.average(np.abs(delta), weights=weights))
        signed_mean_delta = float(np.average(delta, weights=weights))
        max_abs_delta = float(np.max(np.abs(delta)))
        similarity = _weighted_cosine(av, bv, weights)
        rows.append(
            ControlLaneComparisonRow(
                name=name,
                distance=distance,
                similarity=similarity,
                mean_abs_delta=mean_abs_delta,
                signed_mean_delta=signed_mean_delta,
                max_abs_delta=max_abs_delta,
                confidence_mean=float(np.mean(confidence)),
                frames=frames,
                reference_source=a.source,
                candidate_source=b.source,
            )
        )
    return rows


def control_lane_comparison_table(rows: Sequence[ControlLaneComparisonRow]) -> list[dict[str, Any]]:
    """Return JSON/table-friendly lane-comparison rows."""

    return [row.to_dict() for row in rows]


def control_lane_summary_row(
    lane: ControlLane,
    *,
    active_mask: Sequence[float] | None = None,
    min_active_frames: int = 1,
) -> dict[str, Any]:
    """Return a compact descriptive row for one control lane."""

    values = lane.values.astype(np.float32)
    confidence = lane.confidence if lane.confidence is not None else np.ones(lane.frames, dtype=np.float32)
    normalized = normalize_array(values)
    peak_frame = int(np.argmax(values))
    trough_frame = int(np.argmin(values))
    row: dict[str, Any] = {
        "name": lane.name,
        "source": lane.source,
        "frames": int(lane.frames),
        "duration_seconds": float(lane.duration_seconds),
        "rate_hz": float(lane.rate_hz),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "peak_frame": peak_frame,
        "peak_seconds": float(peak_frame / lane.rate_hz),
        "peak_value": float(values[peak_frame]),
        "trough_frame": trough_frame,
        "trough_seconds": float(trough_frame / lane.rate_hz),
        "trough_value": float(values[trough_frame]),
        "high_fraction": float(np.mean(normalized >= 0.75)),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_min": float(np.min(confidence)),
        "normalization": lane.metadata.get("normalization"),
    }
    if active_mask is not None:
        mask = _resample_values(np.asarray(active_mask, dtype=np.float32), lane.frames) >= 0.5
        active_values = values[mask]
        row["active_frames"] = int(mask.sum())
        row["active_fraction"] = float(mask.mean())
        row["active_duration_seconds"] = float(mask.sum() / lane.rate_hz)
        if active_values.shape[0] >= int(min_active_frames):
            active_peak_frame = int(np.argmax(np.where(mask, values, -np.inf)))
            row.update(
                {
                    "active_mean": float(np.mean(active_values)),
                    "active_std": float(np.std(active_values)),
                    "active_min": float(np.min(active_values)),
                    "active_max": float(np.max(active_values)),
                    "active_peak_frame": active_peak_frame,
                    "active_peak_seconds": float(active_peak_frame / lane.rate_hz),
                    "active_peak_value": float(values[active_peak_frame]),
                }
            )
        else:
            row.update(
                {
                    "active_mean": None,
                    "active_std": None,
                    "active_min": None,
                    "active_max": None,
                    "active_peak_frame": None,
                    "active_peak_seconds": None,
                    "active_peak_value": None,
                }
            )
    return row


def control_lane_summary_table(
    lanes: Sequence[ControlLane],
    *,
    active_mask: Sequence[float] | None = None,
    min_active_frames: int = 1,
) -> list[dict[str, Any]]:
    """Return JSON/table-friendly summaries for a lane set."""

    return [
        control_lane_summary_row(lane, active_mask=active_mask, min_active_frames=min_active_frames)
        for lane in lanes
    ]


def active_source_mask_from_lanes(
    lanes: Sequence[ControlLane],
    *,
    confidence_lane_name: str = "audio_confidence",
    min_confidence: float = 0.05,
    target_frames: int | None = None,
    pad_seconds: float = 0.0,
) -> np.ndarray:
    """Return a source-active mask, preferably from the audio-confidence lane."""

    lane_list = list(lanes)
    if not lane_list:
        raise ValueError("at least one lane is required")
    confidence_lane = _lane_map(lane_list).get(confidence_lane_name)
    base_lane = confidence_lane or lane_list[0]
    frames = int(target_frames) if target_frames is not None else base_lane.frames
    if confidence_lane is None:
        return np.ones(frames, dtype=np.float32)
    confidence = resample_control_lane(confidence_lane, frames).values
    mask = confidence >= float(min_confidence)
    if pad_seconds > 0:
        pad_frames = int(round(float(pad_seconds) * confidence_lane.rate_hz))
        if pad_frames > 0 and np.any(mask):
            expanded = mask.copy()
            active_indices = np.flatnonzero(mask)
            start = max(0, int(active_indices[0]) - pad_frames)
            end = min(frames, int(active_indices[-1]) + pad_frames + 1)
            expanded[start:end] = True
            mask = expanded
    return mask.astype(np.float32)


def active_source_span_from_lanes(
    lanes: Sequence[ControlLane],
    *,
    confidence_lane_name: str = "audio_confidence",
    min_confidence: float = 0.05,
    target_rate_hz: float | None = None,
) -> dict[str, Any]:
    """Return start/end metadata for the active source region."""

    lane_list = list(lanes)
    if not lane_list:
        raise ValueError("at least one lane is required")
    confidence_lane = _lane_map(lane_list).get(confidence_lane_name)
    base_lane = confidence_lane or lane_list[0]
    mask = active_source_mask_from_lanes(
        lane_list,
        confidence_lane_name=confidence_lane_name,
        min_confidence=min_confidence,
        target_frames=base_lane.frames,
    ).astype(bool)
    active = np.flatnonzero(mask)
    if active.size == 0:
        return {
            "confidence_lane": confidence_lane_name if confidence_lane is not None else None,
            "status": "no_active_frames",
            "frames": int(base_lane.frames),
            "rate_hz": float(target_rate_hz or base_lane.rate_hz),
            "start_frame": 0,
            "end_frame": 0,
            "start_seconds": 0.0,
            "end_seconds": 0.0,
            "active_duration_seconds": 0.0,
            "active_fraction": 0.0,
            "min_confidence": float(min_confidence),
        }
    start = int(active[0])
    end = int(active[-1]) + 1
    rate = float(target_rate_hz or base_lane.rate_hz)
    return {
        "confidence_lane": confidence_lane_name if confidence_lane is not None else None,
        "status": "ok" if confidence_lane is not None else "fallback_all_active",
        "frames": int(base_lane.frames),
        "rate_hz": rate,
        "start_frame": start,
        "end_frame": end,
        "start_seconds": float(start / rate),
        "end_seconds": float(end / rate),
        "active_duration_seconds": float((end - start) / rate),
        "active_fraction": float(mask.mean()),
        "min_confidence": float(min_confidence),
    }


def control_lane_correlation_table(
    lanes: Sequence[ControlLane],
    *,
    names: Sequence[str] | None = None,
    active_mask: Sequence[float] | None = None,
    use_confidence: bool = True,
) -> list[dict[str, Any]]:
    """Return pairwise lane correlations, optionally over active source frames."""

    lane_by_name = _lane_map(lanes)
    resolved_names = list(names) if names is not None else [lane.name for lane in lanes]
    resolved = [lane_by_name[name] for name in resolved_names if name in lane_by_name]
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(resolved):
        for b in resolved[i + 1 :]:
            frames = max(a.frames, b.frames)
            av = resample_control_lane(a, frames).values
            bv = resample_control_lane(b, frames).values
            weights = _combined_confidence(a, b, frames) if use_confidence else np.ones(frames, dtype=np.float32)
            if active_mask is not None:
                mask = _resample_values(np.asarray(active_mask, dtype=np.float32), frames)
                weights = weights * (mask >= 0.5)
            positive_weight_frames = int(np.sum(weights > 0))
            if active_mask is not None and positive_weight_frames == 0:
                rows.append(
                    {
                        "lane_a": a.name,
                        "lane_b": b.name,
                        "correlation": 0.0,
                        "similarity": 0.0,
                        "frames": int(frames),
                        "active_frames": 0,
                        "active_only": True,
                        "confidence_weighted": bool(use_confidence),
                        "status": "no_active_frames",
                    }
                )
                continue
            weights = _safe_weights(weights)
            active_frames = positive_weight_frames if active_mask is not None else int(np.sum(weights > 0))
            rows.append(
                {
                    "lane_a": a.name,
                    "lane_b": b.name,
                    "correlation": _weighted_correlation(av, bv, weights),
                    "similarity": _weighted_cosine(_weighted_zscore(av, weights), _weighted_zscore(bv, weights), weights),
                    "frames": int(frames),
                    "active_frames": active_frames,
                    "active_only": active_mask is not None,
                    "confidence_weighted": bool(use_confidence),
                    "status": "ok",
                }
            )
    rows.sort(key=lambda row: abs(float(row["correlation"])), reverse=True)
    return rows


def lane_region_table(regions: Sequence[LaneRegion]) -> list[dict[str, Any]]:
    """Return JSON/table-friendly rows for selected lane regions."""

    rows = []
    for region in regions:
        row = region.to_dict()
        row["duration_seconds"] = float(region.end_seconds - region.start_seconds)
        row["frames"] = int(region.end_frame - region.start_frame)
        rows.append(row)
    return rows


def regions_for_control_lanes(
    lanes: Sequence[ControlLane],
    *,
    names: Sequence[str] | None = None,
    mode: str = "peaks",
    threshold: float | None = None,
    percentile: float | None = None,
    min_duration_seconds: float = 0.0,
    merge_gap_seconds: float = 0.0,
    top_k_per_lane: int | None = None,
    min_confidence: float = 0.0,
    active_mask: Sequence[float] | None = None,
    normalize: bool = True,
) -> list[LaneRegion]:
    """Select regions independently for every requested lane."""

    lane_by_name = _lane_map(lanes)
    resolved_names = list(names) if names is not None else [lane.name for lane in lanes]
    regions: list[LaneRegion] = []
    for name in resolved_names:
        lane = lane_by_name.get(name)
        if lane is None:
            continue
        regions.extend(
            regions_from_control_lane(
                lane,
                mode=mode,
                threshold=threshold,
                percentile=percentile,
                min_duration_seconds=min_duration_seconds,
                merge_gap_seconds=merge_gap_seconds,
                top_k=top_k_per_lane,
                min_confidence=min_confidence,
                active_mask=active_mask,
                normalize=normalize,
            )
        )
    regions.sort(key=lambda region: (region.lane_name, region.start_frame))
    return regions


def compare_lane_region_sets(
    reference_regions: Sequence[LaneRegion],
    candidate_regions: Sequence[LaneRegion],
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
) -> list[dict[str, Any]]:
    """Compare two region sets by nearest temporal overlap/center distance."""

    rows: list[dict[str, Any]] = []
    for ref_index, ref in enumerate(reference_regions):
        best = None
        for cand_index, cand in enumerate(candidate_regions):
            overlap = max(0.0, min(ref.end_seconds, cand.end_seconds) - max(ref.start_seconds, cand.start_seconds))
            union = max(ref.end_seconds, cand.end_seconds) - min(ref.start_seconds, cand.start_seconds)
            ref_center = 0.5 * (ref.start_seconds + ref.end_seconds)
            cand_center = 0.5 * (cand.start_seconds + cand.end_seconds)
            row = {
                "reference_label": reference_label,
                "candidate_label": candidate_label,
                "reference_index": int(ref_index),
                "candidate_index": int(cand_index),
                "reference_lane": ref.lane_name,
                "candidate_lane": cand.lane_name,
                "reference_start_seconds": float(ref.start_seconds),
                "reference_end_seconds": float(ref.end_seconds),
                "candidate_start_seconds": float(cand.start_seconds),
                "candidate_end_seconds": float(cand.end_seconds),
                "overlap_seconds": float(overlap),
                "intersection_over_union": float(overlap / max(union, 1e-8)),
                "center_delta_seconds": float(cand_center - ref_center),
                "abs_center_delta_seconds": float(abs(cand_center - ref_center)),
                "reference_score": float(ref.score),
                "candidate_score": float(cand.score),
            }
            key = (-row["intersection_over_union"], row["abs_center_delta_seconds"])
            if best is None or key < best[0]:
                best = (key, row)
        if best is not None:
            rows.append(best[1])
    rows.sort(key=lambda row: (row["reference_lane"], row["abs_center_delta_seconds"]))
    return rows


def regions_from_control_lane(
    lane: ControlLane,
    *,
    mode: str = "peaks",
    threshold: float | None = None,
    percentile: float | None = None,
    min_duration_seconds: float = 0.0,
    merge_gap_seconds: float = 0.0,
    top_k: int | None = None,
    min_confidence: float = 0.0,
    active_mask: Sequence[float] | None = None,
    normalize: bool = True,
    label: str | None = None,
) -> list[LaneRegion]:
    """Select contiguous time regions from a lane.

    Modes: ``above``, ``below``, ``peaks``, ``stable``, and ``silence``.
    Returned frame spans are end-exclusive.
    """

    mode = mode.lower()
    base_values = normalize_control_lane(lane, mode="minmax").values if normalize else lane.values.astype(np.float32)
    values = base_values.copy()
    confidence = lane.confidence if lane.confidence is not None else np.ones(lane.frames, dtype=np.float32)
    if confidence.shape[0] != lane.frames:
        confidence = resample_control_lane(ControlLane("confidence", confidence, lane.rate_hz), lane.frames).values

    if mode in {"above", "high"}:
        resolved = _resolve_threshold(values, threshold=threshold, percentile=percentile, default=0.5)
        mask = values >= resolved
        score_values = values
    elif mode in {"below", "low"}:
        resolved = _resolve_threshold(values, threshold=threshold, percentile=percentile, default=0.5)
        mask = values <= resolved
        score_values = 1.0 - values
    elif mode in {"peaks", "attacks", "events"}:
        resolved = _resolve_threshold(values, threshold=threshold, percentile=percentile, default_percentile=90.0)
        mask = values >= resolved
        score_values = values
    elif mode in {"stable", "sustain"}:
        motion = np.abs(np.diff(values, prepend=values[:1]))
        resolved = _resolve_threshold(motion, threshold=threshold, percentile=percentile, default_percentile=25.0)
        mask = motion <= resolved
        score_values = 1.0 - normalize_array(motion)
    elif mode in {"silence", "quiet"}:
        resolved = _resolve_threshold(values, threshold=threshold, percentile=percentile, default=0.1)
        mask = values <= resolved
        score_values = 1.0 - values
    else:
        raise ValueError("mode must be 'above', 'below', 'peaks', 'stable', or 'silence'")

    if min_confidence > 0:
        mask = mask & (confidence >= float(min_confidence))
    if active_mask is not None:
        active = _resample_values(np.asarray(active_mask, dtype=np.float32), lane.frames) >= 0.5
        mask = mask & active

    return _regions_from_mask(
        lane,
        mask,
        score_values=score_values,
        min_duration_seconds=min_duration_seconds,
        merge_gap_seconds=merge_gap_seconds,
        top_k=top_k,
        label=label or mode,
        metadata={"mode": mode, "threshold": float(resolved), "normalized": bool(normalize)},
    )


def control_lane_mask(
    lane: ControlLane,
    *,
    target_frames: int | None = None,
    regions: Sequence[LaneRegion] | None = None,
    mode: str = "peaks",
    threshold: float | None = None,
    percentile: float | None = None,
    min_duration_seconds: float = 0.0,
    merge_gap_seconds: float = 0.0,
    invert: bool = False,
) -> np.ndarray:
    """Return a float mask from selected lane regions, optionally resampled."""

    resolved_regions = list(regions) if regions is not None else regions_from_control_lane(
        lane,
        mode=mode,
        threshold=threshold,
        percentile=percentile,
        min_duration_seconds=min_duration_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )
    mask = np.zeros(lane.frames, dtype=np.float32)
    for region in resolved_regions:
        start = max(0, min(lane.frames, int(region.start_frame)))
        end = max(start, min(lane.frames, int(region.end_frame)))
        mask[start:end] = 1.0
    if invert:
        mask = 1.0 - mask
    if target_frames is not None and int(target_frames) != lane.frames:
        mask = _resample_values(mask, int(target_frames))
    return mask.astype(np.float32)


def apply_time_mask_to_latents(
    reference_latents: Any,
    edited_latents: Any,
    time_mask: Sequence[float],
    *,
    amount: float = 1.0,
    time_axis: int = -1,
) -> Any:
    """Blend edited latents into reference latents using a time-domain mask."""

    torch = _optional_torch()
    if torch is not None and (isinstance(reference_latents, torch.Tensor) or isinstance(edited_latents, torch.Tensor)):
        ref = reference_latents if isinstance(reference_latents, torch.Tensor) else torch.as_tensor(reference_latents)
        edit = edited_latents if isinstance(edited_latents, torch.Tensor) else torch.as_tensor(edited_latents, device=ref.device, dtype=ref.dtype)
        edit = edit.to(device=ref.device, dtype=ref.dtype)
        if ref.shape != edit.shape:
            raise ValueError(f"latent shapes must match, got {tuple(ref.shape)} and {tuple(edit.shape)}")
        axis = time_axis if time_axis >= 0 else ref.ndim + time_axis
        mask_np = _resample_values(np.asarray(time_mask, dtype=np.float32), int(ref.shape[axis]))
        mask = torch.as_tensor(mask_np, device=ref.device, dtype=ref.dtype)
        view = [1] * ref.ndim
        view[axis] = int(ref.shape[axis])
        return ref + float(amount) * mask.view(*view) * (edit - ref)

    ref_np = np.asarray(reference_latents, dtype=np.float32)
    edit_np = np.asarray(edited_latents, dtype=np.float32)
    if ref_np.shape != edit_np.shape:
        raise ValueError(f"latent shapes must match, got {ref_np.shape} and {edit_np.shape}")
    axis = time_axis if time_axis >= 0 else ref_np.ndim + time_axis
    mask = _resample_values(np.asarray(time_mask, dtype=np.float32), int(ref_np.shape[axis]))
    shape = [1] * ref_np.ndim
    shape[axis] = int(ref_np.shape[axis])
    return ref_np + float(amount) * mask.reshape(shape) * (edit_np - ref_np)


def latent_channel_scores(
    latent: Any,
    *,
    top_k: int | None = None,
    score: str = "motion",
) -> list[dict[str, float | int]]:
    """Rank SAME latent channels by energy, variance, or motion."""

    z = as_time_major(latent)
    diffs = np.diff(z, axis=0, prepend=z[:1])
    rows: list[dict[str, float | int]] = []
    for channel in range(z.shape[1]):
        values = z[:, channel]
        motion = diffs[:, channel]
        row = {
            "channel": int(channel),
            "rms_energy": float(np.sqrt(np.mean(values * values))),
            "mean_abs": float(np.mean(np.abs(values))),
            "std": float(np.std(values)),
            "motion_energy": float(np.sqrt(np.mean(motion * motion))),
            "peak_abs": float(np.max(np.abs(values))),
        }
        rows.append(row)
    key = {
        "energy": "rms_energy",
        "activity": "mean_abs",
        "variance": "std",
        "std": "std",
        "motion": "motion_energy",
        "peak": "peak_abs",
    }.get(score.lower())
    if key is None:
        raise ValueError("score must be 'energy', 'activity', 'variance', 'std', 'motion', or 'peak'")
    rows.sort(key=lambda row: float(row[key]), reverse=True)
    if top_k is not None:
        rows = rows[: max(1, int(top_k))]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
        row["score"] = float(row[key])
    return rows


def latent_channel_lanes(
    latent: Any,
    *,
    latent_rate: float,
    channels: Sequence[int] | None = None,
    top_k: int = 8,
    score: str = "motion",
    absolute: bool = True,
    normalize: bool = True,
    source: str | None = None,
) -> list[ControlLane]:
    """Return individual latent-channel lanes for selected or top-ranked channels."""

    z = as_time_major(latent)
    if channels is None:
        channels = [int(row["channel"]) for row in latent_channel_scores(z, top_k=top_k, score=score)]
    lanes = []
    for channel in channels:
        channel = int(channel)
        values = z[:, channel]
        if absolute:
            values = np.abs(values)
        lane = ControlLane(
            name=f"latent_ch_{channel:03d}_{'abs' if absolute else 'raw'}",
            values=values.astype(np.float32),
            rate_hz=latent_rate,
            source=source,
            metadata={"channel": channel, "absolute": bool(absolute)},
        )
        lanes.append(normalize_control_lane(lane, mode="minmax") if normalize else lane)
    return lanes


def rank_control_lane_matches(
    reference_lanes: Sequence[ControlLane],
    candidates_by_id: Mapping[str, Sequence[ControlLane]],
    *,
    names: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rank candidate artifacts by control-lane distance to a reference."""

    rows = []
    weights = dict(weights or {})
    for item_id, candidate_lanes in candidates_by_id.items():
        comparisons = compare_control_lane_sets(reference_lanes, candidate_lanes, names=names)
        if not comparisons:
            continue
        total_weight = 0.0
        distance_sum = 0.0
        similarity_sum = 0.0
        for row in comparisons:
            weight = float(weights.get(row.name, 1.0))
            total_weight += weight
            distance_sum += weight * row.distance
            similarity_sum += weight * row.similarity
        rows.append(
            {
                "item_id": str(item_id),
                "lane_distance": float(distance_sum / max(total_weight, 1e-8)),
                "lane_similarity": float(similarity_sum / max(total_weight, 1e-8)),
                "matched_lanes": int(len(comparisons)),
                "comparisons": control_lane_comparison_table(comparisons),
            }
        )
    rows.sort(key=lambda row: float(row["lane_distance"]))
    return rows[:top_k] if top_k is not None else rows


def control_lane_transition_cost(
    source_lanes: Sequence[ControlLane],
    target_lanes: Sequence[ControlLane],
    *,
    seconds: float = 1.0,
    names: Sequence[str] | None = None,
) -> float:
    """Cost of placing target after source using lane boundary windows."""

    source_map = _lane_map(source_lanes)
    target_map = _lane_map(target_lanes)
    resolved_names = list(names) if names is not None else sorted(set(source_map) & set(target_map))
    distances = []
    for name in resolved_names:
        if name not in source_map or name not in target_map:
            continue
        a = _boundary_values(source_map[name], side="end", seconds=seconds)
        b = _boundary_values(target_map[name], side="start", seconds=seconds)
        frames = max(a.shape[0], b.shape[0])
        av = _resample_values(a, frames)
        bv = _resample_values(b, frames)
        distances.append(float(np.sqrt(np.mean((av - bv) ** 2))))
    if not distances:
        return float("inf")
    return float(np.mean(distances))


def rank_control_lane_bridges(
    start_lanes: Sequence[ControlLane],
    end_lanes: Sequence[ControlLane],
    candidates_by_id: Mapping[str, Sequence[ControlLane]],
    *,
    seconds: float = 1.0,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rank candidates by start->candidate plus candidate->end lane continuity."""

    rows = []
    for item_id, lanes in candidates_by_id.items():
        start_cost = control_lane_transition_cost(start_lanes, lanes, seconds=seconds, names=names)
        end_cost = control_lane_transition_cost(lanes, end_lanes, seconds=seconds, names=names)
        rows.append(
            {
                "item_id": str(item_id),
                "lane_bridge_cost": float(start_cost + end_cost),
                "start_to_candidate_cost": float(start_cost),
                "candidate_to_end_cost": float(end_cost),
            }
        )
    rows.sort(key=lambda row: float(row["lane_bridge_cost"]))
    return rows[:top_k] if top_k is not None else rows


def normalize_array(values: Sequence[float], *, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize an arbitrary one-dimensional array."""

    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lo = float(arr.min())
    hi = float(arr.max())
    return ((arr - lo) / max(hi - lo, eps)).astype(np.float32)


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


def _spectral_frame_features(frames: np.ndarray, frame: int, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = np.hanning(frame).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32) + 1e-10
    power = (spectrum * spectrum).astype(np.float32) + 1e-10
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate).astype(np.float32)
    return spectrum, power, freqs


def _spectral_band_fraction(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    mask = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(mask):
        return np.zeros(power.shape[0], dtype=np.float32)
    total = np.maximum(power.sum(axis=1), 1e-10)
    return (power[:, mask].sum(axis=1) / total).astype(np.float32)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    window = max(1, int(window))
    if window <= 1 or values.shape[0] <= 1:
        return values.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _lane_map(lanes: Sequence[ControlLane]) -> dict[str, ControlLane]:
    return {lane.name: lane for lane in lanes}


def _rms_to_confidence(rms: np.ndarray, *, floor_db: float, full_db: float) -> np.ndarray:
    db = 20.0 * np.log10(np.asarray(rms, dtype=np.float32).clip(1e-10))
    return np.clip((db - float(floor_db)) / max(float(full_db) - float(floor_db), 1e-8), 0.0, 1.0).astype(np.float32)


def _combined_confidence(a: ControlLane, b: ControlLane, frames: int) -> np.ndarray:
    confidence = np.ones(frames, dtype=np.float32)
    if a.confidence is not None:
        confidence *= resample_control_lane(ControlLane("confidence", a.confidence, a.rate_hz), frames).values
    if b.confidence is not None:
        confidence *= resample_control_lane(ControlLane("confidence", b.confidence, b.rate_hz), frames).values
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _safe_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float32).reshape(-1)
    if weights.size == 0 or float(weights.sum()) <= 1e-8:
        return np.ones_like(weights, dtype=np.float32)
    return weights


def _weighted_zscore(values: np.ndarray, weights: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if weights is None:
        mean = float(values.mean())
        std = float(values.std())
    else:
        weights = _safe_weights(weights)
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
    return ((values - mean) / max(std, eps)).astype(np.float32)


def _weighted_cosine(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    weights = _safe_weights(weights)
    wa = a * np.sqrt(weights)
    wb = b * np.sqrt(weights)
    denom = float(np.linalg.norm(wa) * np.linalg.norm(wb))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(wa, wb) / denom)


def _weighted_correlation(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:
    weights = _safe_weights(weights).astype(np.float32)
    if float(weights.sum()) <= 1e-8:
        return 0.0
    av = _weighted_zscore(a, weights)
    bv = _weighted_zscore(b, weights)
    return _weighted_cosine(av, bv, weights)


def _resolve_threshold(
    values: np.ndarray,
    *,
    threshold: float | None,
    percentile: float | None,
    default: float | None = None,
    default_percentile: float | None = None,
) -> float:
    if threshold is not None:
        return float(threshold)
    if percentile is not None:
        return float(np.percentile(values, float(percentile)))
    if default_percentile is not None:
        return float(np.percentile(values, float(default_percentile)))
    if default is None:
        raise ValueError("threshold resolution needs a default")
    return float(default)


def _regions_from_mask(
    lane: ControlLane,
    mask: np.ndarray,
    *,
    score_values: np.ndarray,
    min_duration_seconds: float,
    merge_gap_seconds: float,
    top_k: int | None,
    label: str,
    metadata: dict[str, Any],
) -> list[LaneRegion]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    min_frames = max(1, int(round(float(min_duration_seconds) * lane.rate_hz)))
    merge_gap_frames = max(0, int(round(float(merge_gap_seconds) * lane.rate_hz)))
    spans: list[tuple[int, int]] = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, mask.shape[0]))

    if merge_gap_frames and spans:
        merged = [spans[0]]
        for start, end in spans[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= merge_gap_frames:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        spans = merged

    regions: list[LaneRegion] = []
    for start, end in spans:
        if end - start < min_frames:
            continue
        score = float(np.mean(score_values[start:end]))
        regions.append(
            LaneRegion(
                lane_name=lane.name,
                start_frame=int(start),
                end_frame=int(end),
                start_seconds=float(start / lane.rate_hz),
                end_seconds=float(end / lane.rate_hz),
                score=score,
                label=label,
                metadata=dict(metadata),
            )
        )
    regions.sort(key=lambda region: region.score, reverse=True)
    if top_k is not None:
        regions = regions[: max(1, int(top_k))]
    regions.sort(key=lambda region: region.start_frame)
    return regions


def _resample_values(values: np.ndarray, target_frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    target_frames = int(target_frames)
    if target_frames < 1:
        raise ValueError("target_frames must be at least 1")
    if values.shape[0] == target_frames:
        return values.astype(np.float32, copy=True)
    if values.shape[0] == 1:
        return np.full(target_frames, float(values[0]), dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    return np.interp(target_x, source_x, values).astype(np.float32)


def _boundary_values(lane: ControlLane, *, side: str, seconds: float) -> np.ndarray:
    values = normalize_control_lane(lane, mode="minmax").values
    frames = max(1, min(values.shape[0], int(round(float(seconds) * lane.rate_hz))))
    if side == "start":
        return values[:frames]
    if side == "end":
        return values[-frames:]
    raise ValueError("side must be 'start' or 'end'")


def _optional_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch
