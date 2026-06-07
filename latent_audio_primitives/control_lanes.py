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
    ``spectral_centroid_hz``.
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
        window = np.hanning(frame).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32) + 1e-10
        freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate).astype(np.float32)
        values = (spectrum * freqs[None, :]).sum(axis=1) / spectrum.sum(axis=1)
        lane_confidence = confidence
    else:
        raise ValueError("name must be 'rms_envelope', 'zero_crossing_rate', or 'spectral_centroid_hz'")
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


def audio_same_control_lanes(
    *,
    audio: Any | None = None,
    sample_rate: int | None = None,
    latent: Any | None = None,
    latent_rate: float | None = None,
    frame_seconds: float = 0.05,
    source: str | None = None,
    normalize: bool = True,
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
                audio_loudness_confidence_lane(audio, sample_rate, frame_seconds=frame_seconds, source=source),
            ]
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


def control_lane_overlay_svg(
    reference: Sequence[ControlLane],
    candidate: Sequence[ControlLane],
    *,
    names: Sequence[str] | None = None,
    width: int = 720,
    lane_height: int = 64,
) -> str:
    """Return an SVG overlay comparing reference and candidate lanes."""

    reference_map = _lane_map(reference)
    candidate_map = _lane_map(candidate)
    resolved_names = list(names) if names is not None else sorted(set(reference_map) & set(candidate_map))
    if not resolved_names:
        return "<div>No matching control lanes.</div>"
    height = len(resolved_names) * lane_height
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="control lane overlay">']
    for lane_index, name in enumerate(resolved_names):
        if name not in reference_map or name not in candidate_map:
            continue
        y0 = lane_index * lane_height
        frames = max(reference_map[name].frames, candidate_map[name].frames)
        ref = resample_control_lane(reference_map[name], frames).values
        cand = resample_control_lane(candidate_map[name], frames).values
        pair = np.concatenate([ref, cand])
        lo = float(pair.min())
        hi = float(pair.max())
        ref_norm = (ref - lo) / max(hi - lo, 1e-8)
        cand_norm = (cand - lo) / max(hi - lo, 1e-8)
        parts.append(f'<rect x="0" y="{y0}" width="{width}" height="{lane_height}" fill="#10131a"/>')
        parts.append(f'<text x="8" y="{y0 + 14}" fill="#dce3ee" font-size="11">{_escape_xml(name)}</text>')
        parts.append(
            f'<polyline points="{_polyline_points(ref_norm, y0, width, lane_height)}" fill="none" stroke="#6dd6b0" stroke-width="2"/>'
        )
        parts.append(
            f'<polyline points="{_polyline_points(cand_norm, y0, width, lane_height)}" fill="none" stroke="#ff8eb3" stroke-width="2"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def control_lane_region_svg(
    lane: ControlLane,
    regions: Sequence[LaneRegion],
    *,
    width: int = 720,
    lane_height: int = 72,
) -> str:
    """Return an SVG lane view with selected regions highlighted."""

    values = normalize_control_lane(lane, mode="minmax").values
    parts = [f'<svg viewBox="0 0 {width} {lane_height}" width="100%" height="{lane_height}" role="img" aria-label="control lane regions">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{lane_height}" fill="#10131a"/>')
    for region in regions:
        x = width * (region.start_frame / max(lane.frames - 1, 1))
        w = width * ((region.end_frame - region.start_frame) / max(lane.frames - 1, 1))
        parts.append(f'<rect x="{x:.1f}" y="0" width="{max(w, 1.0):.1f}" height="{lane_height}" fill="#ff8eb3" opacity="0.28"/>')
    parts.append(f'<text x="8" y="14" fill="#dce3ee" font-size="11">{_escape_xml(lane.name)}</text>')
    parts.append(f'<polyline points="{_polyline_points(values, 0, width, lane_height)}" fill="none" stroke="#6dd6b0" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def latent_channel_heatmap_svg(
    latent: Any,
    *,
    latent_rate: float | None = None,
    channels: Sequence[int] | None = None,
    top_k: int = 16,
    score: str = "motion",
    width: int = 720,
    row_height: int = 14,
    max_columns: int = 180,
) -> str:
    """Return a compact heatmap for selected or top-ranked latent channels."""

    z = as_time_major(latent)
    if channels is None:
        channels = [int(row["channel"]) for row in latent_channel_scores(z, top_k=top_k, score=score)]
    selected = [int(channel) for channel in channels]
    if not selected:
        return "<div>No latent channels selected.</div>"
    columns = min(max_columns, z.shape[0])
    height = len(selected) * row_height + 18
    cell_w = width / max(columns, 1)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="latent channel heatmap">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#10131a"/>')
    if latent_rate is not None:
        parts.append(f'<text x="8" y="12" fill="#dce3ee" font-size="11">latent channel heatmap @ {float(latent_rate):.3f} Hz</text>')
    for row_index, channel in enumerate(selected):
        values = normalize_array(np.abs(z[:, channel]))
        if values.shape[0] != columns:
            values = _resample_values(values, columns)
        y = 18 + row_index * row_height
        parts.append(f'<text x="4" y="{y + 10}" fill="#dce3ee" font-size="10">ch {channel:03d}</text>')
        for col, value in enumerate(values):
            color = _heat_color(float(value))
            parts.append(f'<rect x="{60 + col * cell_w:.1f}" y="{y}" width="{max(cell_w, 1.0):.1f}" height="{row_height - 1}" fill="{color}"/>')
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


def _polyline_points(values: np.ndarray, y0: int, width: int, lane_height: int) -> str:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 1:
        return f"0,{y0 + lane_height / 2:.1f} {width},{y0 + lane_height / 2:.1f}"
    return " ".join(
        f"{x:.1f},{y0 + 16 + (1.0 - float(value)) * (lane_height - 24):.1f}"
        for x, value in zip(np.linspace(0, width, values.size), values)
    )


def _heat_color(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    r = int(30 + 210 * value)
    g = int(48 + 110 * (1.0 - abs(value - 0.55)))
    b = int(72 + 110 * (1.0 - value))
    return f"rgb({r},{g},{b})"


def _optional_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch
