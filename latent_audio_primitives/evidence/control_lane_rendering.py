"""SVG evidence views for control lanes and latent channel activity."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from latent_audio_primitives.control_lanes import (
    ControlLane,
    LaneRegion,
    latent_channel_scores,
    normalize_array,
    normalize_control_lane,
    resample_control_lane,
)
from latent_audio_primitives.latent_math import as_time_major


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


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _lane_map(lanes: Sequence[ControlLane]) -> dict[str, ControlLane]:
    return {lane.name: lane for lane in lanes}


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
