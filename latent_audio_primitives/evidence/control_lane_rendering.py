"""SVG evidence views for control lanes and latent channel activity."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

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


def control_lane_regions_svg(
    lanes: Sequence[ControlLane],
    regions: Sequence[LaneRegion],
    *,
    width: int = 720,
    lane_height: int = 72,
) -> str:
    """Return stacked lane views with each lane's selected regions highlighted."""

    lanes = list(lanes)
    if not lanes:
        return "<div>No control lanes.</div>"
    regions_by_lane: dict[str, list[LaneRegion]] = {}
    for region in regions:
        regions_by_lane.setdefault(region.lane_name, []).append(region)
    height = max(1, len(lanes)) * lane_height
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="control lane regions">']
    for lane_index, lane in enumerate(lanes):
        y0 = lane_index * lane_height
        values = normalize_control_lane(lane, mode="minmax").values
        parts.append(f'<rect x="0" y="{y0}" width="{width}" height="{lane_height}" fill="#10131a"/>')
        for region in regions_by_lane.get(lane.name, []):
            x = width * (region.start_frame / max(lane.frames - 1, 1))
            w = width * ((region.end_frame - region.start_frame) / max(lane.frames - 1, 1))
            parts.append(
                f'<rect x="{x:.1f}" y="{y0}" width="{max(w, 1.0):.1f}" height="{lane_height}" fill="#ff8eb3" opacity="0.24"/>'
            )
        parts.append(f'<text x="8" y="{y0 + 14}" fill="#dce3ee" font-size="11">{_escape_xml(lane.name)}</text>')
        parts.append(f'<polyline points="{_polyline_points(values, y0, width, lane_height)}" fill="none" stroke="#6dd6b0" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def control_lane_probe_heatmap_svg(
    rows: Sequence[Mapping[str, Any] | Any],
    *,
    x_key: str = "layer_index",
    y_key: str = "lane_name",
    value_key: str = "correlation_mean",
    title: str = "control-lane probe heatmap",
    width: int = 720,
    cell_height: int = 24,
    absolute: bool = True,
) -> str:
    """Return a compact heatmap over lane/probe rows."""

    clean_rows = [row for row in rows if str(_row_get(row, "status", "ok") or "ok") in {"", "ok"}]
    if not clean_rows:
        return "<div>No probe rows.</div>"
    x_values = _sorted_axis_values(_row_get(row, x_key) for row in clean_rows)
    y_values = _sorted_axis_values(_row_get(row, y_key) for row in clean_rows)
    if not x_values or not y_values:
        return "<div>No probe rows.</div>"
    left = 150
    top = 26
    cell_w = max(10.0, (width - left - 8) / max(len(x_values), 1))
    height = top + len(y_values) * cell_height + 8
    values: dict[tuple[Any, Any], float] = {}
    for row in clean_rows:
        x = _row_get(row, x_key)
        y = _row_get(row, y_key)
        if x not in x_values or y not in y_values:
            continue
        value = _float_or_zero(_row_get(row, value_key))
        if absolute:
            value = abs(value)
        key = (x, y)
        values[key] = max(values.get(key, float("-inf")), value)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="{_escape_xml(title)}">']
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#10131a"/>')
    parts.append(f'<text x="8" y="16" fill="#dce3ee" font-size="12">{_escape_xml(title)}</text>')
    for col, x in enumerate(x_values):
        parts.append(f'<text x="{left + col * cell_w + 2:.1f}" y="16" fill="#9fb1c7" font-size="9">{_escape_xml(_axis_label(x))}</text>')
    for row_index, y in enumerate(y_values):
        y0 = top + row_index * cell_height
        parts.append(f'<text x="8" y="{y0 + 16}" fill="#dce3ee" font-size="10">{_escape_xml(_axis_label(y))}</text>')
        for col, x in enumerate(x_values):
            value = values.get((x, y), 0.0)
            color = _heat_color(min(max(float(value), 0.0), 1.0))
            parts.append(f'<rect x="{left + col * cell_w:.1f}" y="{y0}" width="{max(cell_w - 1, 1.0):.1f}" height="{cell_height - 1}" fill="{color}"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def control_lane_prediction_svg(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    width: int = 720,
    lane_height: int = 112,
    max_panels: int = 6,
) -> str:
    """Return actual-vs-predicted lane curves from prediction rows."""

    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        key = (
            str(row.get("lane_name", "")),
            int(row.get("layer_index", -1)),
            str(row.get("window_label", "") or "all"),
        )
        groups.setdefault(key, []).append(row)
    items = list(groups.items())[: max(1, int(max_panels))]
    if not items:
        return "<div>No prediction rows.</div>"
    height = len(items) * lane_height
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="actual vs predicted control lanes">']
    for panel_index, ((lane_name, layer_index, window_label), rows) in enumerate(items):
        y0 = panel_index * lane_height
        rows = sorted(rows, key=lambda row: float(row.get("sample_fraction", 0.0)))
        target = np.asarray([float(row.get("target", 0.0)) for row in rows], dtype=np.float32)
        pred = np.asarray([float(row.get("prediction", 0.0)) for row in rows], dtype=np.float32)
        pair = np.concatenate([target, pred])
        lo = float(pair.min())
        hi = float(pair.max())
        target_norm = (target - lo) / max(hi - lo, 1e-8)
        pred_norm = (pred - lo) / max(hi - lo, 1e-8)
        parts.append(f'<rect x="0" y="{y0}" width="{width}" height="{lane_height}" fill="#10131a"/>')
        title = f"{lane_name} layer {layer_index} {window_label}".strip()
        parts.append(f'<text x="8" y="{y0 + 14}" fill="#dce3ee" font-size="11">{_escape_xml(title)}</text>')
        parts.append(f'<polyline points="{_polyline_points(target_norm, y0, width, lane_height)}" fill="none" stroke="#6dd6b0" stroke-width="2"/>')
        parts.append(f'<polyline points="{_polyline_points(pred_norm, y0, width, lane_height)}" fill="none" stroke="#ff8eb3" stroke-width="2"/>')
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


def _row_get(row: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _sorted_axis_values(values: Sequence[Any]) -> list[Any]:
    clean = [value for value in values if value is not None and value != ""]
    unique = list(dict.fromkeys(clean))
    try:
        return sorted(unique, key=lambda value: float(value))
    except (TypeError, ValueError):
        return sorted(unique, key=lambda value: str(value))


def _axis_label(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2g}"
    return str(value)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
