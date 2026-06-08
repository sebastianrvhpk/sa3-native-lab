"""Mechanistic probes from SA3 residual activations to control lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .control_lanes import ControlLane, normalize_array, resample_control_lane


@dataclass(frozen=True, slots=True)
class ControlLaneProbeRow:
    """One held-out linear probe row for a lane/layer/window cell."""

    lane_name: str
    layer_index: int
    method: str
    alignment_mode: str
    correlation_mean: float
    correlation_std: float
    normalized_mse_mean: float
    normalized_mse_std: float
    r2_mean: float
    r2_std: float
    fold_count: int
    sample_count: int
    feature_count: int
    lane_frames: int
    activation_call_count: int
    active_count: int
    quiet_count: int
    active_delta_norm: float
    ridge_active_cosine: float
    active_threshold: float
    quiet_threshold: float
    call_group_count: int = 0
    call_heldout_correlation_mean: float = 0.0
    call_heldout_correlation_std: float = 0.0
    call_heldout_normalized_mse_mean: float = 0.0
    call_heldout_normalized_mse_std: float = 0.0
    call_heldout_r2_mean: float = 0.0
    call_heldout_r2_std: float = 0.0
    call_heldout_fold_count: int = 0
    call_heldout_status: str = ""
    rank: int = 0
    status: str = "ok"
    error: str = ""
    window_index: int | None = None
    window_label: str = ""
    window_start_fraction: float | None = None
    window_end_fraction: float | None = None
    call_start: int | None = None
    call_end: int | None = None
    window_count: int | None = None
    step_index: int | None = None
    sampler_index: int | None = None
    timestep: float | None = None
    sigma: float | None = None
    logsnr: float | None = None
    sampler_type: str = ""
    calls_per_step: int | None = None
    mapping_status: str = ""
    null_kind: str = ""
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_name": self.lane_name,
            "layer_index": int(self.layer_index),
            "window_index": None if self.window_index is None else int(self.window_index),
            "window_label": self.window_label,
            "window_start_fraction": (
                None if self.window_start_fraction is None else float(self.window_start_fraction)
            ),
            "window_end_fraction": None if self.window_end_fraction is None else float(self.window_end_fraction),
            "call_start": None if self.call_start is None else int(self.call_start),
            "call_end": None if self.call_end is None else int(self.call_end),
            "window_count": None if self.window_count is None else int(self.window_count),
            "step_index": None if self.step_index is None else int(self.step_index),
            "sampler_index": None if self.sampler_index is None else int(self.sampler_index),
            "timestep": None if self.timestep is None else float(self.timestep),
            "sigma": None if self.sigma is None else float(self.sigma),
            "logsnr": None if self.logsnr is None else float(self.logsnr),
            "sampler_type": self.sampler_type,
            "calls_per_step": None if self.calls_per_step is None else int(self.calls_per_step),
            "mapping_status": self.mapping_status,
            "null_kind": self.null_kind,
            "method": self.method,
            "alignment_mode": self.alignment_mode,
            "correlation_mean": float(self.correlation_mean),
            "correlation_std": float(self.correlation_std),
            "normalized_mse_mean": float(self.normalized_mse_mean),
            "normalized_mse_std": float(self.normalized_mse_std),
            "r2_mean": float(self.r2_mean),
            "r2_std": float(self.r2_std),
            "fold_count": int(self.fold_count),
            "sample_count": int(self.sample_count),
            "feature_count": int(self.feature_count),
            "lane_frames": int(self.lane_frames),
            "activation_call_count": int(self.activation_call_count),
            "active_count": int(self.active_count),
            "quiet_count": int(self.quiet_count),
            "active_delta_norm": float(self.active_delta_norm),
            "ridge_active_cosine": float(self.ridge_active_cosine),
            "active_threshold": float(self.active_threshold),
            "quiet_threshold": float(self.quiet_threshold),
            "call_group_count": int(self.call_group_count),
            "call_heldout_correlation_mean": float(self.call_heldout_correlation_mean),
            "call_heldout_correlation_std": float(self.call_heldout_correlation_std),
            "call_heldout_normalized_mse_mean": float(self.call_heldout_normalized_mse_mean),
            "call_heldout_normalized_mse_std": float(self.call_heldout_normalized_mse_std),
            "call_heldout_r2_mean": float(self.call_heldout_r2_mean),
            "call_heldout_r2_std": float(self.call_heldout_r2_std),
            "call_heldout_fold_count": int(self.call_heldout_fold_count),
            "call_heldout_status": self.call_heldout_status,
            "rank": int(self.rank),
            "status": self.status,
            "error": self.error,
            "source": self.source,
        }


def control_lane_layer_probe_rows(
    raw_activations: Mapping[int, Sequence[Any] | Any],
    lanes: Sequence[ControlLane],
    *,
    lane_names: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    min_samples: int = 16,
    min_confidence: float = 0.0,
    active_percentile: float = 85.0,
    quiet_percentile: float = 15.0,
) -> list[ControlLaneProbeRow]:
    """Rank layers by how well residual token activations predict each lane.

    Each captured forward call is treated as a sequence of audio/time tokens
    with feature dimension last. Lane values are resampled to each call's token
    count, then repeated across calls. This makes the row a mechanistic
    microscope over token-aligned residual content, not exact sampler-timestep
    attribution.
    """

    selected_lanes = _select_lanes(lanes, lane_names)
    selected_layers = _select_layers(raw_activations, layer_indices)
    rows: list[ControlLaneProbeRow] = []
    for lane in selected_lanes:
        for layer_idx in selected_layers:
            calls = _activation_calls(raw_activations[layer_idx])
            rows.append(
                _probe_calls_for_lane(
                    calls,
                    lane,
                    layer_index=int(layer_idx),
                    alignment_mode="token_time_repeated_over_observed_calls",
                    cv_folds=cv_folds,
                    ridge_alpha=ridge_alpha,
                    min_samples=min_samples,
                    min_confidence=min_confidence,
                    active_percentile=active_percentile,
                    quiet_percentile=quiet_percentile,
                )
            )
    return _rank_rows_within_lane(rows)


def control_lane_window_probe_rows(
    raw_activations: Mapping[int, Sequence[Any] | Any],
    lanes: Sequence[ControlLane],
    *,
    lane_names: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
    window_count: int | None = 5,
    window_size: int | None = None,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    min_samples: int = 16,
    min_confidence: float = 0.0,
    active_percentile: float = 85.0,
    quiet_percentile: float = 15.0,
) -> list[ControlLaneProbeRow]:
    """Rank observed layer/call-window cells by lane predictability."""

    selected_lanes = _select_lanes(lanes, lane_names)
    selected_layers = _select_layers(raw_activations, layer_indices)
    rows: list[ControlLaneProbeRow] = []
    for lane in selected_lanes:
        for layer_idx in selected_layers:
            calls = _activation_calls(raw_activations[layer_idx])
            windows = control_lane_probe_call_windows(
                len(calls),
                window_count=window_count,
                window_size=window_size,
            )
            for window_index, (start, end) in enumerate(windows):
                rows.append(
                    _probe_calls_for_lane(
                        calls[start:end],
                        lane,
                        layer_index=int(layer_idx),
                        alignment_mode="token_time_within_observed_call_window",
                        cv_folds=cv_folds,
                        ridge_alpha=ridge_alpha,
                        min_samples=min_samples,
                        min_confidence=min_confidence,
                        active_percentile=active_percentile,
                        quiet_percentile=quiet_percentile,
                        window_index=window_index,
                        window_label=_window_label(window_index, len(windows), start, end),
                        call_start=start,
                        call_end=end,
                        window_count=len(windows),
                        total_call_count=len(calls),
                    )
                )
    return _rank_rows_within_lane(rows)


def control_lane_timestep_probe_rows(
    layer_timestep_activations: Mapping[int, Mapping[int, Any]] | Mapping[tuple[int, int], Any],
    lanes: Sequence[ControlLane],
    *,
    layer_timestep_metadata: Mapping[int, Mapping[int, Mapping[str, Any]]] | Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    lane_names: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    min_samples: int = 16,
    min_confidence: float = 0.0,
    active_percentile: float = 85.0,
    quiet_percentile: float = 15.0,
) -> list[ControlLaneProbeRow]:
    """Rank layer/sampler-step cells by lane predictability.

    These rows are only exact timestep attribution when the sampler callback and
    residual hook calls map cleanly. The ``mapping_status`` field reports that
    contract for every row.
    """

    selected_lanes = _select_lanes(lanes, lane_names)
    selected_layers = _select_timestep_layers(layer_timestep_activations, layer_indices)
    rows: list[ControlLaneProbeRow] = []
    for lane in selected_lanes:
        for layer_idx in selected_layers:
            for step_idx, activation, metadata in _iter_layer_timestep_activations(
                layer_timestep_activations,
                layer_idx,
                layer_timestep_metadata,
            ):
                rows.append(
                    _probe_calls_for_lane(
                        [activation],
                        lane,
                        layer_index=int(layer_idx),
                        alignment_mode="token_time_within_sampler_timestep",
                        cv_folds=cv_folds,
                        ridge_alpha=ridge_alpha,
                        min_samples=min_samples,
                        min_confidence=min_confidence,
                        active_percentile=active_percentile,
                        quiet_percentile=quiet_percentile,
                        step_metadata={**metadata, "step_index": step_idx},
                    )
                )
    return _rank_rows_within_lane(rows)


def control_lane_null_layer_probe_rows(
    raw_activations: Mapping[int, Sequence[Any] | Any],
    lanes: Sequence[ControlLane],
    *,
    null_kinds: Sequence[str] = ("shuffle", "reverse", "random"),
    seed: int = 0,
    lane_names: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    min_samples: int = 16,
    min_confidence: float = 0.0,
    active_percentile: float = 85.0,
    quiet_percentile: float = 15.0,
) -> list[ControlLaneProbeRow]:
    """Run layer probes against shuffled/reversed/random lane controls."""

    rng = np.random.default_rng(int(seed))
    selected_lanes = _select_lanes(lanes, lane_names)
    selected_layers = _select_layers(raw_activations, layer_indices)
    rows: list[ControlLaneProbeRow] = []
    for null_kind in null_kinds:
        for lane in selected_lanes:
            null_lane = _null_control_lane(lane, null_kind, rng)
            for layer_idx in selected_layers:
                calls = _activation_calls(raw_activations[layer_idx])
                rows.append(
                    _probe_calls_for_lane(
                        calls,
                        null_lane,
                        layer_index=int(layer_idx),
                        alignment_mode="token_time_repeated_over_observed_calls",
                        cv_folds=cv_folds,
                        ridge_alpha=ridge_alpha,
                        min_samples=min_samples,
                        min_confidence=min_confidence,
                        active_percentile=active_percentile,
                        quiet_percentile=quiet_percentile,
                        null_kind=null_kind,
                    )
                )
    return _rank_rows_within_lane(rows)


def control_lane_null_window_probe_rows(
    raw_activations: Mapping[int, Sequence[Any] | Any],
    lanes: Sequence[ControlLane],
    *,
    null_kinds: Sequence[str] = ("shuffle", "reverse", "random"),
    seed: int = 0,
    lane_names: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
    window_count: int | None = 5,
    window_size: int | None = None,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    min_samples: int = 16,
    min_confidence: float = 0.0,
    active_percentile: float = 85.0,
    quiet_percentile: float = 15.0,
) -> list[ControlLaneProbeRow]:
    """Run observed-call window probes against lane-null controls."""

    rng = np.random.default_rng(int(seed))
    selected_lanes = _select_lanes(lanes, lane_names)
    selected_layers = _select_layers(raw_activations, layer_indices)
    rows: list[ControlLaneProbeRow] = []
    for null_kind in null_kinds:
        for lane in selected_lanes:
            null_lane = _null_control_lane(lane, null_kind, rng)
            for layer_idx in selected_layers:
                calls = _activation_calls(raw_activations[layer_idx])
                windows = control_lane_probe_call_windows(
                    len(calls),
                    window_count=window_count,
                    window_size=window_size,
                )
                for window_index, (start, end) in enumerate(windows):
                    rows.append(
                        _probe_calls_for_lane(
                            calls[start:end],
                            null_lane,
                            layer_index=int(layer_idx),
                            alignment_mode="token_time_within_observed_call_window",
                            cv_folds=cv_folds,
                            ridge_alpha=ridge_alpha,
                            min_samples=min_samples,
                            min_confidence=min_confidence,
                            active_percentile=active_percentile,
                            quiet_percentile=quiet_percentile,
                            window_index=window_index,
                            window_label=_window_label(window_index, len(windows), start, end),
                            call_start=start,
                            call_end=end,
                            window_count=len(windows),
                            total_call_count=len(calls),
                            null_kind=null_kind,
                        )
                    )
    return _rank_rows_within_lane(rows)


def control_lane_null_timestep_probe_rows(
    layer_timestep_activations: Mapping[int, Mapping[int, Any]] | Mapping[tuple[int, int], Any],
    lanes: Sequence[ControlLane],
    *,
    layer_timestep_metadata: Mapping[int, Mapping[int, Mapping[str, Any]]] | Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    null_kinds: Sequence[str] = ("shuffle", "reverse", "random"),
    seed: int = 0,
    lane_names: Sequence[str] | None = None,
    layer_indices: Sequence[int] | None = None,
    cv_folds: int = 5,
    ridge_alpha: float = 1.0,
    min_samples: int = 16,
    min_confidence: float = 0.0,
    active_percentile: float = 85.0,
    quiet_percentile: float = 15.0,
) -> list[ControlLaneProbeRow]:
    """Run sampler-step probes against lane-null controls."""

    rng = np.random.default_rng(int(seed))
    selected_lanes = _select_lanes(lanes, lane_names)
    selected_layers = _select_timestep_layers(layer_timestep_activations, layer_indices)
    rows: list[ControlLaneProbeRow] = []
    for null_kind in null_kinds:
        for lane in selected_lanes:
            null_lane = _null_control_lane(lane, null_kind, rng)
            for layer_idx in selected_layers:
                for step_idx, activation, metadata in _iter_layer_timestep_activations(
                    layer_timestep_activations,
                    layer_idx,
                    layer_timestep_metadata,
                ):
                    rows.append(
                        _probe_calls_for_lane(
                            [activation],
                            null_lane,
                            layer_index=int(layer_idx),
                            alignment_mode="token_time_within_sampler_timestep",
                            cv_folds=cv_folds,
                            ridge_alpha=ridge_alpha,
                            min_samples=min_samples,
                            min_confidence=min_confidence,
                            active_percentile=active_percentile,
                            quiet_percentile=quiet_percentile,
                            step_metadata={**metadata, "step_index": step_idx},
                            null_kind=null_kind,
                        )
                    )
    return _rank_rows_within_lane(rows)


def control_lane_probe_table(rows: Sequence[ControlLaneProbeRow]) -> list[dict[str, Any]]:
    """Return JSON/table-friendly control-lane probe rows."""

    return [row.to_dict() for row in rows]


def control_lane_null_margin_table(
    true_rows: Sequence[ControlLaneProbeRow | Mapping[str, Any]],
    null_rows: Sequence[ControlLaneProbeRow | Mapping[str, Any]],
    *,
    score_key: str = "correlation_mean",
    heldout_score_key: str = "call_heldout_correlation_mean",
) -> list[dict[str, Any]]:
    """Compare true-lane probe rows against matched shuffled/reversed/random rows."""

    null_by_key: dict[tuple[Any, ...], list[ControlLaneProbeRow | Mapping[str, Any]]] = {}
    for row in null_rows:
        if str(_row_get(row, "status", "ok") or "ok") not in {"", "ok"}:
            continue
        key = _probe_match_key(row)
        null_by_key.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for true_row in true_rows:
        if str(_row_get(true_row, "status", "ok") or "ok") not in {"", "ok"}:
            continue
        if str(_row_get(true_row, "null_kind", "") or ""):
            continue
        key = _probe_match_key(true_row)
        matched = null_by_key.get(key, [])
        true_score = float(_row_get(true_row, score_key, 0.0) or 0.0)
        true_heldout_status = str(_row_get(true_row, "call_heldout_status", "") or "")
        true_heldout_score = float(_row_get(true_row, heldout_score_key, 0.0) or 0.0)
        null_scores = [float(_row_get(row, score_key, 0.0) or 0.0) for row in matched]
        null_heldout_scores = [
            float(_row_get(row, heldout_score_key, 0.0) or 0.0)
            for row in matched
            if str(_row_get(row, "call_heldout_status", "") or "") == "ok"
        ]
        if null_scores:
            max_null = float(max(null_scores))
            mean_null = float(np.mean(null_scores))
            margin = float(true_score - max_null)
            status = "ok"
        else:
            max_null = 0.0
            mean_null = 0.0
            margin = 0.0
            status = "no_null_rows"

        if true_heldout_status == "ok" and null_heldout_scores:
            max_null_heldout: float | None = float(max(null_heldout_scores))
            mean_null_heldout: float | None = float(np.mean(null_heldout_scores))
            heldout_margin: float | None = float(true_heldout_score - max_null_heldout)
            beats_null_heldout: bool | None = bool(heldout_margin > 0.0)
        else:
            max_null_heldout = None
            mean_null_heldout = None
            heldout_margin = None
            beats_null_heldout = None

        out.append(
            {
                "lane_name": str(_row_get(true_row, "lane_name", "")),
                "layer_index": _optional_int(_row_get(true_row, "layer_index")),
                "alignment_mode": str(_row_get(true_row, "alignment_mode", "")),
                "window_index": _optional_int(_row_get(true_row, "window_index")),
                "window_label": str(_row_get(true_row, "window_label", "") or ""),
                "step_index": _optional_int(_row_get(true_row, "step_index")),
                "sampler_index": _optional_int(_row_get(true_row, "sampler_index")),
                "timestep": _optional_float(_row_get(true_row, "timestep")),
                "sigma": _optional_float(_row_get(true_row, "sigma")),
                "logsnr": _optional_float(_row_get(true_row, "logsnr")),
                "mapping_status": str(_row_get(true_row, "mapping_status", "") or ""),
                "score_key": score_key,
                "true_score": true_score,
                "max_null_score": max_null,
                "mean_null_score": mean_null,
                "null_margin": margin,
                "beats_null": bool(null_scores and margin > 0.0),
                "null_count": int(len(matched)),
                "null_kinds": _matched_null_kinds(matched),
                "call_heldout_score_key": heldout_score_key,
                "true_call_heldout_status": true_heldout_status,
                "true_call_heldout_score": true_heldout_score if true_heldout_status == "ok" else None,
                "max_null_call_heldout_score": max_null_heldout,
                "mean_null_call_heldout_score": mean_null_heldout,
                "call_heldout_null_margin": heldout_margin,
                "beats_null_call_heldout": beats_null_heldout,
                "rank": int(_row_get(true_row, "rank", 0) or 0),
                "status": status,
            }
        )
    out.sort(
        key=lambda row: (
            row["status"] != "ok",
            -float(row["null_margin"]),
            -float(row["true_score"]),
            str(row["lane_name"]),
        )
    )
    for rank, row in enumerate(out, start=1):
        row["null_margin_rank"] = int(rank)
    return out


def control_lane_probe_prediction_table(
    raw_activations: Mapping[int, Sequence[Any] | Any],
    lanes: Sequence[ControlLane],
    probe_rows: Sequence[ControlLaneProbeRow | Mapping[str, Any]],
    *,
    top_k_per_lane: int = 1,
    max_points_per_row: int = 240,
    ridge_alpha: float = 1.0,
    cv_folds: int = 5,
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Return held-out actual-vs-predicted samples for top probe rows."""

    lane_by_name = {lane.name: lane for lane in lanes}
    selected_rows = _select_top_probe_rows(probe_rows, top_k_per_lane=top_k_per_lane)
    out: list[dict[str, Any]] = []
    for row in selected_rows:
        lane_name = str(_row_get(row, "lane_name"))
        lane = lane_by_name.get(lane_name)
        layer_idx = _optional_int(_row_get(row, "layer_index"))
        if lane is None or layer_idx is None or layer_idx not in raw_activations:
            continue
        calls = _calls_for_probe_row(raw_activations[layer_idx], row)
        try:
            x, y = _features_and_targets_for_calls(calls, lane, min_confidence=min_confidence)
        except Exception:
            continue
        predictions = _ridge_blocked_cv_predictions(
            x,
            y,
            cv_folds=cv_folds,
            ridge_alpha=ridge_alpha,
        )
        if not predictions:
            continue
        sample_indices = np.arange(len(predictions), dtype=np.int64)
        if max_points_per_row is not None and len(sample_indices) > int(max_points_per_row):
            sample_indices = np.linspace(0, len(predictions) - 1, int(max_points_per_row), dtype=np.int64)
        for point_index in sample_indices:
            pred = predictions[int(point_index)]
            out.append(
                {
                    "lane_name": lane_name,
                    "layer_index": int(layer_idx),
                    "window_index": _optional_int(_row_get(row, "window_index")),
                    "window_label": str(_row_get(row, "window_label", "") or ""),
                    "rank": int(_row_get(row, "rank", 0) or 0),
                    "sample_index": int(point_index),
                    "sample_fraction": float(point_index / max(len(predictions) - 1, 1)),
                    "target": float(pred["target"]),
                    "prediction": float(pred["prediction"]),
                    "residual": float(pred["prediction"] - pred["target"]),
                    "fold_index": int(pred["fold_index"]),
                    "source": _row_get(row, "source"),
                }
            )
    return out


def control_lane_active_direction_table(
    raw_activations: Mapping[int, Sequence[Any] | Any],
    lanes: Sequence[ControlLane],
    probe_rows: Sequence[ControlLaneProbeRow | Mapping[str, Any]],
    *,
    top_k_per_lane: int = 1,
    top_features: int = 16,
    ridge_alpha: float = 1.0,
    min_confidence: float = 0.0,
    active_percentile: float = 85.0,
    quiet_percentile: float = 15.0,
) -> list[dict[str, Any]]:
    """Return compact active-vs-quiet residual direction previews."""

    lane_by_name = {lane.name: lane for lane in lanes}
    rows = _select_top_probe_rows(probe_rows, top_k_per_lane=top_k_per_lane)
    out: list[dict[str, Any]] = []
    for row in rows:
        lane_name = str(_row_get(row, "lane_name"))
        lane = lane_by_name.get(lane_name)
        layer_idx = _optional_int(_row_get(row, "layer_index"))
        if lane is None or layer_idx is None or layer_idx not in raw_activations:
            continue
        calls = _calls_for_probe_row(raw_activations[layer_idx], row)
        try:
            x, y = _features_and_targets_for_calls(calls, lane, min_confidence=min_confidence)
        except Exception:
            continue
        preview = _active_quiet_direction_preview(
            x,
            y,
            ridge_alpha=ridge_alpha,
            active_percentile=active_percentile,
            quiet_percentile=quiet_percentile,
            top_features=top_features,
        )
        out.append(
            {
                "lane_name": lane_name,
                "layer_index": int(layer_idx),
                "window_index": _optional_int(_row_get(row, "window_index")),
                "window_label": str(_row_get(row, "window_label", "") or ""),
                "rank": int(_row_get(row, "rank", 0) or 0),
                "feature_count": int(x.shape[1]),
                **preview,
            }
        )
    return out


def control_lane_probe_top_table(
    rows: Sequence[ControlLaneProbeRow | Mapping[str, Any]],
    *,
    top_k_per_lane: int = 1,
    include_null: bool = False,
) -> list[dict[str, Any]]:
    """Return compact top rows per lane for repeatability ledgers."""

    return [
        _row_to_dict(row)
        for row in _select_top_probe_rows(rows, top_k_per_lane=top_k_per_lane, include_null=include_null)
    ]


def control_lane_probe_call_windows(
    call_count: int,
    *,
    window_count: int | None = 5,
    window_size: int | None = None,
) -> list[tuple[int, int]]:
    """Return non-empty observed-call windows for lane probing."""

    if call_count <= 0:
        raise ValueError("cannot window an empty activation trace")
    if window_size is not None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        return [(start, min(start + window_size, call_count)) for start in range(0, call_count, window_size)]
    if window_count is None:
        window_count = 1
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    count = min(int(window_count), call_count)
    return [
        (int(index * call_count / count), int((index + 1) * call_count / count))
        for index in range(count)
        if int(index * call_count / count) < int((index + 1) * call_count / count)
    ]


def _probe_calls_for_lane(
    calls: Sequence[Any],
    lane: ControlLane,
    *,
    layer_index: int,
    alignment_mode: str,
    cv_folds: int,
    ridge_alpha: float,
    min_samples: int,
    min_confidence: float,
    active_percentile: float,
    quiet_percentile: float,
    window_index: int | None = None,
    window_label: str = "",
    call_start: int | None = None,
    call_end: int | None = None,
    window_count: int | None = None,
    total_call_count: int | None = None,
    step_metadata: Mapping[str, Any] | None = None,
    null_kind: str = "",
) -> ControlLaneProbeRow:
    step_metadata = dict(step_metadata or {})
    try:
        x, y, groups = _features_targets_groups_for_calls(calls, lane, min_confidence=min_confidence)
    except Exception as exc:
        return _empty_probe_row(
            lane,
            layer_index=layer_index,
            alignment_mode=alignment_mode,
            method="ridge_blocked_cv",
            status="alignment_error",
            error=str(exc),
            window_index=window_index,
            window_label=window_label,
            call_start=call_start,
            call_end=call_end,
            window_count=window_count,
            activation_call_count=total_call_count if total_call_count is not None else len(calls),
            step_metadata=step_metadata,
            null_kind=null_kind,
        )
    if x.shape[0] < min_samples or x.shape[1] < 1:
        return _empty_probe_row(
            lane,
            layer_index=layer_index,
            alignment_mode=alignment_mode,
            method="ridge_blocked_cv",
            status="insufficient",
            error=f"need at least {min_samples} samples and one feature",
            window_index=window_index,
            window_label=window_label,
            call_start=call_start,
            call_end=call_end,
            window_count=window_count,
            sample_count=x.shape[0],
            feature_count=x.shape[1],
            activation_call_count=total_call_count if total_call_count is not None else len(calls),
            step_metadata=step_metadata,
            null_kind=null_kind,
        )
    metrics = _ridge_blocked_cv(
        x,
        y,
        cv_folds=cv_folds,
        ridge_alpha=ridge_alpha,
    )
    call_metrics = _ridge_call_heldout_cv(
        x,
        y,
        groups,
        cv_folds=cv_folds,
        ridge_alpha=ridge_alpha,
    )
    contrast = _active_quiet_contrast(
        x,
        y,
        ridge_alpha=ridge_alpha,
        active_percentile=active_percentile,
        quiet_percentile=quiet_percentile,
    )
    return ControlLaneProbeRow(
        lane_name=lane.name,
        layer_index=layer_index,
        method="ridge_blocked_cv",
        alignment_mode=alignment_mode,
        correlation_mean=metrics["correlation_mean"],
        correlation_std=metrics["correlation_std"],
        normalized_mse_mean=metrics["normalized_mse_mean"],
        normalized_mse_std=metrics["normalized_mse_std"],
        r2_mean=metrics["r2_mean"],
        r2_std=metrics["r2_std"],
        fold_count=metrics["fold_count"],
        sample_count=int(x.shape[0]),
        feature_count=int(x.shape[1]),
        lane_frames=lane.frames,
        activation_call_count=total_call_count if total_call_count is not None else len(calls),
        active_count=contrast["active_count"],
        quiet_count=contrast["quiet_count"],
        active_delta_norm=contrast["active_delta_norm"],
        ridge_active_cosine=contrast["ridge_active_cosine"],
        active_threshold=contrast["active_threshold"],
        quiet_threshold=contrast["quiet_threshold"],
        call_group_count=call_metrics["call_group_count"],
        call_heldout_correlation_mean=call_metrics["correlation_mean"],
        call_heldout_correlation_std=call_metrics["correlation_std"],
        call_heldout_normalized_mse_mean=call_metrics["normalized_mse_mean"],
        call_heldout_normalized_mse_std=call_metrics["normalized_mse_std"],
        call_heldout_r2_mean=call_metrics["r2_mean"],
        call_heldout_r2_std=call_metrics["r2_std"],
        call_heldout_fold_count=call_metrics["fold_count"],
        call_heldout_status=str(call_metrics["status"]),
        window_index=window_index,
        window_label=window_label,
        window_start_fraction=None if call_start is None else call_start / max(total_call_count or len(calls), 1),
        window_end_fraction=None if call_end is None else call_end / max(total_call_count or len(calls), 1),
        call_start=call_start,
        call_end=call_end,
        window_count=window_count,
        step_index=_optional_int(step_metadata.get("step_index")),
        sampler_index=_optional_int(step_metadata.get("sampler_index")),
        timestep=_optional_float(step_metadata.get("timestep")),
        sigma=_optional_float(step_metadata.get("sigma")),
        logsnr=_optional_float(step_metadata.get("logsnr")),
        sampler_type=str(step_metadata.get("sampler_type", "") or ""),
        calls_per_step=_optional_int(step_metadata.get("calls_per_step")),
        mapping_status=str(step_metadata.get("mapping_status", "") or ""),
        null_kind=str(null_kind or ""),
        source=lane.source,
    )


def _features_and_targets_for_calls(
    calls: Sequence[Any],
    lane: ControlLane,
    *,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y, _groups = _features_targets_groups_for_calls(calls, lane, min_confidence=min_confidence)
    return x, y


def _features_targets_groups_for_calls(
    calls: Sequence[Any],
    lane: ControlLane,
    *,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    groups = []
    for call_index, activation in enumerate(calls):
        x = _token_feature_matrix(activation)
        y_lane = resample_control_lane(lane, x.shape[0]).values.astype(np.float32)
        keep = np.ones(x.shape[0], dtype=bool)
        if lane.confidence is not None and min_confidence > 0:
            confidence = resample_control_lane(
                ControlLane("confidence", lane.confidence, lane.rate_hz),
                x.shape[0],
            ).values
            keep &= confidence >= float(min_confidence)
        if np.any(keep):
            features.append(x[keep])
            targets.append(y_lane[keep])
            groups.append(np.full(int(np.sum(keep)), int(call_index), dtype=np.int64))
    if not features:
        raise ValueError("no activation/lane samples after confidence filtering")
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0), np.concatenate(groups, axis=0)


def _token_feature_matrix(activation: Any) -> np.ndarray:
    arr = _to_numpy(activation)
    if arr.ndim == 0:
        raise ValueError("activation tensor must have at least one dimension")
    if arr.ndim == 1:
        return arr.reshape(1, -1).astype(np.float32)
    return arr.reshape(-1, arr.shape[-1]).astype(np.float32)


def _ridge_blocked_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int,
    ridge_alpha: float,
) -> dict[str, float | int]:
    folds = _blocked_fold_indices(x.shape[0], cv_folds=cv_folds)
    return _ridge_cv_metrics(x, y, folds=folds, ridge_alpha=ridge_alpha)


def _ridge_call_heldout_cv(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    cv_folds: int,
    ridge_alpha: float,
) -> dict[str, float | int | str]:
    call_group_count = int(np.unique(groups).shape[0])
    folds = _group_heldout_fold_indices(groups, cv_folds=cv_folds)
    if not folds:
        return {
            "correlation_mean": 0.0,
            "correlation_std": 0.0,
            "normalized_mse_mean": 0.0,
            "normalized_mse_std": 0.0,
            "r2_mean": 0.0,
            "r2_std": 0.0,
            "fold_count": 0,
            "call_group_count": call_group_count,
            "status": "insufficient_call_groups",
        }
    metrics = _ridge_cv_metrics(x, y, folds=folds, ridge_alpha=ridge_alpha)
    return {**metrics, "call_group_count": call_group_count, "status": "ok"}


def _ridge_cv_metrics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    ridge_alpha: float,
) -> dict[str, float | int]:
    correlations = []
    normalized_mses = []
    r2s = []
    for train_idx, test_idx in folds:
        pred = _fit_predict_ridge(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            ridge_alpha=ridge_alpha,
        )
        target = y[test_idx].astype(np.float32)
        mse = float(np.mean((pred - target) ** 2))
        target_var = float(np.var(target))
        baseline = float(np.mean((target - float(np.mean(y[train_idx]))) ** 2))
        normalized_mses.append(mse / max(target_var, 1e-8))
        r2s.append(1.0 - mse / max(baseline, 1e-8))
        correlations.append(_safe_correlation(pred, target))
    if not correlations:
        return {
            "correlation_mean": 0.0,
            "correlation_std": 0.0,
            "normalized_mse_mean": 0.0,
            "normalized_mse_std": 0.0,
            "r2_mean": 0.0,
            "r2_std": 0.0,
            "fold_count": 0,
        }
    return {
        "correlation_mean": float(np.mean(correlations)),
        "correlation_std": float(np.std(correlations)),
        "normalized_mse_mean": float(np.mean(normalized_mses)),
        "normalized_mse_std": float(np.std(normalized_mses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "fold_count": int(len(folds)),
    }


def _ridge_blocked_cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int,
    ridge_alpha: float,
) -> list[dict[str, float | int]]:
    folds = _blocked_fold_indices(x.shape[0], cv_folds=cv_folds)
    if not folds:
        return []
    predictions: list[dict[str, float | int]] = [
        {"target": float(value), "prediction": 0.0, "fold_index": -1}
        for value in y.astype(np.float32)
    ]
    for fold_index, (train_idx, test_idx) in enumerate(folds):
        pred = _fit_predict_ridge(
            x[train_idx],
            y[train_idx],
            x[test_idx],
            ridge_alpha=ridge_alpha,
        )
        for local_index, sample_index in enumerate(test_idx):
            predictions[int(sample_index)] = {
                "target": float(y[int(sample_index)]),
                "prediction": float(pred[local_index]),
                "fold_index": int(fold_index),
            }
    return predictions


def _fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    ridge_alpha: float,
) -> np.ndarray:
    x_train_std, x_test_std = _standardize_train_test(x_train, x_test)
    y_mean = float(np.mean(y_train))
    y_centered = y_train.astype(np.float32) - y_mean
    weights = _ridge_weights(x_train_std, y_centered, ridge_alpha=ridge_alpha)
    return (x_test_std @ weights + y_mean).astype(np.float32)


def _active_quiet_contrast(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_alpha: float,
    active_percentile: float,
    quiet_percentile: float,
) -> dict[str, float | int]:
    x_std, _ = _standardize_train_test(x, x)
    y_norm = normalize_array(y)
    active_threshold = float(np.percentile(y_norm, active_percentile))
    quiet_threshold = float(np.percentile(y_norm, quiet_percentile))
    active_mask = y_norm >= active_threshold
    quiet_mask = y_norm <= quiet_threshold
    if not np.any(active_mask) or not np.any(quiet_mask):
        return {
            "active_count": int(np.sum(active_mask)),
            "quiet_count": int(np.sum(quiet_mask)),
            "active_delta_norm": 0.0,
            "ridge_active_cosine": 0.0,
            "active_threshold": active_threshold,
            "quiet_threshold": quiet_threshold,
        }
    delta = x_std[active_mask].mean(axis=0) - x_std[quiet_mask].mean(axis=0)
    weights = _ridge_weights(x_std, y.astype(np.float32) - float(np.mean(y)), ridge_alpha=ridge_alpha)
    return {
        "active_count": int(np.sum(active_mask)),
        "quiet_count": int(np.sum(quiet_mask)),
        "active_delta_norm": float(np.linalg.norm(delta)),
        "ridge_active_cosine": _safe_cosine(delta, weights),
        "active_threshold": active_threshold,
        "quiet_threshold": quiet_threshold,
    }


def _active_quiet_direction_preview(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge_alpha: float,
    active_percentile: float,
    quiet_percentile: float,
    top_features: int,
) -> dict[str, Any]:
    x_std, _ = _standardize_train_test(x, x)
    y_norm = normalize_array(y)
    active_threshold = float(np.percentile(y_norm, active_percentile))
    quiet_threshold = float(np.percentile(y_norm, quiet_percentile))
    active_mask = y_norm >= active_threshold
    quiet_mask = y_norm <= quiet_threshold
    if not np.any(active_mask) or not np.any(quiet_mask):
        return {
            "active_count": int(np.sum(active_mask)),
            "quiet_count": int(np.sum(quiet_mask)),
            "active_threshold": active_threshold,
            "quiet_threshold": quiet_threshold,
            "active_delta_norm": 0.0,
            "ridge_active_cosine": 0.0,
            "top_feature_indices": [],
            "top_feature_values": [],
            "top_feature_abs_values": [],
        }
    delta = x_std[active_mask].mean(axis=0) - x_std[quiet_mask].mean(axis=0)
    weights = _ridge_weights(x_std, y.astype(np.float32) - float(np.mean(y)), ridge_alpha=ridge_alpha)
    order = np.argsort(-np.abs(delta))[: max(1, int(top_features))]
    return {
        "active_count": int(np.sum(active_mask)),
        "quiet_count": int(np.sum(quiet_mask)),
        "active_threshold": active_threshold,
        "quiet_threshold": quiet_threshold,
        "active_delta_norm": float(np.linalg.norm(delta)),
        "ridge_active_cosine": _safe_cosine(delta, weights),
        "top_feature_indices": [int(index) for index in order],
        "top_feature_values": [float(delta[index]) for index in order],
        "top_feature_abs_values": [float(abs(delta[index])) for index in order],
    }


def _ridge_weights(x: np.ndarray, y: np.ndarray, *, ridge_alpha: float) -> np.ndarray:
    alpha = float(ridge_alpha)
    sample_count, feature_count = x.shape
    if sample_count < feature_count:
        system = x @ x.T + alpha * np.eye(sample_count, dtype=np.float32)
        try:
            dual = np.linalg.solve(system, y.astype(np.float32))
        except np.linalg.LinAlgError:
            dual = np.linalg.lstsq(system, y.astype(np.float32), rcond=None)[0]
        return (x.T @ dual).astype(np.float32)
    system = x.T @ x + alpha * np.eye(feature_count, dtype=np.float32)
    rhs = x.T @ y.astype(np.float32)
    try:
        return np.linalg.solve(system, rhs).astype(np.float32)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0].astype(np.float32)


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def _blocked_fold_indices(n: int, *, cv_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_count = min(max(2, int(cv_folds)), max(2, int(n)))
    folds = np.array_split(np.arange(n, dtype=np.int64), fold_count)
    result = []
    all_indices = np.arange(n, dtype=np.int64)
    for test_idx in folds:
        if test_idx.size == 0:
            continue
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        if train_idx.size == 0:
            continue
        result.append((train_idx, test_idx))
    return result


def _group_heldout_fold_indices(groups: np.ndarray, *, cv_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    unique_groups = np.unique(groups)
    if unique_groups.shape[0] < 2:
        return []
    fold_count = min(max(2, int(cv_folds)), int(unique_groups.shape[0]))
    group_folds = np.array_split(unique_groups, fold_count)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    indices = np.arange(groups.shape[0], dtype=np.int64)
    for test_groups in group_folds:
        if test_groups.size == 0:
            continue
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            continue
        result.append((indices[train_mask], indices[test_mask]))
    return result


def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _activation_calls(value: Sequence[Any] | Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("activation call list is empty")
        return list(value)
    return [value]


def _select_lanes(lanes: Sequence[ControlLane], lane_names: Sequence[str] | None) -> list[ControlLane]:
    if lane_names is None:
        return list(lanes)
    lane_map = {lane.name: lane for lane in lanes}
    missing = [name for name in lane_names if name not in lane_map]
    if missing:
        raise KeyError(f"missing control lanes: {missing}")
    return [lane_map[name] for name in lane_names]


def _select_layers(
    raw_activations: Mapping[int, Sequence[Any] | Any],
    layer_indices: Sequence[int] | None,
) -> list[int]:
    if layer_indices is None:
        return sorted(int(layer) for layer in raw_activations)
    missing = [int(layer) for layer in layer_indices if int(layer) not in raw_activations]
    if missing:
        raise KeyError(f"missing activation layers: {missing}")
    return [int(layer) for layer in layer_indices]


def _select_timestep_layers(
    layer_timestep_activations: Mapping[int, Mapping[int, Any]] | Mapping[tuple[int, int], Any],
    layer_indices: Sequence[int] | None,
) -> list[int]:
    if not layer_timestep_activations:
        return []
    first_key = next(iter(layer_timestep_activations))
    if isinstance(first_key, tuple):
        available = sorted({int(key[0]) for key in layer_timestep_activations})  # type: ignore[index]
    else:
        available = sorted(int(layer) for layer in layer_timestep_activations)
    if layer_indices is None:
        return available
    missing = [int(layer) for layer in layer_indices if int(layer) not in available]
    if missing:
        raise KeyError(f"missing timestep activation layers: {missing}")
    return [int(layer) for layer in layer_indices]


def _iter_layer_timestep_activations(
    layer_timestep_activations: Mapping[int, Mapping[int, Any]] | Mapping[tuple[int, int], Any],
    layer_idx: int,
    metadata: Mapping[int, Mapping[int, Mapping[str, Any]]] | Mapping[tuple[int, int], Mapping[str, Any]] | None,
) -> list[tuple[int, Any, dict[str, Any]]]:
    if not layer_timestep_activations:
        return []
    first_key = next(iter(layer_timestep_activations))
    out: list[tuple[int, Any, dict[str, Any]]] = []
    if isinstance(first_key, tuple):
        flat_activations = layer_timestep_activations  # type: ignore[assignment]
        flat_metadata = metadata or {}
        for key in sorted(flat_activations):
            if int(key[0]) != int(layer_idx):
                continue
            step_idx = int(key[1])
            out.append((step_idx, flat_activations[key], dict(flat_metadata.get(key, {}))))  # type: ignore[arg-type]
        return out
    nested_activations = layer_timestep_activations.get(layer_idx, {})  # type: ignore[union-attr]
    nested_metadata = {} if metadata is None else metadata.get(layer_idx, {})  # type: ignore[union-attr]
    for step_idx in sorted(nested_activations):
        out.append((int(step_idx), nested_activations[step_idx], dict(nested_metadata.get(step_idx, {}))))
    return out


def _rank_rows_within_lane(rows: Sequence[ControlLaneProbeRow]) -> list[ControlLaneProbeRow]:
    ranked: list[ControlLaneProbeRow] = []
    for lane_name in sorted({row.lane_name for row in rows}):
        lane_rows = [row for row in rows if row.lane_name == lane_name]
        lane_rows.sort(
            key=lambda row: (
                row.status != "ok",
                row.call_group_count < 2,
                -_row_rank_score(row),
                -row.correlation_mean,
                row.normalized_mse_mean,
                row.layer_index,
                -1 if row.window_index is None else row.window_index,
            )
        )
        for rank, row in enumerate(lane_rows, start=1):
            ranked.append(_copy_row_with_rank(row, rank=rank))
    return ranked


def _row_rank_score(row: ControlLaneProbeRow) -> float:
    if row.call_heldout_status == "ok":
        return float(row.call_heldout_correlation_mean)
    return float(row.correlation_mean)


def _copy_row_with_rank(row: ControlLaneProbeRow, *, rank: int) -> ControlLaneProbeRow:
    return ControlLaneProbeRow(
        lane_name=row.lane_name,
        layer_index=row.layer_index,
        method=row.method,
        alignment_mode=row.alignment_mode,
        correlation_mean=row.correlation_mean,
        correlation_std=row.correlation_std,
        normalized_mse_mean=row.normalized_mse_mean,
        normalized_mse_std=row.normalized_mse_std,
        r2_mean=row.r2_mean,
        r2_std=row.r2_std,
        fold_count=row.fold_count,
        sample_count=row.sample_count,
        feature_count=row.feature_count,
        lane_frames=row.lane_frames,
        activation_call_count=row.activation_call_count,
        active_count=row.active_count,
        quiet_count=row.quiet_count,
        active_delta_norm=row.active_delta_norm,
        ridge_active_cosine=row.ridge_active_cosine,
        active_threshold=row.active_threshold,
        quiet_threshold=row.quiet_threshold,
        call_group_count=row.call_group_count,
        call_heldout_correlation_mean=row.call_heldout_correlation_mean,
        call_heldout_correlation_std=row.call_heldout_correlation_std,
        call_heldout_normalized_mse_mean=row.call_heldout_normalized_mse_mean,
        call_heldout_normalized_mse_std=row.call_heldout_normalized_mse_std,
        call_heldout_r2_mean=row.call_heldout_r2_mean,
        call_heldout_r2_std=row.call_heldout_r2_std,
        call_heldout_fold_count=row.call_heldout_fold_count,
        call_heldout_status=row.call_heldout_status,
        rank=rank,
        status=row.status,
        error=row.error,
        window_index=row.window_index,
        window_label=row.window_label,
        window_start_fraction=row.window_start_fraction,
        window_end_fraction=row.window_end_fraction,
        call_start=row.call_start,
        call_end=row.call_end,
        window_count=row.window_count,
        step_index=row.step_index,
        sampler_index=row.sampler_index,
        timestep=row.timestep,
        sigma=row.sigma,
        logsnr=row.logsnr,
        sampler_type=row.sampler_type,
        calls_per_step=row.calls_per_step,
        mapping_status=row.mapping_status,
        null_kind=row.null_kind,
        source=row.source,
    )


def _probe_match_key(row: ControlLaneProbeRow | Mapping[str, Any]) -> tuple[Any, ...]:
    layer_index = _optional_int(_row_get(row, "layer_index"))
    return (
        str(_row_get(row, "lane_name", "")),
        -1 if layer_index is None else int(layer_index),
        str(_row_get(row, "alignment_mode", "")),
        _optional_int(_row_get(row, "window_index")),
        _optional_int(_row_get(row, "call_start")),
        _optional_int(_row_get(row, "call_end")),
        _optional_int(_row_get(row, "step_index")),
        _optional_int(_row_get(row, "sampler_index")),
    )


def _matched_null_kinds(rows: Sequence[ControlLaneProbeRow | Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            str(_row_get(row, "null_kind", "") or "")
            for row in rows
            if str(_row_get(row, "null_kind", "") or "")
        }
    )


def _empty_probe_row(
    lane: ControlLane,
    *,
    layer_index: int,
    alignment_mode: str,
    method: str,
    status: str,
    error: str,
    window_index: int | None,
    window_label: str,
    call_start: int | None,
    call_end: int | None,
    window_count: int | None,
    sample_count: int = 0,
    feature_count: int = 0,
    activation_call_count: int = 0,
    step_metadata: Mapping[str, Any] | None = None,
    null_kind: str = "",
) -> ControlLaneProbeRow:
    step_metadata = dict(step_metadata or {})
    return ControlLaneProbeRow(
        lane_name=lane.name,
        layer_index=layer_index,
        method=method,
        alignment_mode=alignment_mode,
        correlation_mean=0.0,
        correlation_std=0.0,
        normalized_mse_mean=0.0,
        normalized_mse_std=0.0,
        r2_mean=0.0,
        r2_std=0.0,
        fold_count=0,
        sample_count=int(sample_count),
        feature_count=int(feature_count),
        lane_frames=lane.frames,
        activation_call_count=int(activation_call_count),
        active_count=0,
        quiet_count=0,
        active_delta_norm=0.0,
        ridge_active_cosine=0.0,
        active_threshold=0.0,
        quiet_threshold=0.0,
        status=status,
        error=error,
        window_index=window_index,
        window_label=window_label,
        call_start=call_start,
        call_end=call_end,
        window_count=window_count,
        step_index=_optional_int(step_metadata.get("step_index")),
        sampler_index=_optional_int(step_metadata.get("sampler_index")),
        timestep=_optional_float(step_metadata.get("timestep")),
        sigma=_optional_float(step_metadata.get("sigma")),
        logsnr=_optional_float(step_metadata.get("logsnr")),
        sampler_type=str(step_metadata.get("sampler_type", "") or ""),
        calls_per_step=_optional_int(step_metadata.get("calls_per_step")),
        mapping_status=str(step_metadata.get("mapping_status", "") or ""),
        null_kind=str(null_kind or ""),
        source=lane.source,
    )


def _window_label(window_index: int, window_count: int, start: int, end: int) -> str:
    if window_count == 1:
        return "all"
    if window_count == 2:
        return ("early", "late")[window_index]
    if window_count == 3:
        return ("early", "middle", "late")[window_index]
    return f"window_{window_index:02d}_{start}_{end}"


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _null_control_lane(lane: ControlLane, null_kind: str, rng: np.random.Generator) -> ControlLane:
    kind = str(null_kind).lower()
    values = lane.values.astype(np.float32)
    confidence = None if lane.confidence is None else lane.confidence.astype(np.float32)
    if kind in {"shuffle", "shuffled", "permute", "permutation"}:
        values = values[rng.permutation(values.shape[0])]
    elif kind in {"reverse", "reversed", "time_reverse"}:
        values = values[::-1]
        confidence = None if confidence is None else confidence[::-1]
    elif kind in {"random", "gaussian"}:
        values = rng.normal(float(values.mean()), max(float(values.std()), 1e-6), size=values.shape).astype(np.float32)
    else:
        raise ValueError("null_kind must be 'shuffle', 'reverse', or 'random'")
    return ControlLane(
        name=lane.name,
        values=values,
        rate_hz=lane.rate_hz,
        confidence=confidence,
        source=lane.source,
        metadata={**lane.metadata, "null_kind": kind},
    )


def _calls_for_probe_row(value: Sequence[Any] | Any, row: ControlLaneProbeRow | Mapping[str, Any]) -> list[Any]:
    calls = _activation_calls(value)
    start = _optional_int(_row_get(row, "call_start"))
    end = _optional_int(_row_get(row, "call_end"))
    if start is None or end is None:
        return calls
    start = max(0, min(len(calls), start))
    end = max(start, min(len(calls), end))
    return calls[start:end]


def _select_top_probe_rows(
    rows: Sequence[ControlLaneProbeRow | Mapping[str, Any]],
    *,
    top_k_per_lane: int,
    include_null: bool = False,
) -> list[ControlLaneProbeRow | Mapping[str, Any]]:
    grouped: dict[str, list[ControlLaneProbeRow | Mapping[str, Any]]] = {}
    for row in rows:
        if str(_row_get(row, "status", "ok") or "ok") not in {"", "ok"}:
            continue
        if not include_null and str(_row_get(row, "null_kind", "") or ""):
            continue
        lane_name = str(_row_get(row, "lane_name", ""))
        if not lane_name:
            continue
        grouped.setdefault(lane_name, []).append(row)
    selected: list[ControlLaneProbeRow | Mapping[str, Any]] = []
    for lane_name in sorted(grouped):
        lane_rows = grouped[lane_name]
        lane_rows.sort(
            key=lambda row: (
                int(_row_get(row, "rank", 999999) or 999999),
                -float(_row_get(row, "correlation_mean", 0.0) or 0.0),
                float(_row_get(row, "normalized_mse_mean", 0.0) or 0.0),
            )
        )
        selected.extend(lane_rows[: max(1, int(top_k_per_lane))])
    return selected


def _row_get(row: ControlLaneProbeRow | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_to_dict(row: ControlLaneProbeRow | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    return row.to_dict()


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
