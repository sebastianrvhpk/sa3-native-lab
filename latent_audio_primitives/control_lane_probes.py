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


def control_lane_probe_table(rows: Sequence[ControlLaneProbeRow]) -> list[dict[str, Any]]:
    """Return JSON/table-friendly control-lane probe rows."""

    return [row.to_dict() for row in rows]


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
) -> ControlLaneProbeRow:
    try:
        x, y = _features_and_targets_for_calls(calls, lane, min_confidence=min_confidence)
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
        )
    metrics = _ridge_blocked_cv(
        x,
        y,
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
        window_index=window_index,
        window_label=window_label,
        window_start_fraction=None if call_start is None else call_start / max(total_call_count or len(calls), 1),
        window_end_fraction=None if call_end is None else call_end / max(total_call_count or len(calls), 1),
        call_start=call_start,
        call_end=call_end,
        window_count=window_count,
        source=lane.source,
    )


def _features_and_targets_for_calls(
    calls: Sequence[Any],
    lane: ControlLane,
    *,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for activation in calls:
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
    if not features:
        raise ValueError("no activation/lane samples after confidence filtering")
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


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
    return {
        "correlation_mean": float(np.mean(correlations)),
        "correlation_std": float(np.std(correlations)),
        "normalized_mse_mean": float(np.mean(normalized_mses)),
        "normalized_mse_std": float(np.std(normalized_mses)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "fold_count": int(len(folds)),
    }


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


def _rank_rows_within_lane(rows: Sequence[ControlLaneProbeRow]) -> list[ControlLaneProbeRow]:
    ranked: list[ControlLaneProbeRow] = []
    for lane_name in sorted({row.lane_name for row in rows}):
        lane_rows = [row for row in rows if row.lane_name == lane_name]
        lane_rows.sort(
            key=lambda row: (
                row.status != "ok",
                -row.correlation_mean,
                row.normalized_mse_mean,
                row.layer_index,
                -1 if row.window_index is None else row.window_index,
            )
        )
        for rank, row in enumerate(lane_rows, start=1):
            ranked.append(_copy_row_with_rank(row, rank=rank))
    return ranked


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
        source=row.source,
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
) -> ControlLaneProbeRow:
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
