"""Linear probes for whether controls are visible in latent summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .latent_math import latent_summary
from .schema import LatentItem


@dataclass(frozen=True, slots=True)
class LinearControlProbe:
    """Ridge-regression probe from latent summaries to a scalar control."""

    control_name: str
    weights: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    r2_train: float
    item_count: int


def fit_linear_control_probe(
    items: Sequence[LatentItem],
    control_name: str,
    *,
    ridge: float = 1e-3,
) -> LinearControlProbe:
    """Fit an observability probe for one numeric descriptor or label."""

    x_rows = []
    y_values = []
    for item in items:
        if control_name in item.descriptors:
            value = item.descriptors[control_name]
        elif control_name in item.labels:
            value = item.labels[control_name]
        else:
            continue
        x_rows.append(latent_summary(item))
        y_values.append(float(value))
    if len(x_rows) < 2:
        raise ValueError("at least two labelled items are required")
    x = np.stack(x_rows).astype(np.float32)
    y = np.asarray(y_values, dtype=np.float32)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    x_norm = (x - mean) / scale
    design = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float32)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float32) * float(ridge)
    penalty[-1, -1] = 0.0
    params = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    weights = params[:-1].astype(np.float32)
    intercept = float(params[-1])
    pred = design @ params
    r2 = r2_score(y, pred)
    return LinearControlProbe(
        control_name=control_name,
        weights=weights,
        intercept=intercept,
        feature_mean=mean,
        feature_scale=scale,
        r2_train=r2,
        item_count=len(y_values),
    )


def predict_control(probe: LinearControlProbe, latent: LatentItem | np.ndarray) -> float:
    """Predict a scalar control from one latent item or array."""

    summary = latent_summary(latent)
    if summary.shape != probe.weights.shape:
        raise ValueError(f"summary shape {summary.shape} does not match probe weights {probe.weights.shape}")
    x = (summary - probe.feature_mean) / probe.feature_scale
    return float(x @ probe.weights + probe.intercept)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return coefficient of determination for probe diagnostics."""

    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def intervention_effect(
    probe: LinearControlProbe,
    before: LatentItem | np.ndarray,
    after: LatentItem | np.ndarray,
) -> dict[str, float]:
    """Estimate whether a latent edit moved a predicted control."""

    y0 = predict_control(probe, before)
    y1 = predict_control(probe, after)
    return {
        "before": y0,
        "after": y1,
        "delta": y1 - y0,
        "train_r2": probe.r2_train,
    }
