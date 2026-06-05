"""Residual activation bases and directions for SA3 feature atlas probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ResidualFeatureBasis:
    """PCA feature basis for SA3 residual-stream activation vectors."""

    layer: str
    mean: np.ndarray
    components: np.ndarray
    variances: np.ndarray


def fit_residual_feature_basis(
    activations: Sequence[np.ndarray],
    *,
    layer: str = "unknown",
    n_components: int = 16,
) -> ResidualFeatureBasis:
    """Fit a PCA basis over flattened residual activations."""

    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in activations]
    if len(rows) < 2:
        raise ValueError("at least two activation samples are required")
    dim = rows[0].shape[0]
    if any(row.shape[0] != dim for row in rows):
        raise ValueError("all activation samples must have the same flattened size")
    x = np.stack(rows)
    mean = x.mean(axis=0)
    centered = x - mean
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    keep = max(1, min(int(n_components), vh.shape[0]))
    variances = (s[:keep] ** 2 / max(x.shape[0] - 1, 1)).astype(np.float32)
    return ResidualFeatureBasis(
        layer=layer,
        mean=mean.astype(np.float32),
        components=vh[:keep].astype(np.float32),
        variances=variances,
    )


def project_residual_features(activation: np.ndarray, basis: ResidualFeatureBasis, *, whiten: bool = False) -> np.ndarray:
    """Project one residual activation into a fitted feature basis."""

    row = np.asarray(activation, dtype=np.float32).reshape(-1)
    if row.shape != basis.mean.shape:
        raise ValueError(f"activation shape {row.shape} does not match basis shape {basis.mean.shape}")
    coeffs = (row - basis.mean) @ basis.components.T
    if whiten:
        coeffs = coeffs / np.sqrt(np.maximum(basis.variances, 1e-8))
    return coeffs.astype(np.float32, copy=False)


def residual_feature_direction(
    positive: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Mean-difference direction over flattened residual activations."""

    pos = _activation_matrix(positive)
    ref = _activation_matrix(reference)
    if pos.shape[1] != ref.shape[1]:
        raise ValueError("positive and reference activations must have the same dimension")
    direction = pos.mean(axis=0) - ref.mean(axis=0)
    if normalize:
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            direction = direction / norm
    return direction.astype(np.float32, copy=False)


def _activation_matrix(values: Sequence[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32).reshape(-1) for value in values]
    if not rows:
        raise ValueError("at least one activation is required")
    dim = rows[0].shape[0]
    if any(row.shape[0] != dim for row in rows):
        raise ValueError("all activations must have the same flattened size")
    return np.stack(rows)
