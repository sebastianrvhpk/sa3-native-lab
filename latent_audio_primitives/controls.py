"""Small descriptor-target scoring helpers for memory and selection tasks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def control_distance(
    descriptors: Mapping[str, float],
    target: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    missing: str = "ignore",
) -> float:
    """Weighted Euclidean distance between descriptor values and targets."""

    if missing not in {"ignore", "error"}:
        raise ValueError("missing must be 'ignore' or 'error'")
    weights = weights or {}
    total = 0.0
    used = 0
    for key, target_value in target.items():
        if key not in descriptors:
            if missing == "error":
                raise KeyError(f"descriptor {key!r} is missing")
            continue
        weight = float(weights.get(key, 1.0))
        delta = float(descriptors[key]) - float(target_value)
        total += weight * delta * delta
        used += 1
    if used == 0:
        return float("inf") if missing == "error" else 0.0
    return float(np.sqrt(total))


def control_score(
    descriptors: Mapping[str, float],
    target: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    missing: str = "ignore",
) -> float:
    """Convert control distance into a higher-is-better score."""

    return -control_distance(descriptors, target, weights=weights, missing=missing)
