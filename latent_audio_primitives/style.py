"""SAME latent style profiles, contrastive directions, and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .latent_math import as_time_major, latent_velocity
from .schema import LatentItem


@dataclass(slots=True)
class LatentStyleProfile:
    """Dataset-level SAME latent statistics for directional style transfer."""

    name: str
    dim: int
    item_count: int
    mean: np.ndarray
    std: np.ndarray
    mean_abs_velocity: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean = _vector(self.mean, self.dim, "mean")
        self.std = np.maximum(_vector(self.std, self.dim, "std"), 1e-6)
        self.mean_abs_velocity = _vector(self.mean_abs_velocity, self.dim, "mean_abs_velocity")


@dataclass(slots=True)
class LatentStyleDirection:
    """Difference between two dataset style profiles."""

    name: str
    target_name: str
    reference_name: str
    dim: int
    mean_delta: np.ndarray
    std_delta: np.ndarray
    velocity_delta: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mean_delta = _vector(self.mean_delta, self.dim, "mean_delta")
        self.std_delta = _vector(self.std_delta, self.dim, "std_delta")
        self.velocity_delta = _vector(self.velocity_delta, self.dim, "velocity_delta")


def fit_style_profile(items: list[LatentItem], *, name: str = "style", metadata: dict[str, Any] | None = None) -> LatentStyleProfile:
    """Fit dataset-level latent statistics from memory items."""

    if not items:
        raise ValueError("at least one item is required")
    dim = items[0].dim
    if any(item.dim != dim for item in items):
        raise ValueError("all items must have the same latent dimension")

    means = []
    stds = []
    velocities = []
    for item in items:
        z = as_time_major(item)
        means.append(z.mean(axis=0))
        stds.append(np.maximum(z.std(axis=0), 1e-6))
        velocity = latent_velocity(z)
        if velocity.shape[0] == 0:
            velocities.append(np.zeros(dim, dtype=np.float32))
        else:
            velocities.append(np.abs(velocity).mean(axis=0))

    return LatentStyleProfile(
        name=name,
        dim=dim,
        item_count=len(items),
        mean=np.stack(means).mean(axis=0),
        std=np.stack(stds).mean(axis=0),
        mean_abs_velocity=np.stack(velocities).mean(axis=0),
        metadata=metadata or {},
    )


def style_direction(
    target: LatentStyleProfile,
    reference: LatentStyleProfile,
    *,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> LatentStyleDirection:
    """Compute a target-minus-reference style direction."""

    if target.dim != reference.dim:
        raise ValueError("target and reference profiles must have the same dimension")
    return LatentStyleDirection(
        name=name or f"{target.name}_minus_{reference.name}",
        target_name=target.name,
        reference_name=reference.name,
        dim=target.dim,
        mean_delta=target.mean - reference.mean,
        std_delta=target.std - reference.std,
        velocity_delta=target.mean_abs_velocity - reference.mean_abs_velocity,
        metadata=metadata or {},
    )


def apply_profile_attraction(
    latent: LatentItem | np.ndarray,
    profile: LatentStyleProfile,
    *,
    alpha: float = 1.0,
    match_std: bool = True,
    eps: float = 1e-5,
) -> np.ndarray:
    """Move a latent toward dataset style statistics.

    At ``alpha=1`` with ``match_std=True``, this is an AdaIN-style operation:

    ``z' = target_std * (z - mean(z)) / std(z) + target_mean``

    Lower alpha interpolates between the original latent and the styled latent.
    Alpha above 1 extrapolates past the dataset statistics.
    """

    z = as_time_major(latent).astype(np.float32, copy=True)
    _check_dim(z, profile.dim)

    current_mean = z.mean(axis=0)
    current_std = np.maximum(z.std(axis=0), eps)

    if match_std:
        target_std = np.maximum(profile.std, eps)
        styled = ((z - current_mean) / current_std) * target_std + profile.mean
    else:
        styled = z + (profile.mean - current_mean)
    return ((1.0 - alpha) * z + alpha * styled).astype(np.float32, copy=False)


def apply_style_direction(
    latent: LatentItem | np.ndarray,
    direction: LatentStyleDirection,
    *,
    alpha: float = 1.0,
    std_alpha: float = 0.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """Apply a fixed dataset difference direction to a latent sequence."""

    z = as_time_major(latent).astype(np.float32, copy=True)
    _check_dim(z, direction.dim)
    shifted = z + alpha * direction.mean_delta

    if std_alpha == 0:
        return shifted.astype(np.float32, copy=False)

    current_mean = shifted.mean(axis=0)
    current_std = np.maximum(shifted.std(axis=0), eps)
    target_std = np.maximum(current_std + std_alpha * direction.std_delta, eps)
    scaled = ((shifted - current_mean) / current_std) * target_std + current_mean
    return scaled.astype(np.float32, copy=False)


def apply_profile_to_item(
    item: LatentItem,
    profile: LatentStyleProfile,
    *,
    alpha: float = 1.0,
    match_std: bool = True,
    item_id: str | None = None,
) -> LatentItem:
    """Return a new LatentItem after profile attraction, preserving metadata."""

    latent = apply_profile_attraction(item, profile, alpha=alpha, match_std=match_std)
    metadata = dict(item.metadata)
    metadata["style_profile"] = profile.name
    metadata["style_alpha"] = alpha
    metadata["style_match_std"] = match_std
    return LatentItem(
        item_id=item_id or f"{item.item_id}__styled_{profile.name}",
        latent=latent,
        latent_rate=item.latent_rate,
        sample_rate=item.sample_rate,
        prompt=item.prompt,
        descriptors=dict(item.descriptors),
        labels=dict(item.labels),
        metadata=metadata,
    )


def save_style_profile(profile: LatentStyleProfile, path: str | Path) -> Path:
    """Save a SAME latent style profile as an ``.npz`` artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        kind="LatentStyleProfile",
        name=profile.name,
        dim=profile.dim,
        item_count=profile.item_count,
        mean=profile.mean,
        std=profile.std,
        mean_abs_velocity=profile.mean_abs_velocity,
        metadata=json.dumps(profile.metadata),
    )
    return path


def load_style_profile(path: str | Path) -> LatentStyleProfile:
    """Load a SAME latent style profile from an ``.npz`` artifact."""

    with np.load(Path(path), allow_pickle=False) as data:
        return LatentStyleProfile(
            name=str(data["name"]),
            dim=int(data["dim"]),
            item_count=int(data["item_count"]),
            mean=data["mean"],
            std=data["std"],
            mean_abs_velocity=data["mean_abs_velocity"],
            metadata=json.loads(str(data["metadata"])),
        )


def save_style_direction(direction: LatentStyleDirection, path: str | Path) -> Path:
    """Save a SAME latent style direction as an ``.npz`` artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        kind="LatentStyleDirection",
        name=direction.name,
        target_name=direction.target_name,
        reference_name=direction.reference_name,
        dim=direction.dim,
        mean_delta=direction.mean_delta,
        std_delta=direction.std_delta,
        velocity_delta=direction.velocity_delta,
        metadata=json.dumps(direction.metadata),
    )
    return path


def load_style_direction(path: str | Path) -> LatentStyleDirection:
    """Load a SAME latent style direction from an ``.npz`` artifact."""

    with np.load(Path(path), allow_pickle=False) as data:
        return LatentStyleDirection(
            name=str(data["name"]),
            target_name=str(data["target_name"]),
            reference_name=str(data["reference_name"]),
            dim=int(data["dim"]),
            mean_delta=data["mean_delta"],
            std_delta=data["std_delta"],
            velocity_delta=data["velocity_delta"],
            metadata=json.loads(str(data["metadata"])),
        )


def _vector(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (dim,):
        raise ValueError(f"{name} must have shape ({dim},), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def _check_dim(latent: np.ndarray, dim: int) -> None:
    if latent.shape[1] != dim:
        raise ValueError(f"latent dim {latent.shape[1]} does not match style dim {dim}")
