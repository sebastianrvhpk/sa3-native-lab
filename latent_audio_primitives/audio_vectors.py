from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .latent_math import as_time_major, latent_summary
from .schema import LatentItem
from .style import LatentStyleDirection, fit_style_profile, save_style_direction, style_direction


@dataclass(slots=True)
class AudioSetDirection:
    """A paired-audio direction in SAME memory space."""

    name: str
    positive_name: str
    negative_name: str
    vector: np.ndarray
    dim: int
    item_count_positive: int
    item_count_negative: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32).reshape(-1)
        if self.vector.shape != (self.dim,):
            raise ValueError(f"vector must have shape ({self.dim},), got {self.vector.shape}")
        if not np.isfinite(self.vector).all():
            raise ValueError("vector contains NaN or infinite values")


def summary_direction(
    positive: list[LatentItem],
    negative: list[LatentItem],
    *,
    name: str = "audio_summary_direction",
    normalize: bool = True,
    metadata: dict[str, Any] | None = None,
) -> AudioSetDirection:
    """Compute mean(summary(positive)) - mean(summary(negative))."""

    if not positive or not negative:
        raise ValueError("positive and negative item lists must be non-empty")
    pos = np.stack([latent_summary(item) for item in positive]).mean(axis=0)
    neg = np.stack([latent_summary(item) for item in negative]).mean(axis=0)
    vector = pos - neg
    if normalize:
        norm = np.linalg.norm(vector)
        if norm > 1e-8:
            vector = vector / norm
    return AudioSetDirection(
        name=name,
        positive_name="positive",
        negative_name="negative",
        vector=vector,
        dim=vector.shape[0],
        item_count_positive=len(positive),
        item_count_negative=len(negative),
        metadata=metadata or {},
    )


def frame_mean_direction(
    positive: list[LatentItem],
    negative: list[LatentItem],
    *,
    name: str = "audio_frame_mean_direction",
    normalize: bool = False,
    metadata: dict[str, Any] | None = None,
) -> LatentStyleDirection:
    """Compute target-minus-reference framewise SAME direction.

    This returns a ``LatentStyleDirection`` so it can be applied directly to
    generated latents with ``apply_style_direction``.
    """

    pos_profile = fit_style_profile(positive, name="audio_positive")
    neg_profile = fit_style_profile(negative, name="audio_negative")
    direction = style_direction(pos_profile, neg_profile, name=name, metadata=metadata)
    if normalize:
        norm = np.linalg.norm(direction.mean_delta)
        if norm > 1e-8:
            direction.mean_delta = direction.mean_delta / norm
    return direction


def apply_frame_direction(
    latent: LatentItem | np.ndarray,
    direction: LatentStyleDirection,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply an audio-derived frame mean direction to every latent frame."""

    z = as_time_major(latent).astype(np.float32, copy=True)
    if z.shape[1] != direction.dim:
        raise ValueError(f"latent dim {z.shape[1]} does not match direction dim {direction.dim}")
    return (z + alpha * direction.mean_delta).astype(np.float32, copy=False)


def save_summary_direction(direction: AudioSetDirection, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        kind="AudioSetDirection",
        name=direction.name,
        positive_name=direction.positive_name,
        negative_name=direction.negative_name,
        vector=direction.vector,
        dim=direction.dim,
        item_count_positive=direction.item_count_positive,
        item_count_negative=direction.item_count_negative,
        metadata=json.dumps(direction.metadata),
    )
    return path


def load_summary_direction(path: str | Path) -> AudioSetDirection:
    with np.load(Path(path), allow_pickle=False) as data:
        return AudioSetDirection(
            name=str(data["name"]),
            positive_name=str(data["positive_name"]),
            negative_name=str(data["negative_name"]),
            vector=data["vector"],
            dim=int(data["dim"]),
            item_count_positive=int(data["item_count_positive"]),
            item_count_negative=int(data["item_count_negative"]),
            metadata=json.loads(str(data["metadata"])),
        )


def save_audio_frame_direction(direction: LatentStyleDirection, path: str | Path) -> Path:
    return save_style_direction(direction, path)
