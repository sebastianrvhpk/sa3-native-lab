"""Shared notebook records for searchable SAME latent audio items."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class LatentItem:
    """A searchable unit of latent audio memory.

    Latents are stored time-major as ``(time, dim)``. SAME latents that arrive
    as ``(channels, time)`` should be transposed before construction or passed
    through ``from_channel_first``.
    """

    item_id: str
    latent: np.ndarray
    latent_rate: float
    sample_rate: int | None = None
    prompt: str | None = None
    descriptors: dict[str, float] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        latent = np.asarray(self.latent, dtype=np.float32)
        if latent.ndim != 2:
            raise ValueError(f"latent must be 2D time-major (T, D), got shape {latent.shape}")
        if latent.shape[0] < 1 or latent.shape[1] < 1:
            raise ValueError(f"latent must have non-empty time and feature axes, got {latent.shape}")
        if not np.isfinite(latent).all():
            raise ValueError("latent contains NaN or infinite values")
        if self.latent_rate <= 0:
            raise ValueError("latent_rate must be positive")
        self.latent = np.ascontiguousarray(latent)
        self.descriptors = {str(k): float(v) for k, v in self.descriptors.items()}

    @property
    def duration_seconds(self) -> float:
        return float(self.latent.shape[0] / self.latent_rate)

    @property
    def dim(self) -> int:
        return int(self.latent.shape[1])

    @property
    def frames(self) -> int:
        return int(self.latent.shape[0])

    @classmethod
    def from_channel_first(
        cls,
        item_id: str,
        latent: np.ndarray,
        latent_rate: float,
        **kwargs: Any,
    ) -> "LatentItem":
        """Construct from a ``(channels, time)`` latent array such as SAME."""

        arr = np.asarray(latent, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"channel-first latent must be 2D, got shape {arr.shape}")
        return cls(item_id=item_id, latent=arr.T, latent_rate=latent_rate, **kwargs)

    def shallow_metadata(self) -> dict[str, Any]:
        """Return JSON-friendly metadata without the latent array."""

        return {
            "item_id": self.item_id,
            "frames": self.frames,
            "dim": self.dim,
            "latent_rate": self.latent_rate,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "prompt": self.prompt,
            "descriptors": dict(self.descriptors),
            "labels": dict(self.labels),
            "metadata": dict(self.metadata),
        }
