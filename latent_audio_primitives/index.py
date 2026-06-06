"""In-memory retrieval over LatentItem summaries and descriptor controls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .latent_math import cosine_similarity, euclidean_distance, latent_summary
from .schema import LatentItem


Metric = Literal["cosine", "euclidean"]


def control_distance(
    descriptors: Mapping[str, float],
    target: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
    missing: str = "ignore",
) -> float:
    """Weighted descriptor-target distance for memory selection."""

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
    """Convert descriptor-target distance into a higher-is-better score."""

    return -control_distance(descriptors, target, weights=weights, missing=missing)


@dataclass(slots=True)
class SearchResult:
    """One ranked latent-memory retrieval result."""

    item: LatentItem
    score: float
    distance: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return self.item.item_id


class LatentMemoryIndex:
    """A small in-memory nearest-neighbor index for latent audio items."""

    def __init__(self, items: list[LatentItem] | None = None) -> None:
        self._items: list[LatentItem] = []
        self._summary_matrix: np.ndarray | None = None
        if items:
            for item in items:
                self.add(item)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[LatentItem, ...]:
        return tuple(self._items)

    def add(self, item: LatentItem) -> None:
        if any(existing.item_id == item.item_id for existing in self._items):
            raise ValueError(f"duplicate item_id {item.item_id!r}")
        self._items.append(item)
        self._summary_matrix = None

    def get(self, item_id: str) -> LatentItem:
        for item in self._items:
            if item.item_id == item_id:
                return item
        raise KeyError(item_id)

    def _matrix(self) -> np.ndarray:
        if not self._items:
            raise ValueError("index is empty")
        if self._summary_matrix is None:
            self._summary_matrix = np.stack([latent_summary(item) for item in self._items])
        return self._summary_matrix

    def query(
        self,
        query: LatentItem | np.ndarray,
        *,
        top_k: int = 5,
        metric: Metric = "cosine",
        exclude_id: str | None = None,
    ) -> list[SearchResult]:
        """Return nearest latent memories for a query item or latent array."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        q = latent_summary(query) if isinstance(query, LatentItem) or np.asarray(query).ndim == 2 else np.asarray(query, dtype=np.float32)
        matrix = self._matrix()
        if q.shape != matrix.shape[1:]:
            q = q.reshape(-1)
        if q.shape[0] != matrix.shape[1]:
            raise ValueError(f"query summary has dim {q.shape[0]}, index expects {matrix.shape[1]}")

        results: list[SearchResult] = []
        for item, vector in zip(self._items, matrix):
            if exclude_id is not None and item.item_id == exclude_id:
                continue
            distance = euclidean_distance(q, vector)
            if metric == "cosine":
                score = cosine_similarity(q, vector)
            elif metric == "euclidean":
                score = -distance
            else:
                raise ValueError("metric must be 'cosine' or 'euclidean'")
            results.append(
                SearchResult(
                    item=item,
                    score=score,
                    distance=distance,
                    components={"latent_score": score, "latent_distance": distance},
                )
            )
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def query_controls(
        self,
        target: dict[str, float],
        *,
        top_k: int = 5,
        weights: dict[str, float] | None = None,
    ) -> list[SearchResult]:
        """Rank memories by descriptor closeness to a target control vector."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        results = []
        for item in self._items:
            score = control_score(item.descriptors, target, weights=weights)
            results.append(SearchResult(item=item, score=score, distance=-score, components={"control_score": score}))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def query_hybrid(
        self,
        query: LatentItem | np.ndarray,
        *,
        target_controls: dict[str, float] | None = None,
        top_k: int = 5,
        latent_weight: float = 1.0,
        control_weight: float = 1.0,
        exclude_id: str | None = None,
    ) -> list[SearchResult]:
        """Rank by latent similarity plus optional descriptor target score."""

        base = self.query(query, top_k=len(self._items), metric="cosine", exclude_id=exclude_id)
        results: list[SearchResult] = []
        for result in base:
            latent_score = result.components["latent_score"]
            ctrl_score = 0.0
            if target_controls:
                ctrl_score = control_score(result.item.descriptors, target_controls)
            score = latent_weight * latent_score + control_weight * ctrl_score
            components = dict(result.components)
            components["control_score"] = ctrl_score
            results.append(SearchResult(item=result.item, score=score, distance=result.distance, components=components))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
