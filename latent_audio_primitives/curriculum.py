"""Dataset-memory clustering and heldout selection for notebook curricula."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .index import LatentMemoryIndex
from .latent_math import latent_summary
from .schema import LatentItem


@dataclass(frozen=True, slots=True)
class MemoryCurriculumCluster:
    """A cluster of latent-memory items for prompt/control experiment design."""

    cluster_id: int
    item_ids: list[str]
    representative_id: str
    prompt_seed: str
    descriptor_means: dict[str, float] = field(default_factory=dict)
    label_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "item_ids": list(self.item_ids),
            "representative_id": self.representative_id,
            "prompt_seed": self.prompt_seed,
            "descriptor_means": dict(self.descriptor_means),
            "label_counts": {key: dict(value) for key, value in self.label_counts.items()},
            "centroid": self.centroid.astype(float).tolist(),
        }


def build_memory_curriculum(
    items: Sequence[LatentItem],
    *,
    cluster_count: int = 4,
    seed: int = 0,
    iterations: int = 25,
    prompt_token_limit: int = 8,
) -> list[MemoryCurriculumCluster]:
    """Cluster latent memory items and summarize each cluster for experiments."""

    items = list(items)
    if not items:
        raise ValueError("at least one item is required")
    clusters = cluster_latent_items(items, cluster_count=cluster_count, seed=seed, iterations=iterations)
    summaries = np.stack([latent_summary(item) for item in items]).astype(np.float32)
    by_id = {item.item_id: item for item in items}
    item_index = {item.item_id: index for index, item in enumerate(items)}
    out: list[MemoryCurriculumCluster] = []
    for cluster_id, item_ids in enumerate(clusters):
        cluster_items = [by_id[item_id] for item_id in item_ids]
        indices = [item_index[item_id] for item_id in item_ids]
        centroid = summaries[indices].mean(axis=0)
        representative_id = min(item_ids, key=lambda item_id: float(np.linalg.norm(summaries[item_index[item_id]] - centroid)))
        out.append(
            MemoryCurriculumCluster(
                cluster_id=cluster_id,
                item_ids=list(item_ids),
                representative_id=representative_id,
                prompt_seed=_cluster_prompt_seed(cluster_items, token_limit=prompt_token_limit),
                descriptor_means=_descriptor_means(cluster_items),
                label_counts=_label_counts(cluster_items),
                centroid=centroid.astype(np.float32),
            )
        )
    return out


def cluster_latent_items(
    items: Sequence[LatentItem],
    *,
    cluster_count: int = 4,
    seed: int = 0,
    iterations: int = 25,
) -> list[list[str]]:
    """Small deterministic k-means over latent summaries."""

    items = list(items)
    if not items:
        raise ValueError("at least one item is required")
    cluster_count = max(1, min(int(cluster_count), len(items)))
    x = np.stack([latent_summary(item) for item in items]).astype(np.float32)
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(items), size=cluster_count, replace=False)].copy()
    assignments = np.zeros(len(items), dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_assignments = np.argmin(distances, axis=1)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        for cluster_id in range(cluster_count):
            mask = assignments == cluster_id
            if np.any(mask):
                centers[cluster_id] = x[mask].mean(axis=0)
    clusters: list[list[str]] = []
    for cluster_id in range(cluster_count):
        ids = [item.item_id for item, assigned in zip(items, assignments) if assigned == cluster_id]
        if ids:
            clusters.append(ids)
    clusters.sort(key=lambda ids: ids[0])
    return clusters


def nearest_memory_rows(
    query: LatentItem | np.ndarray,
    items: Sequence[LatentItem],
    *,
    top_k: int = 5,
    exclude_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return JSON/table-friendly nearest latent-memory rows."""

    index = LatentMemoryIndex(list(items))
    rows = []
    for result in index.query(query, top_k=top_k, exclude_id=exclude_id):
        rows.append(
            {
                "item_id": result.item_id,
                "score": float(result.score),
                "distance": float(result.distance),
                "prompt": result.item.prompt or "",
                "duration_seconds": result.item.duration_seconds,
                "descriptors": dict(result.item.descriptors),
                "metadata": dict(result.item.metadata),
            }
        )
    return rows


def heldout_split_item_ids(
    clusters: Sequence[MemoryCurriculumCluster],
    *,
    holdout_fraction: float = 0.2,
) -> dict[str, list[str]]:
    """Create deterministic train/heldout item ids, stratified by cluster."""

    holdout_fraction = min(max(float(holdout_fraction), 0.0), 1.0)
    train: list[str] = []
    heldout: list[str] = []
    for cluster in clusters:
        ids = sorted(cluster.item_ids)
        holdout_count = int(round(len(ids) * holdout_fraction))
        if len(ids) > 1 and holdout_fraction > 0:
            holdout_count = max(1, holdout_count)
        heldout.extend(ids[:holdout_count])
        train.extend(ids[holdout_count:])
    return {"train": train, "heldout": heldout}


def _cluster_prompt_seed(items: Sequence[LatentItem], *, token_limit: int) -> str:
    tokens: list[str] = []
    for item in items:
        tokens.extend(_tokens(item.prompt or ""))
        for key in ("path", "relpath", "source_audio_path", "filename"):
            value = item.metadata.get(key)
            if value:
                tokens.extend(_tokens(str(value)))
        for key, value in item.labels.items():
            if isinstance(value, (str, int, float)):
                tokens.extend(_tokens(str(value)))
            elif isinstance(value, (list, tuple)):
                for entry in value:
                    tokens.extend(_tokens(str(entry)))
    counts = Counter(token for token in tokens if len(token) > 2)
    if not counts:
        return "audio texture"
    ranked = [token for token, _count in counts.most_common(max(1, int(token_limit)))]
    return " ".join(ranked)


def _descriptor_means(items: Sequence[LatentItem]) -> dict[str, float]:
    keys = sorted({key for item in items for key in item.descriptors})
    means = {}
    for key in keys:
        values = [item.descriptors[key] for item in items if key in item.descriptors]
        if values:
            means[key] = float(np.mean(values))
    return means


def _label_counts(items: Sequence[LatentItem]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = {}
    for item in items:
        for key, value in item.labels.items():
            counter = out.setdefault(str(key), Counter())
            if isinstance(value, (list, tuple, set)):
                for entry in value:
                    counter[str(entry)] += 1
            else:
                counter[str(value)] += 1
    return {key: dict(counter) for key, counter in out.items()}


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", value.lower())
