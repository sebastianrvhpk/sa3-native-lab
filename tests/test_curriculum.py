from __future__ import annotations

import numpy as np

from latent_audio_primitives.curriculum import (
    build_memory_curriculum,
    cluster_latent_items,
    heldout_split_item_ids,
    nearest_memory_rows,
)
from latent_audio_primitives.schema import LatentItem


def _item(item_id: str, value: float, *, prompt: str = "", brightness: float = 0.0) -> LatentItem:
    return LatentItem(
        item_id=item_id,
        latent=np.full((5, 3), value, dtype=np.float32),
        latent_rate=10.0,
        prompt=prompt,
        descriptors={"brightness": brightness},
        labels={"family": "bright" if brightness > 0.5 else "dark"},
        metadata={"path": f"/dataset/{item_id}.wav"},
    )


def test_cluster_latent_items_groups_near_summaries():
    items = [
        _item("a", 0.0),
        _item("b", 0.1),
        _item("c", 10.0),
        _item("d", 10.1),
    ]

    clusters = cluster_latent_items(items, cluster_count=2, seed=0)
    sets = [set(cluster) for cluster in clusters]

    assert {"a", "b"} in sets
    assert {"c", "d"} in sets


def test_build_memory_curriculum_summarizes_clusters_and_split():
    items = [
        _item("dark_a", 0.0, prompt="dark tape loop", brightness=0.1),
        _item("dark_b", 0.2, prompt="dark noise loop", brightness=0.2),
        _item("bright_a", 8.0, prompt="bright glass tone", brightness=0.9),
        _item("bright_b", 8.2, prompt="bright bell tone", brightness=0.8),
    ]

    clusters = build_memory_curriculum(items, cluster_count=2, seed=2)
    split = heldout_split_item_ids(clusters, holdout_fraction=0.5)

    assert len(clusters) == 2
    assert all(cluster.prompt_seed for cluster in clusters)
    assert sum(len(cluster.item_ids) for cluster in clusters) == 4
    assert len(split["heldout"]) == 2
    assert len(split["train"]) == 2


def test_nearest_memory_rows_are_table_friendly():
    items = [_item("near", 1.0, prompt="near"), _item("far", 9.0, prompt="far")]
    query = np.full((5, 3), 1.2, dtype=np.float32)

    rows = nearest_memory_rows(query, items, top_k=1)

    assert rows[0]["item_id"] == "near"
    assert rows[0]["prompt"] == "near"
    assert "duration_seconds" in rows[0]
