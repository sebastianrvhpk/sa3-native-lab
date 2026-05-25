import numpy as np

from latent_audio_primitives import LatentItem, LatentMemoryIndex


def item(item_id: str, value: float, *, brightness: float = 0.0) -> LatentItem:
    latent = np.full((6, 3), value, dtype=np.float32)
    return LatentItem(
        item_id=item_id,
        latent=latent,
        latent_rate=10.77,
        descriptors={"brightness": brightness},
    )


def test_query_returns_nearest_latent_summary():
    index = LatentMemoryIndex([item("low", 0.0), item("mid", 5.0), item("high", 10.0)])
    query = np.full((6, 3), 5.2, dtype=np.float32)

    results = index.query(query, top_k=2, metric="euclidean")

    assert [result.item_id for result in results] == ["mid", "high"]


def test_query_controls_ranks_descriptor_targets():
    index = LatentMemoryIndex(
        [
            item("dark", 0.0, brightness=0.1),
            item("bright", 1.0, brightness=0.9),
            item("middle", 2.0, brightness=0.5),
        ]
    )

    results = index.query_controls({"brightness": 0.8}, top_k=2)

    assert [result.item_id for result in results] == ["bright", "middle"]


def test_query_hybrid_can_balance_latent_and_control_scores():
    index = LatentMemoryIndex(
        [
            item("near_dark", 0.0, brightness=0.1),
            item("far_bright", 3.0, brightness=0.9),
        ]
    )
    query = np.zeros((6, 3), dtype=np.float32)

    latent_first = index.query_hybrid(query, target_controls={"brightness": 0.9}, control_weight=0.0)
    control_first = index.query_hybrid(query, target_controls={"brightness": 0.9}, latent_weight=0.0)

    assert latent_first[0].item_id == "near_dark"
    assert control_first[0].item_id == "far_bright"
