import numpy as np

from latent_audio_primitives import (
    LatentItem,
    best_path,
    bridge_cost,
    loop_cost,
    ranked_bridges,
    ranked_continuations,
    transition_cost,
)


def ramp_item(item_id: str, start: float, stop: float, frames: int = 5) -> LatentItem:
    values = np.linspace(start, stop, frames, dtype=np.float32)[:, None]
    latent = np.repeat(values, repeats=2, axis=1)
    return LatentItem(item_id=item_id, latent=latent, latent_rate=10.77)


def test_transition_cost_prefers_matching_boundaries():
    source = ramp_item("source", 0.0, 1.0)
    good_next = ramp_item("good", 1.0, 2.0)
    bad_next = ramp_item("bad", 10.0, 11.0)

    assert transition_cost(source, good_next, k=1) < transition_cost(source, bad_next, k=1)


def test_ranked_continuations_orders_by_boundary_cost():
    source = ramp_item("source", 0.0, 1.0)
    good_next = ramp_item("good", 1.0, 2.0)
    bad_next = ramp_item("bad", 10.0, 11.0)

    ranked = ranked_continuations(source, [bad_next, good_next], k=1)

    assert [candidate.item_id for candidate, _ in ranked] == ["good", "bad"]


def test_loop_cost_prefers_matching_start_and_end():
    loopable = LatentItem(
        item_id="loop",
        latent=np.array([[0.0], [1.0], [0.0]], dtype=np.float32),
        latent_rate=10.77,
    )
    one_way = LatentItem(
        item_id="one_way",
        latent=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        latent_rate=10.77,
    )

    assert loop_cost(loopable, k=1) < loop_cost(one_way, k=1)


def test_bridge_cost_and_ranking_prefers_middle_candidate():
    start = ramp_item("start", 0.0, 1.0)
    bridge = ramp_item("bridge", 1.0, 2.0)
    bad_bridge = ramp_item("bad_bridge", 10.0, 11.0)
    end = ramp_item("end", 2.0, 3.0)

    assert bridge_cost(start, bridge, end, k=1) < bridge_cost(start, bad_bridge, end, k=1)
    ranked = ranked_bridges(start, end, [bad_bridge, bridge], k=1)

    assert ranked[0][0].item_id == "bridge"


def test_best_path_uses_intermediate_when_direct_boundary_is_worse():
    start = ramp_item("start", 0.0, 1.0)
    middle = ramp_item("middle", 1.0, 2.0)
    end = ramp_item("end", 2.0, 3.0)
    distractor = ramp_item("distractor", 50.0, 51.0)

    path, cost = best_path([start, middle, end, distractor], "start", "end", k=1)

    assert [item.item_id for item in path] == ["start", "middle", "end"]
    assert cost == 0.0
