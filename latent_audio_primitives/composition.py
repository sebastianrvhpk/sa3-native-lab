from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

from .latent_math import boundary_summary, euclidean_distance
from .schema import LatentItem


@dataclass(frozen=True, slots=True)
class TransitionWeights:
    state: float = 1.0
    velocity: float = 0.25


def transition_cost(
    source: LatentItem,
    target: LatentItem,
    *,
    k: int = 8,
    weights: TransitionWeights | None = None,
) -> float:
    """Cost of placing ``target`` immediately after ``source``."""

    weights = weights or TransitionWeights()
    source_state, source_velocity = boundary_summary(source, "end", k)
    target_state, target_velocity = boundary_summary(target, "start", k)
    return (
        weights.state * euclidean_distance(source_state, target_state)
        + weights.velocity * euclidean_distance(source_velocity, target_velocity)
    )


def loop_cost(
    item: LatentItem,
    *,
    k: int = 8,
    weights: TransitionWeights | None = None,
) -> float:
    """Cost of treating an item as a loop by matching its end to its start."""

    weights = weights or TransitionWeights()
    start_state, start_velocity = boundary_summary(item, "start", k)
    end_state, end_velocity = boundary_summary(item, "end", k)
    return (
        weights.state * euclidean_distance(start_state, end_state)
        + weights.velocity * euclidean_distance(start_velocity, end_velocity)
    )


def bridge_cost(
    start: LatentItem,
    bridge: LatentItem,
    end: LatentItem,
    *,
    k: int = 8,
    weights: TransitionWeights | None = None,
) -> float:
    """Cost of using ``bridge`` to connect ``start`` to ``end``."""

    return transition_cost(start, bridge, k=k, weights=weights) + transition_cost(bridge, end, k=k, weights=weights)


def ranked_continuations(
    source: LatentItem,
    candidates: list[LatentItem] | tuple[LatentItem, ...],
    *,
    k: int = 8,
    weights: TransitionWeights | None = None,
    exclude_self: bool = True,
) -> list[tuple[LatentItem, float]]:
    """Rank candidates by boundary compatibility after ``source``."""

    ranked = []
    for candidate in candidates:
        if exclude_self and candidate.item_id == source.item_id:
            continue
        ranked.append((candidate, transition_cost(source, candidate, k=k, weights=weights)))
    ranked.sort(key=lambda pair: pair[1])
    return ranked


def ranked_bridges(
    start: LatentItem,
    end: LatentItem,
    candidates: list[LatentItem] | tuple[LatentItem, ...],
    *,
    k: int = 8,
    weights: TransitionWeights | None = None,
) -> list[tuple[LatentItem, float]]:
    """Rank candidates by how well they bridge ``start`` and ``end``."""

    ranked = []
    for candidate in candidates:
        if candidate.item_id in {start.item_id, end.item_id}:
            continue
        ranked.append((candidate, bridge_cost(start, candidate, end, k=k, weights=weights)))
    ranked.sort(key=lambda pair: pair[1])
    return ranked


def best_path(
    items: list[LatentItem] | tuple[LatentItem, ...],
    start_id: str,
    end_id: str,
    *,
    k: int = 8,
    weights: TransitionWeights | None = None,
    max_hops: int | None = None,
    max_edge_cost: float | None = None,
) -> tuple[list[LatentItem], float]:
    """Find a low-cost directed latent-memory path between two items.

    This is a dense all-to-all Dijkstra search, appropriate for small research
    collections. Larger memories should build an approximate candidate graph.
    """

    by_id = {item.item_id: item for item in items}
    if start_id not in by_id:
        raise KeyError(start_id)
    if end_id not in by_id:
        raise KeyError(end_id)
    if max_hops is not None and max_hops < 1:
        raise ValueError("max_hops must be at least 1 when provided")

    heap: list[tuple[float, int, str, list[str]]] = [(0.0, 0, start_id, [start_id])]
    best: dict[tuple[str, int], float] = {(start_id, 0): 0.0}

    while heap:
        cost, hops, current_id, path = heappop(heap)
        if current_id == end_id and hops > 0:
            return [by_id[item_id] for item_id in path], cost
        if max_hops is not None and hops >= max_hops:
            continue

        current = by_id[current_id]
        for candidate in items:
            if candidate.item_id == current_id:
                continue
            if candidate.item_id in path and candidate.item_id != end_id:
                continue
            edge = transition_cost(current, candidate, k=k, weights=weights)
            if max_edge_cost is not None and edge > max_edge_cost:
                continue
            next_cost = cost + edge
            next_hops = hops + 1
            key = (candidate.item_id, next_hops)
            if next_cost >= best.get(key, float("inf")):
                continue
            best[key] = next_cost
            heappush(heap, (next_cost, next_hops, candidate.item_id, path + [candidate.item_id]))

    raise ValueError(f"no path found from {start_id!r} to {end_id!r}")
