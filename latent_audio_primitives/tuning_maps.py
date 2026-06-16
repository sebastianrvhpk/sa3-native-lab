"""Relational tuning-map inference from audio pitch evidence.

The core object here is not a named scale. A ``TuningMap`` is a compact,
JSON-friendly explanation of observed pitch events: centers, interval edges,
low-integer ratio hypotheses, generator fits, period candidates, and optional
Wilson-style CPS fits. It is a microscope/selector for later SAME/SA3 probes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from itertools import combinations
from math import gcd, isfinite, log2
from typing import Any, Mapping, Sequence

import numpy as np

from .tuning_systems import PitchTrackConfig, pitch_track_rows, ratio_to_cents


DEFAULT_PRIMES = (2, 3, 5, 7, 11, 13)
TRITAVE_CENTS = 1200.0 * log2(3.0)


@dataclass(frozen=True, slots=True)
class RatioCandidate:
    """One low-integer ratio explanation for an observed interval."""

    ratio: str
    cents: float
    error_cents: float
    numerator: int
    denominator: int
    tenney_height: float
    prime_limit: int
    monzo: Mapping[str, int]
    score: float

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["monzo"] = dict(self.monzo)
        return row


@dataclass(frozen=True, slots=True)
class PitchEvent:
    """A stable-enough pitch event segmented from framewise f0 evidence."""

    event_id: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    median_hz: float
    median_cents: float
    std_cents: float
    confidence_mean: float
    frame_count: int
    weight: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PitchCenter:
    """Clustered pitch center in a folded period."""

    center_id: str
    cents: float
    absolute_cents_mean: float
    median_hz: float
    event_count: int
    frame_count: int
    weight: float
    spread_cents: float
    event_ids: Sequence[str] = ()

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["event_ids"] = list(self.event_ids)
        return row


@dataclass(frozen=True, slots=True)
class IntervalEdge:
    """Directed interval relation between two pitch centers."""

    source_center_id: str
    target_center_id: str
    interval_cents: float
    folded_distance_cents: float
    event_weight: float
    best_ratio: str | None
    best_error_cents: float | None
    harmonic_entropy_proxy: float | None
    ratio_candidates: Sequence[RatioCandidate] = ()

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["ratio_candidates"] = [candidate.to_row() for candidate in self.ratio_candidates]
        return row


@dataclass(frozen=True, slots=True)
class GeneratorCandidate:
    """Period-plus-generator fit for observed pitch centers."""

    generator_cents: float
    period_cents: float
    mean_abs_error_cents: float
    max_abs_error_cents: float
    within_tolerance_ratio: float
    used_step_count: int
    coordinates_by_center: Mapping[str, int]
    score: float

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["coordinates_by_center"] = dict(self.coordinates_by_center)
        return row


@dataclass(frozen=True, slots=True)
class PeriodCandidate:
    """Summary row for one hypothesized period."""

    period_cents: float
    center_count: int
    edge_count: int
    ratio_fit_error_cents: float | None
    ratio_fit_coverage: float
    generator_cents: float | None
    generator_error_cents: float | None
    cohesion_score: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CPSFit:
    """Fit of observed pitch centers to a Wilson-style combination product set."""

    name: str
    factors: Sequence[str]
    choose: int
    period_cents: float
    vertex_count: int
    best_root_offset_cents: float
    mean_abs_error_cents: float | None
    max_abs_error_cents: float | None
    coverage_ratio: float
    matched_vertex_count: int
    score: float

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["factors"] = list(self.factors)
        return row


@dataclass(frozen=True, slots=True)
class TuningMapConfig:
    """Inference settings for dependency-light tuning cartography."""

    pitch_config: PitchTrackConfig = field(default_factory=PitchTrackConfig)
    reference_hz: float | None = None
    period_candidates_cents: Sequence[float] = (1200.0, TRITAVE_CENTS)
    event_gap_seconds: float = 0.08
    event_jump_cents: float = 80.0
    min_event_duration_seconds: float = 0.06
    min_event_frames: int = 3
    center_tolerance_cents: float = 28.0
    ratio_max_numerator: int = 31
    ratio_max_denominator: int = 31
    ratio_prime_limit: int = 13
    ratio_top_k: int = 5
    ratio_error_tolerance_cents: float = 35.0
    ratio_complexity_weight: float = 1.5
    entropy_sigma_cents: float = 18.0
    generator_min_cents: float = 40.0
    generator_step_cents: float = 2.5
    generator_max_steps: int = 36
    generator_top_k: int = 6
    generator_coordinate_weight: float = 0.12
    max_centers: int = 24


@dataclass(frozen=True, slots=True)
class TuningMap:
    """Relational pitch-organization evidence inferred from one audio item."""

    reference_hz: float
    selected_period_cents: float
    pitch_events: Sequence[PitchEvent]
    pitch_centers: Sequence[PitchCenter]
    interval_edges: Sequence[IntervalEdge]
    generator_candidates: Sequence[GeneratorCandidate]
    period_candidates: Sequence[PeriodCandidate]
    cps_fits: Sequence[CPSFit]
    summary: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "reference_hz": float(self.reference_hz),
            "selected_period_cents": float(self.selected_period_cents),
            "pitch_events": [event.to_row() for event in self.pitch_events],
            "pitch_centers": [center.to_row() for center in self.pitch_centers],
            "interval_edges": [edge.to_row() for edge in self.interval_edges],
            "generator_candidates": [candidate.to_row() for candidate in self.generator_candidates],
            "period_candidates": [candidate.to_row() for candidate in self.period_candidates],
            "cps_fits": [fit.to_row() for fit in self.cps_fits],
            "summary": dict(self.summary),
            "metadata": dict(self.metadata),
        }


def infer_tuning_map(
    audio: Any,
    sample_rate: int,
    *,
    config: TuningMapConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TuningMap:
    """Infer a relational tuning map from audio.

    The method is intentionally dependency-light. It is a first microscope, not
    a replacement for specialist pitch trackers or musicological judgement.
    """

    cfg = config or TuningMapConfig()
    rows = pitch_track_rows(audio, sample_rate, config=cfg.pitch_config)
    reference_hz = _resolve_reference_hz(rows, cfg.reference_hz)
    events = pitch_events_from_rows(rows, reference_hz=reference_hz, config=cfg)
    period_summaries: list[tuple[PeriodCandidate, list[PitchCenter], list[IntervalEdge], list[GeneratorCandidate]]] = []
    for period in cfg.period_candidates_cents:
        centers = pitch_centers_from_events(events, period_cents=float(period), config=cfg)
        edges = interval_edges_from_centers(centers, period_cents=float(period), config=cfg)
        generators = generator_candidates_from_centers(centers, period_cents=float(period), config=cfg)
        period_row = _period_candidate_from_parts(float(period), centers, edges, generators)
        period_summaries.append((period_row, centers, edges, generators))

    if period_summaries:
        selected_period, centers, edges, generators = min(period_summaries, key=lambda item: item[0].cohesion_score)
    else:
        selected_period = PeriodCandidate(
            period_cents=1200.0,
            center_count=0,
            edge_count=0,
            ratio_fit_error_cents=None,
            ratio_fit_coverage=0.0,
            generator_cents=None,
            generator_error_cents=None,
            cohesion_score=float("inf"),
        )
        centers = []
        edges = []
        generators = []

    cps_fits = cps_fit_rows(centers, period_cents=selected_period.period_cents, config=cfg)
    summary = tuning_map_summary(
        events=events,
        centers=centers,
        edges=edges,
        generators=generators,
        period_candidate=selected_period,
        cps_fits=cps_fits,
    )
    return TuningMap(
        reference_hz=reference_hz,
        selected_period_cents=selected_period.period_cents,
        pitch_events=events,
        pitch_centers=centers,
        interval_edges=edges,
        generator_candidates=generators,
        period_candidates=[row for row, _centers, _edges, _generators in period_summaries],
        cps_fits=cps_fits,
        summary=summary,
        metadata=dict(metadata or {}),
    )


def pitch_events_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_hz: float,
    config: TuningMapConfig | None = None,
) -> list[PitchEvent]:
    """Segment voiced f0 rows into stable-ish pitch events."""

    cfg = config or TuningMapConfig()
    voiced = [row for row in rows if row.get("f0_hz") is not None and _finite_positive(row.get("f0_hz"))]
    if not voiced:
        return []
    hop = _median_hop_seconds(voiced)
    events: list[PitchEvent] = []
    current: list[Mapping[str, Any]] = []
    prev_time = None
    prev_cents = None
    for row in voiced:
        time = float(row.get("time_seconds", 0.0))
        cents = 1200.0 * log2(float(row["f0_hz"]) / reference_hz)
        split = False
        if current and prev_time is not None and time - prev_time > cfg.event_gap_seconds + hop:
            split = True
        if current and prev_cents is not None and abs(cents - prev_cents) > cfg.event_jump_cents:
            split = True
        if split:
            event = _event_from_rows(current, reference_hz=reference_hz, event_index=len(events), hop_seconds=hop)
            if _keep_event(event, cfg):
                events.append(event)
            current = []
        current.append(row)
        prev_time = time
        prev_cents = cents
    if current:
        event = _event_from_rows(current, reference_hz=reference_hz, event_index=len(events), hop_seconds=hop)
        if _keep_event(event, cfg):
            events.append(event)
    return events


def pitch_centers_from_events(
    events: Sequence[PitchEvent],
    *,
    period_cents: float,
    config: TuningMapConfig | None = None,
) -> list[PitchCenter]:
    """Cluster pitch events into folded pitch centers."""

    cfg = config or TuningMapConfig()
    if period_cents <= 0:
        raise ValueError("period_cents must be positive")
    if not events:
        return []
    values = [
        {
            "event": event,
            "folded": event.median_cents % period_cents,
            "weight": max(float(event.weight), 1e-6),
        }
        for event in events
    ]
    values.sort(key=lambda row: row["folded"])
    clusters: list[list[dict[str, Any]]] = []
    for row in values:
        if not clusters:
            clusters.append([row])
            continue
        if abs(float(row["folded"]) - float(clusters[-1][-1]["folded"])) <= cfg.center_tolerance_cents:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    if len(clusters) > 1:
        first = clusters[0]
        last = clusters[-1]
        wrap_gap = float(first[0]["folded"]) + period_cents - float(last[-1]["folded"])
        if wrap_gap <= cfg.center_tolerance_cents:
            merged = last + first
            clusters = [merged] + clusters[1:-1]

    centers = [_center_from_cluster(cluster, period_cents=period_cents, center_index=index) for index, cluster in enumerate(clusters)]
    centers.sort(key=lambda center: center.weight, reverse=True)
    centers = centers[: max(1, int(cfg.max_centers))]
    centers.sort(key=lambda center: center.cents)
    return [
        PitchCenter(
            center_id=f"center_{index:02d}",
            cents=center.cents,
            absolute_cents_mean=center.absolute_cents_mean,
            median_hz=center.median_hz,
            event_count=center.event_count,
            frame_count=center.frame_count,
            weight=center.weight,
            spread_cents=center.spread_cents,
            event_ids=center.event_ids,
        )
        for index, center in enumerate(centers)
    ]


def interval_edges_from_centers(
    centers: Sequence[PitchCenter],
    *,
    period_cents: float,
    config: TuningMapConfig | None = None,
) -> list[IntervalEdge]:
    """Build pairwise interval edges between folded pitch centers."""

    cfg = config or TuningMapConfig()
    edges: list[IntervalEdge] = []
    for source, target in combinations(centers, 2):
        forward = (target.cents - source.cents) % period_cents
        reverse = (source.cents - target.cents) % period_cents
        interval = forward if forward <= reverse else reverse
        if interval <= 1e-6:
            continue
        candidates = ratio_candidates_for_cents(
            interval,
            period_cents=period_cents,
            max_numerator=cfg.ratio_max_numerator,
            max_denominator=cfg.ratio_max_denominator,
            prime_limit=cfg.ratio_prime_limit,
            top_k=cfg.ratio_top_k,
            complexity_weight=cfg.ratio_complexity_weight,
        )
        best = candidates[0] if candidates else None
        entropy = harmonic_entropy_proxy(interval, period_cents=period_cents, config=cfg)
        edges.append(
            IntervalEdge(
                source_center_id=source.center_id,
                target_center_id=target.center_id,
                interval_cents=float(interval),
                folded_distance_cents=float(min(forward, reverse)),
                event_weight=float(min(source.weight, target.weight)),
                best_ratio=None if best is None else best.ratio,
                best_error_cents=None if best is None else float(best.error_cents),
                harmonic_entropy_proxy=entropy,
                ratio_candidates=tuple(candidates),
            )
        )
    edges.sort(key=lambda edge: (edge.best_error_cents if edge.best_error_cents is not None else float("inf"), -edge.event_weight))
    return edges


def ratio_candidates_for_cents(
    cents: float,
    *,
    period_cents: float = 1200.0,
    max_numerator: int = 31,
    max_denominator: int = 31,
    prime_limit: int = 13,
    top_k: int = 5,
    complexity_weight: float = 1.5,
) -> list[RatioCandidate]:
    """Return low-integer ratio candidates for an interval in cents."""

    if period_cents <= 0:
        raise ValueError("period_cents must be positive")
    candidates: list[RatioCandidate] = []
    period_fraction = _period_fraction_for_cents(period_cents)
    seen: set[Fraction] = set()
    for numerator in range(1, max(1, int(max_numerator)) + 1):
        for denominator in range(1, max(1, int(max_denominator)) + 1):
            if numerator == denominator or gcd(numerator, denominator) != 1:
                continue
            ratio = Fraction(numerator, denominator)
            if period_fraction is not None:
                ratio = _fold_fraction_to_period(ratio, period=period_fraction)
            if ratio == 1 or ratio in seen:
                continue
            seen.add(ratio)
            monzo = ratio_monzo(ratio.numerator, ratio.denominator, primes=DEFAULT_PRIMES)
            if monzo["unfactored_numerator"] != 1 or monzo["unfactored_denominator"] != 1:
                continue
            ratio_prime_limit = max((prime for prime, exp in monzo.items() if isinstance(prime, int) and exp), default=1)
            if ratio_prime_limit > prime_limit:
                continue
            ratio_cents = ratio_to_cents(ratio) % period_cents
            if ratio_cents <= 1e-6:
                continue
            error = _circular_cents_distance_scalar(float(cents), ratio_cents, period_cents)
            tenney = log2(ratio.numerator * ratio.denominator)
            score = float(error + complexity_weight * tenney)
            candidates.append(
                RatioCandidate(
                    ratio=_fraction_label(ratio),
                    cents=float(ratio_cents),
                    error_cents=float(error),
                    numerator=int(ratio.numerator),
                    denominator=int(ratio.denominator),
                    tenney_height=float(tenney),
                    prime_limit=int(ratio_prime_limit),
                    monzo={str(key): int(value) for key, value in monzo.items() if isinstance(key, int) and value},
                    score=score,
                )
            )
    candidates.sort(key=lambda candidate: candidate.score)
    return candidates[: max(1, int(top_k))]


def harmonic_entropy_proxy(
    cents: float,
    *,
    period_cents: float = 1200.0,
    config: TuningMapConfig | None = None,
) -> float | None:
    """Erlich-inspired uncertainty over nearby low-integer ratios.

    This is a practical row value, not a faithful full harmonic-entropy model.
    Low values mean one/few simple ratios dominate; high values mean the
    interval is spread across many complex or distant candidates.
    """

    cfg = config or TuningMapConfig()
    candidates = ratio_candidates_for_cents(
        cents,
        period_cents=period_cents,
        max_numerator=cfg.ratio_max_numerator,
        max_denominator=cfg.ratio_max_denominator,
        prime_limit=cfg.ratio_prime_limit,
        top_k=64,
        complexity_weight=0.0,
    )
    if not candidates:
        return None
    weights = []
    sigma = max(float(cfg.entropy_sigma_cents), 1e-6)
    for candidate in candidates:
        proximity = np.exp(-0.5 * (candidate.error_cents / sigma) ** 2)
        simplicity = 1.0 / max(candidate.tenney_height, 1.0)
        weights.append(float(proximity * simplicity))
    probs = np.asarray(weights, dtype=np.float64)
    total = float(probs.sum())
    if total <= 1e-12:
        return None
    probs = probs / total
    entropy = -float(np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
    return float(entropy / max(log2(len(probs)), 1e-12))


def generator_candidates_from_centers(
    centers: Sequence[PitchCenter],
    *,
    period_cents: float,
    config: TuningMapConfig | None = None,
) -> list[GeneratorCandidate]:
    """Fit period-plus-generator coordinates to observed centers."""

    cfg = config or TuningMapConfig()
    if len(centers) < 2:
        return []
    max_generator = max(cfg.generator_min_cents, period_cents / 2.0)
    generator_values = np.arange(cfg.generator_min_cents, max_generator + 1e-9, max(cfg.generator_step_cents, 0.1))
    out: list[GeneratorCandidate] = []
    for generator in generator_values:
        errors = []
        coords: dict[str, int] = {}
        used_steps: set[int] = set()
        abs_steps = []
        for center in centers:
            best_step = 0
            best_error = float("inf")
            for step in range(-cfg.generator_max_steps, cfg.generator_max_steps + 1):
                predicted = (step * float(generator)) % period_cents
                error = _circular_cents_distance_scalar(center.cents, predicted, period_cents)
                if error < best_error:
                    best_error = error
                    best_step = step
            errors.append(best_error)
            coords[center.center_id] = int(best_step)
            used_steps.add(int(best_step))
            abs_steps.append(abs(int(best_step)))
        mean_error = float(np.mean(errors)) if errors else float("inf")
        max_error = float(np.max(errors)) if errors else float("inf")
        within = float(np.mean(np.asarray(errors) <= cfg.center_tolerance_cents)) if errors else 0.0
        mean_abs_step = float(np.mean(abs_steps)) if abs_steps else 0.0
        score = mean_error + 0.05 * max_error + 0.15 * len(used_steps) + cfg.generator_coordinate_weight * mean_abs_step
        out.append(
            GeneratorCandidate(
                generator_cents=float(generator),
                period_cents=float(period_cents),
                mean_abs_error_cents=mean_error,
                max_abs_error_cents=max_error,
                within_tolerance_ratio=within,
                used_step_count=len(used_steps),
                coordinates_by_center=coords,
                score=float(score),
            )
        )
    out.sort(key=lambda candidate: candidate.score)
    return out[: max(1, int(cfg.generator_top_k))]


def ratio_monzo(
    numerator: int,
    denominator: int = 1,
    *,
    primes: Sequence[int] = DEFAULT_PRIMES,
) -> dict[int | str, int]:
    """Return prime-exponent coordinates plus unfactored leftovers."""

    n = int(numerator)
    d = int(denominator)
    out: dict[int | str, int] = {}
    for prime in primes:
        count = 0
        while n % prime == 0 and n > 1:
            count += 1
            n //= prime
        while d % prime == 0 and d > 1:
            count -= 1
            d //= prime
        if count:
            out[int(prime)] = int(count)
    out["unfactored_numerator"] = int(n)
    out["unfactored_denominator"] = int(d)
    return out


def cps_ratios(
    factors: Sequence[int | str | Fraction],
    choose: int,
    *,
    period_ratio: int | str | Fraction = 2,
) -> list[dict[str, Any]]:
    """Return Wilson-style combination product set vertices."""

    if choose <= 0:
        raise ValueError("choose must be positive")
    factor_fracs = [_as_fraction(value) for value in factors]
    period = _as_fraction(period_ratio)
    rows: list[dict[str, Any]] = []
    for combo in combinations(range(len(factor_fracs)), int(choose)):
        value = Fraction(1, 1)
        labels = []
        for index in combo:
            value *= factor_fracs[index]
            labels.append(_fraction_label(factor_fracs[index]))
        folded = _fold_fraction_to_period(value, period=period)
        rows.append(
            {
                "label": "*".join(labels),
                "ratio": _fraction_label(folded),
                "cents": ratio_to_cents(folded),
                "factor_indices": list(combo),
            }
        )
    rows.sort(key=lambda row: float(row["cents"]))
    return rows


def default_cps_specs() -> list[dict[str, Any]]:
    """Starter Wilson-style CPS structures for fit rows."""

    return [
        {"name": "hexany_1_3_5_7", "factors": ("1", "3", "5", "7"), "choose": 2, "period_ratio": "2"},
        {"name": "dekany_1_3_5_7_9", "factors": ("1", "3", "5", "7", "9"), "choose": 2, "period_ratio": "2"},
        {"name": "eikosany_1_3_5_7_9_11", "factors": ("1", "3", "5", "7", "9", "11"), "choose": 3, "period_ratio": "2"},
    ]


def cps_fit_rows(
    centers: Sequence[PitchCenter],
    *,
    period_cents: float,
    config: TuningMapConfig | None = None,
    cps_specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[CPSFit]:
    """Fit observed pitch centers against Wilson-style CPS vertex sets."""

    cfg = config or TuningMapConfig()
    if not centers:
        return []
    specs = list(cps_specs or default_cps_specs())
    out: list[CPSFit] = []
    center_values = np.asarray([center.cents for center in centers], dtype=np.float64)
    for spec in specs:
        vertices = cps_ratios(spec["factors"], int(spec["choose"]), period_ratio=spec.get("period_ratio", 2))
        vertex_values = np.asarray([float(vertex["cents"]) % period_cents for vertex in vertices], dtype=np.float64)
        if vertex_values.size == 0:
            continue
        best_offset = 0.0
        best_errors = None
        best_score = float("inf")
        for offset in np.arange(0.0, period_cents, max(cfg.generator_step_cents, 0.5)):
            shifted = (vertex_values + offset) % period_cents
            errors = np.min(_circular_cents_distance(center_values[:, None], shifted[None, :], period_cents), axis=1)
            score = float(np.mean(errors))
            if score < best_score:
                best_score = score
                best_offset = float(offset)
                best_errors = errors
        if best_errors is None:
            continue
        coverage = float(np.mean(best_errors <= cfg.ratio_error_tolerance_cents))
        matched = int(np.sum(best_errors <= cfg.ratio_error_tolerance_cents))
        score = float(best_score + 2.0 * (1.0 - coverage) + 0.05 * len(vertices))
        out.append(
            CPSFit(
                name=str(spec["name"]),
                factors=tuple(str(value) for value in spec["factors"]),
                choose=int(spec["choose"]),
                period_cents=float(period_cents),
                vertex_count=len(vertices),
                best_root_offset_cents=best_offset,
                mean_abs_error_cents=float(np.mean(best_errors)) if best_errors.size else None,
                max_abs_error_cents=float(np.max(best_errors)) if best_errors.size else None,
                coverage_ratio=coverage,
                matched_vertex_count=matched,
                score=score,
            )
        )
    out.sort(key=lambda fit: fit.score)
    return out


def tuning_map_summary(
    *,
    events: Sequence[PitchEvent],
    centers: Sequence[PitchCenter],
    edges: Sequence[IntervalEdge],
    generators: Sequence[GeneratorCandidate],
    period_candidate: PeriodCandidate,
    cps_fits: Sequence[CPSFit],
) -> dict[str, Any]:
    """Return a compact high-level tuning-map summary."""

    ratio_errors = [edge.best_error_cents for edge in edges if edge.best_error_cents is not None]
    entropy_values = [edge.harmonic_entropy_proxy for edge in edges if edge.harmonic_entropy_proxy is not None]
    strong_edges = [edge for edge in edges if edge.best_error_cents is not None and edge.best_error_cents <= 20.0]
    best_generator = generators[0] if generators else None
    best_cps = cps_fits[0] if cps_fits else None
    return {
        "claim_maturity": "microscope",
        "pitch_event_count": len(events),
        "pitch_center_count": len(centers),
        "interval_edge_count": len(edges),
        "selected_period_cents": float(period_candidate.period_cents),
        "cohesion_score": float(period_candidate.cohesion_score),
        "ratio_fit_error_cents": _mean_or_none(ratio_errors),
        "ratio_fit_coverage_35c": float(np.mean(np.asarray(ratio_errors) <= 35.0)) if ratio_errors else 0.0,
        "strong_low_integer_edge_count": len(strong_edges),
        "harmonic_entropy_proxy_mean": _mean_or_none(entropy_values),
        "best_generator_cents": None if best_generator is None else float(best_generator.generator_cents),
        "best_generator_error_cents": None if best_generator is None else float(best_generator.mean_abs_error_cents),
        "best_cps_name": None if best_cps is None else best_cps.name,
        "best_cps_error_cents": None if best_cps is None else best_cps.mean_abs_error_cents,
        "decision_hint": _decision_hint(events, centers, edges, best_generator),
    }


def tuning_map_vector_targets(tuning_map: TuningMap) -> dict[str, float]:
    """Scalar targets useful for later SAME/SA3 probe rows."""

    summary = dict(tuning_map.summary)
    return {
        "pitch_event_count": float(summary.get("pitch_event_count", 0.0) or 0.0),
        "pitch_center_count": float(summary.get("pitch_center_count", 0.0) or 0.0),
        "selected_period_cents": float(summary.get("selected_period_cents", 0.0) or 0.0),
        "cohesion_score": float(summary.get("cohesion_score", 0.0) or 0.0),
        "ratio_fit_error_cents": float(summary.get("ratio_fit_error_cents", 0.0) or 0.0),
        "ratio_fit_coverage_35c": float(summary.get("ratio_fit_coverage_35c", 0.0) or 0.0),
        "harmonic_entropy_proxy_mean": float(summary.get("harmonic_entropy_proxy_mean", 0.0) or 0.0),
        "best_generator_cents": float(summary.get("best_generator_cents", 0.0) or 0.0),
        "best_generator_error_cents": float(summary.get("best_generator_error_cents", 0.0) or 0.0),
    }


def _resolve_reference_hz(rows: Sequence[Mapping[str, Any]], reference_hz: float | None) -> float:
    if reference_hz is not None:
        if reference_hz <= 0:
            raise ValueError("reference_hz must be positive")
        return float(reference_hz)
    freqs = [float(row["f0_hz"]) for row in rows if _finite_positive(row.get("f0_hz"))]
    if not freqs:
        return 440.0
    return float(np.percentile(freqs, 10.0))


def _event_from_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_hz: float,
    event_index: int,
    hop_seconds: float,
) -> PitchEvent:
    times = np.asarray([float(row.get("time_seconds", 0.0)) for row in rows], dtype=np.float64)
    freqs = np.asarray([float(row["f0_hz"]) for row in rows], dtype=np.float64)
    cents = 1200.0 * np.log2(freqs / reference_hz)
    confidence = np.asarray([float(row.get("pitch_confidence", 0.0) or 0.0) for row in rows], dtype=np.float64)
    start = float(times[0] - 0.5 * hop_seconds)
    end = float(times[-1] + 0.5 * hop_seconds)
    duration = max(end - start, hop_seconds)
    weight = float(duration * max(float(np.mean(confidence)), 1e-3))
    return PitchEvent(
        event_id=f"event_{event_index:03d}",
        start_seconds=max(0.0, start),
        end_seconds=max(0.0, end),
        duration_seconds=float(duration),
        median_hz=float(np.median(freqs)),
        median_cents=float(np.median(cents)),
        std_cents=float(np.std(cents)),
        confidence_mean=float(np.mean(confidence)),
        frame_count=len(rows),
        weight=weight,
    )


def _keep_event(event: PitchEvent, cfg: TuningMapConfig) -> bool:
    return event.frame_count >= cfg.min_event_frames and event.duration_seconds >= cfg.min_event_duration_seconds


def _center_from_cluster(cluster: Sequence[Mapping[str, Any]], *, period_cents: float, center_index: int) -> PitchCenter:
    events = [row["event"] for row in cluster]
    folded = np.asarray([float(row["folded"]) for row in cluster], dtype=np.float64)
    weights = np.asarray([float(row["weight"]) for row in cluster], dtype=np.float64)
    center = _weighted_circular_mean(folded, weights, period_cents)
    spreads = _circular_cents_distance(folded, center, period_cents)
    absolute = np.asarray([event.median_cents for event in events], dtype=np.float64)
    freqs = np.asarray([event.median_hz for event in events], dtype=np.float64)
    return PitchCenter(
        center_id=f"center_{center_index:02d}",
        cents=float(center),
        absolute_cents_mean=float(np.average(absolute, weights=weights)),
        median_hz=float(np.median(freqs)),
        event_count=len(events),
        frame_count=sum(int(event.frame_count) for event in events),
        weight=float(np.sum(weights)),
        spread_cents=float(np.average(spreads, weights=weights)) if spreads.size else 0.0,
        event_ids=tuple(event.event_id for event in events),
    )


def _period_candidate_from_parts(
    period_cents: float,
    centers: Sequence[PitchCenter],
    edges: Sequence[IntervalEdge],
    generators: Sequence[GeneratorCandidate],
) -> PeriodCandidate:
    ratio_errors = [edge.best_error_cents for edge in edges if edge.best_error_cents is not None]
    coverage = float(np.mean(np.asarray(ratio_errors) <= 35.0)) if ratio_errors else 0.0
    ratio_error = _mean_or_none(ratio_errors)
    best_generator = generators[0] if generators else None
    generator_error = None if best_generator is None else float(best_generator.mean_abs_error_cents)
    generator_cents = None if best_generator is None else float(best_generator.generator_cents)
    cohesion = (ratio_error if ratio_error is not None else 200.0) + 0.5 * (generator_error if generator_error is not None else 200.0)
    cohesion += 5.0 * (1.0 - coverage) + 0.15 * max(len(centers) - 1, 0)
    return PeriodCandidate(
        period_cents=float(period_cents),
        center_count=len(centers),
        edge_count=len(edges),
        ratio_fit_error_cents=ratio_error,
        ratio_fit_coverage=coverage,
        generator_cents=generator_cents,
        generator_error_cents=generator_error,
        cohesion_score=float(cohesion),
    )


def _decision_hint(
    events: Sequence[PitchEvent],
    centers: Sequence[PitchCenter],
    edges: Sequence[IntervalEdge],
    best_generator: GeneratorCandidate | None,
) -> str:
    if len(events) < 2:
        return "insufficient_pitch_events"
    if len(centers) < 2:
        return "single_center_or_unclustered"
    strong_edges = [edge for edge in edges if edge.best_error_cents is not None and edge.best_error_cents <= 20.0]
    if not strong_edges:
        return "weak_low_integer_interval_evidence"
    if best_generator is not None and best_generator.mean_abs_error_cents <= 20.0:
        return "cohesive_ratio_and_generator_map"
    return "ratio_edges_without_clear_generator"


def _median_hop_seconds(rows: Sequence[Mapping[str, Any]]) -> float:
    times = [float(row.get("time_seconds", 0.0)) for row in rows]
    if len(times) < 2:
        return 0.01
    diffs = np.diff(np.asarray(times, dtype=np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.01
    return float(np.median(diffs))


def _weighted_circular_mean(values: np.ndarray, weights: np.ndarray, period_cents: float) -> float:
    angles = 2.0 * np.pi * values / period_cents
    z = np.sum(weights * np.exp(1j * angles))
    if abs(z) <= 1e-12:
        return float(np.average(values, weights=weights) % period_cents)
    angle = np.angle(z)
    return float((angle % (2.0 * np.pi)) * period_cents / (2.0 * np.pi))


def _circular_cents_distance(a: np.ndarray, b: np.ndarray | float, period_cents: float) -> np.ndarray:
    return np.abs(np.mod(a - b + 0.5 * period_cents, period_cents) - 0.5 * period_cents)


def _circular_cents_distance_scalar(a: float, b: float, period_cents: float) -> float:
    return float(abs((a - b + 0.5 * period_cents) % period_cents - 0.5 * period_cents))


def _finite_positive(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(f) and f > 0


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and isfinite(float(value))]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _as_fraction(value: int | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def _fraction_label(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _fold_fraction_to_period(value: Fraction, *, period: Fraction) -> Fraction:
    if value <= 0:
        raise ValueError("ratio must be positive")
    folded = Fraction(value)
    while folded < 1:
        folded *= period
    while folded >= period:
        folded /= period
    return folded


def _period_fraction_for_cents(period_cents: float, *, tolerance_cents: float = 1e-3) -> Fraction | None:
    if abs(float(period_cents) - 1200.0) <= tolerance_cents:
        return Fraction(2, 1)
    if abs(float(period_cents) - TRITAVE_CENTS) <= tolerance_cents:
        return Fraction(3, 1)
    return None
