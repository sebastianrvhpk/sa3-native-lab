"""Tuning-system prompt recipes and deterministic intonation evidence.

These helpers are microscopes/selectors for SA3/SAME runs. They can show
whether rendered audio has a monophonic pitch trace that fits a requested
tuning lattice better than nearby/null systems, but they do not prove causal
intonation control without repeat runs and listening evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from math import isfinite, log2
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TuningSystem:
    """A pitch-class lattice used for prompt recipes and f0 evidence rows."""

    name: str
    family: str
    degrees_cents: Sequence[float]
    period_cents: float = 1200.0
    root_hz: float = 440.0
    ratios: Sequence[str] = ()
    prompt_terms: Sequence[str] = ()
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.period_cents <= 0:
            raise ValueError("period_cents must be positive")
        if self.root_hz <= 0:
            raise ValueError("root_hz must be positive")
        degrees = _unique_sorted_degrees(self.degrees_cents, self.period_cents)
        if not degrees:
            raise ValueError("TuningSystem requires at least one degree")
        object.__setattr__(self, "degrees_cents", tuple(degrees))
        object.__setattr__(self, "ratios", tuple(str(value) for value in self.ratios))
        object.__setattr__(self, "prompt_terms", tuple(str(value) for value in self.prompt_terms))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def degree_count(self) -> int:
        """Number of pitch classes in the folded scale/lattice."""

        return len(self.degrees_cents)

    def to_row(self) -> dict[str, Any]:
        """Return a JSON-friendly tuning-system manifest row."""

        row = asdict(self)
        row["degrees_cents"] = [float(value) for value in self.degrees_cents]
        row["period_cents"] = float(self.period_cents)
        row["root_hz"] = float(self.root_hz)
        row["degree_count"] = int(self.degree_count)
        return row


@dataclass(frozen=True, slots=True)
class PitchTrackConfig:
    """Settings for the dependency-light autocorrelation pitch tracker."""

    frame_length: int = 4096
    hop_length: int = 512
    fmin_hz: float = 55.0
    fmax_hz: float = 1760.0
    min_confidence: float = 0.25
    active_rms_percentile: float = 20.0
    min_active_rms: float = 1e-4
    max_frames: int | None = None


@dataclass(frozen=True, slots=True)
class TuningMatchConfig:
    """Settings for matching tracked f0 values to a tuning lattice."""

    tolerance_cents: float = 25.0
    fit_root_offset: bool = True
    root_offset_step_cents: float = 2.5
    min_voiced_frames: int = 4
    min_voiced_coverage: float = 0.08


@dataclass(frozen=True, slots=True)
class TuningPromptSpec:
    """One prompt/seed recipe for a tuning-system SA3 probe."""

    prompt_id: str
    system_name: str
    prompt: str
    role: str = "target"
    seed: int = 0
    root_note: str = "A4"
    root_hz: float = 440.0
    duration: float = 8.0
    negative_prompt: str = "wide vibrato, continuous glissando, pitch bend, detuned chorus"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Return a JSON-friendly prompt recipe row."""

        row = asdict(self)
        row["root_hz"] = float(self.root_hz)
        row["duration"] = float(self.duration)
        row["metadata"] = dict(self.metadata)
        return row


def ratio_to_cents(ratio: str | float | int | Fraction) -> float:
    """Convert a frequency ratio to cents."""

    value = _ratio_to_float(ratio)
    if value <= 0:
        raise ValueError("ratio must be positive")
    return float(1200.0 * log2(value))


def cents_to_ratio(cents: float) -> float:
    """Convert cents to a frequency ratio."""

    return float(2.0 ** (float(cents) / 1200.0))


def tuning_system_from_ratios(
    name: str,
    ratios: Sequence[str | float | int | Fraction],
    *,
    family: str = "just_intonation",
    period_ratio: str | float | int | Fraction = 2,
    root_hz: float = 440.0,
    prompt_terms: Sequence[str] = (),
    description: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> TuningSystem:
    """Build a folded tuning system from exact or numeric ratios."""

    period_cents = ratio_to_cents(period_ratio)
    ratio_strings = tuple(_ratio_to_label(value) for value in ratios)
    degrees = [ratio_to_cents(value) for value in ratios]
    return TuningSystem(
        name=name,
        family=family,
        degrees_cents=degrees,
        period_cents=period_cents,
        root_hz=root_hz,
        ratios=ratio_strings,
        prompt_terms=tuple(prompt_terms),
        description=description,
        metadata=dict(metadata or {}),
    )


def equal_temperament_system(
    name: str,
    divisions: int,
    *,
    steps: Sequence[int] | None = None,
    family: str = "equal_temperament",
    period_ratio: str | float | int | Fraction = 2,
    root_hz: float = 440.0,
    prompt_terms: Sequence[str] = (),
    description: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> TuningSystem:
    """Build an equal-tempered scale over an octave or other period."""

    if divisions <= 0:
        raise ValueError("divisions must be positive")
    period_cents = ratio_to_cents(period_ratio)
    selected_steps = tuple(range(divisions)) if steps is None else tuple(int(step) for step in steps)
    degrees = [(step % divisions) * period_cents / divisions for step in selected_steps]
    terms = tuple(prompt_terms) or (f"{divisions}-EDO", f"{divisions} equal divisions")
    meta = {"divisions": int(divisions), "steps": list(selected_steps), **dict(metadata or {})}
    return TuningSystem(
        name=name,
        family=family,
        degrees_cents=degrees,
        period_cents=period_cents,
        root_hz=root_hz,
        prompt_terms=terms,
        description=description,
        metadata=meta,
    )


def odd_limit_ratio_strings(limit: int) -> list[str]:
    """Return octave-folded odd-limit ratios with odd numerator/denominator <= limit.

    This is a compact notebook lattice, not a full harmonic-space theory. It is
    useful for making 5-limit, 7-limit, or 11-limit prompt/evidence candidates
    explicit before a run.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    ratios: set[Fraction] = set()
    odd_values = [value for value in range(1, int(limit) + 1, 2)]
    for numerator in odd_values:
        for denominator in odd_values:
            folded = _fold_fraction_to_period(Fraction(numerator, denominator), period=Fraction(2, 1))
            ratios.add(folded)
    return [_ratio_to_label(value) for value in sorted(ratios, key=float)]


def odd_limit_tuning_system(
    limit: int,
    *,
    root_hz: float = 440.0,
    name: str | None = None,
    description: str | None = None,
) -> TuningSystem:
    """Build an octave-folded odd-limit JI lattice."""

    ratios = odd_limit_ratio_strings(limit)
    label = name or f"ji_{int(limit)}_odd_limit_lattice"
    return tuning_system_from_ratios(
        label,
        ratios,
        family="just_intonation",
        root_hz=root_hz,
        prompt_terms=(f"{int(limit)}-limit just intonation", "JI ratios", "pure intervals"),
        description=description or f"Octave-folded odd-limit just-intonation lattice up to {int(limit)}.",
        metadata={"limit": int(limit), "lattice": "odd_limit"},
    )


def default_tuning_systems(root_hz: float = 440.0) -> list[TuningSystem]:
    """Return a compact starter bank for xenharmonic and JI prompt probes."""

    return [
        equal_temperament_system(
            "12_tet_major",
            12,
            steps=(0, 2, 4, 5, 7, 9, 11),
            family="12_tet_reference",
            root_hz=root_hz,
            prompt_terms=("12-TET", "twelve-tone equal temperament", "major scale"),
            description="Reference 12-tone equal-tempered major scale.",
        ),
        equal_temperament_system(
            "19_edo_diatonic",
            19,
            steps=(0, 3, 6, 8, 11, 14, 17),
            family="xenharmonic_equal",
            root_hz=root_hz,
            prompt_terms=("19-EDO", "19 equal divisions of the octave", "xenharmonic"),
            description="Seven-step diatonic subset inside 19 equal divisions of the octave.",
        ),
        equal_temperament_system(
            "31_edo_major",
            31,
            steps=(0, 5, 10, 13, 18, 23, 28),
            family="xenharmonic_equal",
            root_hz=root_hz,
            prompt_terms=("31-EDO", "31 equal divisions of the octave", "xenharmonic"),
            description="Seven-step major-like subset inside 31 equal divisions of the octave.",
        ),
        equal_temperament_system(
            "bohlen_pierce_13_ed3",
            13,
            steps=(0, 2, 4, 5, 7, 9, 11),
            family="xenharmonic_non_octave",
            period_ratio=3,
            root_hz=root_hz,
            prompt_terms=("Bohlen-Pierce", "13 equal divisions of the tritave", "non-octave scale"),
            description="Bohlen-Pierce-style 13 equal divisions of a 3:1 tritave.",
            metadata={"period": "tritave", "divisions": 13},
        ),
        tuning_system_from_ratios(
            "ji_5_limit_major",
            ("1/1", "9/8", "5/4", "4/3", "3/2", "5/3", "15/8"),
            root_hz=root_hz,
            prompt_terms=("5-limit just intonation", "pure thirds and fifths", "JI major scale"),
            description="5-limit just major scale.",
            metadata={"limit": 5},
        ),
        tuning_system_from_ratios(
            "ji_7_limit_septimal",
            ("1/1", "8/7", "7/6", "5/4", "4/3", "3/2", "7/4"),
            root_hz=root_hz,
            prompt_terms=("7-limit just intonation", "septimal intervals", "JI scale"),
            description="Septimal 7-limit just-intonation scale.",
            metadata={"limit": 7},
        ),
        tuning_system_from_ratios(
            "ji_11_limit_neutral",
            ("1/1", "12/11", "11/10", "9/8", "5/4", "4/3", "3/2", "8/5", "7/4", "15/8"),
            root_hz=root_hz,
            prompt_terms=("11-limit just intonation", "neutral intervals", "JI scale"),
            description="11-limit just-intonation scale with neutral interval candidates.",
            metadata={"limit": 11},
        ),
    ]


def tuning_system_rows(systems: Sequence[TuningSystem] | None = None) -> list[dict[str, Any]]:
    """Return manifest rows for a tuning-system bank."""

    return [system.to_row() for system in (systems or default_tuning_systems())]


def tuning_prompt_specs(
    systems: Sequence[TuningSystem] | None = None,
    *,
    seeds: Sequence[int] = (0, 1, 2),
    root_note: str = "A4",
    root_hz: float = 440.0,
    duration: float = 8.0,
    base_prompt: str = "solo dry monophonic lead melody, sparse notes, stable pitch centers",
    include_nulls: bool = True,
) -> list[TuningPromptSpec]:
    """Create prompt rows for a first SA3 tuning-system sweep."""

    resolved_systems = list(systems or default_tuning_systems(root_hz=root_hz))
    specs: list[TuningPromptSpec] = []
    for system in resolved_systems:
        system = _system_with_root(system, root_hz)
        scale_text = _scale_prompt_text(system)
        not_12tet = "" if system.name.startswith("12_tet") else ", do not collapse it to ordinary 12-TET"
        for seed in seeds:
            prompt = (
                f"{base_prompt}; tune the melody to {scale_text}; root {root_note} = {root_hz:g} Hz"
                f"{not_12tet}; avoid vibrato, portamento, pitch bend, and thick chords"
            )
            specs.append(
                TuningPromptSpec(
                    prompt_id=f"{system.name}_seed{int(seed)}",
                    system_name=system.name,
                    prompt=prompt,
                    role="target",
                    seed=int(seed),
                    root_note=root_note,
                    root_hz=root_hz,
                    duration=duration,
                    metadata={"system": system.to_row()},
                )
            )

    if include_nulls:
        for seed in seeds:
            specs.append(
                TuningPromptSpec(
                    prompt_id=f"null_ordinary_pitch_seed{int(seed)}",
                    system_name="",
                    prompt=(
                        f"{base_prompt}; ordinary tonal melody with no named tuning system; "
                        f"root {root_note} = {root_hz:g} Hz; avoid vibrato and pitch bend"
                    ),
                    role="null_no_named_tuning",
                    seed=int(seed),
                    root_note=root_note,
                    root_hz=root_hz,
                    duration=duration,
                    negative_prompt="",
                    metadata={"null": "no_named_tuning"},
                )
            )
            specs.append(
                TuningPromptSpec(
                    prompt_id=f"null_pitch_bend_seed{int(seed)}",
                    system_name="",
                    prompt=(
                        f"{base_prompt}; continuous glissando and expressive pitch bends; "
                        f"do not quantize to a scale; root area near {root_note}"
                    ),
                    role="null_unquantized_pitch",
                    seed=int(seed),
                    root_note=root_note,
                    root_hz=root_hz,
                    duration=duration,
                    negative_prompt="",
                    metadata={"null": "unquantized_pitch"},
                )
            )
    return specs


def tuning_prompt_rows(specs: Sequence[TuningPromptSpec]) -> list[dict[str, Any]]:
    """Return JSON-friendly prompt recipe rows."""

    return [spec.to_row() for spec in specs]


def pitch_track_rows(
    audio: Any,
    sample_rate: int,
    *,
    config: PitchTrackConfig | None = None,
) -> list[dict[str, Any]]:
    """Track monophonic f0 candidates with framewise autocorrelation."""

    cfg = config or PitchTrackConfig()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if cfg.frame_length < 16:
        raise ValueError("frame_length must be at least 16")
    if cfg.hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if cfg.fmin_hz <= 0 or cfg.fmax_hz <= 0 or cfg.fmin_hz >= cfg.fmax_hz:
        raise ValueError("expected 0 < fmin_hz < fmax_hz")

    mono = _as_mono_audio(audio)
    frame_length = int(cfg.frame_length)
    hop_length = int(cfg.hop_length)
    if mono.size < frame_length:
        mono = np.pad(mono, (0, frame_length - mono.size))
    frame_count = 1 + max(0, (mono.size - frame_length) // hop_length)
    if cfg.max_frames is not None:
        frame_count = min(frame_count, max(0, int(cfg.max_frames)))
    if frame_count == 0:
        return []

    rms_values = np.empty((frame_count,), dtype=np.float32)
    for index in range(frame_count):
        frame = mono[index * hop_length : index * hop_length + frame_length]
        rms_values[index] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    active_percentile = min(max(float(cfg.active_rms_percentile), 0.0), 100.0)
    active_threshold = max(float(cfg.min_active_rms), float(np.percentile(rms_values, active_percentile)))

    rows: list[dict[str, Any]] = []
    for index in range(frame_count):
        start = index * hop_length
        frame = mono[start : start + frame_length]
        rms = float(rms_values[index])
        active = bool(rms >= active_threshold)
        f0_hz = None
        confidence = 0.0
        if active:
            f0_hz, confidence = _autocorrelation_f0(
                frame,
                sample_rate,
                fmin_hz=float(cfg.fmin_hz),
                fmax_hz=float(cfg.fmax_hz),
            )
            if confidence < cfg.min_confidence:
                f0_hz = None
        rows.append(
            {
                "frame_index": int(index),
                "time_seconds": float((start + 0.5 * frame_length) / sample_rate),
                "rms": rms,
                "active": active,
                "f0_hz": None if f0_hz is None else float(f0_hz),
                "pitch_confidence": float(confidence),
                "voiced": bool(f0_hz is not None),
            }
        )
    return rows


def match_pitch_frequencies_to_tuning(
    frequencies_hz: Sequence[float],
    system: TuningSystem,
    *,
    root_hz: float | None = None,
    config: TuningMatchConfig | None = None,
) -> dict[str, Any]:
    """Match f0 values to one tuning system and return cent-error evidence."""

    cfg = config or TuningMatchConfig()
    root = float(system.root_hz if root_hz is None else root_hz)
    if root <= 0:
        raise ValueError("root_hz must be positive")
    f0 = np.asarray([float(value) for value in frequencies_hz if _finite_positive(value)], dtype=np.float64)
    if f0.size:
        cents = np.mod(1200.0 * np.log2(f0 / root), float(system.period_cents))
    else:
        cents = np.empty((0,), dtype=np.float64)

    fixed_errors, fixed_degrees = _nearest_degree_errors(cents, system, root_offset_cents=0.0)
    if cfg.fit_root_offset and cents.size:
        offsets = np.arange(0.0, float(system.period_cents), max(float(cfg.root_offset_step_cents), 0.1))
        best_offset = 0.0
        best_errors = fixed_errors
        best_degrees = fixed_degrees
        best_score = _mean_or_inf(best_errors)
        for offset in offsets:
            errors, degree_indices = _nearest_degree_errors(cents, system, root_offset_cents=float(offset))
            score = _mean_or_inf(errors)
            if score < best_score:
                best_score = score
                best_offset = float(offset)
                best_errors = errors
                best_degrees = degree_indices
    else:
        best_offset = 0.0
        best_errors = fixed_errors
        best_degrees = fixed_degrees

    fixed_within = _within_ratio(fixed_errors, cfg.tolerance_cents)
    best_within = _within_ratio(best_errors, cfg.tolerance_cents)
    degree_coverage = len(set(int(value) for value in best_degrees.tolist())) if best_degrees.size else 0
    voiced_count = int(f0.size)
    status = _tuning_status(
        voiced_count=voiced_count,
        voiced_coverage=None,
        within_ratio=best_within,
        config=cfg,
    )
    return {
        "candidate_system": system.name,
        "candidate_family": system.family,
        "period_cents": float(system.period_cents),
        "system_degree_count": int(system.degree_count),
        "root_hz": root,
        "voiced_frame_count": voiced_count,
        "fixed_mean_abs_error_cents": _mean_or_none(fixed_errors),
        "fixed_median_abs_error_cents": _median_or_none(fixed_errors),
        "fixed_within_tolerance_ratio": fixed_within,
        "best_root_offset_cents": float(best_offset),
        "best_mean_abs_error_cents": _mean_or_none(best_errors),
        "best_median_abs_error_cents": _median_or_none(best_errors),
        "best_within_tolerance_ratio": best_within,
        "degree_coverage": int(degree_coverage),
        "degree_coverage_ratio": float(degree_coverage / max(system.degree_count, 1)),
        "tolerance_cents": float(cfg.tolerance_cents),
        "status": status,
        "claim_maturity": "microscope",
    }


def tuning_match_report(
    audio: Any,
    sample_rate: int,
    system: TuningSystem,
    *,
    root_hz: float | None = None,
    pitch_config: PitchTrackConfig | None = None,
    match_config: TuningMatchConfig | None = None,
) -> dict[str, Any]:
    """Track f0 from audio and score it against one tuning system."""

    rows = pitch_track_rows(audio, sample_rate, config=pitch_config)
    frequencies = [row["f0_hz"] for row in rows if row.get("f0_hz") is not None]
    report = match_pitch_frequencies_to_tuning(frequencies, system, root_hz=root_hz, config=match_config)
    active_count = sum(1 for row in rows if row.get("active"))
    voiced_count = sum(1 for row in rows if row.get("voiced"))
    frame_count = len(rows)
    voiced_coverage = voiced_count / max(frame_count, 1)
    active_voiced_ratio = voiced_count / max(active_count, 1)
    confidence_values = [float(row["pitch_confidence"]) for row in rows if row.get("voiced")]
    report.update(
        {
            "frame_count": int(frame_count),
            "active_frame_count": int(active_count),
            "voiced_frame_count": int(voiced_count),
            "voiced_coverage": float(voiced_coverage),
            "active_voiced_ratio": float(active_voiced_ratio),
            "pitch_confidence_mean": _mean_or_none(np.asarray(confidence_values, dtype=np.float64)),
            "status": _tuning_status(
                voiced_count=voiced_count,
                voiced_coverage=voiced_coverage,
                within_ratio=report["best_within_tolerance_ratio"],
                config=match_config or TuningMatchConfig(),
            ),
        }
    )
    return report


def tuning_comparison_rows(
    audio_by_output: Mapping[str, Any],
    sample_rate: int,
    candidate_systems: Sequence[TuningSystem] | None = None,
    *,
    target_system_by_output: Mapping[str, str] | None = None,
    root_hz_by_output: Mapping[str, float] | None = None,
    pitch_config: PitchTrackConfig | None = None,
    match_config: TuningMatchConfig | None = None,
) -> list[dict[str, Any]]:
    """Score each rendered output against each candidate tuning system."""

    systems = list(candidate_systems or default_tuning_systems())
    targets = dict(target_system_by_output or {})
    roots = dict(root_hz_by_output or {})
    rows: list[dict[str, Any]] = []
    for output_id, audio in audio_by_output.items():
        pitch_rows = pitch_track_rows(audio, sample_rate, config=pitch_config)
        frequencies = [row["f0_hz"] for row in pitch_rows if row.get("f0_hz") is not None]
        target_system = targets.get(output_id)
        output_rows: list[dict[str, Any]] = []
        for system in systems:
            report = match_pitch_frequencies_to_tuning(
                frequencies,
                system,
                root_hz=roots.get(output_id, system.root_hz),
                config=match_config,
            )
            output_rows.append(
                {
                    "output_id": str(output_id),
                    "target_system": target_system,
                    "is_target_system": bool(target_system and system.name == target_system),
                    "frame_count": len(pitch_rows),
                    "active_frame_count": sum(1 for row in pitch_rows if row.get("active")),
                    "voiced_coverage": sum(1 for row in pitch_rows if row.get("voiced")) / max(len(pitch_rows), 1),
                    **report,
                }
            )
        output_rows.sort(key=lambda row: _score_for_rank(row))
        for rank, row in enumerate(output_rows, start=1):
            row["rank"] = int(rank)
            rows.append(row)
    return rows


def tuning_selectivity_rows(comparison_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize whether each output fits its prompted system better than alternatives."""

    by_output: dict[str, list[Mapping[str, Any]]] = {}
    for row in comparison_rows:
        by_output.setdefault(str(row.get("output_id", "")), []).append(row)

    out: list[dict[str, Any]] = []
    for output_id, rows in sorted(by_output.items()):
        ranked = sorted(rows, key=_score_for_rank)
        best = ranked[0] if ranked else {}
        target_rows = [row for row in ranked if row.get("is_target_system")]
        target = target_rows[0] if target_rows else None
        best_non_target = next((row for row in ranked if not row.get("is_target_system")), None)
        target_score = None if target is None else _optional_float(target.get("best_mean_abs_error_cents"))
        non_target_score = None if best_non_target is None else _optional_float(best_non_target.get("best_mean_abs_error_cents"))
        margin = None if target_score is None or non_target_score is None else non_target_score - target_score
        target_rank = None if target is None else int(target.get("rank", ranked.index(target) + 1))
        if target is None:
            decision = "no_target_label"
        elif target_rank == 1 and margin is not None and margin >= 5.0:
            decision = "target_wins"
        elif target_rank == 1:
            decision = "target_wins_small_margin"
        else:
            decision = "ambiguous_or_non_target_wins"
        out.append(
            {
                "output_id": output_id,
                "target_system": None if target is None else target.get("candidate_system"),
                "target_rank": target_rank,
                "target_best_mean_abs_error_cents": target_score,
                "best_candidate_system": best.get("candidate_system"),
                "best_candidate_error_cents": _optional_float(best.get("best_mean_abs_error_cents")),
                "best_non_target_system": None if best_non_target is None else best_non_target.get("candidate_system"),
                "best_non_target_error_cents": non_target_score,
                "target_margin_cents": margin,
                "voiced_coverage": best.get("voiced_coverage"),
                "decision_hint": decision,
                "claim_maturity": "selector" if decision.startswith("target_wins") else "microscope",
            }
        )
    return out


def _system_with_root(system: TuningSystem, root_hz: float) -> TuningSystem:
    if abs(float(system.root_hz) - float(root_hz)) < 1e-9:
        return system
    return TuningSystem(
        name=system.name,
        family=system.family,
        degrees_cents=system.degrees_cents,
        period_cents=system.period_cents,
        root_hz=root_hz,
        ratios=system.ratios,
        prompt_terms=system.prompt_terms,
        description=system.description,
        metadata=system.metadata,
    )


def _scale_prompt_text(system: TuningSystem) -> str:
    terms = ", ".join(system.prompt_terms) if system.prompt_terms else system.name.replace("_", " ")
    if system.ratios:
        ratios = " ".join(system.ratios[:12])
        suffix = " ..." if len(system.ratios) > 12 else ""
        return f"{terms} using ratios {ratios}{suffix}"
    degrees = " ".join(f"{value:.1f}" for value in system.degrees_cents[:12])
    suffix = " ..." if len(system.degrees_cents) > 12 else ""
    return f"{terms} with pitch classes at {degrees}{suffix} cents"


def _as_mono_audio(audio: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(audio, torch.Tensor):
        arr = audio.detach().float().cpu().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("audio must contain samples")
    if arr.ndim == 1:
        mono = arr.astype(np.float32, copy=False)
    elif arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            mono = arr.mean(axis=0)
        else:
            mono = arr.mean(axis=1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        return _as_mono_audio(arr[0])
    else:
        raise ValueError(f"audio must have shape N, C x N, or N x C; got {arr.shape}")
    if not np.isfinite(mono).all():
        raise ValueError("audio contains NaN or infinite values")
    return np.ascontiguousarray(mono, dtype=np.float32)


def _autocorrelation_f0(
    frame: np.ndarray,
    sample_rate: int,
    *,
    fmin_hz: float,
    fmax_hz: float,
) -> tuple[float | None, float]:
    x = np.asarray(frame, dtype=np.float64)
    x = x - float(np.mean(x))
    x = x * np.hanning(x.size)
    energy = float(np.dot(x, x))
    if energy <= 1e-12:
        return None, 0.0
    n_fft = 1 << int((2 * x.size - 1).bit_length())
    spectrum = np.fft.rfft(x, n=n_fft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft)[: x.size]
    if corr[0] <= 1e-12:
        return None, 0.0
    corr = corr / max(float(corr[0]), 1e-12)
    max_f = min(float(fmax_hz), sample_rate / 2.0)
    lag_min = max(1, int(np.floor(sample_rate / max_f)))
    lag_max = min(x.size - 2, int(np.ceil(sample_rate / float(fmin_hz))))
    if lag_max <= lag_min:
        return None, 0.0

    search = corr[lag_min : lag_max + 1]
    
    # Identify local peaks in the correlation within the search range
    peaks = []
    global_max_val = -1.0
    for lag_idx in range(lag_min, lag_max + 1):
        val = float(corr[lag_idx])
        if val > global_max_val:
            global_max_val = val
        # Check if it is a local maximum
        if corr[lag_idx] > corr[lag_idx - 1] and corr[lag_idx] > corr[lag_idx + 1]:
            peaks.append((lag_idx, val))

    selected_lag = None
    if peaks and global_max_val > 0.0:
        # Threshold-based first-peak picker (YIN-style thresholding at ~0.15 relative to max)
        # Select the first local peak (smallest lag) with a value >= 85% of global max
        threshold_val = 0.85 * global_max_val
        for lag_idx, val in peaks:
            if val >= threshold_val:
                selected_lag = lag_idx
                break

    # Fallback to global argmax if no local peaks meet the threshold
    if selected_lag is None:
        local_index = int(np.argmax(search))
        selected_lag = lag_min + local_index

    confidence = float(corr[selected_lag])
    refined_lag = _parabolic_peak_lag(corr, selected_lag)
    if refined_lag <= 0:
        return None, confidence
    return float(sample_rate / refined_lag), confidence


def _parabolic_peak_lag(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(index)
    y0 = float(values[index - 1])
    y1 = float(values[index])
    y2 = float(values[index + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(index)
    delta = 0.5 * (y0 - y2) / denom
    return float(index + max(min(delta, 1.0), -1.0))


def _nearest_degree_errors(
    cents: np.ndarray,
    system: TuningSystem,
    *,
    root_offset_cents: float,
) -> tuple[np.ndarray, np.ndarray]:
    if cents.size == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    degrees = np.asarray(system.degrees_cents, dtype=np.float64)
    shifted = np.mod(degrees + float(root_offset_cents), float(system.period_cents))
    distances = _circular_cents_distance(cents[:, None], shifted[None, :], float(system.period_cents))
    degree_indices = np.argmin(distances, axis=1)
    errors = distances[np.arange(cents.size), degree_indices]
    return errors.astype(np.float64, copy=False), degree_indices.astype(np.int64, copy=False)


def _circular_cents_distance(a: np.ndarray, b: np.ndarray, period_cents: float) -> np.ndarray:
    delta = np.abs(np.mod(a - b + 0.5 * period_cents, period_cents) - 0.5 * period_cents)
    return delta.astype(np.float64, copy=False)


def _unique_sorted_degrees(values: Sequence[float], period_cents: float) -> list[float]:
    degrees = []
    for value in values:
        degree = float(value) % float(period_cents)
        if isfinite(degree):
            degrees.append(degree)
    unique = sorted({round(value, 6): value for value in degrees}.values())
    return [float(value) for value in unique]


def _ratio_to_float(ratio: str | float | int | Fraction) -> float:
    if isinstance(ratio, Fraction):
        return float(ratio)
    if isinstance(ratio, str):
        return float(Fraction(ratio.strip()))
    return float(ratio)


def _ratio_to_label(ratio: str | float | int | Fraction) -> str:
    if isinstance(ratio, Fraction):
        return f"{ratio.numerator}/{ratio.denominator}"
    if isinstance(ratio, str):
        return ratio
    return f"{float(ratio):.8g}"


def _fold_fraction_to_period(value: Fraction, *, period: Fraction) -> Fraction:
    if value <= 0:
        raise ValueError("ratio must be positive")
    folded = Fraction(value)
    while folded < 1:
        folded *= period
    while folded >= period:
        folded /= period
    return folded


def _finite_positive(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return isfinite(f) and f > 0


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(result):
        return None
    return result


def _mean_or_inf(values: np.ndarray) -> float:
    if values.size == 0:
        return float("inf")
    return float(np.mean(values))


def _mean_or_none(values: np.ndarray | Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _median_or_none(values: np.ndarray | Sequence[float]) -> float | None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _within_ratio(values: np.ndarray, tolerance_cents: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values <= float(tolerance_cents)))


def _score_for_rank(row: Mapping[str, Any]) -> float:
    score = _optional_float(row.get("best_mean_abs_error_cents"))
    if score is None:
        return float("inf")
    return score


def _tuning_status(
    *,
    voiced_count: int,
    voiced_coverage: float | None,
    within_ratio: float | None,
    config: TuningMatchConfig,
) -> str:
    if voiced_count < config.min_voiced_frames:
        return "insufficient_voiced_pitch"
    if voiced_coverage is not None and voiced_coverage < config.min_voiced_coverage:
        return "low_voiced_coverage"
    if within_ratio is None:
        return "unscored"
    if within_ratio >= 0.70:
        return "strong_pitch_class_fit"
    if within_ratio >= 0.45:
        return "partial_pitch_class_fit"
    return "weak_pitch_class_fit"
