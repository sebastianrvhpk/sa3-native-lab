"""Native-evidence disagreement rows for SA3 notebook review."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class DisagreementRow:
    """One artifact or prompt variant reviewed across native evidence lanes."""

    item_id: str
    label: str = ""
    same_distance: float | None = None
    same_neighbor_score: float | None = None
    flow_loss: float | None = None
    descriptor_delta_norm: float | None = None
    listening_rating: float | None = None
    conflict_score: float = 0.0
    decision: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def descriptor_delta_norm(delta: Mapping[str, float] | Sequence[float] | None) -> float | None:
    """Summarize descriptor movement as mean absolute change."""

    if delta is None:
        return None
    if isinstance(delta, Mapping):
        values = [float(value) for value in delta.values()]
    else:
        values = [float(value) for value in delta]
    if not values:
        return None
    return float(sum(abs(value) for value in values) / len(values))


def make_disagreement_row(
    item_id: str,
    *,
    label: str = "",
    same_distance: float | None = None,
    same_neighbor_score: float | None = None,
    flow_loss: float | None = None,
    descriptor_delta: Mapping[str, float] | Sequence[float] | None = None,
    descriptor_delta_norm_value: float | None = None,
    listening_rating: float | None = None,
    decision: str = "",
    notes: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> DisagreementRow:
    """Build a disagreement row with a transparent native-evidence conflict score."""

    resolved_descriptor_delta_norm = (
        descriptor_delta_norm(descriptor_delta)
        if descriptor_delta_norm_value is None
        else float(descriptor_delta_norm_value)
    )
    values = {
        "same_distance": same_distance,
        "same_neighbor_score": same_neighbor_score,
        "flow_loss": flow_loss,
        "descriptor_delta_norm": resolved_descriptor_delta_norm,
        "listening_rating": listening_rating,
    }
    return DisagreementRow(
        item_id=str(item_id),
        label=str(label),
        same_distance=_float_or_none(same_distance),
        same_neighbor_score=_float_or_none(same_neighbor_score),
        flow_loss=_float_or_none(flow_loss),
        descriptor_delta_norm=_float_or_none(resolved_descriptor_delta_norm),
        listening_rating=_float_or_none(listening_rating),
        conflict_score=native_evidence_conflict_score(values),
        decision=str(decision),
        notes=str(notes),
        metadata=dict(metadata or {}),
    )


def native_evidence_conflict_score(values: Mapping[str, float | None]) -> float:
    """Estimate disagreement among available native evidence lanes.

    The score is only a triage signal. Lower flow loss and lower SAME distance
    are treated as better; higher neighbor score and listening rating are better.
    Descriptor movement is treated as an amount, not goodness, and therefore
    only contributes when other lanes are present.
    """

    normalized: list[float] = []
    if values.get("same_distance") is not None:
        normalized.append(_bounded_inverse(float(values["same_distance"])))
    if values.get("same_neighbor_score") is not None:
        normalized.append(_bounded(float(values["same_neighbor_score"])))
    if values.get("flow_loss") is not None:
        normalized.append(_bounded_inverse(float(values["flow_loss"])))
    if values.get("listening_rating") is not None:
        normalized.append(_bounded(float(values["listening_rating"]) / 5.0))
    if len(normalized) < 2:
        return 0.0
    spread = max(normalized) - min(normalized)
    descriptor_amount = values.get("descriptor_delta_norm")
    if descriptor_amount is not None:
        spread += 0.1 * _bounded(float(descriptor_amount))
    return float(spread)


def rank_disagreement_rows(rows: Sequence[DisagreementRow]) -> list[DisagreementRow]:
    """Put the most contradictory rows first for notebook review."""

    return sorted(rows, key=lambda row: (-row.conflict_score, row.item_id))


def disagreement_rows_to_table(rows: Sequence[DisagreementRow]) -> list[dict[str, Any]]:
    """Convert rows to dictionaries for pandas or JSON display."""

    return [
        {
            "item_id": row.item_id,
            "label": row.label,
            "same_distance": row.same_distance,
            "same_neighbor_score": row.same_neighbor_score,
            "flow_loss": row.flow_loss,
            "descriptor_delta_norm": row.descriptor_delta_norm,
            "listening_rating": row.listening_rating,
            "conflict_score": row.conflict_score,
            "decision": row.decision,
            "notes": row.notes,
            **{f"meta_{key}": value for key, value in row.metadata.items()},
        }
        for row in rows
    ]


def _float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bounded_inverse(value: float) -> float:
    return 1.0 / (1.0 + max(0.0, float(value)))
