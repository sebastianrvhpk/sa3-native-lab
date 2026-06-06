"""Prompt semantic bookkeeping for SA3-native prompt experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SEMANTIC_TAGS = (
    "material",
    "gesture",
    "time",
    "energy",
    "space",
    "affect",
    "production",
    "metadata",
    "instruction",
)

_TAG_ALIASES = {
    "instrument": "material",
    "instruments": "material",
    "instrumentation": "material",
    "timbre": "material",
    "texture": "material",
    "rhythm": "gesture",
    "motion": "gesture",
    "groove": "gesture",
    "tempo": "time",
    "duration": "time",
    "structure": "time",
    "intensity": "energy",
    "density": "energy",
    "loudness": "energy",
    "room": "space",
    "spatial": "space",
    "reverb": "space",
    "mood": "affect",
    "emotion": "affect",
    "mix": "production",
    "mastering": "production",
    "artist": "metadata",
    "reference": "metadata",
    "era": "metadata",
    "negative": "instruction",
    "constraint": "instruction",
}


@dataclass(frozen=True, slots=True)
class PromptVariant:
    """One candidate wording with explicit semantic intent."""

    variant_id: str
    prompt: str
    semantic_tags: tuple[str, ...] = ()
    source: str = "manual"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromptSemanticRow:
    """Prompt variant plus SA3-native evidence fields."""

    variant_id: str
    prompt: str
    semantic_tags: tuple[str, ...]
    source: str
    flow_loss: float | None = None
    flow_score: float | None = None
    descriptor_delta_norm: float | None = None
    listening_rating: float | None = None
    decision: str = ""
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_semantic_tags(tags: Sequence[str] | str | None) -> tuple[str, ...]:
    """Normalize prompt tags into the local SA3 notebook vocabulary."""

    if tags is None:
        return ()
    raw_tags = [tags] if isinstance(tags, str) else list(tags)
    normalized: list[str] = []
    for tag in raw_tags:
        key = str(tag).strip().lower().replace("_", " ").replace("-", " ")
        if not key:
            continue
        key = _TAG_ALIASES.get(key, key)
        if key not in SEMANTIC_TAGS:
            key = "instruction"
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def make_prompt_variants(
    prompts: Sequence[str] | Mapping[str, str],
    *,
    tags_by_variant: Mapping[str, Sequence[str] | str] | None = None,
    tags_by_prompt: Mapping[str, Sequence[str] | str] | None = None,
    source: str = "manual",
    notes_by_variant: Mapping[str, str] | None = None,
) -> list[PromptVariant]:
    """Create prompt variants from a list or ``id -> prompt`` mapping."""

    tags_by_variant = tags_by_variant or {}
    tags_by_prompt = tags_by_prompt or {}
    notes_by_variant = notes_by_variant or {}
    if isinstance(prompts, Mapping):
        items = [(str(key), str(value)) for key, value in prompts.items()]
    else:
        items = [(f"p{index:02d}", str(prompt)) for index, prompt in enumerate(prompts)]
    variants: list[PromptVariant] = []
    for variant_id, prompt in items:
        tags = tags_by_variant.get(variant_id, tags_by_prompt.get(prompt, ()))
        variants.append(
            PromptVariant(
                variant_id=variant_id,
                prompt=prompt,
                semantic_tags=normalize_semantic_tags(tags),
                source=source,
                notes=str(notes_by_variant.get(variant_id, "")),
            )
        )
    return variants


def prompt_semantic_rows(
    variants: Sequence[PromptVariant],
    *,
    flow_losses_by_prompt: Mapping[str, float] | None = None,
    descriptor_delta_norms_by_prompt: Mapping[str, float] | None = None,
    listening_ratings_by_prompt: Mapping[str, float] | None = None,
    decisions_by_prompt: Mapping[str, str] | None = None,
    notes_by_prompt: Mapping[str, str] | None = None,
) -> list[PromptSemanticRow]:
    """Attach SA3-native measurements to prompt variants."""

    flow_losses_by_prompt = flow_losses_by_prompt or {}
    descriptor_delta_norms_by_prompt = descriptor_delta_norms_by_prompt or {}
    listening_ratings_by_prompt = listening_ratings_by_prompt or {}
    decisions_by_prompt = decisions_by_prompt or {}
    notes_by_prompt = notes_by_prompt or {}
    rows: list[PromptSemanticRow] = []
    for variant in variants:
        flow_loss = _lookup_float(flow_losses_by_prompt, variant)
        rows.append(
            PromptSemanticRow(
                variant_id=variant.variant_id,
                prompt=variant.prompt,
                semantic_tags=variant.semantic_tags,
                source=variant.source,
                flow_loss=flow_loss,
                flow_score=None if flow_loss is None else -float(flow_loss),
                descriptor_delta_norm=_lookup_float(descriptor_delta_norms_by_prompt, variant),
                listening_rating=_lookup_float(listening_ratings_by_prompt, variant),
                decision=str(decisions_by_prompt.get(variant.prompt, decisions_by_prompt.get(variant.variant_id, ""))),
                notes=str(notes_by_prompt.get(variant.prompt, notes_by_prompt.get(variant.variant_id, variant.notes))),
                metadata=dict(variant.metadata),
            )
        )
    return rows


def rank_prompt_semantic_rows(rows: Sequence[PromptSemanticRow]) -> list[PromptSemanticRow]:
    """Rank rows by available native evidence without inventing a new score."""

    def sort_key(row: PromptSemanticRow) -> tuple[float, float, str]:
        flow_loss = float("inf") if row.flow_loss is None else float(row.flow_loss)
        rating = 0.0 if row.listening_rating is None else -float(row.listening_rating)
        return (flow_loss, rating, row.variant_id)

    return sorted(rows, key=sort_key)


def semantic_tag_counts(variants: Sequence[PromptVariant] | Sequence[PromptSemanticRow]) -> dict[str, int]:
    """Count how often each semantic tag appears in a prompt set."""

    counts = {tag: 0 for tag in SEMANTIC_TAGS}
    for item in variants:
        for tag in item.semantic_tags:
            counts[tag] = counts.get(tag, 0) + 1
    return {tag: count for tag, count in counts.items() if count}


def prompt_variant_manifest(variants: Sequence[PromptVariant]) -> list[dict[str, Any]]:
    """Return a notebook-friendly manifest for prompt variants."""

    return [
        {
            "variant_id": variant.variant_id,
            "prompt": variant.prompt,
            "semantic_tags": list(variant.semantic_tags),
            "source": variant.source,
            "notes": variant.notes,
            "metadata": dict(variant.metadata),
        }
        for variant in variants
    ]


def _lookup_float(mapping: Mapping[str, float], variant: PromptVariant) -> float | None:
    value = mapping.get(variant.prompt, mapping.get(variant.variant_id))
    return None if value is None else float(value)
