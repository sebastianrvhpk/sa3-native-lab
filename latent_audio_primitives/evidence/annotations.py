"""Annotation persistence and search helpers for notebook listening evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


def load_audio_annotations(path: str | Path) -> dict[str, Any]:
    """Load a JSON annotation store produced by the Colab player."""

    annotation_path = Path(path)
    if not annotation_path.exists():
        return {"annotation_version": 1, "items": []}
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"annotation_version": 1, "items": payload}
    payload.setdefault("annotation_version", 1)
    payload.setdefault("items", [])
    return payload


def save_audio_annotation(path: str | Path, annotation: dict[str, Any]) -> dict[str, Any]:
    """Upsert one track annotation into a JSON store."""

    annotation_path = Path(path)
    store = load_audio_annotations(annotation_path)
    items = list(store.get("items", []))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    normalized = _normalize_annotation(annotation, updated_at=now)
    key = normalized["path"]
    replaced = False
    for index, item in enumerate(items):
        if str(item.get("path", "")) == key:
            normalized["created_at"] = item.get("created_at", now)
            items[index] = normalized
            replaced = True
            break
    if not replaced:
        normalized["created_at"] = now
        items.append(normalized)

    store["annotation_version"] = 1
    store["items"] = items
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, ensure_ascii=True)
    return {"ok": True, "path": str(annotation_path), "count": len(items), "item": normalized}


def search_audio_annotations(
    path: str | Path,
    *,
    query: str = "",
    tags: Iterable[str] | None = None,
    min_rating: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Search annotation text/tags and return matching tracks."""

    store = load_audio_annotations(path)
    query = query.strip().lower()
    required_tags = {tag.strip().lower() for tag in (tags or []) if tag.strip()}
    matches = []
    for item in store.get("items", []):
        item_tags = {str(tag).lower() for tag in item.get("tags", [])}
        if required_tags and not required_tags.issubset(item_tags):
            continue
        rating = item.get("rating")
        if min_rating is not None and (rating is None or float(rating) < min_rating):
            continue
        searchable = " ".join(
            [
                str(item.get("label", "")),
                str(item.get("value", "")),
                str(item.get("description", "")),
                " ".join(str(tag) for tag in item.get("tags", [])),
            ]
        ).lower()
        if query and query not in searchable:
            continue
        matches.append(item)

    matches.sort(key=lambda item: (float(item.get("rating") or 0.0), item.get("updated_at", "")), reverse=True)
    return matches[:limit] if limit is not None else matches


def _normalize_annotation(annotation: dict[str, Any], *, updated_at: str) -> dict[str, Any]:
    path = str(annotation.get("path", ""))
    if not path:
        raise ValueError("annotation must include a path")
    return {
        "path": path,
        "label": str(annotation.get("label", "")),
        "track_index": _optional_int(annotation.get("track_index")),
        "player_title": str(annotation.get("player_title", "")),
        "rating": _optional_float(annotation.get("rating")),
        "tags": _normalize_tags(annotation.get("tags", [])),
        "value": str(annotation.get("value", "")),
        "description": str(annotation.get("description", "")),
        "metadata": annotation.get("metadata", {}) if isinstance(annotation.get("metadata", {}), dict) else {},
        "updated_at": updated_at,
    }


def _normalize_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        raw_tags = tags.split(",")
    elif isinstance(tags, Iterable):
        raw_tags = tags
    else:
        raw_tags = []
    normalized = []
    seen = set()
    for tag in raw_tags:
        value = str(tag).strip()
        key = value.lower()
        if value and key not in seen:
            normalized.append(value)
            seen.add(key)
    return normalized


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
