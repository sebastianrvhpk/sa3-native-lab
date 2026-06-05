"""JSON and NumPy persistence helpers for LatentItem notebook artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import LatentItem


MANIFEST_VERSION = 1


def save_item(item: LatentItem, directory: str | Path) -> Path:
    """Save one latent-memory item as ``latent.npy`` + ``metadata.json``."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "latent.npy", item.latent.astype(np.float32, copy=False))
    metadata = item.shallow_metadata()
    metadata["manifest_version"] = MANIFEST_VERSION
    with (directory / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return directory


def load_item(directory: str | Path) -> LatentItem:
    """Load one latent-memory item from ``latent.npy`` + ``metadata.json``."""

    directory = Path(directory)
    with (directory / "metadata.json").open("r", encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)
    latent = np.load(directory / "latent.npy")
    return LatentItem(
        item_id=metadata["item_id"],
        latent=latent,
        latent_rate=float(metadata["latent_rate"]),
        sample_rate=metadata.get("sample_rate"),
        prompt=metadata.get("prompt"),
        descriptors=metadata.get("descriptors") or {},
        labels=metadata.get("labels") or {},
        metadata=metadata.get("metadata") or {},
    )


def save_items(items: list[LatentItem], root: str | Path) -> list[Path]:
    """Save multiple latent-memory items and a manifest under one root."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "items": [],
    }
    for item in items:
        item_dir = root / _safe_item_id(item.item_id)
        paths.append(save_item(item, item_dir))
        manifest["items"].append({"item_id": item.item_id, "path": item_dir.name})
    with (root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return paths


def load_items(root: str | Path) -> list[LatentItem]:
    """Load all latent-memory items from a manifest root or item directories."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        return [load_item(root / entry["path"]) for entry in manifest["items"]]
    return [load_item(path) for path in sorted(root.iterdir()) if (path / "metadata.json").exists()]


def _safe_item_id(item_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in item_id)
    return safe[:180] or "item"
