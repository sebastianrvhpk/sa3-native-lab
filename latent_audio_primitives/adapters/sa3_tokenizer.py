"""Stable Audio 3 tokenizer access helpers."""

from __future__ import annotations

from typing import Any


def extract_prompt_tokenizer(model_or_adapter: Any, *, condition_key: str = "prompt") -> Any:
    """Return the tokenizer used by the SA3 text conditioner.

    Works with the official ``StableAudioModel``, the local ``StableAudio3Adapter``,
    or the underlying conditioned diffusion wrapper.
    """

    for candidate in _walk_model_candidates(model_or_adapter):
        conditioner = getattr(candidate, "conditioner", None)
        if conditioner is None:
            continue
        conditioners = getattr(conditioner, "conditioners", None)
        if conditioners is None:
            if hasattr(conditioner, "tokenizer"):
                return conditioner.tokenizer
            continue
        if condition_key in conditioners and hasattr(conditioners[condition_key], "tokenizer"):
            return conditioners[condition_key].tokenizer
        for sub_conditioner in conditioners.values():
            if hasattr(sub_conditioner, "tokenizer"):
                return sub_conditioner.tokenizer
    raise ValueError(f"could not find a tokenizer-backed conditioner for key {condition_key!r}")


def _walk_model_candidates(root: Any):
    seen: set[int] = set()
    queue = [root]
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        yield candidate
        for attr in ("model", "diffusion"):
            child = getattr(candidate, attr, None)
            if child is not None:
                queue.append(child)
