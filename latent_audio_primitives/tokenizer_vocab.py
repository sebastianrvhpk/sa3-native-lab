from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TokenizerVocabularyConfig:
    """Filters for building text candidates from the native SA3 tokenizer."""

    condition_key: str = "prompt"
    max_candidates: int = 2048
    min_chars: int = 2
    max_chars: int = 24
    require_word_start: bool = True
    ascii_only: bool = True
    lowercase: bool = True
    allow_punctuation: str = " -'/.&"


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


def native_tokenizer_vocabulary(
    model_or_tokenizer: Any,
    *,
    config: TokenizerVocabularyConfig | None = None,
    condition_key: str = "prompt",
    max_candidates: int | None = None,
    min_chars: int | None = None,
    max_chars: int | None = None,
    require_word_start: bool | None = None,
    ascii_only: bool | None = None,
    lowercase: bool | None = None,
    allow_punctuation: str | None = None,
) -> list[str]:
    """Build prompt-search candidates from SA3's native text tokenizer.

    The returned strings are decoded text pieces, not SAME latents and not
    learned SA3 audio tokens. They are useful for PEZ-like hard prompt search
    because every candidate is known to be representable by the tokenizer that
    SA3 actually uses.
    """

    cfg = config or TokenizerVocabularyConfig(condition_key=condition_key)
    if max_candidates is not None:
        cfg = _replace(cfg, max_candidates=max_candidates)
    if min_chars is not None:
        cfg = _replace(cfg, min_chars=min_chars)
    if max_chars is not None:
        cfg = _replace(cfg, max_chars=max_chars)
    if require_word_start is not None:
        cfg = _replace(cfg, require_word_start=require_word_start)
    if ascii_only is not None:
        cfg = _replace(cfg, ascii_only=ascii_only)
    if lowercase is not None:
        cfg = _replace(cfg, lowercase=lowercase)
    if allow_punctuation is not None:
        cfg = _replace(cfg, allow_punctuation=allow_punctuation)

    tokenizer = (
        model_or_tokenizer
        if hasattr(model_or_tokenizer, "get_vocab")
        else extract_prompt_tokenizer(model_or_tokenizer, condition_key=cfg.condition_key)
    )
    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_by_id = sorted(((int(token_id), token) for token, token_id in vocab.items()), key=lambda item: item[0])

    candidates: list[str] = []
    seen: set[str] = set()
    for token_id, token in token_by_id:
        if token_id in special_ids:
            continue
        if cfg.require_word_start and not _looks_like_word_start(token):
            continue
        text = _token_piece_to_text(tokenizer, token)
        text = _normalize_candidate(text, lowercase=cfg.lowercase)
        if not _candidate_allowed(text, cfg):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(text)
        if len(candidates) >= cfg.max_candidates:
            break

    return candidates


def preview_native_tokenizer_vocabulary(vocabulary: list[str], *, columns: int = 8, rows: int = 8) -> str:
    """Format a compact preview table for notebook logging."""

    if columns <= 0 or rows <= 0:
        raise ValueError("columns and rows must be positive")
    shown = vocabulary[: columns * rows]
    lines = []
    for row in range(rows):
        cells = shown[row * columns : (row + 1) * columns]
        if not cells:
            break
        lines.append(" | ".join(f"{cell:<14.14}" for cell in cells))
    return "\n".join(lines)


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


def _looks_like_word_start(token: str) -> bool:
    return token.startswith("\u2581") or token.startswith(" ") or token.startswith("\u0120")


def _token_piece_to_text(tokenizer: Any, token: str) -> str:
    try:
        text = tokenizer.convert_tokens_to_string([token])
    except Exception:
        text = token
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        text = token.replace("\u2581", " ").replace("\u0120", " ")
    return text


def _normalize_candidate(text: str, *, lowercase: bool) -> str:
    text = text.replace("\u2581", " ").replace("\u0120", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("_")
    return text.lower() if lowercase else text


def _candidate_allowed(text: str, cfg: TokenizerVocabularyConfig) -> bool:
    if len(text) < cfg.min_chars or len(text) > cfg.max_chars:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if text.startswith("<") or text.endswith(">"):
        return False
    if cfg.ascii_only and not text.isascii():
        return False
    allowed_punctuation = re.escape(cfg.allow_punctuation)
    if re.search(rf"[^A-Za-z0-9{allowed_punctuation}]", text):
        return False
    if re.fullmatch(r"[a-zA-Z]\d+", text):
        return False
    return True


def _replace(config: TokenizerVocabularyConfig, **changes: Any) -> TokenizerVocabularyConfig:
    values = {
        "condition_key": config.condition_key,
        "max_candidates": config.max_candidates,
        "min_chars": config.min_chars,
        "max_chars": config.max_chars,
        "require_word_start": config.require_word_start,
        "ascii_only": config.ascii_only,
        "lowercase": config.lowercase,
        "allow_punctuation": config.allow_punctuation,
    }
    values.update(changes)
    return TokenizerVocabularyConfig(**values)
