from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Callable, Sequence


PromptScorer = Callable[[str], float]
BatchPromptScorer = Callable[[list[str]], Sequence[float]]


@dataclass(frozen=True, slots=True)
class PromptOptimizationResult:
    prompt: str
    score: float
    history: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GreedyPromptSearchResult:
    prompt: str
    score: float
    tokens: list[str]
    history: list[tuple[str, float]] = field(default_factory=list)


def prompt_seed_from_audio_path(path: str | Path, *, extra_tags: list[str] | None = None) -> str:
    """Make a crude prompt seed from path tokens.

    This is not a captioner. It is a deterministic fallback that turns file and
    folder names into a starting prompt for later CLAP/caption/LLM optimization.
    """

    path = Path(path)
    tokens = []
    for part in [*path.parent.parts[-2:], path.stem]:
        tokens.extend(_tokens(part))
    if extra_tags:
        tokens.extend(extra_tags)
    deduped = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return " ".join(deduped) if deduped else "audio texture"


def coordinate_prompt_search(
    seed_prompt: str,
    modifier_axes: list[list[str]],
    scorer: PromptScorer,
    *,
    rounds: int = 2,
) -> PromptOptimizationResult:
    """Small coordinate search over prompt modifiers using a pluggable scorer.

    This does not generate audio by itself. In Colab, the scorer can be CLAP
    audio-text similarity, a prompt-adherence model, or a custom human-in-loop
    score. It is useful for audio-to-prompt work, but it is still text search,
    not soft-prompt gradient optimization.
    """

    best_prompt = seed_prompt.strip()
    best_score = float(scorer(best_prompt))
    history = [(best_prompt, best_score)]

    for _round in range(rounds):
        improved = False
        for axis in modifier_axes:
            candidates = [_join_prompt(best_prompt, modifier) for modifier in axis]
            scored = [(candidate, float(scorer(candidate))) for candidate in candidates]
            history.extend(scored)
            candidate_prompt, candidate_score = max(scored, key=lambda item: item[1])
            if candidate_score > best_score:
                best_prompt = candidate_prompt
                best_score = candidate_score
                improved = True
        if not improved:
            break
    return PromptOptimizationResult(prompt=best_prompt, score=best_score, history=history)


def greedy_token_prompt_search(
    vocabulary: Sequence[str],
    scorer: BatchPromptScorer,
    *,
    tokens_generated: int = 16,
    runs: int = 8,
    token_subset: int | None = None,
    candidate_batch_size: int | None = None,
    seed: int = 0,
    prefix: str = "",
    suffix: str = "",
    separator: str = " ",
    higher_is_better: bool = True,
) -> GreedyPromptSearchResult:
    """Greedy hard-token prompt synthesis with a pluggable batch scorer.

    This is the transferable part of the old CLIP "image to text" notebook:
    build a prompt one token at a time by evaluating many candidate next tokens
    against a target embedding/objective. The scorer can be CLAP, an SA3
    teacher-forcing loss wrapped as a negative score, or a human-in-loop model.
    """

    if not vocabulary:
        raise ValueError("vocabulary must be non-empty")
    if tokens_generated < 1:
        raise ValueError("tokens_generated must be at least 1")
    if runs < 1:
        raise ValueError("runs must be at least 1")

    rng = random.Random(seed)
    best_global: GreedyPromptSearchResult | None = None
    full_history: list[tuple[str, float]] = []

    for _run in range(runs):
        selected: list[str] = []
        run_history: list[tuple[str, float]] = []
        for _position in range(tokens_generated):
            if token_subset is None or token_subset >= len(vocabulary):
                candidates = list(vocabulary)
            else:
                candidates = rng.sample(list(vocabulary), token_subset)
            prompts = [_assemble_prompt([*selected, token], prefix, suffix, separator) for token in candidates]
            scores = _score_prompts_in_batches(scorer, prompts, candidate_batch_size)
            if len(scores) != len(prompts):
                raise ValueError("batch scorer must return one score per prompt")
            pick = max(range(len(scores)), key=scores.__getitem__) if higher_is_better else min(range(len(scores)), key=scores.__getitem__)
            selected.append(candidates[pick])
            run_history.extend(zip(prompts, scores))

        final_prompt = _assemble_prompt(selected, prefix, suffix, separator)
        final_score = _score_prompts_in_batches(scorer, [final_prompt], candidate_batch_size)[0]
        result = GreedyPromptSearchResult(
            prompt=final_prompt,
            score=final_score,
            tokens=list(selected),
            history=run_history,
        )
        full_history.extend(run_history)
        if best_global is None:
            best_global = result
        elif higher_is_better and result.score > best_global.score:
            best_global = result
        elif not higher_is_better and result.score < best_global.score:
            best_global = result

    assert best_global is not None
    return GreedyPromptSearchResult(
        prompt=best_global.prompt,
        score=best_global.score,
        tokens=best_global.tokens,
        history=full_history,
    )


def default_modifier_axes() -> list[list[str]]:
    return [
        ["bright", "dark", "warm", "cold", "muted", "shimmering"],
        ["sparse", "dense", "minimal", "layered", "busy"],
        ["calm", "tense", "euphoric", "melancholic", "aggressive"],
        ["ambient", "cinematic", "electronic", "acoustic", "experimental"],
        ["wide stereo", "narrow centered", "reverberant", "dry close"],
    ]


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _join_prompt(prompt: str, modifier: str) -> str:
    prompt = prompt.strip()
    modifier = modifier.strip()
    if not prompt:
        return modifier
    if modifier in prompt:
        return prompt
    return f"{prompt}, {modifier}"


def _assemble_prompt(tokens: list[str], prefix: str, suffix: str, separator: str) -> str:
    body = separator.join(token.strip() for token in tokens if token.strip())
    parts = [part.strip() for part in [prefix, body, suffix] if part.strip()]
    return separator.join(parts)


def _score_prompts_in_batches(
    scorer: BatchPromptScorer,
    prompts: list[str],
    batch_size: int | None,
) -> list[float]:
    if batch_size is None or batch_size <= 0 or batch_size >= len(prompts):
        return [float(score) for score in scorer(prompts)]

    scores: list[float] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        scores.extend(float(score) for score in scorer(chunk))
    return scores
