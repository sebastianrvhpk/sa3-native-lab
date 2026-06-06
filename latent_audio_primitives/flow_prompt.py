"""Frozen-SA3 flow prompt scoring and prompt attribution helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence


BatchLossScorer = Callable[[list[str]], Sequence[float]]


@dataclass(frozen=True, slots=True)
class FlowPromptLossRow:
    """One prompt/probe loss row for notebook diagnostics."""

    prompt: str
    timestep: float
    logsnr: float | None
    loss: float
    probe_index: int


@dataclass(frozen=True, slots=True)
class PromptAttributionRow:
    """Leave-one-out prompt contribution row.

    ``contribution`` is positive when removing the token/phrase increases loss,
    so the original token helped the prompt explain the target under the loss.
    """

    token: str
    position: int
    prompt_without_token: str
    base_loss: float
    ablated_loss: float
    contribution: float
    best_replacement: str | None = None
    best_replacement_prompt: str | None = None
    best_replacement_loss: float | None = None
    best_replacement_delta: float | None = None


def summarize_flow_loss_rows(rows: Sequence[FlowPromptLossRow]) -> list[dict[str, float | str]]:
    """Aggregate diagnostic rows by prompt for compact notebook display."""

    by_prompt: dict[str, list[FlowPromptLossRow]] = {}
    for row in rows:
        by_prompt.setdefault(row.prompt, []).append(row)
    summary = []
    for prompt, prompt_rows in by_prompt.items():
        losses = [row.loss for row in prompt_rows]
        summary.append(
            {
                "prompt": prompt,
                "loss_mean": float(sum(losses) / max(len(losses), 1)),
                "loss_min": float(min(losses)),
                "loss_max": float(max(losses)),
                "probe_count": float(len(prompt_rows)),
            }
        )
    summary.sort(key=lambda row: float(row["loss_mean"]))
    return summary


def prompt_leave_one_out_attribution(
    prompt: str,
    loss_scorer: BatchLossScorer,
    *,
    replacement_candidates: Sequence[str] | None = None,
    token_pattern: str = r"[^,\s]+",
) -> list[PromptAttributionRow]:
    """Score how much each prompt token/phrase helps under a lower-is-better loss.

    The scorer receives batched prompts and returns one loss per prompt. Positive
    contribution means the original token reduced the loss.
    """

    tokens = _prompt_tokens(prompt, token_pattern=token_pattern)
    if not tokens:
        return []
    base_loss = float(loss_scorer([prompt])[0])
    ablated_prompts = [_prompt_without_index(tokens, index) for index in range(len(tokens))]
    ablated_losses = [float(value) for value in loss_scorer(ablated_prompts)]
    rows: list[PromptAttributionRow] = []
    replacements = [value.strip() for value in (replacement_candidates or []) if value.strip()]
    for index, (token, without_prompt, ablated_loss) in enumerate(zip(tokens, ablated_prompts, ablated_losses)):
        best_replacement = None
        best_replacement_prompt = None
        best_replacement_loss = None
        best_replacement_delta = None
        if replacements:
            replacement_pairs = [
                (candidate, _prompt_with_replacement(tokens, index, candidate))
                for candidate in replacements
                if candidate != token
            ]
            replacement_prompts = [pair[1] for pair in replacement_pairs]
            if replacement_prompts:
                replacement_losses = [float(value) for value in loss_scorer(replacement_prompts)]
                best_index = min(range(len(replacement_losses)), key=replacement_losses.__getitem__)
                best_replacement_prompt = replacement_prompts[best_index]
                best_replacement_loss = replacement_losses[best_index]
                best_replacement = replacement_pairs[best_index][0]
                best_replacement_delta = base_loss - best_replacement_loss
        rows.append(
            PromptAttributionRow(
                token=token,
                position=index,
                prompt_without_token=without_prompt,
                base_loss=base_loss,
                ablated_loss=ablated_loss,
                contribution=ablated_loss - base_loss,
                best_replacement=best_replacement,
                best_replacement_prompt=best_replacement_prompt,
                best_replacement_loss=best_replacement_loss,
                best_replacement_delta=best_replacement_delta,
            )
        )
    rows.sort(key=lambda row: row.contribution, reverse=True)
    return rows


def timesteps_from_logsnr_values(logsnr_values: Sequence[float] | str | None) -> list[float]:
    """Convert logSNR probe values to SA3 flow timesteps."""

    values = parse_float_sequence(logsnr_values)
    return [float(1.0 / (1.0 + math.exp(float(value)))) for value in values]


def logsnr_from_timestep(timestep: float, *, eps: float = 1e-8) -> float:
    """Convert a flow timestep to the matching logSNR value."""

    t = min(max(float(timestep), eps), 1.0 - eps)
    return float(math.log((1.0 - t) / t))


def parse_float_sequence(value: Sequence[float] | str | None) -> list[float]:
    """Parse comma/semicolon text or numeric sequences into float probes."""

    if value is None:
        return []
    if isinstance(value, str):
        chunks = value.replace(";", ",").split(",")
        return [float(chunk.strip()) for chunk in chunks if chunk.strip()]
    return [float(item) for item in value]


def flow_velocity_target(target: Any, noise: Any, *, convention: str):
    """Return the straight-flow velocity target under an explicit convention."""

    if convention == "noise_minus_data":
        return noise - target
    if convention == "data_minus_noise":
        return target - noise
    raise ValueError("velocity_convention must be 'noise_minus_data' or 'data_minus_noise'")


def _prompt_tokens(prompt: str, *, token_pattern: str) -> list[str]:
    return re.findall(token_pattern, prompt.strip())


def _prompt_without_index(tokens: Sequence[str], index: int) -> str:
    return " ".join(token for token_index, token in enumerate(tokens) if token_index != index)


def _prompt_with_replacement(tokens: Sequence[str], index: int, replacement: str) -> str:
    return " ".join(replacement if token_index == index else token for token_index, token in enumerate(tokens))
