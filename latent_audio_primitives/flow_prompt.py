"""Frozen-SA3 flow prompt scoring and prompt attribution helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
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
    noise_seed: int | None = None
    noise_sign: float = 1.0


@dataclass(frozen=True, slots=True)
class FlowProbeSpec:
    """One reusable SA3 flow probe.

    The probe stores the scalar flow time, the equivalent logSNR when known,
    and the deterministic noise seed/sign used to form ``z_t``. It deliberately
    stores coordinates, not tensors, so a notebook can display and serialize the
    probe bank before running any model code.
    """

    probe_index: int
    timestep: float
    logsnr: float | None
    noise_seed: int
    noise_sign: float = 1.0


@dataclass(frozen=True, slots=True)
class FlowProbeBank:
    """Reusable flow probes for fair prompt and intervention comparisons."""

    probes: Sequence[FlowProbeSpec]
    velocity_convention: str = "noise_minus_data"
    shared_noise: bool = True
    antithetic_noise: bool = False
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.velocity_convention not in {"noise_minus_data", "data_minus_noise"}:
            raise ValueError("velocity_convention must be 'noise_minus_data' or 'data_minus_noise'")
        if not self.probes:
            raise ValueError("FlowProbeBank requires at least one probe")
        object.__setattr__(self, "probes", tuple(self.probes))

    @property
    def probe_count(self) -> int:
        """Number of concrete timestep/noise probes in the bank."""

        return len(self.probes)


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


def flow_probe_bank_from_values(
    *,
    logsnr_values: Sequence[float] | str | None = None,
    timestep_values: Sequence[float] | str | None = None,
    seed: int = 0,
    velocity_convention: str = "noise_minus_data",
    antithetic_noise: bool = False,
    shared_noise: bool = True,
    metadata: dict[str, Any] | None = None,
) -> FlowProbeBank:
    """Create deterministic flow probes from logSNR or timestep values."""

    if logsnr_values is not None and timestep_values is not None:
        raise ValueError("Provide either logsnr_values or timestep_values, not both")
    if timestep_values is not None:
        timesteps = parse_float_sequence(timestep_values)
        logsnrs: list[float | None] = [logsnr_from_timestep(value) for value in timesteps]
    else:
        logsnrs = parse_float_sequence(logsnr_values)
        timesteps = timesteps_from_logsnr_values(logsnrs)
    if not timesteps:
        raise ValueError("At least one logSNR or timestep value is required")

    signs = (1.0, -1.0) if antithetic_noise else (1.0,)
    probes: list[FlowProbeSpec] = []
    probe_index = 0
    base_seed = int(seed)
    for value_index, (timestep, logsnr) in enumerate(zip(timesteps, logsnrs)):
        for sign in signs:
            probes.append(
                FlowProbeSpec(
                    probe_index=probe_index,
                    timestep=float(timestep),
                    logsnr=None if logsnr is None else float(logsnr),
                    noise_seed=base_seed + value_index * 1009,
                    noise_sign=float(sign),
                )
            )
            probe_index += 1
    return FlowProbeBank(
        probes=probes,
        velocity_convention=velocity_convention,
        shared_noise=shared_noise,
        antithetic_noise=antithetic_noise,
        seed=base_seed,
        metadata=dict(metadata or {}),
    )


def flow_probe_bank_to_manifest(probe_bank: FlowProbeBank) -> dict[str, Any]:
    """Return a JSON-friendly manifest for a flow probe bank."""

    return {
        "velocity_convention": probe_bank.velocity_convention,
        "shared_noise": bool(probe_bank.shared_noise),
        "antithetic_noise": bool(probe_bank.antithetic_noise),
        "seed": int(probe_bank.seed),
        "probe_count": int(probe_bank.probe_count),
        "metadata": dict(probe_bank.metadata),
        "probes": [
            {
                "probe_index": int(probe.probe_index),
                "timestep": float(probe.timestep),
                "logsnr": None if probe.logsnr is None else float(probe.logsnr),
                "noise_seed": int(probe.noise_seed),
                "noise_sign": float(probe.noise_sign),
            }
            for probe in probe_bank.probes
        ],
    }


def flow_probe_bank_from_manifest(manifest: dict[str, Any]) -> FlowProbeBank:
    """Reconstruct a flow probe bank from ``flow_probe_bank_to_manifest``."""

    probes = [
        FlowProbeSpec(
            probe_index=int(item["probe_index"]),
            timestep=float(item["timestep"]),
            logsnr=None if item.get("logsnr") is None else float(item["logsnr"]),
            noise_seed=int(item["noise_seed"]),
            noise_sign=float(item.get("noise_sign", 1.0)),
        )
        for item in manifest.get("probes", [])
    ]
    return FlowProbeBank(
        probes=probes,
        velocity_convention=str(manifest.get("velocity_convention", "noise_minus_data")),
        shared_noise=bool(manifest.get("shared_noise", True)),
        antithetic_noise=bool(manifest.get("antithetic_noise", False)),
        seed=int(manifest.get("seed", 0)),
        metadata=dict(manifest.get("metadata", {})),
    )


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
    """Convert local straight-path logSNR probe values to SA3 flow timesteps.

    This project uses ``log((1 - t) / t)`` for the linear path
    ``z_t = (1 - t) z0 + t epsilon``. That is an amplitude log-ratio, not the
    squared-power logSNR convention used in some diffusion literature.
    """

    values = parse_float_sequence(logsnr_values)
    return [float(1.0 / (1.0 + math.exp(float(value)))) for value in values]


def logsnr_from_timestep(timestep: float, *, eps: float = 1e-8) -> float:
    """Convert a flow timestep to the matching local straight-path logSNR."""

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
