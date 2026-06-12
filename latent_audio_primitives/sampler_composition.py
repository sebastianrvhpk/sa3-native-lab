"""SA3-native sampler-composition math and JSON-friendly rows.

This module owns small, explicit primitives for trajectory-level experiments:

- scalar schedules over step fraction, sigma/timestep, or logSNR,
- prompt phase selection over the same coordinates,
- rectified-flow denoised-state anchoring,
- compact rows that say what the sampler intended to do at each step.

It does not load SA3, build conditioning tensors, or run a sampler loop. Those
source-sensitive operations belong in ``procedures/`` and ``adapters/``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SamplerStepContext:
    """One sampler coordinate used by schedules and prompt phases."""

    step_index: int
    step_count: int
    timestep: float
    sigma: float
    logsnr: float | None = None

    @property
    def step_fraction(self) -> float:
        if self.step_count <= 1:
            return 0.0
        return float(self.step_index) / float(self.step_count - 1)

    def value(self, coordinate: str) -> float | None:
        coordinate = coordinate.lower()
        if coordinate == "step_index":
            return float(self.step_index)
        if coordinate == "step_fraction":
            return self.step_fraction
        if coordinate in {"timestep", "t"}:
            return float(self.timestep)
        if coordinate in {"sigma", "noise"}:
            return float(self.sigma)
        if coordinate == "logsnr":
            return self.logsnr
        raise ValueError(
            "coordinate must be 'step_index', 'step_fraction', 'timestep', "
            "'sigma', or 'logsnr'"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "step_count": int(self.step_count),
            "step_fraction": float(self.step_fraction),
            "timestep": float(self.timestep),
            "sigma": float(self.sigma),
            "logsnr": None if self.logsnr is None else float(self.logsnr),
        }


@dataclass(frozen=True, slots=True)
class ScheduleKnot:
    """One scalar schedule knot."""

    x: float
    value: float

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "value": float(self.value)}


@dataclass(frozen=True, slots=True)
class ScalarSchedule:
    """A scalar value as a function of sampler coordinate.

    The coordinate is intentionally explicit. ``logsnr`` schedules are usually
    the most portable when comparing sampler variants, while ``step_fraction``
    schedules are easy to audition.
    """

    knots: Sequence[ScheduleKnot | tuple[float, float] | list[float]]
    coordinate: str = "step_fraction"
    interpolation: str = "linear"
    default: float = 0.0
    clamp: bool = True
    name: str = ""

    def __post_init__(self) -> None:
        if self.interpolation not in {"linear", "previous", "nearest"}:
            raise ValueError("interpolation must be 'linear', 'previous', or 'nearest'")
        normalized = []
        for knot in self.knots:
            if isinstance(knot, ScheduleKnot):
                normalized.append(knot)
            else:
                if len(knot) != 2:
                    raise ValueError("schedule knots must be (x, value) pairs")
                normalized.append(ScheduleKnot(float(knot[0]), float(knot[1])))
        normalized.sort(key=lambda item: item.x)
        object.__setattr__(self, "knots", tuple(normalized))
        object.__setattr__(self, "coordinate", self.coordinate.lower())

    def value_for(self, context: SamplerStepContext) -> float:
        x = context.value(self.coordinate)
        if x is None or not self.knots:
            return float(self.default)
        x = float(x)
        first = self.knots[0]
        last = self.knots[-1]
        if x <= first.x:
            return float(first.value if self.clamp else self.default)
        if x >= last.x:
            return float(last.value if self.clamp else self.default)
        lower = first
        upper = last
        for left, right in zip(self.knots[:-1], self.knots[1:]):
            if left.x <= x <= right.x:
                lower, upper = left, right
                break
        if self.interpolation == "previous":
            return float(lower.value)
        if self.interpolation == "nearest":
            return float(lower.value if abs(x - lower.x) <= abs(upper.x - x) else upper.value)
        denom = max(upper.x - lower.x, 1e-12)
        frac = (x - lower.x) / denom
        return float(lower.value * (1.0 - frac) + upper.value * frac)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "coordinate": self.coordinate,
            "interpolation": self.interpolation,
            "default": float(self.default),
            "clamp": bool(self.clamp),
            "knots": [knot.to_dict() for knot in self.knots],
        }


@dataclass(frozen=True, slots=True)
class PromptPhaseSpec:
    """A prompt/condition phase active over a sampler coordinate interval."""

    name: str
    prompt: str
    coordinate: str = "logsnr"
    start: float | None = None
    end: float | None = None
    include_start: bool = True
    include_end: bool = True
    negative_prompt: str | None = None
    source: str = "manual"
    maturity: str = "intervention_candidate"
    note: str = ""

    def matches(self, context: SamplerStepContext) -> bool:
        x = context.value(self.coordinate)
        if x is None:
            return False
        if self.start is not None:
            if self.include_start and x < float(self.start):
                return False
            if not self.include_start and x <= float(self.start):
                return False
        if self.end is not None:
            if self.include_end and x > float(self.end):
                return False
            if not self.include_end and x >= float(self.end):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SamplerCompositionStepRow:
    """One intended sampler-composition action at a step."""

    step_index: int
    step_fraction: float
    timestep: float
    sigma: float
    logsnr: float | None
    cfg_scale: float
    apg_scale: float
    anchor_strength: float
    prompt_phase: str
    prompt: str
    maturity: str = "intervention_candidate"
    source_status: str = "repo-inferred"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SamplerCompositionPlan:
    """Notebook-facing manifest for SA3 trajectory-composition experiments."""

    cfg_schedule: ScalarSchedule | None = None
    apg_schedule: ScalarSchedule | None = None
    anchor_schedule: ScalarSchedule | None = None
    prompt_phases: Sequence[PromptPhaseSpec] = field(default_factory=tuple)
    default_prompt: str = ""
    default_negative_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_phases", tuple(self.prompt_phases))

    def phase_for(self, context: SamplerStepContext) -> PromptPhaseSpec | None:
        for phase in self.prompt_phases:
            if phase.matches(context):
                return phase
        return None

    def values_for(
        self,
        context: SamplerStepContext,
        *,
        default_cfg_scale: float = 1.0,
        default_apg_scale: float = 1.0,
    ) -> dict[str, Any]:
        phase = self.phase_for(context)
        cfg_scale = (
            self.cfg_schedule.value_for(context)
            if self.cfg_schedule is not None
            else float(default_cfg_scale)
        )
        apg_scale = (
            self.apg_schedule.value_for(context)
            if self.apg_schedule is not None
            else float(default_apg_scale)
        )
        anchor_strength = (
            self.anchor_schedule.value_for(context)
            if self.anchor_schedule is not None
            else 0.0
        )
        return {
            "cfg_scale": float(cfg_scale),
            "apg_scale": float(apg_scale),
            "anchor_strength": float(anchor_strength),
            "phase": phase,
            "prompt": phase.prompt if phase is not None else self.default_prompt,
            "negative_prompt": (
                phase.negative_prompt
                if phase is not None and phase.negative_prompt is not None
                else self.default_negative_prompt
            ),
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "cfg_schedule": None if self.cfg_schedule is None else self.cfg_schedule.to_dict(),
            "apg_schedule": None if self.apg_schedule is None else self.apg_schedule.to_dict(),
            "anchor_schedule": None if self.anchor_schedule is None else self.anchor_schedule.to_dict(),
            "prompt_phases": [phase.to_dict() for phase in self.prompt_phases],
            "default_prompt": self.default_prompt,
            "default_negative_prompt": self.default_negative_prompt,
            "metadata": dict(self.metadata),
        }


def sampler_logsnr_from_sigma(sigma: float | None) -> float | None:
    """Convert SA3 RF sigma/timestep to amplitude logSNR.

    SA3's RF sampler uses the same unit interval as the flow prompt helper:

        logSNR = log((1 - sigma) / sigma)
    """

    if sigma is None:
        return None
    eps = 1e-8
    clipped = min(max(float(sigma), eps), 1.0 - eps)
    return float(math.log((1.0 - clipped) / clipped))


def sampler_step_contexts(sigmas: Any) -> list[SamplerStepContext]:
    """Create schedule contexts from a 1D or per-batch SA3 sigma schedule."""

    values = _sigma_values(sigmas)
    if len(values) < 2:
        raise ValueError("sigma schedule must contain at least two points")
    step_count = len(values) - 1
    return [
        SamplerStepContext(
            step_index=index,
            step_count=step_count,
            timestep=float(values[index]),
            sigma=float(values[index]),
            logsnr=sampler_logsnr_from_sigma(float(values[index])),
        )
        for index in range(step_count)
    ]


def sampler_composition_step_rows(
    sigmas: Any,
    *,
    plan: SamplerCompositionPlan | Mapping[str, Any] | None = None,
    cfg_scale: float = 1.0,
    apg_scale: float = 1.0,
    default_prompt: str = "",
) -> list[SamplerCompositionStepRow]:
    """Return per-step rows for planned sampler-composition actions."""

    resolved = coerce_sampler_composition_plan(plan, default_prompt=default_prompt)
    rows = []
    for context in sampler_step_contexts(sigmas):
        values = resolved.values_for(
            context,
            default_cfg_scale=cfg_scale,
            default_apg_scale=apg_scale,
        )
        phase = values["phase"]
        rows.append(
            SamplerCompositionStepRow(
                step_index=context.step_index,
                step_fraction=context.step_fraction,
                timestep=context.timestep,
                sigma=context.sigma,
                logsnr=context.logsnr,
                cfg_scale=values["cfg_scale"],
                apg_scale=values["apg_scale"],
                anchor_strength=values["anchor_strength"],
                prompt_phase=phase.name if phase is not None else "default",
                prompt=values["prompt"],
                note="planned sampler-state intervention, not promoted evidence",
            )
        )
    return rows


def rf_denoised_from_velocity(state: Any, velocity: Any, sigma: Any) -> Any:
    """Return RF denoised estimate ``x0_hat = x_t - sigma * v``."""

    return state - _broadcast_sigma(sigma, state) * velocity


def rf_velocity_from_denoised(state: Any, denoised: Any, sigma: Any, *, eps: float = 1e-6) -> Any:
    """Return RF velocity implied by a denoised estimate.

    This is the inverse of ``x0_hat = x_t - sigma * v``:

        v = (x_t - x0_hat) / sigma
    """

    sigma_broadcast = _broadcast_sigma(sigma, state).clamp_min(float(eps))
    return (state - denoised) / sigma_broadcast


def apply_denoised_anchor(
    denoised: Any,
    anchor: Any,
    *,
    strength: float,
    mask: Any = None,
) -> Any:
    """Blend an RF denoised state toward a source/target anchor.

    ``strength=0`` leaves the sampler unchanged. ``strength=1`` replaces the
    denoised estimate by the anchor where the optional mask is active. Values
    outside ``[0, 1]`` are allowed for deliberate creative extrapolation.
    """

    if float(strength) == 0.0:
        return denoised
    anchor_tensor = anchor.to(device=denoised.device, dtype=denoised.dtype)
    if tuple(anchor_tensor.shape) != tuple(denoised.shape):
        raise ValueError(
            f"anchor shape {tuple(anchor_tensor.shape)} does not match denoised shape {tuple(denoised.shape)}"
        )
    weight = float(strength)
    if mask is None:
        return denoised * (1.0 - weight) + anchor_tensor * weight
    mask_tensor = _broadcast_mask(mask, denoised)
    return denoised * (1.0 - weight * mask_tensor) + anchor_tensor * (weight * mask_tensor)


def coerce_scalar_schedule(
    value: ScalarSchedule | Mapping[str, Any] | Sequence[float] | float | int | None,
    *,
    default: float,
    coordinate: str = "step_fraction",
    name: str = "",
) -> ScalarSchedule | None:
    """Normalize user-friendly schedule inputs into ``ScalarSchedule``."""

    if value is None:
        return None
    if isinstance(value, ScalarSchedule):
        return value
    if isinstance(value, Mapping):
        return ScalarSchedule(
            knots=value.get("knots", [(0.0, value.get("default", default))]),
            coordinate=str(value.get("coordinate", coordinate)),
            interpolation=str(value.get("interpolation", "linear")),
            default=float(value.get("default", default)),
            clamp=bool(value.get("clamp", True)),
            name=str(value.get("name", name)),
        )
    if isinstance(value, (int, float)):
        scalar = float(value)
        return ScalarSchedule(
            knots=[(0.0, scalar), (1.0, scalar)],
            coordinate=coordinate,
            interpolation="previous",
            default=scalar,
            name=name,
        )
    values = [float(item) for item in value]
    if not values:
        return None
    if len(values) == 1:
        knots = [(0.0, values[0]), (1.0, values[0])]
    else:
        knots = [
            (index / (len(values) - 1), scalar)
            for index, scalar in enumerate(values)
        ]
    return ScalarSchedule(knots=knots, coordinate=coordinate, default=default, name=name)


def coerce_prompt_phases(phases: Sequence[PromptPhaseSpec | Mapping[str, Any]] | None) -> tuple[PromptPhaseSpec, ...]:
    """Normalize prompt phase dictionaries into dataclasses."""

    out = []
    for phase in phases or []:
        if isinstance(phase, PromptPhaseSpec):
            out.append(phase)
        else:
            out.append(
                PromptPhaseSpec(
                    name=str(phase.get("name", f"phase_{len(out)}")),
                    prompt=str(phase.get("prompt", "")),
                    coordinate=str(phase.get("coordinate", "logsnr")),
                    start=None if phase.get("start") is None else float(phase.get("start")),
                    end=None if phase.get("end") is None else float(phase.get("end")),
                    include_start=bool(phase.get("include_start", True)),
                    include_end=bool(phase.get("include_end", True)),
                    negative_prompt=phase.get("negative_prompt"),
                    source=str(phase.get("source", "manual")),
                    maturity=str(phase.get("maturity", "intervention_candidate")),
                    note=str(phase.get("note", "")),
                )
            )
    return tuple(out)


def coerce_sampler_composition_plan(
    plan: SamplerCompositionPlan | Mapping[str, Any] | None,
    *,
    default_prompt: str = "",
    default_negative_prompt: str | None = None,
    cfg_scale: float = 1.0,
    apg_scale: float = 1.0,
) -> SamplerCompositionPlan:
    """Normalize a user dictionary into a sampler-composition plan."""

    if isinstance(plan, SamplerCompositionPlan):
        return plan
    payload = dict(plan or {})
    return SamplerCompositionPlan(
        cfg_schedule=coerce_scalar_schedule(
            payload.get("cfg_schedule"),
            default=cfg_scale,
            name="cfg_schedule",
        ),
        apg_schedule=coerce_scalar_schedule(
            payload.get("apg_schedule"),
            default=apg_scale,
            name="apg_schedule",
        ),
        anchor_schedule=coerce_scalar_schedule(
            payload.get("anchor_schedule"),
            default=0.0,
            name="anchor_schedule",
        ),
        prompt_phases=coerce_prompt_phases(payload.get("prompt_phases")),
        default_prompt=str(payload.get("default_prompt", default_prompt)),
        default_negative_prompt=payload.get("default_negative_prompt", default_negative_prompt),
        metadata=dict(payload.get("metadata", {})),
    )


def schedule_rows_to_table(rows: Sequence[SamplerCompositionStepRow]) -> list[dict[str, Any]]:
    """Return JSON/table-friendly sampler-composition rows."""

    return [row.to_dict() for row in rows]


def _sigma_values(sigmas: Any) -> list[float]:
    try:
        import torch

        if isinstance(sigmas, torch.Tensor):
            tensor = sigmas.detach().float().cpu()
            if tensor.ndim == 2:
                tensor = tensor[0]
            return [float(value) for value in tensor.reshape(-1).tolist()]
    except Exception:
        pass
    if hasattr(sigmas, "ndim"):
        arr = sigmas
        if arr.ndim == 2:
            arr = arr[0]
        return [float(value) for value in list(arr.reshape(-1))]
    return [float(value) for value in sigmas]


def _broadcast_sigma(sigma: Any, reference: Any) -> Any:
    try:
        import torch

        if isinstance(sigma, torch.Tensor):
            tensor = sigma.to(device=reference.device, dtype=reference.dtype)
        else:
            tensor = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 0:
            return tensor
        while tensor.ndim < reference.ndim:
            tensor = tensor.unsqueeze(-1)
        return tensor
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for sampler tensor math.") from exc


def _broadcast_mask(mask: Any, reference: Any) -> Any:
    try:
        import torch

        tensor = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask)
        tensor = tensor.to(device=reference.device, dtype=reference.dtype)
        if tensor.ndim == 1 and reference.ndim == 3:
            tensor = tensor.view(1, 1, -1)
        elif tensor.ndim == 2 and reference.ndim == 3:
            tensor = tensor.unsqueeze(1)
        while tensor.ndim < reference.ndim:
            tensor = tensor.unsqueeze(0)
        return tensor.clamp(0.0, 1.0)
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for sampler mask math.") from exc
