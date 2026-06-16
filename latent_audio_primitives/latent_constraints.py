"""Scalar objectives over SAME/SA3 latent tensors.

These helpers define lightweight latent constraints that can be inspected in a
notebook row or used as differentiable objectives by procedures/guidance cells.
They operate on tensor-like latents and do not call SA3/SAME.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class LatentConstraintSpec:
    """A scalar latent measurement or optimization target."""

    name: str
    kind: str
    target: float = 0.0
    weight: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["params"] = dict(self.params)
        return row


def default_latent_constraint_specs() -> list[LatentConstraintSpec]:
    """A small constraint library for first-pass latent optimization."""

    return [
        LatentConstraintSpec(name="match_reference", kind="reference_distance", target=0.0, weight=1.0),
        LatentConstraintSpec(name="preserve_rms", kind="rms", target=1.0, weight=0.25),
        LatentConstraintSpec(name="motion_energy", kind="motion_energy", target=0.0, weight=0.5),
        LatentConstraintSpec(name="loop_boundary", kind="loop_boundary", target=0.0, weight=0.5),
        LatentConstraintSpec(
            name="channel_energy_focus",
            kind="channel_energy",
            target=1.0,
            weight=0.5,
            params={"channels": [0, 1, 2, 3]},
        ),
    ]


def latent_constraint_value(latents: Any, spec: LatentConstraintSpec, *, reference: Any | None = None) -> Any:
    """Return a scalar tensor/number for a single latent constraint."""

    x = _as_constraint_latents(latents)
    if spec.kind == "reference_distance":
        if reference is None:
            return _tensor_zero_like(x)
        reference = _as_constraint_latents(reference)
        delta = x - reference
        return (delta * delta).mean()
    if spec.kind == "rms":
        return _sqrt((x * x).mean() + 1e-12)
    if spec.kind == "motion_energy":
        if x.shape[-1] < 2:
            return _tensor_zero_like(x)
        delta = x[..., 1:] - x[..., :-1]
        return (delta * delta).mean()
    if spec.kind == "loop_boundary":
        window = int(spec.params.get("window", 1))
        if window <= 1 or x.shape[-1] < 2 * window:
            delta = x[..., 0] - x[..., -1]
        else:
            if hasattr(x, "detach"):
                start = x[..., :window].mean(dim=-1)
                end = x[..., -window:].mean(dim=-1)
            else:
                start = x[..., :window].mean(axis=-1)
                end = x[..., -window:].mean(axis=-1)
            delta = start - end
        return (delta * delta).mean()
    if spec.kind == "channel_energy":
        channels = list(spec.params.get("channels", []))
        channel_dim = -2 if x.ndim >= 2 else 0
        channels = [channel for channel in channels if 0 <= int(channel) < x.shape[channel_dim]]
        if not channels:
            return _tensor_zero_like(x)
        selected = _select_channels(x, channels, dim=channel_dim)
        return (selected * selected).mean()
    if spec.kind == "mean":
        return x.mean()
    raise ValueError(f"Unknown latent constraint kind: {spec.kind!r}")


def latent_constraint_loss(
    latents: Any,
    specs: Sequence[LatentConstraintSpec],
    *,
    reference: Any | None = None,
) -> Any:
    """Combine constraint specs into a scalar objective."""

    total = _tensor_zero_like(latents)
    for spec in specs:
        value = latent_constraint_value(latents, spec, reference=reference)
        target = value.new_tensor(float(spec.target)) if hasattr(value, "new_tensor") else float(spec.target)
        delta = value - target
        total = total + float(spec.weight) * (delta * delta)
    return total


def evaluate_latent_constraints(
    before_latents: Any,
    after_latents: Any,
    specs: Sequence[LatentConstraintSpec],
    *,
    reference: Any | None = None,
) -> list[dict[str, Any]]:
    """Report how constraints changed between two latent states."""

    rows: list[dict[str, Any]] = []
    for spec in specs:
        before = _to_float(latent_constraint_value(before_latents, spec, reference=reference))
        after = _to_float(latent_constraint_value(after_latents, spec, reference=reference))
        target_error_before = abs(before - float(spec.target))
        target_error_after = abs(after - float(spec.target))
        rows.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "target": spec.target,
                "weight": spec.weight,
                "before": before,
                "after": after,
                "delta": after - before,
                "target_error_before": target_error_before,
                "target_error_after": target_error_after,
                "status": "improved" if target_error_after < target_error_before else "worse_or_unchanged",
            }
        )
    return rows


def _tensor_zero_like(x: Any) -> Any:
    if hasattr(x, "new_zeros"):
        return x.new_zeros(())
    dtype = np.asarray(x).dtype
    return np.asarray(0.0, dtype=dtype)


def _as_constraint_latents(value: Any) -> Any:
    if hasattr(value, "detach") or hasattr(value, "new_zeros"):
        return value
    return np.asarray(value, dtype=np.float32)


def _select_channels(x: Any, channels: Sequence[int], *, dim: int) -> Any:
    if hasattr(x, "index_select"):
        return x.index_select(dim, _torch_index(channels, device=x.device))
    return np.take(np.asarray(x), [int(channel) for channel in channels], axis=dim)


def _torch_index(values: Sequence[int], *, device: Any) -> Any:
    import torch

    return torch.tensor(list(values), dtype=torch.long, device=device)


def _sqrt(value: Any) -> Any:
    if hasattr(value, "sqrt"):
        return value.sqrt()
    return np.sqrt(value)


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().float().cpu().item())
    return float(value)
