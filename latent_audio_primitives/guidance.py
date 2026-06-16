"""Differentiable latent guidance primitives for notebook sampler probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class GuidanceStepResult:
    """Result of one differentiable latent guidance update."""

    latents: Any
    loss: float
    grad_norm: float


def gradient_guidance_step(
    latents: Any,
    loss_fn: Callable[[Any], Any],
    *,
    scale: float = 1.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> GuidanceStepResult:
    """One generic differentiable latent guidance step.

    This implements the primitive behind training-free guidance:

        z' = z - scale * grad_z L(z)

    If ``normalize=True``, the gradient is divided by its RMS norm so ``scale``
    is easier to tune across losses.
    """

    torch = _require_torch()
    z = latents.detach().clone().requires_grad_(True)
    loss = loss_fn(z)
    if loss.ndim != 0:
        loss = loss.mean()
    grad = torch.autograd.grad(loss, z)[0]
    grad_norm = torch.sqrt(torch.mean(grad.float() ** 2)).clamp_min(eps)
    step = grad / grad_norm if normalize else grad
    updated = (z - float(scale) * step).detach().to(dtype=latents.dtype)
    return GuidanceStepResult(
        latents=updated,
        loss=float(loss.detach().cpu()),
        grad_norm=float(grad_norm.detach().cpu()),
    )


def combine_guidance_losses(*weighted_losses: tuple[float, Callable[[Any], Any]]) -> Callable[[Any], Any]:
    """Combine weighted differentiable losses into one loss function."""

    if not weighted_losses:
        raise ValueError("at least one weighted loss is required")

    def loss_fn(latents: Any) -> Any:
        total = None
        for weight, fn in weighted_losses:
            value = float(weight) * fn(latents)
            total = value if total is None else total + value
        return total

    return loss_fn


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for guidance helpers.") from exc
    return torch
