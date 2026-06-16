"""Level 3: Tuning-steered rectified-flow sampling procedures."""

from __future__ import annotations

from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for tuning-steered sampling.") from exc
    return torch


def latent_periodicity_loss(x: Any, target_lag_frames: float) -> Any:
    """Compute differentiable normalized autocorrelation loss at target_lag_frames."""
    torch = _require_torch()
    B, C, T = x.shape
    x_centered = x - x.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(x_centered.float(), dim=-1)
    psd = (spectrum * spectrum.conj()).real

    bins = torch.arange(spectrum.shape[-1], device=x.device, dtype=torch.float32)
    phase = 2.0 * torch.pi * bins * float(target_lag_frames) / T

    r_tau = (psd * torch.cos(phase).view(1, 1, -1)).sum(dim=-1)
    r_0 = psd.sum(dim=-1)

    autocorr = r_tau / r_0.clamp_min(1e-8)
    return -autocorr.mean()


def sample_tuning_steered_euler(
    model: Any,
    x: Any,
    sigmas: Any,
    *,
    target_frequency_hz: float,
    latent_rate: float,
    steering_scale: float = 1.0,
    disable_tqdm: bool = False,
    **extra_args: Any,
) -> Any:
    """Euler rectified-flow sampler steered toward target_frequency_hz pitch targets."""
    torch = _require_torch()
    state = x.clone()
    t = torch.as_tensor(sigmas).to(state.device)

    target_lag_frames = float(latent_rate) / float(target_frequency_hz)
    ones = state.new_ones([state.shape[0]])
    num_steps = t.shape[-1] - 1

    # Ensure steering_scale is float
    scale = float(steering_scale)

    for i in range(num_steps):
        t_curr = t[i].to(dtype=state.dtype)
        t_next = t[i + 1].to(dtype=state.dtype)
        t_curr_tensor = t_curr * ones
        dt = t_next - t_curr

        if scale > 0.0:
            # Differentiable forward step for guidance calculation
            state_with_grad = state.detach().requires_grad_(True)
            velocity = model(state_with_grad, t_curr_tensor, **extra_args)
            denoised = state_with_grad - t_curr * velocity
            loss = latent_periodicity_loss(denoised, target_lag_frames)
            grad = torch.autograd.grad(loss, state_with_grad)[0]
            steering_step = -scale * grad
        else:
            steering_step = 0.0

        # Update using regular velocity
        with torch.no_grad():
            velocity = model(state, t_curr_tensor, **extra_args)
            state = state + dt * velocity + steering_step

    return state
