from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LoopBoundaryMetrics:
    """Simple boundary diagnostics for loop experiments."""

    state_l2: float
    velocity_l2: float
    window_frames: int

    @property
    def total(self) -> float:
        return self.state_l2 + self.velocity_l2


def cyclic_roll_latents(latents: Any, shift_frames: int) -> Any:
    """Cyclically roll SAME/SA3 latents along the time axis."""

    torch = _require_torch()
    x = _as_tensor(latents, torch)
    if x.shape[-1] == 0:
        return x.clone()
    return torch.roll(x, shifts=int(shift_frames), dims=-1)


def cyclic_roll_audio(audio: Any, shift_samples: int) -> Any:
    """Cyclically roll an audio tensor along the sample axis."""

    torch = _require_torch()
    x = _as_tensor(audio, torch)
    if x.shape[-1] == 0:
        return x.clone()
    return torch.roll(x, shifts=int(shift_samples), dims=-1)


def repeated_loop_preview_audio(audio: Any, repeats: int = 4) -> Any:
    """Concatenate repeated copies of an audio tensor for seam listening."""

    torch = _require_torch()
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    x = _as_tensor(audio, torch)
    return torch.cat([x] * int(repeats), dim=-1)


def loop_boundary_metrics(latents: Any, *, window_frames: int = 8) -> LoopBoundaryMetrics:
    """Measure start/end latent state and velocity mismatch."""

    torch = _require_torch()
    x = _as_bct(latents, torch).detach().float()
    frames = int(x.shape[-1])
    if frames < 2:
        return LoopBoundaryMetrics(state_l2=0.0, velocity_l2=0.0, window_frames=0)
    k = max(1, min(int(window_frames), frames // 2))
    start = x[..., :k].mean(dim=-1)
    end = x[..., -k:].mean(dim=-1)
    state_l2 = torch.linalg.vector_norm(start - end, dim=-1).mean().item()

    if k < 2:
        velocity_l2 = 0.0
    else:
        start_velocity = (x[..., 1:k] - x[..., : k - 1]).mean(dim=-1)
        end_velocity = (x[..., -k + 1 :] - x[..., -k:-1]).mean(dim=-1)
        velocity_l2 = torch.linalg.vector_norm(start_velocity - end_velocity, dim=-1).mean().item()
    return LoopBoundaryMetrics(state_l2=state_l2, velocity_l2=velocity_l2, window_frames=k)


def seam_inpaint_bounds(duration: float, shift_fraction: float, window_seconds: float) -> tuple[float, float]:
    """Return an inpaint window around the rolled loop seam."""

    duration = float(duration)
    center = duration * float(shift_fraction)
    half = float(window_seconds) / 2.0
    return max(0.0, center - half), min(duration, center + half)


def frames_from_fraction(latents: Any, shift_fraction: float) -> int:
    """Convert a cyclic shift fraction to an integer latent-frame shift."""

    torch = _require_torch()
    x = _as_tensor(latents, torch)
    return int(round(x.shape[-1] * float(shift_fraction)))


def samples_from_fraction(audio: Any, shift_fraction: float) -> int:
    """Convert a cyclic shift fraction to an integer audio-sample shift."""

    torch = _require_torch()
    x = _as_tensor(audio, torch)
    return int(round(x.shape[-1] * float(shift_fraction)))


def _as_bct(latents: Any, torch):
    tensor = _as_tensor(latents, torch)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"expected latents shaped B x C x T or C x T, got {tuple(tensor.shape)}")
    return tensor


def _as_tensor(value: Any, torch):
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for loop helpers.") from exc
    return torch
