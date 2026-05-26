from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tqdm import tqdm


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


def sample_cyclic_roll_euler(
    model: Any,
    x: Any,
    sigmas: Any,
    *,
    roll_frames: int,
    mode: str = "alternate",
    unroll_output: bool = True,
    callback: Any = None,
    disable_tqdm: bool = False,
    **extra_args: Any,
) -> Any:
    """Euler rectified-flow sampler with cyclic time-roll interventions.

    ``mode="alternate"`` literally rolls the latent state after every Euler
    update. With a half-length roll, the model sees alternating temporal
    origins across denoising steps:

        x <- R_s(x + dt * v_theta(x, t, c))

    ``mode="paired_average"`` keeps the state in one orientation but evaluates
    the model under two temporal origins at each step:

        v_cyc = 0.5 * (v_theta(x, t, c) + R_-s v_theta(R_s x, t, c))

    This helper is intentionally Euler-only. It is an experimental probe, not a
    replacement for SA3's full sampler zoo.
    """

    torch = _require_torch()
    state = _as_bct(x, torch)
    t = _as_tensor(sigmas, torch).to(state.device)
    if t.ndim not in {1, 2}:
        raise ValueError("sample_cyclic_roll_euler expects a 1D or 2D sigma schedule")
    per_element_schedule = t.ndim == 2
    if per_element_schedule and t.shape[0] != state.shape[0]:
        raise ValueError("2D sigma schedule batch dimension must match latent batch dimension")
    mode = mode.lower()
    if mode not in {"alternate", "paired_average"}:
        raise ValueError("mode must be 'alternate' or 'paired_average'")
    roll_frames = int(roll_frames)
    net_shift = 0
    ones = state.new_ones([state.shape[0]])
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element_schedule:
            t_curr_tensor = t[:, i].to(dtype=state.dtype)
            t_next_tensor = t[:, i + 1].to(dtype=state.dtype)
            t_broadcast = t_curr_tensor.view(-1, 1, 1)
            dt = (t_next_tensor - t_curr_tensor).view(-1, 1, 1)
        else:
            t_curr = t[i].to(dtype=state.dtype)
            t_next = t[i + 1].to(dtype=state.dtype)
            t_curr_tensor = t_curr * ones
            t_broadcast = t_curr
            dt = t_next - t_curr

        if mode == "paired_average":
            direct_v = model(state, t_curr_tensor, **extra_args)
            rolled_state = cyclic_roll_latents(state, roll_frames)
            rolled_v = model(rolled_state, t_curr_tensor, **extra_args)
            velocity = 0.5 * (direct_v + cyclic_roll_latents(rolled_v, -roll_frames))
        else:
            velocity = model(state, t_curr_tensor, **extra_args)

        denoised = state - t_broadcast * velocity
        if callback is not None:
            callback(
                {
                    "x": state,
                    "i": i,
                    "t": t_curr_tensor,
                    "sigma": t_curr_tensor,
                    "denoised": denoised,
                    "roll_frames": roll_frames,
                    "net_shift_frames": net_shift,
                    "mode": mode,
                }
            )

        state = state + dt * velocity

        if mode == "alternate":
            state = cyclic_roll_latents(state, roll_frames)
            net_shift += roll_frames

    if mode == "alternate" and unroll_output and net_shift:
        state = cyclic_roll_latents(state, -net_shift)
    return state


def sa3_cyclic_roll_sample_from_init_latents(
    stable_model: Any,
    init_latents: Any,
    *,
    prompt: str = "",
    duration: float = 9.0,
    steps: int = 20,
    cfg_scale: float = 1.0,
    init_noise_level: float = 0.35,
    roll_fraction: float = 0.5,
    roll_frames: int | None = None,
    mode: str = "alternate",
    unroll_output: bool = True,
    seed: int = 0,
    negative_prompt: str | None = None,
    duration_padding_sec: float = 6.0,
    apg_scale: float = 1.0,
    dist_shift: Any = None,
    callback: Any = None,
    disable_tqdm: bool = False,
) -> Any:
    """Run SA3 audio-to-audio sampling with cyclic rolls inside each step.

    This starts from the usual variation state:

        x_T = (1 - sigma) z_audio + sigma * eps

    then integrates the rectified-flow sampler with ``sample_cyclic_roll_euler``.
    It does not use inpainting, continuation masks, or boundary repair.
    """

    torch = _require_torch()
    from stable_audio_3.data.utils import (
        compute_effective_seq_len_from_conditioning,
        create_padding_mask_from_lengths,
    )
    from stable_audio_3.inference.sampling import build_schedule

    device = str(stable_model.device)
    core = stable_model.model
    if core.diffusion_objective not in {"rectified_flow", "rf_denoiser"}:
        raise ValueError(f"cyclic roll sampling only supports RF objectives, got {core.diffusion_objective}")

    model_dtype = next(core.model.parameters()).dtype
    init = _as_bct(init_latents, torch).to(device=device, dtype=model_dtype)
    roll_frames = frames_from_fraction(init, roll_fraction) if roll_frames is None else int(roll_frames)
    conditioning, negative_conditioning = stable_model._build_conditioning_dicts(
        prompt,
        negative_prompt,
        duration,
        batch_size=init.shape[0],
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(init.shape, device=device, dtype=model_dtype, generator=generator)
    sigma_max = float(init_noise_level)
    state = init * (1.0 - sigma_max) + noise * sigma_max

    conditioning_tensors = core.conditioner(conditioning, device)
    negative_conditioning_tensors = {}
    if negative_conditioning is not None:
        negative_conditioning_tensors = core.conditioner(negative_conditioning, device)

    mask = torch.zeros((init.shape[0], 1, init.shape[-1]), device=device)
    inpaint_input = torch.zeros_like(init)
    conditioning_tensors["inpaint_mask"] = [mask]
    conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
    conditioning_inputs = core.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors["inpaint_mask"] = [mask]
        negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        negative_conditioning_tensors = core.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

    downsampling_ratio = core.pretransform.downsampling_ratio if core.pretransform is not None else 1
    effective_seq_len = compute_effective_seq_len_from_conditioning(
        conditioning,
        core.sample_rate,
        downsampling_ratio,
        device,
    )
    padding_mask = None
    if effective_seq_len is not None:
        headroom_tokens = int(duration_padding_sec * core.sample_rate / downsampling_ratio)
        valid_lengths = (effective_seq_len + headroom_tokens).clamp(max=init.shape[-1]).long()
        padding_mask = create_padding_mask_from_lengths(valid_lengths, init.shape[-1])

    sigmas = build_schedule(
        steps=steps,
        sigma_max=sigma_max,
        dist_shift=dist_shift if dist_shift is not None else core.sampling_dist_shift,
        effective_seq_len=effective_seq_len,
        fallback_seq_len=init.shape[-1],
        include_endpoint=True,
        device=device,
    )

    common_kwargs = {
        **_cast_cond_inputs(conditioning_inputs, model_dtype, torch),
        **_cast_cond_inputs(negative_conditioning_tensors, model_dtype, torch),
        "cfg_scale": cfg_scale,
        "batch_cfg": True,
        "rescale_cfg": True,
        "padding_mask": padding_mask,
        "apg_scale": apg_scale,
    }

    with torch.inference_mode():
        return sample_cyclic_roll_euler(
            core.model,
            state,
            sigmas,
            roll_frames=roll_frames,
            mode=mode,
            unroll_output=unroll_output,
            callback=callback,
            disable_tqdm=disable_tqdm,
            **common_kwargs,
        )


def _as_bct(latents: Any, torch):
    tensor = _as_tensor(latents, torch)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"expected latents shaped B x C x T or C x T, got {tuple(tensor.shape)}")
    return tensor


def _as_tensor(value: Any, torch):
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _cast_cond_inputs(cond_inputs: dict[str, Any], dtype, torch):
    return {
        key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in cond_inputs.items()
    }


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for loop helpers.") from exc
    return torch
