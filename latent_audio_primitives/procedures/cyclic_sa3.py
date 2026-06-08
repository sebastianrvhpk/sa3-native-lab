"""SA3 sampler procedures with cyclic latent-time interventions."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from latent_audio_primitives.looping import cyclic_mix_latents, cyclic_roll_latents, frames_from_fraction


def sample_cyclic_roll_euler(
    model: Any,
    x: Any,
    sigmas: Any,
    *,
    roll_frames: int,
    mode: str = "alternate",
    roll_mix: float = 0.10,
    roll_mix_schedule: Any = None,
    mix_every_n: int = 1,
    symmetric_mix: bool = True,
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

    ``mode="cyclic_mix"`` is the most direct tiling-style intervention. After
    each Euler update it softly projects the latent state toward agreement with
    its half-rolled copy:

        x <- x + beta * (0.5 * (x + R_s x) - x)

    This mode changes the state even when the init-noise schedule has dt=0.

    ``roll_mix_schedule`` may provide a per-step mix strength. This lets
    residual-timestep cartography turn cyclic projection into a sampler-phase
    intervention instead of applying the same mix at every denoising step.

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
    if mode not in {"alternate", "paired_average", "cyclic_mix", "mix", "projection"}:
        raise ValueError("mode must be 'alternate', 'paired_average', or 'cyclic_mix'")
    roll_frames = int(roll_frames)
    mix_every_n = max(1, int(mix_every_n))
    net_shift = 0
    ones = state.new_ones([state.shape[0]])
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        step_roll_mix = _scheduled_roll_mix(roll_mix_schedule, step_index=i, default=roll_mix)
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
                    "roll_mix": step_roll_mix,
                    "net_shift_frames": net_shift,
                    "mode": mode,
                }
            )

        state = state + dt * velocity

        if mode == "alternate":
            state = cyclic_roll_latents(state, roll_frames)
            net_shift += roll_frames
        elif mode in {"cyclic_mix", "mix", "projection"} and i % mix_every_n == 0:
            state = cyclic_mix_latents(
                state,
                roll_frames,
                strength=step_roll_mix,
                symmetric=symmetric_mix,
            )

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
    roll_mix: float = 0.10,
    roll_mix_schedule: Any = None,
    mix_every_n: int = 1,
    symmetric_mix: bool = True,
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
            roll_mix=roll_mix,
            roll_mix_schedule=roll_mix_schedule,
            mix_every_n=mix_every_n,
            symmetric_mix=symmetric_mix,
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


def _scheduled_roll_mix(schedule: Any, *, step_index: int, default: float) -> float:
    if schedule is None:
        return float(default)
    if callable(schedule):
        return float(schedule(step_index, default))
    try:
        return float(schedule[step_index])
    except IndexError:
        return 0.0
    except KeyError:
        return 0.0


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for cyclic SA3 procedures.") from exc
    return torch
