"""SA3 procedures for selective SAME-channel renoise and graft experiments."""

from __future__ import annotations

from typing import Any

from latent_audio_primitives.selective_renoise import (
    LatentGraftResult,
    LatentMaskSpec,
    SelectiveRenoiseResult,
    sampler_noise_for_channels,
    sampler_noise_from_donor_channels,
    select_latent_channels,
)


def selective_renoise_sa3(
    stable_model: Any,
    audio: Any,
    sample_rate: int,
    *,
    spec: LatentMaskSpec,
    prompt: str = "",
    duration: float = 9.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    init_noise_level: float = 0.4,
    seed: int = 0,
    negative_prompt: str | None = None,
    sample_size: int = 5292032,
    duration_padding_sec: float = 6.0,
    apg_scale: float = 1.0,
    dist_shift: Any = None,
    sampler_type: str | None = None,
    return_mixed_latents: bool = True,
    **sampler_kwargs: Any,
) -> SelectiveRenoiseResult:
    """Run SA3 variation with noise injected only into selected latent channels."""

    torch = _require_torch()
    from stable_audio_3.inference.sampling import sample_diffusion

    device = str(stable_model.device)
    conditioning, negative_conditioning = stable_model._build_conditioning_dicts(
        prompt,
        negative_prompt,
        duration,
        batch_size=1,
    )
    audio_sample_size = stable_model._adapt_sample_size(
        conditioning,
        sample_size,
        duration_padding_sec,
    )
    init_latents, _inpaint_mask = stable_model._encode_audio_input(
        (sample_rate, audio),
        audio_sample_size,
        inpaint_mask=None,
    )

    core = stable_model.model
    model_dtype = next(core.model.parameters()).dtype
    init_latents = init_latents.to(device=device, dtype=model_dtype)
    selected_channels = select_latent_channels(init_latents, spec)
    sampler_noise = sampler_noise_for_channels(init_latents, selected_channels, seed=seed).to(dtype=model_dtype)
    mixed_latents = init_latents
    if return_mixed_latents:
        mixed_latents = init_latents * (1 - init_noise_level) + sampler_noise * init_noise_level

    conditioning_tensors = core.conditioner(conditioning, device)
    negative_conditioning_tensors = {}
    if negative_conditioning is not None:
        negative_conditioning_tensors = core.conditioner(negative_conditioning, device)

    latent_frames = init_latents.shape[-1]
    mask = torch.zeros((1, 1, latent_frames), device=device)
    inpaint_input = torch.zeros_like(init_latents)
    conditioning_tensors["inpaint_mask"] = [mask]
    conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
    conditioning_inputs = core.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors["inpaint_mask"] = [mask]
        negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        negative_conditioning_tensors = core.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

    conditioning_inputs = _cast_cond_inputs(conditioning_inputs, model_dtype, torch)
    negative_conditioning_tensors = _cast_cond_inputs(negative_conditioning_tensors, model_dtype, torch)
    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

    with torch.inference_mode():
        sampled_latents = sample_diffusion(
            model=core.model,
            noise=sampler_noise,
            cond_inputs=cond_inputs,
            diffusion_objective=core.diffusion_objective,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_rate=core.sample_rate,
            pretransform=core.pretransform,
            mask_padding_attention=True,
            use_effective_length_for_schedule=True,
            headroom_seconds=duration_padding_sec,
            dist_shift=dist_shift if dist_shift is not None else core.sampling_dist_shift,
            sampler_type=sampler_type,
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=apg_scale,
            init_data=init_latents,
            init_noise_level=init_noise_level,
            decode=False,
            **sampler_kwargs,
        )

    return SelectiveRenoiseResult(
        sampled_latents=sampled_latents,
        init_latents=init_latents,
        mixed_latents=mixed_latents,
        selected_channels=selected_channels,
        metadata={
            "spec": {
                "name": spec.name,
                "mode": spec.mode,
                "fraction": spec.fraction,
                "seed": spec.seed,
                "channels": list(spec.channels) if spec.channels is not None else None,
                "start_channel": spec.start_channel,
                "block_size": spec.block_size,
            },
            "prompt": prompt,
            "duration": duration,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "init_noise_level": init_noise_level,
            "seed": seed,
            "selected_channel_count": len(selected_channels),
            "selected_channels": selected_channels,
        },
    )


def selective_graft_sa3(
    stable_model: Any,
    source_audio: Any,
    source_sample_rate: int,
    donor_audio: Any,
    donor_sample_rate: int,
    *,
    spec: LatentMaskSpec,
    prompt: str = "",
    duration: float = 9.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    init_noise_level: float = 0.4,
    seed: int = 0,
    negative_prompt: str | None = None,
    sample_size: int = 5292032,
    duration_padding_sec: float = 6.0,
    apg_scale: float = 1.0,
    dist_shift: Any = None,
    sampler_type: str | None = None,
    return_mixed_latents: bool = True,
    **sampler_kwargs: Any,
) -> LatentGraftResult:
    """Run SA3 variation with selected latent channels borrowed from donor audio."""

    torch = _require_torch()
    from stable_audio_3.inference.sampling import sample_diffusion

    device = str(stable_model.device)
    conditioning, negative_conditioning = stable_model._build_conditioning_dicts(
        prompt,
        negative_prompt,
        duration,
        batch_size=1,
    )
    audio_sample_size = stable_model._adapt_sample_size(
        conditioning,
        sample_size,
        duration_padding_sec,
    )
    init_latents, _source_mask = stable_model._encode_audio_input(
        (source_sample_rate, source_audio),
        audio_sample_size,
        inpaint_mask=None,
    )
    donor_latents, _donor_mask = stable_model._encode_audio_input(
        (donor_sample_rate, donor_audio),
        audio_sample_size,
        inpaint_mask=None,
    )

    core = stable_model.model
    model_dtype = next(core.model.parameters()).dtype
    init_latents = init_latents.to(device=device, dtype=model_dtype)
    donor_latents = donor_latents.to(device=device, dtype=model_dtype)
    selected_channels = select_latent_channels(init_latents, spec)
    sampler_noise = sampler_noise_from_donor_channels(init_latents, donor_latents, selected_channels).to(
        dtype=model_dtype
    )
    mixed_latents = init_latents
    if return_mixed_latents:
        mixed_latents = init_latents * (1 - init_noise_level) + sampler_noise * init_noise_level

    conditioning_tensors = core.conditioner(conditioning, device)
    negative_conditioning_tensors = {}
    if negative_conditioning is not None:
        negative_conditioning_tensors = core.conditioner(negative_conditioning, device)

    latent_frames = init_latents.shape[-1]
    mask = torch.zeros((1, 1, latent_frames), device=device)
    inpaint_input = torch.zeros_like(init_latents)
    conditioning_tensors["inpaint_mask"] = [mask]
    conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
    conditioning_inputs = core.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning_tensors:
        negative_conditioning_tensors["inpaint_mask"] = [mask]
        negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        negative_conditioning_tensors = core.get_conditioning_inputs(negative_conditioning_tensors, negative=True)

    conditioning_inputs = _cast_cond_inputs(conditioning_inputs, model_dtype, torch)
    negative_conditioning_tensors = _cast_cond_inputs(negative_conditioning_tensors, model_dtype, torch)
    cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

    with torch.inference_mode():
        sampled_latents = sample_diffusion(
            model=core.model,
            noise=sampler_noise,
            cond_inputs=cond_inputs,
            diffusion_objective=core.diffusion_objective,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_rate=core.sample_rate,
            pretransform=core.pretransform,
            mask_padding_attention=True,
            use_effective_length_for_schedule=True,
            headroom_seconds=duration_padding_sec,
            dist_shift=dist_shift if dist_shift is not None else core.sampling_dist_shift,
            sampler_type=sampler_type,
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=apg_scale,
            init_data=init_latents,
            init_noise_level=init_noise_level,
            decode=False,
            **sampler_kwargs,
        )

    return LatentGraftResult(
        sampled_latents=sampled_latents,
        init_latents=init_latents,
        donor_latents=donor_latents,
        mixed_latents=mixed_latents,
        selected_channels=selected_channels,
        metadata={
            "spec": {
                "name": spec.name,
                "mode": spec.mode,
                "fraction": spec.fraction,
                "seed": spec.seed,
                "channels": list(spec.channels) if spec.channels is not None else None,
                "start_channel": spec.start_channel,
                "block_size": spec.block_size,
            },
            "prompt": prompt,
            "duration": duration,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "init_noise_level": init_noise_level,
            "seed": seed,
            "selected_channel_count": len(selected_channels),
            "selected_channels": selected_channels,
            "intervention": "donor_channel_graft",
        },
    )


def _cast_cond_inputs(cond_inputs: dict[str, Any], dtype, torch):
    return {
        key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in cond_inputs.items()
    }


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for selective SA3 procedures.") from exc
    return torch
