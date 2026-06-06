"""SA3 procedures for polishing or resampling edited SAME latents."""

from __future__ import annotations

from typing import Any


def sa3_sample_from_init_latents(
    stable_model: Any,
    init_latents: Any,
    *,
    prompt: str = "",
    duration: float = 9.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    init_noise_level: float = 0.12,
    seed: int = 0,
    negative_prompt: str | None = None,
    duration_padding_sec: float = 6.0,
    apg_scale: float = 1.0,
    dist_shift: Any = None,
    sampler_type: str | None = None,
    **sampler_kwargs: Any,
) -> Any:
    """Use edited latents as SA3 init_data and sample back toward the manifold."""

    torch = _require_torch()
    from stable_audio_3.inference.sampling import sample_diffusion

    device = str(stable_model.device)
    core = stable_model.model
    model_dtype = next(core.model.parameters()).dtype
    init = _as_bct(init_latents, torch).to(device=device, dtype=model_dtype)
    conditioning, negative_conditioning = stable_model._build_conditioning_dicts(
        prompt,
        negative_prompt,
        duration,
        batch_size=init.shape[0],
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(init.shape, device=device, dtype=model_dtype, generator=generator)
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

    conditioning_inputs = _cast_cond_inputs(conditioning_inputs, model_dtype, torch)
    negative_conditioning_tensors = _cast_cond_inputs(negative_conditioning_tensors, model_dtype, torch)

    with torch.inference_mode():
        return sample_diffusion(
            model=core.model,
            noise=noise,
            cond_inputs={**conditioning_inputs, **negative_conditioning_tensors},
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
            init_data=init,
            init_noise_level=init_noise_level,
            decode=False,
            **sampler_kwargs,
        )


def _as_bct(latents: Any, torch):
    tensor = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"expected latents shaped B x C x T or C x T, got {tuple(tensor.shape)}")
    return tensor


def _cast_cond_inputs(cond_inputs: dict[str, Any], dtype, torch):
    return {
        key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in cond_inputs.items()
    }


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for SA3 latent sampling procedures.") from exc
    return torch
