"""Selective SAME channel/time renoise and donor-graft helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class LatentMaskSpec:
    """A reproducible latent-channel selection recipe."""

    name: str
    mode: str = "random_channels"
    fraction: float = 0.25
    seed: int = 0
    channels: tuple[int, ...] | None = None
    start_channel: int | None = None
    block_size: int | None = None


@dataclass(frozen=True, slots=True)
class SelectiveRenoiseResult:
    """Latents and metadata for one selective renoise experiment."""

    sampled_latents: Any
    init_latents: Any
    mixed_latents: Any
    selected_channels: list[int]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LatentGraftResult:
    """Latents and metadata for one cross-audio channel graft experiment."""

    sampled_latents: Any
    init_latents: Any
    donor_latents: Any
    mixed_latents: Any
    selected_channels: list[int]
    metadata: dict[str, Any]


def select_latent_channels(latents: Any, spec: LatentMaskSpec) -> list[int]:
    """Select SAME latent channels using a deterministic mask recipe."""

    torch = _require_torch()
    tensor = _as_bct(latents, torch).detach().float().cpu()
    channel_count = int(tensor.shape[1])
    if channel_count <= 0:
        raise ValueError("latents must have at least one channel")

    if spec.channels is not None:
        selected = sorted({int(channel) for channel in spec.channels})
        if not selected:
            raise ValueError("explicit channel list is empty")
        if selected[0] < 0 or selected[-1] >= channel_count:
            raise ValueError(f"channel index out of range for {channel_count} channels")
        return selected

    count = _selection_count(channel_count, spec.fraction, spec.block_size)
    mode = spec.mode.lower()
    if mode in {"random", "random_channels"}:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(spec.seed)
        return sorted(torch.randperm(channel_count, generator=generator)[:count].tolist())
    if mode in {"high_variance", "high_var"}:
        scores = tensor.var(dim=(0, 2), unbiased=False)
        return sorted(torch.topk(scores, k=count, largest=True).indices.tolist())
    if mode in {"low_variance", "low_var"}:
        scores = tensor.var(dim=(0, 2), unbiased=False)
        return sorted(torch.topk(scores, k=count, largest=False).indices.tolist())
    if mode in {"high_activity", "activity"}:
        scores = tensor.abs().mean(dim=(0, 2))
        return sorted(torch.topk(scores, k=count, largest=True).indices.tolist())
    if mode in {"low_activity"}:
        scores = tensor.abs().mean(dim=(0, 2))
        return sorted(torch.topk(scores, k=count, largest=False).indices.tolist())
    if mode in {"channel_block", "block"}:
        block_size = spec.block_size or count
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        start = spec.start_channel
        if start is None:
            start = (spec.seed * block_size) % channel_count
        return sorted(((start + offset) % channel_count) for offset in range(min(block_size, channel_count)))
    if mode in {"every_n", "comb"}:
        step = max(1, round(1.0 / max(spec.fraction, 1e-6)))
        offset = spec.seed % step
        return list(range(offset, channel_count, step))[:count]
    raise ValueError(f"unknown latent mask mode: {spec.mode}")


def channel_mask_like(latents: Any, channels: Sequence[int], *, level: float = 1.0) -> Any:
    """Create a broadcastable channel mask shaped ``1 x C x 1``."""

    torch = _require_torch()
    tensor = _as_bct(latents, torch)
    mask = torch.zeros((1, tensor.shape[1], 1), device=tensor.device, dtype=tensor.dtype)
    if channels:
        mask[:, list(channels), :] = level
    return mask


def masked_latent_noise(
    init_latents: Any,
    channels: Sequence[int],
    *,
    sigma: float,
    seed: int = 0,
) -> Any:
    """Mix Gaussian noise into selected latent channels only."""

    torch = _require_torch()
    init = _as_bct(init_latents, torch)
    mask = channel_mask_like(init, channels, level=float(sigma))
    generator = torch.Generator(device=init.device)
    generator.manual_seed(seed)
    noise = torch.randn(init.shape, device=init.device, dtype=init.dtype, generator=generator)
    return (1 - mask) * init + mask * noise


def sampler_noise_for_channels(
    init_latents: Any,
    channels: Sequence[int],
    *,
    seed: int = 0,
) -> Any:
    """Build the noise tensor passed to SA3's sampler for channel-selective variation.

    ``sample_diffusion`` later computes:

    ``start = init_data * (1 - sigma) + noise * sigma``

    If unselected channels in ``noise`` equal ``init_data``, those channels start
    unchanged while selected channels start with the usual renoise mixture.
    """

    torch = _require_torch()
    init = _as_bct(init_latents, torch)
    mask = channel_mask_like(init, channels, level=1.0)
    generator = torch.Generator(device=init.device)
    generator.manual_seed(seed)
    noise = torch.randn(init.shape, device=init.device, dtype=init.dtype, generator=generator)
    return (1 - mask) * init + mask * noise


def graft_latent_channels(
    source_latents: Any,
    donor_latents: Any,
    channels: Sequence[int],
    *,
    amount: float = 1.0,
) -> Any:
    """Mix donor latents into selected source channels.

    ``amount=1`` is a direct channel transplant. Smaller values make a linear
    source-to-donor interpolation only on the selected channels.
    """

    torch = _require_torch()
    source, donor = _paired_bct(source_latents, donor_latents, torch)
    mask = channel_mask_like(source, channels, level=float(amount))
    return (1 - mask) * source + mask * donor


def sampler_noise_from_donor_channels(
    source_latents: Any,
    donor_latents: Any,
    channels: Sequence[int],
) -> Any:
    """Build SA3 sampler noise from donor latents on selected channels.

    ``sample_diffusion`` later computes:

    ``start = source * (1 - sigma) + sampler_noise * sigma``

    so selected channels start at ``source + sigma * (donor - source)``, while
    unselected channels start unchanged.
    """

    torch = _require_torch()
    source, donor = _paired_bct(source_latents, donor_latents, torch)
    mask = channel_mask_like(source, channels, level=1.0)
    return (1 - mask) * source + mask * donor


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


def _selection_count(channel_count: int, fraction: float, block_size: int | None) -> int:
    if block_size is not None:
        return max(1, min(channel_count, int(block_size)))
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in the interval (0, 1]")
    return max(1, min(channel_count, int(round(channel_count * fraction))))


def _as_bct(latents: Any, torch):
    tensor = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"expected latents shaped B x C x T or C x T, got {tuple(tensor.shape)}")
    return tensor


def _paired_bct(source_latents: Any, donor_latents: Any, torch):
    source = _as_bct(source_latents, torch)
    donor = _as_bct(donor_latents, torch).to(device=source.device, dtype=source.dtype)
    if donor.shape[0] != source.shape[0]:
        if donor.shape[0] == 1:
            donor = donor.expand(source.shape[0], -1, -1)
        else:
            raise ValueError(f"batch mismatch: source {tuple(source.shape)}, donor {tuple(donor.shape)}")
    if donor.shape[1] != source.shape[1]:
        raise ValueError(f"channel mismatch: source {tuple(source.shape)}, donor {tuple(donor.shape)}")
    if donor.shape[2] != source.shape[2]:
        raise ValueError(f"time mismatch: source {tuple(source.shape)}, donor {tuple(donor.shape)}")
    return source, donor


def _cast_cond_inputs(cond_inputs: dict[str, Any], dtype, torch):
    return {
        key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in cond_inputs.items()
    }


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for selective latent renoise.") from exc
    return torch
