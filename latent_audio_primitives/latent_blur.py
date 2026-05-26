from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LatentBlurSpec:
    """A reproducible latent blur recipe."""

    name: str
    mode: str = "temporal"
    strength: float = 1.0
    temporal_radius: int = 4
    temporal_sigma: float | None = None
    temporal_kernel: str = "gaussian"
    temporal_direction: str = "centered"
    channel_radius: int = 2
    channel_sigma: float | None = None
    rank: int = 16
    detail_gain: float = 0.25


def apply_latent_blur(latents: Any, spec: LatentBlurSpec) -> Any:
    """Apply a latent blur recipe to ``B x C x T`` or ``C x T`` latents."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    mode = spec.mode.lower()
    if mode in {"temporal", "time", "time_blur", "frame_mix", "video_blur", "motion_blur"}:
        target = temporal_blur_latents(
            x,
            radius=spec.temporal_radius,
            sigma=spec.temporal_sigma,
            kernel=spec.temporal_kernel,
            direction=spec.temporal_direction,
        )
    elif mode in {"channel", "channel_blur"}:
        target = channel_blur_latents(x, radius=spec.channel_radius, sigma=spec.channel_sigma)
    elif mode in {"temporal_channel", "time_channel", "both"}:
        target = temporal_blur_latents(
            x,
            radius=spec.temporal_radius,
            sigma=spec.temporal_sigma,
            kernel=spec.temporal_kernel,
            direction=spec.temporal_direction,
        )
        target = channel_blur_latents(target, radius=spec.channel_radius, sigma=spec.channel_sigma)
    elif mode in {"low_rank", "pca", "svd"}:
        target = low_rank_latents(x, rank=spec.rank)
    elif mode in {"detail_attenuate", "detail", "soften"}:
        target = detail_attenuate_latents(
            x,
            radius=spec.temporal_radius,
            sigma=spec.temporal_sigma,
            detail_gain=spec.detail_gain,
        )
    elif mode in {"mean_blend", "time_mean", "static"}:
        target = x.mean(dim=-1, keepdim=True).expand_as(x)
    else:
        raise ValueError(f"unknown latent blur mode: {spec.mode}")

    return x + float(spec.strength) * (target - x)


def temporal_box_blur_latents(
    latents: Any,
    *,
    radius: int = 4,
    direction: str = "centered",
) -> Any:
    """Mix contiguous latent frames with equal weights."""

    return temporal_blur_latents(latents, radius=radius, kernel="box", direction=direction)


def temporal_blur_latents(
    latents: Any,
    *,
    radius: int = 4,
    sigma: float | None = None,
    kernel: str = "gaussian",
    direction: str = "centered",
) -> Any:
    """Blur along latent time only.

    ``kernel="box"`` is the literal contiguous-frame mix, analogous to a
    video box blur:

        z'[t] = mean(z[t-r : t+r+1])

    ``direction="past"`` or ``direction="future"`` makes the blur one-sided,
    which behaves more like simple motion blur.
    """

    torch = _require_torch()
    x = _as_bct(latents, torch)
    if radius <= 0:
        return x.clone()
    left, right = _temporal_padding(radius, direction)
    weights = _temporal_kernel(
        radius,
        sigma=sigma,
        kernel=kernel,
        direction=direction,
        device=x.device,
        dtype=x.dtype,
        torch=torch,
    )
    weight = weights.view(1, 1, -1).repeat(x.shape[1], 1, 1)
    padded = _pad_1d(x, left, right, torch=torch)
    return torch.nn.functional.conv1d(padded, weight, groups=x.shape[1])


def channel_blur_latents(latents: Any, *, radius: int = 2, sigma: float | None = None) -> Any:
    """Gaussian blur across latent channel index.

    SAME channels are learned, so adjacent channel index is not guaranteed to be
    semantically adjacent. This is still a useful probe for whether channel
    order has exploitable structure.
    """

    torch = _require_torch()
    x = _as_bct(latents, torch)
    if radius <= 0:
        return x.clone()
    batch, channels, frames = x.shape
    kernel = _gaussian_kernel(radius, sigma=sigma, device=x.device, dtype=x.dtype, torch=torch)
    folded = x.permute(0, 2, 1).reshape(batch * frames, 1, channels)
    padded = _pad_1d(folded, radius, radius, torch=torch)
    blurred = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1))
    return blurred.reshape(batch, frames, channels).permute(0, 2, 1)


def low_rank_latents(latents: Any, *, rank: int = 16) -> Any:
    """Low-rank SVD reconstruction over the time x channel latent matrix."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    if rank <= 0:
        raise ValueError("rank must be positive")
    outputs = []
    for item in x:
        time_major = item.transpose(0, 1).float()
        mean = time_major.mean(dim=0, keepdim=True)
        centered = time_major - mean
        u, s, vh = torch.linalg.svd(centered, full_matrices=False)
        keep = min(rank, s.shape[0])
        reconstructed = (u[:, :keep] * s[:keep]) @ vh[:keep, :]
        outputs.append((reconstructed + mean).transpose(0, 1).to(dtype=x.dtype))
    return torch.stack(outputs, dim=0)


def detail_attenuate_latents(
    latents: Any,
    *,
    radius: int = 4,
    sigma: float | None = None,
    detail_gain: float = 0.25,
) -> Any:
    """Attenuate temporal detail residuals after a latent low-pass."""

    x = _as_bct(latents, _require_torch())
    low = temporal_blur_latents(x, radius=radius, sigma=sigma)
    return low + float(detail_gain) * (x - low)


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
    """Use blurred latents as SA3 init_data and sample back toward the manifold."""

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


def _gaussian_kernel(radius: int, *, sigma: float | None, device, dtype, torch):
    if radius <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    sigma = float(sigma) if sigma is not None else max(radius / 2.0, 1e-6)
    positions = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(positions**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.to(dtype=dtype)


def _temporal_kernel(
    radius: int,
    *,
    sigma: float | None,
    kernel: str,
    direction: str,
    device,
    dtype,
    torch,
):
    if radius <= 0:
        return torch.ones(1, device=device, dtype=dtype)

    direction = _normalize_direction(direction)
    if direction == "centered":
        positions = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    elif direction == "past":
        positions = torch.arange(-radius, 1, device=device, dtype=torch.float32)
    else:
        positions = torch.arange(0, radius + 1, device=device, dtype=torch.float32)

    kernel = kernel.lower()
    if kernel in {"box", "mean", "average", "frame_mix", "video"}:
        weights = torch.ones_like(positions)
    elif kernel in {"triangle", "triangular", "linear"}:
        weights = radius + 1 - positions.abs()
        weights = weights.clamp_min(1.0)
    elif kernel in {"gaussian", "gauss"}:
        sigma = float(sigma) if sigma is not None else max(radius / 2.0, 1e-6)
        weights = torch.exp(-(positions**2) / (2 * sigma**2))
    else:
        raise ValueError(f"unknown temporal blur kernel: {kernel}")

    weights = weights / weights.sum().clamp_min(1e-12)
    return weights.to(dtype=dtype)


def _temporal_padding(radius: int, direction: str):
    direction = _normalize_direction(direction)
    if direction == "centered":
        return radius, radius
    if direction == "past":
        return radius, 0
    return 0, radius


def _normalize_direction(direction: str):
    direction = direction.lower()
    if direction in {"center", "centered", "symmetric", "both"}:
        return "centered"
    if direction in {"past", "trailing", "backward", "left", "causal"}:
        return "past"
    if direction in {"future", "leading", "forward", "right", "anti_causal"}:
        return "future"
    raise ValueError(f"unknown temporal blur direction: {direction}")


def _pad_1d(x: Any, left: int, right: int, *, torch):
    if left <= 0 and right <= 0:
        return x
    mode = "reflect" if x.shape[-1] > max(left, right) else "replicate"
    return torch.nn.functional.pad(x, (left, right), mode=mode)


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
        raise RuntimeError("PyTorch is required for latent blur.") from exc
    return torch
