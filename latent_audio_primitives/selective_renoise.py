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
    grafted_latents: Any | None = None


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
    """Build legacy SA3 sampler noise from donor latents on selected channels.

    This treats donor latents as the sampler ``noise`` argument, not as a
    deterministic edited ``init_data``. It is useful as a diagnostic of prior
    pull, but direct donor-channel grafting should usually use
    ``graft_latent_channels`` first and then polish the grafted init latents.

    ``sample_diffusion`` later computes approximately:

    ``start = source * (1 - sigma) + sampler_noise * sigma``

    so selected channels start at ``source + sigma * (donor - source)``, while
    unselected channels start unchanged.
    """

    torch = _require_torch()
    source, donor = _paired_bct(source_latents, donor_latents, torch)
    mask = channel_mask_like(source, channels, level=1.0)
    return (1 - mask) * source + mask * donor


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


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for selective latent renoise.") from exc
    return torch
