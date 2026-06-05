"""DSP-like operations over learned SAME latent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class LatentDSPSpec:
    """A reproducible neural-latent DSP operation.

    Operations expect SAME/SA3 latents as ``B x C x T`` or ``C x T`` tensors.
    They process the low-rate learned latent trajectories, not waveform samples.
    """

    name: str
    mode: str = "gain"
    strength: float = 1.0
    gain: float = 1.0
    center: str = "channel_mean"
    threshold: float = 1.0
    ratio: float = 4.0
    makeup_gain: float = 1.0
    drive: float = 1.0
    ceiling: float = 2.0
    fft_low_cutoff: float = 0.15
    fft_high_cutoff: float = 0.65
    fft_low_gain: float = 1.0
    fft_mid_gain: float = 1.0
    fft_high_gain: float = 1.0
    phase_shift_fraction: float = 0.0
    phase_random_amount: float = 1.0
    phase_blend_amount: float = 1.0
    magnitude_amount: float = 1.0
    pca_rank: int | None = None
    pca_component_gains: tuple[float, ...] | None = None
    seed: int = 0


def apply_latent_dsp(latents: Any, spec: LatentDSPSpec, *, donor_latents: Any | None = None) -> Any:
    """Apply a latent-DSP spec and blend it against the input by ``strength``."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    mode = spec.mode.lower()

    if mode in {"gain", "latent_gain"}:
        target = latent_gain(x, gain=spec.gain, center=spec.center)
    elif mode in {"compress", "dynamics", "dynamics_compress", "latent_compress"}:
        target = latent_dynamics(
            x,
            threshold=spec.threshold,
            ratio=spec.ratio,
            mode="compress",
            center=spec.center,
            makeup_gain=spec.makeup_gain,
        )
    elif mode in {"expand", "dynamics_expand", "latent_expand"}:
        target = latent_dynamics(
            x,
            threshold=spec.threshold,
            ratio=spec.ratio,
            mode="expand",
            center=spec.center,
            makeup_gain=spec.makeup_gain,
        )
    elif mode in {"softclip", "saturate", "tanh", "soft_clip"}:
        target = latent_soft_clip(
            x,
            drive=spec.drive,
            ceiling=spec.ceiling,
            center=spec.center,
            makeup_gain=spec.makeup_gain,
        )
    elif mode in {"fft_eq", "latent_eq", "fft_shelf", "spectral_eq"}:
        target = latent_fft_eq(
            x,
            low_cutoff=spec.fft_low_cutoff,
            high_cutoff=spec.fft_high_cutoff,
            low_gain=spec.fft_low_gain,
            mid_gain=spec.fft_mid_gain,
            high_gain=spec.fft_high_gain,
        )
    elif mode in {"fft_phase_shift", "phase_shift", "latent_phase_shift"}:
        target = latent_fft_phase_shift(x, shift_fraction=spec.phase_shift_fraction)
    elif mode in {"fft_phase_randomize", "phase_randomize", "phase_scramble"}:
        target = latent_fft_phase_randomize(
            x,
            amount=spec.phase_random_amount,
            seed=spec.seed,
        )
    elif mode in {"fft_phase_blend", "phase_blend"}:
        donor = _require_donor(donor_latents, torch)
        target = latent_fft_phase_blend(
            x,
            donor,
            amount=spec.phase_blend_amount,
        )
    elif mode in {"fft_mag_phase_graft", "mag_phase_graft", "magnitude_from_donor", "donor_magnitude"}:
        donor = _require_donor(donor_latents, torch)
        target = latent_fft_magnitude_phase_graft(
            magnitude_latents=donor,
            phase_latents=x,
            magnitude_amount=spec.magnitude_amount,
        )
    elif mode in {"fft_phase_from_donor", "donor_phase"}:
        donor = _require_donor(donor_latents, torch)
        target = latent_fft_phase_blend(
            x,
            donor,
            amount=1.0,
        )
    elif mode in {"pca_gain", "component_gain", "latent_pca_eq"}:
        target = pca_component_gain(
            x,
            component_gains=spec.pca_component_gains,
            rank=spec.pca_rank,
        )
    else:
        raise ValueError(f"unknown latent DSP mode: {spec.mode}")

    return _blend(x, target.to(dtype=x.dtype), spec.strength)


def latent_gain(latents: Any, *, gain: float = 1.0, center: str = "channel_mean") -> Any:
    """Scale latent excursions around a center."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    c = _center(x, center)
    return c + float(gain) * (x - c)


def latent_dynamics(
    latents: Any,
    *,
    threshold: float = 1.0,
    ratio: float = 4.0,
    mode: str = "compress",
    center: str = "channel_mean",
    makeup_gain: float = 1.0,
    eps: float = 1e-8,
) -> Any:
    """Compressor/expander over normalized latent excursions.

    ``threshold`` is in per-channel standard-deviation units. Compression maps
    excursions above threshold through slope ``1 / ratio``. Expansion maps them
    through slope ``ratio``.
    """

    torch = _require_torch()
    x = _as_bct(latents, torch)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if ratio <= 0:
        raise ValueError("ratio must be positive")

    c = _center(x, center)
    residual = x - c
    scale = residual.float().std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    magnitude = residual.float().abs() / scale
    threshold_tensor = torch.full_like(magnitude, float(threshold))
    above = magnitude > threshold_tensor

    mode = mode.lower()
    if mode == "compress":
        shaped = threshold_tensor + (magnitude - threshold_tensor) / float(ratio)
    elif mode == "expand":
        shaped = threshold_tensor + (magnitude - threshold_tensor) * float(ratio)
    else:
        raise ValueError("mode must be 'compress' or 'expand'")

    shaped = torch.where(above, shaped, magnitude)
    gain = shaped / magnitude.clamp_min(eps)
    target = c.float() + float(makeup_gain) * residual.float() * gain
    return target.to(dtype=x.dtype)


def latent_soft_clip(
    latents: Any,
    *,
    drive: float = 1.0,
    ceiling: float = 2.0,
    center: str = "channel_mean",
    makeup_gain: float = 1.0,
    eps: float = 1e-8,
) -> Any:
    """Softly limit latent excursions with a tanh transfer curve."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    if drive <= 0:
        raise ValueError("drive must be positive")
    if ceiling <= 0:
        raise ValueError("ceiling must be positive")
    c = _center(x, center)
    residual = x - c
    scale = residual.float().std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    norm = residual.float() / (scale * float(ceiling))
    denom = torch.tanh(torch.tensor(float(drive), device=x.device, dtype=torch.float32)).clamp_min(eps)
    clipped = torch.tanh(float(drive) * norm) / denom
    target = c.float() + float(makeup_gain) * clipped * scale * float(ceiling)
    return target.to(dtype=x.dtype)


def latent_fft_eq(
    latents: Any,
    *,
    low_cutoff: float = 0.15,
    high_cutoff: float = 0.65,
    low_gain: float = 1.0,
    mid_gain: float = 1.0,
    high_gain: float = 1.0,
) -> Any:
    """Three-band gain over latent-time FFT bins."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    if x.shape[-1] <= 1:
        return x.clone()
    if not 0.0 <= low_cutoff <= 1.0 or not 0.0 <= high_cutoff <= 1.0:
        raise ValueError("FFT cutoffs must be normalized to [0, 1]")
    if high_cutoff < low_cutoff:
        raise ValueError("high_cutoff must be >= low_cutoff")

    spectrum = torch.fft.rfft(x.float(), dim=-1)
    freqs = torch.linspace(0.0, 1.0, spectrum.shape[-1], device=x.device, dtype=torch.float32)
    gains = torch.full_like(freqs, float(mid_gain))
    gains = torch.where(freqs < float(low_cutoff), torch.full_like(gains, float(low_gain)), gains)
    gains = torch.where(freqs > float(high_cutoff), torch.full_like(gains, float(high_gain)), gains)
    out = torch.fft.irfft(spectrum * gains.view(1, 1, -1), n=x.shape[-1], dim=-1)
    return out.to(dtype=x.dtype)


def latent_fft_phase_shift(latents: Any, *, shift_fraction: float = 0.0) -> Any:
    """Fractional circular shift by changing latent-time FFT phase."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    if x.shape[-1] <= 1 or shift_fraction == 0:
        return x.clone()
    spectrum = torch.fft.rfft(x.float(), dim=-1)
    bins = torch.arange(spectrum.shape[-1], device=x.device, dtype=torch.float32)
    phase = -2.0 * torch.pi * bins * float(shift_fraction)
    shifted = spectrum * torch.exp(1j * phase).view(1, 1, -1)
    out = torch.fft.irfft(shifted, n=x.shape[-1], dim=-1)
    return out.to(dtype=x.dtype)


def latent_fft_phase_randomize(
    latents: Any,
    *,
    amount: float = 1.0,
    seed: int = 0,
) -> Any:
    """Blend latent FFT phase toward random phase while preserving magnitudes."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    amount = _clamp01(amount)
    if x.shape[-1] <= 2 or amount == 0:
        return x.clone()

    spectrum = torch.fft.rfft(x.float(), dim=-1)
    magnitude = spectrum.abs()
    phase = torch.angle(spectrum)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    random_phase = (torch.rand(phase.shape, device=x.device, generator=generator) * 2.0 - 1.0) * torch.pi
    random_phase[..., 0] = 0.0
    if x.shape[-1] % 2 == 0:
        random_phase[..., -1] = 0.0
    unit = _blend_phase_units(torch.exp(1j * phase), torch.exp(1j * random_phase), amount)
    out = torch.fft.irfft(magnitude * unit, n=x.shape[-1], dim=-1)
    return out.to(dtype=x.dtype)


def latent_fft_phase_blend(latents: Any, donor_latents: Any, *, amount: float = 1.0) -> Any:
    """Keep source FFT magnitudes while blending latent-time phase toward donor."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    donor = _match_bct_shape(_as_bct(donor_latents, torch).to(device=x.device), x, torch)
    amount = _clamp01(amount)
    if x.shape[-1] <= 1 or amount == 0:
        return x.clone()

    source_spectrum = torch.fft.rfft(x.float(), dim=-1)
    donor_spectrum = torch.fft.rfft(donor.float(), dim=-1)
    magnitude = source_spectrum.abs()
    source_unit = _safe_complex_unit(source_spectrum)
    donor_unit = _safe_complex_unit(donor_spectrum)
    unit = _blend_phase_units(source_unit, donor_unit, amount)
    out = torch.fft.irfft(magnitude * unit, n=x.shape[-1], dim=-1)
    return out.to(dtype=x.dtype)


def latent_fft_magnitude_phase_graft(
    *,
    magnitude_latents: Any,
    phase_latents: Any,
    magnitude_amount: float = 1.0,
) -> Any:
    """Use magnitudes from one latent and phase from another in latent-time FFT.

    This is the latent analogue of keeping temporal organization from one signal
    while imposing modulation-spectrum energy from another.
    """

    torch = _require_torch()
    phase_source = _as_bct(phase_latents, torch)
    magnitude_source = _match_bct_shape(
        _as_bct(magnitude_latents, torch).to(device=phase_source.device),
        phase_source,
        torch,
    )
    amount = _clamp01(magnitude_amount)
    if phase_source.shape[-1] <= 1 or amount == 0:
        return phase_source.clone()

    phase_spectrum = torch.fft.rfft(phase_source.float(), dim=-1)
    magnitude_spectrum = torch.fft.rfft(magnitude_source.float(), dim=-1)
    magnitude = (1.0 - amount) * phase_spectrum.abs() + amount * magnitude_spectrum.abs()
    unit = _safe_complex_unit(phase_spectrum)
    out = torch.fft.irfft(magnitude * unit, n=phase_source.shape[-1], dim=-1)
    return out.to(dtype=phase_source.dtype)


def pca_component_gain(
    latents: Any,
    *,
    component_gains: Sequence[float] | None = None,
    rank: int | None = None,
) -> Any:
    """Apply gains to per-clip PCA/SVD components over the ``T x C`` latent matrix."""

    torch = _require_torch()
    x = _as_bct(latents, torch)
    outputs = []
    for item in x:
        time_major = item.transpose(0, 1).float()
        mean = time_major.mean(dim=0, keepdim=True)
        centered = time_major - mean
        u, s, vh = torch.linalg.svd(centered, full_matrices=False)
        component_count = s.shape[0]
        gains = torch.ones(component_count, device=x.device, dtype=torch.float32)
        if rank is not None:
            keep = max(0, min(int(rank), component_count))
        else:
            keep = component_count
        if component_gains is not None:
            provided = torch.tensor(list(component_gains), device=x.device, dtype=torch.float32)
            length = min(keep, provided.shape[0])
            gains[:length] = provided[:length]
        reconstructed = (u * (s * gains).view(1, -1)) @ vh
        outputs.append((reconstructed + mean).transpose(0, 1).to(dtype=x.dtype))
    return torch.stack(outputs, dim=0)


def latent_change_report(before: Any, after: Any) -> dict[str, float]:
    """Return JSON-friendly magnitude diagnostics for a latent edit."""

    torch = _require_torch()
    b = _as_bct(before, torch).float()
    a = _match_bct_shape(_as_bct(after, torch).float(), b, torch)
    delta = a - b
    b_flat = b.reshape(b.shape[0], -1)
    a_flat = a.reshape(a.shape[0], -1)
    cosine = torch.nn.functional.cosine_similarity(b_flat, a_flat, dim=-1).mean()
    return {
        "latent_rms_before": float(torch.sqrt(torch.mean(b * b)).detach().cpu()),
        "latent_rms_after": float(torch.sqrt(torch.mean(a * a)).detach().cpu()),
        "delta_rms": float(torch.sqrt(torch.mean(delta * delta)).detach().cpu()),
        "mean_abs_delta": float(delta.abs().mean().detach().cpu()),
        "cosine_similarity": float(cosine.detach().cpu()),
        "std_ratio": float((a.std(unbiased=False) / b.std(unbiased=False).clamp_min(1e-8)).detach().cpu()),
    }


def _blend(x: Any, target: Any, strength: float) -> Any:
    return x + float(strength) * (target - x)


def _center(x: Any, mode: str) -> Any:
    mode = mode.lower()
    if mode in {"zero", "origin", "none"}:
        return x.new_zeros((x.shape[0], x.shape[1], 1)).expand_as(x)
    if mode in {"channel_mean", "time_mean", "per_channel"}:
        return x.mean(dim=-1, keepdim=True)
    if mode in {"global_mean", "clip_mean"}:
        return x.mean(dim=(-2, -1), keepdim=True)
    raise ValueError(f"unknown latent DSP center mode: {mode}")


def _safe_complex_unit(spectrum: Any, eps: float = 1e-8) -> Any:
    return spectrum / spectrum.abs().clamp_min(eps)


def _blend_phase_units(source_unit: Any, target_unit: Any, amount: float) -> Any:
    blended = (1.0 - amount) * source_unit + amount * target_unit
    return _safe_complex_unit(blended)


def _match_bct_shape(value: Any, reference: Any, torch) -> Any:
    x = value
    if x.shape[0] == 1 and reference.shape[0] > 1:
        x = x.expand(reference.shape[0], -1, -1)
    if x.shape[0] != reference.shape[0]:
        raise ValueError("batch dimension must match, or donor batch must be 1")
    if x.shape[1] != reference.shape[1]:
        raise ValueError("channel dimension must match")
    if x.shape[2] != reference.shape[2]:
        x = torch.nn.functional.interpolate(
            x.float(),
            size=reference.shape[2],
            mode="linear",
            align_corners=False,
        ).to(dtype=value.dtype)
    return x


def _require_donor(donor_latents: Any | None, torch) -> Any:
    if donor_latents is None:
        raise ValueError("this latent DSP mode requires donor_latents")
    return donor_latents


def _clamp01(value: float) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError("amount values must be in [0, 1]")
    return value


def _as_bct(latents: Any, torch):
    tensor = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"expected latents shaped B x C x T or C x T, got {tuple(tensor.shape)}")
    return tensor


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for latent DSP.") from exc
    return torch
