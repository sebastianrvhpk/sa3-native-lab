"""Teacher-forced SA3 flow scoring procedures for prompt inversion."""

from __future__ import annotations

from typing import Any, Sequence

from latent_audio_primitives.flow_prompt import (
    FlowPromptLossRow,
    flow_velocity_target,
    logsnr_from_timestep,
    parse_float_sequence,
    timesteps_from_logsnr_values,
)


def sa3_flow_losses_for_prompts(
    stable_model: Any,
    target_latents: Any,
    prompts: Sequence[str],
    *,
    duration: float,
    seed: int = 0,
    min_t: float = 0.05,
    max_t: float = 0.95,
    score_samples: int = 1,
    shared_noise: bool = True,
    timestep_values: Sequence[float] | None = None,
    cosine_weight: float = 0.0,
    antithetic_noise: bool = False,
    normalize_mse: bool = True,
    conditional_delta_weight: float = 0.0,
    velocity_convention: str = "noise_minus_data",
) -> list[float]:
    """Score prompts against target SAME/SA3 latents with a frozen flow model.

    Lower loss means the frozen SA3 vector field better matches the straight
    data-to-noise velocity for the target latent under that text condition.
    Notebook ranking cells can negate these losses when a higher-is-better
    score is more convenient.
    """

    torch = _require_torch()
    core = stable_model.model
    device = str(stable_model.device)
    dtype = next(core.model.parameters()).dtype
    prompts = list(prompts)
    target = _as_target_batch(target_latents, torch=torch, device=device, dtype=dtype)
    if target.shape[0] == 1 and len(prompts) > 1:
        target = target.expand(len(prompts), -1, -1).contiguous()
    if target.shape[0] != len(prompts):
        raise ValueError("target batch must be 1 or match number of prompts")

    if timestep_values is not None:
        resolved_timesteps = [float(value) for value in timestep_values]
        score_samples = len(resolved_timesteps)
    else:
        resolved_timesteps = None
        score_samples = max(1, int(score_samples))

    conditioning = [{"prompt": prompt, "seconds_total": duration} for prompt in prompts]
    with torch.inference_mode():
        cond_inputs = _conditioning_inputs(core, conditioning, target, device=device, dtype=dtype, torch=torch)
        null_cond_inputs = None
        if conditional_delta_weight:
            null_conditioning = [{"prompt": "", "seconds_total": duration} for _prompt in prompts]
            null_cond_inputs = _conditioning_inputs(core, null_conditioning, target, device=device, dtype=dtype, torch=torch)

        batch, channels, frames = target.shape
        losses = torch.zeros((batch,), device=device, dtype=torch.float32)
        loss_terms = 0
        for sample_index in range(score_samples):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + sample_index * 1009)
            if shared_noise:
                if resolved_timesteps is not None:
                    t_scalar = torch.tensor(resolved_timesteps[sample_index], device=device)
                else:
                    t_scalar = min_t + (max_t - min_t) * torch.rand((), device=device, generator=generator)
                t = t_scalar.expand(batch)
                noise_base = torch.randn((1, channels, frames), device=device, dtype=dtype, generator=generator)
                noise = noise_base.expand(batch, -1, -1).contiguous()
            else:
                if resolved_timesteps is not None:
                    t = torch.full((batch,), resolved_timesteps[sample_index], device=device)
                else:
                    t = min_t + (max_t - min_t) * torch.rand(batch, device=device, generator=generator)
                noise = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)

            for noise_sign in ([1.0, -1.0] if antithetic_noise else [1.0]):
                signed_noise = noise * noise_sign
                t_view = t[:, None, None].to(dtype)
                z_t = (1 - t_view) * target + t_view * signed_noise
                velocity_target = flow_velocity_target(target, signed_noise, convention=velocity_convention)
                pred = core.model(z_t, t, **cond_inputs, cfg_scale=1.0, batch_cfg=True)
                mse = _flow_residual_loss(pred, velocity_target, normalize_mse=normalize_mse, cosine_weight=cosine_weight, torch=torch)
                if conditional_delta_weight and null_cond_inputs is not None:
                    null_pred = core.model(z_t, t, **null_cond_inputs, cfg_scale=1.0, batch_cfg=True)
                    delta = (pred.float() - null_pred.float()).reshape(batch, -1)
                    wanted_delta = (velocity_target.float() - null_pred.float()).reshape(batch, -1)
                    delta_cosine = torch.nn.functional.cosine_similarity(delta, wanted_delta, dim=1, eps=1e-8)
                    mse = mse + float(conditional_delta_weight) * (1.0 - delta_cosine)
                losses = losses + mse
                loss_terms += 1
        losses = losses / max(loss_terms, 1)
    return [float(loss.detach().cpu()) for loss in losses]


def sa3_flow_loss_rows_for_prompts(
    stable_model: Any,
    target_latents: Any,
    prompts: Sequence[str],
    *,
    duration: float,
    logsnr_values: Sequence[float] | str | None = None,
    timestep_values: Sequence[float] | None = None,
    seed: int = 0,
    shared_noise: bool = True,
    cosine_weight: float = 0.0,
    antithetic_noise: bool = False,
    normalize_mse: bool = True,
    conditional_delta_weight: float = 0.0,
    velocity_convention: str = "noise_minus_data",
) -> list[FlowPromptLossRow]:
    """Return per-prompt/per-probe losses instead of only aggregate losses."""

    prompts = list(prompts)
    if timestep_values is None:
        logsnr_list = parse_float_sequence(logsnr_values if logsnr_values is not None else [2.0, 0.0, -2.0])
        timesteps = timesteps_from_logsnr_values(logsnr_list)
    else:
        timesteps = [float(value) for value in timestep_values]
        logsnr_list = [logsnr_from_timestep(value) for value in timesteps]
    rows: list[FlowPromptLossRow] = []
    for probe_index, (timestep, logsnr) in enumerate(zip(timesteps, logsnr_list)):
        losses = sa3_flow_losses_for_prompts(
            stable_model,
            target_latents,
            prompts,
            duration=duration,
            seed=int(seed) + probe_index * 1009,
            timestep_values=[float(timestep)],
            shared_noise=shared_noise,
            cosine_weight=cosine_weight,
            antithetic_noise=antithetic_noise,
            normalize_mse=normalize_mse,
            conditional_delta_weight=conditional_delta_weight,
            velocity_convention=velocity_convention,
        )
        rows.extend(
            FlowPromptLossRow(
                prompt=prompt,
                timestep=float(timestep),
                logsnr=float(logsnr) if logsnr is not None else None,
                loss=float(loss),
                probe_index=probe_index,
            )
            for prompt, loss in zip(prompts, losses)
        )
    return rows


def _as_target_batch(value: Any, *, torch, device: str, dtype):
    target = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError(f"target latents must have shape B x C x T or C x T, got {tuple(target.shape)}")
    return target.to(device=device, dtype=dtype)


def _conditioning_inputs(core: Any, conditioning: list[dict[str, Any]], target: Any, *, device: str, dtype, torch) -> dict[str, Any]:
    cond = dict(core.conditioner(conditioning, device))
    batch, channels, frames = target.shape
    cond["inpaint_mask"] = [torch.zeros((batch, 1, frames), device=device)]
    cond["inpaint_masked_input"] = [torch.zeros((batch, channels, frames), device=device, dtype=dtype)]
    return {
        key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in core.get_conditioning_inputs(cond).items()
    }


def _flow_residual_loss(pred: Any, velocity_target: Any, *, normalize_mse: bool, cosine_weight: float, torch):
    batch = pred.shape[0]
    residual = pred.float() - velocity_target.float()
    mse = torch.mean(residual**2, dim=(1, 2))
    if normalize_mse:
        target_scale = torch.mean(velocity_target.float() ** 2, dim=(1, 2)).clamp_min(1e-8)
        mse = mse / target_scale
    if cosine_weight:
        pred_flat = pred.float().reshape(batch, -1)
        target_flat = velocity_target.float().reshape(batch, -1)
        cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8)
        mse = mse + float(cosine_weight) * (1.0 - cosine)
    return mse


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - exercised only when torch is missing
        raise RuntimeError("PyTorch is required for SA3 flow prompt scoring") from exc
    return torch
