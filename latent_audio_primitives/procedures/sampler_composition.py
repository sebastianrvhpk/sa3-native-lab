"""Notebook procedures for SA3 native sampler composition."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm

from latent_audio_primitives.sampler_composition import (
    SamplerCompositionPlan,
    apply_denoised_anchor,
    coerce_sampler_composition_plan,
    rf_denoised_from_velocity,
    rf_velocity_from_denoised,
    sampler_composition_step_rows,
    sampler_step_contexts,
    schedule_rows_to_table,
)


@dataclass(slots=True)
class SamplerCompositionResult:
    """Latents and metadata from one sampler-composition run."""

    latents: Any
    rows: list[Any] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save_metadata(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "sampler_composition_rows.json", schedule_rows_to_table(self.rows))
        _write_json(directory / "sampler_step_records.json", self.step_records)
        _write_json(directory / "metadata.json", self.metadata)
        return directory


def sample_trajectory_composition_euler(
    model: Any,
    x: Any,
    sigmas: Any,
    *,
    plan: SamplerCompositionPlan | Mapping[str, Any] | None = None,
    condition_inputs_by_phase: Mapping[str, Mapping[str, Any]] | None = None,
    default_cfg_scale: float = 1.0,
    default_apg_scale: float = 1.0,
    anchor: Any = None,
    anchor_mask: Any = None,
    padding_mask: Any = None,
    callback: Any = None,
    disable_tqdm: bool = False,
) -> tuple[Any, list[dict[str, Any]]]:
    """Euler RF sampler with prompt/guidance/anchor composition.

    This keeps SA3's RF convention explicit:

        x0_hat = x_t - sigma * v_theta(x_t, sigma, C)
        x0_hat' = blend(x0_hat, anchor, w)
        v' = (x_t - x0_hat') / sigma
        x_next = x_t + (sigma_next - sigma) * v'

    ``condition_inputs_by_phase`` should contain a ``"default"`` bundle and
    optional prompt-phase bundles keyed by ``PromptPhaseSpec.name``.
    """

    torch = _require_torch()
    resolved_plan = coerce_sampler_composition_plan(plan)
    condition_inputs_by_phase = dict(condition_inputs_by_phase or {"default": {}})
    if "default" not in condition_inputs_by_phase:
        raise ValueError("condition_inputs_by_phase must include a 'default' bundle")
    state = _as_bct(x, torch)
    t = _as_tensor(sigmas, torch).to(state.device)
    if t.ndim not in {1, 2}:
        raise ValueError("sample_trajectory_composition_euler expects a 1D or 2D sigma schedule")
    per_element_schedule = t.ndim == 2
    contexts = sampler_step_contexts(t)
    step_records: list[dict[str, Any]] = []
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        context = contexts[i]
        values = resolved_plan.values_for(
            context,
            default_cfg_scale=default_cfg_scale,
            default_apg_scale=default_apg_scale,
        )
        phase = values["phase"]
        phase_key = phase.name if phase is not None else "default"
        cond_inputs = dict(condition_inputs_by_phase.get(phase_key) or condition_inputs_by_phase["default"])
        cond_inputs["cfg_scale"] = float(values["cfg_scale"])
        cond_inputs["apg_scale"] = float(values["apg_scale"])
        cond_inputs.setdefault("batch_cfg", True)
        cond_inputs.setdefault("rescale_cfg", True)
        cond_inputs["padding_mask"] = padding_mask

        if per_element_schedule:
            t_curr_tensor = t[:, i].to(dtype=state.dtype)
            t_next_tensor = t[:, i + 1].to(dtype=state.dtype)
            dt = (t_next_tensor - t_curr_tensor).view(-1, 1, 1)
        else:
            t_curr = t[i].to(dtype=state.dtype)
            t_next = t[i + 1].to(dtype=state.dtype)
            t_curr_tensor = t_curr * torch.ones((state.shape[0],), dtype=state.dtype, device=state.device)
            dt = t_next - t_curr

        velocity = model(state, t_curr_tensor, **cond_inputs)
        denoised = rf_denoised_from_velocity(state, velocity, t_curr_tensor)
        anchor_strength = float(values["anchor_strength"])
        if anchor is not None and anchor_strength != 0.0:
            anchored_denoised = apply_denoised_anchor(
                denoised,
                anchor,
                strength=anchor_strength,
                mask=anchor_mask,
            )
            velocity = rf_velocity_from_denoised(state, anchored_denoised, t_curr_tensor)
        else:
            anchored_denoised = denoised

        record = {
            "i": int(i),
            "t": t_curr_tensor,
            "sigma": t_curr_tensor,
            "denoised": anchored_denoised,
            "raw_denoised": denoised,
            "cfg_scale": float(values["cfg_scale"]),
            "apg_scale": float(values["apg_scale"]),
            "anchor_strength": anchor_strength,
            "prompt_phase": phase_key,
            "prompt": values["prompt"],
        }
        step_records.append(_json_step_record(record, context))
        if callback is not None:
            callback(record)
        state = state + dt * velocity

    return state, step_records


def sa3_sampler_composition_from_init_latents(
    stable_model: Any,
    init_latents: Any,
    *,
    prompt: str = "",
    duration: float = 9.0,
    steps: int = 20,
    cfg_scale: float = 1.0,
    init_noise_level: float = 0.35,
    seed: int = 0,
    negative_prompt: str | None = None,
    apg_scale: float = 1.0,
    plan: SamplerCompositionPlan | Mapping[str, Any] | None = None,
    anchor_latents: Any = None,
    anchor_mask: Any = None,
    duration_padding_sec: float = 6.0,
    dist_shift: Any = None,
    callback: Any = None,
    disable_tqdm: bool = False,
) -> SamplerCompositionResult:
    """Run source-latent SA3 RF sampling with trajectory composition.

    This is an intervention candidate, not a promoted sampler. It intentionally
    uses an explicit Euler RF path so the denoised-state anchoring math is
    inspectable from the notebook.
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
        raise ValueError(f"sampler composition only supports RF objectives, got {core.diffusion_objective}")

    model_dtype = next(core.model.parameters()).dtype
    init = _as_bct(init_latents, torch).to(device=device, dtype=model_dtype)
    anchor = init if anchor_latents is None else _as_bct(anchor_latents, torch).to(device=device, dtype=model_dtype)
    resolved_plan = coerce_sampler_composition_plan(
        plan,
        default_prompt=prompt,
        default_negative_prompt=negative_prompt,
        cfg_scale=cfg_scale,
        apg_scale=apg_scale,
    )

    conditioning, _negative_conditioning = stable_model._build_conditioning_dicts(
        prompt,
        negative_prompt,
        duration,
        batch_size=init.shape[0],
    )
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

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise = torch.randn(init.shape, device=device, dtype=model_dtype, generator=generator)
    sigma_max = float(init_noise_level)
    state = init * (1.0 - sigma_max) + noise * sigma_max

    sigmas = build_schedule(
        steps=steps,
        sigma_max=sigma_max,
        dist_shift=dist_shift if dist_shift is not None else core.sampling_dist_shift,
        effective_seq_len=effective_seq_len,
        fallback_seq_len=init.shape[-1],
        include_endpoint=True,
        device=device,
    )
    phase_inputs = _build_phase_condition_inputs(
        stable_model,
        resolved_plan,
        duration=duration,
        batch_size=init.shape[0],
        latent_shape=tuple(init.shape),
        dtype=model_dtype,
        device=device,
        torch=torch,
    )
    rows = sampler_composition_step_rows(
        sigmas,
        plan=resolved_plan,
        cfg_scale=cfg_scale,
        apg_scale=apg_scale,
        default_prompt=prompt,
    )

    with torch.inference_mode():
        latents, step_records = sample_trajectory_composition_euler(
            core.model,
            state,
            sigmas,
            plan=resolved_plan,
            condition_inputs_by_phase=phase_inputs,
            default_cfg_scale=cfg_scale,
            default_apg_scale=apg_scale,
            anchor=anchor,
            anchor_mask=anchor_mask,
            padding_mask=padding_mask,
            callback=callback,
            disable_tqdm=disable_tqdm,
        )

    return SamplerCompositionResult(
        latents=latents,
        rows=rows,
        step_records=step_records,
        metadata={
            "prompt": prompt,
            "duration": float(duration),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "apg_scale": float(apg_scale),
            "init_noise_level": float(init_noise_level),
            "seed": int(seed),
            "plan": resolved_plan.to_manifest(),
            "anchor_source": "init_latents" if anchor_latents is None else "anchor_latents",
            "anchor_mask": "provided" if anchor_mask is not None else None,
            "sampler_type": "explicit_euler_rf_composition",
            "maturity": "intervention_candidate",
            "claim": "SA3 RF sampler-state composition pending audio evidence",
        },
    )


def _build_phase_condition_inputs(
    stable_model: Any,
    plan: SamplerCompositionPlan,
    *,
    duration: float,
    batch_size: int,
    latent_shape: tuple[int, int, int],
    dtype: Any,
    device: str,
    torch: Any,
) -> dict[str, dict[str, Any]]:
    core = stable_model.model
    phases = [None, *plan.prompt_phases]
    out: dict[str, dict[str, Any]] = {}
    mask = None
    inpaint_input = None
    for phase in phases:
        key = "default" if phase is None else phase.name
        prompt = plan.default_prompt if phase is None else phase.prompt
        negative_prompt = (
            plan.default_negative_prompt
            if phase is None or phase.negative_prompt is None
            else phase.negative_prompt
        )
        conditioning, negative_conditioning = stable_model._build_conditioning_dicts(
            prompt,
            negative_prompt,
            duration,
            batch_size=batch_size,
        )
        conditioning_tensors = core.conditioner(conditioning, device)
        negative_conditioning_tensors = {}
        if negative_conditioning is not None:
            negative_conditioning_tensors = core.conditioner(negative_conditioning, device)
        if mask is None:
            mask = torch.zeros((batch_size, 1, latent_shape[-1]), device=device)
            inpaint_input = torch.zeros(latent_shape, device=device, dtype=dtype)
        conditioning_tensors["inpaint_mask"] = [mask]
        conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        cond_inputs = core.get_conditioning_inputs(conditioning_tensors)
        if negative_conditioning_tensors:
            negative_conditioning_tensors["inpaint_mask"] = [mask]
            negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
            negative_conditioning_tensors = core.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
        out[key] = {
            **_cast_cond_inputs(cond_inputs, dtype, torch),
            **_cast_cond_inputs(negative_conditioning_tensors, dtype, torch),
        }
    return out


def _json_step_record(record: Mapping[str, Any], context: Any) -> dict[str, Any]:
    sigma = _tensor_scalar(record.get("sigma"))
    timestep = _tensor_scalar(record.get("t"))
    return {
        "sampler_index": int(record.get("i", context.step_index)),
        "timestep": timestep,
        "sigma": sigma,
        "logsnr": context.logsnr,
        "cfg_scale": float(record.get("cfg_scale", 1.0)),
        "apg_scale": float(record.get("apg_scale", 1.0)),
        "anchor_strength": float(record.get("anchor_strength", 0.0)),
        "prompt_phase": str(record.get("prompt_phase", "default")),
        "prompt": str(record.get("prompt", "")),
        "sampler_type": "explicit_euler_rf_composition",
    }


def _tensor_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().cpu().reshape(-1)
            if tensor.numel() == 0:
                return None
            return float(tensor.mean().item())
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for SA3 sampler composition.") from exc
    return torch
