"""Soft conditioning optimization and audition hooks for frozen SA3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from latent_audio_primitives.flow_prompt import FlowProbeBank, flow_probe_bank_to_manifest


@dataclass(slots=True)
class SoftPromptState:
    """Optimized SA3 conditioning tensors for a target audio/latent."""

    conditioning: list[dict[str, Any]]
    conditioning_tensors: dict[str, Any]
    losses: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        torch = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "conditioning": self.conditioning,
                "conditioning_tensors": _to_cpu(self.conditioning_tensors),
                "losses": self.losses,
                "metadata": self.metadata,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SoftPromptState":
        torch = _require_torch()
        data = torch.load(path, map_location="cpu")
        return cls(
            conditioning=data["conditioning"],
            conditioning_tensors=data["conditioning_tensors"],
            losses=list(data.get("losses", [])),
            metadata=dict(data.get("metadata", {})),
        )


def optimize_soft_prompt_from_latents(
    stable_model: Any,
    target_latents: Any,
    *,
    seed_prompt: str,
    duration: float,
    optimization_steps: int = 100,
    lr: float = 1e-2,
    train_keys: tuple[str, ...] = ("prompt",),
    reg_weight: float = 1e-4,
    min_t: float = 0.05,
    max_t: float = 0.95,
    probe_bank: FlowProbeBank | None = None,
    seed: int = 0,
    velocity_convention: str = "noise_minus_data",
) -> SoftPromptState:
    """Optimize continuous SA3 conditioning tensors against a target SAME latent.

    This is the differentiable prompt route: it does not search over text
    strings. It keeps SA3 weights frozen and optimizes selected conditioning
    tensors so the frozen DiT better predicts the target latent under a
    flow-matching-style objective. ``velocity_convention`` controls the sign
    of the straight-path target and should be checked against the loaded SA3
    wrapper before long optimization runs.
    """

    torch = _require_torch()
    core = stable_model.model
    device = str(stable_model.device)
    model_dtype = next(core.model.parameters()).dtype
    target = _as_latent_tensor(target_latents, torch, device=device, dtype=model_dtype)
    batch_size, channels, latent_frames = target.shape

    conditioning = [{"prompt": seed_prompt, "seconds_total": duration}] * batch_size
    conditioning_tensors = core.conditioner(conditioning, device)
    conditioning_tensors = _clone_conditioning(conditioning_tensors, torch)
    params, originals = _make_trainable(conditioning_tensors, train_keys=train_keys, torch=torch)
    if not params:
        raise ValueError(f"no trainable conditioning tensors found for keys {train_keys}")

    for parameter in core.parameters():
        parameter.requires_grad_(False)
    core.eval()

    probe_specs = tuple(probe_bank.probes) if probe_bank is not None else ()
    if probe_bank is not None:
        velocity_convention = probe_bank.velocity_convention

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(params, lr=lr)
    losses: list[float] = []

    for _step in range(optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        if probe_specs:
            probe = probe_specs[_step % len(probe_specs)]
            t = torch.full((batch_size,), float(probe.timestep), device=device)
            noise_sign = float(probe.noise_sign)
        else:
            t = min_t + (max_t - min_t) * torch.rand(batch_size, device=device, generator=generator)
            noise_sign = 1.0
        noise = torch.randn(target.shape, device=device, dtype=model_dtype, generator=generator)
        if noise_sign != 1.0:
            noise = noise * noise_sign
        t_view = t[:, None, None].to(model_dtype)
        z_t = (1 - t_view) * target + t_view * noise
        velocity_target = _velocity_target(target, noise, convention=velocity_convention)

        cond_with_inpaint = _with_zero_inpaint(
            conditioning_tensors,
            batch_size,
            channels,
            latent_frames,
            device,
            model_dtype,
            torch,
        )
        cond_inputs = core.get_conditioning_inputs(cond_with_inpaint)
        cond_inputs = _cast_cond_inputs(cond_inputs, model_dtype)

        pred = core.model(
            z_t,
            t,
            **cond_inputs,
            cfg_scale=1.0,
            batch_cfg=True,
        )
        loss = torch.nn.functional.mse_loss(pred.float(), velocity_target.float())
        if reg_weight:
            reg = sum(torch.mean((param.float() - original.float()) ** 2) for param, original in originals)
            loss = loss + reg_weight * reg
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return SoftPromptState(
        conditioning=conditioning,
        conditioning_tensors=_detach_conditioning(conditioning_tensors),
        losses=losses,
        metadata={
            "seed_prompt": seed_prompt,
            "duration": duration,
            "optimization_steps": optimization_steps,
            "lr": lr,
            "train_keys": list(train_keys),
            "reg_weight": reg_weight,
            "min_t": min_t,
            "max_t": max_t,
            "probe_bank": None if probe_bank is None else flow_probe_bank_to_manifest(probe_bank),
            "seed": seed,
            "target_shape": list(target.shape),
            "velocity_convention": velocity_convention,
        },
    )


def generate_with_soft_prompt(
    stable_model: Any,
    state: SoftPromptState,
    *,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = 42,
    return_latents: bool = False,
    **kwargs: Any,
) -> Any:
    """Generate with optimized conditioning tensors through official SA3 generate."""

    device = str(stable_model.device)
    core = stable_model.model
    model_dtype = next(core.model.parameters()).dtype
    tensors = _to_device(state.conditioning_tensors, device, dtype=model_dtype)
    return stable_model.generate(
        conditioning=state.conditioning,
        conditioning_tensors=tensors,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        return_latents=return_latents,
        **kwargs,
    )


def _as_latent_tensor(value: Any, torch, *, device: str, dtype):
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"target latents must have shape B x C x T or C x T, got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


def _clone_conditioning(obj: Any, torch):
    if isinstance(obj, torch.Tensor):
        cloned = obj.detach().clone()
        if cloned.is_floating_point():
            cloned = cloned.float()
        return cloned
    if isinstance(obj, dict):
        return {key: _clone_conditioning(value, torch) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clone_conditioning(value, torch) for value in obj]
    return obj


def _detach_conditioning(obj: Any):
    torch = _require_torch()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _detach_conditioning(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_detach_conditioning(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_conditioning(value) for value in obj)
    return obj


def _make_trainable(conditioning_tensors: dict[str, Any], *, train_keys: tuple[str, ...], torch):
    params = []
    originals = []
    for key in train_keys:
        if key not in conditioning_tensors:
            continue
        value = conditioning_tensors[key]
        candidates = value if isinstance(value, list) else [value]
        for tensor in candidates:
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and tensor.ndim >= 2:
                tensor.requires_grad_(True)
                params.append(tensor)
                originals.append((tensor, tensor.detach().clone()))
                break
    return params, originals


def _with_zero_inpaint(
    conditioning_tensors: dict[str, Any],
    batch_size: int,
    channels: int,
    latent_frames: int,
    device: str,
    dtype,
    torch,
):
    out = dict(conditioning_tensors)
    out["inpaint_mask"] = [torch.zeros((batch_size, 1, latent_frames), device=device, dtype=dtype)]
    out["inpaint_masked_input"] = [torch.zeros((batch_size, channels, latent_frames), device=device, dtype=dtype)]
    return out


def _cast_cond_inputs(cond_inputs: dict[str, Any], dtype):
    torch = _require_torch()
    return {key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value for key, value in cond_inputs.items()}


def _velocity_target(target: Any, noise: Any, *, convention: str):
    if convention == "noise_minus_data":
        return noise - target
    if convention == "data_minus_noise":
        return target - noise
    raise ValueError("velocity_convention must be 'noise_minus_data' or 'data_minus_noise'")


def _to_cpu(obj: Any):
    torch = _require_torch()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(value) for value in obj)
    return obj


def _to_device(obj: Any, device: str, *, dtype: Any | None = None):
    torch = _require_torch()
    if isinstance(obj, torch.Tensor):
        if dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {key: _to_device(value, device, dtype=dtype) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_device(value, device, dtype=dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_to_device(value, device, dtype=dtype) for value in obj)
    return obj


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for soft prompt optimization.") from exc
    return torch
