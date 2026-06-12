"""SA3 internal feature capture, CFG/APG recording, and residual patching.

This adapter is deliberately source-sensitive. It talks to released SA3 module
shapes and keeps notebook-facing procedures away from upstream path details.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from latent_audio_primitives.internal_features import (
    ActivationPatchSpec,
    BranchInterventionSpec,
    cfg_apg_component_stats,
    cfg_apg_rows_from_records,
    default_sa3_internal_surfaces,
    summarize_activation_traces,
)
from latent_audio_primitives.adapters.sa3_residual_hooks import (
    AudioscopeIntegrationError,
    get_dit_layers,
)


BRANCH_SURFACE_MODULES = {
    "post_block_residual": "",
    "self_attention_residual_update": "self_attn_scale",
    "cross_attention_residual_update": "cross_attn_scale",
    "feedforward_residual_update": "ff_scale",
    "local_conditioning_projection": "to_local_embed",
}

ADALN_SURFACES = {
    "adaln_scale_self": 0,
    "adaln_shift_self": 1,
    "adaln_gate_self": 2,
    "adaln_scale_ff": 3,
    "adaln_shift_ff": 4,
    "adaln_gate_ff": 5,
}

DEFAULT_CAPTURE_SURFACES = [
    "post_block_residual",
    "self_attention_residual_update",
    "cross_attention_residual_update",
    "feedforward_residual_update",
    "local_conditioning_projection",
    *ADALN_SURFACES,
]


@dataclass
class InternalActivationStore:
    """Captured tensors for one surface/layer pair."""

    surface_name: str
    layer_index: int | None
    activations: list[Any] = field(default_factory=list)

    def clear(self) -> None:
        self.activations.clear()

    def append(self, value: Any, *, cpu_offload: bool) -> None:
        act = _first_tensor(value)
        if act is None:
            return
        act = act.detach()
        if cpu_offload:
            act = act.cpu()
        self.activations.append(act)


class SA3InternalActivationCollector:
    """Capture SA3 post-block, branch-update, and adaLN condition surfaces."""

    def __init__(
        self,
        model: Any,
        *,
        layer_indices: list[int] | None = None,
        surfaces: Sequence[str] | None = None,
        cpu_offload: bool = True,
    ) -> None:
        self.model = model
        self.layer_indices = layer_indices
        self.surfaces = list(surfaces or DEFAULT_CAPTURE_SURFACES)
        self.cpu_offload = cpu_offload
        self._hooks: list[Any] = []
        self._stores: dict[str, dict[int | None, InternalActivationStore]] = {}
        self._unsupported: list[dict[str, Any]] = []

        layers = get_dit_layers(model)
        indices = layer_indices if layer_indices is not None else list(range(len(layers)))
        for index in indices:
            if index < 0 or index >= len(layers):
                raise IndexError(f"layer index {index} out of range for {len(layers)} layers")
            block = layers[index]
            self._register_block_surfaces(block, index)
            self._register_adaln_surfaces(block, index)

    @property
    def unsupported(self) -> list[dict[str, Any]]:
        return list(self._unsupported)

    @property
    def surface_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._stores))

    def clear(self) -> None:
        for by_layer in self._stores.values():
            for store in by_layer.values():
                store.clear()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_raw_activations(self) -> dict[str, dict[int | None, list[Any]]]:
        return {
            surface: {
                layer: list(store.activations)
                for layer, store in by_layer.items()
            }
            for surface, by_layer in self._stores.items()
        }

    def get_summary_rows(self, *, mapping_status: str = ""):
        return summarize_activation_traces(
            self.get_raw_activations(),
            surface_specs=default_sa3_internal_surfaces(),
            mapping_status=mapping_status,
        )

    def __enter__(self) -> "SA3InternalActivationCollector":
        return self

    def __exit__(self, *_args) -> None:
        self.remove_hooks()

    def _register_block_surfaces(self, block: Any, layer_index: int) -> None:
        for surface_name in self.surfaces:
            if surface_name not in BRANCH_SURFACE_MODULES:
                continue
            module_name = BRANCH_SURFACE_MODULES[surface_name]
            module = block if module_name == "" else getattr(block, module_name, None)
            if module is None:
                self._unsupported.append(
                    {
                        "surface_name": surface_name,
                        "layer_index": int(layer_index),
                        "status": "unsupported",
                        "reason": f"block has no {module_name or 'forward'}",
                    }
                )
                continue
            store = self._store(surface_name, layer_index)
            self._hooks.append(module.register_forward_hook(self._make_hook(store)))

    def _register_adaln_surfaces(self, block: Any, layer_index: int) -> None:
        requested = [surface for surface in self.surfaces if surface in ADALN_SURFACES]
        if not requested:
            return
        if getattr(block, "global_cond_dim", None) is None or not hasattr(block, "to_scale_shift_gate"):
            for surface_name in requested:
                self._unsupported.append(
                    {
                        "surface_name": surface_name,
                        "layer_index": int(layer_index),
                        "status": "unsupported",
                        "reason": "block does not expose adaLN global conditioning",
                    }
                )
            return
        for surface_name in requested:
            self._store(surface_name, layer_index)
        self._hooks.append(
            block.register_forward_pre_hook(
                self._make_adaln_pre_hook(layer_index, requested),
                with_kwargs=True,
            )
        )

    def _store(self, surface_name: str, layer_index: int | None) -> InternalActivationStore:
        by_layer = self._stores.setdefault(surface_name, {})
        if layer_index not in by_layer:
            by_layer[layer_index] = InternalActivationStore(surface_name, layer_index)
        return by_layer[layer_index]

    def _make_hook(self, store: InternalActivationStore):
        def hook(_module, _inputs, output):
            store.append(output, cpu_offload=self.cpu_offload)

        return hook

    def _make_adaln_pre_hook(self, layer_index: int, requested: Sequence[str]):
        def hook(module, _args, kwargs):
            global_cond = kwargs.get("global_cond") if isinstance(kwargs, Mapping) else None
            if global_cond is None:
                return None
            values = (module.to_scale_shift_gate + global_cond).unsqueeze(1).chunk(6, dim=-1)
            for surface_name in requested:
                value = values[ADALN_SURFACES[surface_name]]
                self._store(surface_name, layer_index).append(value, cpu_offload=self.cpu_offload)
            return None

        return hook


class CFGAPGInfluenceRecorder:
    """Record SA3's CFG/APG condition-influence decomposition during sampling."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.backbone = get_diffusion_backbone(model)
        self.records: list[dict[str, Any]] = []
        self._original: Any = None

    def clear(self) -> None:
        self.records.clear()

    def rows(self, *, step_records: Sequence[Mapping[str, Any]] | None = None, sampler_type: str = ""):
        return cfg_apg_rows_from_records(
            self.records,
            step_records=step_records,
            sampler_type=sampler_type,
        )

    def __enter__(self) -> "CFGAPGInfluenceRecorder":
        self._original = self.backbone.apg_project

        def wrapped(v0, v1, padding_mask=None):
            parallel, orthogonal = self._original(v0, v1, padding_mask=padding_mask)
            stats = cfg_apg_component_stats(v0, v1, parallel, orthogonal)
            stats["call_index"] = len(self.records)
            self.records.append(stats)
            return parallel, orthogonal

        self.backbone.apg_project = wrapped
        return self

    def __exit__(self, *_args) -> None:
        if self._original is not None:
            self.backbone.apg_project = self._original
            self._original = None


class ResidualActivationPatcher:
    """Patch corrupt post-block residual activations from a clean activation cache."""

    def __init__(
        self,
        model: Any,
        *,
        clean_activations: Mapping[int, Sequence[Any]],
        patch_specs: Sequence[ActivationPatchSpec | Mapping[str, Any]],
    ) -> None:
        self.model = model
        self.clean_activations = {
            int(layer): list(values)
            for layer, values in clean_activations.items()
        }
        self.patch_specs = [_coerce_patch_spec(spec) for spec in patch_specs]
        self._blocks = {
            index: get_dit_layers(model)[index]
            for index in sorted({spec.layer_index for spec in self.patch_specs})
        }

    @contextmanager
    def patch(self):
        originals = {}
        call_counts = {index: 0 for index in self._blocks}
        for layer_index, block in self._blocks.items():
            originals[layer_index] = block.forward

            def make_patched(index, original_forward):
                def patched(*args, **kwargs):
                    call_index = call_counts[index]
                    call_counts[index] += 1
                    output = original_forward(*args, **kwargs)
                    clean_trace = self.clean_activations.get(index, [])
                    if call_index >= len(clean_trace):
                        return output
                    specs = [
                        spec
                        for spec in self.patch_specs
                        if spec.matches_call(index, call_index)
                    ]
                    if not specs:
                        return output
                    patched_output = output
                    for spec in specs:
                        patched_output = _patch_output(
                            patched_output,
                            clean_trace[call_index],
                            spec=spec,
                        )
                    return patched_output

                return patched

            block.forward = make_patched(layer_index, originals[layer_index])

        try:
            yield
        finally:
            for layer_index, block in self._blocks.items():
                block.forward = originals[layer_index]


class SA3BranchOutputPatcher:
    """Patch or scale SA3 branch output modules.

    This patcher targets branch modules registered in ``BRANCH_SURFACE_MODULES``:
    self-attention, cross-attention, feedforward, and local-conditioning
    projection outputs. ``mode='scale'`` and ``mode='ablate'`` do not require a
    clean cache. ``replace``, ``blend``, and ``add_delta`` require matching
    clean activations captured from ``SA3InternalActivationCollector``.
    """

    def __init__(
        self,
        model: Any,
        *,
        specs: Sequence[BranchInterventionSpec | Mapping[str, Any]],
        clean_activations: Mapping[str, Mapping[int, Sequence[Any]]] | None = None,
    ) -> None:
        self.model = model
        self.specs = [_coerce_branch_spec(spec) for spec in specs]
        self.clean_activations = {
            str(surface): {int(layer): list(values) for layer, values in by_layer.items()}
            for surface, by_layer in dict(clean_activations or {}).items()
        }
        layers = get_dit_layers(model)
        self._modules: dict[tuple[str, int], Any] = {}
        for spec in self.specs:
            module_name = BRANCH_SURFACE_MODULES.get(spec.surface_name)
            if module_name is None or module_name == "":
                raise AudioscopeIntegrationError(
                    f"{spec.surface_name!r} is not a branch module surface for SA3BranchOutputPatcher"
                )
            if spec.layer_index < 0 or spec.layer_index >= len(layers):
                raise IndexError(f"layer index {spec.layer_index} out of range for {len(layers)} layers")
            module = getattr(layers[spec.layer_index], module_name, None)
            if module is None:
                raise AudioscopeIntegrationError(
                    f"layer {spec.layer_index} has no module {module_name!r} for {spec.surface_name!r}"
                )
            self._modules[(spec.surface_name, spec.layer_index)] = module

    @contextmanager
    def patch(self):
        originals = {}
        call_counts = {key: 0 for key in self._modules}
        for key, module in self._modules.items():
            surface_name, layer_index = key
            originals[key] = module.forward

            def make_patched(surface, index, original_forward):
                def patched(*args, **kwargs):
                    call_index = call_counts[(surface, index)]
                    call_counts[(surface, index)] += 1
                    output = original_forward(*args, **kwargs)
                    specs = [
                        spec
                        for spec in self.specs
                        if spec.surface_name == surface and spec.matches_call(index, call_index)
                    ]
                    if not specs:
                        return output
                    patched_output = output
                    clean_trace = self.clean_activations.get(surface, {}).get(index, [])
                    for spec in specs:
                        clean_value = clean_trace[call_index] if call_index < len(clean_trace) else None
                        patched_output = _patch_output(
                            patched_output,
                            clean_value,
                            spec=spec,
                        )
                    return patched_output

                return patched

            module.forward = make_patched(surface_name, layer_index, originals[key])

        try:
            yield
        finally:
            for key, module in self._modules.items():
                module.forward = originals[key]


def get_diffusion_backbone(model: Any) -> Any:
    """Locate an object exposing SA3 DiffusionTransformer-style ``apg_project``."""

    candidates = [
        ("model.model.model", ("model", "model", "model")),
        ("model.model", ("model", "model")),
        ("model", ("model",)),
        ("self", ()),
    ]
    for _label, attrs in candidates:
        current = model
        try:
            for attr in attrs:
                current = getattr(current, attr)
            if hasattr(current, "apg_project") and hasattr(current, "transformer"):
                return current
        except AttributeError:
            continue
    raise AudioscopeIntegrationError("Could not locate SA3 diffusion backbone with apg_project")


def get_continuous_transformer(model: Any) -> Any:
    """Locate the ContinuousTransformer object when it is available."""

    backbone = get_diffusion_backbone(model)
    transformer = getattr(backbone, "transformer", None)
    if transformer is None:
        raise AudioscopeIntegrationError("Could not locate SA3 transformer")
    return transformer


def memory_token_parameter_rows(model: Any) -> list[dict[str, Any]]:
    """Return compact rows for SA3 continuous-transformer memory token parameters."""

    try:
        transformer = get_continuous_transformer(model)
    except AudioscopeIntegrationError:
        return [
            {
                "surface_name": "memory_token_parameter",
                "status": "unsupported",
                "maturity": "microscope",
                "note": "Could not locate ContinuousTransformer",
            }
        ]
    token_count = int(getattr(transformer, "num_memory_tokens", 0) or 0)
    if token_count <= 0 or not hasattr(transformer, "memory_tokens"):
        return [
            {
                "surface_name": "memory_token_parameter",
                "status": "absent",
                "token_count": 0,
                "maturity": "microscope",
            }
        ]
    tokens = transformer.memory_tokens.detach().float().cpu()
    return [
        {
            "surface_name": "memory_token_parameter",
            "status": "ok",
            "token_count": int(tokens.shape[0]),
            "feature_count": int(tokens.shape[-1]),
            "shape": "x".join(str(int(dim)) for dim in tokens.shape),
            "rms": float(tokens.square().mean().sqrt().item()),
            "norm": float(tokens.norm().item()),
            "maturity": "microscope",
            "source_status": "source-inferred",
        }
    ]


def _first_tensor(value: Any) -> Any | None:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    return item
    except Exception:
        return None
    return None


def _patch_output(output: Any, clean_value: Any | None, *, spec: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise AudioscopeIntegrationError("PyTorch is required for residual patching.") from exc

    def patch_tensor(corrupt):
        mode = str(spec.mode)
        alpha = float(spec.alpha)
        if mode == "scale":
            target = corrupt * alpha
        elif mode == "ablate":
            target = corrupt * 0.0
        else:
            if clean_value is None:
                return corrupt
            clean = clean_value.to(device=corrupt.device, dtype=corrupt.dtype)
            if tuple(clean.shape) != tuple(corrupt.shape):
                raise AudioscopeIntegrationError(
                    f"clean activation shape {tuple(clean.shape)} does not match corrupt shape {tuple(corrupt.shape)}"
                )
            if mode == "replace":
                target = clean
            elif mode == "blend":
                target = corrupt * (1.0 - alpha) + clean * alpha
            elif mode == "add_delta":
                target = corrupt + alpha * (clean - corrupt)
            else:
                raise ValueError("patch mode must be 'replace', 'blend', 'add_delta', 'scale', or 'ablate'")
        return _apply_tensor_selection(corrupt, target, spec, torch)

    if isinstance(output, torch.Tensor):
        return patch_tensor(output)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (patch_tensor(output[0]), *output[1:])
    if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
        return [patch_tensor(output[0]), *output[1:]]
    raise AudioscopeIntegrationError(f"cannot patch unsupported block output type {type(output)!r}")


def _coerce_patch_spec(spec: ActivationPatchSpec | Mapping[str, Any]) -> ActivationPatchSpec:
    if isinstance(spec, ActivationPatchSpec):
        return spec
    return ActivationPatchSpec(
        layer_index=int(spec["layer_index"]),
        call_start=None if spec.get("call_start") is None else int(spec["call_start"]),
        call_end=None if spec.get("call_end") is None else int(spec["call_end"]),
        step_index=None if spec.get("step_index") is None else int(spec["step_index"]),
        calls_per_step=None if spec.get("calls_per_step") is None else int(spec["calls_per_step"]),
        token_start=None if spec.get("token_start") is None else int(spec["token_start"]),
        token_end=None if spec.get("token_end") is None else int(spec["token_end"]),
        token_mask_name=str(spec.get("token_mask_name", "")),
        batch_selector=str(spec.get("batch_selector", "all")),
        batch_indices=None if spec.get("batch_indices") is None else tuple(int(index) for index in spec["batch_indices"]),
        alpha=float(spec.get("alpha", 1.0)),
        mode=str(spec.get("mode", "blend")),
        source=str(spec.get("source", "mapping")),
        maturity=str(spec.get("maturity", "intervention_candidate")),
        note=str(spec.get("note", "")),
    )


def _coerce_branch_spec(spec: BranchInterventionSpec | Mapping[str, Any]) -> BranchInterventionSpec:
    if isinstance(spec, BranchInterventionSpec):
        return spec
    return BranchInterventionSpec(
        surface_name=str(spec["surface_name"]),
        layer_index=int(spec["layer_index"]),
        call_start=None if spec.get("call_start") is None else int(spec["call_start"]),
        call_end=None if spec.get("call_end") is None else int(spec["call_end"]),
        step_index=None if spec.get("step_index") is None else int(spec["step_index"]),
        calls_per_step=None if spec.get("calls_per_step") is None else int(spec["calls_per_step"]),
        token_start=None if spec.get("token_start") is None else int(spec["token_start"]),
        token_end=None if spec.get("token_end") is None else int(spec["token_end"]),
        token_mask_name=str(spec.get("token_mask_name", "")),
        batch_selector=str(spec.get("batch_selector", "all")),
        batch_indices=None if spec.get("batch_indices") is None else tuple(int(index) for index in spec["batch_indices"]),
        alpha=float(spec.get("alpha", 1.0)),
        mode=str(spec.get("mode", "scale")),
        source=str(spec.get("source", "mapping")),
        maturity=str(spec.get("maturity", "intervention_candidate")),
        note=str(spec.get("note", "")),
    )


def _apply_tensor_selection(corrupt: Any, target: Any, spec: Any, torch) -> Any:
    has_token_selector = spec.token_start is not None or spec.token_end is not None
    has_batch_selector = (
        spec.batch_indices is not None
        or str(spec.batch_selector or "all").lower() not in {"", "all"}
    )
    if not has_token_selector and not has_batch_selector:
        return target
    if corrupt.ndim < 2:
        raise AudioscopeIntegrationError("selected patching requires tensor with batch dimension")
    batch_indices = _selected_batch_indices(spec, int(corrupt.shape[0]))
    if corrupt.ndim == 3:
        token_count = int(corrupt.shape[-2])
        token_start = 0 if spec.token_start is None else max(0, int(spec.token_start))
        token_end = token_count if spec.token_end is None else min(token_count, int(spec.token_end))
        if token_start >= token_end:
            return corrupt
        mask_shape = list(corrupt.shape[:-1]) + [1]
        mask = torch.zeros(mask_shape, device=corrupt.device, dtype=corrupt.dtype)
        mask[batch_indices, token_start:token_end, :] = 1.0
    elif corrupt.ndim > 3:
        raise AudioscopeIntegrationError(
            "selected patching currently supports 2D batch-feature or 3D batch-token-feature tensors"
        )
    else:
        if has_token_selector:
            raise AudioscopeIntegrationError("token selection requires an activation tensor with token axis")
        mask = torch.zeros((corrupt.shape[0], 1), device=corrupt.device, dtype=corrupt.dtype)
        mask[batch_indices, :] = 1.0
    return corrupt * (1.0 - mask) + target * mask


def _selected_batch_indices(spec: Any, batch_size: int) -> list[int]:
    if spec.batch_indices is not None:
        indices = [int(index) for index in spec.batch_indices]
        return [index for index in indices if 0 <= index < batch_size]
    selector = str(spec.batch_selector or "all").lower()
    if selector in {"", "all"}:
        return list(range(batch_size))
    if selector in {"first_half", "conditional", "cond"}:
        if batch_size < 2 or batch_size % 2 != 0:
            raise AudioscopeIntegrationError(
                "conditional batch selection requires even CFG batch layout"
            )
        return list(range(batch_size // 2))
    if selector in {"second_half", "unconditional", "uncond", "negative"}:
        if batch_size < 2 or batch_size % 2 != 0:
            raise AudioscopeIntegrationError(
                "unconditional batch selection requires even CFG batch layout"
            )
        return list(range(batch_size // 2, batch_size))
    if selector.startswith("batch_"):
        index = int(selector.split("_", 1)[1])
        return [index] if 0 <= index < batch_size else []
    raise ValueError(
        "batch_selector must be 'all', 'conditional', 'unconditional', "
        "'first_half', 'second_half', 'negative', or 'batch_N'"
    )
