"""Rows and math for SA3 internal feature cartography.

This module owns JSON-friendly research objects after SA3 internal tensors have
already been captured. It does not know how to find SA3 layers, run SA3, or
patch compiled transformer blocks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


MATURITY_MICROSCOPE = "microscope"
MATURITY_SELECTOR = "selector"
MATURITY_INTERVENTION_CANDIDATE = "intervention_candidate"
MATURITY_PROMOTED = "promoted_method"


@dataclass(frozen=True, slots=True)
class SA3InternalSurface:
    """One SA3-native internal object that can be observed or patched."""

    name: str
    native_object: str
    capture_mode: str
    operation: str
    maturity: str
    intervention_status: str
    source_status: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InternalActivationRow:
    """Summary row for captured activations on one surface/layer."""

    surface_name: str
    layer_index: int | None
    native_object: str
    capture_mode: str
    call_count: int
    shape: str
    token_count_min: int | None
    token_count_max: int | None
    feature_count: int | None
    rms_mean: float
    rms_std: float
    l2_per_element_mean: float
    l2_per_element_std: float
    abs_mean: float
    abs_max: float
    dtype: str
    device: str
    status: str = "ok"
    maturity: str = MATURITY_MICROSCOPE
    source_status: str = "repo-inferred"
    mapping_status: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CFGAPGInfluenceRow:
    """One timestep/call summary for SA3 CFG/APG prompt influence."""

    call_index: int
    step_index: int | None
    sampler_index: int | None
    timestep: float | None
    sigma: float | None
    logsnr: float | None
    shape: str
    diff_norm: float
    diff_rms: float
    cond_denoised_norm: float
    uncond_denoised_norm: float
    cond_uncond_cosine: float
    parallel_norm: float
    orthogonal_norm: float
    orthogonal_fraction: float
    parallel_cosine_to_cond: float
    mapping_status: str
    sampler_type: str = ""
    status: str = "ok"
    maturity: str = MATURITY_MICROSCOPE
    source_status: str = "repo-inferred"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SparseFeatureScaffoldRow:
    """A selected surface/layer that is suitable for later SAE training."""

    surface_name: str
    layer_index: int | None
    sample_source: str
    selection_reason: str
    suggested_training_object: str
    minimum_next_dataset: str
    status: str = "scaffold"
    maturity: str = MATURITY_MICROSCOPE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActivationPatchSpec:
    """Post-block residual activation patch coordinate.

    Clean/corrupt patching is intentionally surface-specific here: this spec
    targets post-block residual outputs from a clean cache. Branch-level
    interventions use ``BranchInterventionSpec`` instead.
    """

    layer_index: int
    call_start: int | None = None
    call_end: int | None = None
    step_index: int | None = None
    calls_per_step: int | None = None
    token_start: int | None = None
    token_end: int | None = None
    token_mask_name: str = ""
    batch_selector: str = "all"
    batch_indices: tuple[int, ...] | None = None
    alpha: float = 1.0
    mode: str = "blend"
    source: str = "manual"
    maturity: str = MATURITY_INTERVENTION_CANDIDATE
    note: str = ""

    def matches_call(self, layer_index: int, call_index: int) -> bool:
        if int(layer_index) != int(self.layer_index):
            return False
        if self.call_start is not None and self.call_end is not None:
            return int(self.call_start) <= int(call_index) < int(self.call_end)
        if self.step_index is not None and self.calls_per_step:
            start = int(self.step_index) * int(self.calls_per_step)
            end = start + int(self.calls_per_step)
            return start <= int(call_index) < end
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": int(self.layer_index),
            "call_start": _optional_int(self.call_start),
            "call_end": _optional_int(self.call_end),
            "step_index": _optional_int(self.step_index),
            "calls_per_step": _optional_int(self.calls_per_step),
            "token_start": _optional_int(self.token_start),
            "token_end": _optional_int(self.token_end),
            "token_mask_name": self.token_mask_name,
            "batch_selector": self.batch_selector,
            "batch_indices": None if self.batch_indices is None else [int(index) for index in self.batch_indices],
            "alpha": float(self.alpha),
            "mode": self.mode,
            "source": self.source,
            "maturity": self.maturity,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class BranchInterventionSpec:
    """Branch output intervention coordinate.

    Branch specs target module outputs such as ``self_attn_scale``,
    ``cross_attn_scale``, ``ff_scale``, and ``to_local_embed``. They can scale
    or ablate a branch without a clean cache, or blend/replace/add a clean
    activation when a matching clean branch trace is provided.
    """

    surface_name: str
    layer_index: int
    call_start: int | None = None
    call_end: int | None = None
    step_index: int | None = None
    calls_per_step: int | None = None
    token_start: int | None = None
    token_end: int | None = None
    token_mask_name: str = ""
    batch_selector: str = "all"
    batch_indices: tuple[int, ...] | None = None
    alpha: float = 1.0
    mode: str = "scale"
    source: str = "manual"
    maturity: str = MATURITY_INTERVENTION_CANDIDATE
    note: str = ""

    def matches_call(self, layer_index: int, call_index: int) -> bool:
        if int(layer_index) != int(self.layer_index):
            return False
        if self.call_start is not None and self.call_end is not None:
            return int(self.call_start) <= int(call_index) < int(self.call_end)
        if self.step_index is not None and self.calls_per_step:
            start = int(self.step_index) * int(self.calls_per_step)
            end = start + int(self.calls_per_step)
            return start <= int(call_index) < end
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_name": self.surface_name,
            "layer_index": int(self.layer_index),
            "call_start": _optional_int(self.call_start),
            "call_end": _optional_int(self.call_end),
            "step_index": _optional_int(self.step_index),
            "calls_per_step": _optional_int(self.calls_per_step),
            "token_start": _optional_int(self.token_start),
            "token_end": _optional_int(self.token_end),
            "token_mask_name": self.token_mask_name,
            "batch_selector": self.batch_selector,
            "batch_indices": None if self.batch_indices is None else [int(index) for index in self.batch_indices],
            "alpha": float(self.alpha),
            "mode": self.mode,
            "source": self.source,
            "maturity": self.maturity,
            "note": self.note,
        }


def default_sa3_internal_surfaces() -> list[SA3InternalSurface]:
    """Return the SA3-native internal objects used by the notebook path."""

    return [
        SA3InternalSurface(
            "post_block_residual",
            "transformer block residual stream",
            "block forward hook; monkey-patch for intervention",
            "observe/select/intervene",
            MATURITY_INTERVENTION_CANDIDATE,
            "patchable post-block residual surface",
            "confirmed",
            "Audioscope-style surface; causal only after patch/steer sweeps.",
        ),
        SA3InternalSurface(
            "self_attention_residual_update",
            "self-attention residual branch update",
            "submodule forward hook on self_attn_scale",
            "observe/select/intervene",
            MATURITY_INTERVENTION_CANDIDATE,
            "branch output scaling, ablation, or clean-cache patching",
            "source-inferred",
            "Closer to branch contribution than raw attention output because it is after scale/gate.",
        ),
        SA3InternalSurface(
            "cross_attention_residual_update",
            "prompt cross-attention residual branch update",
            "submodule forward hook on cross_attn_scale when present",
            "observe/select/intervene",
            MATURITY_INTERVENTION_CANDIDATE,
            "branch output scaling, ablation, or clean-cache patching",
            "source-inferred",
            "Only present on cross-attending SA3 transformer blocks.",
        ),
        SA3InternalSurface(
            "feedforward_residual_update",
            "feedforward residual branch update",
            "submodule forward hook on ff_scale",
            "observe/select/intervene",
            MATURITY_INTERVENTION_CANDIDATE,
            "branch output scaling, ablation, or clean-cache patching",
            "source-inferred",
            "Captures the post-gate feedforward update before residual addition.",
        ),
        SA3InternalSurface(
            "local_conditioning_projection",
            "local/inpaint conditioning projection",
            "submodule hook on to_local_embed when local conditioning is active",
            "observe/select/intervene",
            MATURITY_INTERVENTION_CANDIDATE,
            "local projection scaling, ablation, or clean-cache patching when active",
            "source-inferred",
            "Useful for inpaint/source-preservation studies.",
        ),
        *[
            SA3InternalSurface(
                name,
                "global timestep/condition adaLN scale, shift, or gate term",
                "block pre-hook derives scale/shift/gate from global_cond",
                "observe/select",
                MATURITY_MICROSCOPE,
                "capture-first; gate intervention needs separate causal test",
                "source-inferred",
                "Direct condition/timestep control surface in SA3 blocks.",
            )
            for name in (
                "adaln_scale_self",
                "adaln_shift_self",
                "adaln_gate_self",
                "adaln_scale_ff",
                "adaln_shift_ff",
                "adaln_gate_ff",
            )
        ],
        SA3InternalSurface(
            "memory_token_parameter",
            "continuous transformer memory token parameter",
            "parameter summary",
            "observe/select",
            MATURITY_MICROSCOPE,
            "parameter audit; intervention not local default",
            "source-inferred",
            "Global memory tokens are prepended before transformer layers when configured.",
        ),
        SA3InternalSurface(
            "cfg_apg_condition_influence",
            "conditional minus unconditional denoised state and APG components",
            "wrapped apg_project call plus sampler callback records",
            "observe/select",
            MATURITY_MICROSCOPE,
            "atlas first; prompt intervention remains external CFG/APG setting",
            "source-inferred",
            "SA3-native prompt influence vector over sampler time.",
        ),
    ]


def internal_surface_table(surfaces: Sequence[SA3InternalSurface] | None = None) -> list[dict[str, Any]]:
    """Return JSON-friendly surface inventory rows."""

    return [surface.to_dict() for surface in (surfaces or default_sa3_internal_surfaces())]


def summarize_activation_traces(
    traces_by_surface: Mapping[str, Mapping[int | None, Sequence[Any]]],
    *,
    surface_specs: Sequence[SA3InternalSurface] | None = None,
    mapping_status: str = "",
) -> list[InternalActivationRow]:
    """Summarize captured tensors into compact notebook rows."""

    specs = {surface.name: surface for surface in (surface_specs or default_sa3_internal_surfaces())}
    rows: list[InternalActivationRow] = []
    for surface_name, layer_map in traces_by_surface.items():
        spec = specs.get(
            surface_name,
            SA3InternalSurface(
                surface_name,
                "unknown",
                "unknown",
                "observe",
                MATURITY_MICROSCOPE,
                "unknown",
                "repo-inferred",
            ),
        )
        for layer_index, activations in layer_map.items():
            rows.append(
                summarize_activation_trace(
                    surface_name,
                    layer_index,
                    activations,
                    native_object=spec.native_object,
                    capture_mode=spec.capture_mode,
                    maturity=spec.maturity,
                    source_status=spec.source_status,
                    mapping_status=mapping_status,
                    note=spec.note,
                )
            )
    rows.sort(key=lambda row: (row.surface_name, -1 if row.layer_index is None else row.layer_index))
    return rows


def summarize_activation_trace(
    surface_name: str,
    layer_index: int | None,
    activations: Sequence[Any],
    *,
    native_object: str,
    capture_mode: str,
    maturity: str = MATURITY_MICROSCOPE,
    source_status: str = "repo-inferred",
    mapping_status: str = "",
    note: str = "",
) -> InternalActivationRow:
    """Summarize one captured activation trace."""

    values = [value for value in activations if value is not None]
    if not values:
        return InternalActivationRow(
            surface_name=surface_name,
            layer_index=None if layer_index is None else int(layer_index),
            native_object=native_object,
            capture_mode=capture_mode,
            call_count=0,
            shape="",
            token_count_min=None,
            token_count_max=None,
            feature_count=None,
            rms_mean=0.0,
            rms_std=0.0,
            l2_per_element_mean=0.0,
            l2_per_element_std=0.0,
            abs_mean=0.0,
            abs_max=0.0,
            dtype="",
            device="",
            status="empty",
            maturity=maturity,
            source_status=source_status,
            mapping_status=mapping_status,
            note=note,
        )

    arrays = [_to_numpy(value) for value in values]
    rms_values = np.asarray([float(np.sqrt(np.mean(np.square(arr)))) for arr in arrays], dtype=np.float64)
    l2_values = np.asarray(
        [float(np.linalg.norm(arr.reshape(-1)) / math.sqrt(max(arr.size, 1))) for arr in arrays],
        dtype=np.float64,
    )
    abs_values = np.concatenate([np.abs(arr).reshape(-1) for arr in arrays])
    token_counts = [_token_count(value) for value in values]
    feature_count = _feature_count(values[0])
    first_shape = _shape_string(values[0])
    dtype, device = _dtype_device(values[0])
    return InternalActivationRow(
        surface_name=surface_name,
        layer_index=None if layer_index is None else int(layer_index),
        native_object=native_object,
        capture_mode=capture_mode,
        call_count=len(values),
        shape=first_shape,
        token_count_min=min(token_counts) if token_counts else None,
        token_count_max=max(token_counts) if token_counts else None,
        feature_count=feature_count,
        rms_mean=float(rms_values.mean()),
        rms_std=float(rms_values.std()),
        l2_per_element_mean=float(l2_values.mean()),
        l2_per_element_std=float(l2_values.std()),
        abs_mean=float(abs_values.mean()) if abs_values.size else 0.0,
        abs_max=float(abs_values.max()) if abs_values.size else 0.0,
        dtype=dtype,
        device=device,
        maturity=maturity,
        source_status=source_status,
        mapping_status=mapping_status,
        note=note,
    )


def cfg_apg_component_stats(
    diff: Any,
    cond_denoised: Any,
    parallel: Any,
    orthogonal: Any,
) -> dict[str, Any]:
    """Summarize SA3's prompt influence vector and APG decomposition."""

    diff_np = _to_numpy(diff)
    cond_np = _to_numpy(cond_denoised)
    parallel_np = _to_numpy(parallel)
    orthogonal_np = _to_numpy(orthogonal)
    uncond_np = cond_np - diff_np
    diff_norm = _norm(diff_np)
    parallel_norm = _norm(parallel_np)
    orthogonal_norm = _norm(orthogonal_np)
    total_component_norm = parallel_norm + orthogonal_norm
    return {
        "shape": _shape_string(diff),
        "diff_norm": diff_norm,
        "diff_rms": _rms(diff_np),
        "cond_denoised_norm": _norm(cond_np),
        "uncond_denoised_norm": _norm(uncond_np),
        "cond_uncond_cosine": _cosine(cond_np, uncond_np),
        "parallel_norm": parallel_norm,
        "orthogonal_norm": orthogonal_norm,
        "orthogonal_fraction": orthogonal_norm / total_component_norm if total_component_norm > 0 else 0.0,
        "parallel_cosine_to_cond": _cosine(parallel_np, cond_np),
    }


def cfg_apg_rows_from_records(
    component_records: Sequence[Mapping[str, Any]],
    *,
    step_records: Sequence[Mapping[str, Any]] | None = None,
    sampler_type: str = "",
) -> list[CFGAPGInfluenceRow]:
    """Attach sampler metadata to CFG/APG component summaries."""

    rows: list[CFGAPGInfluenceRow] = []
    step_records = list(step_records or [])
    record_count = len(component_records)
    step_count = len(step_records)
    if step_count == 0:
        mapping_status = "no_sampler_records"
        calls_per_step = None
    elif record_count == step_count:
        mapping_status = "exact_one_call_per_step"
        calls_per_step = 1
    elif record_count > step_count and record_count % step_count == 0:
        mapping_status = "grouped_calls_per_step"
        calls_per_step = record_count // step_count
    else:
        mapping_status = "approximate_even_mapping"
        calls_per_step = None

    for call_index, record in enumerate(component_records):
        if not step_records:
            step_record: Mapping[str, Any] = {}
            step_index = None
        elif calls_per_step:
            step_index = min(call_index // calls_per_step, step_count - 1)
            step_record = step_records[step_index]
        else:
            step_index = min(int(call_index * step_count / max(record_count, 1)), step_count - 1)
            step_record = step_records[step_index]
        rows.append(
            CFGAPGInfluenceRow(
                call_index=int(call_index),
                step_index=None if step_index is None else int(step_index),
                sampler_index=_optional_int(step_record.get("sampler_index", step_record.get("i"))),
                timestep=_optional_float(step_record.get("timestep")),
                sigma=_optional_float(step_record.get("sigma")),
                logsnr=_optional_float(step_record.get("logsnr")),
                shape=str(record.get("shape", "")),
                diff_norm=float(record.get("diff_norm", 0.0)),
                diff_rms=float(record.get("diff_rms", 0.0)),
                cond_denoised_norm=float(record.get("cond_denoised_norm", 0.0)),
                uncond_denoised_norm=float(record.get("uncond_denoised_norm", 0.0)),
                cond_uncond_cosine=float(record.get("cond_uncond_cosine", 0.0)),
                parallel_norm=float(record.get("parallel_norm", 0.0)),
                orthogonal_norm=float(record.get("orthogonal_norm", 0.0)),
                orthogonal_fraction=float(record.get("orthogonal_fraction", 0.0)),
                parallel_cosine_to_cond=float(record.get("parallel_cosine_to_cond", 0.0)),
                mapping_status=mapping_status,
                sampler_type=str(step_record.get("sampler_type", sampler_type or "")),
            )
        )
    return rows


def sparse_feature_scaffold_rows(
    activation_rows: Sequence[InternalActivationRow | Mapping[str, Any]],
    *,
    top_k: int = 8,
    sample_source: str = "selected_internal_surface_capture",
) -> list[SparseFeatureScaffoldRow]:
    """Select surfaces for later SAE training without training an SAE yet."""

    normalized = [_row_to_dict(row) for row in activation_rows]
    candidates = [
        row
        for row in normalized
        if row.get("status") == "ok"
        and int(row.get("call_count") or 0) > 0
        and row.get("surface_name") not in {"adaln_scale_shift_gate", "memory_token_parameter"}
    ]
    candidates.sort(
        key=lambda row: (
            -float(row.get("rms_mean") or 0.0),
            str(row.get("surface_name") or ""),
            int(row.get("layer_index") if row.get("layer_index") is not None else -1),
        )
    )
    out = []
    for row in candidates[: max(0, int(top_k))]:
        surface = str(row.get("surface_name"))
        layer = row.get("layer_index")
        out.append(
            SparseFeatureScaffoldRow(
                surface_name=surface,
                layer_index=None if layer is None else int(layer),
                sample_source=sample_source,
                selection_reason=(
                    "Selected from captured SA3-native surfaces. Train only after this "
                    "surface repeats across prompt/source scouts and has a causal test plan."
                ),
                suggested_training_object=(
                    "token-level activations grouped by sampler logSNR band, not one whole-run mean"
                ),
                minimum_next_dataset=(
                    "at least several prompt/source families, repeated seeds, stored timestep/logSNR metadata"
                ),
            )
        )
    return out


def patch_specs_from_rows(
    rows: Sequence[Mapping[str, Any] | Any],
    *,
    top_k: int = 4,
    alpha: float = 1.0,
    mode: str = "blend",
    source: str = "selected_rows",
) -> list[ActivationPatchSpec]:
    """Convert ranked layer/timestep/window-like rows into patch specs."""

    specs: list[ActivationPatchSpec] = []
    normalized = [_row_to_dict(row) for row in rows]
    normalized.sort(key=lambda row: (int(row.get("rank") or 10**9), -float(row.get("score", row.get("accuracy_mean", 0.0)) or 0.0)))
    for row in normalized[: max(0, int(top_k))]:
        layer = row.get("layer_index", row.get("layer"))
        if layer is None:
            continue
        specs.append(
            ActivationPatchSpec(
                layer_index=int(layer),
                call_start=_optional_int(row.get("call_start")),
                call_end=_optional_int(row.get("call_end")),
                step_index=_optional_int(row.get("step_index", row.get("step"))),
                calls_per_step=_optional_int(row.get("calls_per_step")),
                token_start=_optional_int(row.get("token_start")),
                token_end=_optional_int(row.get("token_end")),
                token_mask_name=str(row.get("token_mask_name", "")),
                batch_selector=str(row.get("batch_selector", "all")),
                batch_indices=_optional_int_tuple(row.get("batch_indices")),
                alpha=float(alpha),
                mode=mode,
                source=source,
                note=str(row.get("mapping_status", "")),
            )
        )
    return specs


def branch_intervention_specs_from_rows(
    rows: Sequence[Mapping[str, Any] | Any],
    *,
    surface_name: str | None = None,
    top_k: int = 4,
    alpha: float = 1.0,
    mode: str = "scale",
    source: str = "selected_rows",
) -> list[BranchInterventionSpec]:
    """Convert ranked surface/layer rows into branch intervention specs."""

    specs: list[BranchInterventionSpec] = []
    normalized = [_row_to_dict(row) for row in rows]
    normalized.sort(
        key=lambda row: (
            int(row.get("rank") or 10**9),
            -float(row.get("score", row.get("accuracy_mean", row.get("rms_mean", 0.0))) or 0.0),
        )
    )
    for row in normalized[: max(0, int(top_k))]:
        layer = row.get("layer_index", row.get("layer"))
        surface = surface_name or row.get("surface_name")
        if layer is None or surface is None:
            continue
        specs.append(
            BranchInterventionSpec(
                surface_name=str(surface),
                layer_index=int(layer),
                call_start=_optional_int(row.get("call_start")),
                call_end=_optional_int(row.get("call_end")),
                step_index=_optional_int(row.get("step_index", row.get("step"))),
                calls_per_step=_optional_int(row.get("calls_per_step")),
                token_start=_optional_int(row.get("token_start")),
                token_end=_optional_int(row.get("token_end")),
                token_mask_name=str(row.get("token_mask_name", "")),
                batch_selector=str(row.get("batch_selector", "all")),
                batch_indices=_optional_int_tuple(row.get("batch_indices")),
                alpha=float(alpha),
                mode=mode,
                source=source,
                note=str(row.get("mapping_status", "")),
            )
        )
    return specs


def activation_rows_to_table(rows: Sequence[InternalActivationRow]) -> list[dict[str, Any]]:
    """Return JSON-friendly activation rows."""

    return [row.to_dict() for row in rows]


def cfg_apg_rows_to_table(rows: Sequence[CFGAPGInfluenceRow]) -> list[dict[str, Any]]:
    """Return JSON-friendly CFG/APG rows."""

    return [row.to_dict() for row in rows]


def sparse_feature_rows_to_table(rows: Sequence[SparseFeatureScaffoldRow]) -> list[dict[str, Any]]:
    """Return JSON-friendly sparse feature scaffold rows."""

    return [row.to_dict() for row in rows]


def patch_specs_to_table(specs: Sequence[ActivationPatchSpec]) -> list[dict[str, Any]]:
    """Return JSON-friendly patch specs."""

    return [spec.to_dict() for spec in specs]


def branch_intervention_specs_to_table(specs: Sequence[BranchInterventionSpec]) -> list[dict[str, Any]]:
    """Return JSON-friendly branch intervention specs."""

    return [spec.to_dict() for spec in specs]


def _row_to_dict(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return asdict(row)


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value, dtype=np.float32)


def _shape_string(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ""
    return "x".join(str(int(dim)) for dim in shape)


def _dtype_device(value: Any) -> tuple[str, str]:
    dtype = str(getattr(value, "dtype", ""))
    device = str(getattr(value, "device", ""))
    return dtype, device


def _token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None:
        arr = np.asarray(value)
        shape = arr.shape
    if len(shape) >= 2:
        return int(shape[-2])
    return 1


def _feature_count(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        arr = np.asarray(value)
        shape = arr.shape
    if len(shape) >= 1:
        return int(shape[-1])
    return None


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value)))) if value.size else 0.0


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(value.reshape(-1))) if value.size else 0.0


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    if denom <= eps:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)
