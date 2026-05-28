from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from latent_audio_primitives.geometry import geometry_report
from latent_audio_primitives.flow_prompt import (
    parse_float_sequence,
    sa3_flow_losses_for_prompts,
    timesteps_from_logsnr_values,
)
from latent_audio_primitives.index import LatentMemoryIndex
from latent_audio_primitives.latent_blur import LatentBlurSpec, apply_latent_blur
from latent_audio_primitives.latent_dsp import LatentDSPSpec, apply_latent_dsp
from latent_audio_primitives.looping import cyclic_mix_latents, cyclic_roll_latents
from latent_audio_primitives.prompt_optimization import (
    beam_token_prompt_search,
    coordinate_prompt_search,
    default_modifier_axes,
    greedy_token_prompt_search,
    prompt_seed_from_audio_path,
)
from latent_audio_primitives.schema import LatentItem
from latent_audio_primitives.selective_renoise import (
    LatentMaskSpec,
    graft_latent_channels,
    masked_latent_noise,
    select_latent_channels,
)

from .contracts import (
    ArtifactKind,
    ArtifactRecord,
    BackendName,
    ModelStatus,
    OperatorFieldOption,
    OperatorFieldSpec,
    OperatorName,
    OperatorSpec,
    Recipe,
)
from .ids import new_id
from .jobs import JobContext, JobResult
from .storage import ArtifactStore


SCRIPT_EXPERIMENT_OPERATORS = {
    OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS,
    OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE,
    OperatorName.EXPERIMENT_STYLE_PROFILE_BUILD,
    OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE,
    OperatorName.EXPERIMENT_STYLE_DIRECTION_GENERATE,
    OperatorName.EXPERIMENT_AUDIO_DIRECTION_GENERATE,
    OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT,
    OperatorName.EXPERIMENT_AUDIO_RESIDUAL_VECTORS_EXTRACT,
    OperatorName.EXPERIMENT_ALPHA_SWEEP,
    OperatorName.EXPERIMENT_SOFT_PROMPT_OPTIMIZE,
    OperatorName.EXPERIMENT_SOFT_PROMPT_GENERATE,
    OperatorName.DATASET_PRE_ENCODE,
    OperatorName.TRAIN_LORA,
}

PROGRESS_PERCENT_RE = re.compile(r"(?<![\d.])(\d{1,3})(?:\.\d+)?%")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}
SA3_MEDIUM_CHECKPOINT_BYTES = 9_222_120_000
DEFAULT_PROMPT_SEARCH_VOCABULARY = [
    "warm",
    "cold",
    "bright",
    "dark",
    "muted",
    "shimmering",
    "granular",
    "textured",
    "pulsing",
    "drifting",
    "wide stereo",
    "close",
    "dry",
    "reverberant",
    "ambient",
    "cinematic",
    "electronic",
    "acoustic",
    "percussive",
    "sustained",
    "metallic",
    "soft",
    "dense",
    "sparse",
    "loop",
    "rhythm",
    "tone",
    "noise",
    "gesture",
    "texture",
]


FIELD_HINTS: dict[str, dict[str, Any]] = {
    "prompt": {"type": "text", "default": "audio texture", "required": True, "description": "Conditioning text sent to the active generation or experiment script."},
    "negative_prompt": {"type": "text", "advanced": True, "placeholder": "optional"},
    "duration_seconds": {"type": "number", "default": 30.0, "min": 0.5, "max": 120.0, "step": 0.5, "description": "Requested output duration in seconds."},
    "duration": {"type": "number", "default": 8.0, "min": 0.5, "max": 120.0, "step": 0.5},
    "steps": {"type": "number", "default": 8, "min": 1, "max": 256, "step": 1},
    "seed": {"type": "number", "step": 1, "description": "Deterministic seed when the backend/script supports seeded runs."},
    "cfg_scale": {"type": "number", "default": 1.0, "min": 0.0, "step": 0.1},
    "apg_scale": {"type": "number", "default": 1.0, "min": 0.0, "step": 0.1},
    "init_noise_level": {"type": "number", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
    "inpaint_start_seconds": {"type": "number", "default": 0.0, "min": 0.0, "step": 0.1},
    "inpaint_end_seconds": {"type": "number", "default": 2.0, "min": 0.1, "step": 0.1},
    "chunked": {"type": "checkbox", "default": False, "advanced": True},
    "chunk_size": {"type": "number", "default": 128, "min": 1, "step": 1, "advanced": True},
    "overlap": {"type": "number", "default": 32, "min": 0, "step": 1, "advanced": True},
    "notes": {"type": "text", "advanced": True, "placeholder": "optional"},
    "model": {"type": "select", "description": "Model family used by the selected backend or script adapter."},
    "decoder": {"type": "select", "options": ["same-s", "same-l"], "advanced": True},
    "backend": {"type": "select", "advanced": True},
    "mode": {"type": "select"},
    "shift_frames": {"type": "number", "default": 1, "min": -4096, "max": 4096, "step": 1},
    "strength": {"type": "range", "default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01},
    "symmetric": {"type": "checkbox", "default": True},
    "fraction": {"type": "range", "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01},
    "amount": {"type": "range", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "sigma": {"type": "range", "default": 0.4, "min": 0.0, "max": 1.5, "step": 0.01},
    "alpha": {"type": "number", "default": 0.6, "step": 0.05},
    "std_alpha": {"type": "number", "default": 0.0, "step": 0.05},
    "alphas": {"type": "text", "default": "-8,-4,0,4,8", "placeholder": "-8,-4,0,4,8", "description": "Comma-separated steering strengths to render as an alpha sweep."},
    "layer": {"type": "number", "default": -1, "step": 1, "advanced": True},
    "layers": {"type": "text", "advanced": True, "placeholder": "blank or 1,4,8"},
    "limit": {"type": "number", "default": 0, "min": 0, "step": 1, "advanced": True},
    "n_components": {"type": "number", "default": 8, "min": 1, "step": 1, "description": "Number of principal components to keep in the geometry audit."},
    "num_pairs": {"type": "number", "default": 2, "min": 1, "step": 1},
    "axis": {"type": "text", "default": "valence", "required": True},
    "baseline": {"type": "select", "default": "prompt", "options": ["prompt", "negative_audio"]},
    "top_k": {"type": "number", "default": 5, "min": 1, "step": 1},
    "metric": {"type": "select", "default": "cosine", "options": ["cosine", "euclidean"]},
    "exclude_self": {"type": "checkbox", "default": True},
    "optimization_steps": {"type": "number", "default": 100, "min": 1, "step": 1},
    "search_mode": {"type": "select", "default": "beam", "options": ["beam", "greedy", "coordinate"], "description": "Prompt search strategy used by the native probe scorer."},
    "scorer": {"type": "select", "default": "lexical_probe", "options": ["lexical_probe", "sa3_flow_probe", "clap"], "description": "Prompt objective. SA3 flow loads the Medium model; CLAP is reserved for a later adapter."},
    "seed_prompt": {"type": "text", "default": "audio texture", "description": "Starting text for prompt search or soft-prompt optimization."},
    "vocabulary": {"type": "text", "default": ", ".join(DEFAULT_PROMPT_SEARCH_VOCABULARY), "advanced": True},
    "tokens_generated": {"type": "number", "default": 4, "min": 1, "max": 32, "step": 1},
    "beam_width": {"type": "number", "default": 4, "min": 1, "max": 24, "step": 1},
    "branch_factor": {"type": "number", "default": 64, "min": 1, "step": 1, "advanced": True},
    "runs": {"type": "number", "default": 4, "min": 1, "max": 64, "step": 1, "advanced": True},
    "rounds": {"type": "number", "default": 2, "min": 1, "max": 12, "step": 1, "advanced": True},
    "candidate_batch_size": {"type": "number", "default": 0, "min": 0, "step": 1, "advanced": True},
    "score_samples": {"type": "number", "default": 1, "min": 1, "step": 1, "advanced": True},
    "logsnr_values": {"type": "text", "default": "", "advanced": True, "placeholder": "2,0,-2"},
    "timestep_values": {"type": "text", "default": "", "advanced": True, "placeholder": "0.12,0.5,0.88"},
    "min_t": {"type": "number", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True},
    "max_t": {"type": "number", "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "advanced": True},
    "shared_noise": {"type": "checkbox", "default": True, "advanced": True},
    "antithetic_noise": {"type": "checkbox", "default": False, "advanced": True},
    "normalize_mse": {"type": "checkbox", "default": True, "advanced": True},
    "cosine_weight": {"type": "number", "default": 0.0, "min": 0.0, "step": 0.05, "advanced": True},
    "conditional_delta_weight": {"type": "number", "default": 0.0, "min": 0.0, "step": 0.05, "advanced": True},
    "prefix": {"type": "text", "advanced": True, "placeholder": "optional"},
    "suffix": {"type": "text", "advanced": True, "placeholder": "optional"},
    "separator": {"type": "text", "default": " ", "advanced": True},
    "modifier_axes": {"type": "text", "advanced": True, "placeholder": "bright|dark|warm; sparse|dense"},
    "lr": {"type": "number", "step": 0.00001, "advanced": True},
    "reg_weight": {"type": "number", "default": 0.0001, "step": 0.0001, "advanced": True},
    "train_keys": {"type": "text", "default": "prompt", "advanced": True},
    "velocity_convention": {"type": "select", "default": "noise_minus_data", "options": ["noise_minus_data", "data_minus_noise"], "advanced": True},
    "batch_size": {"type": "number", "default": 1, "min": 1, "step": 1},
    "sample_size": {"type": "number", "default": 12582912, "min": 1, "step": 1, "advanced": True},
    "pad": {"type": "checkbox", "default": False, "advanced": True},
    "model_half": {"type": "checkbox", "default": False, "advanced": True},
    "rank": {"type": "number", "default": 16, "min": 1, "step": 1},
    "lora_alpha": {"type": "number", "step": 1, "advanced": True},
    "adapter_type": {"type": "select", "default": "lora", "options": ["lora", "lora-xs"], "advanced": True},
    "dropout": {"type": "number", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "advanced": True},
    "logger": {"type": "select", "default": "csv", "options": ["wandb", "comet", "csv", "none"], "advanced": True},
    "include": {"type": "text", "advanced": True},
    "exclude": {"type": "text", "advanced": True},
    "device": {"type": "text", "advanced": True},
    "name": {"type": "text"},
    "positive_path": {"type": "path", "required": True},
    "negative_path": {"type": "path"},
    "input_path": {"type": "path", "required": True},
    "data_dir": {"type": "path"},
    "target_audio_path": {"type": "artifact-path", "artifact_kinds": [ArtifactKind.AUDIO]},
    "vectors_path": {"type": "artifact-path", "required": True, "artifact_kinds": [ArtifactKind.BUNDLE], "description": "Bundle artifact containing reusable steering vectors."},
    "profile_path": {"type": "artifact-path", "required": True, "artifact_kinds": [ArtifactKind.BUNDLE], "description": "Bundle artifact containing a style profile."},
    "direction_path": {"type": "artifact-path", "required": True, "artifact_kinds": [ArtifactKind.BUNDLE], "description": "Bundle artifact containing a direction vector."},
    "soft_prompt_path": {"type": "artifact-path", "required": True, "artifact_kinds": [ArtifactKind.BUNDLE], "description": "Bundle artifact containing optimized soft-prompt tensors."},
    "target_memory_path": {"type": "artifact-path", "required": True, "artifact_kinds": [ArtifactKind.BUNDLE], "description": "Bundle artifact used as the target memory/profile source."},
    "reference_memory_path": {"type": "artifact-path", "artifact_kinds": [ArtifactKind.BUNDLE], "description": "Optional reference bundle for contrastive profile building."},
    "encoded_dir": {"type": "artifact-path", "artifact_kinds": [ArtifactKind.BUNDLE]},
    "svd_bases_path": {"type": "artifact-path", "artifact_kinds": [ArtifactKind.BUNDLE], "advanced": True},
    "lora_checkpoint": {"type": "artifact-path", "artifact_kinds": [ArtifactKind.BUNDLE], "advanced": True},
}


GENERIC_TYPE_PARTS = {"str", "int", "float", "bool", "folder", "file", "csv", "patterns", "null", "..."}


def _with_ui_fields(spec: OperatorSpec) -> OperatorSpec:
    return spec.model_copy(update={"ui_fields": _ui_fields_for_spec(spec)})


def _ui_fields_for_spec(spec: OperatorSpec) -> list[OperatorFieldSpec]:
    fields = [_ui_field_for_param(key, value) for key, value in spec.params.items()]
    fields.append(
        OperatorFieldSpec(
            key="backend",
            label="Backend",
            type="select",
            default=spec.backends[0].value if spec.backends else None,
            advanced=True,
            options=[_field_option(backend.value) for backend in spec.backends],
        )
    )
    return fields


def _ui_field_for_param(key: str, param_type: Any) -> OperatorFieldSpec:
    hint = FIELD_HINTS.get(key, {})
    options = _field_options(key, param_type, hint)
    field_type = str(hint.get("type") or _infer_field_type(param_type, options))
    default = hint.get("default", _default_for_field(key, field_type, options))
    return OperatorFieldSpec(
        key=key,
        label=str(hint.get("label") or _human_label(key)),
        type=field_type,
        default=default,
        required=bool(hint.get("required", False)),
        advanced=bool(hint.get("advanced", False)),
        min=_optional_float(hint.get("min")),
        max=_optional_float(hint.get("max")),
        step=_optional_float(hint.get("step")),
        options=options,
        artifact_kinds=list(hint.get("artifact_kinds", [])),
        placeholder=hint.get("placeholder"),
        description=hint.get("description"),
    )


def _field_options(key: str, param_type: Any, hint: dict[str, Any]) -> list[OperatorFieldOption]:
    raw_options = hint.get("options")
    if raw_options is None:
        raw_options = _literal_options(param_type)
    if not raw_options:
        return []
    return [_field_option(str(value), key=key) for value in raw_options]


def _literal_options(param_type: Any) -> list[str]:
    if not isinstance(param_type, str) or "|" not in param_type:
        return []
    values = [part.strip() for part in param_type.split("|") if part.strip()]
    if not values:
        return []
    meaningful = [value for value in values if value not in GENERIC_TYPE_PARTS]
    if not meaningful:
        return []
    if len(meaningful) != len([value for value in values if value != "null"]):
        return []
    return meaningful


def _field_option(value: str, *, key: str | None = None) -> OperatorFieldOption:
    label = {
        "sm-music": "Small Music",
        "sm-sfx": "Small SFX",
        "medium": "Medium",
        "same-s": "SAME-S",
        "same-l": "SAME-L",
        "torch_mps": "Torch MPS",
        "torch_cpu": "Torch CPU",
        "mlx": "MLX",
        "cpu": "CPU",
    }.get(value)
    if label is None:
        label = _human_label(value if key != "model" else value.replace("-", " "))
    return OperatorFieldOption(value=value, label=label)


def _infer_field_type(param_type: Any, options: list[OperatorFieldOption]) -> str:
    if options:
        return "select"
    value = str(param_type)
    if "bool" in value:
        return "checkbox"
    if value in {"int", "int|null", "float", "float|null"}:
        return "number"
    if "folder" in value or "file" in value:
        return "path"
    return "text"


def _default_for_field(key: str, field_type: str, options: list[OperatorFieldOption]) -> Any:
    if key == "model":
        values = {option.value for option in options}
        if "medium" in values:
            return "medium"
        if "same-l" in values:
            return "same-l"
    if field_type == "checkbox":
        return False
    if field_type == "select" and options:
        return options[0].value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _human_label(value: str) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


class RuntimeDispatcher:
    def __init__(self, store: ArtifactStore, *, repo_root: str | Path | None = None) -> None:
        self.store = store
        env_repo_root = os.environ.get("SA3_REPO_ROOT")
        self.repo_root = (
            Path(repo_root)
            if repo_root is not None
            else Path(env_repo_root)
            if env_repo_root
            else Path(__file__).resolve().parents[2]
        )
        self.mlx_dir = self.repo_root / "optimized" / "mlx"
        self.mlx_wrapper = self.mlx_dir / "sa3"
        self._same_adapters: dict[tuple[str, str], Any] = {}

    def backend_statuses(self) -> list[ModelStatus]:
        return [self._mlx_status(), self._torch_status(), self._cpu_status()]

    def operator_specs(self) -> list[OperatorSpec]:
        specs = [
            OperatorSpec(
                name=OperatorName.TEXT_TO_AUDIO,
                maturity="core",
                backends=[BackendName.MLX],
                inputs=[],
                params={
                    "prompt": "str",
                    "negative_prompt": "str|null",
                    "duration_seconds": "float",
                    "steps": "int",
                    "seed": "int|null",
                    "cfg_scale": "float",
                    "apg_scale": "float",
                    "model": "sm-music|sm-sfx|medium",
                    "decoder": "same-s|same-l|null",
                },
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.AUDIO_TO_AUDIO,
                maturity="core",
                backends=[BackendName.MLX],
                inputs=["source"],
                params={
                    "prompt": "str",
                    "negative_prompt": "str|null",
                    "duration_seconds": "float",
                    "steps": "int",
                    "seed": "int|null",
                    "cfg_scale": "float",
                    "apg_scale": "float",
                    "model": "sm-music|sm-sfx|medium",
                    "decoder": "same-s|same-l|null",
                    "init_noise_level": "float",
                },
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.INPAINT,
                maturity="core",
                backends=[BackendName.MLX],
                inputs=["source"],
                params={
                    "prompt": "str",
                    "negative_prompt": "str|null",
                    "duration_seconds": "float",
                    "steps": "int",
                    "seed": "int|null",
                    "cfg_scale": "float",
                    "apg_scale": "float",
                    "model": "sm-music|sm-sfx|medium",
                    "decoder": "same-s|same-l|null",
                    "init_noise_level": "float",
                    "inpaint_start_seconds": "float",
                    "inpaint_end_seconds": "float",
                },
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_BLUR,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source"],
                params={"mode": "temporal|channel|low_rank|sharpen|fft_lowpass|...", "strength": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_DSP,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source", "donor?"],
                params={"mode": "gain|compress|softclip|fft_eq|phase_blend|...", "strength": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_GRAFT,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source", "donor"],
                params={"mode": "random_channels|high_variance|...", "fraction": "float", "amount": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_RENOISE,
                maturity="lab",
                backends=[BackendName.TORCH_CPU],
                inputs=["source"],
                params={"mode": "random_channels|high_variance|...", "fraction": "float", "sigma": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_CYCLIC_ROLL,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source"],
                params={"shift_frames": "int", "strength": "float|null", "symmetric": "bool"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_ENCODE,
                maturity="core",
                backends=[BackendName.TORCH_MPS],
                inputs=["source"],
                params={
                    "model": "same-s|same-l",
                    "chunked": "bool",
                    "chunk_size": "int",
                    "overlap": "int",
                    "prompt": "str|null",
                    "notes": "str|null",
                },
                produces=[ArtifactKind.LATENT],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_DECODE,
                maturity="core",
                backends=[BackendName.TORCH_MPS],
                inputs=["source"],
                params={
                    "model": "same-s|same-l",
                    "chunked": "bool",
                    "chunk_size": "int",
                    "overlap": "int",
                    "notes": "str|null",
                },
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_PROMPT_SEARCH,
                maturity="probe",
                backends=[BackendName.CPU, BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=["source?"],
                params={
                    "target_audio_path": "file|null",
                    "seed_prompt": "str|null",
                    "scorer": "lexical_probe|sa3_flow_probe|clap",
                    "search_mode": "beam|greedy|coordinate",
                    "vocabulary": "csv",
                    "tokens_generated": "int",
                    "beam_width": "int",
                    "branch_factor": "int|null",
                    "runs": "int",
                    "rounds": "int",
                    "candidate_batch_size": "int|null",
                    "prefix": "str|null",
                    "suffix": "str|null",
                    "modifier_axes": "csv|null",
                    "model": "medium|small-music|small-sfx|null",
                    "duration_seconds": "float|null",
                    "score_samples": "int",
                    "logsnr_values": "csv|null",
                    "timestep_values": "csv|null",
                    "min_t": "float",
                    "max_t": "float",
                    "shared_noise": "bool",
                    "antithetic_noise": "bool",
                    "normalize_mse": "bool",
                    "cosine_weight": "float",
                    "conditional_delta_weight": "float",
                    "velocity_convention": "noise_minus_data|data_minus_noise",
                    "device": "str|null",
                    "model_half": "bool",
                    "seed": "int|null",
                },
                produces=[ArtifactKind.BUNDLE],
                status="implemented with lexical and optional SA3 flow probe scorer; CLAP scorer queued",
            ),
            *self._script_operator_specs(),
        ]
        return [_with_ui_fields(spec) for spec in specs]

    def _script_operator_specs(self) -> list[OperatorSpec]:
        return [
            OperatorSpec(
                name=OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"positive_path": "folder", "negative_path": "folder", "model": "same-s|same-l", "limit": "int"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"input_path": "folder", "model": "same-s|same-l", "name": "str"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_STYLE_PROFILE_BUILD,
                maturity="lab",
                backends=[BackendName.CPU],
                inputs=[],
                params={"target_memory_path": "folder", "reference_memory_path": "folder|null", "name": "str"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"profile_path": "file", "prompt": "str", "alpha": "float", "duration_seconds": "float"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_STYLE_DIRECTION_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"direction_path": "file", "prompt": "str", "alpha": "float", "std_alpha": "float"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_AUDIO_DIRECTION_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"direction_path": "file", "prompt": "str", "alpha": "float"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT,
                maturity="probe",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"axis": "str|all", "num_pairs": "int", "layers": "csv"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_AUDIO_RESIDUAL_VECTORS_EXTRACT,
                maturity="probe",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"positive_path": "folder", "negative_path": "folder|null", "baseline": "prompt|negative_audio"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_ALPHA_SWEEP,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"vectors_path": "folder|file", "prompt": "str", "alphas": "csv", "layer": "int"},
                produces=[ArtifactKind.AUDIO, ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_GEOMETRY_AUDIT,
                maturity="probe",
                backends=[BackendName.CPU],
                inputs=["source?"],
                params={"n_components": "int", "limit": "int"},
                produces=[ArtifactKind.BUNDLE],
                status="implemented for local latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_SOFT_PROMPT_OPTIMIZE,
                maturity="probe",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=["source?"],
                params={"target_audio_path": "file|null", "seed_prompt": "str", "optimization_steps": "int"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_SOFT_PROMPT_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"soft_prompt_path": "file", "steps": "int", "seed": "int"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.DATASET_PRE_ENCODE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={
                    "data_dir": "folder",
                    "model": "same-s|same-l",
                    "batch_size": "int",
                    "sample_size": "int",
                    "pad": "bool",
                    "model_half": "bool",
                },
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.MEMORY_QUERY,
                maturity="lab",
                backends=[BackendName.CPU],
                inputs=["source"],
                params={"top_k": "int", "metric": "cosine|euclidean", "exclude_self": "bool"},
                produces=[ArtifactKind.BUNDLE],
                status="implemented for local latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.TRAIN_LORA,
                maturity="danger",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={
                    "encoded_dir": "folder",
                    "data_dir": "folder|null",
                    "steps": "int",
                    "rank": "int",
                    "lora_alpha": "int",
                    "adapter_type": "lora|lora-xs",
                    "dropout": "float",
                    "logger": "wandb|comet|csv|none",
                    "include": "patterns",
                    "exclude": "patterns",
                },
                produces=[ArtifactKind.BUNDLE],
                status="script adapter; long-running training",
            ),
        ]

    def handler_for_recipe(self, recipe: Recipe):
        if recipe.operator in {
            OperatorName.TEXT_TO_AUDIO,
            OperatorName.AUDIO_TO_AUDIO,
            OperatorName.INPAINT,
        }:
            return lambda context: self._run_mlx_generation(recipe, context)
        if recipe.operator in {
            OperatorName.LATENT_BLUR,
            OperatorName.LATENT_DSP,
            OperatorName.LATENT_GRAFT,
            OperatorName.LATENT_RENOISE,
            OperatorName.LATENT_CYCLIC_ROLL,
        }:
            return lambda context: self._run_latent_operator(recipe, context)
        if recipe.operator == OperatorName.LATENT_ENCODE:
            return lambda context: self._run_same_encode(recipe, context)
        if recipe.operator == OperatorName.LATENT_DECODE:
            return lambda context: self._run_same_decode(recipe, context)
        if recipe.operator == OperatorName.MEMORY_QUERY:
            return lambda context: self._run_memory_query(recipe, context)
        if recipe.operator == OperatorName.EXPERIMENT_GEOMETRY_AUDIT:
            return lambda context: self._run_geometry_audit(recipe, context)
        if recipe.operator == OperatorName.EXPERIMENT_PROMPT_SEARCH:
            return lambda context: self._run_prompt_search(recipe, context)
        if recipe.operator in SCRIPT_EXPERIMENT_OPERATORS:
            return lambda context: self._run_script_experiment(recipe, context)
        raise ValueError(f"operator is not implemented yet: {recipe.operator}")

    def _mlx_status(self) -> ModelStatus:
        apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        venv_python = self.mlx_dir / ".venv" / "bin" / "python"
        available = apple_silicon and self.mlx_wrapper.exists() and venv_python.exists()
        if not apple_silicon:
            message = "MLX backend requires Apple Silicon"
        elif not self.mlx_wrapper.exists():
            message = "optimized/mlx/sa3 wrapper is missing"
        elif not venv_python.exists():
            message = "optimized/mlx/.venv is missing; run optimized/mlx/install.sh"
        else:
            message = "MLX wrapper is ready"
        weights_dir = self.mlx_dir / "models" / "mlx"
        return ModelStatus(
            backend=BackendName.MLX,
            available=available,
            loaded=False,
            device="metal" if apple_silicon else None,
            message=message,
            details={
                "wrapper": str(self.mlx_wrapper),
                "venv_python": str(venv_python),
                "weights_dir": str(weights_dir),
                "weights_dir_exists": weights_dir.exists(),
            },
        )

    def _torch_status(self) -> ModelStatus:
        try:
            import torch

            mps_available = bool(torch.backends.mps.is_available())
            cuda_available = bool(torch.cuda.is_available())
            if mps_available:
                device = "mps"
                available = True
            elif cuda_available:
                device = "cuda"
                available = True
            else:
                device = "cpu"
                available = True
            return ModelStatus(
                backend=BackendName.TORCH_MPS,
                available=available,
                loaded=False,
                device=device,
                message="Torch is importable",
                details={
                    "torch_version": torch.__version__,
                    "mps_available": mps_available,
                    "cuda_available": cuda_available,
                },
            )
        except Exception as exc:
            return ModelStatus(
                backend=BackendName.TORCH_MPS,
                available=False,
                loaded=False,
                message=str(exc),
            )

    def _cpu_status(self) -> ModelStatus:
        return ModelStatus(
            backend=BackendName.CPU,
            available=True,
            loaded=True,
            device="cpu",
            message="CPU artifact, metadata, and latent-file operations are available",
        )

    def _run_mlx_generation(self, recipe: Recipe, context: JobContext) -> JobResult:
        status = self._mlx_status()
        if not status.available:
            raise RuntimeError(status.message or "MLX backend is not available")

        params = recipe.params
        model = str(recipe.model or params.get("model") or "medium")
        decoder = params.get("decoder")
        if decoder is None:
            decoder = "same-l" if model == "medium" else "same-s"
        prompt = str(params.get("prompt", ""))
        duration = float(params.get("duration_seconds", 30.0))
        steps = int(params.get("steps", 8))
        output_id, output_path = self.store.reserve_artifact_path(filename=f"{_operator_value(recipe.operator)}.wav")

        command = [
            str(self.mlx_wrapper),
            "--prompt",
            prompt,
            "--dit",
            model,
            "--decoder",
            str(decoder),
            "--seconds",
            str(duration),
            "--steps",
            str(steps),
            "--cfg",
            str(float(params.get("cfg_scale", 1.0))),
            "--apg",
            str(float(params.get("apg_scale", 1.0))),
            "--out",
            str(output_path),
        ]
        seed = recipe.seed if recipe.seed is not None else params.get("seed")
        if seed is not None:
            command.extend(["--seed", str(int(seed))])
        if params.get("negative_prompt"):
            command.extend(["--negative-prompt", str(params["negative_prompt"])])
        if recipe.operator in {OperatorName.AUDIO_TO_AUDIO, OperatorName.INPAINT}:
            source_id = recipe.inputs.get("source")
            if not source_id:
                raise ValueError("audio generation recipe requires inputs.source")
            source = self.store.get_artifact(source_id)
            command.extend(["--init-audio", str(source.path)])
            command.extend(["--init-noise-level", str(float(params.get("init_noise_level", 0.7)))])
        if recipe.operator == OperatorName.INPAINT:
            start = float(params["inpaint_start_seconds"])
            end = float(params["inpaint_end_seconds"])
            command.extend(["--inpaint-range", f"{start},{end}"])

        context.set_progress(0.05, "starting MLX subprocess")
        process = subprocess.Popen(
            command,
            cwd=str(self.mlx_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if context.cancelled():
                process.terminate()
                context.log("cancel requested; terminating subprocess")
                try:
                    return process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return process.wait()
            clean = line.rstrip()
            _update_subprocess_progress(context, clean)
            context.log(clean)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"MLX generation failed with exit code {return_code}")
        if not output_path.exists():
            raise RuntimeError(f"MLX generation did not create output file: {output_path}")

        source_ids = [recipe.inputs["source"]] if "source" in recipe.inputs else []
        artifact = self.store.finalize_audio_file(
            artifact_id=output_id,
            path=output_path,
            recipe=recipe,
            source_artifact_ids=source_ids,
            prompt=prompt,
            metadata={
                "backend": BackendName.MLX.value,
                "model": model,
                "decoder": decoder,
                "steps": steps,
                "duration_seconds": duration,
            },
        )
        return JobResult(artifact_ids=[artifact.artifact_id], metrics={"return_code": return_code})

    def _run_latent_operator(self, recipe: Recipe, context: JobContext) -> JobResult:
        source_id = recipe.inputs.get("source")
        if not source_id:
            raise ValueError("latent operator recipe requires inputs.source")
        source = self.store.get_artifact(source_id)
        if source.latent is None:
            raise ValueError(f"source artifact {source_id} does not have latent metadata")
        source_array = self.store.load_latent_array(source_id)
        x = _time_major_to_bct(source_array)
        params = dict(recipe.params)
        metadata: dict[str, Any] = {
            "backend": recipe.backend.value if isinstance(recipe.backend, BackendName) else str(recipe.backend),
            "operator": _operator_value(recipe.operator),
        }

        context.set_progress(0.15, f"running {recipe.operator}")
        if recipe.operator == OperatorName.LATENT_BLUR:
            spec = LatentBlurSpec(**_dataclass_kwargs(LatentBlurSpec, {"name": "api", **params}))
            result = apply_latent_blur(x, spec)
            metadata["spec"] = _clean_params(params)
        elif recipe.operator == OperatorName.LATENT_DSP:
            spec = LatentDSPSpec(**_dataclass_kwargs(LatentDSPSpec, {"name": "api", **params}))
            donor = self._load_optional_donor(recipe)
            result = apply_latent_dsp(x, spec, donor_latents=donor)
            metadata["spec"] = _clean_params(params)
        elif recipe.operator == OperatorName.LATENT_GRAFT:
            donor = self._load_required_donor(recipe)
            mask = LatentMaskSpec(**_dataclass_kwargs(LatentMaskSpec, {"name": "api", **params}))
            channels = select_latent_channels(x, mask)
            result = graft_latent_channels(x, donor, channels, amount=float(params.get("amount", 1.0)))
            metadata["selected_channels"] = channels
        elif recipe.operator == OperatorName.LATENT_RENOISE:
            mask = LatentMaskSpec(**_dataclass_kwargs(LatentMaskSpec, {"name": "api", **params}))
            channels = select_latent_channels(x, mask)
            result = masked_latent_noise(
                x,
                channels,
                sigma=float(params.get("sigma", params.get("amount", 0.4))),
                seed=int(recipe.seed if recipe.seed is not None else params.get("seed", 0)),
            )
            metadata["selected_channels"] = channels
        elif recipe.operator == OperatorName.LATENT_CYCLIC_ROLL:
            shift_frames = int(params.get("shift_frames", 0))
            strength = params.get("strength")
            if strength is None:
                result = cyclic_roll_latents(x, shift_frames)
            else:
                result = cyclic_mix_latents(
                    x,
                    shift_frames,
                    strength=float(strength),
                    symmetric=bool(params.get("symmetric", True)),
                )
        else:
            raise ValueError(f"unsupported latent operator: {recipe.operator}")

        output = _bct_to_time_major(result)
        context.set_progress(0.75, "saving latent artifact")
        artifact = self.store.store_latent_array(
            output,
            latent_rate=source.latent.latent_rate,
            recipe=recipe,
            source_artifact_ids=[item for item in [source_id, recipe.inputs.get("donor")] if item],
            prompt=source.prompt,
            metadata=metadata,
        )
        return JobResult(artifact_ids=[artifact.artifact_id])

    def _run_same_encode(self, recipe: Recipe, context: JobContext) -> JobResult:
        source_id = recipe.inputs.get("source")
        if not source_id:
            raise ValueError("latent encode recipe requires inputs.source")
        source = self.store.get_artifact(source_id)
        if source.kind != ArtifactKind.AUDIO or source.audio is None:
            raise ValueError(f"source artifact {source_id} is not an audio artifact")

        params = dict(recipe.params)
        model_name = str(recipe.model or params.get("model") or "same-l")
        adapter = self._same_adapter(model_name, recipe.backend)
        context.set_progress(0.20, f"loaded SAME encoder {model_name}")

        item = adapter.encode_file(
            source.path,
            item_id=source.artifact_id,
            prompt=params.get("prompt") or source.prompt,
            chunked=bool(params.get("chunked", False)),
            metadata={
                "source_artifact_id": source_id,
                "source_path": str(source.path),
                "backend": _backend_value(recipe.backend),
            },
            chunk_size=int(params.get("chunk_size", 128)),
            overlap=int(params.get("overlap", 32)),
        )
        context.set_progress(0.75, "saving encoded latent")
        artifact = self.store.store_latent_array(
            item.latent,
            latent_rate=item.latent_rate,
            sample_rate=item.sample_rate,
            recipe=recipe,
            source_artifact_ids=[source_id],
            prompt=item.prompt,
            metadata=item.metadata,
        )
        return JobResult(artifact_ids=[artifact.artifact_id])

    def _run_same_decode(self, recipe: Recipe, context: JobContext) -> JobResult:
        source_id = recipe.inputs.get("source")
        if not source_id:
            raise ValueError("latent decode recipe requires inputs.source")
        source = self.store.get_artifact(source_id)
        if source.latent is None:
            raise ValueError(f"source artifact {source_id} does not have latent metadata")

        params = dict(recipe.params)
        model_name = str(recipe.model or params.get("model") or "same-l")
        adapter = self._same_adapter(model_name, recipe.backend)
        context.set_progress(0.20, f"loaded SAME decoder {model_name}")

        latents = _time_major_to_bct(self.store.load_latent_array(source_id))
        device = getattr(adapter.autoencoder, "device", None)
        if device is not None:
            latents = latents.to(str(device))
        audio = adapter.decode_latents(
            latents,
            chunked=bool(params.get("chunked", False)),
            chunk_size=int(params.get("chunk_size", 128)),
            overlap=int(params.get("overlap", 32)),
        )
        output_id, output_path = self.store.reserve_artifact_path(filename=f"{source_id}.decoded.wav")
        _write_audio_tensor(output_path, audio, sample_rate=int(adapter.sample_rate))
        context.set_progress(0.80, "saving decoded audio")
        artifact = self.store.finalize_audio_file(
            artifact_id=output_id,
            path=output_path,
            recipe=recipe,
            source_artifact_ids=[source_id],
            prompt=source.prompt,
            metadata={
                "backend": _backend_value(recipe.backend),
                "model": model_name,
                "source_artifact_id": source_id,
            },
        )
        return JobResult(artifact_ids=[artifact.artifact_id])

    def _run_memory_query(self, recipe: Recipe, context: JobContext) -> JobResult:
        source_id = recipe.inputs.get("source")
        if not source_id:
            raise ValueError("memory query recipe requires inputs.source")
        source = self.store.get_artifact(source_id)
        if source.latent is None:
            raise ValueError(f"source artifact {source_id} does not have latent metadata")

        params = dict(recipe.params)
        top_k = max(1, int(params.get("top_k", 5)))
        metric = str(params.get("metric", "cosine"))
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("memory query metric must be 'cosine' or 'euclidean'")
        exclude_self = bool(params.get("exclude_self", True))

        query_latent = self.store.load_latent_array(source_id)
        candidates: list[LatentItem] = []
        for record in self.store.list_artifacts(kind=ArtifactKind.LATENT):
            if exclude_self and record.artifact_id == source_id:
                continue
            if record.latent is None or record.latent.shape[1] != query_latent.shape[1]:
                continue
            latent = self.store.load_latent_array(record.artifact_id)
            candidates.append(_latent_record_to_item(record, latent))

        context.set_progress(0.45, f"indexed {len(candidates)} latent memories")
        results = []
        if candidates:
            index = LatentMemoryIndex(candidates)
            for result in index.query(query_latent, top_k=top_k, metric=metric):
                results.append(
                    {
                        "artifact_id": result.item_id,
                        "score": result.score,
                        "distance": result.distance,
                        "components": result.components,
                        "item": result.item.shallow_metadata(),
                    }
                )

        output_id, output_path = self.store.reserve_artifact_path(filename="memory_query.json")
        payload = {
            "operator": _operator_value(recipe.operator),
            "source_artifact_id": source_id,
            "metric": metric,
            "top_k": top_k,
            "exclude_self": exclude_self,
            "candidate_count": len(candidates),
            "results": results,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        context.set_progress(0.85, "saving memory query")
        artifact = self.store.finalize_bundle_path(
            artifact_id=output_id,
            path=output_path,
            recipe=recipe,
            source_artifact_ids=[source_id],
            label="memory query",
            metadata={
                "operator": _operator_value(recipe.operator),
                "result_count": len(results),
                "metric": metric,
                "top_k": top_k,
            },
        )
        return JobResult(
            artifact_ids=[artifact.artifact_id],
            metrics={"result_count": len(results), "candidate_count": len(candidates), "metric": metric},
        )

    def _run_geometry_audit(self, recipe: Recipe, context: JobContext) -> JobResult:
        params = dict(recipe.params)
        n_components = max(1, int(params.get("n_components", 8)))
        limit = max(0, int(params.get("limit", 0)))
        source_id = recipe.inputs.get("source")
        source = self.store.get_artifact(source_id) if source_id else None
        if source is not None and source.latent is None:
            raise ValueError(f"source artifact {source_id} does not have latent metadata")

        records = self.store.list_artifacts(kind=ArtifactKind.LATENT, session_id=recipe.session_id)
        if not records:
            records = self.store.list_artifacts(kind=ArtifactKind.LATENT)
        if source is not None:
            records = [source, *[record for record in records if record.artifact_id != source.artifact_id]]
        if limit:
            records = records[:limit]

        items: list[LatentItem] = []
        anchor_dim = source.latent.shape[1] if source and source.latent else None
        for record in records:
            if record.latent is None:
                continue
            if anchor_dim is not None and record.latent.shape[1] != anchor_dim:
                continue
            latent = self.store.load_latent_array(record.artifact_id)
            if anchor_dim is None:
                anchor_dim = latent.shape[1]
            if latent.shape[1] != anchor_dim:
                continue
            items.append(_latent_record_to_item(record, latent))

        if not items:
            raise ValueError("geometry audit requires at least one local latent artifact")

        context.set_progress(0.45, f"auditing {len(items)} latent artifacts")
        report = geometry_report(items, n_components=n_components)
        output_id, output_path = self.store.reserve_artifact_path(filename="geometry_report.json")
        payload = {
            "operator": _operator_value(recipe.operator),
            "source_artifact_id": source_id,
            "latent_count": len(items),
            "candidate_count": len(records),
            "n_components": n_components,
            "artifacts": [item.shallow_metadata() for item in items],
            "report": report,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        context.set_progress(0.85, "saving geometry audit")
        artifact = self.store.finalize_bundle_path(
            artifact_id=output_id,
            path=output_path,
            recipe=recipe,
            source_artifact_ids=[source_id] if source_id else [],
            label="geometry audit",
            metadata={
                "operator": _operator_value(recipe.operator),
                "latent_count": len(items),
                "n_components": n_components,
                "kept_variance_fraction": report.get("kept_variance_fraction"),
            },
        )
        return JobResult(
            artifact_ids=[artifact.artifact_id],
            metrics={
                "latent_count": len(items),
                "n_components": n_components,
                "kept_variance_fraction": report.get("kept_variance_fraction"),
            },
        )

    def _run_prompt_search(self, recipe: Recipe, context: JobContext) -> JobResult:
        params = dict(recipe.params)
        source_id = recipe.inputs.get("source")
        source = self.store.get_artifact(source_id) if source_id else None
        if source is not None and source.kind != ArtifactKind.AUDIO:
            raise ValueError(f"source artifact {source_id} must be audio for prompt search")

        target_audio_path = str(params.get("target_audio_path") or "").strip()
        if not target_audio_path and source is not None:
            target_audio_path = str(source.path)
        seed_prompt = str(params.get("seed_prompt") or "").strip()
        if not seed_prompt:
            if source is not None and source.prompt:
                seed_prompt = source.prompt
            elif target_audio_path:
                seed_prompt = prompt_seed_from_audio_path(target_audio_path, extra_tags=list(source.tags if source else []))
            else:
                seed_prompt = "audio texture"

        seed = int(recipe.seed if recipe.seed is not None else params.get("seed", 0) or 0)
        search_mode = str(params.get("search_mode") or "beam").lower()
        vocabulary = _parse_prompt_vocabulary(params.get("vocabulary"))
        target_tokens = _prompt_search_target_tokens(
            target_audio_path=target_audio_path,
            seed_prompt=seed_prompt,
            source=source,
        )
        scorer_kind = str(params.get("scorer") or "lexical_probe").lower()
        if scorer_kind == "lexical_probe":
            scorer = _lexical_prompt_scorer(target_tokens)
            batch_scorer = lambda prompts: [scorer(prompt) for prompt in prompts]
            scorer_metadata: dict[str, Any] = {
                "kind": "lexical_probe",
                "model_backed": False,
                "notes": "Deterministic local scorer for wiring and triage; choose sa3_flow_probe for frozen SA3 flow-loss scoring.",
                "target_tokens": sorted(target_tokens)[:48],
            }
        elif scorer_kind == "sa3_flow_probe":
            context.set_progress(0.10, "loading SA3 flow prompt scorer")
            batch_scorer, scorer_metadata = self._sa3_flow_prompt_batch_scorer(
                recipe,
                context=context,
                params=params,
                target_audio_path=target_audio_path,
                source=source,
                seed=seed,
            )
            scorer = lambda prompt: batch_scorer([prompt])[0]
        elif scorer_kind == "clap":
            raise RuntimeError("CLAP prompt scorer is not implemented yet; choose lexical_probe or sa3_flow_probe.")
        else:
            raise ValueError("prompt search scorer must be 'lexical_probe', 'sa3_flow_probe', or 'clap'")

        context.set_progress(0.20, f"running {search_mode} prompt search with {scorer_kind}")
        if search_mode == "coordinate":
            axes = _parse_modifier_axes(params.get("modifier_axes")) or default_modifier_axes()
            result = coordinate_prompt_search(
                seed_prompt,
                axes,
                scorer,
                rounds=max(1, int(params.get("rounds", 2) or 2)),
            )
            tokens = _prompt_tokens(result.prompt)
            beams: list[dict[str, Any]] = []
        elif search_mode == "greedy":
            result = greedy_token_prompt_search(
                vocabulary,
                batch_scorer,
                tokens_generated=max(1, int(params.get("tokens_generated", 4) or 4)),
                runs=max(1, int(params.get("runs", 4) or 4)),
                candidate_batch_size=_optional_positive_int(params.get("candidate_batch_size")),
                seed=seed,
                prefix=str(params.get("prefix") or seed_prompt),
                suffix=str(params.get("suffix") or ""),
                separator=str(params.get("separator") or " "),
            )
            tokens = result.tokens
            beams = []
        elif search_mode == "beam":
            result = beam_token_prompt_search(
                vocabulary,
                batch_scorer,
                tokens_generated=max(1, int(params.get("tokens_generated", 4) or 4)),
                beam_width=max(1, int(params.get("beam_width", 4) or 4)),
                branch_factor=_optional_positive_int(params.get("branch_factor")),
                candidate_batch_size=_optional_positive_int(params.get("candidate_batch_size")),
                seed=seed,
                prefix=str(params.get("prefix") or seed_prompt),
                suffix=str(params.get("suffix") or ""),
                separator=str(params.get("separator") or " "),
            )
            tokens = result.tokens
            beams = [{"prompt": prompt, "score": float(score)} for prompt, score in result.beams[:12]]
        else:
            raise ValueError("prompt search mode must be 'beam', 'greedy', or 'coordinate'")

        context.set_progress(0.70, "saving prompt search bundle")
        history = [{"prompt": prompt, "score": float(score)} for prompt, score in result.history[:300]]
        payload = {
            "operator": _operator_value(recipe.operator),
            "status": "probe",
            "scorer": scorer_metadata,
            "source_artifact_id": source_id,
            "target_audio_path": target_audio_path or None,
            "search_mode": search_mode,
            "seed": seed,
            "seed_prompt": seed_prompt,
            "prompt": result.prompt,
            "score": float(result.score),
            "tokens": tokens,
            "beams": beams,
            "candidate_count": len(result.history),
            "history": history,
        }
        output_id, output_path = self.store.reserve_artifact_path(filename="prompt_search.json")
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        artifact = self.store.finalize_bundle_path(
            artifact_id=output_id,
            path=output_path,
            recipe=recipe,
            source_artifact_ids=[source_id] if source_id else [],
            label="prompt search",
            prompt=result.prompt,
            metadata={
                "operator": _operator_value(recipe.operator),
                "metric": scorer_metadata["kind"],
                "result_count": 1,
                "candidate_count": len(result.history),
                "search_mode": search_mode,
                "scorer": scorer_metadata["kind"],
            },
        )
        return JobResult(
            artifact_ids=[artifact.artifact_id],
            metrics={
                "prompt": result.prompt,
                "score": float(result.score),
                "candidate_count": len(result.history),
                "scorer": scorer_metadata["kind"],
                "search_mode": search_mode,
            },
        )

    def _sa3_flow_prompt_batch_scorer(
        self,
        recipe: Recipe,
        *,
        context: JobContext,
        params: dict[str, Any],
        target_audio_path: str,
        source: ArtifactRecord | None,
        seed: int,
    ):
        if not target_audio_path:
            raise ValueError("SA3 flow prompt scorer requires params.target_audio_path or an audio source artifact")
        try:
            import torchaudio
            from stable_audio_3 import StableAudioModel
        except Exception as exc:
            raise RuntimeError("SA3 flow prompt scorer requires the Stable Audio 3 torch runtime extras") from exc

        model_name = str(recipe.model or params.get("model") or "medium")
        device = str(params.get("device") or _device_for_backend(recipe.backend))
        _ensure_sa3_model_cache_space(model_name)
        context.set_progress(0.11, f"loading SA3 flow scorer model={model_name} device={device}")
        context.log(f"SA3 flow prompt scorer loading model={model_name} device={device}")
        stable_model = StableAudioModel.from_pretrained(
            model_name,
            device=device,
            model_half=bool(params.get("model_half", False)),
        )
        context.set_progress(0.14, "loading target audio for SA3 flow scorer")
        audio, sample_rate = torchaudio.load(target_audio_path)
        duration = float(params.get("duration_seconds") or params.get("duration") or 0.0)
        if duration <= 0:
            duration = float(audio.shape[-1] / sample_rate)
        seed_prompt = str(params.get("seed_prompt") or (source.prompt if source else "") or "audio texture")
        conditioning = [{"prompt": seed_prompt, "seconds_total": duration}]
        audio_sample_size = stable_model._adapt_sample_size(conditioning, 5292032, duration_padding_sec=6.0)
        context.set_progress(0.16, "encoding target audio to SA3 latents")
        target_latents, _ = stable_model._encode_audio_input((sample_rate, audio), audio_sample_size)
        timestep_values = parse_float_sequence(params.get("timestep_values"))
        if not timestep_values:
            timestep_values = timesteps_from_logsnr_values(params.get("logsnr_values"))
        effective_samples = len(timestep_values) if timestep_values else max(1, int(params.get("score_samples", 1) or 1))
        context.set_progress(0.18, f"prepared SA3 flow scorer with {effective_samples} sample(s)")
        context.log(f"SA3 flow scorer target duration={duration:.2f}s samples={effective_samples}")
        scored_count = 0

        def batch_scorer(prompts: list[str]) -> list[float]:
            nonlocal scored_count
            context.set_progress(
                min(0.68, 0.22 + min(scored_count, 1000) / 1000 * 0.42),
                f"scoring {len(prompts)} prompt candidate(s) with SA3 flow loss",
            )
            losses = sa3_flow_losses_for_prompts(
                stable_model,
                target_latents,
                prompts,
                duration=duration,
                seed=seed,
                min_t=float(params.get("min_t", 0.05) or 0.05),
                max_t=float(params.get("max_t", 0.95) or 0.95),
                score_samples=effective_samples,
                shared_noise=bool(params.get("shared_noise", True)),
                timestep_values=timestep_values or None,
                cosine_weight=float(params.get("cosine_weight", 0.0) or 0.0),
                antithetic_noise=bool(params.get("antithetic_noise", False)),
                normalize_mse=bool(params.get("normalize_mse", True)),
                conditional_delta_weight=float(params.get("conditional_delta_weight", 0.0) or 0.0),
                velocity_convention=str(params.get("velocity_convention") or "noise_minus_data"),
            )
            scored_count += len(prompts)
            context.set_progress(
                min(0.68, 0.22 + min(scored_count, 1000) / 1000 * 0.42),
                f"scored {scored_count} prompt candidate(s) with SA3 flow loss",
            )
            return [-loss for loss in losses]

        metadata = {
            "kind": "sa3_flow_probe",
            "model_backed": True,
            "model": model_name,
            "device": device,
            "duration_seconds": duration,
            "score_samples": effective_samples,
            "timestep_values": timestep_values,
            "shared_noise": bool(params.get("shared_noise", True)),
            "antithetic_noise": bool(params.get("antithetic_noise", False)),
            "normalize_mse": bool(params.get("normalize_mse", True)),
            "cosine_weight": float(params.get("cosine_weight", 0.0) or 0.0),
            "conditional_delta_weight": float(params.get("conditional_delta_weight", 0.0) or 0.0),
            "velocity_convention": str(params.get("velocity_convention") or "noise_minus_data"),
            "target_audio_path": target_audio_path,
            "notes": "Frozen SA3 flow-loss scorer; prompt scores are negative losses against the target latent.",
        }
        return batch_scorer, metadata

    def _run_script_experiment(self, recipe: Recipe, context: JobContext) -> JobResult:
        output_id, marker_path = self.store.reserve_artifact_path(filename=f"{_operator_slug(recipe.operator)}.zip")
        output_root = marker_path.parent / "script_output"
        output_root.mkdir(parents=True, exist_ok=True)
        command, primary_output = self._script_experiment_command(recipe, output_root)

        context.set_progress(0.05, f"starting {_operator_value(recipe.operator)}")
        return_code = self._run_subprocess(command, cwd=self.repo_root, context=context)
        if return_code != 0:
            raise RuntimeError(f"{recipe.operator} failed with exit code {return_code}")
        if not primary_output.exists():
            raise RuntimeError(f"{recipe.operator} did not create expected output: {primary_output}")

        context.set_progress(0.85, "registering experiment artifacts")
        artifact_ids = self._register_script_outputs(
            recipe,
            primary_output=primary_output,
            bundle_artifact_id=output_id,
        )
        return JobResult(
            artifact_ids=artifact_ids,
            metrics={
                "return_code": return_code,
                "primary_output": str(primary_output),
                "command": _redacted_command(command),
            },
        )

    def _script_experiment_command(self, recipe: Recipe, output_root: Path) -> tuple[list[str], Path]:
        params = dict(recipe.params)
        model = str(recipe.model or params.get("model") or _default_script_model(recipe.operator))
        seed = recipe.seed if recipe.seed is not None else params.get("seed")

        def script(name: str) -> list[str]:
            return [sys.executable, str(self.repo_root / "scripts" / name)]

        def torch_common(command: list[str]) -> None:
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            if bool(params.get("no_half", False)):
                command.append("--no-half")

        if recipe.operator == OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS:
            output = output_root / "audio_style_vectors"
            command = script("extract_audio_style_vectors.py")
            _append_required(command, "--positive", params, "positive_path")
            _append_required(command, "--negative", params, "negative_path")
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            _append_bool(command, "--chunked", params.get("chunked"))
            _append_int(command, "--limit", params.get("limit"))
            _append_option(command, "--name", params.get("name"))
            _append_bool(command, "--normalize-frame", params.get("normalize_frame"))
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE:
            output = output_root / "positive_style"
            memory = output / "memory"
            profile = output / "profile.npz"
            command = script("build_positive_audio_style_profile.py")
            _append_required(command, "--input", params, "input_path")
            command.extend(["--memory-output", str(memory), "--profile-output", str(profile), "--model", model])
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            _append_bool(command, "--chunked", params.get("chunked"))
            _append_int(command, "--limit", params.get("limit"))
            _append_option(command, "--name", params.get("name"))
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_STYLE_PROFILE_BUILD:
            output = output_root / "style_profile"
            profile = output / "profile.npz"
            direction = output / "direction.npz"
            command = script("build_same_style_profile.py")
            _append_required(command, "--target", params, "target_memory_path")
            command.extend(["--output", str(profile)])
            _append_option(command, "--name", params.get("name"))
            if params.get("reference_memory_path"):
                command.extend(["--reference", str(params["reference_memory_path"]), "--direction-output", str(direction)])
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE:
            output = output_root / "styled_profile.wav"
            command = script("generate_sa3_with_style_profile.py")
            _append_required(command, "--profile", params, "profile_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--alpha", params.get("alpha"))
            if params.get("match_std") is False or params.get("no_std"):
                command.append("--no-std")
            _append_bool(command, "--save-original", params.get("save_original"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_STYLE_DIRECTION_GENERATE:
            output = output_root / "styled_direction.wav"
            command = script("generate_sa3_with_style_direction.py")
            _append_required(command, "--direction", params, "direction_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--alpha", params.get("alpha"))
            _append_float(command, "--std-alpha", params.get("std_alpha"))
            _append_bool(command, "--save-original", params.get("save_original"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_AUDIO_DIRECTION_GENERATE:
            output = output_root / "audio_direction.wav"
            command = script("generate_sa3_with_audio_direction.py")
            _append_required(command, "--direction", params, "direction_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--alpha", params.get("alpha"))
            _append_bool(command, "--save-original", params.get("save_original"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT:
            output = output_root / "sa3_vectors"
            command = script("extract_sa3_vectors.py")
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--axis", params.get("axis"))
            _append_int(command, "--num-pairs", params.get("num_pairs"))
            _append_generation_params(command, params, seed=seed)
            _append_option(command, "--layers", params.get("layers"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_AUDIO_RESIDUAL_VECTORS_EXTRACT:
            output = output_root / "audio_residual_vectors"
            command = script("extract_audio_residual_vectors.py")
            _append_required(command, "--positive", params, "positive_path")
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--negative", params.get("negative_path"))
            _append_option(command, "--baseline", params.get("baseline"))
            _append_prompt(command, params)
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--init-noise-level", params.get("init_noise_level"))
            _append_option(command, "--layers", params.get("layers"))
            _append_int(command, "--limit", params.get("limit"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_ALPHA_SWEEP:
            output = output_root / "alpha_sweep"
            command = script("run_sa3_alpha_sweep.py")
            _append_required(command, "--vectors", params, "vectors_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--alphas", params.get("alphas"))
            _append_generation_params(command, params, seed=seed)
            _append_int(command, "--layer", params.get("layer"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_SOFT_PROMPT_OPTIMIZE:
            output = output_root / "soft_prompt.pt"
            target_audio = params.get("target_audio_path")
            if not target_audio and recipe.inputs.get("source"):
                target_audio = str(self.store.get_artifact(recipe.inputs["source"]).path)
            if not target_audio:
                raise ValueError("soft prompt optimization requires params.target_audio_path or inputs.source")
            command = script("optimize_sa3_soft_prompt.py")
            command.extend(["--target-audio", str(target_audio), "--output", str(output), "--model", model])
            _append_option(command, "--seed-prompt", params.get("seed_prompt"))
            _append_float(command, "--duration", params.get("duration_seconds", params.get("duration")))
            _append_int(command, "--optimization-steps", params.get("optimization_steps"))
            _append_float(command, "--lr", params.get("lr"))
            _append_float(command, "--reg-weight", params.get("reg_weight"))
            _append_int(command, "--seed", seed)
            _append_option(command, "--train-keys", params.get("train_keys"))
            _append_option(command, "--velocity-convention", params.get("velocity_convention"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_SOFT_PROMPT_GENERATE:
            output = output_root / "soft_prompt.wav"
            command = script("generate_sa3_with_soft_prompt.py")
            _append_required(command, "--soft-prompt", params, "soft_prompt_path")
            command.extend(["--output", str(output), "--model", model])
            _append_int(command, "--steps", params.get("steps"))
            _append_float(command, "--cfg-scale", params.get("cfg_scale"))
            _append_int(command, "--seed", seed)
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.DATASET_PRE_ENCODE:
            output = output_root / "encoded_dataset"
            command = script("pre_encode_dataset.py")
            _append_required(command, "--data_dir", params, "data_dir")
            command.extend(["--output_path", str(output), "--model", model])
            _append_int(command, "--batch_size", params.get("batch_size"))
            _append_int(command, "--sample_size", params.get("sample_size"))
            _append_bool(command, "--model_half", params.get("model_half"))
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            _append_bool(command, "--pad", params.get("pad"))
            return command, output

        if recipe.operator == OperatorName.TRAIN_LORA:
            output = output_root / "lora_checkpoints"
            command = script("train_lora.py")
            command.extend(["--save_dir", str(output), "--model", model])
            if params.get("encoded_dir"):
                command.extend(["--encoded_dir", str(params["encoded_dir"])])
            elif params.get("data_dir"):
                command.extend(["--data_dir", str(params["data_dir"])])
            else:
                raise ValueError("LoRA training requires params.encoded_dir or params.data_dir")
            for flag, key in [
                ("--rank", "rank"),
                ("--lora_alpha", "lora_alpha"),
                ("--adapter_type", "adapter_type"),
                ("--dropout", "dropout"),
                ("--svd_bases_path", "svd_bases_path"),
                ("--base_precision", "base_precision"),
                ("--lora_checkpoint", "lora_checkpoint"),
                ("--lr", "lr"),
                ("--steps", "steps"),
                ("--batch_size", "batch_size"),
                ("--duration", "duration_seconds"),
                ("--device", "device"),
                ("--logger", "logger"),
                ("--name", "name"),
                ("--checkpoint_every", "checkpoint_every"),
                ("--log_every", "log_every"),
                ("--demo_every", "demo_every"),
                ("--num_workers", "num_workers"),
            ]:
                _append_option(command, flag, params.get(key))
            _append_multi_option(command, "--include", params.get("include"))
            _append_multi_option(command, "--exclude", params.get("exclude"))
            _append_int(command, "--seed", seed)
            if not params.get("device"):
                _append_option(command, "--device", _device_for_backend(recipe.backend))
            return command, output

        raise ValueError(f"unsupported script experiment operator: {recipe.operator}")

    def _run_subprocess(self, command: list[str], *, cwd: Path, context: JobContext) -> int:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            _update_subprocess_progress(context, clean)
            context.log(clean)
        return process.wait()

    def _register_script_outputs(self, recipe: Recipe, *, primary_output: Path, bundle_artifact_id: str) -> list[str]:
        source_ids = [value for value in recipe.inputs.values() if value]
        audio_paths = []
        if primary_output.is_dir():
            audio_paths = [path for path in sorted(primary_output.rglob("*")) if path.suffix.lower() in AUDIO_EXTENSIONS]
        elif primary_output.suffix.lower() in AUDIO_EXTENSIONS:
            audio_paths = [primary_output]

        audio_artifact_ids: list[str] = []
        for audio_path in audio_paths:
            artifact = self.store.finalize_audio_file(
                artifact_id=new_id("art"),
                path=audio_path,
                recipe=recipe,
                source_artifact_ids=source_ids,
                prompt=str(recipe.params.get("prompt", "")) or None,
                label=audio_path.stem,
                metadata={
                    "backend": _backend_value(recipe.backend),
                    "operator": _operator_value(recipe.operator),
                    "script_output_path": str(audio_path),
                },
            )
            audio_artifact_ids.append(artifact.artifact_id)

        if primary_output.is_dir() or not audio_paths:
            bundle = self.store.finalize_bundle_path(
                artifact_id=bundle_artifact_id,
                path=primary_output,
                recipe=recipe,
                source_artifact_ids=source_ids,
                label=_operator_value(recipe.operator),
                prompt=str(recipe.params.get("prompt", "")) or None,
                metadata={
                    "backend": _backend_value(recipe.backend),
                    "operator": _operator_value(recipe.operator),
                    "script_output_path": str(primary_output),
                },
            )
            return [*audio_artifact_ids, bundle.artifact_id]

        return audio_artifact_ids

    def _load_optional_donor(self, recipe: Recipe):
        donor_id = recipe.inputs.get("donor")
        if not donor_id:
            return None
        return _time_major_to_bct(self.store.load_latent_array(donor_id))

    def _load_required_donor(self, recipe: Recipe):
        donor = self._load_optional_donor(recipe)
        if donor is None:
            raise ValueError("operator requires inputs.donor")
        return donor

    def _same_adapter(self, model_name: str, backend: BackendName | str):
        from latent_audio_primitives.adapters.stable_audio3 import SAMEAutoencoderAdapter

        device = _device_for_backend(backend)
        key = (model_name, device)
        if key not in self._same_adapters:
            self._same_adapters[key] = SAMEAutoencoderAdapter.from_pretrained(model_name, device=device)
        return self._same_adapters[key]


def _time_major_to_bct(array: np.ndarray):
    import torch

    latent = np.asarray(array, dtype=np.float32)
    if latent.ndim != 2:
        raise ValueError(f"latent must be 2D time-major, got shape {latent.shape}")
    return torch.from_numpy(latent.T.copy()).unsqueeze(0)


def _bct_to_time_major(tensor: Any) -> np.ndarray:
    detached = tensor.detach().cpu().float()
    if detached.ndim == 3:
        detached = detached.squeeze(0)
    if detached.ndim != 2:
        raise ValueError(f"operator produced invalid latent shape {tuple(detached.shape)}")
    return detached.numpy().T.astype(np.float32, copy=False)


def _latent_record_to_item(record: ArtifactRecord, latent: np.ndarray) -> LatentItem:
    if record.latent is None:
        raise ValueError(f"artifact {record.artifact_id} does not have latent metadata")
    return LatentItem(
        item_id=record.artifact_id,
        latent=latent,
        latent_rate=record.latent.latent_rate,
        sample_rate=record.latent.sample_rate,
        prompt=record.prompt,
        descriptors=_numeric_descriptors(record.metadata.get("descriptors")),
        labels={"label": record.label, "tags": list(record.tags)},
        metadata={
            "artifact_id": record.artifact_id,
            "path": str(record.path),
            "recipe_id": record.recipe_id,
            "session_id": record.session_id,
        },
    )


def _numeric_descriptors(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    descriptors: dict[str, float] = {}
    for key, descriptor in value.items():
        try:
            descriptors[str(key)] = float(descriptor)
        except (TypeError, ValueError):
            continue
    return descriptors


def _parse_prompt_vocabulary(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value]
    else:
        text = str(value or "").strip()
        if not text:
            return list(DEFAULT_PROMPT_SEARCH_VOCABULARY)
        tokens = [part.strip() for part in re.split(r"[,;\n]+", text)]
        if len(tokens) == 1:
            tokens = [part.strip() for part in text.split()]
    deduped = []
    for token in tokens:
        normalized = re.sub(r"\s+", " ", token).strip().lower()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped or list(DEFAULT_PROMPT_SEARCH_VOCABULARY)


def _parse_modifier_axes(value: Any) -> list[list[str]]:
    text = str(value or "").strip()
    if not text:
        return []
    axes: list[list[str]] = []
    for axis_text in re.split(r"[;\n]+", text):
        axis = [part.strip().lower() for part in re.split(r"[|,]+", axis_text) if part.strip()]
        if axis:
            axes.append(axis)
    return axes


def _optional_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _prompt_search_target_tokens(
    *,
    target_audio_path: str,
    seed_prompt: str,
    source: ArtifactRecord | None,
) -> set[str]:
    chunks = [seed_prompt]
    if target_audio_path:
        path = Path(target_audio_path)
        chunks.extend([path.stem, *path.parent.parts[-2:]])
    if source is not None:
        chunks.extend(
            [
                source.label or "",
                source.prompt or "",
                source.file.filename if source.file else "",
                *source.tags,
            ]
        )
    tokens = set(_prompt_tokens(" ".join(chunks)))
    return tokens or {"audio", "texture"}


def _lexical_prompt_scorer(target_tokens: set[str]):
    def score(prompt: str) -> float:
        tokens = _prompt_tokens(prompt)
        if not tokens:
            return 0.0
        unique = set(tokens)
        overlap = len(unique & target_tokens)
        coverage = overlap / max(len(target_tokens), 1)
        repeated = max(0, len(tokens) - len(unique))
        diversity = len(unique) / len(tokens)
        length_bonus = min(len(tokens), 14) * 0.01
        overlap_density = sum(1 for token in tokens if token in target_tokens) * 0.05
        return float(coverage + overlap_density + diversity * 0.05 + length_bonus - repeated * 0.02)

    return score


def _prompt_tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token]


def _dataclass_kwargs(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in params.items() if key in valid}


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in params.items()}


def _operator_value(operator: OperatorName | str) -> str:
    return operator.value if isinstance(operator, OperatorName) else str(operator)


def _operator_slug(operator: OperatorName | str) -> str:
    return _operator_value(operator).replace(".", "_").replace("/", "_")


def _backend_value(backend: BackendName | str) -> str:
    return backend.value if isinstance(backend, BackendName) else str(backend)


def _device_for_backend(backend: BackendName | str) -> str:
    value = _backend_value(backend)
    if value == BackendName.TORCH_CPU.value:
        return "cpu"
    if value == BackendName.TORCH_MPS.value:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return "cpu"


def _ensure_sa3_model_cache_space(model_name: str) -> None:
    if model_name != "medium" or _hf_cached_file("stabilityai/stable-audio-3-medium", "model.safetensors"):
        return
    cache_dir = _hf_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_dir).free
    if free_bytes >= SA3_MEDIUM_CHECKPOINT_BYTES:
        return
    raise RuntimeError(
        "SA3 Medium torch checkpoint is not cached and needs about "
        f"{_format_bytes(SA3_MEDIUM_CHECKPOINT_BYTES)} free in the Hugging Face cache; "
        f"{_format_bytes(free_bytes)} is currently free at {cache_dir}. "
        "Free disk space, move HF_HOME to a larger volume, or use the MLX generation path for Medium."
    )


def _hf_cached_file(repo_id: str, filename: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo_id, filename), str)
    except Exception:
        return False


def _hf_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    return (Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface") / "hub"


def _format_bytes(value: int | float) -> str:
    gib = float(value) / (1024**3)
    return f"{gib:.1f} GiB"


def _write_audio_tensor(path: Path, audio: Any, *, sample_rate: int) -> None:
    import soundfile as sf

    arr = audio.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"decoded audio must be shaped channels x samples, got {arr.shape}")
    arr = np.clip(arr, -1.0, 1.0).T
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate)


def _update_subprocess_progress(context: JobContext, line: str) -> None:
    match = PROGRESS_PERCENT_RE.search(line)
    if not match:
        return
    percent = float(match.group(1))
    if not 0 <= percent <= 100:
        return
    progress = 0.05 + (percent / 100.0) * 0.80
    context.set_progress(min(0.84, max(0.05, progress)), line[:240])


def _default_script_model(operator: OperatorName | str) -> str:
    if operator in {
        OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS,
        OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE,
        OperatorName.DATASET_PRE_ENCODE,
    }:
        return "same-l"
    if operator == OperatorName.TRAIN_LORA:
        return "medium-base"
    return "medium"


def _append_required(command: list[str], flag: str, params: dict[str, Any], key: str) -> None:
    value = params.get(key)
    if value in {None, ""}:
        raise ValueError(f"missing required parameter: {key}")
    command.extend([flag, str(value)])


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def _append_multi_option(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value if str(item).strip()]
    else:
        parts = [part.strip() for part in str(value).replace(",", " ").split() if part.strip()]
    if parts:
        command.append(flag)
        command.extend(parts)


def _append_bool(command: list[str], flag: str, value: Any) -> None:
    if bool(value):
        command.append(flag)


def _append_int(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(int(value))])


def _append_float(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(float(value))])


def _append_prompt(command: list[str], params: dict[str, Any]) -> None:
    command.extend(["--prompt", str(params.get("prompt", ""))])


def _append_generation_params(command: list[str], params: dict[str, Any], *, seed: Any) -> None:
    _append_float(command, "--duration", params.get("duration_seconds", params.get("duration")))
    _append_int(command, "--steps", params.get("steps"))
    _append_float(command, "--cfg-scale", params.get("cfg_scale"))
    _append_int(command, "--seed", seed)


def _redacted_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    sensitive_flags = {"--token", "--hf-token", "--huggingface-token"}
    for part in command:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        redacted.append(part)
        if part in sensitive_flags:
            skip_next = True
    return redacted
