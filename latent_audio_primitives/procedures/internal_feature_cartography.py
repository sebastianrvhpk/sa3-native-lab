"""Executable SA3 internal feature cartography procedures for notebook cells."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_audio_primitives.adapters.sa3_internal_hooks import (
    CFGAPGInfluenceRecorder,
    ResidualActivationPatcher,
    SA3InternalActivationCollector,
    memory_token_parameter_rows,
)
from latent_audio_primitives.adapters.sa3_residual_hooks import ActivationCollector
from latent_audio_primitives.internal_features import (
    ActivationPatchSpec,
    CFGAPGInfluenceRow,
    InternalActivationRow,
    SparseFeatureScaffoldRow,
    activation_rows_to_table,
    cfg_apg_rows_to_table,
    default_sa3_internal_surfaces,
    internal_surface_table,
    patch_specs_to_table,
    sparse_feature_scaffold_rows,
    sparse_feature_rows_to_table,
)
from latent_audio_primitives.trajectory import sampler_timestep_recorder


@dataclass(slots=True)
class InternalSurfaceCaptureResult:
    """Captured SA3 internal surface rows and provenance."""

    surface_rows: list[InternalActivationRow] = field(default_factory=list)
    memory_rows: list[dict[str, Any]] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)
    unsupported_rows: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "internal_surface_rows.json", activation_rows_to_table(self.surface_rows))
        _write_json(directory / "memory_token_rows.json", self.memory_rows)
        _write_json(directory / "sampler_step_records.json", self.step_records)
        _write_json(directory / "unsupported_surface_rows.json", self.unsupported_rows)
        _write_json(directory / "metadata.json", self.metadata)
        return directory


@dataclass(slots=True)
class CFGAPGAtlasResult:
    """SA3 CFG/APG condition-influence rows and provenance."""

    rows: list[CFGAPGInfluenceRow] = field(default_factory=list)
    step_records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "cfg_apg_influence_rows.json", cfg_apg_rows_to_table(self.rows))
        _write_json(directory / "sampler_step_records.json", self.step_records)
        _write_json(directory / "metadata.json", self.metadata)
        return directory


@dataclass(slots=True)
class SparseFeatureScaffoldResult:
    """Selected internal surfaces for later sparse-feature/SAE training."""

    rows: list[SparseFeatureScaffoldRow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "sparse_feature_scaffold_rows.json", sparse_feature_rows_to_table(self.rows))
        _write_json(directory / "metadata.json", self.metadata)
        return directory


@dataclass(slots=True)
class ResidualPatchRun:
    """One clean/corrupt residual patch run."""

    alpha: float
    patch_specs: list[ActivationPatchSpec]
    latents: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResidualPatchSweepResult:
    """Clean/corrupt post-block residual patch sweep outputs."""

    runs: list[ResidualPatchRun] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save_metadata(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "runs": [
                {
                    "alpha": run.alpha,
                    "patch_specs": patch_specs_to_table(run.patch_specs),
                    "metadata": run.metadata,
                }
                for run in self.runs
            ],
        }
        _write_json(directory / "residual_patch_sweep_metadata.json", payload)
        return directory


class SA3InternalFeatureCartographer:
    """Run SA3 internal feature cartography notebook procedures."""

    def __init__(
        self,
        model: Any,
        *,
        layer_indices: list[int] | None = None,
        cpu_offload: bool = True,
    ) -> None:
        self.model = model
        self.layer_indices = layer_indices
        self.cpu_offload = cpu_offload

    def surface_inventory(self) -> list[dict[str, Any]]:
        """Return the source-grounded internal surface inventory."""

        return internal_surface_table(default_sa3_internal_surfaces())

    def capture_surfaces(
        self,
        *,
        prompt: str,
        duration: float = 8.0,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        surfaces: Sequence[str] | None = None,
        sampler_type: str | None = None,
        record_timesteps: bool = True,
        return_latents: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> InternalSurfaceCaptureResult:
        """Run one generation and summarize selected SA3 internal surfaces."""

        torch = _require_torch()
        generate_kwargs = dict(generate_kwargs or {})
        step_records: list[dict[str, Any]] = []
        with SA3InternalActivationCollector(
            self.model,
            layer_indices=self.layer_indices,
            surfaces=surfaces,
            cpu_offload=self.cpu_offload,
        ) as collector:
            if sampler_type is not None:
                generate_kwargs.setdefault("sampler_type", sampler_type)
            if record_timesteps:
                user_callback = generate_kwargs.get("callback")
                generate_kwargs["callback"] = sampler_timestep_recorder(
                    step_records,
                    user_callback=user_callback,
                    sampler_type=sampler_type,
                )
            torch.manual_seed(seed)
            self.model.generate(
                prompt=prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                return_latents=return_latents,
                **generate_kwargs,
            )
            mapping_status = _mapping_status_from_records(step_records)
            surface_rows = collector.get_summary_rows(mapping_status=mapping_status)
            unsupported_rows = collector.unsupported
        return InternalSurfaceCaptureResult(
            surface_rows=surface_rows,
            memory_rows=memory_token_parameter_rows(self.model),
            step_records=step_records,
            unsupported_rows=unsupported_rows,
            metadata={
                "prompt": prompt,
                "duration": float(duration),
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "seed": int(seed),
                "layer_indices": self.layer_indices,
                "surfaces": list(surfaces) if surfaces is not None else None,
                "sampler_type": sampler_type,
                "record_timesteps": bool(record_timesteps),
                "mapping_status": mapping_status,
                "maturity": "microscope",
                "claim": "SA3-native internal surface visibility, not causal control",
            },
        )

    def cfg_apg_atlas(
        self,
        *,
        prompt: str,
        duration: float = 8.0,
        steps: int = 8,
        cfg_scale: float = 4.0,
        apg_scale: float = 1.0,
        seed: int = 42,
        sampler_type: str | None = None,
        return_latents: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> CFGAPGAtlasResult:
        """Record timestep-wise CFG/APG prompt influence.

        This requires CFG to be active. If ``cfg_scale`` is 1.0, upstream SA3
        does not enter the CFG branch and no rows are expected.
        """

        torch = _require_torch()
        generate_kwargs = dict(generate_kwargs or {})
        generate_kwargs.setdefault("apg_scale", apg_scale)
        if sampler_type is not None:
            generate_kwargs.setdefault("sampler_type", sampler_type)
        step_records: list[dict[str, Any]] = []
        user_callback = generate_kwargs.get("callback")
        generate_kwargs["callback"] = sampler_timestep_recorder(
            step_records,
            user_callback=user_callback,
            sampler_type=sampler_type,
        )
        with CFGAPGInfluenceRecorder(self.model) as recorder:
            torch.manual_seed(seed)
            self.model.generate(
                prompt=prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                return_latents=return_latents,
                **generate_kwargs,
            )
            rows = recorder.rows(step_records=step_records, sampler_type=sampler_type or "")
        return CFGAPGAtlasResult(
            rows=rows,
            step_records=step_records,
            metadata={
                "prompt": prompt,
                "duration": float(duration),
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "apg_scale": float(apg_scale),
                "seed": int(seed),
                "sampler_type": sampler_type,
                "row_count": int(len(rows)),
                "maturity": "microscope",
                "claim": "SA3-native condition influence over sampler time",
            },
        )

    def sparse_feature_scaffold(
        self,
        activation_rows: Sequence[InternalActivationRow | Mapping[str, Any]],
        *,
        top_k: int = 8,
        sample_source: str = "internal_feature_cartography",
    ) -> SparseFeatureScaffoldResult:
        """Select target surfaces for later SAE training without training yet."""

        rows = sparse_feature_scaffold_rows(
            activation_rows,
            top_k=top_k,
            sample_source=sample_source,
        )
        return SparseFeatureScaffoldResult(
            rows=rows,
            metadata={
                "top_k": int(top_k),
                "sample_source": sample_source,
                "maturity": "microscope",
                "claim": "SAE target selection scaffold, not sparse feature evidence yet",
            },
        )

    def residual_patch_sweep(
        self,
        *,
        clean_prompt: str,
        corrupt_prompt: str,
        patch_specs: Sequence[ActivationPatchSpec | Mapping[str, Any]],
        alphas: Sequence[float],
        duration: float = 8.0,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        return_latents: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> ResidualPatchSweepResult:
        """Run clean/corrupt post-block residual activation patching.

        This is a causal test for selected post-block residual coordinates. It
        does not patch branch internals.
        """

        torch = _require_torch()
        specs = [_coerce_patch_spec(spec) for spec in patch_specs]
        target_layers = sorted({spec.layer_index for spec in specs})
        generate_kwargs = dict(generate_kwargs or {})
        with ActivationCollector(
            self.model,
            layer_indices=target_layers,
            cpu_offload=self.cpu_offload,
        ) as collector:
            torch.manual_seed(seed)
            self.model.generate(
                prompt=clean_prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                return_latents=return_latents,
                **generate_kwargs,
            )
            clean_cache = collector.get_raw_activations()

        runs = []
        for alpha in alphas:
            alpha_specs = [
                ActivationPatchSpec(
                    layer_index=spec.layer_index,
                    call_start=spec.call_start,
                    call_end=spec.call_end,
                    step_index=spec.step_index,
                    calls_per_step=spec.calls_per_step,
                    alpha=float(alpha),
                    mode=spec.mode,
                    source=spec.source,
                    maturity=spec.maturity,
                    note=spec.note,
                )
                for spec in specs
            ]
            patcher = ResidualActivationPatcher(
                self.model,
                clean_activations=clean_cache,
                patch_specs=alpha_specs,
            )
            with patcher.patch():
                torch.manual_seed(seed)
                latents = self.model.generate(
                    prompt=corrupt_prompt,
                    duration=duration,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    return_latents=return_latents,
                    **generate_kwargs,
                )
            runs.append(
                ResidualPatchRun(
                    alpha=float(alpha),
                    patch_specs=alpha_specs,
                    latents=latents,
                    metadata={
                        "clean_prompt": clean_prompt,
                        "corrupt_prompt": corrupt_prompt,
                        "duration": float(duration),
                        "steps": int(steps),
                        "cfg_scale": float(cfg_scale),
                        "seed": int(seed),
                    },
                )
            )
        return ResidualPatchSweepResult(
            runs=runs,
            metadata={
                "clean_prompt": clean_prompt,
                "corrupt_prompt": corrupt_prompt,
                "duration": float(duration),
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "seed": int(seed),
                "patch_specs": patch_specs_to_table(specs),
                "maturity": "intervention_candidate",
                "claim": "causal post-block residual patch test pending audio evidence",
            },
        )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _mapping_status_from_records(records: Sequence[Mapping[str, Any]]) -> str:
    return "sampler_records_present" if records else "no_sampler_records"


def _coerce_patch_spec(spec: ActivationPatchSpec | Mapping[str, Any]) -> ActivationPatchSpec:
    if isinstance(spec, ActivationPatchSpec):
        return spec
    return ActivationPatchSpec(
        layer_index=int(spec["layer_index"]),
        call_start=None if spec.get("call_start") is None else int(spec["call_start"]),
        call_end=None if spec.get("call_end") is None else int(spec["call_end"]),
        step_index=None if spec.get("step_index") is None else int(spec["step_index"]),
        calls_per_step=None if spec.get("calls_per_step") is None else int(spec["calls_per_step"]),
        alpha=float(spec.get("alpha", 1.0)),
        mode=str(spec.get("mode", "blend")),
        source=str(spec.get("source", "mapping")),
        maturity=str(spec.get("maturity", "intervention_candidate")),
        note=str(spec.get("note", "")),
    )


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for internal feature cartography.") from exc
    return torch
