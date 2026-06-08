"""SA3 execution wrapper for control-lane mechanistic probes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from latent_audio_primitives.adapters.sa3_residual_hooks import ActivationCollector
from latent_audio_primitives.control_lanes import ControlLane
from latent_audio_primitives.control_lane_probes import (
    ControlLaneProbeRow,
    control_lane_active_direction_table,
    control_lane_layer_probe_rows,
    control_lane_null_layer_probe_rows,
    control_lane_null_timestep_probe_rows,
    control_lane_null_window_probe_rows,
    control_lane_probe_prediction_table,
    control_lane_probe_table,
    control_lane_probe_top_table,
    control_lane_timestep_probe_rows,
    control_lane_window_probe_rows,
)
from latent_audio_primitives.trajectory import sampler_timestep_recorder


@dataclass(slots=True)
class ControlLaneMechanisticProbeResult:
    """Lane-probe rows and provenance for one SA3 audio-conditioned run."""

    layer_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    window_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    timestep_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    null_layer_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    null_window_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    null_timestep_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    prediction_rows: list[dict[str, Any]] = field(default_factory=list)
    active_direction_rows: list[dict[str, Any]] = field(default_factory=list)
    top_rows: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "control_lane_layer_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.layer_rows), f, indent=2, sort_keys=True)
        with (directory / "control_lane_window_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.window_rows), f, indent=2, sort_keys=True)
        with (directory / "control_lane_timestep_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.timestep_rows), f, indent=2, sort_keys=True)
        with (directory / "control_lane_null_layer_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.null_layer_rows), f, indent=2, sort_keys=True)
        with (directory / "control_lane_null_window_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.null_window_rows), f, indent=2, sort_keys=True)
        with (directory / "control_lane_null_timestep_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.null_timestep_rows), f, indent=2, sort_keys=True)
        with (directory / "control_lane_prediction_rows.json").open("w", encoding="utf-8") as f:
            json.dump(self.prediction_rows, f, indent=2, sort_keys=True)
        with (directory / "control_lane_active_direction_rows.json").open("w", encoding="utf-8") as f:
            json.dump(self.active_direction_rows, f, indent=2, sort_keys=True)
        with (directory / "control_lane_top_rows.json").open("w", encoding="utf-8") as f:
            json.dump(self.top_rows, f, indent=2, sort_keys=True)
        with (directory / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)
        return directory


class SA3ControlLaneProbeExtractor:
    """Capture SA3 residual activations and probe them against control lanes."""

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

    def probe_audio_path(
        self,
        path: str | Path,
        *,
        lanes: Sequence[ControlLane],
        lane_names: Sequence[str] | None = None,
        prompt: str = "audio texture",
        duration: float | None = None,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        init_noise_level: float = 0.35,
        trajectory_probe: bool = True,
        trajectory_window_count: int | None = 5,
        trajectory_window_size: int | None = None,
        timestep_probe: bool = False,
        sampler_type: str | None = None,
        null_probe: bool = False,
        null_kinds: Sequence[str] = ("shuffle", "reverse", "random"),
        null_seed: int = 0,
        prediction_probe: bool = True,
        prediction_top_k_per_lane: int = 1,
        prediction_max_points_per_row: int = 240,
        active_direction_preview: bool = True,
        direction_top_features: int = 16,
        cv_folds: int = 5,
        ridge_alpha: float = 1.0,
        min_samples: int = 16,
        min_confidence: float = 0.0,
        active_percentile: float = 85.0,
        quiet_percentile: float = 15.0,
        return_latents: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> ControlLaneMechanisticProbeResult:
        """Run audio-to-audio SA3 once and return lane/layer probe rows."""

        torch = _require_torch()
        torchaudio = _require_torchaudio()
        path = Path(path)
        generate_kwargs = generate_kwargs or {}
        audio, sample_rate = torchaudio.load(path)
        audio_duration = audio.shape[-1] / sample_rate
        run_duration = duration if duration is not None else audio_duration
        step_records: list[dict[str, Any]] = []
        timestep_activations: dict[int, dict[int, Any]] = {}
        timestep_metadata: dict[int, dict[int, dict[str, Any]]] = {}
        with ActivationCollector(
            self.model,
            layer_indices=self.layer_indices,
            cpu_offload=self.cpu_offload,
        ) as collector:
            collector.clear()
            run_kwargs = dict(generate_kwargs)
            if sampler_type is not None:
                run_kwargs.setdefault("sampler_type", sampler_type)
            if timestep_probe:
                user_callback = run_kwargs.get("callback")
                run_kwargs["callback"] = sampler_timestep_recorder(
                    step_records,
                    user_callback=user_callback,
                    sampler_type=sampler_type,
                )
            torch.manual_seed(seed)
            self.model.generate(
                prompt=prompt,
                duration=run_duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                init_audio=(sample_rate, audio),
                init_noise_level=init_noise_level,
                return_latents=return_latents,
                **run_kwargs,
            )
            raw_activations = collector.get_raw_activations()
            if timestep_probe and step_records:
                timestep_activations, timestep_metadata = collector.get_timestep_mean_activations(step_records)
        layer_rows = control_lane_layer_probe_rows(
            raw_activations,
            lanes,
            lane_names=lane_names,
            cv_folds=cv_folds,
            ridge_alpha=ridge_alpha,
            min_samples=min_samples,
            min_confidence=min_confidence,
            active_percentile=active_percentile,
            quiet_percentile=quiet_percentile,
        )
        window_rows: list[ControlLaneProbeRow] = []
        if trajectory_probe:
            window_rows = control_lane_window_probe_rows(
                raw_activations,
                lanes,
                lane_names=lane_names,
                window_count=trajectory_window_count,
                window_size=trajectory_window_size,
                cv_folds=cv_folds,
                ridge_alpha=ridge_alpha,
                min_samples=min_samples,
                min_confidence=min_confidence,
                active_percentile=active_percentile,
                quiet_percentile=quiet_percentile,
            )
        timestep_rows: list[ControlLaneProbeRow] = []
        if timestep_probe and timestep_activations:
            timestep_rows = control_lane_timestep_probe_rows(
                timestep_activations,
                lanes,
                layer_timestep_metadata=timestep_metadata,
                lane_names=lane_names,
                cv_folds=cv_folds,
                ridge_alpha=ridge_alpha,
                min_samples=min_samples,
                min_confidence=min_confidence,
                active_percentile=active_percentile,
                quiet_percentile=quiet_percentile,
            )
        null_layer_rows: list[ControlLaneProbeRow] = []
        null_window_rows: list[ControlLaneProbeRow] = []
        null_timestep_rows: list[ControlLaneProbeRow] = []
        if null_probe:
            null_layer_rows = control_lane_null_layer_probe_rows(
                raw_activations,
                lanes,
                null_kinds=null_kinds,
                seed=null_seed,
                lane_names=lane_names,
                cv_folds=cv_folds,
                ridge_alpha=ridge_alpha,
                min_samples=min_samples,
                min_confidence=min_confidence,
                active_percentile=active_percentile,
                quiet_percentile=quiet_percentile,
            )
            if trajectory_probe:
                null_window_rows = control_lane_null_window_probe_rows(
                    raw_activations,
                    lanes,
                    null_kinds=null_kinds,
                    seed=null_seed + 1,
                    lane_names=lane_names,
                    window_count=trajectory_window_count,
                    window_size=trajectory_window_size,
                    cv_folds=cv_folds,
                    ridge_alpha=ridge_alpha,
                    min_samples=min_samples,
                    min_confidence=min_confidence,
                    active_percentile=active_percentile,
                    quiet_percentile=quiet_percentile,
                )
            if timestep_probe and timestep_activations:
                null_timestep_rows = control_lane_null_timestep_probe_rows(
                    timestep_activations,
                    lanes,
                    layer_timestep_metadata=timestep_metadata,
                    null_kinds=null_kinds,
                    seed=null_seed + 2,
                    lane_names=lane_names,
                    cv_folds=cv_folds,
                    ridge_alpha=ridge_alpha,
                    min_samples=min_samples,
                    min_confidence=min_confidence,
                    active_percentile=active_percentile,
                    quiet_percentile=quiet_percentile,
                )
        prediction_rows: list[dict[str, Any]] = []
        if prediction_probe:
            prediction_rows = control_lane_probe_prediction_table(
                raw_activations,
                lanes,
                window_rows or layer_rows,
                top_k_per_lane=prediction_top_k_per_lane,
                max_points_per_row=prediction_max_points_per_row,
                ridge_alpha=ridge_alpha,
                cv_folds=cv_folds,
                min_confidence=min_confidence,
            )
        active_direction_rows: list[dict[str, Any]] = []
        if active_direction_preview:
            active_direction_rows = control_lane_active_direction_table(
                raw_activations,
                lanes,
                window_rows or layer_rows,
                top_k_per_lane=prediction_top_k_per_lane,
                top_features=direction_top_features,
                ridge_alpha=ridge_alpha,
                min_confidence=min_confidence,
                active_percentile=active_percentile,
                quiet_percentile=quiet_percentile,
            )
        top_rows = control_lane_probe_top_table(layer_rows, top_k_per_lane=1)
        if window_rows:
            top_rows.extend(control_lane_probe_top_table(window_rows, top_k_per_lane=1))
        if timestep_rows:
            top_rows.extend(control_lane_probe_top_table(timestep_rows, top_k_per_lane=1))
        return ControlLaneMechanisticProbeResult(
            layer_rows=layer_rows,
            window_rows=window_rows,
            timestep_rows=timestep_rows,
            null_layer_rows=null_layer_rows,
            null_window_rows=null_window_rows,
            null_timestep_rows=null_timestep_rows,
            prediction_rows=prediction_rows,
            active_direction_rows=active_direction_rows,
            top_rows=top_rows,
            metadata={
                "source_audio": str(path),
                "source_duration_seconds": float(audio_duration),
                "duration": None if duration is None else float(duration),
                "run_duration": float(run_duration),
                "prompt": prompt,
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
                "seed": int(seed),
                "init_noise_level": float(init_noise_level),
                "lane_names": None if lane_names is None else list(lane_names),
                "layer_indices": self.layer_indices,
                "trajectory_probe": bool(trajectory_probe),
                "trajectory_window_count": trajectory_window_count if trajectory_probe else None,
                "trajectory_window_size": trajectory_window_size if trajectory_probe else None,
                "timestep_probe": bool(timestep_probe),
                "sampler_type": sampler_type,
                "sampler_step_record_count": int(len(step_records)),
                "timestep_mapping_statuses": sorted(
                    {
                        str(row.mapping_status)
                        for row in timestep_rows
                        if row.mapping_status
                    }
                ),
                "null_probe": bool(null_probe),
                "null_kinds": list(null_kinds) if null_probe else [],
                "null_seed": int(null_seed) if null_probe else None,
                "prediction_probe": bool(prediction_probe),
                "prediction_top_k_per_lane": int(prediction_top_k_per_lane) if prediction_probe else None,
                "prediction_max_points_per_row": int(prediction_max_points_per_row) if prediction_probe else None,
                "active_direction_preview": bool(active_direction_preview),
                "direction_top_features": int(direction_top_features) if active_direction_preview else None,
                "cv_folds": int(cv_folds),
                "ridge_alpha": float(ridge_alpha),
                "min_samples": int(min_samples),
                "min_confidence": float(min_confidence),
                "active_percentile": float(active_percentile),
                "quiet_percentile": float(quiet_percentile),
                "alignment_note": (
                    "Lane targets are resampled to residual token count inside each observed forward call. "
                    "Window rows group observed hook calls. Timestep rows use sampler callback metadata when "
                    "available and report mapping_status for exact/grouped/approximate hook-call mapping."
                ),
            },
        )


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for control-lane mechanistic probes.") from exc
    return torch


def _require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for control-lane mechanistic probes.") from exc
    return torchaudio
