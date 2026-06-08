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
    control_lane_layer_probe_rows,
    control_lane_probe_table,
    control_lane_window_probe_rows,
)


@dataclass(slots=True)
class ControlLaneMechanisticProbeResult:
    """Lane-probe rows and provenance for one SA3 audio-conditioned run."""

    layer_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    window_rows: list[ControlLaneProbeRow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "control_lane_layer_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.layer_rows), f, indent=2, sort_keys=True)
        with (directory / "control_lane_window_probe_rows.json").open("w", encoding="utf-8") as f:
            json.dump(control_lane_probe_table(self.window_rows), f, indent=2, sort_keys=True)
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
        with ActivationCollector(
            self.model,
            layer_indices=self.layer_indices,
            cpu_offload=self.cpu_offload,
        ) as collector:
            collector.clear()
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
                **generate_kwargs,
            )
            raw_activations = collector.get_raw_activations()
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
        return ControlLaneMechanisticProbeResult(
            layer_rows=layer_rows,
            window_rows=window_rows,
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
                "cv_folds": int(cv_folds),
                "ridge_alpha": float(ridge_alpha),
                "min_samples": int(min_samples),
                "min_confidence": float(min_confidence),
                "active_percentile": float(active_percentile),
                "quiet_percentile": float(quiet_percentile),
                "alignment_note": (
                    "Lane targets are resampled to residual token count inside each observed forward call. "
                    "Window rows group observed hook calls and are not exact sampler-timestep attribution."
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
