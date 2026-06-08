"""Prompt-pair residual activation vector extraction for SA3 trajectory probes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from latent_audio_primitives.adapters.sa3_residual_hooks import ActivationCollector
from latent_audio_primitives.prompt_pairs import DEFAULT_PROMPT_PAIRS, PromptPair
from latent_audio_primitives.residual_probes import (
    ActivationExample,
    LayerProbeRow,
    SteeringVectors,
    flatten_layer_windows,
    probe_layer_rows,
    probe_layer_timestep_rows,
    probe_layer_window_rows,
    vectors_from_examples,
)
from latent_audio_primitives.trajectory import sampler_timestep_recorder


@dataclass(slots=True)
class VectorExtractionResult:
    """Steering vectors plus provenance for a prompt-pair extraction run."""

    vectors: SteeringVectors
    examples: list[ActivationExample] = field(default_factory=list)
    layer_probe_rows: list[LayerProbeRow] = field(default_factory=list)
    trajectory_probe_rows: list[LayerProbeRow] = field(default_factory=list)
    timestep_probe_rows: list[LayerProbeRow] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vectors.save(directory / "steering_vectors.pt")
        examples_meta = [
            {
                "label": example.label,
                "prompt": example.prompt,
                "pair_index": example.pair_index,
                "axis": example.axis,
                "family": example.family,
            }
            for example in self.examples
        ]
        with (directory / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": self.metadata,
                    "examples": examples_meta,
                    "vectors": {
                        "layers": sorted(self.vectors.vectors),
                        "probe_accuracy": self.vectors.probe_accuracy,
                        "best_layer": self.vectors.best_layer,
                    },
                    "layer_probe": [row.to_dict() for row in self.layer_probe_rows],
                    "trajectory_probe": [row.to_dict() for row in self.trajectory_probe_rows],
                    "timestep_probe": [row.to_dict() for row in self.timestep_probe_rows],
                },
                f,
                indent=2,
                sort_keys=True,
            )
        return directory


class SA3ActivationVectorExtractor:
    """Extract audioscope-style residual directions from released SA3 models."""

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

    def collect_examples(
        self,
        pairs: list[PromptPair] | None = None,
        *,
        num_pairs: int | None = None,
        duration: float = 8.0,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        trajectory_window_count: int | None = None,
        trajectory_window_size: int | None = None,
        timestep_probe: bool = False,
        sampler_type: str | None = None,
        return_latents: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> list[ActivationExample]:
        torch = _require_torch()
        selected_pairs = list(pairs if pairs is not None else DEFAULT_PROMPT_PAIRS)
        if num_pairs is not None:
            selected_pairs = selected_pairs[:num_pairs]
        generate_kwargs = generate_kwargs or {}

        examples: list[ActivationExample] = []
        with ActivationCollector(
            self.model,
            layer_indices=self.layer_indices,
            cpu_offload=self.cpu_offload,
        ) as collector:
            for pair_index, pair in enumerate(selected_pairs):
                for label, prompt in [(1, pair.positive), (0, pair.negative)]:
                    collector.clear()
                    run_kwargs = dict(generate_kwargs)
                    if sampler_type is not None:
                        run_kwargs.setdefault("sampler_type", sampler_type)
                    step_records: list[dict[str, Any]] = []
                    if timestep_probe:
                        user_callback = run_kwargs.get("callback")
                        run_kwargs["callback"] = sampler_timestep_recorder(
                            step_records,
                            user_callback=user_callback,
                            sampler_type=sampler_type,
                        )
                    torch.manual_seed(seed + pair_index)
                    self.model.generate(
                        prompt=prompt,
                        duration=duration,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        seed=seed + pair_index,
                        return_latents=return_latents,
                        **run_kwargs,
                    )
                    activations = collector.get_mean_activations()
                    layer_window_activations: dict[tuple[int, int], Any] = {}
                    layer_window_metadata: dict[tuple[int, int], dict[str, Any]] = {}
                    layer_timestep_activations: dict[tuple[int, int], Any] = {}
                    layer_timestep_metadata: dict[tuple[int, int], dict[str, Any]] = {}
                    if trajectory_window_count is not None or trajectory_window_size is not None:
                        pooled_windows, window_metadata = collector.get_windowed_mean_activations(
                            window_count=trajectory_window_count,
                            window_size=trajectory_window_size,
                        )
                        layer_window_activations, layer_window_metadata = flatten_layer_windows(
                            pooled_windows,
                            window_metadata,
                        )
                    if timestep_probe:
                        pooled_timesteps, timestep_metadata = collector.get_timestep_mean_activations(step_records)
                        layer_timestep_activations, layer_timestep_metadata = flatten_layer_windows(
                            pooled_timesteps,
                            timestep_metadata,
                        )
                    examples.append(
                        ActivationExample(
                            layer_activations=activations,
                            label=label,
                            prompt=prompt,
                            pair_index=pair_index,
                            axis=pair.axis,
                            family=pair.family,
                            layer_window_activations=layer_window_activations,
                            layer_window_metadata=layer_window_metadata,
                            layer_timestep_activations=layer_timestep_activations,
                            layer_timestep_metadata=layer_timestep_metadata,
                        )
                    )
        return examples

    def extract(
        self,
        pairs: list[PromptPair] | None = None,
        *,
        num_pairs: int | None = None,
        duration: float = 8.0,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        normalize: bool = True,
        probe: bool = True,
        probe_method: str = "logistic_cv",
        probe_cv_folds: int = 5,
        probe_require_sklearn: bool = False,
        probe_random_state: int = 0,
        trajectory_probe: bool = False,
        trajectory_window_count: int | None = 5,
        trajectory_window_size: int | None = None,
        timestep_probe: bool = False,
        sampler_type: str | None = None,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> VectorExtractionResult:
        examples = self.collect_examples(
            pairs,
            num_pairs=num_pairs,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            trajectory_window_count=trajectory_window_count if trajectory_probe else None,
            trajectory_window_size=trajectory_window_size if trajectory_probe else None,
            timestep_probe=timestep_probe,
            sampler_type=sampler_type,
            generate_kwargs=generate_kwargs,
        )
        vectors = vectors_from_examples(examples, normalize=normalize)
        layer_probe_rows: list[LayerProbeRow] = []
        trajectory_probe_rows: list[LayerProbeRow] = []
        timestep_probe_rows: list[LayerProbeRow] = []
        if probe:
            layer_probe_rows = probe_layer_rows(
                examples,
                method=probe_method,
                cv_folds=probe_cv_folds,
                require_sklearn=probe_require_sklearn,
                random_state=probe_random_state,
            )
            vectors.probe_accuracy = {
                row.layer_index: row.accuracy_mean
                for row in layer_probe_rows
                if row.fold_count > 0
            }
            ranked_ok = [row for row in layer_probe_rows if row.fold_count > 0]
            if ranked_ok:
                vectors.best_layer = ranked_ok[0].layer_index
        if trajectory_probe:
            trajectory_probe_rows = probe_layer_window_rows(
                examples,
                method=probe_method,
                cv_folds=probe_cv_folds,
                require_sklearn=probe_require_sklearn,
                random_state=probe_random_state,
            )
        if timestep_probe:
            timestep_probe_rows = probe_layer_timestep_rows(
                examples,
                method=probe_method,
                cv_folds=probe_cv_folds,
                require_sklearn=probe_require_sklearn,
                random_state=probe_random_state,
            )
        metadata = {
            "pairs": [asdict(pair) for pair in (pairs if pairs is not None else DEFAULT_PROMPT_PAIRS)[: num_pairs or None]],
            "duration": duration,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "normalize": normalize,
            "probe": probe,
            "probe_method": probe_method if probe else None,
            "probe_cv_folds": probe_cv_folds if probe else None,
            "probe_require_sklearn": probe_require_sklearn if probe else None,
            "probe_random_state": probe_random_state if probe else None,
            "trajectory_probe": trajectory_probe,
            "trajectory_window_count": trajectory_window_count if trajectory_probe else None,
            "trajectory_window_size": trajectory_window_size if trajectory_probe else None,
            "timestep_probe": timestep_probe,
            "sampler_type": sampler_type,
        }
        return VectorExtractionResult(
            vectors=vectors,
            examples=examples,
            layer_probe_rows=layer_probe_rows,
            trajectory_probe_rows=trajectory_probe_rows,
            timestep_probe_rows=timestep_probe_rows,
            metadata=metadata,
        )


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for activation-vector extraction.") from exc
    return torch
