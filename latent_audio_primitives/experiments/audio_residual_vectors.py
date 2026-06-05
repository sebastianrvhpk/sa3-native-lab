"""Audio-set residual vector extraction for SA3 steering probes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from latent_audio_primitives.adapters.audioscope_sa3 import ActivationCollector, SteeringVectors
from latent_audio_primitives.experiments.activation_vectors import (
    ActivationExample,
    probe_layer_accuracy,
    vectors_from_examples,
)


@dataclass(frozen=True, slots=True)
class AudioExample:
    """One audio file assigned to a positive or reference residual set."""

    path: Path
    label: int
    group: str


@dataclass(slots=True)
class AudioResidualExtractionResult:
    """Residual vectors plus provenance for an audio-set extraction run."""

    vectors: SteeringVectors
    examples: list[ActivationExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.vectors.save(directory / "residual_audio_vectors.pt")
        with (directory / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": self.metadata,
                    "vectors": {
                        "layers": sorted(self.vectors.vectors),
                        "probe_accuracy": self.vectors.probe_accuracy,
                        "best_layer": self.vectors.best_layer,
                    },
                    "examples": [
                        {
                            "label": example.label,
                            "prompt": example.prompt,
                            "pair_index": example.pair_index,
                            "axis": example.axis,
                            "family": example.family,
                        }
                        for example in self.examples
                    ],
                },
                f,
                indent=2,
                sort_keys=True,
            )
        return directory


class SA3AudioResidualVectorExtractor:
    """Extract SA3 residual directions from positive/negative audio files.

    This uses audio-to-audio generation as the activation-producing path:

    ``model.generate(prompt=..., init_audio=(sr, audio), init_noise_level=..., return_latents=True)``

    The resulting direction lives in SA3 residual-stream space, not SAME latent
    space. It can be used with ``ResidualSteerer`` during later generations.
    """

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
        positive_paths: list[str | Path],
        negative_paths: list[str | Path] | None = None,
        *,
        prompt: str = "audio texture",
        duration: float | None = None,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        init_noise_level: float = 0.35,
        baseline_mode: str = "prompt",
        generate_kwargs: dict[str, Any] | None = None,
    ) -> list[ActivationExample]:
        torch = _require_torch()
        torchaudio = _require_torchaudio()
        generate_kwargs = generate_kwargs or {}
        negative_paths = negative_paths or []
        if baseline_mode not in {"negative_audio", "prompt"}:
            raise ValueError("baseline_mode must be 'negative_audio' or 'prompt'")
        if not positive_paths:
            raise ValueError("at least one positive audio path is required")
        if baseline_mode == "negative_audio" and not negative_paths:
            raise ValueError("negative_paths are required when baseline_mode='negative_audio'")
        audio_examples = [AudioExample(Path(path), 1, "positive") for path in positive_paths]
        if baseline_mode == "negative_audio":
            audio_examples.extend(AudioExample(Path(path), 0, "negative") for path in negative_paths)

        examples: list[ActivationExample] = []
        with ActivationCollector(
            self.model,
            layer_indices=self.layer_indices,
            cpu_offload=self.cpu_offload,
        ) as collector:
            for index, audio_example in enumerate(audio_examples):
                collector.clear()
                audio, sample_rate = torchaudio.load(audio_example.path)
                audio_duration = audio.shape[-1] / sample_rate
                run_duration = duration if duration is not None else audio_duration
                torch.manual_seed(seed + index)
                self.model.generate(
                    prompt=prompt,
                    duration=run_duration,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed + index,
                    init_audio=(sample_rate, audio),
                    init_noise_level=init_noise_level,
                    return_latents=True,
                    **generate_kwargs,
                )
                activations = collector.get_mean_activations()
                examples.append(
                    ActivationExample(
                        layer_activations=activations,
                        label=audio_example.label,
                        prompt=f"{audio_example.group}:{audio_example.path}",
                        pair_index=index,
                        axis="audio_residual",
                        family=audio_example.group,
                    )
                )
                if baseline_mode == "prompt" and audio_example.label == 1:
                    collector.clear()
                    torch.manual_seed(seed + index)
                    self.model.generate(
                        prompt=prompt,
                        duration=run_duration,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        seed=seed + index,
                        return_latents=True,
                        **generate_kwargs,
                    )
                    baseline_activations = collector.get_mean_activations()
                    examples.append(
                        ActivationExample(
                            layer_activations=baseline_activations,
                            label=0,
                            prompt=f"prompt_baseline:{audio_example.path}",
                            pair_index=index,
                            axis="audio_residual",
                            family="prompt_baseline",
                        )
                    )
        return examples

    def extract(
        self,
        positive_paths: list[str | Path],
        negative_paths: list[str | Path] | None = None,
        *,
        prompt: str = "audio texture",
        duration: float | None = None,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = 42,
        init_noise_level: float = 0.35,
        baseline_mode: str = "prompt",
        normalize: bool = True,
        probe: bool = True,
        generate_kwargs: dict[str, Any] | None = None,
    ) -> AudioResidualExtractionResult:
        examples = self.collect_examples(
            positive_paths,
            negative_paths,
            prompt=prompt,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            init_noise_level=init_noise_level,
            baseline_mode=baseline_mode,
            generate_kwargs=generate_kwargs,
        )
        vectors = vectors_from_examples(examples, normalize=normalize)
        if probe:
            vectors.probe_accuracy = probe_layer_accuracy(examples)
            if vectors.probe_accuracy:
                vectors.best_layer = max(vectors.probe_accuracy, key=vectors.probe_accuracy.get)
        return AudioResidualExtractionResult(
            vectors=vectors,
            examples=examples,
            metadata={
                "prompt": prompt,
                "duration": duration,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "init_noise_level": init_noise_level,
                "baseline_mode": baseline_mode,
                "positive_paths": [str(path) for path in positive_paths],
                "negative_paths": [str(path) for path in (negative_paths or [])],
            },
        )


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for audio residual extraction.") from exc
    return torch


def _require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required for audio residual extraction.") from exc
    return torchaudio
