"""Prompt-pair residual activation vector extraction for SA3 steering probes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from latent_audio_primitives.adapters.sa3_residual_hooks import (
    ActivationCollector,
    SteeringVectors,
    mean_difference_vectors,
)
from latent_audio_primitives.prompt_pairs import DEFAULT_PROMPT_PAIRS, PromptPair


@dataclass(slots=True)
class ActivationExample:
    """One generated prompt example and its mean residual activations."""

    layer_activations: dict[int, Any]
    label: int
    prompt: str
    pair_index: int
    axis: str
    family: str


@dataclass(slots=True)
class VectorExtractionResult:
    """Steering vectors plus provenance for a prompt-pair extraction run."""

    vectors: SteeringVectors
    examples: list[ActivationExample] = field(default_factory=list)
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
                    torch.manual_seed(seed + pair_index)
                    self.model.generate(
                        prompt=prompt,
                        duration=duration,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        seed=seed + pair_index,
                        return_latents=return_latents,
                        **generate_kwargs,
                    )
                    activations = collector.get_mean_activations()
                    examples.append(
                        ActivationExample(
                            layer_activations=activations,
                            label=label,
                            prompt=prompt,
                            pair_index=pair_index,
                            axis=pair.axis,
                            family=pair.family,
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
        generate_kwargs: dict[str, Any] | None = None,
    ) -> VectorExtractionResult:
        examples = self.collect_examples(
            pairs,
            num_pairs=num_pairs,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            generate_kwargs=generate_kwargs,
        )
        vectors = vectors_from_examples(examples, normalize=normalize)
        if probe:
            vectors.probe_accuracy = probe_layer_accuracy(examples)
            if vectors.probe_accuracy:
                vectors.best_layer = max(vectors.probe_accuracy, key=vectors.probe_accuracy.get)
        metadata = {
            "pairs": [asdict(pair) for pair in (pairs if pairs is not None else DEFAULT_PROMPT_PAIRS)[: num_pairs or None]],
            "duration": duration,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "normalize": normalize,
        }
        return VectorExtractionResult(vectors=vectors, examples=examples, metadata=metadata)


def vectors_from_examples(examples: list[ActivationExample], *, normalize: bool = True) -> SteeringVectors:
    """Collapse labeled activation examples into mean-difference vectors."""

    torch = _require_torch()
    if not examples:
        raise ValueError("at least one activation example is required")

    positive: dict[int, list[Any]] = {}
    negative: dict[int, list[Any]] = {}
    for example in examples:
        target = positive if example.label == 1 else negative
        for layer_idx, activation in example.layer_activations.items():
            target.setdefault(layer_idx, []).append(activation)

    pos_mean = {layer: torch.stack(values).mean(dim=0) for layer, values in positive.items()}
    neg_mean = {layer: torch.stack(values).mean(dim=0) for layer, values in negative.items()}
    return mean_difference_vectors(pos_mean, neg_mean, normalize=normalize)


def probe_layer_accuracy(examples: list[ActivationExample]) -> dict[int, float]:
    """Small dependency-free centroid probe for quick layer ranking.

    This is not a replacement for a proper held-out linear probe. It is a cheap
    Colab-friendly sanity check: for each layer, classify examples by which
    class centroid is closer in leave-one-out evaluation.
    """

    import numpy as np

    by_layer: dict[int, list[tuple[Any, int]]] = {}
    for example in examples:
        for layer_idx, activation in example.layer_activations.items():
            by_layer.setdefault(layer_idx, []).append((activation.float().cpu().numpy(), example.label))

    scores: dict[int, float] = {}
    for layer_idx, values in by_layer.items():
        if len(values) < 4:
            scores[layer_idx] = 0.0
            continue
        correct = 0
        total = 0
        for holdout_idx, (x, label) in enumerate(values):
            train_pos = [v for i, (v, y) in enumerate(values) if i != holdout_idx and y == 1]
            train_neg = [v for i, (v, y) in enumerate(values) if i != holdout_idx and y == 0]
            if not train_pos or not train_neg:
                continue
            pos_centroid = np.stack(train_pos).mean(axis=0)
            neg_centroid = np.stack(train_neg).mean(axis=0)
            pred = int(np.linalg.norm(x - pos_centroid) < np.linalg.norm(x - neg_centroid))
            correct += int(pred == label)
            total += 1
        scores[layer_idx] = correct / total if total else 0.0
    return scores


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for activation-vector extraction.") from exc
    return torch
