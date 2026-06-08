"""Prompt-pair residual activation vector extraction for SA3 trajectory probes."""

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
from latent_audio_primitives.trajectory import sampler_timestep_recorder


@dataclass(slots=True)
class ActivationExample:
    """One generated prompt example and its mean residual activations."""

    layer_activations: dict[int, Any]
    label: int
    prompt: str
    pair_index: int
    axis: str
    family: str
    layer_window_activations: dict[tuple[int, int], Any] = field(default_factory=dict)
    layer_window_metadata: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    layer_timestep_activations: dict[tuple[int, int], Any] = field(default_factory=dict)
    layer_timestep_metadata: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LayerProbeRow:
    """One layer or layer-window ranking row for residual activation probes."""

    layer_index: int
    method: str
    accuracy_mean: float
    accuracy_std: float
    fold_count: int
    sample_count: int
    positive_count: int
    negative_count: int
    rank: int = 0
    status: str = "ok"
    error: str = ""
    window_index: int | None = None
    window_label: str = ""
    window_start_fraction: float | None = None
    window_end_fraction: float | None = None
    call_start: int | None = None
    call_end: int | None = None
    call_count: int | None = None
    window_count: int | None = None
    step_index: int | None = None
    sampler_index: int | None = None
    timestep: float | None = None
    sigma: float | None = None
    logsnr: float | None = None
    sampler_type: str = ""
    calls_per_step: int | None = None
    mapping_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": int(self.layer_index),
            "window_index": None if self.window_index is None else int(self.window_index),
            "window_label": self.window_label,
            "window_start_fraction": (
                None if self.window_start_fraction is None else float(self.window_start_fraction)
            ),
            "window_end_fraction": None if self.window_end_fraction is None else float(self.window_end_fraction),
            "call_start": None if self.call_start is None else int(self.call_start),
            "call_end": None if self.call_end is None else int(self.call_end),
            "call_count": None if self.call_count is None else int(self.call_count),
            "window_count": None if self.window_count is None else int(self.window_count),
            "step_index": None if self.step_index is None else int(self.step_index),
            "sampler_index": None if self.sampler_index is None else int(self.sampler_index),
            "timestep": None if self.timestep is None else float(self.timestep),
            "sigma": None if self.sigma is None else float(self.sigma),
            "logsnr": None if self.logsnr is None else float(self.logsnr),
            "sampler_type": self.sampler_type,
            "calls_per_step": None if self.calls_per_step is None else int(self.calls_per_step),
            "mapping_status": self.mapping_status,
            "method": self.method,
            "accuracy_mean": float(self.accuracy_mean),
            "accuracy_std": float(self.accuracy_std),
            "fold_count": int(self.fold_count),
            "sample_count": int(self.sample_count),
            "positive_count": int(self.positive_count),
            "negative_count": int(self.negative_count),
            "rank": int(self.rank),
            "status": self.status,
            "error": self.error,
        }


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


def probe_layer_rows(
    examples: list[ActivationExample],
    *,
    method: str = "logistic_cv",
    cv_folds: int = 5,
    require_sklearn: bool = False,
    random_state: int = 0,
) -> list[LayerProbeRow]:
    """Rank residual layers with audioscope-grade or dependency-light probes.

    ``method="logistic_cv"`` mirrors audioscope's layer probe: fit a linear
    logistic classifier per layer and report stratified cross-validation
    accuracy. The implementation uses scikit-learn when available, but keeps a
    Torch solver because the Colab runtime intentionally removes sklearn to
    avoid optional-import conflicts with SA3/T5Gemma.
    """

    method_key = method.lower()
    if method_key in {"audioscope", "logistic", "linear_probe"}:
        method_key = "logistic_cv"
    if method_key in {"centroid", "centroid_loo", "loo"}:
        method_key = "centroid_loo"
    if method_key not in {"logistic_cv", "centroid_loo"}:
        raise ValueError("method must be 'logistic_cv' or 'centroid_loo'")

    by_layer = _activation_examples_by_layer(examples)
    rows: list[LayerProbeRow] = []
    for layer_idx, values in by_layer.items():
        x, y = _probe_matrix(values)
        positive_count = int((y == 1).sum())
        negative_count = int((y == 0).sum())
        if positive_count < 1 or negative_count < 1:
            rows.append(
                LayerProbeRow(
                    layer_index=layer_idx,
                    method=method_key,
                    accuracy_mean=0.0,
                    accuracy_std=0.0,
                    fold_count=0,
                    sample_count=int(y.shape[0]),
                    positive_count=positive_count,
                    negative_count=negative_count,
                    status="insufficient",
                    error="both positive and negative examples are required",
                )
            )
            continue

        if method_key == "centroid_loo":
            accuracy = _centroid_leave_one_out_accuracy(values)
            rows.append(
                LayerProbeRow(
                    layer_index=layer_idx,
                    method="centroid_loo",
                    accuracy_mean=accuracy,
                    accuracy_std=0.0,
                    fold_count=int(y.shape[0]),
                    sample_count=int(y.shape[0]),
                    positive_count=positive_count,
                    negative_count=negative_count,
                )
            )
            continue

        row = _logistic_cv_probe_row(
            layer_idx,
            x,
            y,
            cv_folds=cv_folds,
            require_sklearn=require_sklearn,
            random_state=random_state,
        )
        rows.append(row)

    rows.sort(key=lambda row: (-row.accuracy_mean, row.accuracy_std, row.layer_index))
    ranked = []
    for rank, row in enumerate(rows, start=1):
        ranked.append(_copy_probe_row(row, rank=rank))
    return ranked


def probe_layer_window_rows(
    examples: list[ActivationExample],
    *,
    method: str = "logistic_cv",
    cv_folds: int = 5,
    require_sklearn: bool = False,
    random_state: int = 0,
) -> list[LayerProbeRow]:
    """Rank residual layer-window cells with the same probes as whole layers.

    Window indices are observed forward-call windows from the SA3 block hooks.
    They are a trajectory microscope over the released runtime, not a guarantee
    of exact sampler timestep semantics.
    """

    method_key = method.lower()
    if method_key in {"audioscope", "logistic", "linear_probe"}:
        method_key = "logistic_cv"
    if method_key in {"centroid", "centroid_loo", "loo"}:
        method_key = "centroid_loo"
    if method_key not in {"logistic_cv", "centroid_loo"}:
        raise ValueError("method must be 'logistic_cv' or 'centroid_loo'")

    by_window, metadata_by_window = _activation_examples_by_layer_window(examples)
    rows: list[LayerProbeRow] = []
    for (layer_idx, window_idx), values in by_window.items():
        x, y = _probe_matrix(values)
        positive_count = int((y == 1).sum())
        negative_count = int((y == 0).sum())
        metadata = metadata_by_window.get((layer_idx, window_idx), {})
        if positive_count < 1 or negative_count < 1:
            rows.append(
                _with_window_metadata(
                    LayerProbeRow(
                        layer_index=layer_idx,
                        method=method_key,
                        accuracy_mean=0.0,
                        accuracy_std=0.0,
                        fold_count=0,
                        sample_count=int(y.shape[0]),
                        positive_count=positive_count,
                        negative_count=negative_count,
                        status="insufficient",
                        error="both positive and negative examples are required",
                    ),
                    window_idx,
                    metadata,
                )
            )
            continue

        if method_key == "centroid_loo":
            row = LayerProbeRow(
                layer_index=layer_idx,
                method="centroid_loo",
                accuracy_mean=_centroid_leave_one_out_accuracy(values),
                accuracy_std=0.0,
                fold_count=int(y.shape[0]),
                sample_count=int(y.shape[0]),
                positive_count=positive_count,
                negative_count=negative_count,
            )
        else:
            row = _logistic_cv_probe_row(
                layer_idx,
                x,
                y,
                cv_folds=cv_folds,
                require_sklearn=require_sklearn,
                random_state=random_state,
            )
        rows.append(_with_window_metadata(row, window_idx, metadata))

    rows.sort(
        key=lambda row: (
            -row.accuracy_mean,
            row.accuracy_std,
            row.layer_index,
            -1 if row.window_index is None else row.window_index,
        )
    )
    return [_copy_probe_row(row, rank=rank) for rank, row in enumerate(rows, start=1)]


def probe_layer_timestep_rows(
    examples: list[ActivationExample],
    *,
    method: str = "logistic_cv",
    cv_folds: int = 5,
    require_sklearn: bool = False,
    random_state: int = 0,
) -> list[LayerProbeRow]:
    """Rank residual layer/timestep cells from sampler callback records."""

    method_key = method.lower()
    if method_key in {"audioscope", "logistic", "linear_probe"}:
        method_key = "logistic_cv"
    if method_key in {"centroid", "centroid_loo", "loo"}:
        method_key = "centroid_loo"
    if method_key not in {"logistic_cv", "centroid_loo"}:
        raise ValueError("method must be 'logistic_cv' or 'centroid_loo'")

    by_timestep, metadata_by_timestep = _activation_examples_by_layer_timestep(examples)
    rows: list[LayerProbeRow] = []
    for (layer_idx, step_idx), values in by_timestep.items():
        x, y = _probe_matrix(values)
        positive_count = int((y == 1).sum())
        negative_count = int((y == 0).sum())
        metadata = metadata_by_timestep.get((layer_idx, step_idx), {})
        if positive_count < 1 or negative_count < 1:
            rows.append(
                _with_timestep_metadata(
                    LayerProbeRow(
                        layer_index=layer_idx,
                        method=method_key,
                        accuracy_mean=0.0,
                        accuracy_std=0.0,
                        fold_count=0,
                        sample_count=int(y.shape[0]),
                        positive_count=positive_count,
                        negative_count=negative_count,
                        status="insufficient",
                        error="both positive and negative examples are required",
                    ),
                    step_idx,
                    metadata,
                )
            )
            continue

        if method_key == "centroid_loo":
            row = LayerProbeRow(
                layer_index=layer_idx,
                method="centroid_loo",
                accuracy_mean=_centroid_leave_one_out_accuracy(values),
                accuracy_std=0.0,
                fold_count=int(y.shape[0]),
                sample_count=int(y.shape[0]),
                positive_count=positive_count,
                negative_count=negative_count,
            )
        else:
            row = _logistic_cv_probe_row(
                layer_idx,
                x,
                y,
                cv_folds=cv_folds,
                require_sklearn=require_sklearn,
                random_state=random_state,
            )
        rows.append(_with_timestep_metadata(row, step_idx, metadata))

    rows.sort(
        key=lambda row: (
            -row.accuracy_mean,
            row.accuracy_std,
            row.layer_index,
            -1 if row.step_index is None else row.step_index,
        )
    )
    return [_copy_probe_row(row, rank=rank) for rank, row in enumerate(rows, start=1)]


def probe_layer_accuracy(
    examples: list[ActivationExample],
    *,
    method: str = "logistic_cv",
    cv_folds: int = 5,
    require_sklearn: bool = False,
    random_state: int = 0,
) -> dict[int, float]:
    """Return a compact layer -> accuracy map for quick notebook display.

    Prefer ``probe_layer_rows`` when displaying or saving layer evidence.
    """

    return {
        row.layer_index: row.accuracy_mean
        for row in probe_layer_rows(
            examples,
            method=method,
            cv_folds=cv_folds,
            require_sklearn=require_sklearn,
            random_state=random_state,
        )
        if row.fold_count > 0
    }


def flatten_layer_windows(
    pooled_windows: dict[int, dict[int, Any]],
    window_metadata: dict[int, dict[int, dict[str, Any]]],
) -> tuple[dict[tuple[int, int], Any], dict[tuple[int, int], dict[str, Any]]]:
    activations: dict[tuple[int, int], Any] = {}
    metadata: dict[tuple[int, int], dict[str, Any]] = {}
    for layer_idx, windows in pooled_windows.items():
        for window_idx, activation in windows.items():
            key = (int(layer_idx), int(window_idx))
            activations[key] = activation
            metadata[key] = dict(window_metadata.get(layer_idx, {}).get(window_idx, {}))
    return activations, metadata


def _activation_examples_by_layer(examples: list[ActivationExample]) -> dict[int, list[tuple[Any, int]]]:
    by_layer: dict[int, list[tuple[Any, int]]] = {}
    for example in examples:
        for layer_idx, activation in example.layer_activations.items():
            by_layer.setdefault(layer_idx, []).append((activation.float().cpu().numpy(), int(example.label)))
    return by_layer


def _activation_examples_by_layer_window(
    examples: list[ActivationExample],
) -> tuple[dict[tuple[int, int], list[tuple[Any, int]]], dict[tuple[int, int], dict[str, Any]]]:
    by_window: dict[tuple[int, int], list[tuple[Any, int]]] = {}
    metadata_by_window: dict[tuple[int, int], dict[str, Any]] = {}
    for example in examples:
        for key, activation in example.layer_window_activations.items():
            layer_idx, window_idx = key
            normalized_key = (int(layer_idx), int(window_idx))
            by_window.setdefault(normalized_key, []).append((activation.float().cpu().numpy(), int(example.label)))
            if normalized_key not in metadata_by_window:
                metadata_by_window[normalized_key] = dict(example.layer_window_metadata.get(key, {}))
    return by_window, metadata_by_window


def _activation_examples_by_layer_timestep(
    examples: list[ActivationExample],
) -> tuple[dict[tuple[int, int], list[tuple[Any, int]]], dict[tuple[int, int], dict[str, Any]]]:
    by_timestep: dict[tuple[int, int], list[tuple[Any, int]]] = {}
    metadata_by_timestep: dict[tuple[int, int], dict[str, Any]] = {}
    for example in examples:
        for key, activation in example.layer_timestep_activations.items():
            layer_idx, step_idx = key
            normalized_key = (int(layer_idx), int(step_idx))
            by_timestep.setdefault(normalized_key, []).append((activation.float().cpu().numpy(), int(example.label)))
            if normalized_key not in metadata_by_timestep:
                metadata_by_timestep[normalized_key] = dict(example.layer_timestep_metadata.get(key, {}))
    return by_timestep, metadata_by_timestep


def _with_window_metadata(row: LayerProbeRow, window_idx: int, metadata: dict[str, Any]) -> LayerProbeRow:
    return LayerProbeRow(
        layer_index=row.layer_index,
        method=row.method,
        accuracy_mean=row.accuracy_mean,
        accuracy_std=row.accuracy_std,
        fold_count=row.fold_count,
        sample_count=row.sample_count,
        positive_count=row.positive_count,
        negative_count=row.negative_count,
        rank=row.rank,
        status=row.status,
        error=row.error,
        window_index=window_idx,
        window_label=str(metadata.get("window_label", "")),
        window_start_fraction=_optional_float(metadata.get("window_start_fraction")),
        window_end_fraction=_optional_float(metadata.get("window_end_fraction")),
        call_start=_optional_int(metadata.get("call_start")),
        call_end=_optional_int(metadata.get("call_end")),
        call_count=_optional_int(metadata.get("call_count")),
        window_count=_optional_int(metadata.get("window_count")),
    )


def _with_timestep_metadata(row: LayerProbeRow, step_idx: int, metadata: dict[str, Any]) -> LayerProbeRow:
    return LayerProbeRow(
        layer_index=row.layer_index,
        method=row.method,
        accuracy_mean=row.accuracy_mean,
        accuracy_std=row.accuracy_std,
        fold_count=row.fold_count,
        sample_count=row.sample_count,
        positive_count=row.positive_count,
        negative_count=row.negative_count,
        rank=row.rank,
        status=row.status,
        error=row.error,
        step_index=step_idx,
        sampler_index=_optional_int(metadata.get("sampler_index", metadata.get("i"))),
        timestep=_optional_float(metadata.get("timestep")),
        sigma=_optional_float(metadata.get("sigma")),
        logsnr=_optional_float(metadata.get("logsnr")),
        sampler_type=str(metadata.get("sampler_type", "")),
        call_start=_optional_int(metadata.get("call_start")),
        call_end=_optional_int(metadata.get("call_end")),
        call_count=_optional_int(metadata.get("call_count")),
        calls_per_step=_optional_int(metadata.get("calls_per_step")),
        mapping_status=str(metadata.get("mapping_status", "")),
    )


def _copy_probe_row(row: LayerProbeRow, *, rank: int) -> LayerProbeRow:
    return LayerProbeRow(
        layer_index=row.layer_index,
        method=row.method,
        accuracy_mean=row.accuracy_mean,
        accuracy_std=row.accuracy_std,
        fold_count=row.fold_count,
        sample_count=row.sample_count,
        positive_count=row.positive_count,
        negative_count=row.negative_count,
        rank=rank,
        status=row.status,
        error=row.error,
        window_index=row.window_index,
        window_label=row.window_label,
        window_start_fraction=row.window_start_fraction,
        window_end_fraction=row.window_end_fraction,
        call_start=row.call_start,
        call_end=row.call_end,
        call_count=row.call_count,
        window_count=row.window_count,
        step_index=row.step_index,
        sampler_index=row.sampler_index,
        timestep=row.timestep,
        sigma=row.sigma,
        logsnr=row.logsnr,
        sampler_type=row.sampler_type,
        calls_per_step=row.calls_per_step,
        mapping_status=row.mapping_status,
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _probe_matrix(values: list[tuple[Any, int]]):
    import numpy as np

    x = np.stack([value for value, _label in values]).astype("float32")
    y = np.asarray([label for _value, label in values], dtype="int64")
    return x, y


def _centroid_leave_one_out_accuracy(values: list[tuple[Any, int]]) -> float:
    import numpy as np

    if len(values) < 4:
        return 0.0
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
    return correct / total if total else 0.0


def _logistic_cv_probe_row(
    layer_idx: int,
    x: Any,
    y: Any,
    *,
    cv_folds: int,
    require_sklearn: bool,
    random_state: int,
) -> LayerProbeRow:
    import numpy as np

    positive_count = int((y == 1).sum())
    negative_count = int((y == 0).sum())
    fold_count = min(max(2, int(cv_folds)), positive_count, negative_count)
    if fold_count < 2:
        return LayerProbeRow(
            layer_index=layer_idx,
            method="logistic_cv",
            accuracy_mean=0.0,
            accuracy_std=0.0,
            fold_count=0,
            sample_count=int(y.shape[0]),
            positive_count=positive_count,
            negative_count=negative_count,
            status="insufficient",
            error="at least two examples per class are required for stratified CV",
        )

    sklearn_error: Exception | None = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        sklearn_error = exc
        if require_sklearn:
            return LayerProbeRow(
                layer_index=layer_idx,
                method="logistic_cv",
                accuracy_mean=0.0,
                accuracy_std=0.0,
                fold_count=0,
                sample_count=int(y.shape[0]),
                positive_count=positive_count,
                negative_count=negative_count,
                status="missing_dependency",
                error=f"scikit-learn is required for logistic_cv: {exc}",
            )
    else:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, C=1.0, class_weight="balanced"),
        )
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_state)
        try:
            scores = cross_val_score(classifier, x, y, cv=splitter, scoring="accuracy")
            return LayerProbeRow(
                layer_index=layer_idx,
                method="logistic_cv_sklearn",
                accuracy_mean=float(np.mean(scores)),
                accuracy_std=float(np.std(scores)),
                fold_count=fold_count,
                sample_count=int(y.shape[0]),
                positive_count=positive_count,
                negative_count=negative_count,
            )
        except Exception as exc:
            if require_sklearn:
                return LayerProbeRow(
                    layer_index=layer_idx,
                    method="logistic_cv_sklearn",
                    accuracy_mean=0.0,
                    accuracy_std=0.0,
                    fold_count=0,
                    sample_count=int(y.shape[0]),
                    positive_count=positive_count,
                    negative_count=negative_count,
                    status="probe_error",
                    error=f"scikit-learn logistic_cv failed: {exc}",
                )
            sklearn_error = exc

    try:
        scores = _torch_logistic_cv_scores(
            x,
            y,
            fold_count=fold_count,
            random_state=random_state,
        )
        return LayerProbeRow(
            layer_index=layer_idx,
            method="logistic_cv_torch",
            accuracy_mean=float(np.mean(scores)),
            accuracy_std=float(np.std(scores)),
            fold_count=int(len(scores)),
            sample_count=int(y.shape[0]),
            positive_count=positive_count,
            negative_count=negative_count,
            status="ok",
            error="",
        )
    except Exception as exc:
        values = [(row, int(label)) for row, label in zip(x, y)]
        return LayerProbeRow(
            layer_index=layer_idx,
            method="centroid_loo",
            accuracy_mean=_centroid_leave_one_out_accuracy(values),
            accuracy_std=0.0,
            fold_count=int(y.shape[0]),
            sample_count=int(y.shape[0]),
            positive_count=positive_count,
            negative_count=negative_count,
            status="fallback_probe_error",
            error=f"logistic_cv failed ({sklearn_error}; {exc}); used centroid leave-one-out",
        )


def _torch_logistic_cv_scores(
    x: Any,
    y: Any,
    *,
    fold_count: int,
    random_state: int,
    train_steps: int = 250,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-3,
) -> list[float]:
    import numpy as np

    torch = _require_torch()
    folds = _stratified_fold_indices(y, fold_count=fold_count, random_state=random_state)
    if not folds:
        raise ValueError("no valid stratified folds")

    scores: list[float] = []
    for train_idx, test_idx in folds:
        x_train_np = x[train_idx].astype("float32")
        x_test_np = x[test_idx].astype("float32")
        mean = x_train_np.mean(axis=0, keepdims=True)
        std = x_train_np.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        x_train_np = (x_train_np - mean) / std
        x_test_np = (x_test_np - mean) / std

        x_train = torch.from_numpy(x_train_np)
        y_train = torch.from_numpy(y[train_idx].astype("float32"))
        x_test = torch.from_numpy(x_test_np)
        y_test = torch.from_numpy(y[test_idx].astype("int64"))

        weight = torch.zeros(x_train.shape[1], dtype=torch.float32, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=learning_rate, weight_decay=weight_decay)

        positive = y_train.sum().clamp_min(1.0)
        negative = (y_train.numel() - y_train.sum()).clamp_min(1.0)
        positive_weight = y_train.numel() / (2.0 * positive)
        negative_weight = y_train.numel() / (2.0 * negative)
        sample_weight = torch.where(y_train > 0.5, positive_weight, negative_weight)

        for _step in range(train_steps):
            optimizer.zero_grad(set_to_none=True)
            logits = x_train @ weight + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_train,
                reduction="none",
            )
            weighted_loss = (loss * sample_weight).mean()
            weighted_loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = ((x_test @ weight + bias) >= 0).to(torch.int64)
            scores.append(float((predictions == y_test).to(torch.float32).mean().item()))
    return scores


def _stratified_fold_indices(y: Any, *, fold_count: int, random_state: int) -> list[tuple[Any, Any]]:
    import numpy as np

    rng = np.random.default_rng(random_state)
    folds: list[list[int]] = [[] for _ in range(fold_count)]
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for offset, sample_idx in enumerate(indices):
            folds[offset % fold_count].append(int(sample_idx))

    all_indices = np.arange(y.shape[0])
    result: list[tuple[Any, Any]] = []
    for fold in folds:
        test_idx = np.asarray(sorted(fold), dtype="int64")
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        if test_idx.size == 0:
            continue
        if len(np.unique(y[test_idx])) < 2 or len(np.unique(y[train_idx])) < 2:
            continue
        result.append((train_idx, test_idx))
    return result


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for activation-vector extraction.") from exc
    return torch
