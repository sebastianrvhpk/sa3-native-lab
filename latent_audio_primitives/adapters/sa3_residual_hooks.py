"""Residual activation capture and steering hooks for SA3 research cells."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from latent_audio_primitives.residual_probes import SteeringVectors


class AudioscopeIntegrationError(RuntimeError):
    """Raised when SA3 residual-stream steering cannot be configured."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise AudioscopeIntegrationError("PyTorch is required for audioscope-style steering.") from exc
    return torch


def get_dit_layers(model: Any) -> list[Any]:
    """Locate Stable Audio 3 DiT transformer blocks across common wrappers.

    Supported shapes include:

    - official ``StableAudioModel``:
      ``model.model.model.model.transformer.layers``
    - official ``ConditionedDiffusionModelWrapper``:
      ``model.model.model.transformer.layers``
    - raw ``DiffusionTransformer``:
      ``model.transformer.layers``

    This replaces audioscope's single hard-coded path so the same steering code
    can run against the released ``stable_audio_3`` wrapper.
    """

    candidates = [
        ("model.model.model.transformer.layers", ("model", "model", "model", "transformer", "layers")),
        ("model.model.transformer.layers", ("model", "model", "transformer", "layers")),
        ("model.transformer.layers", ("model", "transformer", "layers")),
        ("transformer.layers", ("transformer", "layers")),
        ("dit.model.transformer.layers", ("dit", "model", "transformer", "layers")),
    ]
    for label, attrs in candidates:
        current = model
        try:
            for attr in attrs:
                current = getattr(current, attr)
            if hasattr(current, "__len__") and len(current) > 0:
                return list(current)
        except AttributeError:
            continue
    raise AudioscopeIntegrationError(
        "Could not locate SA3 DiT layers. Tried: "
        + ", ".join(label for label, _ in candidates)
    )


@dataclass
class LayerActivationStore:
    """Captured residual activations for one SA3 transformer layer."""

    layer_idx: int
    activations: list[Any] = field(default_factory=list)

    def clear(self) -> None:
        self.activations.clear()

    def mean_over_sequence(self) -> Any:
        if not self.activations:
            raise AudioscopeIntegrationError(f"no activations collected for layer {self.layer_idx}")
        return _mean_activation_tensors(self.activations)

    def mean_over_window(self, start: int, end: int) -> Any:
        if start < 0 or end > len(self.activations) or start >= end:
            raise AudioscopeIntegrationError(
                f"invalid activation window {start}:{end} for layer {self.layer_idx}"
            )
        return _mean_activation_tensors(self.activations[start:end])


class ActivationCollector:
    """Forward-hook collector compatible with SA3/audioscope residual blocks."""

    def __init__(
        self,
        model: Any,
        layer_indices: list[int] | None = None,
        *,
        cpu_offload: bool = True,
    ) -> None:
        self.model = model
        self.cpu_offload = cpu_offload
        self._hooks: list[Any] = []
        self._layers: dict[int, LayerActivationStore] = {}

        layers = get_dit_layers(model)
        indices = layer_indices if layer_indices is not None else list(range(len(layers)))
        for index in indices:
            if index < 0 or index >= len(layers):
                raise IndexError(f"layer index {index} out of range for {len(layers)} layers")
            self._layers[index] = LayerActivationStore(index)
            self._hooks.append(layers[index].register_forward_hook(self._make_hook(index)))

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(self._layers)

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            act = output[0] if isinstance(output, (tuple, list)) else output
            act = act.detach()
            if self.cpu_offload:
                act = act.cpu()
            self._layers[layer_idx].activations.append(act)

        return hook

    def clear(self) -> None:
        for store in self._layers.values():
            store.clear()

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_mean_activations(self) -> dict[int, Any]:
        return {index: store.mean_over_sequence() for index, store in self._layers.items()}

    def get_raw_activations(self) -> dict[int, list[Any]]:
        return {index: list(store.activations) for index, store in self._layers.items()}

    def get_windowed_mean_activations(
        self,
        *,
        window_count: int | None = 5,
        window_size: int | None = None,
    ) -> tuple[dict[int, dict[int, Any]], dict[int, dict[int, dict[str, Any]]]]:
        """Pool each layer's captured forward calls into trajectory windows.

        These windows are indexed by observed block-forward call order. They are
        not guaranteed to be exact sampler timesteps unless the upstream sampler
        exposes timestep metadata.
        """

        pooled: dict[int, dict[int, Any]] = {}
        metadata: dict[int, dict[int, dict[str, Any]]] = {}
        for index, store in self._layers.items():
            windows = activation_call_windows(
                len(store.activations),
                window_count=window_count,
                window_size=window_size,
            )
            pooled[index] = {}
            metadata[index] = {}
            for window_index, (start, end) in enumerate(windows):
                pooled[index][window_index] = store.mean_over_window(start, end)
                metadata[index][window_index] = {
                    "window_index": window_index,
                    "call_start": start,
                    "call_end": end,
                    "call_count": len(store.activations),
                    "window_start_fraction": start / len(store.activations),
                    "window_end_fraction": end / len(store.activations),
                    "window_label": _window_label(window_index, len(windows), start, end),
                    "window_count": len(windows),
                }
        return pooled, metadata

    def get_timestep_mean_activations(
        self,
        step_records: list[dict[str, Any]],
    ) -> tuple[dict[int, dict[int, Any]], dict[int, dict[int, dict[str, Any]]]]:
        """Pool captured calls according to sampler callback step records."""

        if not step_records:
            raise AudioscopeIntegrationError("sampler timestep records are required")
        pooled: dict[int, dict[int, Any]] = {}
        metadata: dict[int, dict[int, dict[str, Any]]] = {}
        record_count = len(step_records)
        for index, store in self._layers.items():
            call_count = len(store.activations)
            if call_count < record_count:
                raise AudioscopeIntegrationError(
                    f"layer {index} captured {call_count} calls for {record_count} sampler records"
                )
            if call_count % record_count == 0:
                calls_per_step = call_count // record_count
                windows = [
                    (step_index * calls_per_step, (step_index + 1) * calls_per_step)
                    for step_index in range(record_count)
                ]
                mapping_status = "exact_one_call_per_step" if calls_per_step == 1 else "grouped_calls_per_step"
            else:
                windows = activation_call_windows(call_count, window_count=record_count)
                calls_per_step = None
                mapping_status = "approximate_even_mapping"
            pooled[index] = {}
            metadata[index] = {}
            for step_index, (start, end) in enumerate(windows):
                record = dict(step_records[step_index])
                pooled[index][step_index] = store.mean_over_window(start, end)
                metadata[index][step_index] = {
                    **record,
                    "step_index": step_index,
                    "call_start": start,
                    "call_end": end,
                    "call_count": call_count,
                    "calls_per_step": calls_per_step,
                    "mapping_status": mapping_status,
                }
        return pooled, metadata

    def __enter__(self) -> "ActivationCollector":
        return self

    def __exit__(self, *_args) -> None:
        self.remove_hooks()


class ResidualSteerer:
    """Monkey-patch SA3 transformer blocks with residual-stream steering vectors."""

    def __init__(
        self,
        model: Any,
        steering_vectors: SteeringVectors,
        *,
        layer: int | None = None,
        top_k: int = 5,
    ) -> None:
        self.model = model
        self.steering_vectors = steering_vectors
        layers = get_dit_layers(model)
        self.layer_indices = steering_vectors.target_layers(layer=layer, top_k=top_k)
        self._blocks = {index: layers[index] for index in self.layer_indices}

    @contextmanager
    def steer(self, alpha: float = 1.0, schedule: Any = None):
        """Temporarily add residual vectors, optionally gated by hook-call order.

        ``schedule`` may be a callable accepting
        ``(layer_index, call_index, base_alpha)`` or an object with
        ``alpha_for(layer_index, call_index, base_alpha=...)``. This keeps the
        adapter independent of the trajectory-cartography module while allowing
        probe-ranked layer/timestep cells to gate steering.
        """

        originals = {}
        call_counts = {index: 0 for index in self._blocks}
        for index, block in self._blocks.items():
            originals[index] = block.forward
            vector = self.steering_vectors.vectors[index]

            def make_patched(layer_index, original_forward, steering_vector):
                def patched(*args, **kwargs):
                    call_index = call_counts[layer_index]
                    call_counts[layer_index] += 1
                    output = original_forward(*args, **kwargs)
                    scheduled_alpha = _scheduled_alpha(
                        schedule,
                        layer_index=layer_index,
                        call_index=call_index,
                        base_alpha=alpha,
                    )
                    if scheduled_alpha == 0:
                        return output
                    return _add_vector_to_output(output, steering_vector, scheduled_alpha)

                return patched

            block.forward = make_patched(index, originals[index], vector)

        try:
            yield
        finally:
            for index, block in self._blocks.items():
                block.forward = originals[index]


def _add_vector_to_output(output: Any, vector: Any, alpha: float) -> Any:
    torch = _require_torch()

    def add(tensor):
        vec = vector.to(device=tensor.device, dtype=tensor.dtype)
        return tensor + alpha * vec.unsqueeze(0).unsqueeze(0)

    if isinstance(output, torch.Tensor):
        return add(output)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (add(output[0]), *output[1:])
    if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
        return [add(output[0]), *output[1:]]
    raise AudioscopeIntegrationError(f"cannot steer unsupported block output type {type(output)!r}")


def _scheduled_alpha(schedule: Any, *, layer_index: int, call_index: int, base_alpha: float) -> float:
    if schedule is None:
        return float(base_alpha)
    if hasattr(schedule, "alpha_for"):
        return float(schedule.alpha_for(layer_index, call_index, base_alpha=base_alpha))
    if callable(schedule):
        return float(schedule(layer_index, call_index, base_alpha))
    raise TypeError("schedule must be callable or expose alpha_for(layer_index, call_index, base_alpha=...)")


def activation_call_windows(
    call_count: int,
    *,
    window_count: int | None = 5,
    window_size: int | None = None,
) -> list[tuple[int, int]]:
    """Return non-empty windows over observed activation-call indices."""

    if call_count <= 0:
        raise AudioscopeIntegrationError("cannot window an empty activation trace")
    if window_size is not None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        return [(start, min(start + window_size, call_count)) for start in range(0, call_count, window_size)]
    if window_count is None:
        window_count = 1
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    count = min(int(window_count), call_count)
    windows: list[tuple[int, int]] = []
    for index in range(count):
        start = int(index * call_count / count)
        end = int((index + 1) * call_count / count)
        if start < end:
            windows.append((start, end))
    return windows


def _mean_activation_tensors(activations: list[Any]) -> Any:
    torch = _require_torch()
    stacked = torch.cat(activations, dim=0)
    if stacked.ndim <= 1:
        return stacked
    return stacked.mean(dim=tuple(range(stacked.ndim - 1)))


def _window_label(window_index: int, window_count: int, start: int, end: int) -> str:
    if window_count == 1:
        return "all"
    if window_count == 2:
        return ("early", "late")[window_index]
    if window_count == 3:
        return ("early", "middle", "late")[window_index]
    return f"window_{window_index:02d}_{start}_{end}"
