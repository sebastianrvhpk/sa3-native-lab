"""Residual activation capture and steering hooks for SA3 research cells."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
        torch = _require_torch()
        if not self.activations:
            raise AudioscopeIntegrationError(f"no activations collected for layer {self.layer_idx}")
        stacked = torch.cat(self.activations, dim=0)
        return stacked.mean(dim=(0, 1))


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

    def __enter__(self) -> "ActivationCollector":
        return self

    def __exit__(self, *_args) -> None:
        self.remove_hooks()


@dataclass
class SteeringVectors:
    """audioscope-compatible steering vector container."""

    vectors: dict[int, Any] = field(default_factory=dict)
    probe_accuracy: dict[int, float] = field(default_factory=dict)
    best_layer: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> "SteeringVectors":
        torch = _require_torch()
        data = torch.load(path, map_location="cpu")
        return cls(
            vectors=dict(data["vectors"]),
            probe_accuracy=dict(data.get("probe_accuracy", {})),
            best_layer=data.get("best_layer"),
        )

    def save(self, path: str | Path) -> None:
        torch = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vectors": self.vectors,
                "probe_accuracy": self.probe_accuracy,
                "best_layer": self.best_layer,
            },
            path,
        )

    def target_layers(self, *, layer: int | None = None, top_k: int = 5) -> list[int]:
        if layer is not None:
            layers = [layer]
        elif self.probe_accuracy:
            layers = [
                index
                for index, _score in sorted(self.probe_accuracy.items(), key=lambda item: -item[1])[:top_k]
            ]
        elif self.best_layer is not None:
            layers = [self.best_layer]
        else:
            layers = sorted(self.vectors)
        missing = [index for index in layers if index not in self.vectors]
        if missing:
            raise KeyError(f"missing steering vectors for layers {missing}")
        return layers


def mean_difference_vectors(
    positive_activations: dict[int, Any],
    negative_activations: dict[int, Any],
    *,
    normalize: bool = True,
) -> SteeringVectors:
    """Compute audioscope-style contrastive mean-difference directions."""

    torch = _require_torch()
    vectors = {}
    for layer_idx, pos in positive_activations.items():
        if layer_idx not in negative_activations:
            raise KeyError(f"negative activations missing layer {layer_idx}")
        diff = pos.float() - negative_activations[layer_idx].float()
        if normalize:
            diff = diff / (torch.linalg.vector_norm(diff) + 1e-8)
        vectors[layer_idx] = diff.detach().cpu()
    return SteeringVectors(vectors=vectors)


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
    def steer(self, alpha: float = 1.0):
        originals = {}
        for index, block in self._blocks.items():
            originals[index] = block.forward
            vector = self.steering_vectors.vectors[index]

            def make_patched(original_forward, steering_vector):
                def patched(*args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    return _add_vector_to_output(output, steering_vector, alpha)

                return patched

            block.forward = make_patched(originals[index], vector)

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
