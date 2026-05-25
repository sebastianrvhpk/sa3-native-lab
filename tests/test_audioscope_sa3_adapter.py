import torch
from torch import nn

from latent_audio_primitives.adapters.audioscope_sa3 import (
    ActivationCollector,
    ResidualSteerer,
    SteeringVectors,
    get_dit_layers,
    mean_difference_vectors,
)


class IdentityBlock(nn.Module):
    def forward(self, x):
        return x


class Transformer(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([IdentityBlock() for _ in range(n_layers)])


class DiffusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()


class DiTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiffusionTransformer()


class ConditionedWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiTWrapper()


class StableAudioModelLike:
    def __init__(self):
        self.model = ConditionedWrapper()


def test_get_dit_layers_handles_official_stable_audio_model_shape():
    fake = StableAudioModelLike()

    layers = get_dit_layers(fake)

    assert len(layers) == 3
    assert all(isinstance(layer, IdentityBlock) for layer in layers)


def test_activation_collector_captures_mean_block_output():
    fake = StableAudioModelLike()
    layers = get_dit_layers(fake)
    x = torch.ones(2, 4, 5)

    with ActivationCollector(fake, layer_indices=[1], cpu_offload=True) as collector:
        layers[1](x)
        means = collector.get_mean_activations()

    assert set(means) == {1}
    torch.testing.assert_close(means[1], torch.ones(5))


def test_mean_difference_vectors_are_normalized():
    vectors = mean_difference_vectors(
        {0: torch.tensor([2.0, 0.0])},
        {0: torch.tensor([0.0, 0.0])},
    )

    torch.testing.assert_close(vectors.vectors[0], torch.tensor([1.0, 0.0]))


def test_residual_steerer_patches_and_restores_block_forward():
    fake = StableAudioModelLike()
    layers = get_dit_layers(fake)
    original_forward = layers[1].forward
    vectors = SteeringVectors(vectors={1: torch.ones(5)}, best_layer=1)
    steerer = ResidualSteerer(fake, vectors, layer=1)
    x = torch.zeros(2, 4, 5)

    with steerer.steer(alpha=2.0):
        y = layers[1](x)

    torch.testing.assert_close(y, torch.full_like(x, 2.0))
    assert layers[1].forward == original_forward
    torch.testing.assert_close(layers[1](x), x)


def test_steering_vectors_save_load_audioscope_format(tmp_path):
    path = tmp_path / "vectors.pt"
    vectors = SteeringVectors(
        vectors={2: torch.tensor([0.0, 1.0])},
        probe_accuracy={2: 0.9},
        best_layer=2,
    )

    vectors.save(path)
    loaded = SteeringVectors.load(path)

    assert loaded.best_layer == 2
    assert loaded.probe_accuracy == {2: 0.9}
    torch.testing.assert_close(loaded.vectors[2], torch.tensor([0.0, 1.0]))
