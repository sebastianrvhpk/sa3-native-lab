import numpy as np
import torch

from latent_audio_primitives import LatentItem, load_items, save_items
from latent_audio_primitives.experiments.activation_vectors import (
    ActivationExample,
    probe_layer_accuracy,
    vectors_from_examples,
)


def test_save_and_load_items_roundtrip(tmp_path):
    item = LatentItem(
        item_id="a/b test",
        latent=np.ones((4, 3), dtype=np.float32),
        latent_rate=10.77,
        sample_rate=44100,
        prompt="test prompt",
        descriptors={"brightness": 0.5},
        labels={"favorite": True},
        metadata={"seed": 1},
    )

    save_items([item], tmp_path)
    loaded = load_items(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].item_id == item.item_id
    assert loaded[0].prompt == item.prompt
    assert loaded[0].descriptors == item.descriptors
    assert loaded[0].labels == item.labels
    assert loaded[0].metadata == item.metadata
    np.testing.assert_allclose(loaded[0].latent, item.latent)


def test_vectors_from_examples_computes_mean_difference():
    examples = [
        ActivationExample({0: torch.tensor([2.0, 0.0])}, 1, "pos1", 0, "axis", "fam"),
        ActivationExample({0: torch.tensor([4.0, 0.0])}, 1, "pos2", 1, "axis", "fam"),
        ActivationExample({0: torch.tensor([0.0, 0.0])}, 0, "neg1", 0, "axis", "fam"),
        ActivationExample({0: torch.tensor([0.0, 0.0])}, 0, "neg2", 1, "axis", "fam"),
    ]

    vectors = vectors_from_examples(examples)

    torch.testing.assert_close(vectors.vectors[0], torch.tensor([1.0, 0.0]))


def test_probe_layer_accuracy_prefers_separable_layer():
    examples = []
    for pair_idx in range(4):
        examples.append(
            ActivationExample(
                {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([float(pair_idx), 0.0])},
                1,
                "pos",
                pair_idx,
                "axis",
                "fam",
            )
        )
        examples.append(
            ActivationExample(
                {0: torch.tensor([-1.0, 0.0]), 1: torch.tensor([float(pair_idx), 0.0])},
                0,
                "neg",
                pair_idx,
                "axis",
                "fam",
            )
        )

    scores = probe_layer_accuracy(examples)

    assert scores[0] == 1.0
    assert scores[1] < scores[0]
