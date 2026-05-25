import numpy as np
import torch

from latent_audio_primitives.adapters.stable_audio3 import latents_to_items, stable_audio3_latent_rate


class FakePretransform:
    downsampling_ratio = 4096


class FakeCore:
    sample_rate = 44100
    pretransform = FakePretransform()


class FakeStableAudioModel:
    model = FakeCore()


def test_stable_audio3_latent_rate_uses_sample_rate_and_downsampling_ratio():
    assert stable_audio3_latent_rate(FakeStableAudioModel()) == 44100 / 4096


def test_latents_to_items_converts_batch_channel_first_to_time_major():
    latents = torch.zeros(2, 256, 12)
    latents[1, :, :] = 3.0

    items = latents_to_items(
        latents,
        item_id_prefix="gen",
        latent_rate=44100 / 4096,
        sample_rate=44100,
        prompt=["a", "b"],
        metadata={"model": "fake"},
    )

    assert [item.item_id for item in items] == ["gen-0000", "gen-0001"]
    assert items[0].latent.shape == (12, 256)
    assert items[1].prompt == "b"
    assert items[1].metadata["model"] == "fake"
    np.testing.assert_allclose(items[1].latent, 3.0)
