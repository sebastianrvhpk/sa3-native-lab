import numpy as np
import pytest

from latent_audio_primitives import LatentItem, boundary_summary, latent_summary


def test_channel_first_constructor_transposes_same_style_latent():
    channel_first = np.zeros((256, 12), dtype=np.float32)
    item = LatentItem.from_channel_first("same-like", channel_first, latent_rate=10.77)

    assert item.latent.shape == (12, 256)
    assert item.frames == 12
    assert item.dim == 256


def test_latent_summary_concatenates_mean_std_and_velocity():
    latent = np.array(
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 5.0],
        ],
        dtype=np.float32,
    )

    summary = latent_summary(latent)

    assert summary.shape == (6,)
    np.testing.assert_allclose(summary[:2], [1.0, 3.0])
    np.testing.assert_allclose(summary[4:], [1.0, 2.0])


def test_boundary_summary_rejects_invalid_side():
    latent = np.zeros((4, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="side"):
        boundary_summary(latent, "middle", k=2)
