import numpy as np

from latent_audio_primitives import LatentItem
from latent_audio_primitives.audio_vectors import (
    apply_frame_direction,
    frame_mean_direction,
    load_summary_direction,
    save_summary_direction,
    summary_direction,
)


def item(item_id: str, values) -> LatentItem:
    return LatentItem(item_id=item_id, latent=np.asarray(values, dtype=np.float32), latent_rate=10.0)


def test_frame_mean_direction_from_audio_sets_applies_to_latent():
    positive = [item("pos", [[2.0, 4.0], [4.0, 6.0]])]
    negative = [item("neg", [[0.0, 1.0], [2.0, 3.0]])]
    direction = frame_mean_direction(positive, negative)
    source = np.zeros((3, 2), dtype=np.float32)

    steered = apply_frame_direction(source, direction, alpha=0.5)

    expected_delta = 0.5 * direction.mean_delta
    np.testing.assert_allclose(steered, np.tile(expected_delta, (3, 1)))


def test_summary_direction_roundtrip(tmp_path):
    positive = [item("pos", [[2.0], [4.0]])]
    negative = [item("neg", [[0.0], [2.0]])]
    direction = summary_direction(positive, negative, normalize=False)
    path = tmp_path / "summary_direction.npz"

    save_summary_direction(direction, path)
    loaded = load_summary_direction(path)

    assert loaded.name == direction.name
    assert loaded.item_count_positive == 1
    assert loaded.item_count_negative == 1
    np.testing.assert_allclose(loaded.vector, direction.vector)
