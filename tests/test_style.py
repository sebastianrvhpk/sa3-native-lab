import numpy as np

from latent_audio_primitives import (
    LatentItem,
    apply_profile_attraction,
    apply_style_direction,
    fit_style_profile,
    load_style_direction,
    load_style_profile,
    save_style_direction,
    save_style_profile,
    style_direction,
)


def item(item_id: str, values) -> LatentItem:
    return LatentItem(item_id=item_id, latent=np.asarray(values, dtype=np.float32), latent_rate=10.0)


def test_fit_style_profile_tracks_dataset_mean_and_std():
    items = [
        item("a", [[0.0, 0.0], [2.0, 2.0]]),
        item("b", [[2.0, 2.0], [4.0, 4.0]]),
    ]

    profile = fit_style_profile(items, name="target")

    np.testing.assert_allclose(profile.mean, [2.0, 2.0])
    np.testing.assert_allclose(profile.std, [1.0, 1.0])
    assert profile.item_count == 2


def test_apply_profile_attraction_alpha_one_matches_target_stats():
    source = np.asarray([[10.0, 20.0], [12.0, 22.0], [14.0, 24.0]], dtype=np.float32)
    target_items = [item("target", [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])]
    profile = fit_style_profile(target_items, name="target")

    styled = apply_profile_attraction(source, profile, alpha=1.0, match_std=True)

    np.testing.assert_allclose(styled.mean(axis=0), profile.mean, atol=1e-5)
    np.testing.assert_allclose(styled.std(axis=0), profile.std, atol=1e-5)


def test_style_direction_shifts_latent_mean_by_delta():
    target = fit_style_profile([item("target", [[2.0], [4.0]])], name="target")
    reference = fit_style_profile([item("reference", [[0.0], [2.0]])], name="reference")
    direction = style_direction(target, reference)
    source = np.asarray([[10.0], [12.0]], dtype=np.float32)

    styled = apply_style_direction(source, direction, alpha=0.5)

    np.testing.assert_allclose(styled.mean(axis=0), source.mean(axis=0) + 0.5 * direction.mean_delta)


def test_style_profile_and_direction_roundtrip(tmp_path):
    target = fit_style_profile([item("target", [[2.0], [4.0]])], name="target")
    reference = fit_style_profile([item("reference", [[0.0], [2.0]])], name="reference")
    direction = style_direction(target, reference)

    profile_path = tmp_path / "profile.npz"
    direction_path = tmp_path / "direction.npz"
    save_style_profile(target, profile_path)
    save_style_direction(direction, direction_path)

    loaded_profile = load_style_profile(profile_path)
    loaded_direction = load_style_direction(direction_path)

    assert loaded_profile.name == "target"
    assert loaded_direction.target_name == "target"
    np.testing.assert_allclose(loaded_profile.mean, target.mean)
    np.testing.assert_allclose(loaded_direction.mean_delta, direction.mean_delta)
