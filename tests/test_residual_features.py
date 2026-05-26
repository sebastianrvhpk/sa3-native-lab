import numpy as np

from latent_audio_primitives.residual_features import (
    fit_residual_feature_basis,
    project_residual_features,
    residual_feature_direction,
)


def test_residual_feature_basis_projects_activation():
    activations = [
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]

    basis = fit_residual_feature_basis(activations, layer="block_0", n_components=2)
    coeffs = project_residual_features(activations[0], basis)

    assert basis.layer == "block_0"
    assert coeffs.shape == (2,)


def test_residual_feature_direction_is_normalized():
    positive = [np.array([2.0, 0.0], dtype=np.float32)]
    reference = [np.array([0.0, 0.0], dtype=np.float32)]

    direction = residual_feature_direction(positive, reference)

    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
