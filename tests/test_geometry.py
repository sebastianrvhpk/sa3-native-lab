import numpy as np

from latent_audio_primitives.geometry import (
    covariance_transport,
    fit_latent_geometry,
    geometry_report,
    latent_barycenter,
    mahalanobis_summary_distance,
    pca_project,
    pca_reconstruct,
)


def test_pca_project_reconstruct_round_trips_full_basis():
    z = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    geometry = fit_latent_geometry(z, n_components=2)

    coeffs = pca_project(z, geometry)
    reconstructed = pca_reconstruct(coeffs, geometry)

    np.testing.assert_allclose(reconstructed, z, atol=1e-5)


def test_mahalanobis_distance_is_zero_for_same_latent():
    z = np.arange(12, dtype=np.float32).reshape(6, 2)
    geometry = fit_latent_geometry(z, n_components=2)

    assert mahalanobis_summary_distance(z, z, geometry) == 0.0


def test_covariance_transport_moves_mean_toward_reference():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    reference = source + np.array([10.0, -3.0], dtype=np.float32)

    transported = covariance_transport(source, reference, alpha=1.0)

    np.testing.assert_allclose(transported.mean(axis=0), reference.mean(axis=0), atol=1e-4)


def test_latent_barycenter_averages_equal_length_latents():
    a = np.zeros((3, 2), dtype=np.float32)
    b = np.ones((3, 2), dtype=np.float32) * 2

    center = latent_barycenter([a, b], weights=[0.25, 0.75])

    np.testing.assert_allclose(center, np.ones((3, 2), dtype=np.float32) * 1.5)


def test_geometry_report_kept_fraction_uses_total_variance():
    z = np.stack(
        [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 2.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 3.0], dtype=np.float32),
        ]
    )

    report = geometry_report([z], n_components=1)

    assert 0.0 < report["kept_variance_fraction"] < 1.0
