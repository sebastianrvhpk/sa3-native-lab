import torch

from latent_audio_primitives.guidance import combine_guidance_losses, gradient_guidance_step


def test_gradient_guidance_step_moves_down_quadratic_loss():
    z = torch.tensor([1.0, -1.0])

    result = gradient_guidance_step(z, lambda value: (value**2).mean(), scale=0.1, normalize=False)

    assert result.loss == 1.0
    assert torch.linalg.vector_norm(result.latents) < torch.linalg.vector_norm(z)


def test_combine_guidance_losses_adds_weighted_terms():
    z = torch.tensor([2.0])
    loss_fn = combine_guidance_losses(
        (1.0, lambda value: value.mean()),
        (2.0, lambda value: (value**2).mean()),
    )

    assert float(loss_fn(z)) == 10.0
