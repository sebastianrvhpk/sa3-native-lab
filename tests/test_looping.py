import torch

from latent_audio_primitives.looping import (
    cyclic_roll_audio,
    cyclic_roll_latents,
    frames_from_fraction,
    loop_boundary_metrics,
    repeated_loop_preview_audio,
    sample_cyclic_roll_euler,
    samples_from_fraction,
    seam_inpaint_bounds,
)


def test_cyclic_roll_latents_rolls_time_axis():
    latents = torch.arange(5, dtype=torch.float32).view(1, 1, 5)

    rolled = cyclic_roll_latents(latents, 2)

    torch.testing.assert_close(rolled, torch.tensor([[[3.0, 4.0, 0.0, 1.0, 2.0]]]))


def test_cyclic_roll_audio_rolls_sample_axis():
    audio = torch.arange(6, dtype=torch.float32).view(1, 6)

    rolled = cyclic_roll_audio(audio, -2)

    torch.testing.assert_close(rolled, torch.tensor([[2.0, 3.0, 4.0, 5.0, 0.0, 1.0]]))


def test_repeated_loop_preview_audio_concatenates_repeats():
    audio = torch.tensor([[0.0, 1.0, 2.0]])

    preview = repeated_loop_preview_audio(audio, repeats=3)

    torch.testing.assert_close(preview, torch.tensor([[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]]))


def test_loop_boundary_metrics_zero_for_constant_latents():
    latents = torch.ones(1, 4, 12)

    metrics = loop_boundary_metrics(latents, window_frames=4)

    assert metrics.total == 0.0
    assert metrics.window_frames == 4


def test_loop_boundary_metrics_detects_boundary_mismatch():
    latents = torch.zeros(1, 1, 8)
    latents[..., -2:] = 4.0

    metrics = loop_boundary_metrics(latents, window_frames=2)

    assert metrics.state_l2 > 0.0


def test_shift_fraction_helpers():
    latents = torch.zeros(1, 2, 10)
    audio = torch.zeros(2, 100)

    assert frames_from_fraction(latents, 0.5) == 5
    assert samples_from_fraction(audio, 0.25) == 25


def test_seam_inpaint_bounds_are_clamped():
    assert seam_inpaint_bounds(10.0, 0.5, 2.0) == (4.0, 6.0)
    assert seam_inpaint_bounds(10.0, 0.0, 2.0) == (0.0, 1.0)
    assert seam_inpaint_bounds(10.0, 1.0, 2.0) == (9.0, 10.0)


def test_cyclic_roll_euler_alternates_state_orientation():
    class ZeroModel:
        def __call__(self, x, t, **kwargs):
            return torch.zeros_like(x)

    latents = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
    sigmas = torch.tensor([1.0, 0.5])

    rolled = sample_cyclic_roll_euler(
        ZeroModel(),
        latents,
        sigmas,
        roll_frames=2,
        mode="alternate",
        unroll_output=False,
        disable_tqdm=True,
    )

    torch.testing.assert_close(rolled, torch.tensor([[[2.0, 3.0, 0.0, 1.0]]]))


def test_cyclic_roll_euler_can_unroll_final_state():
    class ZeroModel:
        def __call__(self, x, t, **kwargs):
            return torch.zeros_like(x)

    latents = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
    sigmas = torch.tensor([1.0, 0.5])

    unrolled = sample_cyclic_roll_euler(
        ZeroModel(),
        latents,
        sigmas,
        roll_frames=2,
        mode="alternate",
        unroll_output=True,
        disable_tqdm=True,
    )

    torch.testing.assert_close(unrolled, latents)


def test_cyclic_roll_euler_accepts_per_element_schedule():
    class ZeroModel:
        def __call__(self, x, t, **kwargs):
            assert t.shape == (1,)
            return torch.zeros_like(x)

    latents = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
    sigmas = torch.tensor([[1.0, 0.5]])

    rolled = sample_cyclic_roll_euler(
        ZeroModel(),
        latents,
        sigmas,
        roll_frames=2,
        mode="alternate",
        unroll_output=False,
        disable_tqdm=True,
    )

    torch.testing.assert_close(rolled, torch.tensor([[[2.0, 3.0, 0.0, 1.0]]]))


def test_cyclic_roll_euler_paired_average_uses_both_origins():
    class PositionModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, x, t, **kwargs):
            self.calls += 1
            weights = torch.tensor([0.0, 0.0, 2.0, 2.0], dtype=x.dtype, device=x.device).view(1, 1, 4)
            return x * weights

    model = PositionModel()
    latents = torch.ones(1, 1, 4)
    sigmas = torch.tensor([1.0, 0.5])

    sampled = sample_cyclic_roll_euler(
        model,
        latents,
        sigmas,
        roll_frames=2,
        mode="paired_average",
        disable_tqdm=True,
    )

    assert model.calls == 2
    torch.testing.assert_close(sampled, torch.full((1, 1, 4), 0.5))
