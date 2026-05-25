import torch

from latent_audio_primitives.experiments.soft_prompt import SoftPromptState


def test_soft_prompt_state_roundtrip(tmp_path):
    state = SoftPromptState(
        conditioning=[{"prompt": "audio texture", "seconds_total": 4.0}],
        conditioning_tensors={
            "prompt": [torch.ones(1, 2, 3), torch.ones(1, 2)],
            "seconds_total": [torch.zeros(1, 1, 3), torch.ones(1, 1)],
        },
        losses=[2.0, 1.0],
        metadata={"target": "file.wav"},
    )
    path = tmp_path / "soft_prompt.pt"

    state.save(path)
    loaded = SoftPromptState.load(path)

    assert loaded.conditioning == state.conditioning
    assert loaded.losses == [2.0, 1.0]
    assert loaded.metadata == {"target": "file.wav"}
    torch.testing.assert_close(loaded.conditioning_tensors["prompt"][0], torch.ones(1, 2, 3))
