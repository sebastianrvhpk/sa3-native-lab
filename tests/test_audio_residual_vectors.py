import sys
import types
from pathlib import Path

import torch
from torch import nn

from latent_audio_primitives.experiments.audio_residual_vectors import SA3AudioResidualVectorExtractor


class BiasBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = 0.0

    def forward(self, x):
        return x + self.bias


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([BiasBlock()])


class DiffusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()


class DiTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiffusionTransformer()


class ConditionedWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DiTWrapper()


class FakeSA3:
    def __init__(self):
        self.model = ConditionedWrapper()
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        audio_tuple = kwargs.get("init_audio")
        mean = float(audio_tuple[1].mean()) if audio_tuple is not None else 0.0
        block = self.model.model.model.transformer.layers[0]
        block.bias = mean
        x = torch.zeros(1, 2, 3)
        return block(x)


def test_audio_residual_extractor_uses_audio_to_audio_path(tmp_path, monkeypatch):
    pos = tmp_path / "pos.wav"
    neg = tmp_path / "neg.wav"
    pos.write_bytes(b"fake")
    neg.write_bytes(b"fake")

    fake_torchaudio = types.SimpleNamespace(
        load=lambda path: (
            torch.full((1, 16), 1.0 if Path(path).name == "pos.wav" else -1.0),
            8000,
        )
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    model = FakeSA3()
    extractor = SA3AudioResidualVectorExtractor(model, layer_indices=[0], cpu_offload=True)

    result = extractor.extract(
        [pos],
        [neg],
        prompt="neutral",
        duration=1.0,
        steps=2,
        init_noise_level=0.25,
        probe=False,
    )

    assert len(model.calls) == 2
    assert model.calls[0]["init_noise_level"] == 0.25
    assert model.calls[0]["init_audio"][0] == 8000
    assert 0 in result.vectors.vectors
    torch.testing.assert_close(result.vectors.vectors[0], torch.full((3,), 1 / (3**0.5)))


def test_audio_residual_extractor_can_use_prompt_baseline_without_negative_audio(tmp_path, monkeypatch):
    pos = tmp_path / "pos.wav"
    pos.write_bytes(b"fake")

    fake_torchaudio = types.SimpleNamespace(
        load=lambda path: (torch.full((1, 16), 1.0), 8000)
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)

    model = FakeSA3()
    extractor = SA3AudioResidualVectorExtractor(model, layer_indices=[0], cpu_offload=True)

    result = extractor.extract(
        [pos],
        negative_paths=None,
        prompt="neutral",
        duration=1.0,
        steps=2,
        init_noise_level=0.25,
        baseline_mode="prompt",
        probe=False,
    )

    assert len(model.calls) == 2
    assert "init_audio" in model.calls[0]
    assert "init_audio" not in model.calls[1]
    assert 0 in result.vectors.vectors
