from __future__ import annotations

import pytest

from latent_audio_primitives.flow_prompt import (
    flow_velocity_target,
    logsnr_from_timestep,
    prompt_leave_one_out_attribution,
    sa3_flow_loss_rows_for_prompts,
    sa3_flow_losses_for_prompts,
    summarize_flow_loss_rows,
    timesteps_from_logsnr_values,
)

torch = pytest.importorskip("torch")


def test_sa3_flow_losses_score_prompts_with_fake_model():
    model = FakeStableModel()
    target = torch.zeros((1, 2, 4), dtype=torch.float32)

    losses = sa3_flow_losses_for_prompts(
        model,
        target,
        ["quiet texture", "warm target"],
        duration=1.0,
        seed=7,
        timestep_values=[0.5],
        shared_noise=True,
        normalize_mse=False,
    )

    assert len(losses) == 2
    assert losses[1] < losses[0]


def test_sa3_flow_losses_support_probe_bank_and_delta_term():
    model = FakeStableModel()
    target = torch.ones((2, 4), dtype=torch.float32)

    losses = sa3_flow_losses_for_prompts(
        model,
        target,
        ["warm target", "cold target"],
        duration=1.0,
        seed=3,
        timestep_values=timesteps_from_logsnr_values("2,0,-2"),
        antithetic_noise=True,
        conditional_delta_weight=0.1,
        cosine_weight=0.25,
    )

    assert len(losses) == 2
    assert all(loss >= 0 for loss in losses)


def test_sa3_flow_loss_rows_keep_per_timestep_diagnostics():
    model = FakeStableModel()
    target = torch.zeros((1, 2, 4), dtype=torch.float32)

    rows = sa3_flow_loss_rows_for_prompts(
        model,
        target,
        ["quiet texture", "warm target"],
        duration=1.0,
        seed=7,
        logsnr_values=[2.0, 0.0],
        shared_noise=True,
        normalize_mse=False,
    )
    summary = summarize_flow_loss_rows(rows)

    assert len(rows) == 4
    assert {row.probe_index for row in rows} == {0, 1}
    assert rows[0].logsnr == pytest.approx(logsnr_from_timestep(rows[0].timestep))
    assert summary[0]["prompt"] == "warm target"


def test_prompt_leave_one_out_attribution_scores_helpful_tokens():
    def loss_scorer(prompts):
        losses = []
        for prompt in prompts:
            loss = 10.0
            if "warm" in prompt:
                loss -= 3.0
            if "wide" in prompt:
                loss -= 1.0
            if "cold" in prompt:
                loss += 2.0
            losses.append(loss)
        return losses

    rows = prompt_leave_one_out_attribution(
        "warm narrow texture",
        loss_scorer,
        replacement_candidates=["wide", "cold"],
    )

    assert rows[0].token == "warm"
    assert rows[0].contribution == pytest.approx(3.0)
    narrow = next(row for row in rows if row.token == "narrow")
    assert narrow.best_replacement == "wide"
    assert narrow.best_replacement_delta == pytest.approx(1.0)


def test_flow_velocity_target_convention_is_explicit():
    target = torch.tensor([1.0])
    noise = torch.tensor([3.0])

    assert flow_velocity_target(target, noise, convention="noise_minus_data").item() == 2.0
    assert flow_velocity_target(target, noise, convention="data_minus_noise").item() == -2.0
    with pytest.raises(ValueError):
        flow_velocity_target(target, noise, convention="sideways")


class FakeStableModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.model = FakeCore()


class FakeCore:
    def __init__(self) -> None:
        self.model = FakeFlowModel()

    def conditioner(self, conditioning, device):
        strengths = []
        for item in conditioning:
            prompt = str(item.get("prompt", ""))
            strengths.append(2.0 if "warm" in prompt else 0.0)
        return {"prompt_strength": torch.tensor(strengths, device=device, dtype=torch.float32)}

    def get_conditioning_inputs(self, cond):
        return {"prompt_strength": cond["prompt_strength"]}


class FakeFlowModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, z_t, t, *, prompt_strength, cfg_scale=1.0, batch_cfg=True):
        del t, cfg_scale, batch_cfg
        return prompt_strength[:, None, None] * z_t * self.anchor
