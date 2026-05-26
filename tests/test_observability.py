import numpy as np

from latent_audio_primitives.observability import (
    fit_linear_control_probe,
    intervention_effect,
    predict_control,
)
from latent_audio_primitives.schema import LatentItem


def _item(value: float, item_id: str) -> LatentItem:
    latent = np.full((4, 2), value, dtype=np.float32)
    return LatentItem(
        item_id=item_id,
        latent=latent,
        latent_rate=1.0,
        descriptors={"brightness": value},
    )


def test_linear_control_probe_predicts_training_control():
    items = [_item(0.0, "a"), _item(1.0, "b"), _item(2.0, "c")]

    probe = fit_linear_control_probe(items, "brightness", ridge=1e-4)

    assert probe.r2_train > 0.99
    assert predict_control(probe, _item(2.0, "d")) > predict_control(probe, _item(0.0, "e"))


def test_intervention_effect_reports_delta():
    items = [_item(0.0, "a"), _item(1.0, "b"), _item(2.0, "c")]
    probe = fit_linear_control_probe(items, "brightness", ridge=1e-4)

    effect = intervention_effect(probe, _item(0.0, "before"), _item(2.0, "after"))

    assert effect["delta"] > 0.0
