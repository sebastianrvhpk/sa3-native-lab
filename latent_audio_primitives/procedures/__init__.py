"""Notebook-facing SA3/SAME research procedures.

These modules run reusable methods that call SA3/SAME, optimize conditioning,
capture hooks, produce sweeps, or create notebook artifacts. They are plain
procedure functions, not a separate runner framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ProcedureStatus:
    """Notebook-visible maturity metadata for one executable procedure module."""

    module: str
    research_layer: str
    maturity: str
    role: str
    promotion_gate: str
    risk: str = ""


PROCEDURE_STATUS: dict[str, ProcedureStatus] = {
    "flow_scoring.py": ProcedureStatus(
        module="flow_scoring.py",
        research_layer="SA3 flow/conditioning",
        maturity="microscope / selector",
        role="teacher-forced frozen-SA3 flow scoring for prompt and condition ranking",
        promotion_gate="Flow rankings predict generated-audio or listening outcomes across shared probe banks.",
    ),
    "soft_prompt.py": ProcedureStatus(
        module="soft_prompt.py",
        research_layer="SA3 flow/conditioning",
        maturity="intervention candidate",
        role="optimize continuous SA3 conditioning against target latents, then audition generations",
        promotion_gate="Optimized conditions beat readable prompt and audio-to-audio baselines without losing reviewability.",
        risk="Can overfit vector-field agreement without audible usefulness.",
    ),
    "sa3_latent_sampling.py": ProcedureStatus(
        module="sa3_latent_sampling.py",
        research_layer="SA3-over-SAME coupled editing",
        maturity="intervention candidate",
        role="polish or resample edited SAME latents through SA3 init-latent sampling",
        promotion_gate="Method packets show which SAME edits survive direct decode and SA3 polish.",
        risk="Version-sensitive sampler boundary around upstream SA3 internals.",
    ),
    "selective_sa3.py": ProcedureStatus(
        module="selective_sa3.py",
        research_layer="SA3-over-SAME coupled editing",
        maturity="intervention candidate",
        role="run selective SAME-channel renoise and donor-channel graft experiments through SA3",
        promotion_gate="Source/baseline/method packets show predictable channel, donor, and prompt effects.",
        risk="Needs copying and source-preservation checks for every donor run.",
    ),
    "cyclic_sa3.py": ProcedureStatus(
        module="cyclic_sa3.py",
        research_layer="SA3 internal trajectory",
        maturity="high-risk sampler microscope / intervention candidate",
        role="insert cyclic latent-time projections or paired evaluations inside an Euler sampler",
        promotion_gate="Loop metrics and auditions improve versus baselines without collapse or sampler artifacts.",
        risk="Euler-only and intentionally tied to upstream sampler assumptions.",
    ),
    "residual_activation_vectors.py": ProcedureStatus(
        module="residual_activation_vectors.py",
        research_layer="SA3 internal trajectory",
        maturity="microscope",
        role="extract prompt-pair residual activation directions and layer probe scores",
        promotion_gate="Layer maps repeat before any steering claim is made.",
    ),
    "audio_residual_vectors.py": ProcedureStatus(
        module="audio_residual_vectors.py",
        research_layer="SA3 internal trajectory",
        maturity="high-risk microscope",
        role="derive residual directions from audio-to-audio activation differences",
        promotion_gate="Audio-derived directions repeat across examples and survive alpha-sweep audition.",
        risk="Can confuse source identity, prompt baseline, and generation artifacts.",
    ),
    "residual_sweeps.py": ProcedureStatus(
        module="residual_sweeps.py",
        research_layer="SA3 internal trajectory",
        maturity="high-risk intervention candidate",
        role="render alpha sweeps for residual steering vectors with saved latent/audio artifacts",
        promotion_gate="Alpha changes are audible, monotonic or interpretable, and not just artifact injection.",
        risk="Depends on fragile residual hook locations and generation settings.",
    ),
}


def procedure_status_table() -> list[dict[str, str]]:
    """Return JSON-friendly procedure maturity rows for manifests and notebooks."""

    return [asdict(status) for status in PROCEDURE_STATUS.values()]


__all__ = ["PROCEDURE_STATUS", "ProcedureStatus", "procedure_status_table"]
