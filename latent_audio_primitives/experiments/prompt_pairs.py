from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptPair:
    positive: str
    negative: str
    axis: str
    family: str = "general"


DEFAULT_PROMPT_PAIRS: list[PromptPair] = [
    PromptPair(
        "a euphoric bright piano house loop with warm uplifting chords",
        "a melancholic dark piano house loop with restrained sorrowful chords",
        axis="valence",
        family="house_piano",
    ),
    PromptPair(
        "a joyful acoustic guitar folk phrase, sunny and hopeful",
        "a sad acoustic guitar folk phrase, muted and lonely",
        axis="valence",
        family="folk_guitar",
    ),
    PromptPair(
        "a bright shimmering ambient pad with airy high frequencies",
        "a dark muted ambient pad with low warm frequencies",
        axis="brightness",
        family="ambient_pad",
    ),
    PromptPair(
        "a sparse minimal techno groove with few elements and open space",
        "a dense busy techno groove with many layered elements and constant motion",
        axis="density",
        family="techno",
    ),
    PromptPair(
        "a calm relaxed cinematic string bed with gentle movement",
        "a tense suspenseful cinematic string bed with rising pressure",
        axis="tension",
        family="cinematic_strings",
    ),
    PromptPair(
        "a clean polished synth texture with smooth transients",
        "a noisy grainy synth texture with rough tape-like artifacts",
        axis="grain",
        family="synth_texture",
    ),
    PromptPair(
        "a narrow centered mono-feeling electronic pulse",
        "a wide immersive stereo electronic pulse surrounding the listener",
        axis="stereo_width",
        family="electronic_pulse",
    ),
    PromptPair(
        "a gentle low-energy intro section that slowly opens",
        "a powerful high-energy drop section with strong impact",
        axis="section_energy",
        family="arrangement",
    ),
]


def pairs_for_axis(axis: str, pairs: list[PromptPair] | None = None) -> list[PromptPair]:
    source = pairs if pairs is not None else DEFAULT_PROMPT_PAIRS
    return [pair for pair in source if pair.axis == axis]
