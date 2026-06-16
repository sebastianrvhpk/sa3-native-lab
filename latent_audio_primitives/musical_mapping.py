"""Musical timeline helpers translating beats/bars to model-native schedules."""

from __future__ import annotations

from typing import Sequence
from .sampler_composition import ScalarSchedule


class MusicalTimeline:
    """A helper for designing schedules in beats, bars, and BPM."""

    def __init__(self, tempo: float, total_duration: float, time_signature_numerator: int = 4) -> None:
        if tempo <= 0:
            raise ValueError("tempo must be positive")
        if total_duration <= 0:
            raise ValueError("total_duration must be positive")
        self.tempo = float(tempo)
        self.total_duration = float(total_duration)
        self.time_signature_numerator = int(time_signature_numerator)

    @property
    def total_beats(self) -> float:
        """Total number of beats in the duration."""
        return (self.total_duration / 60.0) * self.tempo

    @property
    def total_bars(self) -> float:
        """Total number of bars in the duration."""
        return self.total_beats / self.time_signature_numerator

    def beat_to_step_fraction(self, beat: float) -> float:
        """Convert a beat index to a step fraction (0.0 to 1.0)."""
        if self.total_beats <= 0:
            return 0.0
        return float(beat) / self.total_beats

    def bar_to_step_fraction(self, bar: float) -> float:
        """Convert a bar index to a step fraction (0.0 to 1.0)."""
        return self.beat_to_step_fraction(bar * self.time_signature_numerator)

    def make_beat_schedule(
        self,
        knots: Sequence[tuple[float, float]],
        *,
        name: str = "",
        interpolation: str = "linear",
    ) -> ScalarSchedule:
        """Create a ScalarSchedule in 'step_fraction' coordinate from beat-based knots.

        Knots should be formatted as sequence of ``(beat_number, value)`` pairs.
        """
        fraction_knots = []
        for beat, val in knots:
            frac = self.beat_to_step_fraction(beat)
            fraction_knots.append((frac, val))
        return ScalarSchedule(
            knots=fraction_knots,
            coordinate="step_fraction",
            interpolation=interpolation,
            name=name,
        )

    def make_bar_schedule(
        self,
        knots: Sequence[tuple[float, float]],
        *,
        name: str = "",
        interpolation: str = "linear",
    ) -> ScalarSchedule:
        """Create a ScalarSchedule in 'step_fraction' coordinate from bar-based knots.

        Knots should be formatted as sequence of ``(bar_number, value)`` pairs.
        """
        beat_knots = [(bar * self.time_signature_numerator, val) for bar, val in knots]
        return self.make_beat_schedule(beat_knots, name=name, interpolation=interpolation)
