"""Notebook evidence instruments for auditioning, annotation, and review."""

from .annotations import load_audio_annotations, save_audio_annotation, search_audio_annotations
from .audio_player import audio_player_html, display_audio_player
from .disagreement import (
    DisagreementRow,
    descriptor_delta_norm,
    disagreement_rows_to_table,
    make_disagreement_row,
    native_evidence_conflict_score,
    rank_disagreement_rows,
)

__all__ = [
    "DisagreementRow",
    "audio_player_html",
    "descriptor_delta_norm",
    "disagreement_rows_to_table",
    "display_audio_player",
    "load_audio_annotations",
    "make_disagreement_row",
    "native_evidence_conflict_score",
    "rank_disagreement_rows",
    "save_audio_annotation",
    "search_audio_annotations",
]
