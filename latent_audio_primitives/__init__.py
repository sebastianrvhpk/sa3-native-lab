"""Notebook-facing SA3/SAME latent audio primitives.

Import concrete helpers from their altitude modules:

- root modules for native objects, math, measurements, operators, and search,
- ``adapters`` for upstream SA3/SAME access,
- ``procedures`` for executable research methods,
- ``evidence`` for auditioning, annotation, and review helpers.
"""

from .flow_prompt import (
    FlowProbeBank,
    FlowProbeSpec,
    flow_probe_bank_from_manifest,
    flow_probe_bank_from_values,
    flow_probe_bank_to_manifest,
)
from .prompt_semantics import (
    PromptSemanticRow,
    PromptVariant,
    make_prompt_variants,
    prompt_semantic_rows,
    prompt_variant_manifest,
    rank_prompt_semantic_rows,
    semantic_tag_counts,
)
from .schema import LatentItem

__all__ = [
    "FlowProbeBank",
    "FlowProbeSpec",
    "LatentItem",
    "PromptSemanticRow",
    "PromptVariant",
    "flow_probe_bank_from_manifest",
    "flow_probe_bank_from_values",
    "flow_probe_bank_to_manifest",
    "make_prompt_variants",
    "prompt_semantic_rows",
    "prompt_variant_manifest",
    "rank_prompt_semantic_rows",
    "semantic_tag_counts",
]
