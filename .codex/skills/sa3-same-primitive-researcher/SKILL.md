---
name: sa3-same-primitive-researcher
description: Implement and test reusable SA3/SAME latent-audio research primitives. Use when adding or modifying latent_audio_primitives helpers for flow scoring, prompt optimization, SAME memory, latent DSP, geometry, guidance, control lanes, descriptors, residual probes, observability, or scripts that wrap those helpers.
---

# SA3/SAME Primitive Researcher

## Scope

Use this skill for reusable Python research code in:

- `latent_audio_primitives/`
- `latent_audio_primitives/experiments/`
- `latent_audio_primitives/adapters/`
- `scripts/` helpers that call primitives
- `tests/test_*.py`

## Research Contract

Keep helper APIs small, tensor-shape explicit, and testable without model
weights. Prefer fake model objects and synthetic tensors for unit tests. Use
loaded SA3/SAME weights only for Colab or manual smoke validation.

Core object conventions:

```text
audio x
SAME latent z, usually B x C x T or T x D depending on adapter boundary
prompt conditioning C(p)
SA3 flow field v_theta(z_t, t, C(p))
descriptor/control rows
experiment manifest entries
```

## Workflow

1. Locate the owner module and tests.
   - Match existing local style and dataclasses.
   - Keep shape conversions explicit at adapter boundaries.
   - Preserve velocity convention arguments in flow/prompt helpers.

2. Implement the primitive.
   - Put reusable math in `latent_audio_primitives/`.
   - Put notebook-only orchestration in the notebook.
   - Put command-line wrappers in `scripts/` only when they are useful outside
     the notebook.

3. Add tests.
   - Fake model tests for flow, prompt, residual, and guidance helpers.
   - Synthetic latent tests for geometry, DSP, memory, descriptors, and control
     lanes.
   - Assertions should cover shapes, deterministic seeds, metadata, and edge
     cases.

4. Validate.
   - Run focused tests first.
   - Run `uv run pytest` when touching shared primitives.
   - Run notebook validation when notebook imports or examples change.
   - Run `git diff --check`.

## Output

Report:

- primitive added or changed
- shape and velocity conventions
- tests added or updated
- notebook/script touchpoints
