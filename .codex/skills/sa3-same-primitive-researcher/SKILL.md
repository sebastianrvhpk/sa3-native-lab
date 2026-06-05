---
name: sa3-same-primitive-researcher
description: Implement reusable SA3/SAME latent-audio research primitives for notebook cells. Use when adding or modifying latent_audio_primitives helpers for flow scoring, prompt optimization, SAME memory, latent DSP, geometry, guidance, control lanes, descriptors, residual probes, or observability.
---

# SA3/SAME Primitive Researcher

## Scope

Use this skill for reusable Python research code in:

- `latent_audio_primitives/`
- `latent_audio_primitives/experiments/`
- `latent_audio_primitives/adapters/`

## Research Contract

Keep helper APIs small, tensor-shape explicit, and easy to call from notebook
cells. Prefer synthetic tensors or tiny fake model objects for local smoke
checks when that clarifies behavior. Use loaded SA3/SAME weights only for Colab
or manual smoke validation.

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

1. Locate the owner module and notebook touchpoint.
   - Match existing local style and dataclasses.
   - Keep shape conversions explicit at adapter boundaries.
   - Preserve velocity convention arguments in flow/prompt helpers.

2. Implement the primitive.
   - Put reusable math in `latent_audio_primitives/`.
   - Put notebook-only orchestration in the notebook.
   - Keep orchestration in the notebook unless repeated math clearly belongs in
     `latent_audio_primitives/`.

3. Validate.
   - Run small import/smoke snippets for changed helpers when practical.
   - Run notebook JSON validation when notebook cells change.
   - Smoke-test notebook cells in Colab when model weights or audio behavior are involved.
   - Run `git diff --check`.

## Output

Report:

- primitive added or changed
- shape and velocity conventions
- notebook touchpoints
- local or Colab smoke checks performed
