# SA3 Native Lab Experiment Ledger

Status: empty ledger template for actual notebook runs.

This document answers: what was run, what audio came out, what changed in the
measurements, what listening said, and whether the method should be kept,
revised, dropped, or investigated further.

The previous research notes contained method plans and validation rules. This
ledger is the missing results layer. It should be updated from real Colab runs,
not from speculation.

Use [Run protocol](run-protocol.md) for the run spine, evidence panels, minimum
run packet, and promote/revise/drop rules.

## Ledger Rules

- Record only actual runs or deliberately planned run slots.
- Link or name output artifacts where possible.
- Include source audio, baseline output, method output, descriptor deltas, and listening notes.
- Use exact mode names, seeds, prompts, model IDs, durations, and key params.
- Mark decisions explicitly: `keep`, `revise`, `drop`, or `unknown`.
- Do not promote a method based only on latent metrics.

## Run Template

```text
Date:
Notebook commit:
Notebook mode:
Model:
Runtime:

Question:
Hypothesis:

Inputs:
  Source audio:
  Donor/reference audio:
  Dataset/memory:
  Prompt:
  Negative/reference prompt:

Recipe:
  Duration:
  Seed(s):
  Steps:
  CFG:
  Flow convention:
  LogSNR/timestep probes:
  Method params:

Outputs:
  Baseline audio:
  Method audio:
  Direct decode:
  SA3 polish:
  Manifest row:
  Descriptor report:
  Player/annotation row:

Measurements:
  Flow score delta:
  Descriptor delta:
  Latent distance:
  Periodicity/boundary:
  Nearest memory rows:
  Control lane movement:
  Residual/feature effect:

Listening notes:
  Prompt adherence:
  Source preservation:
  Novelty:
  Artifacts:
  Loopability/transition quality:
  Musical usefulness:

Decision:
  keep | revise | drop | unknown

Next action:
```

## Audio Validation Protocol

For every new method, produce:

```text
source audio
baseline generation
method generation
descriptor delta
native flow score delta if relevant
annotation prompt
manifest row
```

Run at least:

```text
three seeds
one in-domain prompt
one out-of-domain prompt
two source clips when available
```

## Promote / Drop Criteria

Drop a method if:

- it cannot move a measured signal,
- it moves the signal but listening repeatedly rejects the output,
- it only creates artifacts,
- it copies a memory item when the goal was source preservation or style transfer,
- it depends on fragile sampler internals without audible payoff.

Revise a method if:

- descriptors move but listening is mixed,
- flow scores improve but generated audio does not,
- the effect works on one clip but fails on another,
- the direct decode works but SA3 polish erases it,
- the result is useful but the UI/recipe is too hard to repeat.

Promote a method if:

- it survives multiple source clips, prompts, and seeds,
- descriptor changes align with listening notes,
- source preservation and novelty are separable,
- baselines are included,
- the notebook recipe is repeatable.

## Measurement Panels To Fill

Each promoted experiment should fill as many of these as relevant:

| Panel | Purpose |
|---|---|
| A/B/C audio player | Source, baseline, method output |
| Descriptor table | RMS, spectral, stereo, flux, and other audio deltas |
| Flow table | Total flow loss, normalized MSE, cosine term, conditional delta |
| Loss-by-timestep panel | Which logSNR/timestep bands moved |
| Memory-nearest rows | Distinguish source preservation from copying |
| Geometry report | PCA, Mahalanobis movement, covariance transport effects |
| Control-lane panel | Whether time-varying controls moved as intended |
| Residual atlas panel | Which layer/features moved and whether intervention was causal |
| Listening annotation | Human judgement with task-specific labels |

## Initial Ledger

No completed runs have been recorded in this reorganized ledger yet.

| Date | Mode | Question | Outputs | Decision | Notes |
|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | unknown | Add first real Colab/listening run here. |

## Open Result Questions

- Does native flow score predict generated-audio similarity, or only teacher-forced agreement?
- Which logSNR bands correspond to style, structure, transient detail, or prompt adherence?
- Does covariance transport outperform mean/std style transfer audibly?
- Which latent DSP operators survive SA3 polish?
- Can null-condition inversion preserve source identity while allowing text edits?
- Which control lanes are observable, predictable, and intervenable?
- Which residual layers produce stable causal audio controls?
- Can sampler-level guidance improve loopability without half-period collapse?
