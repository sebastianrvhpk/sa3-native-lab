---
name: latent-audio-interface-architect
description: Design research interfaces for latent audio and generative music systems from actual model math, operators, artifacts, and listening workflows. Use for SA3/SAME, Stable Audio, latent diffusion or flow, audio-to-latent workflows, renoise, graft, blur, latent DSP, prompt inversion, activation steering, memory retrieval, and creative audio AI instruments.
---

# Latent Audio Interface Architect

## Overview

Use this skill to turn latent-audio research systems into truthful, playable interfaces. It treats the model, latent space, operators, descriptors, and generated artifacts as the source of the interface, then lets visual language support that structure.

Visual references are metaphors, not requirements. Preserve the research affordances first.

## Core Frame

Ground the interface in the signal path:

```text
audio x -> encoder E -> latent z
z in R^(C x T)
operator O_phi: (z, prompt c, donor z_d, mask M, noise eps, params phi) -> z'
decoder D or SA3 polish -> audio y
metadata m = {source, recipe, params, seed, prompt, metrics, notes}
```

For SA3/SAME work, the interface should expose:

- Source audio, donor audio, prompt, seed, operator, recipe, and model settings.
- Direct SAME decode versus SA3-polished result when both exist.
- Latent object shape, selected channels, time windows, masks, noise level, and polish steps.
- Listening, comparison, annotation, lineage, retrieval, and export.

## Design Workflow

1. Identify the latent object.
   - What tensor is being edited?
   - What are its dimensions, rate, and file representation?
   - Is the operation happening before SA3, inside SA3, during sampling, or after decoding?

2. Write the operator equation.
   - Every meaningful control should map to an equation or executable transform.
   - If the equation is unknown, mark the control as speculative and keep it out of core UI.

3. Choose the interaction primitive.
   - Continuous scalar: slider, knob, typed numeric input, macro wheel.
   - Discrete mode: segmented control, tabs, menu.
   - Tensor region: matrix, heatmap, channel lane, time span, mask brush.
   - Lineage or dependency: graph edge, recipe chain, branch stack.
   - Memory or retrieval: gallery, embedding map, sortable table, tag browser.

4. Define feedback.
   - The user must see what changed: waveform, latent slice, descriptor delta, recipe diff, A/B player, provenance.
   - Listening controls are first-class. Audio artifacts without playback and metadata are not usable research artifacts.

5. Decide maturity.
   - `core`: reliable and needed for most experiments.
   - `lab`: experimental but interesting.
   - `probe`: diagnostic or mechinterp tool.
   - `danger`: likely misleading or destructive; isolate it.

6. Define the instrument surface.
   - Decide whether the interface is a listening bench, latent map, sweep field, memory atlas, recipe instrument, or lineage browser.
   - Do not default to a dashboard simply because there are artifacts, jobs, readiness checks, and forms.
   - Keep runtime/readiness state subordinate to the creative/research act unless the current task is setup debugging.

## Visual Translation

Map the provided visual references to function:

- Paper texture: lab notebook, trace history, research artifact surface.
- Gradient cells: continuous latent operators or parameterized transforms.
- Organic routing lines: provenance, recipe lineage, memory reuse, or signal flow.
- Wheels: macro stochastic controls, renoise amount, channel selection, or sampler/polish intensity.
- Waveform bus: perceptual grounding and comparison axis.
- Node clusters: operator families, latent memories, or candidate families.
- Hand labels: human annotations and experiment notes.

Do not let visual motifs imply nonexistent functionality. A graph should mean real dependency, lineage, routing, or pipeline structure.

## Priority Rules

- P0: playback, source/result comparison, recipe traceability, artifact export, reproducibility.
- P1: parameter exploration, annotation, descriptor deltas, memory reuse, batch navigation.
- P2: latent maps, channel families, donor-noise families, residual-stream probes, prompt search.
- P3: paper grain, ornamental routing, advanced motion, decorative texture.

## Anti-Dashboard Rule

If the proposed layout is "artifact list + comparison cards + settings form + status cards", treat it as an engineering scaffold, not the final interface. Require a research-instrument brief before implementation.

Load `references/operator-family-map.md` when designing over SA3/SAME operators.
