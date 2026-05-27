---
name: research-instrument-interface-director
description: Define the interaction grammar, screen-state model, and prototype plan for exploratory AI research instruments before implementation. Use when designing interfaces for complex research systems, generative models, latent spaces, creative AI tools, audio/video model labs, artifact browsers, generation consoles, or experiment workbenches, especially when Codex must avoid generic dashboards, fake DAWs, fake node graphs, and premature product UI.
---

# Research Instrument Interface Director

## Overview

Use this skill before writing React or app code. Its job is to decide what kind of instrument the interface is, what the central interaction surface is, and how model/artifact complexity becomes playable without collapsing into a dashboard.

Do not implement UI from a capability map alone. Produce an interaction grammar first.

## Required Sequence

1. Name the instrument.
   - State the primary research act in one sentence.
   - Examples: listen across generated families, steer latent perturbations, compare recipe descendants, map channel behavior.

2. Define the central surface.
   - Choose one primary surface, not a dashboard grid.
   - Valid examples: specimen board, listening bench, latent map, recipe wheel, sweep field, memory atlas, lineage strip.
   - Invalid default: left list + center cards + right form/status panel unless justified by a stronger interaction model.

3. Define the interaction grammar.
   - Objects: artifact, run, recipe, latent, donor, source, variant, note, descriptor, job.
   - Gestures: select, send to compare, generate, fork, annotate, reuse, inspect, sweep, pin, group.
   - Feedback: audio, waveform, latent shape, recipe diff, provenance, status, note, metric delta.

4. Define screen states.
   - Empty state with no artifacts.
   - Loaded corpus.
   - Selected artifact.
   - Running generation.
   - Failed runtime.
   - Successful generated family.
   - Annotated/reuse-ready artifact.

5. Define visual rules from function.
   - What colors mean.
   - What shapes mean.
   - What lines mean.
   - What is always visible.
   - What is hidden until expanded.

6. Only then create an implementation scope.
   - State which components are needed.
   - State which components are forbidden for the slice.
   - State acceptance tests.

## Anti-Dashboard Rules

Do not allow these patterns unless explicitly justified:

- A three-column admin layout as the primary design.
- Raw file lists as the first object of attention.
- Status cards that dominate the creative task.
- Forms that look like settings pages instead of recipes, instruments, or probes.
- Card grids that hide lineage, listening, or artifact relationships.
- Decorative glyphs added after the layout is already generic.
- Node graphs without real graph semantics.
- DAW timelines when the workflow is generation and artifact exploration.

## Required Deliverable

Before implementation, produce:

- Instrument thesis.
- Central surface choice.
- Interaction grammar.
- Screen-state model.
- Visual semantics.
- Component map.
- Explicit non-goals.
- Acceptance tests.

Use `references/instrument-brief-template.md` for the structured brief.
