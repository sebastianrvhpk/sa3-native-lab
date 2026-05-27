---
name: interaction-triage-loop
description: Evaluate and evolve experimental AI interfaces through a rigorous triage loop. Use when reviewing design references, prototypes, screenshots, interaction ideas, or UI passes for creative and research tools, especially to separate functional interaction value from visual style, prioritize changes, define tests, and decide whether to promote, revise, or discard an interface direction.
---

# Interaction Triage Loop

## Overview

Use this skill to evaluate an experimental interface without getting trapped by surface style. The loop asks what each interaction proves, what model or data capability it exposes, and what should be kept, revised, deferred, or removed.

## Triage Workflow

1. Gather evidence.
   - Screenshot, running UI, code path, data shape, user workflow, generated artifacts, and known constraints.
   - If the UI is not runnable, evaluate it as a concept and mark implementation claims as unverified.

2. Split function from visual language.
   - Function: what the user can do and what system state changes.
   - Visual language: how the action is represented, felt, grouped, and remembered.
   - Keep visual references as pressure on interaction quality, not as literal feature requirements.

3. Classify each idea.
   - P0: functional truth, trust, reproducibility, playback, data visibility.
   - P1: playability, speed, navigation, comparison, undo, annotation.
   - P2: research cognition, latent maps, lineage, probes, parameter landscapes.
   - P3: aesthetic language, motion, texture, vibe, graphic identity.
   - P4: ornament with no operational purpose.

4. Score each interaction.
   - Intended user action.
   - Underlying code/model capability.
   - Data or artifact touched.
   - Required feedback.
   - Verification method.
   - Risk of misleading the user.
   - Decision: promote, prototype, revise, defer, or delete.

5. Define the next pass.
   - Keep the next iteration small enough to verify.
   - Tie every proposed change to a workflow or experiment.
   - Include acceptance tests and explicit non-goals.

## Review Failure Modes

Call these out directly:

- The interface becomes a generic dashboard because the data model has many resource types.
- The first viewport shows infrastructure instead of the primary research act.
- Raw files or sidecars are treated as primary objects when meaningful run/artifact families exist.
- A form is used where the interaction should be a recipe, instrument, sweep, or probe.
- A visual metaphor implies routing, memory, or control that does not exist.
- A graph UI exists without graph semantics.
- A beautiful screen has no artifact provenance.
- A generated result cannot be compared to its source.
- A control has no parameter, equation, or code path.
- The interface hides uncertainty instead of marking experimental status.
- The prototype optimizes for portfolio screenshots before research usability.

## Required Output

When this skill is used, produce:

- Triage table.
- Kept, revised, deferred, and deleted ideas.
- Next-pass implementation scope.
- Acceptance tests.
- Open risks.

Also state whether the next step should be implementation or a research-instrument brief. If the central surface and screen-state model are missing, do not recommend implementation.

Use `references/interface-review-rubric.md` for a stricter rubric.
