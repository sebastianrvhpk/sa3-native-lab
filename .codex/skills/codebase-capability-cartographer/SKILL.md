---
name: codebase-capability-cartographer
description: Inspect and map a codebase into interface-ready capabilities before designing or building UI. Use when Codex needs to understand an existing repo, research notebook, model pipeline, API, data model, experiment inventory, artifact format, tests, or I/O surface before proposing interfaces, app architecture, product concepts, or implementation plans.
---

# Codebase Capability Cartographer

## Overview

Use this skill before designing an interface over an existing technical system. Its job is to convert code into an evidence-backed capability map: what the system actually does, what objects it manipulates, what parameters matter, what artifacts are produced, and what interface affordances are justified.

Do not propose a UI first. Read the codebase until the interface can be grounded in real execution paths.

## Workflow

1. Establish the repo boundary.
   - Inspect `README`, docs, notebooks, scripts, config files, package metadata, tests, examples, and recent git history.
   - Identify primary runtimes: notebook, CLI, web app, service, library, local scripts, model checkpoints, external repos.
   - Mark anything outside the repo as external dependency, not confirmed local capability.

2. Inventory execution surfaces.
   - List user-facing entrypoints: notebooks, cells, commands, APIs, scripts, UI routes, tests, demo files.
   - For each entrypoint, record required inputs, produced outputs, expected runtime, hardware assumptions, and failure modes.
   - If a path cannot be run, say why and keep it in the map as unverified.

3. Extract domain objects.
   - Name the real objects the code moves around: audio files, latents, prompts, model handles, descriptors, memories, metrics, manifests, recipes, checkpoints, UI state.
   - For each object, record shape, file format, owner module, lifecycle, persistence path, and provenance metadata.
   - Prefer code-confirmed names and schemas over invented product language.

4. Build capability cards.
   - A capability is a real operation the system can perform, such as encode, decode, renoise, graft, polish, prompt-search, annotate, retrieve, export, or compare.
   - Each card must include evidence, I/O, parameters, artifacts, known constraints, and interface affordances.
   - Separate confirmed behavior from inferred behavior and unknown behavior.

5. Derive interface affordances.
   - Translate capabilities into interaction primitives only after the map exists.
   - Ask what gesture exposes the operation truthfully: slider, matrix, graph edge, batch table, transport, timeline, latent map, memory browser, inspector, diff view.
   - Do not create decorative controls that are not backed by a code path or a plausible near-term implementation path.

6. Produce a staged backlog.
   - P0: required to operate and trust the system.
   - P1: makes exploration faster or more legible.
   - P2: expands research cognition or creative play.
   - P3: visual atmosphere and polish.

## Evidence Labels

Use these labels whenever making claims:

- `confirmed`: directly observed in files, tests, docs, command output, or executed behavior.
- `repo-inferred`: strongly implied by local code but not executed.
- `paper-inferred`: supported by cited literature or model docs but not confirmed in this repo.
- `hypothesis`: plausible research or design direction.
- `unknown`: cannot be determined from available evidence.

## Required Output

When this skill is used, produce:

- Capability map.
- I/O and artifact graph.
- Parameter and control inventory.
- Runtime and dependency assumptions.
- Interface affordance backlog.
- Unknowns and verification plan.

Use `references/capability-map-template.md` when a structured template is useful.
