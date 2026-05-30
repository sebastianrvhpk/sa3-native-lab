# Visual Design Grammar

This document translates the visual references into app semantics for SA3
Native Lab. It should be read together with
`docs/product-rescue-brief.md`, which is now the product authority for the
interface rescue.

The goal is not to copy the images literally. The goal is to make the local
instrument feel tactile, playful, and expressive while every strong visual motif
still maps to real app state.

## Reference Reading

The references combine three useful ideas:

- A tactile lab surface: matte paper, visible grain, fold marks, pencil lines,
  small labels, and sketch-like boundaries.
- Modular instruments: dense gradient cells, transport wheels, timeline bars,
  waveform lanes, knobs, and clustered controls.
- Organic flow: thin grey tendrils, nodes, and routing-like curves that imply
  signal movement, provenance, memory, or dependency.

The danger is that the references can easily become decoration. In this app,
flow lines should not imply routing unless the data exists. Dense cells should
not hide missing parameters. Wheels should not appear unless they control or
play something real.

## Motif To Function

| Motif | App meaning | Current allowed use |
| --- | --- | --- |
| Paper grain and folds | Research notebook / local lab surface | Global page background and panel texture. |
| Gradient cells | Parameterized controls, operators, presets, bundle affordances | Recipe fields, operator controls, prompt-search tools, bundle workflow chips. |
| Dark bordered modules | Executable surfaces or inspectable material groups | Tune drawer, specimen surface, audition stack, branches, Memory shelf. |
| Thin grey lines | Real lineage, source relationships, or low-priority lab-surface flow | Source -> gesture -> take -> memory/branch relationships; subtle non-interactive workbench background only. |
| Nodes / dots | Source/take/decision branch points | Data-backed lineage rows, decision memory, prompt history, branch points, anchors. |
| Transport wheel | Playback or macro execution control | AudioDeck play button only for now. |
| Rainbow waveform lane | Perceptual listening axis | AudioDeck WaveSurfer surface, persisted markers, draggable loop region, and compact audition playback. |
| Grid clusters | Mode/gesture families or dense parameter sets | RecipeFields, Tune controls, prompt search presets. |
| Hand labels | Human annotation and experiment labels | Eyebrows, rail heads, chips, small metadata labels. |

## Visual Semantics

- `green`: available/runnable, framed playback controls, positive decision.
- `teal/cyan`: audio flow, listening, runtime-ready state.
- `orange/amber`: experimental/probe or cost-warning state.
- `rose`: rejected, danger, or high-cost/queued state.
- `violet/blue`: latent/model-space, scorer/probe, bundle evidence.
- Soft gradients are allowed when they encode family, state, or interaction
  density. Flat color should be rare.

## Layout Rules

- The first viewport should teach the actual instrument: Current Sound,
  Gestures, Pending Takes, Branch / Remember / Tune, and Memory.
- The current sound and playback surface remain the main focal objects.
- Runtime/readiness state should stay available but visually subordinate.
- Dense modules are acceptable when controls remain readable and stable.
- Flow lines remain faint unless they correspond to real lineage or dependency.

## Interaction Rules

- Prompt-search preset/vocabulary/axis cells write directly to recipe params.
- Take rows select real audio material and can be remembered, anchored, reused,
  or opened as the current sound.
- Bundle action cells should only appear when the bundle contains reusable data.
- Current lineage threads must only render nodes that exist in app state:
  source material, producing gesture/job, current sound, branch, or memory
  assignment.
- Future drag/routing interactions must create or inspect real recipe/source
  relationships, not just move decorative nodes.

## Anti-Copy Rules

- Do not add fake node graphs before graph semantics exist.
- Do not add decorative lines that look clickable or imply missing routing.
- Do not turn every panel into a colorful tile; repeated colors must mean
  operation family, status, or object kind.
- Do not let paper texture reduce contrast or readability.
- Do not use large wheels for non-playback controls until macro parameters are
  real.

## Stack Assessment

The current stack is enough for the first visual pass:

- CSS variables and pseudo layers for paper, watercolor gradients, and subtle
  flow fields.
- React component state for real selected sounds, prompt history, decisions,
  recipes, takes, branches, memory, and anchors.
- Lucide icons for symbolic controls.

Near-horizon additions should remain conditional:

- Motion for React: causal transitions for queued/running/produced, sound
  remembered, branch opened, and take selected.
- wavesurfer.js: promoted for real listening work: waveform zoom, persisted
  markers, and editable loop regions.
- React Flow: only when lineage edges become an interactive graph of real
  recipe/source/output relationships.

## Acceptance Criteria

- The app feels more tactile and alive without hiding parameter access.
- Text remains readable on desktop and mobile.
- Flow styling does not imply unavailable routing.
- Play and audition controls become more visually central.
- Prompt, latent, and recipe controls feel like gesture-specific Tune controls
  rather than generic forms.
- Strong visual routes can be audited against actual source/gesture/take/branch
  data.
