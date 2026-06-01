# Visual Design Grammar

This document translates the visual references into app semantics for SA3
Native Lab. It should be read together with
`docs/product-rescue-brief.md`, which is now the product authority for the
interface rescue.

The goal is not to copy the images literally, and it is also not to decorate
the old dashboard/card system. The references point toward a structured sound
machine: track lanes, transport, modular gesture controls, material bays, and
signal evidence. Every strong visual motif still has to map to real app state.

## Reference Reading

The references combine five useful ideas:

- A tactile lab surface: matte paper, visible grain, fold marks, pencil lines,
  small labels, and sketch-like boundaries.
- Tape and timeline structure: long horizontal bars, playheads, track rows,
  lanes, and transport.
- Modular instruments: dense gradient cells, transport wheels, knobs, and
  clustered controls.
- Material bays: small clips, pads, track labels, memory/source tiles, and
  compact browser cells.
- Organic flow: thin grey tendrils, nodes, and routing-like curves, but only
  where they imply signal movement, provenance, memory, or dependency.

The danger is that the references can easily become decoration. In this app,
flow lines should not imply routing unless the data exists. Dense cells should
not hide missing parameters. Wheels should not appear unless they control or
play something real.

The current structural reset now interprets the app as one composed sound
instrument surface rather than a left/center/right workbench. Current Sound is a
wide playback lane with the Material Bay physically attached to it. A real loop
strip summarizes Current Sound -> Gesture -> Pending -> Takes -> Branches ->
Memory from current app state. The Take Field sits under the sound bench, Tune
is a side control bank, and Inspect/Evidence remains a subordinate dock.

## Motif To Function

| Motif | App meaning | Current allowed use |
| --- | --- | --- |
| Paper grain and folds | Research notebook / local lab surface | Global page background and panel texture. |
| Timeline bars | Playable sound, generated take, branch lane, or source clip | Current Sound player, compact take playback, source/memory audio clips. |
| Transport wheel | Playback or macro execution control | AudioDeck play button only for now. |
| Gradient cells | Parameterized controls, operators, presets, bundle affordances | Gesture modules, Tune fields, operator controls, prompt-search tools, bundle workflow chips. |
| Dark bordered modules | Executable surfaces or inspectable material groups | Current Sound lane, Tune bank, take queue, branches, Memory shelf. |
| Material clips | Reusable sources, remembered material, branch takes | Sources rail, Memory browser, take queue, branch detail. |
| Thin grey lines | Real lineage, source relationships, or low-priority lab-surface flow | Source -> gesture -> take -> memory/branch relationships; subtle non-interactive workbench background only. |
| Nodes / dots | Source/take/decision branch points | Data-backed lineage rows, decision memory, prompt history, branch points, anchors. |
| Loop strip cells | Current product-loop state | Counts and labels derived from current sound, active gesture, pending takes, takes, branches, and memory. |
| Rainbow waveform lane | Perceptual listening axis | AudioDeck WaveSurfer surface, persisted markers, draggable loop region, and compact audition playback. |
| Grid clusters | Mode/gesture families or dense parameter sets | RecipeFields, Tune controls, prompt search presets. |
| Hand labels | Human annotation and experiment labels | Eyebrows, rail heads, chips, small metadata labels. |

## Visual Semantics

- `neon pink`: selected/generated audio energy, current take emphasis.
- `cyan/teal`: audio flow, listening, runtime-ready state.
- `yellow/chartreuse`: runnable/keeper/ready material.
- `orange/amber`: experimental/probe, pending, cost warning.
- `rose`: failed, rejected, danger, or high-cost queued state.
- `violet/blue`: latent/model-space, prompt/vector/bundle evidence.
- Soft gradients are allowed only when they encode sound energy, object kind,
  operation family, state, or interaction density.

## Layout Rules

- The first viewport should read as an instrument, not a dashboard: Current
  Sound and Material Bay form the main bench; Tune is a side bank; gesture
  modules attach directly below; the Take Field begins beneath the bench.
- The current sound and playback lane remain the main focal objects.
- Takes should read as lane strips inside the Take Field rather than a right
  rail of cards.
- Sources and Memory should read as usable material bays rather than archive
  rows. Source/Memory can appear as compact lists, but only inside material
  zones.
- Tune should read as a compact control bank while preserving exact submitted
  values.
- Runtime/readiness state should stay available but visually subordinate.
- Dense modules are acceptable when controls remain readable and stable.
- Flow lines remain faint unless they correspond to real lineage or dependency.

## Interaction Rules

- Prompt-search preset/vocabulary/axis cells write directly to recipe params.
- Take rows select real audio material and can be remembered, anchored, reused,
  or opened as the current sound.
- Queue autoplay is a transport behavior over the visible take order, not a
  playlist product.
- Bundle action cells should only appear when the bundle contains reusable data.
- Current lineage threads must only render nodes that exist in app state:
  source material, producing gesture/job, current sound, branch, or memory
  assignment.
- Future drag/routing interactions must create or inspect real recipe/source
  relationships, not just move decorative nodes.
- Mobile order is listening-first: Current Sound, material, gestures, Tune,
  Take Field, Memory, then Evidence.

## Anti-Copy Rules

- Do not add fake node graphs before graph semantics exist.
- Do not add decorative lines that look clickable or imply missing routing.
- Do not turn every panel into a colorful tile; repeated colors must mean
  operation family, status, or object kind.
- Do not preserve the three-column dashboard/card grammar when it fights the
  sound-instrument loop.
- Do not turn lane visuals into fake waveform editing, playlist export, or
  unsupported time-region masks.
- Do not let paper texture reduce contrast or readability.
- Do not use large wheels for non-playback controls until macro parameters are
  real.

## Stack Assessment

The current stack is enough for the structural instrument surface:

- CSS variables and pseudo layers for paper, track grids, neon lane bars,
  module banks, and transport surfaces.
- React component state for real selected sounds, prompt history, decisions,
  recipes, takes, branches, memory, and anchors.
- `SoundInstrumentSurface` as the product composition frame, with `App.tsx`
  kept as the query/mutation composition root.
- `instrumentFrameModel` as a pure loop-strip model so patch-like cells are
  tied to real counts rather than decorative routing.
- Lucide icons for symbolic controls.
- Existing WaveSurfer and AudioDeck playback state for the only large transport
  wheel currently allowed.

Near-horizon additions should remain conditional:

- Motion for React: causal transitions for queued/running/produced, sound
  remembered, branch opened, and take selected.
- wavesurfer.js: promoted for real listening work: waveform zoom, persisted
  markers, and editable loop regions.
- React Flow: only when lineage edges become an interactive graph of real
  recipe/source/output relationships.

## Acceptance Criteria

- The first screenshot reads as a sound instrument: Current Sound and Material
  Bay are composed as one bench, Tune is a side bank, gesture modules are
  attached to the sound, and Take Field is a connected lane surface.
- Text remains readable on desktop and mobile.
- Flow styling does not imply unavailable routing.
- Play and audition controls become more visually central.
- Prompt, latent, and recipe controls feel like gesture-specific Tune controls
  rather than generic forms.
- Strong visual routes can be audited against actual source/gesture/take/branch
  data.
- The app no longer reads as a pastel dashboard made of interchangeable cards.
