# SA3 Native Lab Architecture Ontology

Status: current architecture-layer map for the notebook-first research program
as of 2026-06-06.

This document answers: which research layers the SA3/SAME architecture exposes,
what each layer can be studied on its own, what only exists in the coupled
SA3-over-SAME system, and which notebook implementations should come next.

## Source Grounding

Primary source facts:

- [Stable Audio 3](https://arxiv.org/abs/2605.17991) is a fast latent diffusion
  model family for variable-length generation and editing, operating over a
  semantic-acoustic autoencoder latent space and supporting inpainting and
  continuation.
- [SAME](https://arxiv.org/abs/2605.18613) is a stereo music/general-audio
  autoencoder with 4096x temporal compression, semantic regularization,
  phase-aware reconstruction losses, and released SAME-L/S variants.
- [Low-Resource Guidance for Controllable Latent Audio Diffusion](https://arxiv.org/abs/2603.04366)
  supports the idea that latent-space control heads and selective guidance can
  be worth testing when controls are measured locally.
- [Live Music Diffusion Models](https://arxiv.org/abs/2605.22717) supports
  block-wise, trajectory, and interactive-generation questions as architecture
  context, not as code to import.

Local rule:

```text
External research suggests questions.
SA3/SAME-native notebook evidence decides what survives.
```

## The Architecture Layers

The repo should not collapse every method into one generic workbench. The
architecture exposes distinct layers:

| Layer | Native objects | What can be studied alone | What counts as evidence |
|---|---|---|---|
| SAME Representation Science | waveform `x`, encoder `E`, SAME latent `z0`, decoder `D`, `LatentItem` | compression, direct decode, geometry, latent memory, source preservation, bottleneck stress, latent DSP | direct decodes, descriptor deltas, geometry rows, nearest-memory rows, control lanes, listening notes |
| SA3 Flow and Conditioning Science | prompt `p`, condition `C(p)`, flow state `z_t`, timestep/logSNR, velocity `v_theta` | prompt scoring, condition inversion, flow timestep bands, null/conditional-delta probes | shared flow-probe rows, prompt semantic rows, attribution, generated-audio audition |
| SA3 Internal Trajectory Science | residual activations `a_l`, sampler states, step windows, guidance objectives | residual feature maps, layer/time causality, sampler-state edits, guidance honesty, step/polish tradeoffs | activation rows, alpha/guidance sweeps, flow/descriptor/listening disagreement |
| SA3-over-SAME Coupled Editing | edited SAME `z0'`, SA3 polish/init-audio/inpainting path, source masks | whether SA3 preserves, repairs, erases, or rewrites SAME edits | direct decode vs SA3 polish packets, source-preservation rows, flow loss, listening |
| Evidence and Listening Science | descriptors, memory rows, player notes, manifests, ledger decisions | how claims are reviewed, compared, promoted, revised, or dropped | complete evidence packets and repeated ledger decisions |

These layers are not a linear pipeline. They are separate microscopes that can
be combined after their failure modes are visible.

## Object Transition Map

```text
SAME only:
  x -> E(x) = z0
  z0 -> D(z0) = x_hat
  z0 -> summaries / geometry / memory / control lanes
  z0 -> z0' -> D(z0')

SA3 only over latent states:
  (z_t, t, C(p)) -> v_theta(z_t, t, C(p))
  C(p) -> prompt/condition evidence
  residual a_l -> layer/time feature evidence

SA3 internal trajectory:
  residual or sampler state -> patched/optimized state
  patched state -> next latent state -> decoded audio

SA3 over SAME:
  z0 or z0' -> SA3 init/polish/inpaint/continue -> z_out
  z_out -> D(z_out) -> evidence packet

Evidence:
  output audio + latent rows + prompt rows + listening notes -> maturity decision
```

## Placement Rules

- SAME-only methods should not require SA3 unless the experiment is explicitly
  about polish/rescue/erasure.
- SA3 flow methods should not claim decoded-audio success until auditioned.
- Residual and sampler-state methods should remain microscopes or high-risk
  intervention candidates until causal sweeps survive baseline comparison.
- Coupled editing methods must always compare direct SAME decode against SA3
  polish or continuation.
- Evidence methods belong across every layer; the ledger is the promotion
  authority.

## Existing Coverage

| Layer | Current support | Main gap |
|---|---|---|
| SAME Representation Science | geometry, periodicity, latent DSP, blur/filter, selective renoise/graft, style profile/direction, memory | systematic bottleneck and direct-decode evidence |
| SA3 Flow and Conditioning Science | flow probe banks, prompt scoring, attribution, soft/hard/readable prompt search, null-condition scaffold | predictive validity against generated audio |
| SA3 Internal Trajectory Science | residual hooks, residual vectors, residual feature atlas, cyclic projection, guidance scaffolds | layer/time causal evidence and artifact checks |
| SA3-over-SAME Coupled Editing | SA3 polish, selective SA3, continuation/inpainting, direct decode helpers | survival matrix: what edits SA3 preserves or erases |
| Evidence and Listening Science | player, descriptors, annotations, disagreement rows, manifests, ledger template | first repeated completed ledger packets |

## New Research Programs

### 1. SAME Bottleneck Atlas

Native transition: `x -> z0 -> controlled bottleneck perturbation -> D(z0')`.

Implementation shape:

- Add a notebook packet that sweeps temporal downsample, low-rank projection,
  channel dropout, FFT band attenuation, blur/sharpen, and noise radius.
- For each perturbation, export direct decode, descriptors, SAME summary delta,
  nearest-memory rows, and listening notes.

Promote if: perturbation families reveal stable semantic/acoustic preservation
patterns across clips.

### 2. SAME Edit Survival Matrix

Native transition: `z0 -> z0' -> direct decode / SA3 polish`.

Implementation shape:

- For every SAME edit family, create source, direct decode, plain polish, and
  method polish.
- Add a compact matrix row: intended movement, direct audible movement, polish
  audible movement, erased/amplified/repaired.

Promote if: the matrix identifies which SAME operators are real controls and
which are only probes.

### 3. Flow Timestep Semantic Bands

Native transition: `target z0 -> shared probe bank -> prompt losses by logSNR`.

Implementation shape:

- Aggregate loss-by-timestep panels across several target clips and prompt
  categories.
- Compare logSNR bands against descriptor changes and listening tags.

Promote if: some timesteps reliably correspond to prompt adherence, texture,
  structure, source identity, or transient detail.

### 4. Prompt Condition Counterfactuals

Native transition: `prompt variants -> C(p) -> flow/loss/listening rows`.

Implementation shape:

- Compare raw prompt, semantic rewrite, readable search result, hard-search
  result, blank prompt, and deliberately wrong prompt under one probe bank.
- Keep semantic tags and conditional-delta rows.

Promote if: prompt changes are explainable by native flow evidence and audible
  behavior, not only by text plausibility.

### 5. Residual Layer-Time Atlas

Native transition: `(prompt/audio examples, timestep window, layer) -> residual
activation basis`.

Implementation shape:

- Capture residual activations by layer and sampler step window for prompt-pair
  and audio-pair contrasts.
- Summarize which layers/timesteps separate material, energy, space, and rhythm
  tags.

Promote if: layer/time maps repeat before any steering claim is made.

### 6. Residual Causal Sweep

Native transition: `residual direction -> layer/time/alpha patch -> output`.

Implementation shape:

- Sweep one prompt-derived and one audio-derived vector over layer, time window,
  and alpha.
- Review direct outputs with descriptors, flow scores, and listening notes.

Promote if: effects move monotonically or predictably and remain bounded.

### 7. SA3 Polish/Rescue Audit

Native transition: `degraded or edited z0' -> SA3 polish -> z_out`.

Implementation shape:

- Feed deliberately degraded SAME latents into SA3 polish/init-audio paths.
- Ask whether SA3 restores fidelity, erases source identity, invents content, or
  preserves intended edits.

Promote if: rescue/erasure behavior becomes predictable enough to choose polish
  settings.

### 8. Trajectory Objective Honesty Packet

Native transition: `objective recipe -> sampler/latent update -> output`.

Implementation shape:

- Represent guidance objectives as inspectable recipes: loop boundary,
  descriptor, source-preservation, profile distance, flow agreement.
- Compare objective improvement against listening and source-preservation rows.

Promote if: objective improvement tracks audible improvement better than a
  baseline.

### 9. Segment And Continuation Structure Bench

Native transition: `long source/dataset -> chunks/control lanes -> continuation
or bridge plan`.

Implementation shape:

- Use SAME chunks, memory rows, bridge search, and control lanes to produce
  segment plans.
- Compare one global prompt against segment-level prompts and bridge-selected
  continuations.

Promote if: structure improves without boundary artifacts or copying.

### 10. Architecture-Layer Evidence Packet

Native transition: `run artifacts -> architecture-layer decision`.

Implementation shape:

- Add an `architecture_layer` field to manifest/ledger rows.
- Let each completed run declare whether it tested SAME-only, SA3-only,
  trajectory, coupled editing, or evidence/listening.

Promote if: this makes run comparison clearer and prevents method categories
  from drifting.

## Priority

```text
1. SAME bottleneck atlas
2. SAME edit survival matrix
3. Flow timestep semantic bands
4. Prompt condition counterfactuals
5. SA3 polish/rescue audit
6. Residual layer-time atlas
7. Residual causal sweep
8. Trajectory objective honesty packet
9. Segment and continuation structure bench
10. Architecture-layer evidence packet
```

The first five should come before stronger residual/guidance claims. Residual
work is compatible with SA3, but it becomes meaningful only when the surrounding
SAME/flow/evidence layers are already interpretable.
