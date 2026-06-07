# SA3 Native Lab Research-Layer Ontology

Status: current research-layer map for the notebook-first research program as
of 2026-06-06.

This document answers: which model-native research layers the SA3/SAME
architecture exposes, what each layer can be studied on its own, what only
exists in the coupled SA3-over-SAME system, and how evidence utilities review
claims without becoming a fifth research object.

## Source Grounding

Primary source facts:

- [Stable Audio 3](https://arxiv.org/abs/2605.17991) is a fast latent diffusion
  model family for variable-length generation and editing, operating over a
  semantic-acoustic autoencoder latent space and supporting inpainting and
  continuation.
- [SAME](https://arxiv.org/abs/2605.18613) is a stereo music/general-audio
  autoencoder with 4096x temporal compression, semantic regularization,
  phase-aware reconstruction losses, and released SAME-L/S variants.
- The official [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
  repo exposes text-to-audio, audio-to-audio editing, inpainting/continuation,
  and direct SAME autoencoder workflows, so local research should treat those
  interfaces as model boundaries rather than clone upstream internals.
- [SemanticAudio](https://arxiv.org/abs/2601.21402) and
  [MusicFlow](https://arxiv.org/abs/2410.20478) reinforce the modern trend:
  semantic planning, acoustic rendering, flow trajectories, and continuation
  need to be separated before claims about control are made.
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

## Four Research Layers Plus Evidence Utilities

The repo should not collapse every method into one generic workbench. The
architecture exposes four research layers. Evidence is cross-cutting lab
infrastructure: it audits every layer, but it is not itself another generative
or representation layer.

Research layer means a native object boundary in the model architecture:

```text
SAME representation object
SA3 flow/condition object
SA3 internal trajectory object
SA3-over-SAME coupled editing object
```

Evidence utility means the review surface that decides whether measurements
support a claim:

```text
audio output + latent rows + flow rows + descriptors + listening notes
-> evidence packet
-> ledger decision
```

## Research Layers

| Layer | Native objects | What can be studied alone | What counts as evidence |
|---|---|---|---|
| SAME Representation Science | waveform `x`, encoder `E`, SAME latent `z0`, decoder `D`, `LatentItem` | compression, direct decode, geometry, latent memory, source preservation, bottleneck stress, latent DSP | direct decodes, descriptor deltas, geometry rows, nearest-memory rows, control lanes, listening notes |
| SA3 Flow and Conditioning Science | prompt `p`, condition `C(p)`, flow state `z_t`, timestep/logSNR, velocity `v_theta` | prompt scoring, condition inversion, flow timestep bands, null/conditional-delta probes | shared flow-probe rows, prompt semantic rows, attribution, generated-audio audition |
| SA3 Internal Trajectory Science | residual activations `a_l`, sampler states, step windows, guidance objectives | residual feature maps, layer/time causality, sampler-state edits, guidance honesty, step/polish tradeoffs | activation rows, alpha/guidance sweeps, flow/descriptor/listening disagreement |
| SA3-over-SAME Coupled Editing | edited SAME `z0'`, SA3 polish/init-audio/inpainting path, source masks | whether SA3 preserves, repairs, erases, or rewrites SAME edits | direct decode vs SA3 polish packets, source-preservation rows, flow loss, listening |

These layers are not a linear pipeline. They are separate microscopes that
should be combined only after their failure modes are visible.

## Evidence Utilities

| Utility surface | Native inputs | Job | Promotion authority |
|---|---|---|---|
| Descriptor rows | decoded audio, baseline audio, method audio | make audible changes inspectable before listening | never alone |
| Memory/source rows | SAME latents, source/donor collections | expose copying, novelty, and neighborhood movement | only with audition |
| Control lanes | audio or latent trajectories | summarize, compare, segment, and mask time-varying structure | selector evidence; intervention only after audition |
| Notebook player and annotations | output paths, loop ranges, notes | collect listening evidence in the same run context | required for promotion |
| Disagreement rows | SAME, flow, descriptor, memory, listening lanes | surface metric conflicts instead of hiding them in a scalar | triage only |
| Manifest and ledger | run parameters, artifacts, decisions | make claims reproducible and comparable | final local authority |

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
- Evidence utilities belong across every layer; the ledger is the promotion
  authority, not a fifth model-object layer.

## Existing Coverage

| Layer | Current support | Main gap |
|---|---|---|
| SAME Representation Science | geometry, periodicity, latent DSP, blur/filter, selective renoise/graft, style profile/direction, memory | systematic bottleneck and direct-decode evidence |
| SA3 Flow and Conditioning Science | flow probe banks, prompt scoring, attribution, soft/hard/readable prompt search, null-condition scaffold | predictive validity against generated audio |
| SA3 Internal Trajectory Science | residual hooks, residual vectors, residual feature atlas, cyclic projection, guidance scaffolds | layer/time causal evidence and artifact checks |
| SA3-over-SAME Coupled Editing | SA3 polish, selective SA3, continuation/inpainting, direct decode helpers | survival matrix: what edits SA3 preserves or erases |

Evidence utilities already exist as player, descriptors, annotations,
disagreement rows, manifests, and the ledger template. The main gap is not more
infrastructure; it is the first repeated completed evidence packets.

## Expert Layer Interpretation

### SAME Representation Science

Modern audio generation is increasingly a science of bottlenecks. SAME is not
just a codec-shaped convenience layer; it is the compressed semantic-acoustic
state that decides what can be preserved, reconstructed, edited, retrieved, or
hallucinated back by a decoder or prior. The core questions are:

```text
what information is actually present in z0?
what information is only reconstructed by D or SA3?
which directions are semantic, acoustic, structural, or artifact-prone?
```

Notebook consequence: SAME experiments must start with direct decode,
bottleneck stress, memory/source checks, descriptor deltas, and listening before
SA3 polish is allowed to rescue the result.

### SA3 Flow and Conditioning Science

SA3 exposes a frozen vector field over SAME-shaped latent states. Prompt
research here is not generic text scoring. It is teacher-forced agreement
between a target latent trajectory and the velocity predicted under a condition:

```text
z_t = (1 - t) z0 + t epsilon
v_theta(z_t, t, C(p))
```

The modern question is which parts of the condition are visible to the flow
field at which timesteps, under which velocity convention, and whether that
visibility predicts audible behavior. Flow scores are microscopes/selectors
until generated outputs and listening agree.

### SA3 Internal Trajectory Science

Residuals, sampler states, and step windows are internal computational objects,
not automatically controls. This layer should borrow the discipline of
mechanistic interpretability: map layer/time features first, then test causal
patches with bounded sweeps. A residual direction is promoted only if it moves
audio predictably across prompts/seeds and does not merely exploit fragile
internals.

### SA3-over-SAME Coupled Editing

The coupled path is where most creative promise lives and where most false
claims happen. An edit to `z0` can be preserved, amplified, repaired, ignored,
or overwritten by SA3 polish/inpainting/continuation. The key scientific object
is therefore not the edit alone but the survival relation:

```text
z0 -> z0' -> direct decode
z0 -> z0' -> SA3 polish/inpaint/continue
compare: preserved / amplified / repaired / erased / invented
```

Notebook consequence: every coupled edit needs source, direct decode, plain
polish, method polish, descriptor rows, source-preservation rows, and listening.

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

- Use SAME chunks, memory rows, lane-similar retrieval, lane-continuity bridge
  search, and control-lane regions to produce segment plans.
- Compare one global prompt against segment-level prompts and bridge-selected
  continuations.

Promote if: structure improves without boundary artifacts or copying.

### 10. Research-Layer Evidence Packet

Native transition: `run artifacts -> research-layer/evidence decision`.

Implementation shape:

- Add a `research_layer` field to manifest/ledger rows.
- Let each completed run declare whether it tested SAME-only, SA3-only,
  trajectory, coupled editing, or an evidence utility.

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
10. Research-layer evidence packet
```

The first five should come before stronger residual/guidance claims. Residual
work is compatible with SA3, but it becomes meaningful only when the surrounding
SAME/flow/evidence layers are already interpretable.
