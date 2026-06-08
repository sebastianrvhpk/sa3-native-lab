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
| SA3 Internal Trajectory Science | residual activations `a_l`, sampler states, sampler timesteps, observed windows, guidance objectives | residual feature maps, residual-timestep cartography, layer/timestep and layer/window causality, sampler-state edits, guidance honesty, step/polish tradeoffs | activation rows, trajectory maps, alpha/guidance sweeps, flow/descriptor/listening disagreement |
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
  residual a_l -> layer/timestep and layer/window feature evidence

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
| SA3 Internal Trajectory Science | residual hooks, residual vectors, residual-timestep cartography, residual feature atlas, cyclic projection, guidance scaffolds | layer/timestep and layer/window causal evidence and artifact checks |
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
mechanistic interpretability: capture activations, rank layers with explicit
linear probes, then test causal patches with bounded sweeps. A residual
direction is promoted only if it moves audio predictably across prompts/seeds
and does not merely exploit fragile internals.

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

## Bottom-Up Research Programs

The notebook now has a section named `6. Bottom-Up Research Program
Workbenches`. These are the current ten executable avenues. They emerged from
the native objects and measurements rather than from legacy mode numbering.

| Program | Layer | Native transition | Current maturity |
|---|---|---|---|
| SAME bottleneck tomography | SAME representation | `z0 -> structured perturbation -> D(z0')` | microscope |
| SA3 flow-semantic cartography | SA3 flow/conditioning | `target z0 + prompt family + shared probes -> banded flow rows` | microscope / selector |
| Coupled edit survival | SA3-over-SAME coupled editing | `edited z0' -> direct decode / plain polish / method polish` | measurement scaffold |
| Latent control system identification | SAME representation | `collection descriptors -> latent-summary probe -> maturity row` | microscope |
| Stemless source cartography | SAME representation | `source/donor z0 + mask -> donor-pull and leakage rows` | microscope |
| Melody/rhythm/timbre factor atlas | cross-layer evidence join | `SAME rows + flow rows + trajectory rows + listening notes -> factor row` | atlas scaffold |
| Long-form latent composition | SAME memory / composition | `memory clips -> continuation / bridge / path candidates` | selector |
| Prompt-condition geometry | SA3 flow/conditioning | `C(p_i), C(p_j), soft states -> condition distance rows` | microscope |
| Sampler physiology | SA3 internal trajectory | `sampler settings + observed step records -> path summary` | microscope |
| Latent constraint library | SAME intervention candidates | `constraint specs -> J(z) -> before/after rows` | high-risk intervention candidate |

### Promotion Discipline

These programs are deliberately not claims yet. Promotion still flows through
the same evidence chain:

```text
object -> baseline -> method -> measurement -> audition -> ledger decision
```

The priority is now execution quality, not adding more categories:

1. Run one small packet for SAME bottleneck tomography and coupled edit
   survival on the same source.
2. Run flow-semantic cartography and prompt-condition geometry on the same
   target.
3. Run stemless source cartography with a self-graft control before donor
   claims.
4. Run long-form composition only after a small memory set has reliable
   descriptors and listening notes.
5. Treat sampler physiology and latent constraints as microscopes until their
   outputs survive direct decode / polish evidence.
