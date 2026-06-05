# Frontier Architecture Transfer

Status: source-checked multimodal research scout for SA3 Native Lab as of
2026-06-05.

This document answers: what recent audio and multimodal architecture changes
matter for this notebook, what transfers into SA3/SAME-native experiments, and
what should remain only source context.

## Scope and Source Strategy

This pass uses primary papers, official repos, and official project pages where
available. It does not claim that a source result works inside this repo until
the notebook measures it.

Evidence labels:

- `source-confirmed`: directly supported by a linked source checked on
  2026-06-05.
- `repo-inferred`: plausible from local notebook/primitives, but not run here.
- `hypothesis`: useful research direction, not yet validated locally.
- `unknown`: needs Colab execution, source inspection, or listening evidence.

## Frontier Thesis

The niche is moving toward:

```text
compressed semantic latents
+ flow / diffusion-transformer backbones
+ explicit temporal structure
+ in-context editing or conditioning
+ representation probes and preference/listening loops
```

For SA3 Native Lab, frontier methods translate best through native-object
mapping:

```text
identify the native object each frontier method manipulates
-> map it onto SAME z0, SA3 z_t, prompt condition C(p), residual activations,
   control lanes, memory rows, or listening annotations
-> test with descriptor, latent, flow, source-preservation, and listening evidence
```

## Source Map

| Source | Date | Architecture Delta | Notebook Transfer | Status |
|---|---:|---|---|---|
| [Stable Audio 3](https://arxiv.org/abs/2605.17991) | 2026-05-18 | Fast latent audio generation/editing over SAME, variable length, inpainting/continuation, adversarial post-training. | Treat SA3 flow fields, SAME latents, and sampler states as primary native objects. | source-confirmed |
| [SAME](https://arxiv.org/abs/2605.18613) | 2026-05-18 | 4096x temporal compression, transformer autoencoder, semantic regularization, phase-aware losses. | Make the notebook a science of what the SAME bottleneck preserves, exposes, and hides. | source-confirmed |
| [ACE-Step](https://arxiv.org/abs/2506.00045) | 2025-05-28 | Music foundation model combining diffusion generation, DCAE, lightweight linear transformer, semantic alignment. | Compare SA3's SAME bottleneck against other semantic/music bottleneck claims; use external outputs as `EXTERNAL_COMPARISON.cross_model` baselines. | source-confirmed |
| [ACE-Step 1.5](https://arxiv.org/abs/2602.00744) | 2026-02 | LM planner plus DiT-style rendering, long compositions, editing/personalization claims. | Use "planner vs renderer" as a notebook lens: separate prompt plans, flow scoring, acoustic polish, and listening decisions. | source-confirmed |
| [DiffRhythm 2](https://arxiv.org/abs/2510.22950) | 2025-10 | Semi-autoregressive block flow matching for long songs, low-frame-rate music VAE, preference optimization. | Transfer block/segment ideas into SA3 continuation, bridge search, and control-lane experiments. | source-confirmed |
| [YuE](https://arxiv.org/abs/2503.08638) | 2025-03 | Long-form lyrics-to-song foundation model using LLaMA-style autoregression, track decoupling, structural conditioning. | Treat structure, lyrics/prompt plan, and audio rendering as separable evidence streams; do not fold full-song LM code into this repo. | source-confirmed |
| [SongGen](https://arxiv.org/abs/2502.13128) | 2025-02 | Single-stage autoregressive transformer for controllable song generation with mixed/dual-track modes. | Add external song baselines to `EXTERNAL_COMPARISON.cross_model`; borrow dual-track evaluation ideas for source/donor comparisons. | source-confirmed |
| [MMAudio](https://github.com/hkchengrex/MMAudio) | CVPR 2025 | Multimodal joint training for video-to-audio synthesis, CLIP/synchrony features, audio VAE. | Use video-to-audio as a reminder that temporal synchrony is measurable; transfer only the measurement idea unless video inputs become a notebook source. | source-confirmed |
| [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators) | 2024-02 | Spacetime latent patches with diffusion transformer over variable durations/resolutions. | Map "patch persistence" to audio segments/control lanes/loop regions, not to visual UI metaphors. | source-confirmed |
| [Movie Gen](https://huggingface.co/papers/2410.13720) | 2024-10 | Media foundation models with synchronized audio, video editing, personalization, and video-to-audio tasks. | Use synchronized-audio evaluation as source context for temporal alignment and source-preservation panels. | source-confirmed |
| [DiT](https://arxiv.org/abs/2212.09748) | 2022-12 | Diffusion transformers scale with depth/width/token count. | Supports treating SA3's transformer/flow internals as scalable representation objects for residual probes. | source-confirmed |
| [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742) | 2025-06 | Flow-matching image generation/editing unified by in-context text/image sequence concatenation and multi-turn consistency benchmarks. | Transfer the benchmark shape: test multi-turn prompt/audio edits for preservation, drift, and consistency. | source-confirmed |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 2023-03 | One/few-step generative models and distillation-style acceleration. | Measure step-count, adversarial/post-training, and polish tradeoffs in SA3 rather than implementing a new sampler family. | source-confirmed |
| [ImageBind](https://arxiv.org/abs/2305.05665) | 2023-05 | Joint embedding across image, text, audio, depth, thermal, and IMU using paired data. | Optional external embedding lane for cross-modal retrieval/novelty checks; do not treat it as SAME-native truth. | source-confirmed |
| [T-CLAP](https://arxiv.org/abs/2404.17806) | 2024-04 | Temporal-enhanced language-audio contrastive learning for retrieval/generation tasks. | Strong candidate for external temporal descriptor/lane comparison against SAME control lanes. | source-confirmed |
| [EnCodec](https://arxiv.org/abs/2210.13438) | 2022-10 | High-fidelity neural audio compression with quantized latent space and perceptual/adversarial training. | Historical codec baseline for why SAME is not merely compression; compare bottleneck behavior, not architecture code. | source-confirmed |
| [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) | 2023 | High-fidelity RVQGAN codec, 44.1 kHz support, drop-in codec for audio language modeling. | Baseline for codec-latent thinking; external only unless used as `EXTERNAL_COMPARISON.cross_model` comparison. | source-confirmed |
| [Generative Audio Compression](https://arxiv.org/abs/2602.00648) | 2026-02 | Task-oriented ultra-low-bitrate semantic compression with powerful decoder prior. | Directly motivates SAME bottleneck stress tests: which information is waveform, semantic, structural, or prior-filled? | source-confirmed |

## Architecture Deltas

### 1. Flow/DiT Is The Shared Backbone Language

Image, video, and audio frontier systems increasingly describe generation as
transformer denoising or flow over compressed latents. SA3 already lives here,
so the local opportunity is not adding a generic diffusion abstraction. It is
measuring SA3's own vector field:

```text
z_t = (1 - t) z0 + t epsilon
v_theta(z_t, t, C(p))
```

Notebook impact:

- keep logSNR/timestep panels,
- cache probe banks,
- compare prompts/edit states under identical flow probes,
- make velocity convention explicit everywhere.

### 2. Semantic Compression Is A Research Object

SAME, ACE-Step's DCAE path, neural codecs, and generative compression sources
all converge on bottlenecks that are not transparent waveforms. The notebook
should ask:

```text
what does the bottleneck preserve?
what does it discard?
what does the decoder/prior hallucinate back?
which dimensions are controllable versus merely reconstructive?
```

Notebook impact:

- SAME bottleneck stress tests,
- direct decode versus SA3 polish comparisons,
- descriptor deltas plus nearest-memory checks,
- latent DSP promoted only after listening evidence.

### 3. Long-Form Music Needs Structure Before Sound

ACE-Step 1.5, DiffRhythm 2, YuE, and SongGen all separate structure, lyrics,
segments, tracks, or high-level plans from final acoustic synthesis in some
form. SA3 Native Lab can borrow the experimental discipline:

```text
plan/segment/control lane
-> local SA3/SAME operation
-> bridge/continuation/polish
-> structure and listening evidence
```

Notebook impact:

- segment-level continuation tasks,
- block-level prompt/flow scoring,
- bridge search with control lanes,
- seed-family atlas over longer recipes.

### 4. Temporal Coherence Is Measurable

Video models frame coherence as persistence across spacetime patches. For audio,
the analogue is not visual persistence; it is periodicity, boundary continuity,
control-lane stability, source preservation, and prompt drift over time.

Notebook impact:

- loop metrics and repeated preview remain central,
- temporal control lanes should be part of retrieval,
- multi-turn edit tests should measure drift.

### 5. In-Context Editing Suggests Multi-Turn Audio Tests

FLUX.1 Kontext is useful here because it makes multi-turn consistency an
evaluation surface. Audio equivalent:

```text
source audio
-> edit A
-> edit B
-> edit C
```

Measure whether each edit obeys the new prompt while preserving declared source
attributes.

Notebook impact:

- null-condition edit auditions,
- source-preservation panel,
- flow/descriptor/listening rows after each turn.

### 6. External Embeddings as Comparison Judges

CLAP/T-CLAP/ImageBind can help judge semantic or temporal alignment, but they
belong as optional measurement lanes beside SAME-native evidence:

```text
SAME-native evidence first
external embedding evidence second
listening decision last
```

Notebook impact:

- optional external descriptor table,
- retrieval comparison against SAME memory,
- disagreement report when external embeddings and listening diverge.

### 7. Preference Loops Should Start As Ledger Loops

Long-form music sources increasingly mention preference optimization or human
alignment. This notebook does not have enough data for RLHF. It does have
listening annotations.

Notebook impact:

- use annotations to select/promote recipes,
- keep scalar rewards provisional,
- avoid optimizing to a tiny subjective dataset.

## Cross-Modal Transfer Matrix

| Source Pattern | Native SA3/SAME Analogue | Measurement | Promote If | Caution |
|---|---|---|---|---|
| Flow-matching image edit | Prompt/audio edit over SA3 flow probes | `flow_prompt.py` rows, decoded audition | flow improvement predicts better edits | flow score may only measure teacher-forced agreement |
| Video spacetime patches | Audio segments, control lanes, loop windows | lane similarity, periodicity, boundary metrics | temporal metrics match listening | avoid visual metaphors that hide audio structure |
| In-context image editing | Multi-turn prompt/audio edit chain | source-preservation rows, descriptor drift | edits stack without identity collapse | iterative polish may erase prior edits |
| Long-form song planning | Segment prompts, bridge plans, curriculum rows | continuation/bridge scores, ledger notes | structure improves within notebook evidence packets | full song planning can become a separate project |
| Neural codec/generative compression | SAME bottleneck stress tests | direct decode, SA3 polish, descriptors | separates semantic from acoustic preservation | not all codec lessons transfer to continuous SAME latents |
| Contrastive multimodal embeddings | Optional external semantic/lane descriptors | CLAP/T-CLAP/ImageBind versus SAME memory | improves retrieval or novelty checks | external embeddings can reward wrong semantics |
| Preference optimization | Annotation-weighted recipe selection | player notes, ledger decisions | repeated decisions identify robust recipes | tiny preference sets are easy to overfit |

## Candidate Notebook Experiments

### A. SAME Bottleneck Stress Test

Object: SAME latent `z0` and decoded/polished audio.

Intervention: apply controlled degradations or projections: temporal downsample,
low-rank, channel dropout, FFT band edits, and direct versus SA3-polished decode.

Measurement: descriptor deltas, latent summary deltas, flow prompt loss, nearest
memory rows, listening notes.

Claim: identifies what SAME represents semantically versus acoustically.

Touchpoints: `SAME_REPRESENTATION.blur_bottleneck`, `SAME_REPRESENTATION.neural_dsp`, `SAME_REPRESENTATION.geometry_audit`, `SA3_FLOW_CONDITIONING.flow_attribution`, `SA3_FLOW_CONDITIONING.flow_timestep_panel`.

Promote if: perturbation families reveal stable preservation/failure patterns
across clips.

Drop or revise if: results are only loudness/artifact differences.

### B. Multi-Turn Audio Edit Consistency

Object: source audio latent, prompt condition, null/soft condition, generated
variant chain.

Intervention: run a fixed edit sequence: preserve source, change texture, change
rhythm/energy, restore source trait.

Measurement: descriptor drift, nearest-memory similarity, flow loss for each
turn, listener notes.

Claim: tests whether SA3/SAME can support in-context-like iterative audio
editing.

Touchpoints: `SA3_FLOW_CONDITIONING.soft_prompt_inversion`, `SA3_FLOW_CONDITIONING.hard_prompt_search`, `SA3_FLOW_CONDITIONING.flow_attribution`, `SA3_FLOW_CONDITIONING.flow_timestep_panel`, `SA3_FLOW_CONDITIONING.null_inversion`, `CAUSAL_STEERING.audio_posterior`.

Promote if: edits accumulate predictably without destroying declared source
identity.

Drop or revise if: each turn behaves like unrelated regeneration.

### C. Segment/Block Prompt Plan

Object: segment-level prompts, SAME chunks, bridge candidates, control lanes.

Intervention: split a longer target into segments, score prompts per segment,
then use continuation/bridge search to assemble variants.

Measurement: bridge cost, lane continuity, loop metrics, descriptor continuity,
listening notes.

Claim: imports long-form music "structure before sound" discipline without
training a song model.

Touchpoints: `DATASET_MEMORY_COMPOSITION.prompt_family`, `DATASET_MEMORY_COMPOSITION.continuation`, `DATASET_MEMORY_COMPOSITION.memory_index`, `EVIDENCE_DECISION_PROTOCOL.control_lanes`, `DATASET_MEMORY_COMPOSITION.curriculum`, `DATASET_MEMORY_COMPOSITION.bridge_search`.

Promote if: segment plans improve continuity versus one global prompt.

Drop or revise if: segment boundaries dominate artifacts.

### D. External Temporal Embedding Lane

Object: decoded audio plus SAME memory/control lanes.

Intervention: compute optional CLAP/T-CLAP/ImageBind-style embedding distances
for generated and source clips, then compare against SAME-native memory scores.

Measurement: disagreement table: SAME distance, external embedding distance,
descriptor deltas, listening tags.

Claim: external embeddings help judge semantic or temporal alignment without
becoming native truth.

Touchpoints: `DATASET_MEMORY_COMPOSITION.memory_index`, `EVIDENCE_DECISION_PROTOCOL.control_lanes`, `DATASET_MEMORY_COMPOSITION.curriculum`, `EXTERNAL_COMPARISON.cross_model`, player/ledger cells.

Promote if: external scores catch failures that SAME summaries miss.

Drop or revise if: scores reward semantically wrong but embedding-near outputs.

### E. Step/Polish Tradeoff Audit

Object: SA3 sampler outputs from identical init latents/prompts.

Intervention: sweep steps, init noise, polish strength, and any available
distillation/adversarial-post-training settings exposed by upstream SA3.

Measurement: runtime, flow loss, descriptor deltas, source preservation,
listening notes.

Claim: finds the lowest useful compute for reliable notebook audition.

Touchpoints: `SAME_REPRESENTATION.neighborhood_renoise`, `SAME_REPRESENTATION.blur_bottleneck`, `SAME_REPRESENTATION.neural_dsp`, `CAUSAL_STEERING.cyclic_trajectory`, `CAUSAL_STEERING.gradient_edit`, `CAUSAL_STEERING.audio_posterior`.

Promote if: a small settings grid gives stable quality/runtime guidance.

Drop or revise if: SA3 internals hide or confound the controls.

### F. Annotation-Weighted Recipe Selection

Object: generated artifacts, descriptor rows, player annotations.

Intervention: aggregate listening ratings/tags by method recipe, seed, prompt,
and source cluster.

Measurement: recipe-level success/failure summaries plus descriptor and
nearest-memory rows.

Claim: turns preference-learning ideas into a notebook-scale scientific ledger
without pretending to train an RLHF system.

Touchpoints: player, manifest, experiment ledger, `SAME_REPRESENTATION.neighborhood_renoise`, `SA3_FLOW_CONDITIONING.hard_prompt_search`, `SAME_REPRESENTATION.ot_style_transfer`, `CAUSAL_STEERING.gradient_edit`, `EXTERNAL_COMPARISON.cross_model`.

Promote if: repeated annotations identify robust recipes and failure families.

Drop or revise if: notes are too sparse or inconsistent to guide decisions.

## Priority Order

Use the measurement-first backlog sequence:

```text
probe-bank cache
-> novelty/source-preservation panel
-> SAME bottleneck stress test
-> multi-turn audio edit consistency
-> segment/block prompt plan
-> external temporal embedding lane
-> step/polish tradeoff audit
-> annotation-weighted recipe selection
```

The first two stabilize evidence. The next four import the best architecture
lessons. The last item waits until enough listening notes exist.

## Unknowns

- Which external embeddings are practical in Colab after SA3 dependencies load?
- Does SAME geometry correlate with semantic embeddings or only with acoustic
  reconstruction features?
- Do SA3 flow scores predict multi-turn edit success?
- Which prompt/segment plans survive SA3 polish without becoming generic?
- Does direct SAME decode reveal useful bottleneck failures that SA3 polish
  hides?

## Docs and Backlog Impact

- `source-context.md` should keep primary sources and point here for transfer
  interpretation.
- `backlog.md` should add frontier-informed candidates only when they become
  concrete notebook runs.
- `experiment-ledger.md` should receive entries only after real Colab outputs
  and listening notes exist.
- `methods-and-math.md` should only change when an experiment's math is precise
  enough to implement.
