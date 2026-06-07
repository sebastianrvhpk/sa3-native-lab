# SA3 Native Lab Source Context

Status: source context for notebook-first SA3 Native Lab research as of
2026-06-06.

This document answers: which external papers/repos shaped the notebook, what
idea each source contributes, how that idea affects the local research, and
whether it is source context, implemented, or still a hypothesis.

These notes are a curated research map, not primary sources. The linked papers
and repositories are the sources.

For a source-checked current architecture scout and translation into concrete
SA3/SAME notebook experiments, see
[Frontier architecture transfer](frontier-architecture-transfer.md).
For the local research layers those sources must land in, see
[Architecture ontology](architecture-ontology.md).

## 2026-06-06 Source Re-Audit Lens

This pass re-checked primary arXiv pages, official repos, official project
pages, and model/dataset pages where available. The cleaner local repo makes the
source-to-notebook mapping more direct:

| Source pressure | What it means locally | Notebook landing |
|---|---|---|
| SA3/SAME release | frozen SA3 flow and SAME latents are the native objects | runtime boundary, SAME representation, SA3 flow/conditioning |
| Semantic latent planning | several sources separate semantic/event layout from acoustic rendering | SAME bottleneck stress tests, prompt semantic audits, flow prompt panels |
| Time-varying controls | audio control papers emphasize loudness, pitch, event rolls, structure, and local edits | control lanes, residual/trajectory probes, source-preservation packets |
| Long-form music | newer systems split lyrics, structure, segments, tracks, or blocks from synthesis | memory/composition, bridge search, segment prompt plans |
| Evaluation and preference data | music evaluation is moving toward detailed listening preference records | player annotations, evidence ledger, promote/revise/drop discipline |
| Semantic codecs/tokenizers | audio-tokenizer work is converging on semantic/acoustic disentanglement | compare against SAME, but do not replace SAME as the local object |

The practical rule is unchanged: external sources motivate experiments; local
claims require notebook artifacts, descriptor rows, memory/source checks, and
listening notes.

The source map now lands into four research layers: SAME representation, SA3
flow/conditioning, SA3 internal trajectory, and SA3-over-SAME coupled editing.
Evidence utilities then review results across those layers; they should not be
mistaken for another model-object layer.

## Primary SA3/SAME Sources

### Stable Audio 3 Paper

- Source: [Stable Audio 3 paper](https://arxiv.org/abs/2605.17991)
- Relevant idea: SA3 is a family of fast latent diffusion/flow audio models for variable-length generation and editing, with inpainting/continuation and a semantic-acoustic autoencoder.
- Notebook impact: Treat SAME latents and SA3 flow fields as the native objects of study rather than generic audio embeddings.
- Status: source context; local notebook methods are built around the released SA3 runtime.

### SAME: Semantically-Aligned Music Autoencoder

- Source: [SAME paper](https://arxiv.org/abs/2605.18613)
- Relevant idea: SAME reaches high temporal compression for stereo music/general audio while preserving fidelity and encouraging semantic latent structure.
- Notebook impact: SAME latents are the primary space for memory, DSP, style profiles, geometry, control lanes, and source/donor edits.
- Status: source context; local operators are hypotheses about what SAME latents expose.

### Stable Audio 3 Official Repo

- Source: [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
- Relevant idea: Released SA3/SAME code provides model loading, generation, audio-to-audio, inpainting, and latent interfaces.
- Notebook impact: The Colab notebook clones and installs the upstream repo rather than vendoring it. Local adapters stay thin.
- Status: external runtime dependency.

### Stability AI Stable Audio 3 Research Article

- Source: [Stability AI Stable Audio 3 article](https://stability.ai/research/stable-audio-3)
- Relevant idea: High-level framing for released SA3 capabilities and model family.
- Notebook impact: Supports the notebook's focus on native SA3/SAME operations, editing, and generation surfaces.
- Status: source context.

### Stable Audio Open

- Source: [Stable Audio Open paper](https://arxiv.org/abs/2407.14358)
- Relevant idea: Earlier open latent audio generation context from Stability.
- Notebook impact: Background for latent audio generation before SA3/SAME-specific methods.
- Status: background source context.

### Fast Timing-Conditioned Latent Audio Diffusion

- Source: [Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825)
- Relevant idea: Timing conditioning and latent diffusion for audio generation.
- Notebook impact: Supports attention to duration, timing, continuation, and time-varying controls.
- Status: background source context.

## Underfit and Adapter Training Context

### Underfit

- Source: [dada-bots/underfit](https://github.com/dada-bots/underfit)
- Relevant idea: MIT-licensed SA3 LoRA trainer emphasizing pre-encoded latents, metadata/prompt composition, DoRA defaults, short crops, frequent demos/checkpoints, loss-by-timestep monitoring, checkpoint audition, and stopping near a creative underfit elbow.
- Notebook impact: Underfit is the LoRA path. This repo imports Underfit audio/checkpoint artifacts for descriptor, memory, player, and annotation comparison.
- Status: external tool and comparison workflow.

### LoRA

- Source: [LoRA](https://arxiv.org/abs/2106.09685)
- Relevant idea: Freeze base weights and inject trainable low-rank updates.
- Notebook impact: Provides adapter-training context for interpreting Underfit results, but this repo does not implement LoRA training.
- Status: source context.

### DoRA

- Source: [DoRA](https://arxiv.org/abs/2402.09353)
- Relevant idea: Decompose pretrained weights into magnitude and direction, using LoRA for direction.
- Notebook impact: Helps explain Underfit's adapter choices and why checkpoint audition matters.
- Status: source context.

### BoRA

- Source: [BoRA](https://arxiv.org/abs/2412.06441)
- Relevant idea: Symmetric weight decomposition over row and column magnitudes.
- Notebook impact: Adapter-method context only. Any training remains external to this notebook repo.
- Status: source context.

### LoRA-XS

- Source: [LoRA-XS](https://arxiv.org/abs/2405.17604)
- Relevant idea: Freeze SVD-derived low-rank bases and train a smaller core.
- Notebook impact: Adapter-efficiency context for future Underfit comparisons.
- Status: source context.

## Flow, Prompt Scoring, and Inversion

### Flow Matching

- Source: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Relevant idea: Train vector fields along probability paths.
- Notebook impact: Supports the teacher-forced prompt score:

```text
z_t = (1 - t) z0 + t epsilon
u_t = epsilon - z0
L_flow(prompt) = E ||v_theta(z_t, t, C(prompt)) - u_t||^2
```

- Status: source context; implemented locally through `flow_prompt.py`,
  `procedures/flow_scoring.py`, and SA3 flow/conditioning science.

### EDM

- Source: [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- Relevant idea: Noise schedules, preconditioning, and design-space clarity for diffusion sampling.
- Notebook impact: Encourages explicit timestep/logSNR handling and diagnostic loss-by-timestep panels.
- Status: source context.

### Classifier-Free Guidance

- Source: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- Relevant idea: Conditional/unconditional branch combination for prompt guidance.
- Notebook impact: Motivates explicit null/blank conditioning probes in SA3 flow/conditioning science.
- Status: source context; local null-condition work is a probe.

### Textual Inversion

- Source: [Textual Inversion](https://arxiv.org/abs/2208.01618)
- Relevant idea: Learn pseudo-token embeddings in a frozen model's conditioning space.
- Notebook impact: soft prompt inversion is the audio-native analogue over SA3 conditioning.
- Status: source context; local soft prompt optimization is implemented.

### DreamBooth

- Source: [DreamBooth](https://arxiv.org/abs/2208.12242)
- Relevant idea: Personalization by adapting model behavior to a subject.
- Notebook impact: Local personalization work routes through frozen-model probes or external Underfit fine-tuning artifacts.
- Status: source context.

### Prompt-to-Prompt

- Source: [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)
- Relevant idea: Prompt edits can control generated content through model internals/attention.
- Notebook impact: Supports prompt microscope, token attribution, and prompt replacement panels.
- Status: source context; SA3 flow/conditioning science implements flow-based prompt attribution.

### Null-Text Inversion

- Source: [Null-text Inversion](https://arxiv.org/abs/2211.09794)
- Relevant idea: Optimize unconditional/null conditioning while preserving a conditional prompt for editing.
- Notebook impact: motivates null-condition inversion: optimize SA3 null/unconditional conditioning against a target audio flow while keeping the human prompt editable.
- Status: source context; local method is scaffold/probe.

### In-Context Prompt Editing For Conditional Audio Generation

- Source: [In-Context Prompt Editing](https://arxiv.org/abs/2311.00895)
- Relevant idea: Retrieval-edited prompts can reduce distribution shift between free-form user language and model training captions.
- Notebook impact: Supports a prompt semantic transparency packet: raw prompt, retrieved/rewrite variants, flow-ranked prompts, generated audio, and listening notes.
- Status: source context; local transfer is a concrete backlog candidate.

### MusicFlow

- Source: [MusicFlow](https://arxiv.org/abs/2410.20478)
- Relevant idea: Cascaded flow matching over semantic and acoustic features, with masked prediction for infilling and continuation.
- Notebook impact: Reinforces the local split between prompt/semantic planning, SA3 flow scoring, SAME latent edits, and acoustic polish.
- Status: source context; local transfer belongs in segment prompt plans and direct-decode versus polish audits.

### SemanticAudio

- Source: [SemanticAudio](https://arxiv.org/abs/2601.21402)
- Relevant idea: A two-stage flow-matching system first generates compact semantic/event features, then synthesizes acoustic latents; it also edits by steering semantic trajectories from source/target text velocity fields.
- Notebook impact: Strongly supports treating SA3 flow loss, SAME summaries, and prompt variants as separate evidence lanes: semantic agreement is not the same as acoustic preservation.
- Status: source context; local transfer is prompt semantic audit plus flow-score predictive validity.

## Guidance, Inverse Problems, and Control

### Diffusion Posterior Sampling

- Source: [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)
- Relevant idea: Inject measurement consistency into diffusion sampling.
- Notebook impact: Supports audio-to-audio posterior guidance and source-preservation losses.
- Status: source context; local audio posterior guidance is a residual/trajectory scaffold.

### FreeDoM

- Source: [FreeDoM](https://arxiv.org/abs/2303.09833)
- Relevant idea: Use frozen generative priors with optimization/guidance for downstream constraints.
- Notebook impact: Supports guidance-gradient latent edits and measurement-driven sampler probes.
- Status: source context; local guidance is scaffold.

### Universal Guidance

- Source: [Universal Guidance for Diffusion Models](https://arxiv.org/abs/2302.07121)
- Relevant idea: Generic differentiable guidance over diffusion models.
- Notebook impact: Supports the local `gradient_guidance_step` primitive and future sampler integration.
- Status: source context.

### ControlNet

- Source: [ControlNet](https://arxiv.org/abs/2302.05543)
- Relevant idea: Train side networks for spatial/structured conditioning.
- Notebook impact: Serves as a caution: build measurable control lanes and sidecar probes before training heavy control networks.
- Status: source context; this repo currently avoids local ControlNet-style training.

### Controllable Music Production with Diffusion Models and Guidance Gradients

- Source: [Controllable Music Production with Diffusion Models and Guidance Gradients](https://arxiv.org/abs/2311.00613)
- Relevant idea: Use differentiable music/audio controls during generation.
- Notebook impact: supports gradient-edit probes and objective mixers in the residual and trajectory bench.
- Status: source context; local guidance is scaffold.

### Music ControlNet

- Source: [Music ControlNet](https://arxiv.org/abs/2311.07069)
- Relevant idea: Time-structured conditioning for music generation.
- Notebook impact: Supports the need for time-varying control lanes instead of only text prompts.
- Status: source context.

### MusicMagus

- Source: [MusicMagus](https://arxiv.org/abs/2402.06178)
- Relevant idea: Music editing/control through generative models.
- Notebook impact: Background for structured editing and prompt/control evaluation.
- Status: source context.

### InstructME

- Source: [InstructME](https://arxiv.org/abs/2308.14360)
- Relevant idea: Instruction-guided music editing.
- Notebook impact: Background for editing tasks and evaluation framing.
- Status: source context.

### LiLAC

- Source: [LiLAC](https://arxiv.org/abs/2506.11476)
- Relevant idea: Time-varying control context for audio/music generation.
- Notebook impact: Supports control-lane measurement before training heavier control models.
- Status: source context.

### Audio ControlNet

- Source: [Audio ControlNet](https://arxiv.org/abs/2602.04680)
- Relevant idea: ControlNet-like conditioning for audio.
- Notebook impact: Reinforces the control-lane path: observe, retrieve, guide, and only later train if needed.
- Status: source context.

### MusRec

- Source: [MusRec](https://arxiv.org/abs/2511.04376)
- Relevant idea: Music recommendation/generation context with structured audio features.
- Notebook impact: Background for dataset memory, retrieval, and curricula.
- Status: source context.

### FluxMusic

- Source: [FluxMusic](https://arxiv.org/abs/2409.00587)
- Relevant idea: Music generation/editing context.
- Notebook impact: Baseline context for cross-model comparisons.
- Status: source context.

### TangoFlux

- Source: [TangoFlux](https://arxiv.org/abs/2412.21037)
- Relevant idea: Fast text-to-audio flow matching and efficient generation context.
- Notebook impact: Background for runtime and flow-matching comparisons only; local prompt evidence remains frozen-SA3 flow loss plus listening.
- Status: source context; local transfer is runtime/evidence discipline, not preference training.

## Audio Generation Baselines

### AudioLDM

- Source: [AudioLDM](https://arxiv.org/abs/2301.12503)
- Relevant idea: Text-to-audio latent diffusion baseline.
- Notebook impact: cross-model baseline context for the external comparison bench.
- Status: source context.

### TANGO

- Source: [TANGO](https://arxiv.org/abs/2304.13731)
- Relevant idea: Text-to-audio generation baseline.
- Notebook impact: Cross-model baseline context for fixed task reports.
- Status: source context.

### MusicGen / AudioCraft

- Sources:
  - [MusicGen](https://arxiv.org/abs/2306.05284)
  - [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
- Relevant idea: Strong autoregressive/audio-token baseline family for music generation.
- Notebook impact: the external comparison bench can compare SA3 outputs against external model commands when available.
- Status: source context; no local model management.

### ACE-Step and ACE-Step 1.5

- Sources:
  - [ACE-Step](https://arxiv.org/abs/2506.00045)
  - [ACE-Step 1.5](https://arxiv.org/abs/2602.00744)
- Relevant idea: Open music foundation model work combining planning, diffusion-transformer rendering, semantic alignment, long compositions, and editing/personalization claims.
- Notebook impact: source context for separating prompt/structure planning from acoustic rendering and external comparison packets.
- Status: source context; no local model management.

### DiffRhythm 2

- Source: [DiffRhythm 2](https://arxiv.org/abs/2510.22950)
- Relevant idea: Semi-autoregressive block flow matching for long, controllable song generation.
- Notebook impact: Supports segment/block prompt plans, bridge search, and control-lane continuity experiments.
- Status: source context; local transfer is a hypothesis.

### YuE

- Source: [YuE](https://arxiv.org/abs/2503.08638)
- Relevant idea: Long-form lyrics-to-song foundation model using LLaMA-style autoregression, track decoupling, structural conditioning, and music representations.
- Notebook impact: Source context for treating long-form structure, source/reference preservation, and acoustic rendering as separate evidence streams.
- Status: source context; no local model management.

### SongGen

- Source: [SongGen](https://arxiv.org/abs/2502.13128)
- Relevant idea: Single-stage autoregressive text-to-song model with controllable attributes and mixed/dual-track generation modes.
- Notebook impact: external baseline context for the external comparison bench and dual-track/source-donor evaluation ideas.
- Status: source context.

### AudioX

- Sources:
  - [AudioX paper](https://arxiv.org/abs/2503.10522)
  - [AudioX project](https://zeyuet.github.io/AudioX/)
- Relevant idea: Unified diffusion-transformer audio/music generation conditioned by text, video, image, music, and audio signals.
- Notebook impact: Source context for external comparison packets and multimodal-control thinking; local work should only import artifacts or metrics, not another runtime stack.
- Status: source context; external comparison only.

## Multimodal Architecture Transfer Sources

### DiT

- Source: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- Relevant idea: Diffusion transformers scale with model depth/width/token count.
- Notebook impact: Supports treating SA3 transformer internals as measurable residual objects, not black-box UI outputs.
- Status: source context.

### Sora Technical Report

- Source: [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators)
- Relevant idea: Variable-duration/resolution video generation through compressed latent spacetime patches and a diffusion transformer.
- Notebook impact: Transfers as temporal segment/control-lane reasoning, not as visual interface metaphor.
- Status: source context.

### Movie Gen

- Source: [Movie Gen](https://huggingface.co/papers/2410.13720)
- Relevant idea: Media foundation models covering video, synchronized audio, editing, and personalization tasks.
- Notebook impact: Source context for temporal alignment, source preservation, and multi-turn edit evaluation.
- Status: source context.

### FLUX.1 Kontext

- Source: [FLUX.1 Kontext](https://arxiv.org/abs/2506.15742)
- Relevant idea: Flow-matching in-context image generation/editing with multi-turn consistency and preservation benchmarks.
- Notebook impact: Motivates multi-turn audio edit consistency tests over source identity, descriptor drift, and flow losses.
- Status: source context; local transfer is a hypothesis.

### Consistency Models

- Source: [Consistency Models](https://arxiv.org/abs/2303.01469)
- Relevant idea: One/few-step generation and distillation-style acceleration for generative models.
- Notebook impact: Motivates SA3 step/polish/runtime audits rather than adding a new sampler abstraction.
- Status: source context.

### PrismAudio

- Sources:
  - [PrismAudio paper](https://arxiv.org/abs/2511.18833)
  - [PrismAudio model page](https://huggingface.co/FunAudioLLM/PrismAudio)
- Relevant idea: Video-to-audio generation decomposes evaluation into semantic consistency, temporal synchrony, aesthetic quality, and spatial accuracy.
- Notebook impact: The useful transfer is the evaluation decomposition: split evidence panels rather than collapse all quality into one scalar.
- Status: source context; local transfer is evidence packet design, not reward training.

## Neural Audio Representation Anchors

### SoundStream

- Source: [SoundStream](https://arxiv.org/abs/2107.03312)
- Relevant idea: Neural audio codecs can provide compact, meaningful compressed representations.
- Notebook impact: Background for treating SAME as a playable/editable compressed signal.
- Status: source context.

### EnCodec

- Source: [EnCodec](https://arxiv.org/abs/2210.13438)
- Relevant idea: High-quality neural audio codec representation.
- Notebook impact: Background for latent audio memory and codec-style editing comparisons.
- Status: source context.

### Descript Audio Codec

- Source: [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec)
- Relevant idea: High-fidelity RVQGAN codec for 44.1 kHz audio and audio language modeling pipelines.
- Notebook impact: Background for codec-latent baselines and external comparisons; SAME remains the local native object.
- Status: source context.

### MOSS-Audio-Tokenizer

- Sources:
  - [MOSS-Audio-Tokenizer paper](https://arxiv.org/abs/2602.10934)
  - [OpenMOSS/MOSS-Audio-Tokenizer](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)
- Relevant idea: Scalable transformer-based discrete audio tokenization trained end-to-end across speech, sound, and music.
- Notebook impact: Context for why SAME should be stress-tested as a semantic/acoustic bottleneck; local work does not need another tokenizer unless used for external comparison.
- Status: source context; docs-only unless a comparison artifact is imported.

### SemanticVocoder

- Source: [SemanticVocoder](https://arxiv.org/abs/2602.23333)
- Relevant idea: Replace acoustic VAE latents with semantic latents for generation, trying to bridge audio understanding and generation.
- Notebook impact: Adds pressure to measure whether SAME summaries behave semantically, acoustically, or as a mixed bottleneck under edits.
- Status: source context; local transfer is SAME representation measurement and prompt semantic audit.

### Generative Audio Compression

- Source: [High-Fidelity Generative Audio Compression at 0.275kbps](https://arxiv.org/abs/2602.00648)
- Relevant idea: Task-oriented ultra-low-bitrate compression that relies on a powerful generative receiver prior.
- Notebook impact: Motivates SAME bottleneck stress tests and direct-decode versus SA3-polish comparisons.
- Status: source context; local transfer is a hypothesis.

### S-PRESSO

- Sources:
  - [S-PRESSO paper](https://arxiv.org/abs/2602.15082)
  - [S-PRESSO project](https://zineblahrichi.github.io/s-presso/)
- Relevant idea: Ultra-low-bitrate sound-effect compression with diffusion autoencoders and offline quantization, where a generative decoder reconstructs plausible audio from sparse embeddings.
- Notebook impact: Sharpens the SAME question: what is preserved in the latent, and what does SA3/SAME reconstruct or hallucinate during direct decode and polish?
- Status: source context; local transfer is the bottleneck and polish audit.

### RAVE

- Source: [RAVE](https://arxiv.org/abs/2111.05011)
- Relevant idea: Latent spaces can become playable real-time synthesis surfaces.
- Notebook impact: Motivates neural latent DSP as a creative/probing interface over SAME.
- Status: source context.

### DDSP

- Source: [DDSP](https://arxiv.org/abs/2001.04643)
- Relevant idea: Interpretable signal-processing structure can be integrated with neural systems.
- Notebook impact: Use classic DSP descriptors as measurement hooks and differentiable side losses, not as fake labels for what SAME channels mean.
- Status: source context.

## Evaluation, Semantics, and Listening Sources

### MusicSem

- Sources:
  - [MusicSem paper](https://arxiv.org/abs/2602.17769)
  - [MusicSem project](https://music-sem-web.vercel.app/)
- Relevant idea: Natural music descriptions contain descriptive, atmospheric, situational, metadata-related, and contextual semantics, which are broader than ordinary caption datasets.
- Notebook impact: Provides a source-backed vocabulary for prompt semantic audits and listening tags without pretending those tags are model-ground-truth.
- Status: source context; local transfer is evidence vocabulary and prompt audit design.

### Music Arena

- Sources:
  - [Music Arena paper](https://arxiv.org/abs/2507.20900)
  - [Music Arena dataset page](https://huggingface.co/datasets/music-arena/music-arena-dataset)
- Relevant idea: Text-to-music evaluation benefits from live pairwise listening preferences, natural-language feedback, and transparent data release policy.
- Notebook impact: Strong support for the current ledger/player approach: repeated pairwise listening notes are more honest than premature scalar reward training.
- Status: source context; local transfer is annotation and report discipline.

## Activation and Representation Steering

### Activation Engineering / ActAdd

- Source: [Activation Engineering / ActAdd](https://arxiv.org/abs/2308.10248)
- Relevant idea: Add contrastive activation directions at inference time.
- Notebook impact: supports prompt residual probes, audio residual probes, residual feature atlas work, and cautious steering audits in the residual and trajectory bench.
- Status: source context; local residual probing primitives are implemented, while steering remains an intervention candidate.

### Representation Engineering

- Source: [Representation Engineering](https://arxiv.org/abs/2310.01405)
- Relevant idea: Treat internal representations as controllable and measurable objects.
- Notebook impact: Supports residual feature discovery and intervention audits.
- Status: source context.

### Inference-Time Intervention

- Source: [Inference-Time Intervention](https://arxiv.org/abs/2306.03341)
- Relevant idea: Steer model behavior through inference-time representation edits.
- Notebook impact: Supports residual activation patching and layer/control atlas ideas.
- Status: source context; causal validation remains open.

## Source-Use Rules

- External sources motivate notebook experiments; they do not prove local audio behavior.
- Local claims need notebook runs, descriptors, manifests, and listening notes.
- Adapter-training sources stay in the Underfit lane unless exported artifacts are brought back for comparison.
- Prompt/guidance/control sources are translated into SA3/SAME-native measurements before becoming notebook controls.
