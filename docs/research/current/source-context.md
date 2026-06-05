# SA3 Native Lab Source Context

Status: migrated source context for notebook-first SA3 Native Lab research as
of 2026-06-05.

This document answers: which external papers/repos shaped the notebook, what
idea each source contributes, how that idea affects the local research, and
whether it is source context, implemented, or still a hypothesis.

These notes are a curated research map, not primary sources. The linked papers
and repositories are the sources.

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

- Status: source context; implemented locally in `flow_prompt.py` and Modes 2, 16, and 17.

### EDM

- Source: [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- Relevant idea: Noise schedules, preconditioning, and design-space clarity for diffusion sampling.
- Notebook impact: Encourages explicit timestep/logSNR handling and diagnostic loss-by-timestep panels.
- Status: source context.

### Classifier-Free Guidance

- Source: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- Relevant idea: Conditional/unconditional branch combination for prompt guidance.
- Notebook impact: Motivates explicit null/blank conditioning probes and Mode 23 null-condition inversion.
- Status: source context; local null-condition work is a probe.

### Textual Inversion

- Source: [Textual Inversion](https://arxiv.org/abs/2208.01618)
- Relevant idea: Learn pseudo-token embeddings in a frozen model's conditioning space.
- Notebook impact: Mode 1 soft prompt inversion is the audio-native analogue over SA3 conditioning.
- Status: source context; local soft prompt optimization is implemented.

### DreamBooth

- Source: [DreamBooth](https://arxiv.org/abs/2208.12242)
- Relevant idea: Personalization by adapting model behavior to a subject.
- Notebook impact: Contrast point for this repo: prefer frozen-model probes and Underfit for external fine-tuning rather than local personalization code.
- Status: source context.

### Prompt-to-Prompt

- Source: [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)
- Relevant idea: Prompt edits can control generated content through model internals/attention.
- Notebook impact: Supports prompt microscope, token attribution, and prompt replacement panels.
- Status: source context; Mode 16 implements flow-based prompt attribution.

### Null-Text Inversion

- Source: [Null-text Inversion](https://arxiv.org/abs/2211.09794)
- Relevant idea: Optimize unconditional/null conditioning while preserving a conditional prompt for editing.
- Notebook impact: Motivates Mode 23: optimize SA3 null/unconditional conditioning against a target audio flow while keeping the human prompt editable.
- Status: source context; local method is scaffold/probe.

## Guidance, Inverse Problems, and Control

### Diffusion Posterior Sampling

- Source: [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)
- Relevant idea: Inject measurement consistency into diffusion sampling.
- Notebook impact: Supports audio-to-audio posterior guidance and source-preservation losses.
- Status: source context; local Mode 25 is scaffold.

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
- Notebook impact: Supports Mode 24 guidance-gradient latent edit probes and objective mixers.
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

## Audio Generation Baselines

### AudioLDM

- Source: [AudioLDM](https://arxiv.org/abs/2301.12503)
- Relevant idea: Text-to-audio latent diffusion baseline.
- Notebook impact: Cross-model baseline context for Mode 26.
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
- Notebook impact: Mode 26 can compare SA3 outputs against external model commands when available.
- Status: source context; no local model management.

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

## Activation and Representation Steering

### Activation Engineering / ActAdd

- Source: [Activation Engineering / ActAdd](https://arxiv.org/abs/2308.10248)
- Relevant idea: Add contrastive activation directions at inference time.
- Notebook impact: Supports Modes 8, 9, and 22 residual steering and feature atlas work.
- Status: source context; local residual steering primitives are implemented.

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

