# SA3 Native Lab Notebook Research Map and Next Methods

Status: research synthesis for the notebook-only SA3 Native Lab direction as of
2026-06-04.

Scope:

- Keep the expanded Colab notebook as the main research instrument.
- Keep methods cell-based, inspectable, and runnable without app scaffolding.
- Treat SA3 and SAME as the native objects of study.
- Prefer frozen-model probes, sidecar predictors, and measurement cells before
  heavier training.
- Treat LoRA as an external handoff to Underfit, not as an active local
  implementation backlog.

This document combines:

- Local repo capabilities confirmed from `colab/`,
  `latent_audio_primitives/`, `scripts/`, `tests/`,
  `docs/research/current/`, and `docs/research/methods/`.
- External research and repo survey across Stable Audio 3, SAME, Underfit as
  the external LoRA reference, flow matching, music/audio control, prompt
  inversion, activation steering, neural codecs, and older neural audio systems.

## Evidence Labels

- `confirmed`: directly present in this repo or in a cited primary source.
- `repo-inferred`: strongly implied by local code but not executed here.
- `paper-inferred`: supported by cited papers/repos, not yet implemented here.
- `hypothesis`: plausible notebook experiment to test.
- `unknown`: needs measurement or source inspection before relying on it.

## External Sources Checked

Primary SA3/SAME:

- [Stable Audio 3 paper](https://arxiv.org/abs/2605.17991)
- [SAME: A Semantically-Aligned Music Autoencoder](https://arxiv.org/abs/2605.18613)
- [Stable Audio 3 official repo](https://github.com/Stability-AI/stable-audio-3)
- [Stability AI Stable Audio 3 research article](https://stability.ai/research/stable-audio-3)
- [Stable Audio Open paper](https://arxiv.org/abs/2407.14358)
- [Fast Timing-Conditioned Latent Audio Diffusion](https://arxiv.org/abs/2402.04825)

External LoRA and adapter context:

- [dada-bots/underfit](https://github.com/dada-bots/underfit)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [DoRA](https://arxiv.org/abs/2402.09353)
- [BoRA](https://arxiv.org/abs/2412.06441)
- [LoRA-XS](https://arxiv.org/abs/2405.17604)

Flow, guidance, inversion, and control:

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)
- [FreeDoM](https://arxiv.org/abs/2303.09833)
- [Universal Guidance for Diffusion Models](https://arxiv.org/abs/2302.07121)
- [ControlNet](https://arxiv.org/abs/2302.05543)
- [Textual Inversion](https://arxiv.org/abs/2208.01618)
- [DreamBooth](https://arxiv.org/abs/2208.12242)
- [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)
- [Null-text Inversion](https://arxiv.org/abs/2211.09794)

Audio/music generation and editing:

- [Controllable Music Production with Diffusion Models and Guidance Gradients](https://arxiv.org/abs/2311.00613)
- [Music ControlNet](https://arxiv.org/abs/2311.07069)
- [MusicMagus](https://arxiv.org/abs/2402.06178)
- [InstructME](https://arxiv.org/abs/2308.14360)
- [LiLAC](https://arxiv.org/abs/2506.11476)
- [Audio ControlNet](https://arxiv.org/abs/2602.04680)
- [MusRec](https://arxiv.org/abs/2511.04376)
- [FluxMusic](https://arxiv.org/abs/2409.00587)
- [AudioLDM](https://arxiv.org/abs/2301.12503)
- [TANGO](https://arxiv.org/abs/2304.13731)
- [MusicGen](https://arxiv.org/abs/2306.05284)
- [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)

Older neural-audio anchors:

- [SoundStream](https://arxiv.org/abs/2107.03312)
- [EnCodec](https://arxiv.org/abs/2210.13438)
- [RAVE](https://arxiv.org/abs/2111.05011)
- [DDSP](https://arxiv.org/abs/2001.04643)

Activation and representation steering:

- [Activation Engineering / ActAdd](https://arxiv.org/abs/2308.10248)
- [Representation Engineering](https://arxiv.org/abs/2310.01405)
- [Inference-Time Intervention](https://arxiv.org/abs/2306.03341)

## Local Research State

Core native graph, `confirmed`:

```text
audio waveform x
  -> SAME encoder E
  -> SAME latent z, B x C x T, C = 256, T ~= samples / 4096
  -> latent operators, prompt objectives, memory summaries, residual probes
  -> SA3 flow field v_theta(z_t, t, C(prompt))
  -> SAME decoder D
  -> decoded audio y
```

The notebook's current research stance, `confirmed`:

```text
freeze SA3
freeze SAME
edit prompts, soft conditioning, SAME latents, SA3 sampler states, residual activations,
or train small sidecars/control heads
measure with native flow losses, latent geometry, audio descriptors, listening notes,
and experiment manifests
```

Primary source of truth, `confirmed`:

```text
colab/sa3_same_native_experimental_modes.ipynb
```

## Research Discipline

Carried forward from the earlier broad research notes:

- Treat latent metrics as candidate generators, not final truth. Every promising
  latent/control score needs decoded audio, descriptor deltas, and listening
  notes before it becomes a claim.
- Separate every control into three questions: observability, predictability,
  and intervenability. A descriptor can be measurable without being controllable;
  a probe can predict a control without proving a causal edit.
- Establish prompt-only and branch-and-rank baselines before internal steering,
  sampler guidance, or sidecar training. This keeps model variance from being
  mistaken for a new control method.
- Prefer pairwise human labels for subjective qualities such as tension,
  section role, prompt adherence, transition quality, loop usability, and
  musical coherence.
- Log exact checkpoint/config/runtime details for every experiment; SA3 wrapper
  paths, conditioning tensors, hooks, and sampler defaults are part of the
  result.
- Drop a method when it cannot move a measured signal, or when it moves the
  signal but listening repeatedly rejects the output.

## Capability Map

### Notebook Modes

| Mode | Capability | Evidence | Main object | Main artifact |
|---|---|---:|---|---|
| 0 | renoise variations | confirmed | SA3 init/audio latent | generated wavs + manifest |
| 0c | latent-selective renoise | confirmed | SAME channel/time masks | variants + annotations |
| 0e | cross-audio graft | confirmed | donor/source SAME channels | grafted variants |
| 0d | latent blur/sharpen/filter | confirmed | SAME latent trajectories | direct/polished audio |
| 0h | neural latent DSP | confirmed | latent gain, dynamics, FFT, PCA | audio + descriptor reports |
| 0f | cyclic roll loop lab | confirmed | waveform/latent boundary repair | loop previews |
| 0g | cyclic projection inside sampler | confirmed | denoising trajectory | loop variants |
| sign diagnostic | flow-sign check | confirmed | SA3 velocity convention | diagnostic metrics |
| 1 | audio to soft prompt | confirmed | continuous conditioning | `.pt` soft prompt |
| 1b | generate with soft prompt | confirmed | saved soft prompt | generated wav |
| 2 | audio to babble/hard prompt | confirmed | text tokens scored by SA3 flow | prompt candidates |
| 3 | audio to readable prompt | confirmed | descriptor prompt axes | readable prompt |
| 4 | dataset to soft prompt | confirmed | shared conditioning vector | dataset soft prompt |
| 5 | dataset to prompt family | confirmed | SAME clusters + text search | prompt families |
| 6 | SAME style profile | confirmed | mean/std profile | JSON profile |
| 7 | SAME style direction | confirmed | positive-reference direction | JSON direction |
| 8 | prompt residual steering | confirmed | SA3 residual activations | steering vectors |
| 9 | audio residual steering | confirmed | audio-derived residual directions | `.pt` vectors |
| 10 | flow-state optimization scaffold | confirmed | intermediate flow state | optimized latent state |
| 11 | continuation/inpainting composition | confirmed | mask + continuation region | generated extension/edit |
| 12 | LatCH-style sidecar head | confirmed | control predictor over z | `.pt` sidecar |
| 13 | external LoRA handoff | confirmed | Underfit reference workflow | external checkpoints/audio |
| 14 | latent memory instrument | confirmed | indexed `LatentItem`s | memory index |
| 15 | geometry/intervention audit | confirmed | latent collection | geometry report |

### Reusable Local Modules

`latent_audio_primitives`, `confirmed`:

- `flow_prompt.py`: SA3 frozen flow prompt loss, logSNR timesteps, velocity target conventions.
- `prompt_optimization.py`: coordinate, greedy-token, and beam hard-prompt search.
- `latent_dsp.py`: latent gain, dynamics, soft clipping, FFT EQ, phase, donor magnitude/phase, PCA gain.
- `latent_blur.py`: temporal/channel blur, low-rank projection, unsharp masking, FFT filters, SA3 polish.
- `selective_renoise.py`: channel selection, masks, masked noise, grafting, sampler helpers.
- `looping.py`: cyclic roll, cyclic mix, repeated previews, loop metrics, sampler-level cyclic projection.
- `geometry.py`: PCA geometry, whitening, Mahalanobis distance, covariance transport, barycenters.
- `periodic.py`: autocorrelation, best period lag, FFT energy, spectral centroid, boundary loss.
- `style.py`: style profiles and directions.
- `audio_vectors.py`: positive/reference SAME directions.
- `residual_features.py`: residual activation basis and directions.
- `observability.py`: linear control probes and intervention effect.
- `audio_descriptors.py`: lightweight audio descriptor reports.
- `composition.py`: transition, loop, bridge, and path ranking.
- `index.py`, `schema.py`, `io.py`: latent memory item, search, persistence.
- `colab_audio_player.py`: waveform player, annotation save/search, notebook listening bench.

Research scripts, `confirmed`:

- `encode_dataset_same.py`: encode audio folders into SAME memory.
- `pre_encode_dataset.py`: encode captioned/audio folders into reusable latent
  datasets when useful for notebook experiments.
- `optimize_sa3_soft_prompt.py`: soft prompt optimization from target audio.
- `extract_sa3_vectors.py`: prompt-derived SA3 residual vectors.
- `extract_audio_residual_vectors.py`: audio-derived residual vectors.
- `run_sa3_alpha_sweep.py`: steering alpha sweeps.
- `generate_sa3_with_*`: apply soft prompts, style profiles, style directions, audio directions.
- `validate_colab_notebook.py`: notebook cell validation.

## I/O and Artifact Graph

```text
audio files
  -> SAME encode
  -> LatentItem / memory folder
  -> latent summaries, geometry, periodicity, descriptors
  -> retrieval / clustering / positive-reference sets

target audio
  -> SAME latent z0
  -> SA3 frozen flow scoring
  -> soft prompt, hard prompt, readable prompt, prompt family

source audio + donor audio
  -> SAME latents
  -> masks, blur, DSP, graft, covariance transport, cyclic operators
  -> direct SAME decode
  -> optional SA3 polish
  -> audio outputs + descriptors + annotations

prompt pairs or labeled audio sets
  -> SA3 residual activation capture
  -> steering vectors / residual feature basis
  -> alpha sweeps
  -> generated outputs + probe reports

LoRA/style fine-tuning need
  -> leave this repo's active method path
  -> use Underfit
  -> optionally bring back generated audio/checkpoints as external comparison artifacts
```

## Runtime and Dependency Assumptions

`confirmed`:

- Colab L4 is the intended notebook runtime.
- SA3 Medium uses SAME-L and requires CUDA plus FlashAttention for normal use.
- SA3 Small Music/SFX use SAME-S and are CPU-capable according to the upstream repo.
- The notebook can validate without model loading via `scripts/validate_colab_notebook.py --skip-setup`.
- LoRA training is externalized to Underfit. This repo should not grow another
  LoRA training/control surface unless there is a specific research reason
  Underfit cannot cover.

`unknown`:

- Whether SA3 internal sampler hooks remain stable under every optimized/compiled path.
- Which residual layers are best for audio controls.
- Which SAME latent axes remain perceptually stable across datasets, durations, and prompt families.

## Parameter and Control Inventory

High-value existing controls, `confirmed`:

- Flow convention: `noise_minus_data` versus `data_minus_noise`.
- Flow probe bank: logSNR/timestep values, shared noise, antithetic noise, normalized MSE, cosine term, conditional-delta term.
- RENOISE: noise level, source prompt, duration, seed, polish steps.
- Selective masks: channel strategy, channel count/range, time window, mask strength.
- Latent blur/DSP: operator family, alpha, radius, cutoff, gain, phase shift, compressor threshold/ratio, PCA component gains.
- Looping: roll fraction, projection strength, polish noise, loop metric windows.
- Style/profile: alpha, mean/std attraction, direction alpha, covariance transport alpha.
- Residual steering: layer, alpha, prompt/audio pair set, baseline strategy.
- Mode 15: PCA components, covariance shrinkage, periodicity windows, linear probe labels.

Missing but promising controls, `hypothesis`:

- Time-varying control lanes for loudness, density, onset strength, pitch/chroma, stereo width, brightness, and periodicity.
- Prompt-token attribution over SA3 flow score.
- Per-timestep loss curves for prompt inversion, soft prompts, residual steering,
  and guidance experiments.
- Residual-layer causal patching strength by flow timestep.
- Dataset nearest-neighbor / memory score for generated outputs.

## Research Reading: What Matters for This Notebook

### Stable Audio 3 and SAME

`confirmed from primary sources`: Stable Audio 3 is a family of fast latent
diffusion models for variable-length generation and editing, with inpainting and
continuation. The paper says the models operate on a new semantic-acoustic
autoencoder and use adversarial post-training to accelerate and improve
inference. SAME reaches 4096x temporal compression for stereo music/general
audio and is explicitly designed to preserve audio fidelity while encouraging
semantic latent structure.

Implication for this repo:

- The notebook's choice to treat SAME as the primary latent object is aligned
  with SA3's released architecture.
- The strongest research questions are not generic audio ML questions. They are:
  "what can be measured and edited in SAME latents?" and "how does SA3's frozen
  vector field react to those latents under text conditioning?"

### Underfit and SA3 LoRA

`confirmed from underfit repo`: Underfit is an MIT-licensed SA3 LoRA trainer. It
emphasizes pre-encoded latents, metadata/prompt composition from JSON/TXT/tags
and path text, DoRA as a default, short random crops, frequent demos/checkpoints,
loss-by-timestep monitoring, checkpoint audition, and stopping around the
creative underfit elbow rather than maximum memorization.

Decision for this repo:

- Underfit is the LoRA path.
- Do not expand local LoRA training, checkpoint inspection, dashboarding, or
  adapter-control methods here.
- Keep only boundary knowledge that helps this notebook interpret external
  Underfit artifacts if needed.
- Reuse Underfit's research discipline more generally: loss-by-timestep views,
  fixed audition grids, and early stopping by listening are useful for prompt,
  residual, and guidance experiments too.

### LoRA, DoRA, BoRA, LoRA-XS

`paper-inferred`: LoRA freezes base weights and injects trainable low-rank
updates. DoRA decomposes pretrained weights into magnitude and direction, using
LoRA for direction and improving capacity/stability. BoRA extends weight
decomposition symmetrically across row and column magnitudes. LoRA-XS freezes
SVD-derived low-rank bases and trains a much smaller core matrix.

Implication for this repo:

- Adapter papers stay as context for understanding external Underfit results.
- They are not active local implementation targets.
- If adapter experiments become necessary later, start by running them in
  Underfit and importing only the resulting audio/analysis artifacts.

### Flow Matching and Native Prompt Scoring

`paper-inferred`: Flow Matching trains vector fields along probability paths.
This supports the notebook's teacher-forced scoring idea:

```text
z_t = (1 - t) z0 + t epsilon
u_t = epsilon - z0
L_flow(prompt) = E ||v_theta(z_t, t, C(prompt)) - u_t||^2
```

Implication for this repo:

- The extracted `flow_prompt.py` helper is central, not incidental.
- Prompt inversion should become more diagnostic: score by timestep/logSNR,
  conditional delta, vector direction, and score stability across seeds.
- The next serious Mode 2 improvement is a prompt attribution panel:
  remove/replace one token, rescore across the same probe bank, and display
  which token moved the SA3 flow field toward the target.

### Guidance Gradients and Inverse Problems

`paper-inferred`: Diffusion Posterior Sampling, FreeDoM, Universal Guidance, and
Apple's controllable music production work all support the same broad pattern:
use a frozen generative prior, then inject measurement or classifier gradients at
sampling time.

Implication for this repo:

- `latent_audio_primitives.guidance.gradient_guidance_step` should become a
  real notebook mode once integrated carefully into the SA3 sampler.
- A first target should not be "perfect music control." It should be a small
  control loss that is cheap and differentiable over SAME latents:

```text
L_control(z_t) =
  lambda_boundary * loop_boundary_loss(z_t)
  + lambda_profile * ||summary(z_t) - target_profile||^2
  + lambda_period * periodicity_loss(z_t)
```

This is the most direct bridge from Mode 15 measurement to generation.

### Time-Varying Control

`paper-inferred`: Music ControlNet, LiLAC, and Audio ControlNet all point toward
time-varying controls as the missing layer for music/audio generation. Text
prompts are weak for precise beat, dynamics, event, and pitch timing.

Implication for this repo:

- Avoid training a ControlNet first. Build a notebook "control lane" format:

```text
control_lane = {
  name: "onset_density" | "rms" | "brightness" | "pitch" | "stereo_width",
  rate_hz: latent_rate or descriptor_rate,
  values: T-control vector,
  confidence: optional mask,
}
```

- First use lanes for measurement, retrieval, and guidance gradients.
- Only after a lane is observable and useful should it become a sidecar control
  or sampler-guidance experiment.

### Textual Inversion, Null-Text Inversion, and Prompt Editing

`paper-inferred`: Textual Inversion learns pseudo-words in the text embedding
space of a frozen model. Null-text inversion optimizes the unconditional
conditioning path for image editing while preserving the conditional prompt.

Implication for this repo:

- Mode 1 soft prompt is an audio-native textual inversion analogue.
- Mode 2 hard prompt is a decoded/prompt-token analogue.
- New method: "SA3 null-audio inversion." Optimize the unconditional/null
  conditioning or CFG branch against target SAME flow while keeping the human
  prompt fixed. Then edit the human prompt and test whether the null conditioning
  preserves source identity.

### Residual and Representation Steering

`paper-inferred`: Activation Addition, Representation Engineering, and
Inference-Time Intervention support the general idea of contrastive activation
directions. The repo already has SA3 residual steering primitives.

Implication for this repo:

- Residual vectors should be measured as causal interventions, not just stored.
- Add activation patching tests:

```text
capture activations for source prompt/audio
capture activations for target prompt/audio
replace or blend one layer/time slice during generation
measure descriptor, flow score, and listening result
```

- A residual feature atlas should rank layers by predictive control accuracy,
  intervention effect, and side effects.

### Neural Codecs, RAVE, DDSP, and Older Audio Work

`paper-inferred`: SoundStream and EnCodec show the power of compressed audio
representations; RAVE shows that latent spaces can become playable real-time
synthesis surfaces; DDSP shows that interpretable signal-processing structure
can be integrated with neural models.

Implication for this repo:

- Mode 0h's "neural latent DSP" is well-motivated, but it must be audited.
- Latent DSP controls should be judged by:
  - direct decode effect,
  - SA3 polish survival,
  - descriptor movement,
  - listening annotations,
  - cross-clip repeatability.
- DDSP suggests a split: use classic DSP descriptors as measurements and
  differentiable side losses, not as fake labels for what SAME channels mean.

## New Notebook-Native Method Proposals

### 1. Flow Attribution Prompt Microscope

Evidence: `confirmed` flow scorer, `paper-inferred` prompt inversion/editing.

Goal: show which words/tokens make SA3's frozen vector field better explain a
target audio latent.

Notebook cells:

1. Encode target audio.
2. Build a shared probe bank: logSNR values, antithetic noise, fixed seed.
3. Score a base prompt.
4. For each token or phrase, run leave-one-out and replacement candidates.
5. Display:
   - total score,
   - per-logSNR score,
   - token contribution,
   - best replacement,
   - candidate prompt audio audition.

Core metric:

```text
contribution(token_i) = L(prompt without token_i) - L(prompt)
```

Why it is useful:

- Turns Mode 2 from "babble search" into a prompt microscope.
- Helps decide which hard prompts are meaningful versus accidental score hacks.

Risk:

- A token can improve teacher-forced flow but hurt generated audio. Must pair
  with short auditions and annotation.

### 2. Loss-by-Timestep Flow Panel

Evidence: `confirmed` Mode 2 flow helper, `paper-inferred` flow matching and
guidance-diagnostic practice.

Goal: expose whether a prompt, soft prompt, residual intervention, or guided
sample explains clean, middle, or noisy parts of the SA3 path.

Notebook cells:

1. Score prompts/interventions over a logSNR grid.
2. Plot loss curves by logSNR.
3. Mark where conditional-delta helps or hurts.
4. Compare:
   - base prompt,
   - hard inverted prompt,
   - readable prompt,
   - soft prompt,
   - residual-steered output,
   - guidance-edited output.

Why it is useful:

- Early/clean timesteps may correspond to structure preservation.
- Noisy timesteps may correspond to global style/prompt prior.
- Helps explain why some prompts sound good but score poorly in aggregate.

### 3. SA3 Null-Condition Inversion

Evidence: `paper-inferred` Null-text Inversion, `confirmed` SA3 CFG/null path
available through conditioning.

Goal: preserve target/source identity with an optimized null branch while using
editable human prompts.

Objective:

```text
min_{c_null} E ||v_theta(z_t, t, CFG(C(prompt), c_null)) - u_t||^2
```

Notebook cells:

1. Pick a source audio and a readable prompt.
2. Optimize only the null/unconditional embedding or null-conditioning tensor.
3. Generate with prompt edits while keeping optimized null conditioning.
4. Compare against Mode 1 soft prompt and Mode 2 hard prompt.

Why it is useful:

- Separates "what should remain source-like" from "what the user wants to edit."
- Potentially better for audio-to-audio restyling than optimizing all
  conditioning.

Risk:

- SA3 wrapper may not expose the exact null path cleanly. Start as a probe over
  available conditioning inputs.

### 4. SAME Control Lanes

Evidence: `paper-inferred` Music ControlNet/LiLAC/Audio ControlNet,
`confirmed` audio descriptors and latent summaries.

Goal: turn audio descriptors into time-varying notebook controls before
training any model.

Lane examples:

```text
rms envelope
spectral centroid envelope
onset density
periodicity/beat energy
stereo width
latent motion energy ||z_t - z_{t-1}||
```

Notebook cells:

1. Extract lane from source or draw/edit it in a small HTML/SVG lane editor.
2. Align lane to SAME latent rate.
3. Use lane for retrieval: find memory items with similar lane.
4. Use lane as a differentiable guidance loss if possible.
5. Record whether the generated audio follows the lane.

Why it is useful:

- Bridges descriptor measurement and controllable generation.
- Keeps time-varying control honest: first observe, then guide, then maybe train.

### 5. Guidance-Gradient SA3 Sampler Mode

Evidence: `confirmed` local `guidance.py`, `paper-inferred` DPS/FreeDoM/Apple
guidance-gradient work.

Goal: integrate a small differentiable SAME-space control loss into the SA3 flow
sampler.

First losses:

```text
L_profile = ||summary(z_t) - target_summary||^2
L_boundary = loop_boundary_loss(z_t)
L_period = ||fft_energy(z_t) - target_fft_energy||^2
```

Notebook cells:

1. Generate baseline.
2. Generate with one guidance loss.
3. Sweep guidance alpha.
4. Display loss trace, descriptor deltas, and audio player rows.

Why it is useful:

- Directly tests whether Mode 15 metrics can become interventions.
- Avoids training new control models prematurely.

Risk:

- Gradients through the SA3 sampler may be memory-heavy. Start with short
  duration, few steps, and one loss.

### 6. Latent OT Style Transfer Bench

Evidence: `confirmed` covariance transport and barycenters in `geometry.py`,
`paper-inferred` flow-matching OT paths.

Goal: compare mean/std style attraction against full-covariance transport and
barycenter mixing.

Notebook cells:

1. Fit source and reference geometry from memory folders.
2. Apply:
   - mean/std profile,
   - frame direction,
   - covariance transport,
   - barycenter blend.
3. Decode direct and polished versions.
4. Show Mahalanobis movement and audio descriptor deltas.

Why it is useful:

- Tests whether cross-channel covariance matters for SAME style movement.
- Provides a rigorous alternative to naive latent vector arithmetic.

### 7. Residual Feature Atlas

Evidence: `confirmed` residual feature primitives, `paper-inferred` activation
steering/representation engineering.

Goal: make residual steering measurable by layer, timestep, and feature.

Notebook cells:

1. Capture residual activations for prompt/audio contrast sets.
2. Fit SVD/PCA feature basis per layer.
3. Rank features by probe accuracy and generation effect.
4. Try alpha sweeps on selected layers.
5. Store a layer-feature report.

Metrics:

```text
probe_accuracy(layer, feature)
descriptor_delta(alpha)
flow_score_delta(alpha)
listening_annotation_tags
side_effect_score
```

Why it is useful:

- Prevents random residual steering from becoming magic knobs.
- Helps identify stable SA3 layers for mood, density, brightness, rhythm, and
  section role.

### 8. Dataset Memory as Prompt and Control Curriculum

Evidence: `confirmed` latent memory, clustering, prompt family, geometry,
observability, and residual-vector primitives.

Goal: use latent memory and geometry to build better prompt search seeds,
control-lane targets, residual contrast sets, and evaluation splits.

Notebook cells:

1. Encode dataset to memory.
2. Cluster by SAME summaries plus descriptor lanes.
3. Generate prompt seeds and readable axes per cluster.
4. Extract representative control lanes per cluster.
5. Build positive/reference sets for residual vectors and SAME directions.
6. Create held-out evaluation splits for every method.
7. Retrieve nearest memory items for any generated output.

Why it is useful:

- Converts the dataset from passive material into an experiment curriculum.
- Gives prompt inversion, residual steering, control lanes, and geometry tests
  the same source/evaluation structure.
- Provides a local memorization/novelty check without adding fine-tuning here.

### 9. Continuation as Bridge Search

Evidence: `confirmed` composition module and Mode 11, `paper-inferred` music
production guidance tasks.

Goal: choose continuation/inpainting candidates by transition cost, not only by
manual listening.

Notebook cells:

1. Generate N continuations or bridges.
2. Compute boundary, descriptor, and latent transition costs.
3. Rank with `composition.py`.
4. Play top/bottom examples for calibration.

Why it is useful:

- Turns continuation into a measurable composition primitive.
- Supports loop and transition experiments without a DAW/app layer.

### 10. Cross-Model Baseline Harness

Evidence: `paper-inferred` AudioCraft/MusicGen, AudioLDM, TANGO, Stable Audio
Open.

Goal: keep SA3/SAME research honest by comparing against other model families
for a few prompt/control tasks.

Notebook cells:

1. Define a small prompt/audio task set.
2. Generate outputs with SA3 and optional external models if installed.
3. Run the same descriptor and listening annotation pipeline.

Why it is useful:

- If a proposed SA3 method only beats a weak baseline, it is not enough.
- If SA3/SAME has unique strengths, the comparison will show where.

Boundary:

- Do not add app scaffolding or package-lock churn. Keep this optional and
  notebook-local.

### 11. Audio-to-Audio Posterior Guidance

Evidence: `paper-inferred` DPS, FreeDoM, Universal Guidance, Apple music
guidance, `confirmed` SA3 audio-to-audio and inpainting routes.

Goal: use source audio as a measurement, not just an init.

Objective sketch:

```text
L_measure(z_t) =
  ||E(D(z_t)) - z_source||^2 over preserved regions
  + descriptor_loss(D(z_t), target_descriptor)
```

Notebook cells:

1. Start from audio-to-audio or inpaint source.
2. Preserve some regions with reconstruction loss.
3. Guide other regions toward prompt/style/descriptor.
4. Compare to plain `init_noise_level` restyling.

Why it is useful:

- SA3 already supports editing; this makes edit strength measurable.
- It may produce better source preservation than global init noise alone.

## Interface Backlog for the Notebook

P0, required for trust:

- Per-output manifest row with source, operator, seed, prompt, model, params,
  descriptors, and annotation link.
- A/B/C audio player rows: source, direct decode, SA3 polish.
- Loss-by-timestep panel for Mode 2, soft prompts, residual steering, and
  guidance experiments.
- Memory nearest-neighbor panel for any generated output.

P1, makes research faster:

- Prompt microscope: token contribution and replacement suggestions.
- Control lane editor: compact SVG/HTML envelope editor stored as JSON.
- Geometry report panel: PCA variance, Mahalanobis movement, boundary loss,
  descriptor deltas.
- Dataset curriculum panel: clusters, representative clips, prompt seeds,
  control lanes, and held-out splits.

P2, expands research cognition:

- Residual feature atlas: layer x feature heatmap with probe accuracy and
  audition links.
- Latent OT bench: profile/direction/covariance transport/barycenter comparison.
- Continuation bridge search panel with transition costs and calibration clips.
- Guidance-gradient sweep panel with loss traces and audio rows.
- Null-condition inversion comparison panel.
- Cross-model baseline report for a small fixed task set.

P3, polish only:

- Better visual styling for notebook panels.
- Exportable HTML experiment reports.
- Compact printable experiment manifest summaries.

## Verification Plan

Short-term notebook tests:

1. Validate all new cells with toggles off using
   `uv run python scripts/validate_colab_notebook.py --skip-setup`.
2. Add fake-model tests for any scorer/guidance helper.
3. Add small synthetic-latent tests for control lanes, transport, and ranking.

Audio validation protocol:

1. For every new method, produce:
   - source audio,
   - baseline generation,
   - method generation,
   - descriptor delta,
   - native flow score delta if relevant,
   - annotation prompt.
2. Run at least three seeds.
3. Test one in-domain prompt and one out-of-domain prompt.
4. Store all rows in the experiment manifest.

Stop criteria:

- Drop a method if it cannot move a measured signal.
- Drop a method if it moves the measured signal but listening consistently
  rejects it.
- Promote a method only if it survives at least two source clips, two prompts,
  and a seed sweep.

## Recommended Implementation Order

1. Flow attribution prompt microscope.
   - Minimal code, builds on `flow_prompt.py` and Mode 2.
2. Loss-by-timestep flow panel.
   - Uses the same probe-bank machinery and immediately improves Mode 2
     interpretability.
3. SAME control lanes as JSON plus visual editor.
   - Measurement first, no sampler integration yet.
4. Dataset memory as prompt and control curriculum.
   - Gives later prompt, residual, lane, and guidance methods shared evaluation
     splits and nearest-neighbor checks.
5. Latent OT style transfer bench.
   - Mostly existing `geometry.py`.
6. Continuation as bridge search.
   - Mostly existing `composition.py`, looping metrics, descriptors, and player
     rows.
7. Residual feature atlas.
   - Needs SA3 hooks and careful runtime testing.
8. SA3 null-condition inversion.
   - Requires inspecting conditioner/CFG internals.
9. Guidance-gradient sampler mode.
   - Highest payoff, highest integration risk; benefits from control lanes and
     geometry/loss panels first.
10. Audio-to-audio posterior guidance.
   - Builds on sampler guidance plus source-preservation losses.
11. Cross-model baseline harness.
   - Useful after core SA3/SAME methods are stabilized.

## Implementation Status

Implemented in the notebook after the backlog pass:

| Backlog item | Notebook mode | Local helper support | Status |
|---|---:|---|---|
| Flow attribution prompt microscope | 16 | `flow_prompt.py` attribution rows | implemented |
| Loss-by-timestep flow panel | 17 | `flow_prompt.py` loss rows and summaries | implemented |
| SAME control lanes | 18 | `control_lanes.py` | implemented |
| Dataset memory curriculum | 19 | `curriculum.py` | implemented |
| Latent OT style transfer bench | 20 | existing `geometry.py`, `style.py` | implemented |
| Continuation as bridge search | 21 | existing `composition.py` | implemented |
| Residual feature atlas | 22 | existing `residual_features.py` | implemented |
| SA3 null-condition inversion probe | 23 | notebook probe over SA3 conditioning tensors | implemented as probe |
| Guidance-gradient latent edit | 24 | existing `guidance.py` | implemented as latent edit plus SA3 polish |
| Audio-to-audio posterior guidance | 25 | existing `guidance.py` plus source/reference summaries | implemented as scaffold |
| Cross-model baseline harness | 26 | notebook command harness plus descriptors/player | implemented |

Implementation boundary:

- Modes 16-22 are ordinary notebook research cells around existing or new
  helper APIs.
- Modes 23-25 are intentionally labeled probes/scaffolds because they depend on
  exact SA3 conditioner/sampler behavior with loaded weights.
- Mode 26 does not vendor other model repos; it accepts external commands and
  compares returned audio through this notebook's descriptor/player path.
- LoRA remains external to Underfit; local LoRA training cells and scripts have
  been removed from the active repo.

## Post-Implementation Re-Review Ideas

Now that the backlog has notebook cells and helper APIs, the next implementable
ideas are smaller and sharper:

1. Probe-bank cache.
   - Save target latent, timesteps, noise seeds, and per-prompt rows so Mode 16
     and Mode 17 reuse identical probes across sessions.
2. Flow-plus-descriptor prompt search.
   - Score hard/readable prompts by a weighted combination of SA3 flow loss and
     decoded descriptor deltas after a short audition generation.
3. Control-lane retrieval.
   - Add lane similarity to `LatentMemoryIndex` so memory retrieval can combine
     latent summary, descriptor target, and time-varying lane shape.
4. Geometry-aware donor selector.
   - Rank donor candidates for graft/DSP/OT by Mahalanobis distance, lane
     similarity, and boundary compatibility.
5. Residual temporal patching.
   - Extend Mode 22 from layer-level feature atlas to layer x denoising-step or
     layer x latent-time patch tests.
6. Guidance objective mixer.
   - Let Mode 24 choose profile, boundary, period, lane, and preservation losses
     from a compact JSON recipe.
7. Null-condition edit audition.
   - After Mode 23 optimizes null conditioning, generate fixed prompt edits and
     compare source preservation against Mode 1 soft prompt and plain
     audio-to-audio.
8. Novelty/source-preservation panel.
   - For every generated artifact, show nearest memory rows plus descriptor
     deltas so "kept source identity" and "copied dataset item" are separated.
9. Seed-family atlas.
   - For a fixed method recipe, generate a seed grid and cluster outputs by SAME
     summaries, descriptor deltas, and listening tags.
10. Notebook report packager.
    - Export selected manifest rows, audio paths, descriptor tables, and
      annotations into a static HTML report without introducing an app server.

## Open Questions

- Does the native flow score predict generated-audio similarity, or only
  teacher-forced vector-field agreement?
- Which logSNR bands correspond to style, structure, transient detail, or prompt
  adherence in SA3?
- Can null-condition inversion preserve source identity while allowing text
  edits?
- Which SAME descriptor/control lanes are observable, predictable, and
  intervenable?
- Does covariance transport outperform mean/std style transfer audibly?
- Which residual layers carry stable music/audio controls?
- Can memory nearest-neighbor checks separate useful source preservation from
  memorization-like copying?
- Which dataset clusters produce prompt/control curricula that generalize?
- Can sampler-level guidance improve loopability without half-period collapse?

## Bottom Line

The project already has a strong notebook-native research base. The most valuable
next move is not more scaffolding. It is better measurement and sharper
interventions:

```text
flow microscope -> loss-by-timestep panel -> control lanes -> memory curriculum
-> geometry/OT bench -> bridge search -> residual atlas -> sampler guidance
```

This sequence keeps every new method grounded in SA3/SAME objects, preserves the
expanded notebook, and avoids building product infrastructure before the research
knows what its controls actually mean.

LoRA is intentionally outside that sequence. When LoRA is needed, use Underfit
and treat any returned audio/checkpoints as external artifacts for comparison,
not as a reason to grow local adapter infrastructure here.
