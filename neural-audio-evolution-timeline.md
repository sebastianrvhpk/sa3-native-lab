# Neural Audio Generation: Architecture, Math, and Research Timeline

This note is a conceptual bridge from earlier neural audio practice, such as SampleRNN, WaveNet, mel-spectrogram VAEs, Dance Diffusion, and MusicGen, to the current latent/audio-diffusion/control landscape. It is written for experimentation: to make the model families, equations, infrastructure shifts, and internal-representation perspectives legible enough to support later notebooks and implementations.

Core framing:

```text
audio generation has evolved by repeatedly changing the object being modeled

waveform samples
-> spectrogram frames
-> learned continuous latents
-> discrete codec tokens
-> diffusion/flow latents
-> editable conditional latent trajectories
-> internal representations and controllable sidecar maps
```

The broad AI field has moved in a similar direction across modalities: learn a representation, build a powerful conditional prior over that representation, then steer the prior with prompts, examples, gradients, adapters, or activation edits.

## Primary Anchors

Representative references, not exhaustive:

- SampleRNN: https://arxiv.org/abs/1612.07837
- WaveNet: https://arxiv.org/abs/1609.03499
- NSynth: https://arxiv.org/abs/1704.01279
- VQ-VAE: https://arxiv.org/abs/1711.00937
- WaveRNN: https://arxiv.org/abs/1802.08435
- MelGAN: https://arxiv.org/abs/1910.06711
- DDSP: https://arxiv.org/abs/2001.04643
- Jukebox: https://arxiv.org/abs/2005.00341
- DiffWave: https://arxiv.org/abs/2009.09761
- WaveGrad: https://arxiv.org/abs/2009.00713
- RAVE: https://arxiv.org/abs/2111.05011
- SoundStream: https://arxiv.org/abs/2107.03312
- AudioLM: https://arxiv.org/abs/2209.03143
- EnCodec: https://arxiv.org/abs/2210.13438
- MusicLM: https://arxiv.org/abs/2301.11325
- AudioLDM: https://arxiv.org/abs/2301.12503
- Make-An-Audio: https://arxiv.org/abs/2301.12661
- TANGO: https://arxiv.org/abs/2304.13731
- MusicGen: https://arxiv.org/abs/2306.05284
- AudioLDM 2: https://arxiv.org/abs/2308.05734
- SoundStorm: https://arxiv.org/abs/2305.09636
- VampNet: https://arxiv.org/abs/2307.04686
- DITTO: https://arxiv.org/abs/2401.12179
- Stable Audio Open: https://arxiv.org/abs/2407.14358
- MusicFlow: https://arxiv.org/abs/2410.20478
- Low-Resource Guidance / LatCH: https://arxiv.org/abs/2603.04366
- Stable Audio 3: https://arxiv.org/abs/2605.17991
- SAME: https://arxiv.org/abs/2605.18613
- Live Music Diffusion Models: https://arxiv.org/abs/2605.22717

## The Core Difficulty of Audio

Audio is a brutal generative modeling target because it is simultaneously:

- high-rate: 44.1 kHz stereo means 88,200 scalar samples per second,
- long-context: musical form needs seconds to minutes of dependency,
- phase-sensitive: tiny waveform errors can matter or be inaudible depending on context,
- perceptually nonlinear: waveform MSE is poorly aligned with listening,
- multi-scale: samples, pitch periods, notes, bars, sections, and style all coexist,
- weakly labeled: captions are noisy and often describe vibe rather than acoustic facts.

So neural audio history can be read as a sequence of answers to one question:

```text
What representation makes audio generation tractable without destroying what matters?
```

The rest follows from that.

## Era 1: Raw Waveform Likelihood

Approximate period: 2016-2018.

Representative systems:

- WaveNet,
- SampleRNN,
- WaveRNN,
- early neural vocoders.

### Object Being Modeled

Raw audio samples:

```text
x = (x_1, x_2, ..., x_T)
```

The canonical autoregressive factorization:

```text
p(x) = product_t p(x_t | x_<t)
```

For conditional generation:

```text
p(x | c) = product_t p(x_t | x_<t, c)
```

where `c` might be text, linguistic features, pitch, speaker embedding, mel frames, or instrument note metadata.

### WaveNet View

WaveNet uses causal dilated convolutions:

```text
h_t^l = f(h_{t - d_l}^{l-1}, h_t^{l-1}, c)
```

where dilation `d_l` expands receptive field without processing every prior sample through recurrence.

Its key lesson:

```text
audio sample prediction works, but the sequence length is punishing
```

This was philosophically important. It showed that a neural net could learn a waveform distribution directly. But it also made the bottleneck obvious: 44,100 decisions per second per channel is not a friendly musical interface.

### SampleRNN View

SampleRNN uses hierarchical recurrent modules at different temporal scales:

```text
h_k = RNN_k(h_k, frame_k)
p(x_t | context) = softmax(g(h_1, h_2, ..., h_K))
```

Artistically, SampleRNN was compelling because it exposed a strange internal musicality: it could create texture, pseudo-language, rhythm-like behavior, and instrument-like gestures without explicit symbolic structure.

But it was difficult to control. The learned hidden state was rich, but not cleanly aligned with human controls.

### What This Era Taught

Confirmed pattern:

- likelihood can model waveform detail,
- local audio texture is learnable,
- long-range musical structure is hard,
- generation is slow,
- control is mostly indirect.

Research perspective:

```text
raw waveform models maximize fidelity to the signal object,
but pay with sequence length and weak semantic handles
```

## Era 2: Spectrograms, Vocoders, and the Two-Stage Split

Approximate period: 2017-2021.

Representative systems:

- Tacotron-style speech pipelines,
- MelGAN,
- WaveGlow, Parallel WaveGAN, HiFi-GAN,
- mel-spectrogram VAEs and GANs,
- many Colab-era music notebooks.

### Object Being Modeled

Instead of waveform samples, model a time-frequency representation:

```text
S = STFT(x)
M = Mel(|S|)
```

Then split generation:

```text
condition/text/latent -> mel spectrogram -> vocoder -> waveform
```

The vocoder learns:

```text
p(x | M)
```

while the creative model learns:

```text
p(M | c)
```

### Why This Mattered

Mel spectrograms are much shorter than waveforms. A typical mel representation might be around 50-100 frames/sec, with 80-128 frequency bins.

This made training and visualization easier:

- spectrograms can be plotted,
- VAEs can compress them,
- convolutions can see local time-frequency patterns,
- artists can inspect and edit the object directly.

### VAE View

A mel VAE learns:

```text
q_phi(z | M)
p_theta(M | z)
```

with evidence lower bound:

```text
L = E_{q_phi(z|M)}[log p_theta(M|z)] - KL(q_phi(z|M) || p(z))
```

In practice, the reconstruction term often dominates the artistic feel:

```text
loss ~= || M - M_hat || + beta * KL
```

The VAE gave a latent space one could interpolate, perturb, and map. But the latent often blurred transients and phase-dependent detail.

### GAN Vocoder View

GAN vocoders learn a waveform decoder by adversarial training:

```text
min_G max_D E[log D(x)] + E[log(1 - D(G(M)))]
```

usually with reconstruction-like spectral losses:

```text
L_G = L_adv + lambda * L_mel_or_STFT
```

The practical shift was huge: once vocoders became good, generative models could stop carrying the full burden of sample-level realism.

### What This Era Taught

```text
separate content/structure from waveform rendering
```

This is one of the lasting ideas. Modern audio systems still do this, just with learned codecs or SAME-like autoencoders instead of hand-designed mels.

Weaknesses:

- phase is discarded or approximated,
- mels are hand-designed, not learned for controllable generation,
- spectrogram image models can make plausible pictures that sound wrong,
- semantic control is still not solved.

## Era 3: Differentiable Synthesizers and Structured Priors

Approximate period: 2019-2021.

Representative systems:

- DDSP,
- NSynth,
- differentiable effects,
- neural synthesis hybrids.

### DDSP Equation

DDSP makes synthesizer parameters differentiable:

```text
y(t) = sum_k a_k(t) sin(phi_k(t)) + filtered_noise(t)
```

with phase:

```text
phi_k(t) = integral_0^t 2*pi*k*f0(tau) d tau
```

The model predicts interpretable controls:

```text
neural_net(audio_or_features) -> {f0(t), amplitudes(t), noise_filter(t), ...}
```

Then the synthesizer renders audio.

### Why This Mattered

DDSP made a different wager:

```text
do not learn all of audio if known signal-processing structure is available
```

This is important for artists because the latent controls are closer to musical parameters:

- pitch,
- loudness,
- harmonic distribution,
- noise,
- timbre transfer.

It also foreshadows current control work: not everything should be hidden in an opaque model if a control surface can be made explicit.

### What This Era Taught

Structured priors improve controllability and data efficiency. But they constrain the sound world. A harmonic-plus-noise synthesizer is beautiful for some instruments and insufficient for many others.

Research perspective:

```text
explicit synthesis gives interpretable handles;
deep generative priors give breadth;
modern systems try to get both
```

## Era 4: Learned Audio Autoencoders and Codebooks

Approximate period: 2017-2023, then consolidated after 2022.

Representative systems:

- VQ-VAE,
- Jukebox,
- SoundStream,
- EnCodec,
- RAVE,
- DAC-style neural codecs,
- AudioLM/MusicGen token pipelines.

### VQ-VAE Core

An encoder maps audio or spectrogram to a continuous vector:

```text
e = E(x)
```

Then quantization picks the nearest codebook entry:

```text
k = argmin_j || e - c_j ||_2
z_q = c_k
```

Training loss:

```text
L = reconstruction_loss(x, D(z_q))
  + || sg[E(x)] - z_q ||_2^2
  + beta || E(x) - sg[z_q] ||_2^2
```

where `sg` means stop-gradient.

### Residual Vector Quantization

Modern neural codecs often use residual quantizers:

```text
r_0 = e
k_i = argmin_j || r_{i-1} - c_{i,j} ||_2
q_i = c_{i,k_i}
r_i = r_{i-1} - q_i
z_q = sum_i q_i
```

This produces multiple codebook streams per time step.

### Why Codecs Changed the Field

Neural codecs convert waveform generation into sequence modeling:

```text
audio x -> codec tokens c_1, ..., c_N
model p(c | text/audio/etc.)
codec decoder D(c) -> audio
```

Now audio can be treated more like language:

```text
p(c | condition) = product_i p(c_i | c_<i, condition)
```

This was one of the most important bridges from text/information AI to audio.

### Jukebox

Jukebox uses a multi-level VQ-VAE and autoregressive priors over compressed codes. The main idea:

```text
x -> z_top, z_middle, z_bottom
p(z_top | metadata/lyrics)
p(z_middle | z_top)
p(z_bottom | z_middle)
D(z_bottom) -> audio
```

It showed long-form music generation was possible, but with heavy compute, slow sampling, and difficult control.

### RAVE

RAVE focused on real-time neural audio synthesis with an autoencoder-like representation and adversarial/spectral training. Artistically, it mattered because it was playable:

```text
audio stream -> latent -> decoder -> audio stream
```

This made latent performance, timbre transfer, and continuous control feel like instrument design rather than offline generation.

### What This Era Taught

```text
the representation bottleneck is the instrument
```

If the bottleneck is discrete, transformers can model it. If the bottleneck is continuous, diffusion/flow can model it. If the bottleneck is structured, humans can steer it. The choice of bottleneck determines the artistic affordance.

## Era 5: Diffusion Enters Audio

Approximate period: 2020-2023.

Representative systems:

- DiffWave,
- WaveGrad,
- Dance Diffusion,
- audio latent diffusion,
- early text-to-audio diffusion systems.

### Diffusion Core

Forward noising:

```text
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
```

Noise prediction training:

```text
epsilon ~ N(0, I)
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
L = E[ || epsilon - epsilon_theta(x_t, t, c) ||_2^2 ]
```

Reverse generation:

```text
x_T ~ N(0, I)
x_{t-1} = denoise_step(x_t, epsilon_theta(x_t, t, c))
```

### Score View

Diffusion also learns a score:

```text
s_theta(x_t, t) ~= grad_{x_t} log p_t(x_t)
```

This connects audio generation to a broader mathematical view of generative modeling as learning gradients of probability density.

### Dance Diffusion

Dance Diffusion made unconditional waveform diffusion artistically accessible. It could generate coherent textures and timbral worlds from a dataset, but long-range form and prompt control were limited.

From an artist-engineer perspective, Dance Diffusion made several internal concepts tactile:

- noise schedules,
- sampler choices,
- checkpoints as aesthetic objects,
- dataset curation as composition,
- interpolation through noise/latents,
- denoising trajectory as a creative process.

### Why Diffusion Felt Different

Autoregressive models generate one step at a time:

```text
x_t depends on x_<t
```

Diffusion starts with the whole noisy object and repeatedly refines:

```text
global noisy field -> less noisy field -> coherent object
```

This is closer to sculpting than typing. It supports inpainting, editing, guidance gradients, and trajectory-level interventions.

### Weaknesses

- waveform diffusion is expensive,
- long audio remains hard,
- conditioning was initially weak,
- many samplers/knobs were poorly understood,
- no explicit musical representation unless one is built in.

## Era 6: Text-Audio Alignment and Embedding Conditioning

Approximate period: 2021-2024.

Representative systems:

- CLAP-style audio-text embeddings,
- AudioLDM,
- TANGO,
- Make-An-Audio,
- MusicLM,
- MusicGen,
- MusicLDM-like systems.

### Conditioning as Geometry

Text and audio are embedded into related spaces:

```text
e_text = F_text(prompt)
e_audio = F_audio(audio)
```

Contrastive alignment:

```text
L = - log exp(sim(e_text_i, e_audio_i) / tau)
        / sum_j exp(sim(e_text_i, e_audio_j) / tau)
```

Then the generator conditions on `e_text`:

```text
p(audio_latent | e_text)
```

or:

```text
p(codec_tokens | e_text)
```

### Why This Mattered

The control interface became language. This connected audio to the broader AI shift: text became the universal conditioning surface.

But language is not the same as control:

```text
"make it more tense" is a semantic instruction,
not a guaranteed operation on density, harmony, rhythm, and timbre
```

Text conditioning gives reach. It does not automatically give precision.

### MusicGen

MusicGen uses compressed discrete audio tokens and a transformer prior. The core object is not waveform:

```text
x -> EnCodec tokens
p(tokens | text, optional melody)
tokens -> decoder -> audio
```

MusicGen was an important practical turning point because it was accessible, coherent enough to experiment with, and clearly tied audio generation to the transformer/token paradigm familiar from language models.

### MusicLM

MusicLM combines hierarchical audio token representations and text conditioning. It demonstrated high-level text-to-music capability and long-range musical plausibility, though public reproducibility was limited.

### AudioLDM / TANGO / Make-An-Audio

These systems brought image-style latent diffusion patterns into audio:

```text
text embedding -> latent diffusion -> audio representation -> waveform
```

The key infrastructure insight:

```text
caption quality and text encoder quality matter enormously
```

This mirrors text/image models: the conditioning model is part of the generator's intelligence.

## Era 7: Masked and Parallel Token Generation

Approximate period: 2023-2024.

Representative systems:

- SoundStorm,
- VampNet,
- MAGNeT-like methods,
- masked acoustic token modeling.

### Masked Modeling View

Instead of left-to-right:

```text
p(c) = product_i p(c_i | c_<i)
```

masked models learn:

```text
p(c_M | c_notM, condition)
```

where `M` is a masked subset of tokens.

Iterative generation:

```text
initialize masked tokens
repeat:
    predict distributions for masked positions
    fill high-confidence positions
    keep uncertain positions masked
```

### Why This Mattered

Audio has long sequences, and strict autoregression is slow. Masked/parallel generation trades some factorization purity for speed and editability.

Artistically, this resembles inpainting:

```text
keep some musical material fixed;
resample the uncertain or missing parts
```

This idea now appears across modalities: image inpainting, text infilling, audio continuation, video prediction, and masked token generation are variations on conditional completion.

## Era 8: Latent Diffusion, Flow Matching, and DiTs

Approximate period: 2022-2026.

Representative systems:

- latent audio diffusion,
- Stable Audio Open,
- MusicFlow,
- Stable Audio 3,
- SAME,
- DiT-based audio generators.

### Latent Diffusion

Learn an autoencoder:

```text
z = E(x)
x_hat = D(z)
```

Then model:

```text
p_theta(z | c)
```

instead of:

```text
p_theta(x | c)
```

This is the same strategic move as image latent diffusion: move expensive pixel/sample generation into a compressed representation.

### Flow Matching

Flow matching learns a vector field:

```text
z_t = (1 - t) z_0 + t epsilon
v_target = epsilon - z_0
L = E[ || v_theta(z_t, t, c) - v_target ||^2 ]
```

Sampling solves an ODE-like process:

```text
dz / dt = v_theta(z, t, c)
```

Compared with older diffusion parameterizations, flow matching makes the "learned object" easier to describe:

```text
a conditional vector field over latent space
```

### DiT View

A diffusion transformer treats latent frames like a sequence:

```text
z_t in R^(channels x time)
project to tokens h in R^(time x width)
h <- Transformer(h, text, duration, timestep, masks)
project back to latent velocity/noise
```

This imports transformer scaling laws and infrastructure into audio diffusion:

- attention,
- cross-attention,
- adaptive normalization,
- memory tokens,
- FlashAttention,
- activation patching,
- hidden-state probing.

### Stable Audio 3 and SAME

SA3/SAME is a current version of the representation/prior split:

```text
audio x -> SAME encoder E -> z in R^(256 x T)
SA3 learns p_theta(z | prompt, duration, mask/context)
SAME decoder D maps z -> audio
```

At 44.1 kHz:

```text
T ~= audio_samples / 4096
latent_rate ~= 10.77 frames/sec
```

This is an extreme compression of time. It means SA3's DiT operates over a much shorter sequence than waveform or mel models. The tradeoff is that detailed waveform reconstruction is delegated to SAME.

### Why This Era Matters

The model is now modular:

```text
representation model: what is audio compressed into?
generative prior: what distribution over that representation is learned?
conditioning system: what controls are baked in?
sampler: how is the trajectory traversed?
post-training: how is speed/quality improved?
intervention layer: where can we steer?
```

This modularity is what makes mechanistic and artistic experimentation richer.

## Era 9: Guidance, Adapters, and Inference-Time Control

Approximate period: 2022-2026.

Representative methods:

- classifier-free guidance,
- classifier/universal/training-free guidance,
- DITTO,
- LatCH,
- activation steering,
- prompt-to-prompt-like interventions,
- branch-and-rank.

### Classifier-Free Guidance

Train conditional and unconditional behavior:

```text
epsilon_cond = epsilon_theta(x_t, t, c)
epsilon_uncond = epsilon_theta(x_t, t, empty)
epsilon_guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
```

or equivalently:

```text
epsilon_guided = epsilon_cond + (s - 1) * (epsilon_cond - epsilon_uncond)
```

This is the canonical prompt-adherence knob. It also shows a broader theme:

```text
generation can be steered by vector arithmetic in model prediction space
```

### Gradient Guidance

Given a differentiable control loss:

```text
L_control(x_t, target)
```

guide sampling with:

```text
x_t <- x_t - eta * grad_{x_t} L_control
```

or modify the score/vector field:

```text
score_guided = score_theta + lambda * grad_x log p(y | x)
```

The hard part is not writing the equation. The hard part is defining a loss that is:

- perceptually meaningful,
- differentiable,
- cheap,
- stable across noise levels,
- hard to game.

### DITTO

DITTO optimizes initial noise:

```text
epsilon* = argmin_epsilon L(f(Sampler_theta(epsilon, c)), y_target)
```

This is powerful because it can target arbitrary differentiable features without retraining the generator. But it is slow because it differentiates through sampling.

Artistically, DITTO reframes seed search:

```text
the seed is not random trivia;
it is an optimizable latent cause of structure
```

### LatCH

LatCH trains small heads to predict controls from latent states:

```text
h_psi(z_t, t) -> y
```

Then those heads can score, rank, or guide generation.

This is a pragmatic answer to the cost of audio-domain guidance:

```text
do not decode to waveform and run expensive descriptors at every step;
learn a cheap latent-space proxy
```

For SA3/SAME, this suggests:

```text
h_psi(SAME latent, optional t, optional prompt) -> control values
```

### Activation Steering

Contrastive direction:

```text
v_l = mean(h_l^positive) - mean(h_l^negative)
```

Inference intervention:

```text
h_l <- h_l + alpha * v_l
```

This sits between mechanistic interpretability and instrument design. It treats the network's residual stream as a performance surface.

Important distinction:

- LatCH reads or guides latent audio states.
- CFG changes prediction mixing.
- DITTO changes the initial noise.
- Activation steering changes internal hidden states during the forward pass.

## Era 10: Live, Block-Wise, and Interactive Diffusion

Approximate period: emerging 2025-2026.

Representative system:

- Live Music Diffusion Models.

### The Problem

Offline diffusion assumes the whole object is available:

```text
generate z[0:T] together
```

Live music wants:

```text
given committed past blocks, generate the next block now
```

Naive outpainting works but is inefficient. In a bidirectional transformer, the clean context representation can depend on the noisy target at every denoising step, so it cannot simply be cached.

### LMDM Idea

Separate clean context and noisy target routing:

```text
context blocks -> cached context representation
target block -> denoising with attention to context
```

Block-causal extension:

```text
committed block k becomes cached context for block k+1
```

The shift is architectural, not just prompt engineering.

### Why This Matters Artistically

This is the path from "generate me a file" toward "play with a model in time":

- generative delay,
- call-and-response,
- live continuation,
- controllable drift,
- branching futures,
- block-level memory,
- human-in-the-loop selection.

For SA3/SAME today, the approximation is rolling continuation. True LMDM behavior would require attention routing, cache policy, and training/post-training.

## The Big Timeline

| Period | Main object | Main architecture | Artistic affordance | Main limitation |
|---|---|---|---|---|
| 2016-2018 | waveform samples | autoregressive RNN/CNN | raw learned sound texture | slow, hard to control |
| 2017-2021 | mel/spectrogram | VAE/GAN/vocoder split | visible/editable time-frequency canvas | phase/detail loss |
| 2019-2021 | synthesis parameters | DDSP/hybrid neural synths | interpretable musical controls | constrained sound world |
| 2020-2022 | waveform/spectrogram | diffusion | denoising as sculptural generation | expensive, weak long form |
| 2020-2023 | learned codes | VQ/RVQ + transformer | audio as token language | token hierarchy complexity |
| 2022-2024 | codec tokens | AR/masked transformers | text/music generation at scale | prompt control still fuzzy |
| 2023-2025 | audio latents | latent diffusion/DiT | editability, inpainting, scale | representation bottleneck matters |
| 2024-2026 | flow latents | flow matching/rectified flow | faster latent trajectories | controls still need measurement |
| 2025-2026 | latent blocks | live/block diffusion | interactive generation | requires architectural changes |
| 2026+ | internal states | steering, probes, sidecars | models as latent instruments | validation and causality |

## What Changed Since the MusicGen / Dance Diffusion Era

If the last deep hands-on period was around 2023, the main changes are:

1. Compression got more central.
   Audio generation moved further away from raw waveform modeling and toward learned latents/codecs.

2. Diffusion got faster and more latent.
   The field shifted from many-step waveform diffusion toward compressed latent diffusion, flow matching, distillation, and post-training.

3. Transformers became the common substrate.
   Whether tokens or DiT latent frames, transformer infrastructure now appears everywhere.

4. Conditioning became multi-path.
   Prompt, duration, reference audio, masks, negative prompt, CFG, adapters, and latent controls can all coexist.

5. Control became a separate research layer.
   The question is no longer only "can it generate music?" but:

   ```text
   can we observe, predict, and intervene on properties of the generated latent?
   ```

6. Internal representations became experimentally accessible.
   Activation steering, probes, hooks, and sidecar heads make it practical to treat models as objects of study, not only black boxes.

7. Live generation is no longer just low-latency inference.
   True live diffusion requires block-wise architecture, caching, and rollout training.

## Cross-Modality Perspective

Audio is not isolated. The same pattern appears across text, image, video, and robotics.

General form:

```text
raw world object x
-> representation z = E(x)
-> generative prior p_theta(z | c)
-> decoded object x_hat = D(z)
```

Text:

```text
x = document
z = tokens
p_theta(tokens | context)
```

Images:

```text
x = pixels
z = VAE latent patches
p_theta(z | text, image refs, masks)
```

Audio:

```text
x = waveform
z = codec tokens or continuous audio latents
p_theta(z | text, audio refs, masks, duration)
```

Video:

```text
x = frames
z = spatiotemporal latents
p_theta(z | text, image refs, motion, past frames)
```

Robotics:

```text
x = sensor/action trajectories
z = state/action tokens or embeddings
p_theta(actions | observations, goals)
```

The shared theoretical move:

```text
learning becomes representation + conditional generation + optimization/control
```

The modality-specific work is deciding what `z` should be and which interventions are meaningful.

## Infrastructure Evolution

### Compute and Kernels

Early neural audio was bottlenecked by sequence length and slow sampling. The modern stack benefits from:

- mixed precision,
- FlashAttention,
- compiled graphs,
- fused kernels,
- distributed training,
- gradient checkpointing,
- faster samplers,
- post-training distillation,
- optimized neural codecs/decoders.

These details are not secondary. Architecture and infrastructure co-evolve. A model family becomes artistically available when its sampling loop is cheap enough to iterate.

### Tooling

The practical research surface moved from custom scripts to:

- PyTorch-first training loops,
- Hugging Face model distribution,
- safetensors checkpoints,
- AudioCraft-like toolkits,
- Gradio demos,
- Colab notebooks,
- experiment trackers,
- CLAP/audio embedding evaluators,
- local model wrappers with `return_latents`.

This matters because experimentation requires state capture:

```text
prompt + seed + checkpoint + sampler + guidance + latent + metrics + audio
```

Without this, artistic exploration becomes unreproducible.

### Data

The data story evolved from:

```text
small curated datasets
-> speech and instrument datasets
-> web-scale audio
-> captioned audio/music
-> licensed catalogs
-> manually labeled control datasets
```

The bottleneck is increasingly not just audio quantity but annotation quality:

- captions are vague,
- genre tags are inconsistent,
- subjective labels are noisy,
- licensing affects reproducibility,
- control labels often do not exist.

### Evaluation

Evaluation moved from reconstruction loss and listening examples toward:

- FAD,
- CLAP/audio-text scores,
- MOS/preference tests,
- descriptor metrics,
- beat/pitch/harmony metrics,
- prompt adherence,
- loop and transition metrics,
- human listening notes.

But no automatic metric is enough. Neural audio remains evaluation-hard because musical value is contextual.

## Mental Models for Experimentation

### 1. Model as Probability Distribution

Question:

```text
what distribution did this model learn?
```

Tools:

- sampling,
- likelihood/proxy losses,
- seed sweeps,
- prompt sweeps,
- calibration and diversity measures.

### 2. Model as Vector Field

Question:

```text
what trajectory transforms noise into audio?
```

Tools:

- denoising trajectory inspection,
- timestep-specific perturbation,
- guidance schedules,
- intermediate latent decoding,
- flow vector analysis.

### 3. Model as Representation Geometry

Question:

```text
what directions and neighborhoods exist in latent space?
```

Tools:

- interpolation,
- PCA/ICA,
- linear probes,
- latent arithmetic,
- descriptor correlations,
- clustering.

### 4. Model as Instrument

Question:

```text
what knobs are playable, stable, and expressive?
```

Tools:

- prompt templates,
- seed morphing,
- latent sliders,
- activation steering,
- branch-and-rank,
- live continuation.

### 5. Model as Editable Program

Question:

```text
which internal computations can be patched?
```

Tools:

- hooks,
- monkey-patching,
- activation replacement,
- attention inspection,
- residual stream steering,
- layer ablations.

## A Research Vocabulary for Controls

Every proposed control should be classified by:

```text
observable: can we measure it?
predictable: can a model infer it from latents?
intervenable: can we reliably change it?
independent: can we change it without changing everything else?
playable: can a human use it in real time or near real time?
```

Example:

```text
loudness:
    observable: yes, RMS/LUFS
    predictable: likely yes from latents
    intervenable: likely yes
    independent: partially, because loudness affects perceived density
    playable: yes

tension:
    observable: weakly
    predictable: maybe with labels
    intervenable: maybe through prompt/activation/LatCH
    independent: hard
    playable: only after careful validation
```

This avoids pretending all controls are equal.

## Equations as a Unified Map

### Autoregressive

```text
p(x | c) = product_t p(x_t | x_<t, c)
```

Good for local causality and token sequences. Slow for high-rate audio.

### Autoencoder

```text
z = E(x)
x_hat = D(z)
```

Good for representation discovery. Quality depends on bottleneck.

### VAE

```text
L = E_q[log p_theta(x | z)] - KL(q_phi(z | x) || p(z))
```

Good for smooth latent spaces. Can blur or average details.

### VQ / Codec

```text
z_q = nearest_code(E(x))
p(codes | condition)
```

Good for transformer priors. Quantization can introduce hierarchy and codebook artifacts.

### GAN Vocoder

```text
min_G max_D L_adv(G, D) + spectral_losses
```

Good for sharp audio rendering. Training stability and mode coverage matter.

### Diffusion

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
epsilon_theta(x_t, t, c) -> epsilon
```

Good for iterative refinement and editing. Sampling cost matters.

### Flow Matching

```text
z_t = (1 - t) z_0 + t epsilon
v_theta(z_t, t, c) -> epsilon - z_0
```

Good for learning transport/vector fields in latent space.

### CFG

```text
pred_guided = pred_uncond + s * (pred_cond - pred_uncond)
```

Good for prompt adherence. Can reduce diversity or overcook outputs.

### Activation Steering

```text
h_l <- h_l + alpha * (mean(h_l^+) - mean(h_l^-))
```

Good as a mechanistic control hypothesis. Needs causal validation.

### LatCH

```text
h_psi(z_t, t, prompt?) -> controls
```

Good for latent-space measurement, ranking, and possible guidance.

### Inpainting / Continuation

```text
p(z_missing | z_known, mask, prompt, duration)
```

Good for composition as conditional completion.

## What to Implement Later

The knowledge-building path should not start with a giant architecture. It should start with small, inspectable experiments.

Suggested sequence:

1. Recreate small historical baselines.
   - SampleRNN-style toy sequence model.
   - Mel VAE.
   - Tiny DDSP synth.
   - Tiny diffusion on spectrogram patches.

2. Build representation intuition.
   - Encode/decode with a neural codec.
   - Encode/decode with SAME.
   - Compare latent rates, reconstruction, artifacts, and controllability.

3. Build descriptor maps.
   - loudness,
   - brightness,
   - density,
   - stereo width,
   - pitch/register,
   - beat strength.

4. Build prompt and seed baselines.
   - paired prompts,
   - multi-seed branch-and-rank,
   - descriptor/human evaluation.

5. Probe internals.
   - hidden state capture,
   - layerwise linear probes,
   - contrastive activation directions,
   - alpha sweeps.

6. Train sidecars, not full models.
   - LatCH-style heads,
   - prompt-aware scoring heads,
   - pairwise preference heads.

7. Only then consider adaptation.
   - sampler guidance for objective controls,
   - block-wise continuation experiments,
   - architecture changes if measurement justifies them.

## High-Level Takeaways

1. Neural audio did not evolve linearly from "worse" to "better"; it evolved by changing representations.

2. The field repeatedly trades signal fidelity, compression, semantic control, and speed.

3. Modern systems are modular: autoencoder, prior, conditioner, sampler, adapter, and evaluator are distinct research surfaces.

4. The most important current question is not only "can it generate convincing audio?" It is:

```text
what parts of the generated object are observable, predictable, and intervenable?
```

5. For an artist working with internals, the model is not just a generator. It is:

```text
a probability distribution,
a vector field,
a representation geometry,
a programmable instrument,
and a partially interpretable computational system
```

That is the useful bridge from neural audio to the wider AI field: the same mathematical objects and intervention ideas now recur across text, images, video, code, and audio. Audio remains special because time, perception, and musical meaning make the representation problem unusually exposed.
