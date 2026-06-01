# Neural Audio Math Methods for Creative Technologists

This note threads the mathematical evolution of neural audio generation from a creative-technologist point of view. The goal is not to turn the math into vague metaphors. The goal is to keep the equations visible while translating what each equation means as an artistic machine.

Working perspective:

```text
AI sound work is not only "generate audio".
It is choosing what object a model learns,
what pressure the loss applies,
what representation becomes playable,
and where a human can intervene.
```

The history of neural audio can be read as a history of changing mathematical objects:

```text
waveform samples
-> spectra
-> latent variables
-> discrete codes
-> noise-to-signal trajectories
-> vector fields
-> hidden activations
-> controllable maps over representation space
```

## 1. Signal Math: The Old Foundation Still Matters

Before neural networks, audio was already math-heavy. Neural audio did not replace this math. It absorbed it.

### Waveform

Digital audio is a sampled pressure signal:

```text
x[n], n = 0, 1, ..., N-1
```

For stereo:

```text
x in R^(2 x N)
```

At 44.1 kHz:

```text
1 second = 44,100 samples per channel
```

Pop translation:

```text
waveform modeling is pixel art at insane temporal resolution
```

Every sample matters locally, but musical meaning lives across thousands or millions of samples.

### Fourier / STFT

The short-time Fourier transform cuts audio into overlapping windows and measures frequency content:

```text
X[m, k] = sum_n x[n] w[n - mH] exp(-j 2 pi k n / N)
```

where:

- `m` is time frame,
- `k` is frequency bin,
- `w` is the window,
- `H` is hop size.

Magnitude and phase:

```text
X[m, k] = |X[m, k]| exp(j phi[m, k])
```

Mel spectrogram:

```text
M = MelFilterbank(|STFT(x)|)
```

Creative meaning:

```text
STFT turns sound into a time-frequency canvas.
Waveform says "what is the pressure now?"
Spectrogram says "what energies are moving where?"
```

But phase is slippery. If you throw it away, the picture can look right while the sound feels smeared, hollow, or phasey.

## 2. Supervised Learning: The First Basic Spell

The simplest neural audio setup:

```text
model(input) -> target
```

Train by minimizing prediction error:

```text
L(theta) = mean_i || y_i - f_theta(x_i) ||^2
```

Gradient descent:

```text
theta <- theta - eta * grad_theta L(theta)
```

This is the bedrock. Everything later still uses some version of:

```text
define loss
differentiate loss
update parameters or inputs
```

Creative translation:

```text
training is pressure.
The loss tells the model what kind of mistake hurts.
The dataset tells it what world exists.
The architecture tells it what shapes of answer are easy.
```

If the loss is waveform MSE, the model learns average pressure. If the loss is spectral, it learns energy shape. If the loss is adversarial, it learns to fool a critic. If the loss is contrastive, it learns relational meaning.

## 3. Autoregression: Sound as a Sentence of Samples

This is the WaveNet / SampleRNN worldview.

### Core Equation

Model the joint probability of audio as a chain:

```text
p(x) = product_t p(x_t | x_1, ..., x_{t-1})
```

Conditional version:

```text
p(x | c) = product_t p(x_t | x_<t, c)
```

where `c` might be speaker, notes, text, class label, or previous features.

### What the Equation Means

The model asks:

```text
given everything I have heard so far,
what is the next sample?
```

At generation time:

```text
sample x_1
sample x_2 from p(x_2 | x_1)
sample x_3 from p(x_3 | x_1, x_2)
...
```

Creative translation:

```text
the model improvises one microscopic audio decision at a time
```

### Why It Worked

Autoregression is honest. It models the actual waveform distribution.

It also gives crisp probability math:

```text
negative log likelihood = - sum_t log p_theta(x_t | x_<t)
```

### Why It Hurt

Audio has brutal length:

```text
10 seconds at 44.1 kHz = 441,000 samples per channel
```

So a raw autoregressive model must make hundreds of thousands of decisions for a short phrase.

Creative cost:

```text
great texture, poor macro-control, slow iteration
```

This is why later models try to shorten the object.

## 4. Convolutions and Receptive Field: How Much Past Can the Model Hear?

WaveNet uses causal dilated convolutions.

One layer:

```text
h_t^l = f(h_t^(l-1), h_{t-d_l}^(l-1))
```

where `d_l` is dilation.

Stack dilations:

```text
d = 1, 2, 4, 8, 16, ...
```

This expands receptive field quickly.

Pop translation:

```text
dilation is memory zoom.
The model listens to nearby samples and also skips back in widening jumps.
```

Creative implication:

```text
architecture decides what time scales are easy to hear
```

If the receptive field covers milliseconds, it learns timbre. If it covers seconds, it may learn rhythm or phrase behavior. But covering long musical form with raw samples is expensive.

## 5. Spectrogram Losses: Stop Punishing the Wrong Mistakes

Waveform MSE:

```text
L_wave = || x - x_hat ||_2^2
```

Problem: tiny phase shifts can create huge waveform error even when audio sounds similar.

Spectral loss:

```text
L_stft = sum_r || |STFT_r(x)| - |STFT_r(x_hat)| ||_1
```

Multi-resolution STFT:

```text
L_mrstft = sum_r L_stft_r
```

where each `r` uses different window/hop settings.

Creative translation:

```text
instead of asking "did every pressure sample match?",
ask "did the sound energy move similarly across time and frequency?"
```

This is why modern autoencoders and vocoders often use spectral losses. They align better with hearing.

## 6. Autoencoders: Learn the Hidden Control Surface

Autoencoder:

```text
z = E_phi(x)
x_hat = D_theta(z)
L = reconstruction_loss(x, x_hat)
```

The encoder compresses. The decoder reconstructs.

Creative translation:

```text
the encoder invents a control voltage for the decoder
```

The latent `z` is not automatically meaningful. It becomes meaningful because:

- it is forced through a bottleneck,
- it must preserve what the decoder needs,
- losses decide what counts as preservation.

### Bottleneck Pressure

If `z` is too large:

```text
model can copy
```

If `z` is too small:

```text
model must summarize
```

Artistically, the bottleneck is where the instrument begins.

### SAME as Modern Autoencoder Logic

For SAME:

```text
x in R^(2 x N)
z = E(x) in R^(256 x T)
T ~= N / 4096
x_hat = D(z)
```

At 44.1 kHz:

```text
latent_rate ~= 10.77 frames/sec
```

Creative translation:

```text
SAME turns high-rate waveform into a slow, dense, continuous musical film strip
```

The DiT does not generate samples. It generates frames in this latent film strip.

## 7. VAEs: Make Latent Space Walkable

A vanilla autoencoder can create a messy latent space. VAEs regularize it.

Encoder outputs a distribution:

```text
q_phi(z | x) = N(mu_phi(x), sigma_phi(x)^2)
```

Sample:

```text
z = mu + sigma * epsilon
epsilon ~ N(0, I)
```

Decoder:

```text
p_theta(x | z)
```

Training objective, the ELBO:

```text
L = E_{q_phi(z|x)}[log p_theta(x | z)]
    - KL(q_phi(z | x) || p(z))
```

Usually optimized as a loss:

```text
loss = reconstruction_loss + beta * KL
```

### What the KL Does

The KL term pushes the encoded distribution toward a simple prior:

```text
p(z) = N(0, I)
```

Creative translation:

```text
the VAE tries to make the latent room navigable,
not just useful for reconstruction
```

You can interpolate:

```text
z(alpha) = (1 - alpha) z_a + alpha z_b
```

But VAEs can blur because the model is rewarded for covering uncertainty smoothly.

Artistic trade:

```text
smooth space, sometimes soft sound
```

## 8. GANs: Add a Critic With Taste

GAN objective:

```text
min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

Generator:

```text
x_hat = G(z)
```

Discriminator:

```text
D(x) -> probability_real
```

Creative translation:

```text
the generator makes audio;
the critic says "sounds fake";
the generator learns the critic's taste
```

In audio, GANs often pair with spectral losses:

```text
L_G = L_adv + lambda * L_stft
```

Why useful:

- sharper attacks,
- less averaged texture,
- more realistic high-frequency detail.

Why dangerous:

- unstable training,
- mode collapse,
- artifacts that fool the critic but annoy the ear.

Creative-technologist lens:

```text
GANs are not just math.
They are adversarial taste engines.
The discriminator becomes an implicit aesthetic judge.
```

## 9. DDSP: Put the Synthesizer Back Into the Network

DDSP says: if we already know useful audio physics, make it differentiable.

Harmonic oscillator bank:

```text
y_harm(t) = sum_k a_k(t) sin(phi_k(t))
```

Phase:

```text
phi_k(t) = integral 2 pi k f0(t) dt
```

Noise branch:

```text
y_noise = filtered_noise(filter_params(t))
```

Output:

```text
y = y_harm + y_noise
```

Neural network predicts parameters:

```text
network(features) -> f0(t), loudness(t), a_k(t), filter_params(t)
```

Creative translation:

```text
instead of asking a network to invent physics,
ask it to play a differentiable synth
```

This is powerful because the controls are musically named. Pitch is pitch. Loudness is loudness. Harmonic weights are timbre.

But the world is constrained by the synth.

Trade:

```text
more interpretable, less universal
```

## 10. VQ and Neural Codecs: Turn Audio Into Tokens

Vector quantization:

```text
e = E(x)
k = argmin_j || e - c_j ||_2
z_q = c_k
```

The codebook:

```text
C = {c_1, c_2, ..., c_K}
```

The model stores index `k`, not the full vector.

VQ-VAE loss:

```text
L = reconstruction_loss(x, D(z_q))
  + || stopgrad(E(x)) - z_q ||_2^2
  + beta || E(x) - stopgrad(z_q) ||_2^2
```

### Residual Quantization

Codec models often use multiple quantizers:

```text
r_0 = e
q_i = nearest_code_i(r_{i-1})
r_i = r_{i-1} - q_i
z_q = sum_i q_i
```

Creative translation:

```text
first codebook sketches the sound;
later codebooks add detail and correction
```

### Why This Was Huge

Once audio is tokens:

```text
audio -> token sequence
```

then music generation can borrow language-model machinery:

```text
p(tokens) = product_i p(token_i | token_<i)
```

Creative translation:

```text
the model stops drawing the waveform and starts writing a compressed sound score
```

This leads to Jukebox, AudioLM, MusicGen, SoundStorm-like methods, and masked token models.

## 11. Transformers: Attention as Learned Routing

Self-attention:

```text
Q = X W_Q
K = X W_K
V = X W_V
Attention(X) = softmax(Q K^T / sqrt(d)) V
```

Each token asks:

```text
which other tokens matter to me right now?
```

Creative translation:

```text
attention is a dynamic patch cable across time
```

For music:

- a drum hit can attend to earlier hits,
- a phrase can attend to its beginning,
- a token can attend to text,
- a latent frame can attend to memory tokens,
- a target inpaint region can attend to known context.

Cross-attention:

```text
Q = audio_tokens W_Q
K = text_tokens W_K
V = text_tokens W_V
```

Meaning:

```text
audio asks the prompt what matters
```

This is one of the deep connections to text AI. The same attention math that routes meaning in language now routes musical and sonic dependencies.

## 12. Contrastive Learning: Align Words and Sound

Audio-text contrastive setup:

```text
e_audio = F_audio(audio)
e_text = F_text(text)
```

Similarity:

```text
sim(a, t) = dot(normalize(e_audio), normalize(e_text))
```

InfoNCE-style loss:

```text
L_i = - log exp(sim(a_i, t_i) / tau)
          / sum_j exp(sim(a_i, t_j) / tau)
```

Creative translation:

```text
pull the matching caption and sound together;
push mismatched ones apart
```

This creates semantic geometry:

```text
"bright piano" near bright piano audio
"thunder" near thunder audio
"melancholic strings" near melancholic strings audio
```

But this space is not perfect truth. It is a learned cultural/descriptor map. It can confuse genre, mood, production, and instrumentation.

## 13. Diffusion: Learn to Undo Destruction

Forward process:

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon
epsilon ~ N(0, I)
```

Train model:

```text
epsilon_theta(x_t, t, c) ~= epsilon
```

Loss:

```text
L = E[ || epsilon - epsilon_theta(x_t, t, c) ||_2^2 ]
```

Sampling:

```text
x_T ~ N(0, I)
for t = T ... 1:
    x_{t-1} = denoise(x_t, epsilon_theta(x_t, t, c))
```

Creative translation:

```text
diffusion teaches a model how to restore signal from controlled damage
```

This is why diffusion is naturally good at:

- inpainting,
- variation,
- gradual transformation,
- guidance,
- editing trajectories.

The model is not writing one sample after another. It is sculpting an entire noisy object into coherence.

## 14. Score Matching: The Gradient of "More Real"

Score:

```text
score(x_t, t) = grad_x log p_t(x_t)
```

This vector points toward higher probability under the data distribution at noise level `t`.

Diffusion models can be viewed as learning:

```text
s_theta(x_t, t) ~= grad_x log p_t(x_t)
```

Creative translation:

```text
at every noise level, the model learns which direction sounds more like the world it was trained on
```

Guidance then becomes intuitive:

```text
base score: go toward realistic audio
control gradient: go toward my desired property
combined score: go toward realistic audio with that property
```

Mathematically:

```text
score_guided = score_model + lambda * grad_x log p(control | x)
```

## 15. Classifier-Free Guidance: Prompt Pressure as Vector Arithmetic

Conditional prediction:

```text
pred_cond = model(x_t, t, prompt)
```

Unconditional prediction:

```text
pred_uncond = model(x_t, t, empty_prompt)
```

Guided prediction:

```text
pred_guided = pred_uncond + s * (pred_cond - pred_uncond)
```

Creative translation:

```text
take the direction from "generic audio" to "prompt audio",
then turn up that direction
```

The knob `s` is not magic. It amplifies a difference vector.

Too low:

```text
more free, less obedient
```

Too high:

```text
more prompt pressure, possible artifacts, less diversity
```

CFG is one of the clearest examples of modern model control as geometry.

## 16. Latent Diffusion: Stop Denoising the Expensive Object

Instead of diffusion over waveform:

```text
p_theta(x)
```

learn an autoencoder:

```text
z = E(x)
x = D(z)
```

then generate:

```text
p_theta(z | c)
```

Creative translation:

```text
do the dreaming in compressed space;
let the decoder render the dream as sound
```

This is the move behind many modern systems. It is also why the autoencoder becomes philosophically central. It decides what the generator can easily imagine.

For SA3/SAME:

```text
audio -> SAME latent z
SA3 DiT generates z
SAME decoder renders audio
```

Question for experimentation:

```text
what did SAME decide is worth preserving?
```

## 17. Flow Matching: Learn the Motion, Not Just the Denoising

Flow matching defines an interpolation between data and noise:

```text
z_t = (1 - t) z_0 + t epsilon
```

The target velocity is:

```text
v_target = d z_t / d t = epsilon - z_0
```

Train:

```text
L = E[ || v_theta(z_t, t, c) - (epsilon - z_0) ||_2^2 ]
```

Sample by following the vector field:

```text
dz / dt = v_theta(z, t, c)
```

Creative translation:

```text
the model learns the wind that carries noise into music
```

This is a clean mental model for SA3:

```text
SA3 learns a conditional vector field over SAME latent space
```

The generated piece is a trajectory through latent space, not a single isolated output.

## 18. Rectified Flow: Make the Path Straighter

Rectified flow tries to learn straighter transports between noise and data.

If:

```text
z_t = (1 - t) z_0 + t z_1
```

then:

```text
v_target = z_1 - z_0
```

Learn:

```text
v_theta(z_t, t) ~= z_1 - z_0
```

Creative translation:

```text
instead of wandering through a foggy denoising maze,
learn a cleaner route from noise to signal
```

Why it matters:

- fewer sampling steps may work,
- trajectories can be easier to reason about,
- vector-field interventions become conceptually clearer.

## 19. Inpainting: Conditional Completion as Composition

Mask:

```text
m[t] = 1 for known region
m[t] = 0 for missing region
```

Known latent:

```text
z_known = E(audio_reference)
```

Generate:

```text
p(z_missing | m * z_known, m, prompt, duration)
```

Creative translation:

```text
do not ask the model for a whole song;
ask it to solve a musical hole with constraints
```

Continuation is the causal-looking special case:

```text
known prefix -> generate future
```

But inpainting is more general:

```text
known before and after -> generate bridge
known stems/sections -> regenerate one region
known loop edges -> repair boundary
```

This is where generation becomes composition.

## 20. DITTO and Input Optimization: The Seed Is a Handle

Normal generation:

```text
epsilon ~ N(0, I)
z = Sampler_theta(epsilon, prompt)
```

DITTO-like optimization:

```text
epsilon* = argmin_epsilon L(f(Sampler_theta(epsilon, prompt)), target)
```

Creative translation:

```text
instead of rolling random seeds, sculpt the seed until the output matches a goal
```

This reframes "seed" as a latent cause.

The expensive part:

```text
gradients must pass through the sampler
```

So it is powerful but not casual. It is like slow studio rendering, not a live knob.

## 21. Activation Steering: Hidden State as a Performance Surface

Collect activations at layer `l`:

```text
h_l(prompt)
```

Contrastive direction:

```text
v_l = mean(h_l(positive_prompts)) - mean(h_l(negative_prompts))
```

Patch at inference:

```text
h_l <- h_l + alpha * v_l
```

Creative translation:

```text
find an internal mood fader and move it while the model thinks
```

This is different from prompt engineering:

```text
prompting changes input language
activation steering changes computation
```

It is different from LatCH:

```text
LatCH reads or guides latent outputs
activation steering edits inner representations
```

Key danger:

```text
linear direction does not guarantee clean causal control
```

A "happy" vector might actually control brightness, tempo, density, or major-mode cues. The only honest answer is measurement.

## 23. LatCH: Make Control Predictable in Latent Space

A LatCH-style head:

```text
h_psi(z_t, t) -> y
```

or prompt-aware:

```text
h_psi(z_t, t, prompt) -> y
```

where `y` might be:

- loudness curve,
- beat probability,
- pitch bins,
- brightness,
- density,
- prompt adherence,
- tension label.

Creative translation:

```text
train a small dashboard that reads the model's latent audio before it becomes waveform
```

Use cases:

1. Measurement:

```text
what did the model generate?
```

2. Ranking:

```text
which seed best matches the control?
```

3. Guidance:

```text
push the trajectory toward desired control values
```

The important philosophical move:

```text
control begins as observability
```

If you cannot measure a property, you probably cannot steer it responsibly.

## 24. Branch-and-Rank: Brute Force With Taste

Generate candidates:

```text
z_i = generate(prompt, seed_i)
```

Score:

```text
s_i = score(z_i)
```

Select:

```text
i* = argmax_i s_i
```

Creative translation:

```text
let the model improvise multiple takes,
then become the producer or train a producer
```

This is underrated. It is simple, measurable, and often strong.

It also gives a baseline for any fancy method:

```text
does steering beat just sampling more candidates?
```

If not, the steering method is not yet earning its complexity.

## 25. Live Diffusion and Block Models: Time Becomes Infrastructure

Offline generation:

```text
generate z[0:T] all at once
```

Live generation:

```text
given committed context blocks z[0:k],
generate next block z[k+1]
```

Naive continuation recomputes too much and can drift.

LMDM-style approach:

```text
context -> cached representation
target noisy block -> denoise while attending to context
```

Creative translation:

```text
the model stops being a file renderer and starts becoming a bandmate with memory
```

This requires architecture, not just faster hardware:

- context routing,
- attention masks,
- cache policy,
- block-causal training,
- rollout stability,
- latency-aware samplers.

## 26. Method Evolution as One Big Equation

A modern audio system often looks like this:

```text
z = E(audio)
condition = C(prompt, duration, references, masks)
z_hat = Sampler_theta(noise, condition)
audio_hat = D(z_hat)
score = R(audio_hat, z_hat, condition)
```

Then intervention can happen at different places:

```text
prompt: change C
seed: change noise
sampler: change trajectory
latent: change z or z_hat
decoder: change D or postprocess
weights: change theta through external fine-tuning
activations: patch hidden state h_l
ranker: change R
dataset: change the world the model learns
```

Creative-technologist translation:

```text
the instrument is the whole pipeline,
not only the model weights
```

## 27. A Map of Math to Creative Knobs

| Math object | Equation shape | Creative knob | Risk |
|---|---|---|---|
| Autoregressive probability | `p(x_t | x_<t)` | temperature, sampling, context | slow, local |
| Spectral loss | `||STFT(x)-STFT(x_hat)||` | texture/fidelity pressure | phase blind spots |
| VAE latent | `q(z|x), p(x|z)` | interpolation, latent walking | blur, weak semantics |
| VQ codes | `nearest_code(E(x))` | token editing, code hierarchy | quantization artifacts |
| Attention | `softmax(QK^T)V` | context routing, prompt attention | hard to interpret |
| Diffusion noise | `x_t = a x_0 + b eps` | denoising trajectory | slow, schedule-sensitive |
| Score gradient | `grad_x log p(x)` | realism direction | abstract, unstable |
| CFG vector | `cond - uncond` | prompt strength | over-guidance |
| Flow field | `dz/dt = v_theta` | trajectory steering | field not directly visible |
| Inpainting mask | `p(z_missing | z_known,m)` | composition by holes | boundary artifacts |
| Activation vector | `h + alpha v` | internal fader | causal ambiguity |
| LatCH head | `h_psi(z)->y` | measurement/ranking/guidance | descriptor gaming |
| Branch ranker | `argmax score(z_i)` | producer selection | compute cost |
| Block cache | `context -> KV cache` | live continuation | architecture cost |

## 28. The Creative Technologist's Loop

A rigorous creative loop:

```text
1. choose representation
2. expose or learn controls
3. generate variations
4. measure descriptors
5. listen and annotate
6. probe latent/internal states
7. intervene lightly
8. compare against branch-and-rank
9. only then train adapters or new heads
```

This prevents the common trap:

```text
building a huge mechanism before knowing if the property is observable
```

For each desired control, ask:

```text
Can I hear it?
Can I measure it?
Can the latent predict it?
Can generation move it?
Can it move independently?
Can a human play it?
```

## 29. A Short Evolution Timeline of the Math

```text
Signal processing:
    Fourier, STFT, filters, phase, envelopes

Autoregressive modeling:
    p(x) = product p(x_t | x_<t)

Spectrogram modeling:
    model lower-rate time-frequency images

Autoencoding:
    z = E(x), x_hat = D(z)

VAE:
    smooth probabilistic latent space with KL pressure

GAN:
    adversarial perceptual realism

DDSP:
    differentiable synthesizer parameters

VQ/codecs:
    learned audio tokens and residual codebooks

Transformers:
    attention as learned routing across time and condition

Diffusion:
    learn to reverse noise corruption

Score/guidance:
    steer with gradients of realism and control

Latent diffusion:
    denoise compressed learned representations

Flow matching:
    learn vector fields from noise to data

External fine-tuning:
    style/domain adaptation outside this local instrument

Mechinterp:
    probes and activation steering over hidden states

Latent controls:
    LatCH, rankers, descriptor maps

Live/block generation:
    cache, masks, continuation, rollout dynamics
```

## 30. Closing Frame

The math evolution is not just academic. Each method changes what kind of artistic object the AI becomes.

```text
autoregression -> an improviser of next moments
spectrogram models -> a painter of frequency images
DDSP -> a differentiable synth player
VQ/codecs -> a writer of sound tokens
diffusion -> a sculptor of noisy fields
flow matching -> a navigator of latent motion
activation steering -> a playable internal circuit
LatCH -> a dashboard and steering proxy
LMDM -> a possible real-time partner
```

For an AI sound creative technologist, the central skill is not memorizing every architecture. It is learning to ask:

```text
What object is this model learning?
What loss shaped that object?
Where is the bottleneck?
Where is the trajectory?
Where can I measure?
Where can I intervene?
What changes the sound, and what only changes my story about the sound?
```

That is the useful math. It is not separate from the art. It is the map of where the instrument can be touched.
