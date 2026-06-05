# SA3/SAME Native Experimental Modes: Current Math and Implementation Notes

Status: current core math for this repo's Colab notebook and helper package.
This document describes experimental primitives built on top of the released
Stable Audio 3 code installed from the upstream repository. For the full current
repo/backlog map, see `docs/research/current/notebook-research-map-and-next-methods.md`.

Main notebook:

```text
colab/sa3_same_native_experimental_modes.ipynb
```

## Evidence Levels

- Confirmed SA3/SAME fact: matches released repo docs, model interfaces, or paper/model-card framing.
- Implemented primitive: local code exists in `latent_audio_primitives/` or the Colab notebook.
- Research hypothesis: plausible behavior being probed by the primitive.
- Unknown: behavior must be measured on audio outputs, latent metrics, and listening notes.

## Core Objects

Confirmed SA3/SAME framing:

```text
audio waveform          x
SAME encoder            E
SAME latent             z = E(x)
SA3 prompt conditioner  c = C(p)
SA3 flow field          v_theta(z_t, t, c)
SAME decoder            D
decoded audio           x_hat = D(z)
```

SAME latents are treated as:

```text
z in R^{B x C x T}
C = 256
T ~= audio_samples / 4096
latent_rate ~= 44100 / 4096 ~= 10.77 frames/sec
```

Local memory items use time-major arrays:

```text
z_memory in R^{T x D}
D = 256
```

The adapter conversion is:

```text
z_memory = z_sa3.transpose(time, channel)
z_sa3 = z_memory.transpose(channel, time)
```

## Frozen-Model Principle

Most modes keep SA3 and SAME frozen:

```text
theta_SA3 unchanged
theta_SAME unchanged
```

Only inference-time objects are changed:

```text
prompt text p
soft conditioning vectors c
initial latents / noise
SAME latent edits z -> z'
SA3 residual activations a_l -> a_l + alpha v_l
sidecar predictors h_psi for LatCH-style experiments
```

This matters because a successful result is evidence about existing latent affordances, not evidence that the base model was retrained.

## Native Flow-Matching Score

The central scoring primitive for prompt inversion and soft prompt optimization is a frozen SA3 teacher-forcing loss.

For a target SAME latent:

```text
z_0 = E(x_target)
```

Choose a noise sample:

```text
epsilon ~ N(0, I)
```

Choose a flow time:

```text
t in [0, 1]
```

Construct the straight-path latent:

```text
z_t = (1 - t) z_0 + t epsilon
```

One velocity convention is:

```text
u_t = epsilon - z_0
```

The opposite convention is:

```text
u_t = z_0 - epsilon
```

The notebook keeps this explicit as:

```text
FLOW_TARGET_CONVENTION = "noise_minus_data" or "data_minus_noise"
```

The native prompt score is:

```text
L_flow(p) = E_{t,epsilon} || v_theta(z_t, t, C(p)) - u_t ||^2
score(p) = -L_flow(p)
```

Interpretation:

```text
Good prompt = SA3's frozen vector field points in the target audio direction.
```

This is not a captioning model. It is a prompt-inversion surrogate over SA3's own latent dynamics.

## Better Mode 2 Prompt-Inversion Objective

Mode 2 now uses a more stable flow-matching approximation.

### Log-SNR Timesteps

Instead of arbitrary linear timesteps, Mode 2 can define probes in log-SNR:

```text
logSNR = log((1 - t) / t)
t = sigmoid(-logSNR)
```

Default:

```python
BABBLE_LOGSNR_VALUES = [2.0, 0.0, -2.0]
```

This gives clean-ish, middle, and noisy probes without overcommitting to one diffusion region.

### Shared Probe Bank

For one candidate comparison, all prompts see the same:

```text
t_k
epsilon_k
z_t,k
```

Only the prompt changes:

```text
v_theta(z_t,k, t_k, C(p_i))
```

This is the fair A/B test. Without shared probes, the score mixes prompt quality with random noise difficulty.

### Antithetic Noise

For each noise sample, Mode 2 can score:

```text
epsilon
-epsilon
```

This reduces Monte Carlo variance:

```text
L ~= 0.5 [L(epsilon) + L(-epsilon)]
```

Default:

```python
BABBLE_ANTITHETIC_NOISE = True
```

### Normalized Residual

Raw MSE can be dominated by high-energy target velocities. The normalized term is:

```text
L_norm =
|| v_theta(z_t,t,C(p)) - u_t ||^2
/ (||u_t||^2 + eps)
```

Default:

```python
BABBLE_NORMALIZE_MSE = True
```

### Velocity-Direction Term

Magnitude is not the only useful signal. Mode 2 adds a cosine direction penalty:

```text
L_cos = 1 - cos(v_theta(z_t,t,C(p)), u_t)
```

Combined:

```text
L = L_norm + lambda_cos L_cos
```

Default:

```python
BABBLE_COSINE_WEIGHT = 0.25
```

### Optional Conditional-Delta Term

The prompt-specific vector-field contribution can be approximated by subtracting a null-prompt prediction:

```text
Delta_p = v_theta(z_t,t,C(p)) - v_theta(z_t,t,C(""))
```

The desired prompt-specific contribution is:

```text
Delta_target = u_t - v_theta(z_t,t,C(""))
```

Optional penalty:

```text
L_delta = 1 - cos(Delta_p, Delta_target)
```

Combined:

```text
L = L_norm + lambda_cos L_cos + lambda_delta L_delta
```

Default is off because it adds null-prompt forwards:

```python
BABBLE_CONDITIONAL_DELTA_WEIGHT = 0.0
```

Suggested probe values:

```python
BABBLE_CONDITIONAL_DELTA_WEIGHT = 0.10  # exploratory
BABBLE_CONDITIONAL_DELTA_WEIGHT = 0.25  # stronger, slower
```

## Hard Prompt Search

Mode 2 uses tokenizer-derived candidate text pieces from SA3's T5Gemma conditioner when available:

```text
candidate vocabulary V = decoded tokenizer pieces filtered by readability/audio prior
```

Beam search keeps several partial prompts alive:

```text
beam_width = K
branch_factor = M
```

At each token position:

```text
for each beam b:
    expand b with M candidate tokens
score all expanded prompts with native SA3 flow score
keep top K
```

Default:

```python
BABBLE_SEARCH_STRATEGY = "beam"
BABBLE_BEAM_WIDTH = 4
BABBLE_BRANCH_FACTOR = 256
BABBLE_CANDIDATE_BATCH_SIZE = 128
BABBLE_TOKENS_GENERATED = 16
```

The GPU batch controls how many prompt candidates are scored per forward chunk:

```text
larger BABBLE_CANDIDATE_BATCH_SIZE -> more VRAM, fewer forward chunks
larger BABBLE_BRANCH_FACTOR        -> wider search
larger BABBLE_BEAM_WIDTH           -> less greedy, more compute
larger BABBLE_LOGSNR_VALUES        -> better probe coverage, more compute
```

## Mode Taxonomy

| Mode | Primitive | Object Edited | Training? | Main Equation |
|---|---|---:|---:|---|
| 0 | Renoise variation | SA3 init latent/noise | no | `z_T=(1-sigma)E(x)+sigma epsilon` |
| 0c | Latent-selective renoise | SAME channel/time mask | no | `z_T[M]=(1-sigma)z[M]+sigma epsilon[M]` |
| 0d | Latent blur/sharpen/filter | SAME latent `z` | no | `z'=z+alpha(f(z)-z)` |
| 0e | Cross-audio graft | SAME channel subset | no | `z'=z_a+alpha M*(z_b-z_a)` |
| 0f | Cyclic roll loop lab | rolled audio/latents | no | repair after cyclic roll |
| 0g | Cyclic roll inside denoising | SA3 sampler state | no | `x <- x + beta(0.5(x+R_s x)-x)` |
| 0h | Neural latent DSP | SAME latent trajectories | no | dynamics, FFT phase/gain, PCA gain |
| 1 | Audio -> soft prompt | conditioning tensor `c` | no base training | optimize `c` against `L_flow` |
| 2 | Audio -> babble prompt | hard tokens `p` | no | beam search over `score(p)=-L_flow(p)` |
| 3 | Audio -> readable prompt | constrained text | no | coordinate search over readable axes |
| 4 | Dataset -> soft prompt | shared `c_D` | no base training | `min_c E_i L_flow(z_i,c)` |
| 5 | Dataset -> prompt family | cluster prompts | no base training | cluster SAME summaries, optimize per cluster |
| 6 | SAME style profile | latent stats | no | AdaIN-style profile attraction |
| 7 | SAME direction | contrastive latent vector | no | `v=mean(z_pos)-mean(z_ref)` |
| 8 | Residual steering from prompts | SA3 DiT activations | no | `a_l <- a_l + alpha v_l` |
| 9 | Residual steering from audio | SA3 DiT activations | no | audio-labeled residual directions |
| 10 | Flow-state optimization | intermediate `z_t` | no base training | optimize flow-state loss |
| 11 | Inpainting/continuation | masked latent/audio | no | `p(z_missing | z_known, mask, c)` |
| 12 | LatCH-style head | sidecar `h_psi(z)` | sidecar only | `h_psi(z)->controls` |
| 14 | Latent memory | indexed latent items | no | retrieval over summaries/metadata |
| 15 | SAME geometry audit | latent set + control labels | no | PCA, periodicity, probes, transport |

## Mode 0: Renoise Variations

Encode arbitrary audio:

```text
z = E(x)
```

Partially renoise:

```text
z_T = (1 - sigma) z + sigma epsilon
```

Let SA3 sample back:

```text
z' = sample_theta(z_T, c, sigma)
```

Research hypothesis: small `sigma` explores a local neighborhood around the input, while larger `sigma` lets SA3 reinterpret the audio under the prompt prior.

## Mode 0c: Latent-Selective Renoise

Given a binary mask:

```text
M in {0,1}^{C x T}
```

Only masked channels/frames receive noise:

```text
z_T = (1 - M) z + M [(1 - sigma) z + sigma epsilon]
```

Direct SAME decode often sounds off-manifold. SA3 sampler polish is more useful because the model can reproject the edited state toward plausible latents.

Research hypothesis: channel subsets and frame subsets may reveal emergent control axes in SAME latent space.

## Mode 0d: Latent Blur, Sharpen, and Filters

General form:

```text
z' = z + alpha (f(z) - z)
```

Examples:

Temporal blur:

```text
f(z)_t = sum_k w_k z_{t+k}
```

Low-rank projection:

```text
Z ~= U_r S_r V_r^T
```

Unsharp mask:

```text
z' = z + alpha (z - blur(z))
```

FFT filter over latent time:

```text
Z_f = FFT_t(z)
Z'_f = g(frequency) Z_f
z' = IFFT_t(Z'_f)
```

Research hypothesis: SAME latent time contains musically meaningful low/high temporal variation components, but channel order is learned and must be treated as a probe, not a guaranteed topology.

## Mode 0e: Cross-Audio Latent Graft

Source and donor:

```text
z_a = E(x_a)
z_b = E(x_b)
```

Masked graft:

```text
z' = z_a + alpha M (z_b - z_a)
```

Research hypothesis: donor latent channels can transfer texture, density, or gesture families without full prompt conditioning.

## Mode 0h: Neural Latent DSP

Mode 0h treats SAME latents as a learned compressed signal:

```text
z in R^{B x C x T}
C = 256
T ~= seconds * 10.77
```

The key warning is:

```text
D(O_z(E(x))) != O_x(x)
```

so these operators are DSP-like probes over learned feature trajectories, not
ordinary waveform processors.

Latent gain:

```text
z' = c + g(z - c)
```

Latent compressor/expander:

```text
u = z - c
s = std_t(u)
m = |u|/(s+eps)

m_compress =
  m                           if m <= theta
  theta + (m-theta)/ratio     if m > theta

m_expand =
  m                           if m <= theta
  theta + (m-theta)*ratio     if m > theta

z' = c + makeup * u * (m'/m)
```

Latent soft clipping:

```text
z' = c + makeup * s * ceiling * tanh(drive * u/(s*ceiling)) / tanh(drive)
```

Latent-time FFT EQ:

```text
Z_{c,k} = FFT_t(z_{c,t})
Z'_{c,k} = G_k Z_{c,k}
z' = IFFT_t(Z')
```

Latent-time phase shift:

```text
Z'_{c,k} = Z_{c,k} exp(-i 2 pi k tau/T)
```

Donor magnitude/source phase graft:

```text
Z' = [(1-alpha)|Z_source| + alpha|Z_donor|] exp(i angle(Z_source))
```

PCA component gain:

```text
X = z^T
X - mu = U S V^T
X' = U diag(g) S V^T + mu
```

Mode 0h can direct-decode edited latents:

```text
x_direct = D(z')
```

or polish them through SA3:

```text
z_polished = sample_theta(init_data=z', init_noise_level=sigma, prompt=p)
x_polished = D(z_polished)
```

It records lightweight MIR descriptors after decode. These include RMS, spectral
centroid, rolloff, flatness, flux, band-energy ratios, stereo width, and stereo
correlation. The descriptors are audit signals for repeatability, not proof that
a subjective control exists.

## Mode 0f: Cyclic Time-Roll Loop Lab

This is the older boundary-repair experiment:

```text
R_s(z)_t = z_{(t-s) mod T}
```

One path rolls, repairs/polishes, then unrolls:

```text
z_rolled = R_s(z)
z_repaired = F_theta(z_rolled)
z_out = R_{-s}(z_repaired)
```

Another path rolls waveform audio and uses SA3 inpainting on the internal seam. This is useful but distinct from sampler-level cyclic constraints.

## Mode 0g: Cyclic Roll Inside Denoising

This is the closer analogue to old image tiling tricks.

Let the sampler state be:

```text
x_i in R^{C x T}
```

Define half-roll:

```text
R_s(x)_t = x_{(t-s) mod T}, s ~= T/2
```

Soft cyclic projection:

```text
P_beta(x) = x + beta (0.5(x + R_s x) - x)
```

Sampler step:

```text
x_{i+1/2} = x_i + Delta t_i v_theta(x_i,t_i,c)
x_{i+1} = P_beta(x_{i+1/2})
```

If `init_noise = 0`, the flow update may be zero, but the projection still changes the latent state:

```text
x_{i+1} = P_beta(x_i)
```

Research hypothesis: repeated cyclic projection pressures the latent trajectory toward half-roll consistency, which may produce loop-like continuity or half-period collapse depending on `beta`.

## Modes 1 and 4: Soft Prompt Optimization

Instead of searching hard text tokens, optimize a continuous conditioning object:

```text
c* = argmin_c E_{t,epsilon} L_flow(z_0,c)
```

The base text encoder, SA3 DiT, and SAME stay frozen. This is prompt tuning / inversion, not base-model training.

## Mode 3: Readable Prompt Search

Readable mode constrains prompts to hand-authored descriptor axes:

```text
p = base + modifiers
```

This is less expressive than Mode 2 but easier to use in normal SA3 prompting.

## Modes 6 and 7: SAME Statistical Controls

Style profile:

```text
mu_D = E_i mean_t z_i
sigma_D = E_i std_t z_i
```

AdaIN-like attraction:

```text
z_norm = (z - mu_z) / sigma_z
z' = (1-alpha)z + alpha (z_norm sigma_D + mu_D)
```

Contrastive SAME direction:

```text
v = mean_i summary(z_i^positive) - mean_j summary(z_j^reference)
z' = z + alpha v
```

Research hypothesis: some dataset-level timbral or structural properties are linearly visible in simple SAME statistics. Unknown until measured.

## Modes 8 and 9: SA3 Residual Steering

Prompt-derived direction:

```text
v_l = mean a_l(positive prompts) - mean a_l(negative prompts)
```

Audio-derived direction uses audio labels or audio sets rather than only prompt pairs.

Inference-time intervention:

```text
a_l <- a_l + alpha v_l
```

This edits internal DiT representations. It does not train SA3 weights. Hooking may fail under `torch.compile`; monkey-patching block forwards is more reliable based on audioscope-style reports.

## Mode 12: LatCH-Style Sidecar Heads

A LatCH-style head is a small predictor over SAME latents:

```text
h_psi(z, optional t) -> y_control
```

It can be trained from automatic descriptors or human labels:

```text
loss = ||h_psi(z) - y||^2
```

It is not initially part of SA3. Uses:

```text
observability: can the control be measured?
predictability: can h_psi predict it from z?
intervenability: can a sampler/edit change it reliably?
```

## Mode 13: Underfit LoRA Handoff

Mode 13 points LoRA work to [dada-bots/underfit](https://github.com/dada-bots/underfit)
for training, monitoring, checkpoint selection, and adapter-specific logic.

The external path is:

```text
dataset folder
-> metadata/prompt audit
-> caption staging folder
-> optional SAME pre-encoded latents
-> SA3 base LoRA/DoRA/BoRA training in Underfit
-> Underfit checkpoint/loss monitor
-> Underfit checkpoint auditions
```

This repo can import audio outputs or analysis artifacts from Underfit for
comparison against frozen SA3/SAME methods.

## Mode 15: SAME Geometry and Intervention Audit

Mode 15 is a measurement mode for the seven stronger operators documented in:

```text
docs/research/methods/seven-better-operators.md
```

It starts from a latent collection:

```text
{z_i}, z_i in R^{T_i x D}
```

and concatenates latent frames:

```text
X = concat_i z_i, X in R^{N x D}
mu = mean(X)
Sigma = Cov(X)
Sigma = Q Lambda Q^T
```

The report stores:

```text
explained_variance_k = lambda_k / trace(Sigma)
kept_variance_fraction = sum_{k in kept} lambda_k / trace(Sigma)
```

This matters because retained components explain only the measured retained
fraction of dataset variance.

Whole-clip latent distances use the PCA covariance:

```text
d_M(a,b) =
sqrt((mean_t z_a - mean_t z_b)^T Sigma^{-1} (mean_t z_a - mean_t z_b))
```

Periodicity probes use latent-time autocorrelation:

```text
rho(k) = <z_{0:T-k}, z_{k:T}> / ||z||^2
```

and boundary mismatch:

```text
L_boundary =
||mean(z_start)-mean(z_end)||_2
+ lambda ||mean(Delta z_start)-mean(Delta z_end)||_2
```

Optional covariance transport tests full-covariance style movement:

```text
z' = (z - mu_s) Sigma_s^{-1/2} Sigma_r^{1/2} + mu_r
```

Optional control probes use existing numeric descriptors or labels:

```text
s(z) = concat(mean_t z, std_t z, mean_t |Delta z|)
h(z) = w^T normalize(s(z)) + b
```

with ridge regression:

```text
min_{w,b} ||Xw + b - y||^2 + lambda ||w||^2
```

Interpretation:

```text
observable     -> descriptor/label y exists and is robust enough
predictable    -> h(z) predicts y better than a trivial baseline
intervenable   -> an edit changes h(z) and the heard audio in the intended way
```

Mode 15 does not prove a control exists. It tells us which latent statistics are
worth probing before building more invasive steering modes.

## Open Measurement Questions

- Does Mode 2's flow score correlate with actual generation similarity or only teacher-forced vector-field agreement?
- Which timestep/log-SNR probes are most predictive of audible prompt usefulness?
- Does conditional-delta scoring improve prompt specificity enough to justify the extra forward pass?
- Which SAME channels or channel groups repeatedly produce stable perceptual families under Mode 0c?
- Which latent filters in Mode 0d are merely destructive, and which become musical after SA3 polish?
- Which Mode 0h latent-DSP operators survive SA3 polish, and which are erased by the SA3 prior?
- Do latent dynamics or soft clipping reduce artifacts after harsher latent edits?
- Do latent FFT phase/magnitude grafts form reproducible variation families?
- Does Mode 0g produce useful cyclic continuity, or mostly half-period collapse?
- Which residual layers in SA3 carry stable mood, density, brightness, or section-role information?
- Which Mode 15 geometry signals are stable across chunk length, musical style, and dataset size?
- Which controls are observable in simple SAME summaries but fail under actual intervention?

## Implementation Safety Notes

- Direct SAME decode of heavily edited latents can sound broken because edits may leave the learned latent manifold.
- SA3 polish/noise can repair some off-manifold edits but may also rewrite content.
- High beam/search budgets can use substantial compute even when VRAM is modest.
- Prompt inversion outputs are steering strings, not necessarily human-readable captions.
- The current notebook is an experiment harness. Treat all controls as hypotheses until audio, latent metrics, and listening notes agree.
