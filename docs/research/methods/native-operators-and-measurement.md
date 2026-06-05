# Native Operators and Measurement for SA3/SAME Research

Status: current method note for the notebook's measurable SA3/SAME operators.

This note maps reusable operators to the measurements that decide whether they
are useful. The working rule is simple: an operator is not a control until it
moves a measurable signal and survives listening review.

The main objects are:

```text
SAME latent geometry     z
SA3 flow field           v_theta(z_t, t, c)
SA3 residual stream      a_l
control observations     y
```

## 1. Latent Geometry

Implemented primitives:

```text
latent_audio_primitives.geometry
```

Core equations:

```text
X = concat latent frames, X in R^{N x D}
mu = mean(X)
Sigma = Cov(X)
Sigma = Q Lambda Q^T
```

Projection:

```text
c = (z - mu) Q_k
```

Whitening:

```text
c_white = c / sqrt(lambda_k + eps)
```

Mahalanobis distance:

```text
d_M(a,b) = sqrt((mu_a - mu_b)^T Sigma^{-1} (mu_a - mu_b))
```

Audit quantities:

```text
explained_variance_k = lambda_k / trace(Sigma)
kept_variance_fraction = sum_k lambda_k / trace(Sigma)
```

Why it matters: this gives retrieval and intervention distances that respect
dataset covariance instead of raw Euclidean scale.

## 2. Optimal Transport / Barycenters

Implemented primitives:

```text
covariance_transport
latent_barycenter
```

Full-covariance Gaussian transport:

```text
z' = (z - mu_s) Sigma_s^{-1/2} Sigma_r^{1/2} + mu_r
```

Soft edit:

```text
z_alpha = (1 - alpha) z + alpha z'
```

Why it matters: this extends mean/std style profiles with cross-channel
covariance instead of treating every SAME channel as independent.

## 3. Fourier / Periodic Latent Operators

Implemented primitives:

```text
latent_audio_primitives.periodic
latent_audio_primitives.latent_dsp
```

Autocorrelation:

```text
rho(k) = <z_{0:T-k}, z_{k:T}> / ||z||^2
```

Latent-time FFT:

```text
Z_f = FFT_t(z)
energy(f) = mean_channels |Z_f|^2
```

Boundary loss:

```text
L_loop = ||mean(z_start) - mean(z_end)||_2
       + lambda ||mean(Delta z_start) - mean(Delta z_end)||_2
```

Why it matters: loopability and continuity should be measured through periodic
structure, not only by listening to repeated waveform previews.

The same latent-time frequency view now supports neural-latent DSP:

```text
Z_{c,k} = FFT_t(z_{c,t})
Z'_{c,k} = G_k Z_{c,k}
Z'_{c,k} = |Z_{c,k}| exp(i phi'_{c,k})
```

This gives FFT gain, phase shift, phase randomization, and donor
magnitude/source phase grafting over SAME trajectories. It is a modulation
operator over neural latents, not an audio EQ.

## 4. Direct Guidance During Sampling

Implemented low-level primitive:

```text
latent_audio_primitives.guidance.gradient_guidance_step
```

Generic update:

```text
z' = z - gamma grad_z L_control(z)
```

Sampler-level target:

```text
v_guided = v_theta(z_t,t,c) - gamma grad_{z_t} L_control(z_t)
```

Current status: the generic differentiable step exists. A full SA3
sampler-guidance mode still needs careful integration into the RF sampler loop
and validation against VRAM/runtime.

## 5. Native Prompt Inversion

Implemented in Mode 2:

```text
beam_token_prompt_search
SA3-native flow score
log-SNR probe bank
antithetic noise
normalized residual
cosine direction term
optional conditional-delta term
```

Main score:

```text
z_t = (1 - t)z_0 + t epsilon
u_t = epsilon - z_0
L(p) = E ||v_theta(z_t,t,C(p)) - u_t||^2
score(p) = -L(p)
```

Better score:

```text
L = L_norm
  + lambda_cos (1 - cos(v_theta, u_t))
  + lambda_delta (1 - cos(Delta_p, Delta_target))
```

where:

```text
Delta_p = v_theta(z_t,t,C(p)) - v_theta(z_t,t,C(""))
Delta_target = u_t - v_theta(z_t,t,C(""))
```

## 6. Residual Stream Feature Discovery

Implemented primitives:

```text
latent_audio_primitives.residual_features
```

Feature basis:

```text
A = stacked residual activations
A - mean(A) = U S V^T
features = V_k
```

Contrastive residual direction:

```text
v_l = mean(a_l^positive) - mean(a_l^reference)
```

Current status: feature fitting and projection exist as math primitives. Full
causal validation still requires generation-time activation patching and careful
handling of compiled model internals.

## 7. Control Observability

Implemented primitives:

```text
latent_audio_primitives.observability
```

Probe:

```text
s(z) = concat(mean_t z, std_t z, mean_t |Delta z|)
h(z) = w^T normalize(s(z)) + b
```

Ridge fit:

```text
min_{w,b} ||Xw + b - y||^2 + lambda ||w||^2
```

Intervention audit:

```text
effect = h(z_after) - h(z_before)
```

Why it matters: a control should be separated into:

```text
observability: can we measure y?
predictability: can h(z) predict y?
intervenability: can an edit reliably move h(z) and the audio?
```

## Current Colab Bridge

Mode 15 now performs a first geometry/intervention audit:

```text
PCA geometry
Mahalanobis summary distances
periodicity reports
optional covariance transport demo
optional linear control probes
```

This should be run before trusting a steering mode. If an edit has no measurable
latent effect, or a control is not predictable from latent summaries, it is not
ready to become a notebook control.

## Implementation Status Matrix

| Operator | Local Code | Colab Exposure | Needs SA3 Weights? | Training? |
|---|---|---:|---:|---:|
| Latent geometry | `geometry.py` | Mode 15 | no for saved latents, yes for fresh encoding | no |
| Covariance transport | `geometry.py` | Mode 15 demo | no for saved latents, yes for fresh encode/decode | no |
| Fourier/periodic latent probes | `periodic.py` | Mode 15 | no for saved latents | no |
| Neural latent DSP | `latent_dsp.py`, `audio_descriptors.py` | Mode 0h | yes for decode/polish | no |
| Direct gradient guidance | `guidance.py` | primitive only | yes for sampler integration | no base training |
| Prompt inversion | `prompt_optimization.py` | Mode 2 | yes | no |
| Residual feature discovery | `residual_features.py` | primitive only | yes for activation capture | no |
| Control observability | `observability.py` | Mode 15 | no for saved labelled latents | sidecar/probe only |

The deliberate gap is sampler integration. The math primitives are in place; the
next step is choosing operators with audible promise and then wiring them into
SA3 generation paths only after measurement and listening agree.
