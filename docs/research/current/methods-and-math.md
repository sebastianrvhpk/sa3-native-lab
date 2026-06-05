# SA3 Native Lab Methods and Math

Status: current technical specification for SA3/SAME notebook methods as of
2026-06-05.

This document answers: what objects are manipulated, which equations define the
notebook methods, how measurements are interpreted, and which conventions must
stay explicit.

## Evidence Levels

- Confirmed SA3/SAME fact: matches released repo docs, model interfaces, paper/model-card framing, or current notebook code.
- Implemented primitive: local code exists in `latent_audio_primitives/` or the Colab notebook.
- Research hypothesis: plausible behavior being probed by a primitive.
- Unknown: must be measured on decoded audio, latent metrics, descriptor deltas, and listening notes.

## Core Objects

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

Adapter conversion:

```text
z_memory = transpose z_sa3 from C x T to T x C
z_sa3 = transpose z_memory from T x C to C x T
```

## Frozen-Model Principle

Most experiments keep SA3 and SAME frozen:

```text
theta_SA3 unchanged
theta_SAME unchanged
```

Only inference-time objects are changed:

```text
prompt text p
soft conditioning vectors c
initial latents / noise
SAME latent edits z -> z_prime
SA3 residual activations a_l -> a_l + alpha v_l
sidecar predictors h_psi for LatCH-style probes
```

A successful result is evidence about existing latent affordances, not evidence
that the base model was retrained.

## Native Flow-Matching Score

For target audio:

```text
z_0 = E(x_target)
epsilon ~ N(0, I)
t in [0, 1]
z_t = (1 - t) z_0 + t epsilon
```

The target velocity convention must remain explicit:

```text
noise_minus_data: u_t = epsilon - z_0
data_minus_noise: u_t = z_0 - epsilon
```

Notebook setting:

```text
FLOW_TARGET_CONVENTION = "noise_minus_data" or "data_minus_noise"
```

Native prompt loss:

```text
L_flow(p) = E_{t,epsilon} || v_theta(z_t, t, C(p)) - u_t ||^2
score(p) = -L_flow(p)
```

Interpretation:

```text
Good prompt = SA3's frozen vector field points in the target audio direction.
```

This is prompt inversion through SA3's own latent dynamics: choose text whose
conditioning makes the frozen flow field agree with the target latent trajectory.

### LogSNR Timesteps

Conditioning and flow-microscope panels can use logSNR values:

```text
logSNR = log((1 - t) / t)
t = sigmoid(-logSNR)
```

Default probe idea:

```text
logSNR values [2.0, 0.0, -2.0]
```

This probes clean-ish, middle, and noisy flow regions without overcommitting to
one timestep.

### Shared Probe Bank

For candidate comparison, all prompts see the same:

```text
t_k
epsilon_k
z_t,k
```

Only the prompt changes:

```text
v_theta(z_t,k, t_k, C(p_i))
```

This makes prompt A/B scoring fair. Without shared probes, the score mixes
prompt quality with random noise difficulty.

### Antithetic Noise

For each noise sample, score both:

```text
epsilon
-epsilon
```

Monte Carlo estimate:

```text
L ~= 0.5 [L(epsilon) + L(-epsilon)]
```

### Normalized Residual

Raw MSE can be dominated by target velocity energy:

```text
L_norm =
|| v_theta(z_t,t,C(p)) - u_t ||^2
/ (||u_t||^2 + eps)
```

### Velocity-Direction Term

Magnitude is not the only useful signal. A direction term penalizes vector-field
misalignment:

```text
L_cos = 1 - cos(v_theta(z_t,t,C(p)), u_t)
```

### Optional Conditional-Delta Term

Compare the prompt-conditioned velocity change against the target change from
the blank/null condition:

```text
Delta_p = v_theta(z_t,t,C(p)) - v_theta(z_t,t,C(""))
Delta_target = u_t - v_theta(z_t,t,C(""))

L_delta = 1 - cos(Delta_p, Delta_target)
```

Combined score:

```text
L = L_norm
  + lambda_cos L_cos
  + lambda_delta L_delta
```

## Hard Prompt Search

Hard prompt search scores text candidates through the frozen flow
loss:

```text
score(prompt) = -L_flow(prompt)
```

Supported search patterns:

```text
coordinate_prompt_search
greedy_token_prompt_search
beam_token_prompt_search
prompt_seed_from_audio_path
```

Readable prompt search constrains the search to descriptor axes:

```text
p = base_prompt + selected modifiers
```

Tradeoff:

```text
`SA3_FLOW_CONDITIONING.hard_prompt_search` = more expressive but may become opaque
`SA3_FLOW_CONDITIONING.readable_prompt_search` = less expressive but usable as normal SA3 prompting
```

## Ontology Math

### `SAME_REPRESENTATION.neighborhood_renoise`: Local Neighborhood Sampling

Encode arbitrary audio:

```text
z = E(x)
```

Partially renoise:

```text
z_T = (1 - sigma) z + sigma epsilon
```

Sample back through SA3:

```text
z_prime = sample_theta(z_T, c, sigma)
```

Hypothesis: small `sigma` explores a local neighborhood around the input, while
larger `sigma` lets SA3 reinterpret the audio under the prompt prior.

### `SAME_REPRESENTATION.selective_renoise`: Channel-Selective Renoise

Given a binary mask:

```text
M in {0,1}^{C x T}
```

Only masked channels/frames receive noise:

```text
z_T = (1 - M) z + M [(1 - sigma) z + sigma epsilon]
```

Direct SAME decode often sounds off-manifold. SA3 sampler polish can reproject
the edited state toward plausible latents.

### `SAME_REPRESENTATION.blur_bottleneck`: Blur, Sharpen, and Filters

General form:

```text
z_prime = z + alpha (f(z) - z)
```

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
z_prime = z + alpha (z - blur(z))
```

FFT filter over latent time:

```text
Z_f = FFT_t(z)
Z_prime_f = g(frequency) Z_f
z_prime = IFFT_t(Z_prime_f)
```

Hypothesis: SAME latent time contains meaningful low/high temporal variation,
but channel order is learned and must be treated as a probe rather than a
guaranteed topology.

### `SAME_REPRESENTATION.audio_graft`: Cross-Audio Latent Graft

Source and donor:

```text
z_a = E(x_a)
z_b = E(x_b)
```

Masked graft:

```text
z_prime = z_a + alpha M (z_b - z_a)
```

Hypothesis: donor latent channels can transfer texture, density, or gesture
families without full prompt conditioning.

### `SAME_REPRESENTATION.neural_dsp`: Neural Latent DSP

SAME gives a learned latent signal:

```text
x audio -> z = E(x)
z in R^{B x C x T}
C = 256
T ~= seconds * 10.77
```

Latent-DSP path:

```text
x_prime = D(O_z(E(x)))
```

This is not waveform DSP:

```text
D(O_z(z)) != O_x(D(z))
```

Interpretation:

```text
waveform DSP: direct operation over pressure-signal samples
latent DSP: operation over learned feature trajectories
SA3 polish: prior-guided reprojection after latent edit
MIR audit: descriptor measurements after decode
```

Latent gain:

```text
z_prime = c + g (z - c)
```

Latent dynamics:

```text
u = z - c
s = std_t(u)
m = |u| / (s + eps)

m_compress =
  m                           if m <= theta
  theta + (m - theta)/ratio   if m > theta

m_expand =
  m                           if m <= theta
  theta + (m - theta)*ratio   if m > theta

z_prime = c + makeup * u * (m_prime / m)
```

Latent soft clipping:

```text
u = (z - c) / (s * ceiling)
z_prime = c + makeup * s * ceiling * tanh(drive * u) / tanh(drive)
```

Latent-time FFT EQ:

```text
Z_{c,k} = FFT_t(z_{c,t})
Z_prime_{c,k} = G_k Z_{c,k}
z_prime = IFFT_t(Z_prime)
```

Since SAME latent rate is about 10.77 Hz, this is modulation EQ over learned
feature trajectories, not audio EQ.

Latent-time phase shift:

```text
Z_prime_{c,k} = Z_{c,k} exp(-i 2 pi k tau/T)
```

Phase randomization:

```text
Z_prime_{c,k} = |Z_{c,k}| exp(i phi_prime_{c,k})
phi_prime = blend(phi, random_phi, alpha)
```

Donor magnitude/source phase graft:

```text
Z_source = FFT(z_source)
Z_donor = FFT(z_donor)
Z_prime = [(1-alpha)|Z_source| + alpha|Z_donor|] exp(i angle(Z_source))
```

PCA component gain:

```text
X = z^T in R^{T x C}
X - mu = U S V^T
X_prime = U diag(g) S V^T + mu
```

Direct SAME decode reveals the literal latent edit:

```text
x_direct = D(z_prime)
```

SA3 polish asks whether the edit can be pulled back toward the learned manifold:

```text
z_polished = sample_theta(init_data=z_prime, prompt=p, init_noise_level=sigma)
x_polished = D(z_polished)
```

Descriptor audit signals:

```text
rms_dbfs
peak
crest_factor_db
zero_crossing_rate
spectral_centroid_hz
spectral_bandwidth_hz
spectral_rolloff_hz
spectral_flatness
spectral_flux
low/mid/high energy ratios
stereo_width
stereo_correlation
```

These are measurement hooks, not subjective labels.

Practical neural-latent-DSP use:

```text
SAME_DSP_RUN_DIRECT_DECODE = False
SAME_DSP_RUN_SA3_POLISH = True
SAME_DSP_STEPS = 20
SAME_DSP_POLISH_NOISE = 0.04 to 0.10
```

For microscope/debugging:

```text
SAME_DSP_RUN_DIRECT_DECODE = True
SAME_DSP_RUN_SA3_POLISH = False
```

For donor experiments:

```text
SAME_DSP_DONOR_AUDIO = INPUT_DIR / "donor_9s.wav"
SAME_DSP_INCLUDE_DONOR_SPECS = True
```

Open neural-latent-DSP questions:

- Which latent DSP operators survive SA3 polish without being erased?
- Which descriptor deltas correlate with listening notes?
- Are PCA component gains reusable across clips, or mostly clip-local?
- Do latent FFT phase edits create repeatable variation families?
- Can donor magnitude/source phase grafting transfer activity without replacing identity?
- Can latent dynamics reduce artifacts after harsher blur or channel edits?
- Which operations are useful only as microscopes, and which become instrument controls?

### `SAME_REPRESENTATION.loop_repair`: Cyclic Time-Roll Loop Repair

Relocate a boundary into the interior, repair it, then restore the temporal
origin:

```text
R_s(z)_t = z_{(t-s) mod T}
z_rolled = R_s(z)
z_repaired = F_theta(z_rolled)
z_out = R_{-s}(z_repaired)
```

Waveform rolling plus inpainting is useful, but distinct from sampler-level
cyclic constraints.

### `CAUSAL_STEERING.cyclic_trajectory`: Cyclic Projection Inside Denoising

Sampler state:

```text
x_i in R^{C x T}
```

Half-roll:

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

Hypothesis: repeated cyclic projection may produce loop continuity or
half-period collapse depending on `beta`.

### `SA3_FLOW_CONDITIONING.soft_prompt_inversion` and `SA3_FLOW_CONDITIONING.dataset_soft_prompt`: Soft Prompt Optimization

Optimize continuous conditioning while base model weights stay frozen:

```text
c_star = argmin_c E_{t,epsilon} L_flow(z_0,c)
```

Single-audio soft prompt inversion targets one item. Dataset soft prompt
inversion targets a shared conditioning object.

### `DATASET_MEMORY_COMPOSITION.prompt_family`: Dataset Prompt Family

Cluster SAME summaries, then search prompt candidates per cluster:

```text
{z_i} -> summaries s_i -> clusters K_j -> prompt_j = argmax score(prompt | K_j)
```

The result is a family of prompts assigned to dataset regions.

### `SAME_REPRESENTATION.style_profile` and `SAME_REPRESENTATION.style_direction`: SAME Statistical Controls

Style profile:

```text
mu_D = E_i mean_t z_i
sigma_D = E_i std_t z_i
```

AdaIN-like attraction:

```text
z_norm = (z - mu_z) / sigma_z
z_prime = (1-alpha)z + alpha (z_norm sigma_D + mu_D)
```

Contrastive SAME direction:

```text
v = mean_i summary(z_i^positive) - mean_j summary(z_j^reference)
z_prime = z + alpha v
```

Hypothesis: some dataset-level timbral or structural properties are linearly
visible in simple SAME statistics.

### `CAUSAL_STEERING.prompt_residual` and `CAUSAL_STEERING.audio_residual`: SA3 Residual Steering

Prompt-derived direction:

```text
v_l = mean a_l(positive prompts) - mean a_l(reference prompts)
```

Audio-derived direction uses audio labels or audio sets rather than only prompt
pairs.

Inference-time intervention:

```text
a_l <- a_l + alpha v_l
```

This edits internal DiT representations without training SA3 weights. Hooking
may fail under optimized/compiled paths, so generation-time validation is
required.

### `CAUSAL_STEERING.flow_state_optimization`: Flow-State Optimization

Optimize an intermediate flow state or related latent variable while keeping
model weights frozen:

```text
z_t_star = argmin_z L_target(z, t, c)
```

This is a scaffold until the target loss and sampler integration prove useful.

### `DATASET_MEMORY_COMPOSITION.continuation` and `DATASET_MEMORY_COMPOSITION.bridge_search`: Continuation, Inpainting, Bridge Search

Continuation/inpainting treats known and missing regions separately:

```text
z = M z_known + (1-M) z_missing
```

Bridge search ranks candidate middle clips or latent items by boundary and
transition costs:

```text
cost(path) = cost_transition(a,b) + cost_transition(b,c) + optional loop terms
```

Useful bridge candidates must be measured and auditioned; transition cost is a
candidate generator.

### `CAUSAL_STEERING.control_head`: LatCH-Style Sidecar Heads

A sidecar head predicts controls over SAME latents:

```text
h_psi(z, optional t) -> y_control
```

Training target:

```text
loss = ||h_psi(z) - y||^2
```

The sidecar head is an auxiliary notebook model over SAME latents. It supports:

```text
observability: can the control be measured?
predictability: can h_psi predict it from z?
intervenability: can a sampler/edit change it reliably?
```

### `EXTERNAL_COMPARISON.underfit_handoff`: Underfit LoRA Handoff

LoRA training uses Underfit. Notebook comparison path:

```text
dataset folder
-> metadata/prompt audit
-> caption staging folder
-> optional SAME pre-encoded latents
-> SA3 base LoRA/DoRA/BoRA training in Underfit
-> Underfit checkpoint/loss monitor
-> Underfit checkpoint auditions
-> imported audio/checkpoints for this notebook
```

### `DATASET_MEMORY_COMPOSITION.memory_index`: Latent Memory

Memory items:

```text
LatentItem = {
  item_id,
  latent in R^{T x D},
  latent_rate,
  prompt,
  descriptors,
  labels,
  metadata
}
```

Retrieval can use latent summaries, descriptor targets, hybrid scores, and later
control lanes or geometry-aware donor selection.

### `SAME_REPRESENTATION.geometry_audit`: SAME Geometry and Intervention Audit

Start from a latent collection:

```text
{z_i}, z_i in R^{T_i x D}
```

Concatenate frames:

```text
X = concat_i z_i, X in R^{N x D}
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

Variance:

```text
explained_variance_k = lambda_k / trace(Sigma)
kept_variance_fraction = sum_{k in kept} lambda_k / trace(Sigma)
```

Mahalanobis summary distance:

```text
d_M(a,b) =
sqrt((mean_t z_a - mean_t z_b)^T Sigma^{-1} (mean_t z_a - mean_t z_b))
```

Covariance transport:

```text
z_prime = (z - mu_s) Sigma_s^{-1/2} Sigma_r^{1/2} + mu_r
z_alpha = (1 - alpha) z + alpha z_prime
```

Latent barycenter:

```text
z_bar = weighted average over aligned latent trajectories
```

### `SA3_FLOW_CONDITIONING.flow_attribution` and `SA3_FLOW_CONDITIONING.flow_timestep_panel`: Flow Microscope Panels

`SA3_FLOW_CONDITIONING.flow_attribution` computes token/phrase attribution over the same flow probe bank:

```text
contribution(token_i) = L(prompt without token_i) - L(prompt)
```

`SA3_FLOW_CONDITIONING.flow_timestep_panel` displays losses by timestep/logSNR:

```text
loss_row = {
  prompt,
  timestep,
  logSNR,
  mse,
  normalized_mse,
  cosine_penalty,
  total_loss
}
```

These panels turn prompt inversion into diagnostics rather than a single scalar.

### `EVIDENCE_DECISION_PROTOCOL.control_lanes`: SAME Control Lanes

Control lanes are time-varying notebook controls:

```text
control_lane = {
  name,
  rate_hz,
  values,
  confidence,
  metadata
}
```

Examples:

```text
rms / envelope
latent motion
latent channel energy
brightness
stereo width
onset density
periodicity
```

Lane status:

```text
measurement first
retrieval second
guidance/control only after observability and listening agree
```

### `DATASET_MEMORY_COMPOSITION.curriculum`: Dataset Memory Curriculum

Dataset memory can become a curriculum:

```text
memory items -> clusters -> representative rows -> heldout rows -> prompts/control targets
```

Use cases:

- choose representative clips,
- avoid overfitting to one source,
- build prompt/control families,
- test whether edits generalize to heldout memory rows.

### `SAME_REPRESENTATION.ot_style_transfer`: Latent OT Style Transfer Bench

Compare style/profile methods:

```text
mean/std profile attraction
contrastive direction
covariance transport
barycenter
```

Measurement should include descriptor deltas, nearest-memory rows, flow score
changes, and listening notes.

### `CAUSAL_STEERING.residual_feature_atlas`: Residual Feature Atlas

Residual feature basis:

```text
A = stacked residual activations
A - mean(A) = U S V^T
features = V_k
```

Contrastive residual direction:

```text
v_l = mean(a_l^positive) - mean(a_l^reference)
```

Atlas goals:

- rank layers by predictive control accuracy,
- measure feature projection,
- test causal interventions by generation-time patching.

### `SA3_FLOW_CONDITIONING.null_inversion`: Null-Condition Inversion Probe

Analogy to null-text inversion:

```text
optimize null/unconditional conditioning against target SAME flow
keep human prompt fixed
edit human prompt later
```

Goal: preserve source identity through the null branch while allowing prompt
edits through the conditional branch.

### `CAUSAL_STEERING.gradient_edit` and `CAUSAL_STEERING.audio_posterior`: Guidance and Posterior Guidance

Generic differentiable guidance:

```text
z_prime = z - gamma grad_z L_control(z)
```

Sampler-level target:

```text
v_guided = v_theta(z_t,t,c) - gamma grad_{z_t} L_control(z_t)
```

Example control loss:

```text
L_control(z_t) =
  lambda_boundary * loop_boundary_loss(z_t)
  + lambda_profile * ||summary(z_t) - target_profile||^2
  + lambda_period * periodicity_loss(z_t)
```

Audio-to-audio posterior-style guidance adds source/reference preservation:

```text
L = L_prompt_or_flow
  + lambda_preserve ||E(D(z_t)) - z_source||^2 over preserved regions
  + lambda_descriptor L_descriptor(z_t)
```

These are scaffolds until sampler integration and audio results are validated.

### `EXTERNAL_COMPARISON.cross_model`: Cross-Model Baseline Harness

Use fixed prompts and optional external generation commands:

```text
SA3 output
external model output
descriptor report
notebook player comparison
```

The harness is a comparison surface, not a claim that external models are
installed or managed by this repo.

## Measurement Logic

An operator is not a control until it:

```text
1. moves a measurable signal,
2. survives decoded audio/listening review,
3. repeats across more than one clip/seed/prompt,
4. avoids obvious copying or off-manifold artifacts.
```

Separate:

```text
observability: can we measure y?
predictability: can h(z) predict y?
intervenability: can an edit reliably move h(z) and the audio?
```

## Operator Status Matrix

| Operator | Local Code | Colab Exposure | Needs SA3 Weights? | Training? |
|---|---|---:|---:|---:|
| Latent geometry | `geometry.py` | `SAME_REPRESENTATION.geometry_audit` | no for saved latents, yes for fresh encoding | no |
| Covariance transport | `geometry.py` | `SAME_REPRESENTATION.geometry_audit`, `SAME_REPRESENTATION.ot_style_transfer` | no for saved latents, yes for fresh encode/decode | no |
| Fourier/periodic latent probes | `periodic.py` | `SAME_REPRESENTATION.geometry_audit` | no for saved latents | no |
| Neural latent DSP | `latent_dsp.py`, `audio_descriptors.py` | `SAME_REPRESENTATION.neural_dsp` | yes for decode/polish | no |
| Direct gradient guidance | `guidance.py` | `CAUSAL_STEERING.gradient_edit`, `CAUSAL_STEERING.audio_posterior` probes | yes for sampler integration | no base training |
| Prompt inversion | `prompt_optimization.py`, `flow_prompt.py` | `SA3_FLOW_CONDITIONING.hard_prompt_search`, `SA3_FLOW_CONDITIONING.readable_prompt_search`, flow microscopes | yes | no |
| Residual feature discovery | `residual_features.py` | `CAUSAL_STEERING.residual_feature_atlas` | yes for activation capture | no |
| Control observability | `observability.py` | `CAUSAL_STEERING.control_head`, `SAME_REPRESENTATION.geometry_audit` | no for saved labeled latents | sidecar/probe only |
| Control lanes | `control_lanes.py` | `EVIDENCE_DECISION_PROTOCOL.control_lanes` | no for saved latents/audio descriptors, yes for fresh encode/decode | no |
| Memory curriculum | `curriculum.py`, `index.py` | `DATASET_MEMORY_COMPOSITION.memory_index`, `DATASET_MEMORY_COMPOSITION.curriculum` | no for saved memory | no |

The deliberate gap is full sampler integration. The math primitives are in
place; the notebook should promote only operators with audible promise after
measurement and listening agree.

## Periodicity and Loop Metrics

Autocorrelation:

```text
rho(k) = <z_{0:T-k}, z_{k:T}> / ||z||^2
```

Latent-time FFT energy:

```text
Z_f = FFT_t(z)
energy(f) = mean_channels |Z_f|^2
```

Boundary loss:

```text
L_loop = ||mean(z_start) - mean(z_end)||_2
       + lambda ||mean(Delta z_start) - mean(Delta z_end)||_2
```

Loopability should be measured through periodic structure and listening to
repeated previews, not by one scalar alone.

## Implementation Safety Notes

- Keep velocity convention explicit for every flow/prompt objective.
- Use shared probe banks when comparing prompts.
- Do not treat SAME channel order as semantic topology.
- Distinguish direct SAME decode from SA3 polish.
- Treat descriptors as audit signals, not subjective truth.
- Treat residual hooks and sampler internals as version-sensitive.
- Do not promote a scaffold until real model-weight runs and listening notes support it.
