# SA3 Native Lab Methods and Math

Status: current technical specification for SA3/SAME notebook methods as of
2026-06-06.

This document answers: what native objects are manipulated, which transitions
the notebook tests, how measurements are interpreted, and which conventions
must stay explicit.

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

Local SAME summary convention:

```text
summary(z) = concat(
  mean_t z_t,
  std_t z_t,
  mean_t |z_t - z_{t-1}|
)
```

For ordinary SAME `D=256`, the default summary has dimension `3D = 768`.
This is not a learned semantic embedding. It is a deliberately plain baseline
for memory search, geometry rows, source-preservation checks, curriculum
clustering, and guidance scaffolds.

Boundary summaries use explicit start/end windows:

```text
boundary_summary(z, side="start"|"end", k)
  -> (mean boundary state, mean boundary velocity)
```

## Research-Layer Math

The same notebook can study SAME on its own, SA3 on its own over SAME-shaped
latents, SA3's internal trajectory, or the coupled SA3-over-SAME editing path.
These are the four research layers. Evidence utilities review the measurements
from all four without becoming a separate model-object layer. Keep the native
objects separate:

```text
SAME Representation:
  x -> E(x) = z0
  z0 -> D(z0) = x_hat
  z0 -> summaries / geometry / memory / control lanes

SA3 Flow and Conditioning:
  z_t = (1 - t) z0 + t epsilon
  (z_t, t, C(p)) -> v_theta(z_t, t, C(p))

SA3 Internal Trajectory:
  a_l = residual activation at layer l
  state_i -> state_i + Delta t_i v_theta(state_i, t_i, C(p))
  patched a_l or patched state_i -> changed output trajectory

SA3-over-SAME Coupled Editing:
  z0 -> edited z0'
  z0' -> SA3 polish / inpaint / continue -> z_out
  z_out -> D(z_out)
```

Placement rule:

```text
SAME-only claim = prove with direct decode / SAME rows first.
SA3-only claim = prove with flow, condition, residual, or trajectory rows first.
Coupled claim = compare direct SAME decode against SA3 polish.
Promoted claim = add descriptors, listening notes, and ledger decisions.
```

## Measurement Recipes

The notebook includes row-producing method cells over SA3/SAME native objects.
They are placed inside the relevant SAME, flow/conditioning, trajectory,
memory/composition, or coupled-editing sections. These are implemented
scaffolds, not promoted claims.

### SAME Bottleneck Tomography

Let `T_i` be a structured SAME-latent perturbation:

```text
z_i = T_i(z0)
delta_i = z_i - z0
x_i = D(z_i)
```

Tomography rows record:

```text
delta_rms_i = sqrt(mean(delta_i^2))
cos_i = cosine(flatten(z0), flatten(z_i))
descriptor_delta_i = descriptors(D(z_i)) - descriptors(D(z0))
```

Perturbation families include temporal blur, channel blur/dropout, low-rank
projection, latent-time FFT bands, noise radius, and latent dynamics. A family
is only meaningful if repeated descriptor/listening packets expose stable SAME
preservation or failure behavior.

### Coupled Edit Survival

For a SAME edit `T`:

```text
z_edit = T(z0)
z_plain = SA3_polish(z0)
z_polish = SA3_polish(z_edit)
```

The survival ratio is:

```text
r_survive = rms(z_polish - z0) / max(rms(z_edit - z0), eps)
```

Local labels are deliberately coarse:

```text
no_op_or_failed_edit, erased, preserved, amplified,
prior_dominated_or_unstable, prior_invention_or_plain_polish
```

This is a measurement of the coupled SAME/SA3 path. It should not be read as
proof that the latent edit is useful until direct decode, plain polish, method
polish, descriptors, and listening agree.

### Flow-Semantic Cartography

Given a prompt family `P_f`, reuse the shared flow probe bank and aggregate:

```text
L_band(p, b) = mean_{k in band b} L_flow(p; probe_k)
```

Bands are currently coarse logSNR regions:

```text
high_noise: logSNR < -2
mid_trajectory: -2 <= logSNR <= 2
low_noise: logSNR > 2
```

The point is not to claim semantic truth from text labels. The point is to ask
whether prompt families produce stable flow-field differences at different
trajectory regions.

### Latent Control System Identification

For an observed scalar lane or descriptor `y`, fit a plain probe on SAME
summaries:

```text
y_hat = w^T normalize(summary(z)) + b
R2_train = 1 - SSE / SST
```

A control is only observable when `R2_train` survives held-out examples. It is
only steerable when an intervention moves `y_hat` and decoded audio in the
intended direction.

### Stemless Source Cartography

For source `z_s`, donor `z_d`, selected channels or regions `M`, and amount
`a`:

```text
z_graft = (1 - M a) z_s + (M a) z_d
donor_pull = rms(z_graft - z_s) / max(rms(z_graft - z_s) + rms(z_graft - z_d), eps)
```

The self-graft control is mandatory:

```text
graft(z_s, z_s, M, a)
```

If the self-graft behaves like an arbitrary donor graft, the mask/edit math is
not measuring source content.

### Factor Atlas

The factor atlas is an evidence join, not a classifier:

```text
factor row = join(SAME rows, flow rows, trajectory rows, listening notes)
```

Candidate factors currently include rhythm, timbre, melody/harmony, and
density. A factor becomes worth using only when multiple microscopes agree.

### Long-Form Latent Composition

Composition remains a selector over boundary states:

```text
cost(a -> b) =
  w_state ||end_state(a) - start_state(b)||_2
  + w_velocity ||end_velocity(a) - start_velocity(b)||_2
```

Bridge and path rows rank candidates before audition:

```text
bridge_cost(a, m, b) = cost(a -> m) + cost(m -> b)
```

Low cost is a candidate for continuity, not proof of musical continuity.

### Prompt-Condition Geometry

For SA3 condition tensors or soft-prompt states:

```text
d(c_i, c_j) = ||flatten(c_i) - flatten(c_j)||_2
cos(c_i, c_j) = cosine(flatten(c_i), flatten(c_j))
```

Useful condition geometry should explain flow-score neighborhoods or soft-prompt
behavior. It is not enough for nearby text to be nearby in condition space.

### Sampler Physiology

Sampler physiology records the settings and observed path metadata:

```text
sampler_type, steps, init_noise, CFG, sigma range, logSNR range, timestep range
```

When sampler callbacks expose step records, rows summarize coverage and compare
the final latent against the source:

```text
output_delta = rms(z_out - z0)
```

These rows are an observed trajectory microscope. Exact sampler timestep
attribution still depends on upstream sampler metadata.

### Latent Constraint Library

A latent constraint is an inspectable scalar objective:

```text
J(z) = sum_i lambda_i (m_i(z) - target_i)^2
```

Initial constraints include reference distance, RMS, motion energy, loop
boundary distance, channel energy, and mean. A constraint is not a control until
before/after rows, direct decode, optional SA3 polish, descriptors, and
listening show bounded movement.

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

### SA3 Noise-State Inversion

This backlog direction should be stated as continuous noise-state optimization,
not literal seed inversion. A seed is a discrete recipe for sampling noise; the
research object is the sampled tensor or an explicit noise trajectory.

Let a frozen SA3 sampler map an initial noise tensor and conditioning to a final
latent:

```text
z_hat_0 = Phi_theta(epsilon, C(p), S)
```

where `S` is the sampler schedule and settings. The basic reconstruction form is:

```text
epsilon* = argmin_epsilon
  || Phi_theta(epsilon, C(p), S) - z_0 ||_2^2
  + lambda_prior R_prior(epsilon)
```

with `z_0 = SAME(x_target)`. A trajectory version replaces one tensor with
per-step corrections:

```text
{delta_k}* = argmin_{delta_k}
  || Phi_theta(epsilon + delta_0, C(p), S, {delta_k}) - z_0 ||_2^2
  + lambda_prior sum_k R_prior(delta_k)
```

Native evidence must include:

```text
target z0
optimized noise tensor or trajectory metadata
Gaussian-prior deviation
reconstruction distance
random-noise baseline
audio-to-audio baseline
descriptor and listening deltas
```

Only after reconstruction/preservation is credible should the notebook try
edited prompts from the optimized noise state.

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

This is the local straight-path amplitude log-ratio for
`z_t = (1 - t) z_0 + t epsilon`. Some diffusion papers use squared-power
logSNR, `log((1 - t)^2 / t^2)`, which differs by a factor of two. SA3 Native
Lab reports the local convention explicitly in probe manifests.

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

Code object:

```text
FlowProbeBank = {
  velocity_convention,
  shared_noise,
  antithetic_noise,
  seed,
  probe_count,
  probes: [{probe_index, timestep, logSNR, noise_seed, noise_sign}]
}
```

The bank stores coordinates and seeds, not tensors, so notebook cells can
display and serialize the evidence plan before running SA3.

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
hard prompt search = more expressive but may become opaque
readable prompt search = less expressive but usable as normal SA3 prompting
```

## Prompt Semantic Transparency

Prompt variants are evidence objects, not just strings:

```text
PromptVariant = {
  variant_id,
  prompt,
  semantic_tags,
  source,
  notes
}
```

Semantic tags name which aspect of the wording changed:

```text
material, gesture, time, energy, space, affect, production, metadata, instruction
```

Native prompt rows attach flow and listening evidence:

```text
PromptSemanticRow = PromptVariant
  + flow_loss
  + descriptor_delta_norm
  + listening_rating
  + decision
```

The claim is modest: tags help explain prompt changes and failures. They do not
replace frozen-SA3 flow loss, decoded-audio descriptors, or listening notes.

## Object Transition Math

This section is organized by native-object transition. The same transition may
act as a microscope, selector, or intervention candidate depending on the
evidence packet.

### Audio To SAME Neighborhood Sampling

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

### SAME Channel-Selective Renoise

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

### SAME Blur, Sharpen, And Filter Perturbations

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

### Cross-Audio SAME Channel Graft

Source and donor:

```text
z_a = E(x_a)
z_b = E(x_b)
```

Masked graft:

```text
z_prime = z_a + alpha M (z_b - z_a)
```

Here `alpha` is only the literal donor interpolation amount inside selected
SAME channels. It is not the SA3 sampler noise amount.

Optional SA3 polish is a second operation:

```text
z_polished = sample_theta(init_data=z_prime, prompt=p, init_noise_level=rho)
```

Self-graft control:

```text
z_b = z_a  ->  z_prime = z_a
```

If self-graft direct decode changes, the graft math or encoding path is wrong.
If only the polished self-graft changes, the change comes from SA3 polish/noise,
not donor transfer.

Hypothesis: donor latent channels can transfer texture, density, or gesture
families without full prompt conditioning.

### SAME Neural Latent DSP

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

### Cyclic Time-Roll Loop Repair

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

### SA3 Trajectory Cyclic Projection

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

### Target Audio To Soft Prompt Condition

Optimize continuous conditioning while base model weights stay frozen:

```text
c_star = argmin_c E_{t,epsilon} L_flow(z_0,c)
```

Single-audio soft prompt inversion targets one item. Dataset soft prompt
inversion targets a shared conditioning object.

### Dataset To Prompt Family

Cluster SAME summaries, then search prompt candidates per cluster:

```text
{z_i} -> summaries s_i -> clusters K_j -> prompt_j = argmax score(prompt | K_j)
```

The result is a family of prompts assigned to dataset regions.

### SAME Statistical Profiles And Directions

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

### Prompt Or Audio Examples To SA3 Residual Probing And Steering

Prompt-derived direction:

```text
v_l = mean a_l(positive prompts) - mean a_l(reference prompts)
```

Audio-derived direction uses audio labels or audio sets rather than only prompt
pairs.

Layer probe selector:

```text
x_i^l = pooled residual activation for example i at layer l
y_i in {positive, reference}
q_l(y | x) = linear logistic probe
score_l = stratified_cv_accuracy(q_l, {(x_i^l, y_i)})
```

Trajectory-window selector:

```text
W_w = observed forward-call window w
x_i^{l,w} = pooled residual activation for example i at layer l inside W_w
score_{l,w} = stratified_cv_accuracy(q_{l,w}, {(x_i^{l,w}, y_i)})
```

Sampler-timestep selector:

```text
s_k = upstream sampler callback row {step index k, timestep t_k, sigma_k}
x_i^{l,k} = pooled residual activation for example i at layer l during sampler step k
score_{l,k} = stratified_cv_accuracy(q_{l,k}, {(x_i^{l,k}, y_i)})
```

The probe is not a display accessory. It is the required selector that ranks
which layers visibly separate the contrast before any residual direction is
treated as a steering candidate. `logistic_cv` uses the same cross-validated
linear-probe idea as audioscope; the notebook implementation keeps a Torch
solver because the SA3 Colab runtime intentionally removes sklearn. The
`centroid_loo` probe remains a dependency-light diagnostic, not the preferred
layer selector. Sampler-timestep rows use the upstream SA3 sampler callback and
record `mapping_status`; `exact_one_call_per_step` is the clean attribution
case. Forward-call window rows remain a fallback microscope when sampler
metadata is unavailable or when extra model evaluations need grouping.

### Control-Lane Mechanistic Probing

Control-lane probes ask whether SA3 residual activations expose measurable
audio/SAME control lanes. The target is continuous, not a positive/reference
class label.

For a control lane:

```text
y(t) = lane value at audio or SAME latent time t
```

and a captured residual activation with feature dimension last:

```text
a_l^{c,i} = residual activation at layer l, observed forward call c, token i
```

the lane target is resampled to the token count inside each observed call:

```text
y^{c,i} = interpolate(y, i / token_count_c)
```

Layer probe:

```text
x_j = a_l^{c,i}
y_j = y^{c,i}
```

Fit a blocked-CV ridge probe:

```text
w_l, b_l = argmin_{w,b} sum_j (w_l^T x_j + b_l - y_j)^2
              + lambda ||w_l||_2^2
```

Score rows with held-out correlation, normalized MSE, and R2:

```text
score_l = corr(y_holdout, y_hat_holdout)
NMSE_l = MSE(y_holdout, y_hat_holdout) / var(y_holdout)
```

There are two evidence levels:

```text
token_blocked_cv:
  hold out contiguous token-time samples from the concatenated call trace

call_heldout_cv:
  hold out whole observed forward calls when more than one call is available
```

`token_blocked_cv` is the sensitive microscope: it asks whether a layer exposes
the lane along token-time. `call_heldout_cv` is the stricter check: it asks
whether the same lane/readout relation survives when complete hook calls are
held out. A row with strong token-blocked score but weak call-held-out score may
still be useful for inspection, but it should not be treated as a robust
selector for residual interventions.

Within a lane, row ranks prefer call-held-out correlation when it exists and
fall back to token-blocked correlation for single-call sampler-step cells.

Observed-call window probe:

```text
W_h = observed forward-call window h
score_{l,h} = score using only calls c in W_h
```

Sampler-timestep probe:

```text
s_k = sampler callback record {step index k, timestep t_k, sigma_k, logSNR_k}
C_k = token-preserving hook-call group mapped to sampler step k
score_{l,k} = score using token activations in C_k
```

This is exact timestep attribution only when the hook calls and sampler callback
records map cleanly. Rows therefore carry:

```text
mapping_status in {
  exact_one_call_per_step,
  grouped_calls_per_step,
  approximate_even_mapping
}
```

Observed-call windows remain an honest fallback microscope when sampler
metadata is unavailable or when multiple hook calls cannot be assigned exactly.
For continuous control-lane targets, sampler-step activations must preserve
token samples inside the step. Mean-pooled timestep vectors are valid for some
contrast diagnostics, but they collapse a time-varying lane to one sample and
are insufficient for lane regression.

Null controls:

```text
y_null(t) in {
  shuffled y(t),
  reversed y(t),
  random lane with mean/std of y(t)
}
```

The useful evidence is not only `score(y)`, but the margin:

```text
visibility_margin = score(y) - max_null_score(y_null)
call_heldout_visibility_margin =
  call_heldout_score(y) - max_call_heldout_null_score(y_null)
```

If a lane probe does not beat its null rows, the result is treated as overfit or
clip-specific noise.

Held-out prediction rows export samples:

```text
{lane, layer, window/timestep, sample_fraction, target y_j, prediction y_hat_j}
```

These rows are for visual diagnosis: a smooth curve fit over the active source
duration is more convincing than a single high average score.

Active/quiet contrast:

```text
d_l = mean(x_j | y_j >= q_active) - mean(x_j | y_j <= q_quiet)
cos_l = cosine(d_l, w_l)
```

The active-direction preview stores only compact feature-index summaries, not a
steering vector. It is a candidate microscope for later residual intervention,
not an intervention by itself.

Interpretation:

- high held-out correlation means a lane is linearly visible in a layer/window,
- call-held-out correlation means visibility is less likely to be an artifact
  of repeated lane curves across observed calls,
- low normalized MSE means the probe beats a variance baseline,
- a strong active/quiet delta means high-lane and low-lane regions occupy
  separated residual states,
- null margins test whether the lane relationship is stronger than trivial
  time/order controls,
- `exact_one_call_per_step` is the cleanest sampler-coordinate evidence; grouped
  or approximate rows must be described as less certain,
- none of these prove causal control until a later residual patch or steering
  sweep moves decoded audio without artifacts.

Residual-timestep cartography turns those probe rows into explicit selectable
cells:

```text
m_{l,k} = {
  layer l,
  sampler step k,
  timestep t_k,
  sigma_k,
  logSNR_k,
  score_{l,k},
  mapping_status
}
```

The trajectory map is a ranked set:

```text
M = sort_by_score({m_{l,k}})
```

It is a microscope and selector. It says where a contrast is visible in SA3's
observed internal trajectory. It does not, by itself, prove causal control.

Trajectory-selected flow probes:

```text
{m_{l,k}} -> {t_k, logSNR_k, noise_seed_k, noise_sign_k}
```

These probes can focus prompt scoring or soft-prompt optimization on sampler
regions where SA3 internally separates the relevant contrast.

Trajectory-gated residual steering:

```text
alpha_{l,k} = alpha * score_{l,k} / max score
a_l(k) <- a_l(k) + alpha_{l,k} v_l
```

This is safer than whole-run steering only if the chosen cells repeat across
seeds/examples and the resulting alpha sweep survives audio review.

Trajectory-gated cyclic projection:

```text
beta_k = beta * score_k / max score
z_{k+1} <- cyclic_mix(z_{k+1}, beta_k)
```

Here `k` is sampler step, not audio time. The schedule changes when the sampler
receives cyclic pressure; it does not say which part of the waveform is cyclic.

Inference-time intervention:

```text
a_l <- a_l + alpha v_l
```

This edits internal DiT representations without training SA3 weights. Hooking
may fail under optimized/compiled paths, so generation-time validation is
required.

### Target Objective To Flow-State Optimization

Optimize an intermediate flow state or related latent variable while keeping
model weights frozen:

```text
z_t_star = argmin_z L_target(z, t, c)
```

This is a scaffold until the target loss and sampler integration prove useful.

### Memory And Masked Latents To Continuation Or Bridge

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

### SAME Latents To Control Sidecar Heads

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

### External Training Artifacts To Notebook Comparison

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

### Latent Items To Memory Search

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

### SAME Collections To Geometry And Intervention Audit

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

### Flow Probe Bank To Attribution And Timestep Panels

Flow attribution computes token/phrase attribution over the same flow probe bank:

```text
contribution(token_i) = L(prompt without token_i) - L(prompt)
```

The flow timestep panel displays losses by timestep/logSNR:

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

### Native Evidence Disagreement Rows

When SAME distance, nearest-memory evidence, flow loss, descriptors, and
listening notes disagree, keep the conflict visible:

```text
DisagreementRow = {
  item_id,
  same_distance,
  same_neighbor_score,
  flow_loss,
  descriptor_delta_norm,
  listening_rating,
  conflict_score,
  decision,
  notes
}
```

The conflict score is only a triage sort. The decision remains a listening and
ledger decision.

### Audio Or SAME Latents To Control Lanes

Control lanes are time-varying notebook measurements that can become selectors
or masks. They are not assumed to be reliable controls until decoded audio,
flow rows, and listening agree.

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
audio confidence from RMS
latent motion
latent channel energy
individual latent-channel traces
spectral flux
spectral centroid and bandwidth
spectral entropy / flatness / contrast
low / mid / high spectral-density band fractions
onset density
```

Core lane measurements:

```text
latent_motion_energy(t) = RMS_c(z_t,c - z_{t-1,c})
latent_channel_energy(t) = RMS_c(z_t,c)
audio_confidence(t) = clamp((RMS_dB(t) - floor_dB) / (full_dB - floor_dB), 0, 1)
spectral_flux(t) = RMS_f(norm(|STFT_t|)_f - norm(|STFT_{t-1}|)_f)
spectral_density_band(t) = sum_{f in band} |STFT_t(f)|^2 / sum_f |STFT_t(f)|^2
spectral_entropy(t) = -sum_f p_t(f) log p_t(f) / log(F)
spectral_flatness(t) = geometric_mean(|STFT_t|^2) / arithmetic_mean(|STFT_t|^2)
spectral_contrast(t) = percentile_90(dB_t(f)) - percentile_10(dB_t(f))
```

The audio-confidence lane gates features that become unstable in near silence,
especially spectral centroid and zero-crossing rate. Active masking uses the
confidence signal itself, not a display-normalized lane value.

Active source masking:

```text
active_mask(t) = 1[audio_confidence(t) >= tau]
active_span = first/last active frame under active_mask
```

When a requested notebook duration exceeds the source duration, padded tails can
inflate correlations by making every lane go flat together. Active-window
summary and correlation rows therefore use `active_mask` and report
`active_only=true`.

Lane comparison:

```text
reference lanes + candidate lanes
-> matched names
-> frame alignment
-> confidence-weighted distance, similarity, delta rows
```

Lane correlation:

```text
lane_a, lane_b, active_mask
-> confidence-weighted active-window correlation and cosine similarity
```

Lane region selection:

```text
lane -> peaks / stable / silence / above / below regions
regions -> time mask
time mask + latent edit -> lane-masked latent intervention
```

Every lane can export its own region rows. Audio-event regions such as RMS,
spectral flux, onset density, or band-density peaks can be compared against SAME
event regions such as latent motion or latent-channel energy by overlap and
center-distance rows. This helps separate "audio event occurred" from "SAME
latent state moved" without treating either as a control yet.

Channel atlas:

```text
z0 -> per-channel rms, mean_abs, std, motion_energy, peak_abs
top channels -> individual channel lanes / heatmap
```

Lane status:

```text
measurement first
comparison / retrieval / bridge ranking second
mask-based latent edits third
guidance/control only after repeated observability and listening agree
```

### Memory Rows To Dataset Curriculum

Dataset memory can become a curriculum:

```text
memory items -> clusters -> representative rows -> heldout rows -> prompts/control targets
```

Use cases:

- choose representative clips,
- avoid overfitting to one source,
- build prompt/control families,
- test whether edits generalize to heldout memory rows.

### SAME Statistics To Latent OT Style Transfer

Compare style/profile methods:

```text
mean/std profile attraction
contrastive direction
covariance transport
barycenter
```

Measurement should include descriptor deltas, nearest-memory rows, flow score
changes, and listening notes.

### Residual Activations To Feature Atlas

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

- rank layers with cross-validated residual probes before steering,
- rank exact sampler-timestep residual cells when callback metadata is available,
- rank layer/window cells with cross-validated residual trajectory probes,
- rank layers by predictive control accuracy,
- measure feature projection,
- test causal interventions by generation-time patching.

### Target Audio To Null-Condition Inversion

Analogy to null-text inversion:

```text
optimize null/unconditional conditioning against target SAME flow
keep human prompt fixed
edit human prompt later
```

Goal: preserve source identity through the null branch while allowing prompt
edits through the conditional branch.

### Objective Recipes To Guidance And Posterior Guidance

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

### External Outputs To Cross-Model Baseline Packets

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

| Operator | Local Code | Notebook Workbench | Needs SA3 Weights? | Training? |
|---|---|---:|---:|---:|
| Latent geometry | `geometry.py` | SAME representation bench | no for saved latents, yes for fresh encoding | no |
| Covariance transport | `geometry.py` | SAME representation / coupled editing benches | no for saved latents, yes for fresh encode/decode | no |
| Fourier/periodic latent probes | `periodic.py` | SAME representation bench | no for saved latents | no |
| Neural latent DSP | `latent_dsp.py`, `audio_descriptors.py` | SAME representation / coupled editing benches | yes for decode/polish | no |
| Direct gradient guidance | `guidance.py` | SA3 internal trajectory science | yes for sampler integration | no base training |
| Latent constraints | `latent_constraints.py` | SAME representation / SA3 internal trajectory / coupled editing benches | no for tensor-only objective rows, yes for decode/polish evidence | no |
| Noise-state inversion | backlog only; future sampler/noise procedure | SA3 internal trajectory science | yes | no base training |
| Prompt inversion | `prompt_optimization.py`, `flow_prompt.py`, `procedures/flow_scoring.py` | SA3 flow/conditioning science | yes | no |
| Residual probe rows/vectors | `residual_probes.py`, residual procedures | SA3 internal trajectory science | yes for activation capture | no |
| Residual feature discovery | `residual_features.py` | SA3 internal trajectory science | yes for activation capture | no |
| Control observability | `observability.py` | SAME representation / SA3 internal trajectory benches | no for saved labeled latents | sidecar/probe only |
| Control lanes | `control_lanes.py`, `evidence/control_lane_rendering.py` | Evidence packet setup / SAME memory and composition bench | no for saved latents/audio descriptors, yes for fresh encode/decode | no |
| Memory curriculum | `curriculum.py`, `index.py` | SAME memory and composition bench | no for saved memory | no |

The deliberate gap is full sampler integration. The math primitives are in
place; the notebook should promote only operators with audible promise after
measurement and listening agree.

Procedure maturity is review metadata, not math. Record it in notebook
narrative cells, experiment manifests for specific runs, and ledger decisions,
not as a package-level registry.

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
