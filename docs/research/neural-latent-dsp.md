# Neural Latent DSP Over SAME/SA3

Status: implemented as experimental primitives and Colab Mode 0h. These are
research operators over neural compressed signals, not official SA3/SAME
features.

## Core Frame

SAME gives a learned latent signal:

```text
x audio -> z = E(x)
z in R^{B x C x T}
C = 256
T ~= seconds * 10.77
```

The basic latent-DSP path is:

```text
x' = D(O_z(E(x)))
```

This is not equivalent to waveform DSP:

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

## Implemented Modules

```text
latent_audio_primitives.latent_dsp
latent_audio_primitives.audio_descriptors
```

Colab exposure:

```text
Mode 0h. Neural latent DSP playground
```

## Latent Gain

Scale latent excursions around a center:

```text
z' = c + g (z - c)
```

where:

```text
c = 0
c = mean_t z[:, :, t]
c = mean_{c,t} z
```

Possible audible effects:

```text
g < 1  -> less latent motion, potentially more stable/flattened
g > 1  -> exaggerated latent gesture, potentially more intense/unstable
```

Unknown: which channels or PCA directions correspond to loudness, density,
brightness, rhythm, or timbral pressure.

## Latent Dynamics

Normalize each channel's excursions:

```text
u = z - c
s = std_t(u)
m = |u| / (s + eps)
```

Compressor above threshold:

```text
m' =
  m                                  if m <= theta
  theta + (m - theta) / ratio        if m > theta
```

Expander above threshold:

```text
m' =
  m                                  if m <= theta
  theta + (m - theta) * ratio        if m > theta
```

Apply signed gain:

```text
z' = c + makeup * u * (m' / m)
```

Why it is interesting: extreme latent excursions may encode salient events,
instability, transients, density spikes, or off-manifold damage. Compression and
expansion make those excursions controllable without assuming they are audio
amplitude.

## Latent Soft Clipping

Soft limiting with a tanh curve:

```text
u = (z - c) / (s * ceiling)
z' = c + makeup * s * ceiling * tanh(drive * u) / tanh(drive)
```

This is a nonlinear latent waveshaper. It may stabilize harsh edits or erase
useful extremes. Treat it as an operator to audition and measure.

## Latent-Time FFT EQ

Apply FFT along SAME latent time, independently per channel:

```text
Z_{c,k} = FFT_t(z_{c,t})
Z'_{c,k} = G_k Z_{c,k}
z' = IFFT_t(Z')
```

Normalized frequency:

```text
0.0 = DC / static latent offset
1.0 = Nyquist at latent frame rate
```

Since SAME latent rate is about 10.77 Hz, this is not audio EQ. It is modulation
EQ over learned feature trajectories.

Interpretation:

```text
low bins   -> slow phrase/texture drift
mid bins   -> repeated gestural motion
high bins  -> fast latent-frame detail, jitter, transient-like structure
```

## Latent-Time Phase

Fractional circular shift:

```text
Z'_{c,k} = Z_{c,k} exp(-i 2 pi k tau / T)
```

Phase randomization:

```text
Z'_{c,k} = |Z_{c,k}| exp(i phi'_{c,k})
phi' = blend(phi, random_phi, alpha)
```

Phase blend with donor:

```text
Z'_k = |Z_source,k| exp(i blend_angle(phi_source,k, phi_donor,k))
```

Why it is interesting: magnitude describes how much latent modulation exists at
each temporal rate; phase describes where those modulation events sit in time.
This may generate families of variations that preserve motion energy while
rewriting temporal placement.

## Magnitude/Phase Graft

Donor magnitude, source phase:

```text
Z_source = FFT(z_source)
Z_donor = FFT(z_donor)
Z' = [(1-alpha)|Z_source| + alpha|Z_donor|] exp(i angle(Z_source))
```

Creative reading:

```text
keep source temporal organization
borrow donor latent modulation spectrum
```

The inverse, source magnitude with donor phase, is exposed as phase blending.

## PCA Component Gain

For each latent clip, work over the time-major matrix:

```text
X = z^T in R^{T x C}
X - mu = U S V^T
```

Apply gains to principal components:

```text
X' = U diag(g) S V^T + mu
```

This is like a learned macro-EQ for the strongest latent trajectory axes of a
clip. It is not global unless the PCA basis is fit over a dataset; Mode 0h uses
per-clip PCA for immediate exploration.

## SA3 Polish

Direct SAME decode:

```text
x_direct = D(z')
```

SA3 polish:

```text
z_polished = sample_theta(init_data=z', prompt=p, init_noise_level=sigma)
x_polished = D(z_polished)
```

The direct path reveals what the latent edit literally did. The SA3 path asks
whether the edit can be pulled back toward the model's learned latent manifold.

## MIR Audit

After decode, Mode 0h records:

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

These are not subjective labels. They are measurement hooks for asking:

```text
Did the edit actually move the audio?
Did variants cluster by descriptor?
Did a latent delta predict an audible family?
Did SA3 polish erase or preserve the edit?
```

## Practical Use

Start with:

```text
LATENT_DSP_RUN_DIRECT_DECODE = False
LATENT_DSP_RUN_SA3_POLISH = True
LATENT_DSP_STEPS = 20
LATENT_DSP_POLISH_NOISE = 0.04 to 0.10
```

For microscope/debugging:

```text
LATENT_DSP_RUN_DIRECT_DECODE = True
LATENT_DSP_RUN_SA3_POLISH = False
```

For donor experiments:

```text
LATENT_DSP_DONOR_AUDIO = INPUT_DIR / "donor_9s.wav"
LATENT_DSP_INCLUDE_DONOR_SPECS = True
```

## Research Questions

- Which latent DSP operators survive SA3 polish without being erased?
- Which descriptor deltas correlate with listening notes?
- Are PCA component gains reusable across clips, or mostly clip-local?
- Do latent FFT phase edits create repeatable variation families?
- Can donor magnitude/source phase grafting transfer activity without replacing identity?
- Can latent dynamics reduce artifacts after harsher blur or channel edits?
- Which operations are useful only as microscopes, and which become instrument controls?
