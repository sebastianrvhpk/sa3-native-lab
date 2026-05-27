# Latent Audio Operator Family Map

Use this reference when translating latent-audio math into interface primitives.

## Base Objects

```text
x: audio waveform
E: audio encoder
D: audio decoder
z = E(x): neural audio latent
z in R^(C x T)
c: text conditioning
M: mask over channels, time, or latent regions
eps: noise or donor residual
O_phi: operator with parameters phi
```

## RENOISE

```text
z_t = sqrt(1 - sigma^2) * z + sigma * eps
z' = SA3_denoise_or_polish(z_t, c, steps, cfg)
```

Interface controls:

- sigma
- seed
- prompt or unconditional mode
- selected channels or time windows
- polish steps
- direct decode versus SA3 polish

## DONOR NOISE

```text
eps_d = normalize(z_d - smooth(z_d))
z_t = sqrt(1 - sigma^2) * z_s + sigma * eps_d
```

Use when testing whether another audio file can provide structured perturbation rather than Gaussian noise.

## GRAFT

```text
z' = z_s + alpha * M * (z_d - z_s)
```

Use for channel, time, or bandwise transfer from donor to source.

## BLUR AND SHARPEN

```text
blur_t(z)[i] = sum_k w_k * z[i - k]
z_blur = (1 - alpha) * z + alpha * blur_t(z)
z_sharp = z + beta * (z - blur_t(z))
```

Blur is a temporal or channel smoothing operator over neural latents. It is not guaranteed to correspond to audio low-pass filtering.

## LATENT DSP

```text
Z_c(f) = FFT_t(z_c)
Z'_c(f) = G_c(f) * exp(i * phi_c(f)) * Z_c(f)
z' = IFFT_t(Z')
```

Use carefully. Latent frequency is frequency over latent frame time, not direct waveform frequency.

## PROMPT SEARCH

```text
p* = argmax_p score(F_theta(p), target(z_audio))
```

This may produce readable or babble prompts. Treat it as reverse-conditioning search, not semantic transcription.

## MEMORY

```text
memory item = {
  latent_summary: pool_t(z),
  descriptors: d(z, x),
  annotations: human labels,
  recipe: operator provenance,
  audio_path: artifact
}
```

Memory supports retrieval, reuse, donor selection, and creative navigation. It does not steer the model by itself unless connected to an operator.
