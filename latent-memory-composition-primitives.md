# Latent Memory to Composition: Primitive Architecture

This document turns the previous research ideas into implementation primitives. It is intentionally staged: the first code layer works with already-computed latents or synthetic latents. SAME/SA3 integration comes later, after the local math and data contracts are stable.

Core idea:

```text
memory becomes composition when retrieval produces actions over time
```

The compositional object is not only an audio file. It is:

```text
audio
latent sequence
descriptors
labels
boundaries
prompts
human notes
generation metadata
```

## System Layers

```text
Layer 0: Data contracts
    LatentItem, descriptors, labels, metadata

Layer 1: Latent math
    summaries, distances, velocities, boundaries, loop costs

Layer 2: Memory index
    nearest neighbors, descriptor filters, target-control search

Layer 3: Composition primitives
    continuation, bridge, contrast, variation, loop, path search

Layer 4: Model adapters
    SAME encode/decode, SA3 generate/inpaint/continue

Layer 5: Learned sidecars
    LatCH heads, prompt-aware rankers, preference models

Layer 6: Interactive workflows
    branch-and-rank, timeline assembly, live rolling continuation
```

The initial implementation covers Layers 0-3. Layers 4-6 should be added only after the primitives are tested on synthetic and precomputed latent data.

## Data Contract

A latent memory entry:

```text
LatentItem:
    item_id: stable unique id
    latent: time-major array, shape T x D
    latent_rate: frames/sec
    sample_rate: optional waveform sample rate
    prompt: optional generation or source prompt
    descriptors: numeric automatic descriptors
    labels: manual or categorical labels
    metadata: arbitrary provenance
```

For SAME:

```text
SAME output commonly appears as C x T with C = 256
local primitive format is T x D
```

So:

```text
z_time_major = z_same.T
```

## Latent Summaries

Given latent sequence:

```text
z in R^(T x D)
```

Whole-clip summary:

```text
mu = mean_t z[t]
sigma = std_t z[t]
velocity = z[t+1] - z[t]
mean_speed = mean_t |velocity[t]|
```

Implementation vector:

```text
summary(z) = concat(mu, sigma, mean_abs_velocity)
```

This is intentionally simple. It is the baseline that future learned embeddings must beat.

## Retrieval

Nearest-neighbor retrieval:

```text
score(q, i) = cosine(summary(q), summary(i))
```

or distance:

```text
d(q, i) = || summary(q) - summary(i) ||_2
```

Multi-objective retrieval:

```text
score(q, i)
= w_latent * cosine(summary(q), summary(i))
- w_distance * euclidean(summary(q), summary(i))
+ w_controls * control_score(target, descriptors_i)
```

This allows searching by:

- latent similarity,
- target descriptors,
- prompt/tags later,
- transition compatibility later.

## Boundary Compatibility

For a source item `A` and candidate next item `B`:

```text
end_A = mean(z_A[T-k:T])
start_B = mean(z_B[0:k])
vel_A = mean(diff(z_A[T-k:T]))
vel_B = mean(diff(z_B[0:k]))
```

Transition cost:

```text
C(A -> B)
= w_state || end_A - start_B ||_2
+ w_velocity || vel_A - vel_B ||_2
+ w_control || controls_A_end - controls_B_start ||_2
```

The initial implementation includes state and velocity. Control trajectories can be added once framewise descriptors exist.

Creative meaning:

```text
find a next section whose beginning lives near the current section's ending
and whose motion continues the current gesture
```

## Loop Cost

A latent loop should have compatible beginning and ending:

```text
loop_cost(z)
= w_state || mean(z[0:k]) - mean(z[T-k:T]) ||_2
+ w_velocity || mean(diff(z[0:k])) - mean(diff(z[T-k:T])) ||_2
```

This is not a complete audio-loop metric. It is a latent pre-filter. Audio-domain click/beat checks come later.

## Bridge Search

Given start `A`, bridge candidate `B`, and end `C`:

```text
bridge_cost(A, B, C)
= transition_cost(A -> B)
+ transition_cost(B -> C)
+ optional target_control_cost(B)
```

Creative meaning:

```text
find a fragment that can leave A and arrive at C
```

Later SA3 integration:

```text
retrieved bridge -> prompt/reference/control target -> SA3 inpaint/continue
```

## Composition Graph

Each latent segment is a node. Directed edge cost:

```text
edge(i, j) = transition_cost(i -> j)
```

Then a composition path is:

```text
path = [node_0, node_1, ..., node_n]
cost(path) = sum edge(node_k, node_{k+1})
```

Use Dijkstra or beam search to find low-cost paths. Later, add constraints:

- target energy curve,
- section roles,
- avoid repeated source,
- require contrast after similarity,
- prefer human favorites,
- match prompt cluster.

## Primitive List

### Retrieve Similar

```text
query latent -> nearest latent summaries
```

Purpose: find sounds living in the same latent neighborhood.

### Retrieve Contrast

```text
query latent + target descriptor shift -> candidates far in selected controls but near in style
```

Purpose: variation with controlled difference.

### Retrieve Continuation

```text
current ending -> candidate starts with low transition cost
```

Purpose: what can happen next?

### Retrieve Bridge

```text
A ending + C beginning -> candidate B with low A->B and B->C cost
```

Purpose: connect two musical states.

### Retrieve Loop

```text
rank items by loop_cost
```

Purpose: find or generate loopable material.

### Dataset Style Push

```text
dataset audio -> SAME memory -> style profile -> latent edit -> SAME decode
```

Purpose: use an arbitrary dataset as an actual direction, not a ranker.

Implemented with:

- `LatentStyleProfile`
- `LatentStyleDirection`
- `fit_style_profile`
- `apply_profile_attraction`
- `apply_style_direction`

Profile attraction:

```text
z = generated SAME latent
mu_z = mean_t z[t]
sigma_z = std_t z[t]

z_style = sigma_dataset * (z - mu_z) / sigma_z + mu_dataset
z_final = (1 - alpha) z + alpha z_style
```

Target-minus-reference direction:

```text
v_mean = mean(target_dataset) - mean(reference_dataset)
z_final[t] = z[t] + alpha * v_mean
```

This is direct latent steering. It does not require batch generation and
filtering. It also does not modify SA3's denoising trajectory yet; that later
requires sampler-time guidance.

### Audio-Derived Steering Vectors

Prompt-derived audioscope vectors live in SA3 residual space:

```text
h_l <- h_l + alpha * v_l
```

Audio-derived vectors can also be extracted, but the first practical version
lives in SAME latent space:

```text
positive_audio_folder -> SAME latents
negative_audio_folder -> SAME latents
v_audio = mean_t,pos(z) - mean_t,neg(z)
z_generated'[t] = z_generated[t] + alpha * v_audio
```

Implemented with:

- `summary_direction`
- `frame_mean_direction`
- `apply_frame_direction`
- `scripts/extract_audio_style_vectors.py`
- `scripts/generate_sa3_with_audio_direction.py`

This is not prompt steering and not ranking. It is an audio-pair-derived latent
edit before decoding. The harder future variant is residual-space extraction
from audio-conditioned SA3 runs, where audio files provide the contrastive
conditions for DiT activations.

That residual-space version is now implemented experimentally:

```text
positive audio file
  -> SA3 generate(init_audio=positive_audio, init_noise_level=rho)
  -> collect residual activations h_l^+

baseline can be either:
  a) prompt-only generation with the same prompt/duration/seed policy
  b) matched negative/reference audio-to-audio generation

v_l = mean(h_l^+) - mean(h_l^baseline)
h_l <- h_l + alpha * v_l
```

Implemented with:

- `SA3AudioResidualVectorExtractor`
- `scripts/extract_audio_residual_vectors.py`

This is closer to audioscope than SAME-frame vectors because it edits the SA3
residual stream during generation. It is also more fragile: positive and
negative audio sets need careful matching for duration, loudness, source type,
and prompt/context, otherwise the vector may represent the dataset mismatch
rather than the intended sonic quality.

### Audio to Prompt

Audio-to-prompt can mean two different things:

```text
caption/search prompt:
    audio -> prompt candidates -> score with CLAP/caption/human model

soft prompt inversion:
    audio target -> optimize T5/text-conditioning embeddings for SA3
```

The first is scaffolded with:

- `prompt_seed_from_audio_path`
- `coordinate_prompt_search`
- `default_modifier_axes`
- `greedy_token_prompt_search`

The greedy hard-token search is the transferable part of older CLIP
image-to-text notebooks:

```text
target embedding/objective fixed
best_tokens = []
for position in prompt:
    score every candidate next token, or a random subset
    append the best token
repeat with many random restarts
```

For SA3 this can be used with several scorers:

```text
CLAP audio-text score
negative SA3 teacher-forcing loss against target SAME latent
distance to optimized soft conditioning state
human/preference score
```

The second is not implemented yet. It would require differentiating through
SA3 generation or optimizing conditioning tensors against a target SAME latent.
The first version is now scaffolded as soft conditioning inversion:

```text
target audio -> SAME latent z_0
seed prompt -> conditioning tensors c
optimize c while SA3 weights stay frozen

z_t = (1 - t) z_0 + t eps
loss = || v_theta(z_t, t, c) - (eps - z_0) ||^2
```

This produces a reusable `.pt` soft prompt state. It is related to PEZ-style
prompt optimization, but it does not yet project the optimized continuous
conditioning back to readable hard tokens.

### Branch-and-Rank

```text
generate candidates -> encode latents -> score -> select
```

Purpose: make generation controllable without modifying SA3.

### Latent Timeline

```text
[LatentItem, LatentItem, ...] + transition costs + labels
```

Purpose: compose by walking through memory.

## Implementation Plan

### Phase 1: Local math package

Implemented now:

- `LatentItem`
- summary vectors
- cosine/euclidean metrics
- memory index
- transition cost
- loop cost
- bridge cost
- graph/path search
- tests on synthetic latents

### Phase 2: Stable Audio 3 / SAME adapter

Implemented as a lazy integration layer:

- `latent_audio_primitives.adapters.stable_audio3.StableAudio3Adapter`
- `latent_audio_primitives.adapters.stable_audio3.SAMEAutoencoderAdapter`
- `latent_audio_primitives.adapters.stable_audio3.latents_to_items`

These bind to the official released `stable_audio_3` package without importing it
at package import time. The important official paths are:

```text
stable_audio_3.StableAudioModel.generate(..., return_latents=True)
stable_audio_3.AutoencoderModel.encode(...)
stable_audio_3.AutoencoderModel.decode(...)
```

Example:

```python
from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter

sa3 = StableAudio3Adapter.from_pretrained("small-music")
items = sa3.generate_items(
    prompt="a sparse glassy ambient loop",
    duration=10.0,
    item_id_prefix="glass-loop",
    steps=8,
    cfg_scale=1.0,
    seed=42,
)
```

SAME folder encoding is intended to use:

```python
from latent_audio_primitives.adapters.stable_audio3 import SAMEAutoencoderAdapter

same = SAMEAutoencoderAdapter.from_pretrained("same-s")
item = same.encode_file("audio/example.wav")
```

Still pending:

- write `LatentItem` JSON/NPY records,
- batch encode audio folders,
- decode selected latents into listening files,
- verify real checkpoint shape/rate on this machine.

### Phase 2b: audioscope / residual steering adapter

Implemented as audioscope-compatible utilities:

- `latent_audio_primitives.adapters.audioscope_sa3.get_dit_layers`
- `ActivationCollector`
- `SteeringVectors`
- `ResidualSteerer`
- `mean_difference_vectors`

The adapter supports the released `stable_audio_3.StableAudioModel` wrapper as
well as audioscope's expected lower-level model shape. It saves and loads the
same basic `.pt` vector format used by audioscope:

```text
{
  "vectors": {layer_idx: tensor},
  "probe_accuracy": {layer_idx: float},
  "best_layer": int | null
}
```

Example:

```python
from latent_audio_primitives.adapters.audioscope_sa3 import (
    ResidualSteerer,
    SteeringVectors,
)

vectors = SteeringVectors.load("cache/mood_vectors.pt")
steerer = ResidualSteerer(sa3.model, vectors, layer=11)

with steerer.steer(alpha=8.0):
    latents = sa3.generate_latents(
        prompt="a cinematic ambient piano phrase",
        duration=10.0,
        seed=123,
    )
```

Pending:

- full prompt-pair extraction loop over real SA3 generations,
- probe training/evaluation harness,
- alpha sweep metadata logging,
- audio descriptor evaluation.

### Phase 3: Descriptor extraction

Not implemented yet:

- RMS/LUFS,
- brightness,
- spectral flatness,
- onset density,
- stereo width,
- beat estimates,
- CLAP/audio-text scores.

### Phase 4: SA3 integration

Not implemented yet:

- prompt generation wrapper,
- continuation/inpainting wrapper,
- `return_latents=True` path,
- branch-and-rank by memory score,
- prompt augmentation from memory retrieval.

### Phase 5: Learned sidecars

Not implemented yet:

- LatCH-SAME heads,
- prompt-aware ranker,
- pairwise preference model,
- activation capture/steering harness.

## Research Discipline

Every primitive should report:

```text
input object
output object
metric
failure mode
audio verification path
human listening note
```

For example:

```text
transition_cost says two latent boundaries are close
but the decoded audio may still click, clash harmonically, or fail rhythmically
```

So latent metrics should be treated as candidate generators, not final truth.

The first milestone is not a full AI DAW. It is a reliable latent memory kernel that can answer:

```text
what is similar?
what can follow this?
what can bridge these?
what loops?
what belongs to this private cluster?
```

Once those answers are measurable, SA3/SAME generation can be wrapped around them.
