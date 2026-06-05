# SA3 Native Lab Run Protocol

Status: operational protocol for turning notebook runs into research evidence.

This document answers: how to run the expanded Colab notebook as a lab
instrument, how to decide which evidence panels are required, and when a method
can move from idea to kept primitive.

## Research Frame

Every run starts with five fields:

```text
Object:
Intervention:
Measurement:
Claim:
Decision:
```

- `Object`: native object under study: audio waveform, SAME `z0`, SA3 flow
  state `z_t`, prompt condition `C(p)`, residual activation, control lane,
  memory row, dataset cluster, or listening annotation.
- `Intervention`: what changes: prompt, latent, channel subset, residual vector,
  sampler path, bridge, style profile, guidance loss, or source/donor choice.
- `Measurement`: evidence collected: descriptors, flow loss, latent geometry,
  control lanes, periodicity, nearest-memory rows, residual probes, runtime, or
  listening notes.
- `Claim`: what success would mean: controllability, source preservation,
  prompt inversion, loopability, novelty, style transfer, or microscope value.
- `Decision`: `promote`, `revise`, `drop`, `unknown`, or `microscope only`.

## Run Spine

Use the same spine for ordinary notebook experiments and frontier-informed
probes:

```text
choose source audio / dataset / prompt family
-> encode or load SAME latents
-> record baseline output or baseline measurements
-> choose one ontology stratum and one intervention
-> run the method with explicit seed, steps, duration, and prompt
-> measure descriptor, flow, latent, memory, geometry, lane, or residual evidence
-> audition with the custom player
-> annotate listening result
-> write or update the experiment ledger
-> decide promote / revise / drop / unknown / microscope only
```

Do not skip the baseline. For prompt and flow methods, reuse the same probe bank
when comparing prompts. For audio edit methods, include the source audio and a
plain SA3/audio-to-audio baseline.

## Science Ontology

| Stratum | Kind | Experiments | Primary Question |
|---|---|---|---|
| `SAME_REPRESENTATION` | native representation stratum | neighborhood renoise, selective renoise, graft, blur bottleneck, neural DSP, style statistics, geometry, OT, loop repair | What does SAME preserve, erase, linearize, or make editable in audio latents? |
| `SA3_FLOW_CONDITIONING` | flow-conditioning stratum | flow sign diagnostic, soft prompt inversion/audition, hard/readable prompt search, dataset soft prompt, attribution, timestep panel, null inversion | What prompt or conditioning object best explains target audio through frozen SA3 flow dynamics? |
| `CAUSAL_STEERING` | intervention stratum | prompt/audio residual steering, feature atlas, control head, cyclic trajectory, flow-state optimization, gradient edit, audio posterior guidance | Which inference-time interventions causally move generated audio rather than only measuring it? |
| `DATASET_MEMORY_COMPOSITION` | collection/composition stratum | prompt family, memory index, curriculum, continuation, bridge search | How do collections become memory, donor context, structure, or continuity without simply copying? |
| `EVIDENCE_DECISION_PROTOCOL` | lab infrastructure | player, annotations, descriptors, control lanes, manifest, ledger, combined chain | What evidence decides whether an effect is real, audible, useful, repeatable, or only a microscope? |
| `EXTERNAL_COMPARISON` | lab infrastructure | Underfit handoff, cross-model harness, optional external embedding lanes | What should frozen SA3/SAME be compared against without importing local app/training scaffolding? |

## Evidence Panels

Choose the panels that match the claim. A run does not need every panel, but it
must include enough evidence for its decision.

| Claim Type | Required Evidence | Useful Optional Evidence |
|---|---|---|
| Prompt inversion | flow loss rows, prompt candidates, baseline prompt | loss-by-timestep, attribution, decoded audition |
| Source-preserving edit | source audio, baseline, method output, descriptor delta, nearest-memory rows | flow score, control lanes, geometry report |
| Latent operator control | direct decode or polish, descriptor delta, listening notes | SAME bottleneck stress rows, flow score |
| Loop/continuation | loop preview, boundary/periodicity metrics, listening notes | bridge cost, lane continuity |
| Residual steering | layer/vector metadata, alpha sweep, descriptor/listening deltas | residual feature atlas, probe accuracy |
| Dataset/memory method | memory rows, cluster/donor selection evidence, output audition | heldout rows, geometry/lane scores |
| Frontier hypothesis | source link, native-object mapping, one concrete notebook run | external embedding disagreement, runtime audit |

## Minimum Run Packet

For a method to be reviewable, collect:

```text
commit hash
ontology stratum and experiment slug
model ID / SA3 checkpoint
runtime and GPU
source/donor audio paths
prompt and negative prompt
duration, seed, steps, CFG, init noise
flow convention and logSNR/timestep probes when relevant
baseline output
method output
descriptor report
listening note
decision
```

For a method to be promoted, repeat across at least:

```text
three seeds
two source clips when possible
one in-domain prompt
one out-of-domain or stress prompt
one baseline comparison
```

## Decision Rules

Promote when:

- descriptor/latent evidence and listening notes agree,
- the effect repeats across clips, prompts, or seeds,
- the recipe is compact enough to rerun from the notebook,
- source preservation and copying are separable when relevant.

Revise when:

- metrics move but listening is mixed,
- the effect works only for one clip or prompt,
- the direct SAME decode works but SA3 polish erases the effect,
- the recipe is useful but too hard to repeat.

Drop when:

- the method only creates artifacts,
- the effect is not measurable,
- it copies a memory item when novelty/source preservation is claimed,
- the sampler internals are too fragile for the payoff.

Keep as microscope only when:

- the method reveals SA3/SAME structure,
- it helps diagnose prompts, latents, flow, residuals, or geometry,
- it is not yet a reliable creative control.

## Ledger Update

After a run, update `experiment-ledger.md` with:

```text
run question
hypothesis
inputs
recipe
outputs
measurements
listening notes
decision
next action
```

Backlog items should not move forward until at least one ledger entry exists for
the relevant run family.
