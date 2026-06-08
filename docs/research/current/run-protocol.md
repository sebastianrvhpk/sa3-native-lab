# SA3 Native Lab Run Protocol

Status: operational protocol for turning notebook runs into evidence and
decisions.

This document answers: how a Colab cell becomes a research claim, what evidence
is required, and when a method moves from microscope to kept method.

## Research Frame

Every run starts with seven fields:

```text
Research layer / evidence utility:
Object:
Transition:
Operation:
Measurement:
Claim:
Decision:
```

- `Research layer / evidence utility`: one of the four research layers
  (`SAME representation`, `SA3 flow/conditioning`, `SA3 internal trajectory`,
  `SA3-over-SAME coupled editing`) or `evidence utility` when the run only
  improves review, annotation, disagreement, manifests, or ledger decisions.
- `Object`: native object under study: audio waveform, SAME `z0`, SA3 flow
  state `z_t`, prompt condition `C(p)`, residual activation, memory item,
  dataset cluster, control lane, or evidence packet.
- `Transition`: what object path is tested, such as `audio -> z0`,
  `z0 -> z0'`, `z0 -> flow probes`, `prompt -> condition`,
  `activation -> steering vector`, `memory -> donor`, or
  `output -> evidence packet`.
- `Operation`: `observe`, `select`, `intervene`, `render`, `compare`, or
  `decide`.
- `Measurement`: descriptors, flow loss, latent geometry, control lanes,
  periodicity, nearest-memory rows, residual probes, runtime, or listening
  notes.
- `Claim`: what success would mean: controllability, source preservation,
  prompt inversion, loopability, novelty, style transfer, selection value, or
  microscope value.
- `Decision`: `promote`, `revise`, `drop`, `unknown`, or `microscope only`.

## Claim Ladder

Do not call a method a control merely because a tensor changed. Promote it only
as far as evidence allows.

| Maturity | Meaning | Required evidence |
|---|---|---|
| Microscope | Reveals structure but is not a reliable control. | Native-object rationale plus measurement rows. |
| Selector | Helps choose prompts, donors, seeds, chunks, channels, recipes, or baselines. | Ranking rows plus at least one audition or review packet. |
| Intervention candidate | Changes decoded or polished audio in the intended direction. | Baseline, method output, descriptor/latent evidence, and listening note. |
| Promoted method | Repeats across sources, prompts, or seeds and survives baselines. | Evidence packets across repeated runs plus ledger decisions. |

## Run Spine

Use this spine for ordinary notebook runs and frontier-informed probes:

```text
choose source audio / dataset / prompt family
-> encode or load native objects
-> record baseline output or baseline measurements
-> choose one object transition and operation
-> run with explicit seed, steps, duration, prompt, and convention
-> collect descriptor / flow / latent / memory / geometry / lane / residual evidence
-> audition with the notebook player
-> annotate listening result
-> write or update the experiment ledger
-> decide maturity: microscope / selector / intervention candidate / promoted / dropped
```

Do not skip the baseline. For prompt and flow methods, reuse the same probe bank
when comparing prompts. For audio edit methods, include the source audio and a
plain SA3/audio-to-audio or direct-decode baseline.

## Native Object Workbenches

The notebook is organized as workbenches over object transitions. A workbench is
not a claim by itself; it is where claims are tested.

Research layer explains why the workbench exists. Evidence utility workbenches
review claims across layers:

```text
SAME representation -> SAME representation, direct decode, memory, control lanes
SA3 flow/conditioning -> SA3 flow and conditioning science
SA3 internal trajectory -> SA3 internal trajectory science
SA3-over-SAME coupled editing -> edited SAME latents entering SA3 polish, continuation, inpainting
evidence utility -> evidence packet setup, ledger and decision board
```

| Workbench | Native objects | Typical transitions | Main question |
|---|---|---|---|
| Runtime and model boundary | upstream SA3/SAME handles | checkpoint -> model handle; audio -> encoded latent | Are external model assumptions explicit and stable? |
| Evidence packet setup | audio paths, descriptors, annotations, manifests | output -> evidence packet | Can each result be reviewed and compared? |
| Audio and SAME preparation | audio waveform, `LatentItem`, SAME `z0` | audio -> `z0`; `z0` -> saved item | What object exactly enters each method? |
| SAME representation bench | SAME `z0`, summaries, geometry, lanes, direct decodes | `z0` -> rows/reports/audio | What does SAME expose before SA3 intervenes? |
| SAME memory and composition bench | memory rows, clusters, bridges, donors | collection -> selector -> output | How do collections support selection without copying? |
| SA3 flow and conditioning science | `z_t`, prompt condition `C(p)` | target `z0` -> probe bank -> prompt/condition score | Which prompts or conditions explain audio under frozen SA3 flow? |
| SA3 internal trajectory science | residual activations, sampler states, trajectory cells | activation/state -> trajectory map -> scheduled candidate -> output | Which internal signals are visible, schedulable, and eventually causal? |
| SA3-over-SAME coupled editing bench | edited SAME latents | `z0` -> `z0'` -> direct decode / SA3 polish | Which latent edits survive, get erased, or get amplified by SA3? |
| External comparison bench | imported audio/checkpoints/run notes | external audio output -> evidence packet | How do imported outputs compare under the same local evidence? |
| Ledger and decision board | evidence packets and decisions | evidence -> maturity update | What is real, repeatable, useful, or only diagnostic? |

## Evidence Panels

Choose panels that match the claim. A run does not need every panel, but it must
include enough evidence for its decision.

| Claim type | Required evidence | Useful optional evidence |
|---|---|---|
| Prompt inversion | flow loss rows, prompt candidates, baseline prompt | loss-by-timestep, attribution, decoded audition |
| Source-preserving edit | source audio, baseline, method output, descriptor delta, nearest-memory rows | flow score, control lanes, geometry report |
| Latent operator control | direct decode or polish, descriptor delta, listening notes | SAME bottleneck stress rows, flow score |
| Latent objective/guidance | objective rows, before/after latent values, direct decode or polish, listening notes | control-lane renderings, geometry report, source-preservation rows |
| Loop/continuation | loop preview, boundary/periodicity metrics, listening notes | bridge cost, lane continuity |
| Residual steering | layer-probe rows, sampler-timestep rows, trajectory map, selected top-k cells/layers, vector metadata, alpha schedule, alpha sweep, descriptor/listening deltas | trajectory window rows, residual feature atlas, repeat probe run, trajectory-derived flow probe bank |
| Dataset/memory method | memory rows, cluster/donor selection evidence, output audition | heldout rows, geometry/lane scores |
| Frontier hypothesis | source link, native-object mapping, one concrete notebook run | native disagreement panel, runtime audit |

For control-lane mechanistic probing, run the notebook preflight before a long
audit. The evidence preflight should report a non-empty active-source span,
typed region-sweep rows, audio-event/SAME-event region comparisons,
all-channel lane correlations, channel-region rows, a channel-region overlap
summary/top-row packet, and a non-empty mech-target manifest. Treat the manifest
as a spending plan for the expensive probe, not as proof of control. The
mechanistic probe should then run on the strongest non-redundant manifest rows
with null controls, typed region rows, prediction curves, and repeatability
metadata. If timestep probing is enabled, the preflight should also report at
least one `ok` token-preserving timestep row; otherwise fix the lane mask, hook
mapping, timestep metadata, or layer/lane selection before launching the
expensive null/repeatability run.

## Minimum Evidence Packet

For a method to be reviewable, collect:

```text
commit hash
research layer / evidence utility
notebook workbench
object transition and method name
claim maturity before the run
model ID / SA3 checkpoint
runtime and GPU
source/donor audio paths
prompt and negative prompt
duration, seed, steps, CFG, init noise
flow convention and logSNR/timestep probes when relevant
baseline output
method output
descriptor report
nearest-memory rows when relevant
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
- sampler internals are too fragile for the payoff.

Keep as microscope only when:

- the method reveals SA3/SAME structure,
- it helps diagnose prompts, latents, flow, residuals, or geometry,
- it is not yet a reliable creative control.

## Ledger Update

After a run, update `experiment-ledger.md` with:

```text
run question
hypothesis
object transition
inputs
recipe
outputs
measurements
listening notes
maturity update
decision
next action
```

Backlog items should not move forward until at least one evidence packet exists
for the relevant run family.
