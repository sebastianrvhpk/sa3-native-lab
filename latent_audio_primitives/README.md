# Latent Audio Primitives

This package is the notebook library for SA3 Native Lab.

The library supports one Colab-first research loop:

```text
audio/prompt/dataset
  -> SA3/SAME objects
  -> measurable latent state
  -> prompt, edit, retrieval, steering, or control probe
  -> decoded/polished audio plus descriptors, annotations, and decisions
```

## Primitive Contract

Every primitive should make these fields legible from a notebook cell:

```text
Object: native object under study
Transition: what maps into what, or what state is compared
Operation: observe, select, intervene, render, compare, or decide
Measurement: what evidence is collected
Evidence artifact: row, dataclass, audio path, latent item, descriptor, or note
Maturity/decision use: how the result supports microscope / selector /
  intervention candidate / promoted method / revise / drop
```

Prefer compact functions, dataclasses, and JSON-friendly rows. Keep upstream
runtime access in `adapters/`, executable method runs in `procedures/`, and
audition/annotation support in `evidence/`. Do not hide
upstream-version-sensitive behavior behind generic abstractions.

## Code Altitude Layers

| Layer | Modules | Object | Research Role |
|---|---|---|---|
| Root primitives | `schema.py`, `latent_math.py`, `geometry.py`, `latent_blur.py`, `selective_renoise.py`, `flow_prompt.py`, `trajectory.py`, `prompt_semantics.py`, etc. | native objects, math, measurements, operators, search | define what the lab manipulates and measures |
| Model boundary | `adapters/` | upstream SA3/SAME wrappers, residual hooks, tokenizer access | isolate external runtime coupling |
| Procedures | `procedures/` | soft prompts, flow scoring, SA3 polish, selective SA3, cyclic SA3, residual probes and sweeps | run executable notebook methods |
| Evidence | `evidence/`, `audio_descriptors.py`, `control_lanes.py` | player panels, annotations, descriptor/lane/disagreement rows, lane masks | support auditioning, selectors, and decisions |

Research layers are different from code altitude:

```text
SAME representation: root SAME math/operators plus evidence.
SA3 flow-conditioning: flow rows plus SA3 procedures.
SA3 internal trajectory: residual adapters/procedures plus trajectory cartography.
SA3-over-SAME coupled editing: SAME edits plus SA3 polish/inpaint/continue procedures.
```

Evidence utilities are shared review surfaces: evidence modules, manifests, and
ledger docs audit every research layer instead of defining a fifth layer.

`control_lanes.py` is measurement-first: it extracts audio/SAME temporal lanes,
compares source/output lanes, builds silence confidence, ranks lane-similar
memory rows, selects lane regions, and creates masks that other latent
operators may use. A lane mask is an intervention surface only after direct
decode, polish, and listening evidence support it.

`trajectory.py` is microscope/selector-first: it turns residual layer/timestep
probe rows into trajectory cells, band summaries, flow probe banks, residual
alpha schedules, and cyclic mix schedules. A trajectory-derived schedule is not
a promoted control until sweeps survive repeated audio evidence.

The full map lives in
[`docs/research/current/primitive-map.md`](../docs/research/current/primitive-map.md).
The research-layer ontology lives in
[`docs/research/current/architecture-ontology.md`](../docs/research/current/architecture-ontology.md).
The bottom-up object/capability map lives in
[`docs/research/current/capability-map.md`](../docs/research/current/capability-map.md).

## Maintenance Rule

Keep new code notebook-facing. A primitive should expose a compact function,
dataclass, or row object that a Colab cell can call directly. State, artifact
paths, and presentation details should remain explicit at the notebook boundary.

Use this placement rule:

```text
root module = define or transform a native object
adapter     = find or talk to external machinery
procedure   = run a research method with SA3/SAME
evidence    = audition, annotate, display, or review results
```

Procedure maturity belongs in the notebook narrative, docs, and experiment
ledger. Do not add package-level registries for static research status.
