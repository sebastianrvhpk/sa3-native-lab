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
| Root primitives | `schema.py`, `latent_math.py`, `geometry.py`, `control_lanes.py`, `latent_constraints.py`, `residual_probes.py`, `latent_blur.py`, `selective_renoise.py`, `flow_prompt.py`, `trajectory.py`, `prompt_semantics.py`, etc. | native objects, math, measurements, operators, search | define what the lab manipulates and measures |
| Model boundary | `adapters/` | upstream SA3/SAME wrappers, residual hooks, tokenizer access | isolate external runtime coupling |
| Procedures | `procedures/` | soft prompts, flow scoring, SA3 polish, selective SA3, cyclic SA3, residual probes and sweeps | run executable notebook methods |
| Evidence | `evidence/`, `audio_descriptors.py`, `evidence/control_lane_rendering.py` | player panels, annotations, descriptor/lane/disagreement rows, SVG views | support auditioning, selectors, and decisions |

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
including deterministic MIR/DSP lanes, computes source-active masks, compares
source/output lanes, builds active-window correlations, ranks lane-similar
memory rows, selects lane regions, compares audio-event and SAME-event regions,
exports complete SAME-channel atlases, correlates every latent channel against
every lane, and creates masks that other latent operators may use. Region modes
are typed temporal predicates over measured lanes: state, event, transition,
persistence, source-validity, and signed-channel selectors. A lane mask or
channel family is an intervention surface only after direct decode, polish, and
listening evidence support it.

`latent_constraints.py` is objective-first: it defines scalar latent constraints
that can be reported as rows or used by guidance/optimization procedures. It
does not own evidence packet aggregation.

`residual_probes.py` is residual-math-first: it owns activation examples,
steering vector containers, and probe rows after activations have already been
captured. SA3 layer discovery and hook execution remain in `adapters/` and
`procedures/`.

`control_lane_probes.py` is residual-lane-probe-first: it owns continuous
ridge probes from captured SA3 residual activations to control lanes, including
layer rows, observed-call window rows, token-preserving sampler-timestep rows,
token-blocked and call-held-out CV scores, null controls, true-vs-null margins,
held-out prediction rows, and active/quiet direction previews. It does not run
SA3.

`evidence/control_lane_rendering.py` owns notebook SVG views for lane overlays,
regions, latent-channel heatmaps, probe heatmaps, and prediction curves.
Rendering is evidence presentation, not the definition of a control lane.

`trajectory.py` is microscope/selector-first: it turns residual layer/timestep
probe rows into trajectory cells, band summaries, flow probe banks, residual
alpha schedules, and cyclic mix schedules. A trajectory-derived schedule is not
a promoted control until sweeps survive repeated audio evidence.

The full map lives in
[`docs/research/current/primitive-map.md`](../docs/research/current/primitive-map.md).
The research-layer ontology lives in
[`docs/research/current/architecture-ontology.md`](../docs/research/current/architecture-ontology.md).
The native-object capability map lives in
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
