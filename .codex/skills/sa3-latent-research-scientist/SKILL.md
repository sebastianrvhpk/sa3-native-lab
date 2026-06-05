---
name: sa3-latent-research-scientist
description: Use when formalizing, auditing, extending, or reviewing SA3/SAME latent-audio notebook research; designing experiments over SAME latents, SA3 flow fields, prompt conditioning, residual activations, control lanes, memory, descriptors, or listening evidence; deciding whether a primitive is a real control, a microscope, or noise.
---

# SA3 Latent Research Scientist

Use this skill to keep SA3 Native Lab scientifically coherent. The goal is not
to add methods quickly; it is to make claims, interventions, measurements, and
decisions explicit.

## First Move

Before editing code or docs, state the research frame:

```text
Object: what native object is under study?
Intervention: what changes?
Measurement: what evidence is collected?
Claim: what would success mean?
Decision: promote, revise, drop, or keep as microscope only?
```

If the user asks to implement immediately, still write this frame first in one
compact paragraph, then proceed.

## Repo Context To Inspect

Prefer local evidence before invention:

- `colab/sa3_native_science_lab.ipynb`
- `latent_audio_primitives/README.md`
- `docs/research/current/primitive-map.md`
- `docs/research/current/research-state.md`
- `docs/research/current/methods-and-math.md`
- `docs/research/current/experiment-ledger.md`
- `docs/research/current/backlog.md`

Use `rg` for references and notebook imports before moving or deleting anything.

## Scientific Grammar

Every method should fit one of these roles:

- Native object probe: reveals structure in SAME latents, SA3 flow states,
  residual activations, prompt conditions, or memory geometry.
- Intervention: changes a native object and asks whether decoded/polished audio
  moves predictably.
- Measurement: turns audio/latent behavior into rows, descriptors, geometry,
  control probes, flow losses, nearest-memory evidence, or listening notes.
- Selection tool: helps choose prompts, donors, seeds, clusters, channels, or
  method recipes.
- Microscope only: useful for understanding the model, but not yet a creative
  control.

Keep service, dashboard, product, session, route, artifact-UI, and control-plane
framing outside SA3/SAME notebook research.

## Evidence Standard

Do not call an operator a control until it has:

1. a clear mathematical or architectural rationale,
2. a compact notebook-facing API or cell,
3. descriptor/latent evidence,
4. listening notes or a plan for listening notes,
5. promote/revise/drop criteria.

Label claims as:

- `confirmed`: present in local code/docs/notebook or directly observed.
- `repo-inferred`: implied by local code but not executed.
- `source-inferred`: supported by an external source that was actually checked.
- `hypothesis`: plausible but unverified.
- `unknown`: cannot be determined yet.

## Output Shape

For audits or proposals, return:

```text
Research frame
Capability/object map
Experiment cards
Measurement and artifact plan
Failure modes
Promote/revise/drop criteria
Notebook/docs/code impact
```

For implementation, update the relevant docs after code changes:

- Method math: `docs/research/current/methods-and-math.md`
- Current state/module role: `docs/research/current/research-state.md` or
  `docs/research/current/primitive-map.md`
- Future work or promotion criteria: `docs/research/current/backlog.md`
- Real runs/listening decisions: `docs/research/current/experiment-ledger.md`

## References

- Use `references/experiment-card.md` when drafting experiment proposals.
- Use `references/evidence-standards.md` when reviewing whether a method is
  ready to promote.
