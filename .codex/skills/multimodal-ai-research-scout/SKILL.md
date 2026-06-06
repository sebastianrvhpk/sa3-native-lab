---
name: multimodal-ai-research-scout
description: Use when surveying current state of the art, recent papers, repos, model architectures, training methods, or cross-modal ideas for audio, music, video, image, text, diffusion, flow matching, transformers, latent spaces, controllability, editing, retrieval, representation learning, or multimodal generative AI; especially to translate frontier research into SA3/SAME notebook workbenches with explicit object transitions, claim maturity, evidence packets, and code altitude.
---

# Multimodal AI Research Scout

Use this skill for frontier research scouting. The job is to connect current
papers, repos, and architecture changes to concrete SA3 Native Lab experiments.
The local lab is notebook-first: upstream SA3 and Underfit stay external unless
the notebook consumes exported artifacts.

## Currentness Rule

For anything described as current, latest, state of the art, recent, frontier,
or best available, browse current sources first. Prefer:

1. primary papers and official project pages,
2. official repos and model cards,
3. benchmark pages or leaderboards with dates,
4. high-quality technical articles only as secondary context.

Always include access dates or publication/release dates when they matter.
Do not present remembered SOTA as current SOTA.

## Research Axes

Scout across modalities, but translate back to audio/SAME/SA3:

- Audio/music generation: codecs, latent diffusion/flow, text-to-audio, music
  structure, controllability, editing, source preservation.
- Image/video diffusion and flow: rectified flow, consistency/distillation,
  inversion, attention control, trajectory editing, conditioning adapters.
- Language/model reasoning: retrieval, synthetic data, preference optimization,
  tool use, eval harnesses, agentic experiment loops.
- Representation learning: contrastive embeddings, self-supervised audio,
  disentanglement, probing, causal/linear control directions.
- 3D/world models: temporal coherence, scene memory, object permanence, motion
  controls, long-horizon generation.
- Scientific ML: experiment design, ablation, uncertainty, reproducibility,
  causal claims, measurement discipline.

## Architecture Map

For each source or family, extract:

```text
Source:
Date:
Modality:
Architecture:
Training/inference trick:
Native object it manipulates:
Evidence/benchmark:
What transfers to SA3/SAME:
What probably does not transfer:
Notebook experiment candidate:
Local object transition:
Local workbench:
Claim maturity target: microscope / selector / intervention candidate / promoted method
Local altitude landing: root primitive / adapter / procedure / evidence / docs-only
Evidence packet needed:
Risk or unknown:
Status: source-confirmed / repo-inferred / hypothesis
```

## Cross-Modal Transfer Rules

Do not copy architecture terms blindly. Translate by native object, transition,
workbench, and evidence:

- Image latent edit -> SAME latent edit only if the operation has a meaningful
  time/channel/feature analogue.
- Video temporal control -> audio loop/structure/control-lane experiment only
  if temporal coherence is measurable.
- Text embedding steering -> prompt/condition/residual experiment only if SA3
  exposes a condition or activation path.
- Diffusion inversion -> SA3 flow prompt/state inversion only if velocity,
  timestep, and convention are explicit.
- Adapter/LoRA method -> external Underfit workflow unless the notebook only
  consumes exported audio/checkpoints for comparison.

When proposing local implementation, state the workbench, maturity target, and
altitude explicitly:

- Root primitive: source suggests new math, rows, statistics, search, or
  operators over existing native objects.
- Adapter: source requires touching upstream SA3/SAME, tokenizer, checkpoints,
  hooks, or external artifact formats.
- Procedure: source requires running SA3/SAME, optimizing, sweeping, capturing
  activations, or generating comparison audio.
- Evidence: source improves auditioning, annotation, comparison, reports, or
  promote/revise/drop decisions.
- Docs-only: source is promising but not yet concrete enough for a notebook cell.

## Output Shape

Return:

```text
Scope and source strategy
Source map
Architecture deltas
Cross-modal transfer matrix
Candidate SA3/SAME experiments
Object transitions
Workbench, maturity, and altitude landing
Evidence packets required
Priority order
Unknowns and verification plan
Docs/backlog updates needed
```

Use concise tables where useful. Cite sources with links. Clearly separate
confirmed source facts from notebook hypotheses.

## References

- Use `references/source-card.md` for source extraction.
- Use `references/architecture-transfer.md` for translating other modalities
  into SA3/SAME experiments.
