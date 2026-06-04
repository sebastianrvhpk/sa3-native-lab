# SA3 Native Lab Repo Structure Map

Status: current as of 2026-06-04. This map reflects the notebook-first research
direction after removal of the abandoned app/control-plane path.

## Direction

SA3 Native Lab is a Colab and Python research workspace for frozen SA3/SAME
latent experiments. The source of truth is the notebook builder plus the
generated Colab notebook:

```text
scripts/build_sa3_same_native_notebook.py
colab/sa3_same_native_experimental_modes.ipynb
```

This repo should grow notebook cells, helper primitives, tests, and research
notes. It should not grow app shells, dashboards, API servers, product
interfaces, or control-plane plumbing.

LoRA is externalized to [dada-bots/underfit](https://github.com/dada-bots/underfit).
This repo may document an Underfit handoff and compare Underfit outputs, but it
should not rebuild Underfit locally.

## Top-Level Map

| Path | Role | Status |
|---|---|---|
| `colab/` | Generated notebook and Colab-facing helper scripts. | Active |
| `scripts/` | Notebook builder, validation, dataset encoding, vector extraction, soft prompt/style helpers. | Active |
| `latent_audio_primitives/` | Reusable SAME/SA3 latent math, prompt scoring, controls, DSP, geometry, observability, and experiment helpers. | Active |
| `tests/` | Unit tests for reusable primitives, adapters, notebook helpers, and math utilities. | Active |
| `docs/research/current/` | Current notebook map, mode math, and repo structure. | Active |
| `docs/research/methods/` | Method notes that support active or near-term notebook modes. | Active reference |
| `docs/archive/` | Historical research surveys and pre-consolidation notes. | Archived reference |
| `docs/guides/` | Stable Audio 3 upstream/reference model docs and images. | Reference |
| `docs/workflows/` | Stable Audio 3 upstream/reference workflow docs. | Reference; LoRA is legacy for this repo |
| `stable_audio_3/` | Vendored Stable Audio 3 source package. | Upstream base |
| `optimized/mlx/` | Experimental MLX implementation/reference path. | Separate reference |
| `.codex/skills/` | Repo-local Codex skill instructions for research/interface work. | Agent workflow |
| `README.stable-audio-3.md` | Preserved upstream Stable Audio 3 README. | Upstream reference |
| `LICENSE.stability-ai-stable-audio-3` | Preserved upstream license. | Required reference |

Empty app-era directories are not part of the active repo.

## Execution Surfaces

| Surface | Entry Point | Inputs | Outputs | Notes |
|---|---|---|---|---|
| Notebook builder | `scripts/build_sa3_same_native_notebook.py` | Local helper code, cell templates | `colab/sa3_same_native_experimental_modes.ipynb` | Source of truth for notebook edits |
| Colab notebook | `colab/sa3_same_native_experimental_modes.ipynb` | Audio files, prompts, SA3/SAME checkpoints, datasets | Audio files, tables, manifests, plots, player cells | Main research instrument |
| Notebook validation | `scripts/validate_colab_notebook.py` | Notebook JSON | validation result | Run after builder/notebook changes |
| Primitive tests | `tests/test_*.py` | Fake tensors/models and helper fixtures | pytest results | Protect reusable math without model weights |
| SA3 CLI | `stable-audio` / `stable_audio_3.cli` | Prompts/audio/checkpoints | generated audio | Upstream package surface |
| Research scripts | `scripts/*.py` | datasets, prompts, model handles | latents, vectors, profiles, audio | Script support for notebook experiments |

## Capability Map

| Capability | Primary Files | Evidence |
|---|---|---|
| SA3/SAME native flow prompt scoring | `latent_audio_primitives/flow_prompt.py`, `tests/test_flow_prompt.py`, Mode 2/16/17 notebook cells | Confirmed |
| Soft prompt optimization | `latent_audio_primitives/prompt_optimization.py`, `scripts/optimize_sa3_soft_prompt.py`, notebook Modes 1/4 | Confirmed |
| SAME latent memory/indexing | `latent_audio_primitives/index.py`, `schema.py`, `io.py` | Confirmed |
| Latent statistics and style controls | `controls.py`, `style.py`, `geometry.py` | Confirmed |
| Selective renoise and blur/filter edits | `selective_renoise.py`, `latent_blur.py` | Confirmed |
| Neural latent DSP | `latent_dsp.py`, `docs/research/methods/neural-latent-dsp.md` | Confirmed |
| Looping and periodic operators | `looping.py`, `periodic.py` | Confirmed |
| Guidance and posterior-style edit scaffolds | `guidance.py`, notebook Modes 24/25 | Confirmed scaffold |
| Residual steering and feature discovery | `adapters/audioscope_sa3.py`, `residual_features.py` | Confirmed |
| Control lanes and curricula | `control_lanes.py`, `curriculum.py` | Confirmed |
| Audio descriptors and audits | `audio_descriptors.py`, tests, notebook tables | Confirmed |
| Cross-model comparison harness | notebook Mode 26 | Confirmed scaffold |
| LoRA fine-tuning | external Underfit; local `scripts/train_lora.py` and `docs/workflows/lora.md` are reference only | Externalized |

## Artifact Graph

```text
audio files
  -> SAME encoder
  -> SAME latents
  -> latent memory / style profile / control lanes / DSP edits / geometry ops
  -> SAME decoder or SA3 polish
  -> audio outputs + descriptor tables + manifests + listening notes

prompts
  -> SA3 conditioner
  -> frozen flow field probes
  -> prompt scores / attribution rows / loss-by-timestep panels
  -> readable prompt candidates or soft prompt optimization

dataset folders
  -> pre-encoded latents + descriptors
  -> retrieval, curriculum, bridge search, donor selection
  -> notebook comparison cells
```

## Markdown Inventory

### Current Project Docs

| File | Role |
|---|---|
| `README.md` | Project entrypoint and current direction. |
| `docs/research/README.md` | Research doc index and documentation rules. |
| `docs/research/current/repo-structure-map.md` | This repo map and Markdown inventory. |
| `docs/research/current/notebook-research-map-and-next-methods.md` | Current capability map, research synthesis, backlog status, and next methods. |
| `docs/research/current/native-experimental-modes-math.md` | Current notebook math and mode implementation notes. |
| `docs/research/methods/seven-better-operators.md` | Active method reference for stronger SA3/SAME operators. |
| `docs/research/methods/neural-latent-dsp.md` | Active method reference for SAME latent DSP. |
| `colab/sa3_medium_l4_runbook.md` | Practical Colab L4 runbook and historical notebook notes. |
| `docs/codex_skills.md` | Repo-local Codex skill usage and install notes. |

### Stable Audio 3 Reference Docs

| File | Role |
|---|---|
| `README.stable-audio-3.md` | Upstream Stable Audio 3 README. |
| `docs/guides/model-overview.md` | Upstream/reference SA3 model overview. |
| `docs/guides/prompting.md` | Upstream/reference prompt guide. |
| `docs/workflows/autoencoder.md` | Upstream/reference SAME autoencoder workflow. |
| `docs/workflows/inference.md` | Upstream/reference inference workflow. |
| `docs/workflows/lora.md` | Legacy/upstream LoRA reference; active LoRA work is Underfit. |

### Archived Research Notes

| File | Role |
|---|---|
| `docs/archive/README.md` | Archive index and usage rules. |
| `docs/archive/latent-audio-research-notes.md` | Earlier broad SA3/SAME/LatCH/guidance research map. |
| `docs/archive/latent-memory-composition-primitives.md` | Earlier latent memory and composition primitive architecture. |
| `docs/archive/neural-audio-evolution-timeline.md` | Historical neural audio architecture and research timeline. |
| `docs/archive/neural-audio-math-methods-creative-tech.md` | Educational neural audio math survey. |

### Local Agent Workflow Docs

| File | Role |
|---|---|
| `.codex/skills/*/SKILL.md` | Project-local Codex skill definitions. |
| `.codex/skills/*/references/*.md` | Skill templates and rubrics. |

### Separate Reference

| File | Role |
|---|---|
| `optimized/mlx/README.md` | Experimental pure-MLX SA3 reference path. |

## Documentation Ownership

- Current notebook implementation claims belong in `docs/research/current/`.
- Reusable method notes belong in `docs/research/methods/`.
- Older literature surveys belong in `docs/archive/`.
- Stable Audio 3 upstream/reference docs stay under `docs/guides/` and
  `docs/workflows/` unless they are replaced by repo-specific notebook docs.
- Notebook behavior should be changed in
  `scripts/build_sa3_same_native_notebook.py` and regenerated into
  `colab/sa3_same_native_experimental_modes.ipynb`.

## Unknowns And Verification Plan

- Full SA3/SAME audio quality claims require loaded model weights and listening
  tests; unit tests only verify helper math and fake-model behavior.
- Notebook cells that depend on GPU Colab runtimes should be validated with
  `scripts/validate_colab_notebook.py` locally and then smoke-tested on Colab L4.
- External LoRA behavior should be validated in Underfit, then imported back as
  audio/checkpoint comparison artifacts rather than reimplemented here.
