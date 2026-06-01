# Colab Capability Map

This map tracks the migration from notebook/script experiments into typed app
contracts. It is intentionally evidence-based: every row should point at a local
execution surface and the app surface that now owns it or still needs it.

For the more granular section-by-section parity gate, see
`docs/colab-parity-matrix.md`.

## Current Coverage

Default app exploration targets SA3 `medium` and SAME `same-l`, matching the
notebook's Medium/SAME-L setup. Small checkpoints remain selectable for explicit
smoke tests, but they are not the app default.

| Capability | Evidence | Current app surface | Status |
| --- | --- | --- | --- |
| Text to audio | `optimized/mlx/sa3`, `generate.text_to_audio` | `/generate/text`, Make gesture, Current Sound | implemented |
| Audio to audio | `optimized/mlx/sa3 --init-audio` | `/generate/audio-to-audio` | implemented |
| Inpaint | `optimized/mlx/sa3 --inpaint-range` | `/generate/inpaint` | implemented |
| SAME encode/decode | SAME adapter scripts and tests | `/latents/encode`, `/latents/decode` | implemented |
| Latent blur/DSP/graft/renoise/roll | `latent_audio_primitives` wrappers | `/operators/run`, Morph/Borrow Texture Tune | implemented with native parameter controls and local presets |
| Real waveform peaks | audio artifacts via `soundfile` | `/artifacts/{id}/peaks` | implemented |
| Audio descriptor comparison | `latent_audio_primitives.audio_descriptors` | `/artifacts/{target}/descriptor-comparison/{take}`, prompt-search take delta strip | implemented first slice |
| Artifact annotation/search | `ArtifactStore.annotate_artifact`, `/artifacts?q=&tags=` | Specimen annotation panel, Memory role/reuse intent, session/archive filters by decision, tag, kind, model, gesture, branch, text, and lineage | implemented |
| Latent memory query | `LatentMemoryIndex`, `memory.query` | `/experiments/run`, Advanced Gestures/Tune | implemented for local latent artifacts |
| Audio style vectors | `scripts/extract_audio_style_vectors.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| Positive style profile | `scripts/build_positive_audio_style_profile.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| SAME style profile build | `scripts/build_same_style_profile.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| SA3 style profile generation | `scripts/generate_sa3_with_style_profile.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| SA3 style direction generation | `scripts/generate_sa3_with_style_direction.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| SA3 audio direction generation | `scripts/generate_sa3_with_audio_direction.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| Prompt residual vectors | `scripts/extract_sa3_vectors.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| Audio residual vectors | `scripts/extract_audio_residual_vectors.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| Residual alpha sweep | `scripts/run_sa3_alpha_sweep.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| Prompt search | `latent_audio_primitives.prompt_optimization`, `latent_audio_primitives.flow_prompt` | `/experiments/run`, Advanced Gestures/Tune | native recipe with `lexical_probe` fallback and optional `sa3_flow_probe` |
| Soft prompt optimize/generate | `scripts/optimize_sa3_soft_prompt.py`, `scripts/generate_sa3_with_soft_prompt.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| Dataset pre-encode | `scripts/pre_encode_dataset.py` | `/experiments/run`, Advanced Gestures/Tune | script-job adapter with native controls |
| SAME geometry audit | `latent_audio_primitives.geometry.geometry_report` | `/experiments/run`, Advanced Gestures/Tune | implemented for local latent artifacts |
| Colab mode inventory | `colab/sa3_same_native_experimental_modes.ipynb` | `/colab/modes`, Mode Atlas | typed parity/status map |

## Artifact Graph

```text
Current Sound / Memory / vector file / profile file
  -> Gesture or Advanced Gesture recipe
  -> Pending Take
  -> audio artifact(s), latent artifact(s), and/or zipped bundle artifact
  -> branch + recipe + logs + metrics + lineage + annotation
```

Script-job adapters are a bridge, not the final ideal surface. They make Colab
workflows durable and replayable first. Advanced Gestures now expose
script-backed experiments through typed Tune controls, advanced parameters,
artifact selectors, and selected-artifact fallbacks where the script can consume
the current artifact. The Mode Atlas remains a triage/inspect surface, not the
primary product loop.

Morph and Borrow Texture expose direct latent operators as native controls
rather than notebook snippets: Roll, Blur, DSP/Reroute, Borrow Texture, and
Renoise all map to executable request params, with donor-latent selection shown
only when the chosen move needs a donor. Browser-local presets still save and
reload named parameter sets per operator mode and show selected-preset diffs for
changed params and donor latents; backend-backed preset history remains future
promotion work.

The Memory shelf now supports label, notes, tags, listening decisions, role,
reuse intent, active reuse actions, and recovery into the current session.
Latent memory query is promoted from concept to recipe: selecting a latent
artifact can produce a `memory_query.json` bundle ranked by cosine or Euclidean
summary distance against other local latent artifacts. Those local memory hits
are actionable in the bundle preview: select the artifact, play audio hits, or
reuse latent hits as donor latents.

Prompt search is now promoted from helper-only code to a native recipe:
`experiment.prompt_search` runs beam, greedy, or coordinate hard-token search,
stores `prompt_search.json`, and exposes the resulting prompt back into
generation fields. It keeps `lexical_probe` as a cheap fallback and adds `sa3_flow_probe`
for Medium-backed flow-loss scoring against a target audio latent. This is still
marked as a probe until the real Medium/MPS path has short-audio listening
validation, runtime-cost notes, and richer candidate comparison. The first
candidate branch slice now shows target-vs-take descriptor deltas for
generated prompt-search takes and summarizes those deltas against saved
keeper/maybe/reject decisions. Prompt memory also groups generated prompt takes
by prompt text across runs.

## Next Promotion Targets

1. Keep the product loop smoke and playback/session smoke green as capability
   surfaces evolve.
2. Add prompt-search sweep/layer comparisons and document useful Medium/MPS
   score-sample/timestep settings.
3. Build the richer Memory browser over current metadata before vector search:
   role, reuse intent, tags, notes, kind, decision, branch, source.
4. Control-head recipes for Mode 12 and the labelled-probe part of Mode 15.
5. Promote local latent presets into backend history only after preset semantics
   are stable.
6. Keep fine-tuning out of SA3 Native Lab; use
   [dada-bots/underfit](https://github.com/dada-bots/underfit) on Colab A100 for
   style/domain adaptation work.
