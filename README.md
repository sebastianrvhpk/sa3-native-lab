# SA3 Native Lab

This repository is a combined Colab/research workspace:

- official Stable Audio 3 source package in `stable_audio_3/`
- SAME/SA3 native latent-memory, steering, and prompt-inversion primitives in `latent_audio_primitives/`
- Colab notebooks in `colab/`
- research scripts in `scripts/`
- current experimental math notes in `docs/research/`

The goal is exploratory research over native SA3/SAME spaces, not a finished product.

For the current app-level description and improvement queue, see:

- `docs/app-overview.md`
- `docs/codebase-review.md`
- `docs/improvement-roadmap.md`
- `docs/architecture-horizon.md`

## Upstream

The Stable Audio 3 source in this repository comes from:

https://github.com/Stability-AI/stable-audio-3

The upstream README is preserved as:

`README.stable-audio-3.md`

The upstream license is preserved as:

`LICENSE.stability-ai-stable-audio-3`

## Colab L4

Push this repo to GitHub, then use the Colab notebook:

`colab/sa3_same_native_experimental_modes.ipynb`

The notebook is already configured to clone:

```python
COMBINED_REPO_URL = "https://github.com/sebastianrvhpk/sa3-native-lab.git"
```

The notebook installs this single repo. No zip upload is needed.

## Local Install

For a local Mac or workstation checkout, prefer a repo-local `uv` environment:

```bash
uv sync --extra dev
uv run pytest
```

This works even when the shell has no bare `python` command. Use `uv run
python ...`, `uv run stable-audio ...`, or activate `.venv/` if you want a
traditional prompt.

For the full local app/research surface, including the API, frontend-backed
scripts, notebook tools, and LoRA scaffold, sync all local extras:

```bash
uv sync --extra app --extra ui --extra lora --extra notebook --extra dev
```

### Model Policy

The active exploration target for this repo is SA3 Medium. Local app defaults
therefore use `medium` for SA3 generation and `same-l` for SAME encode/decode,
matching the medium checkpoint's SAME-L latent space. Smaller checkpoints remain
available for explicit quick tests, but the app and docs assume Medium unless a
command says otherwise.

### Apple Silicon / M1

For fast local generation on M1/M2/M3/M4, use the MLX implementation:

```bash
cd optimized/mlx
./install.sh -y --download medium
./sa3 --prompt "lofi house loop" --dit medium --decoder same-l --seconds 5 --out smoke.wav --play
```

The MLX path auto-downloads missing weights from Hugging Face on first use.
You may need to accept the Stability AI license and log in with `hf auth login`
or set `HF_TOKEN`.

The PyTorch research layer also runs locally and auto-selects `cuda -> mps ->
cpu` when a script exposes `--device`:

```bash
uv run stable-audio --model medium --device mps --no-half -p "short test tone" --duration 5 -o outputs/smoke.wav
```

On Apple Silicon, PyTorch SA3 Medium is mainly a compatibility path; the MLX
CLI is the practical path for generation speed and memory.

For the Gradio UI, install the UI extra and launch locally:

```bash
uv sync --extra ui
uv run python run_gradio.py --model medium --device mps --no-half
```

### Local App Runner

For the native app, install the API/dev extras and frontend dependencies once:

```bash
uv sync --extra app --extra dev
npm install --prefix frontend
npm install --prefix apps/control-plane
```

Then check readiness:

```bash
uv run sa3-lab doctor
uv run sa3-lab doctor --with-control-plane
```

Start the API daemon and Vite workbench together:

```bash
uv run sa3-lab dev
```

Start the modern tRPC control-plane path as well:

```bash
uv run sa3-lab dev --with-control-plane
```

The runner prints backend readiness, artifact storage, and the local URLs. It
reuses already-running services on `127.0.0.1:8733`, `127.0.0.1:8787`, and
`127.0.0.1:5173` instead of starting duplicates.

### TypeScript Control Plane

`apps/control-plane` is the tRPC app-contract layer. It does not replace the
Python runtime; it shapes Python runtime records into app-native procedures such
as `workbench.load`, `system.readiness`, `jobs.cancel`, `jobs.retry`,
`jobs.events`, `recipes.replay`, `recipes.fork`, `artifacts.inspect`, and
`families.load`.

```bash
npm install --prefix apps/control-plane
npm run test --prefix apps/control-plane
SA3_PYTHON_API_BASE=http://127.0.0.1:8733 npm run dev --prefix apps/control-plane
```

The frontend uses the control plane when `VITE_SA3_CONTROL_PLANE_URL` is set.
`uv run sa3-lab dev --with-control-plane` wires that env var automatically.
Without that flag, the frontend keeps using the Python API read endpoints
directly.

See `docs/control-plane-architecture.md` for the staged tRPC/Postgres/pgvector
plan.

### Local API Daemon

The notebook experiments are being wrapped as a typed local app/runtime layer.
Install the API extra and start the daemon:

```bash
uv sync --extra app
uv run sa3-lab-api --host 127.0.0.1 --port 8733
```

The daemon persists local artifacts, recipes, and background jobs under
`.sa3_lab/` by default. Override that location with:

```bash
uv run sa3-lab-api --artifact-root /path/to/sa3-lab-artifacts
```

Useful first endpoints:

```bash
curl http://127.0.0.1:8733/health
curl http://127.0.0.1:8733/models/status
curl http://127.0.0.1:8733/operators/specs
```

`/operators/specs` is now more than an inventory endpoint: each operator also
returns `ui_fields` with labels, defaults, bounds, select options, artifact-kind
hints, and required/advanced flags. The React instrument merges those backend
field contracts into Operator Studio and Recipe Studio, so duration, seed,
model, sweep, path, and backend params stay reachable as the Colab scripts move
into native UI.

Text generation runs through the Apple Silicon MLX backend when
`optimized/mlx/install.sh` has been completed:

```bash
curl -X POST http://127.0.0.1:8733/generate/text \
  -H "Content-Type: application/json" \
  -d '{"prompt":"lofi house loop","duration_seconds":5,"model":"medium","decoder":"same-l","steps":8}'
```

Latent `.npy` artifacts can be imported and transformed through typed operator
jobs such as `latent.blur`, `latent.dsp`, `latent.graft`, `latent.renoise`, and
`latent.cyclic_roll`. Every run stores a recipe, job record, source lineage, and
output artifact instead of relying on notebook cell state.

Colab-style experiment scripts are also available as background recipe jobs
through `/experiments/run`. These bridge the notebook migration for style
profiles, audio directions, residual vectors, alpha sweeps, prompt search, soft
prompts, dataset pre-encoding, and LoRA training. Script runs save audio
artifacts when they produce listenable WAVs and zipped bundle artifacts for
vector/profile outputs and training folders. `experiment.prompt_search` is a
native recipe today: it keeps a deterministic `lexical_probe` scorer for cheap
wiring tests and exposes an optional `sa3_flow_probe` scorer that ranks prompts
with Medium flow losses against a target audio latent. CLAP scoring is still
queued behind the same explicit scorer contract.

Bundle artifacts can be inspected with `/artifacts/{id}/inspect`; embedded audio
inside a bundle can be streamed through `/artifacts/{id}/bundle-file` or promoted
into a normal audio artifact through `/artifacts/{id}/bundle-audio/promote`.

Audio artifacts can also be encoded to SAME latents and decoded back through the
Torch/MPS autoencoder path:

```bash
curl -X POST http://127.0.0.1:8733/latents/encode \
  -H "Content-Type: application/json" \
  -d '{"source_artifact_id":"art_...","model":"same-l","backend":"torch_mps"}'

curl -X POST http://127.0.0.1:8733/latents/decode \
  -H "Content-Type: application/json" \
  -d '{"source_artifact_id":"art_...","model":"same-l","backend":"torch_mps"}'
```

### Listening Bench Frontend

The first local app slice lives in `frontend/` and talks to the API daemon:

```bash
cd frontend
npm install
npm run dev
```

Open the printed Vite URL, normally `http://127.0.0.1:5173`. The first bench
supports audio import, MLX text generation, SAME encode/decode, latent operator
jobs, Recipe Studio script experiments, Colab Mode Atlas parity/status, job
polling plus durable live job replay, job phase labels, cancellation/retry,
recipe replay/fork editing, recipe diff/reset controls, archive-and-new session
cleanup, result-family grouping with detail playback, sortable alpha-sweep
variant promotion and metric tables, memory-query result reuse,
bundle-to-recipe actions, backend-parsed typed bundle inspectors, inline bundle
plot previews, readiness checks, artifact selection, kind-specific artifact
vitals, real waveform peaks, region looping, download, and A/B audio playback.

### Notebook Parity Check

The Colab notebook can be validated locally with all mode toggles off:

```bash
uv run python scripts/validate_colab_notebook.py
```

To run the model-loading and smoke cells, authenticate Hugging Face without
putting the token in commands or files, then opt in:

```bash
export HF_TOKEN=...
uv run python scripts/validate_colab_notebook.py --with-models --with-smoke
```

Notebook mode switches can also be enabled from the environment, for example:

```bash
RUN_MODE_15_GEOMETRY_AUDIT=1 uv run python scripts/validate_colab_notebook.py --with-models
```

### Colab / System Install

```bash
uv pip install --system -e .
```

For SA3 Medium on Linux CUDA:

```bash
uv pip install --system torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --system -e .
uv pip install --system --force-reinstall numpy==2.2.6
python -m pip uninstall -y scipy scikit-learn sklearn torchvision
uv pip install --system flash-attn --no-build-isolation --no-cache-dir --no-deps
```

The NumPy/scipy/sklearn/torchvision cleanup is mainly for Colab. It prevents
Transformers' optional scientific/vision import paths from reaching stale binary
packages after Torch/NumPy are repinned. Restart the Colab runtime once after
the install phase so old binary modules are cleared from memory.

## Native Spaces

```text
audio x -> SAME encoder E -> latent z
prompt p -> SA3 conditioner C(p)
SA3 DiT/flow model v_theta(z_t, t, C(p))
SAME decoder D(z) -> audio
```

The research layer focuses on:

- SAME latent memory and statistics
- audio-to-soft-prompt inversion using SA3-native flow losses
- hard/babble prompt search with lexical and SA3 flow-loss scorers
- SAME latent style profiles and directions
- audioscope-style SA3 residual steering
- audio-derived residual vectors
- continuation/inpainting as composition
- local SAME geometry audit over saved latents
- LatCH-style control heads

## Research Docs

The current repo-specific math and implementation map is:

`docs/research/native-experimental-modes-math.md`

That document covers the current Colab modes, including renoise, selective latent
renoise, blur/sharpen/filtering, cross-audio grafting, cyclic sampler mixing,
neural latent DSP, soft prompt inversion, Mode 2 beam prompt inversion, SAME
statistical controls, residual steering, LatCH-style sidecars, LoRA boundaries,
and Mode 15 geometry audits.

The seven-operator research layer is:

`docs/research/seven-better-operators.md`

It maps latent geometry, covariance transport, periodic operators, direct
guidance, prompt inversion, residual feature discovery, and control
observability to the current helper modules and Colab exposure.

The neural-latent DSP notes are:

`docs/research/neural-latent-dsp.md`

They document Mode 0h: latent dynamics, soft clipping, latent-time FFT gain and
phase operators, magnitude/phase grafting, PCA component gain, SA3 polish, and
MIR descriptor audits.

Older root-level research notes are historical context from before this repo was
consolidated around the combined SA3 Native Lab implementation.
