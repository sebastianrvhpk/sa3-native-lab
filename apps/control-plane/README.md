# SA3 Native Lab Control Plane

Experimental TypeScript control plane for SA3 Native Lab.

The goal is not to replace the Python runtime. Python remains the local model
worker for MLX, PyTorch, SAME, scripts, and artifact file IO. This package adds
tRPC procedures that shape those runtime records into app-native contracts.

## Run

```bash
npm install --prefix apps/control-plane
npm run test --prefix apps/control-plane
SA3_PYTHON_API_BASE=http://127.0.0.1:8733 npm run dev --prefix apps/control-plane
```

By default the server listens on `127.0.0.1:8787` and expects the Python runtime
at `http://127.0.0.1:8733`.

```bash
uv run sa3-lab dev --with-control-plane
```

The second command is the preferred local path because it starts the Python API,
this control plane, and the frontend with `VITE_SA3_CONTROL_PLANE_URL` already
set. `GET /health` is available for the dev runner.

## Current Procedures

`workbench.load` aggregates health, sessions, artifacts, jobs, the mode atlas,
operator specs, and result-family records into one UI-ready workbench state.
The frontend currently presents those records as branches and owns product-layer
models for Memory reuse, Next actions, Pending Take landing, Branch summaries,
and Tune field grouping. `workbench.load` intentionally contains app logic such
as active-session resolution, archive/session grouping, running job filtering,
result-family grouping, and readiness summaries without replacing the Python
runtime.

The control plane also exposes:

- `system.readiness`
- `jobs.list/get/cancel/retry`
- `recipes.replay/fork`
- `artifacts.inspect`
- `families.load`
- `archive.search`
- `archive.annotateAndSearch`

The frontend uses this path when the control-plane URL is configured. Heavy
model execution and artifact file IO still flow through the Python worker.
