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

## First Procedure

`workbench.load` aggregates health, sessions, artifacts, jobs, the mode atlas,
and operator specs into one UI-ready workbench state. It intentionally contains
app logic such as active-session resolution, archive/session grouping, running
job filtering, and readiness summaries.

The frontend uses this procedure for workbench reads when the control-plane URL
is configured. Model execution and mutations still flow through the Python
worker until those app-level procedures are designed and tested.
