# SA3 Native Lab Control Plane

Experimental TypeScript control plane for SA3 Native Lab.

The goal is not to replace the Python runtime. Python remains the local model
worker for MLX, PyTorch, SAME, scripts, and artifact file IO. This package adds
tRPC procedures that shape those runtime records into app-native contracts.

## Run

```bash
npm install
npm run test
npm run dev
```

By default the server listens on `127.0.0.1:8787` and expects the Python runtime
at `http://127.0.0.1:8733`.

```bash
SA3_PYTHON_API_BASE=http://127.0.0.1:8733 npm run dev
```

## First Procedure

`workbench.load` aggregates health, sessions, artifacts, jobs, the mode atlas,
and operator specs into one UI-ready workbench state. It intentionally contains
app logic such as active-session resolution, archive/session grouping, running
job filtering, and readiness summaries.
