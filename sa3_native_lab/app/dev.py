from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import (
    ArtifactAnnotationRequest,
    ArtifactKind,
    BackendName,
    JobRecord,
    JobStatus,
    OperatorName,
    Recipe,
    SessionCreateRequest,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8733
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_CONTROL_PLANE_PORT = 8787
TERMINAL_JOB_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}


@dataclass(frozen=True)
class DevCheck:
    name: str
    status: str
    message: str
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != "error"


@dataclass(frozen=True)
class SmokeFixtureResult:
    status: str
    message: str
    artifact_root: str
    session_id: str
    job_id: str
    source_artifact_id: str
    output_artifact_id: str | None
    phase: str | None
    progress: float
    elapsed_seconds: float
    metrics: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(prog="sa3-lab", description="SA3 Native Lab local app tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local app/runtime readiness.")
    _add_common_args(doctor)
    doctor.add_argument("--with-control-plane", action="store_true", help="Also check optional tRPC control-plane readiness.")
    doctor.add_argument("--json", action="store_true", help="Print checks as JSON.")

    dev = subparsers.add_parser("dev", help="Run the API daemon and frontend together.")
    _add_common_args(dev)
    dev.add_argument("--with-control-plane", action="store_true", help="Run the tRPC control plane and point the frontend at it.")
    dev.add_argument("--reload-api", action="store_true", help="Run the API with uvicorn reload.")
    dev.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for local servers.")

    smoke = subparsers.add_parser(
        "smoke-fixture",
        help="Run a fast fixture-backed runtime job through storage, jobs, and the runtime dispatcher.",
    )
    _add_common_args(smoke)
    smoke.add_argument("--timeout", type=float, default=15.0, help="Seconds to wait for the fixture job.")
    smoke.add_argument("--json", action="store_true", help="Print the smoke result as JSON.")

    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(run_doctor(args))
    if args.command == "dev":
        raise SystemExit(run_dev(args))
    if args.command == "smoke-fixture":
        raise SystemExit(run_fixture_smoke_command(args))
    parser.error(f"unknown command: {args.command}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--control-plane-port", type=int, default=DEFAULT_CONTROL_PLANE_PORT)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)


def run_doctor(args: argparse.Namespace) -> int:
    root = resolve_repo_root(args.repo_root)
    checks = collect_checks(
        root,
        host=args.host,
        api_port=args.api_port,
        frontend_port=args.frontend_port,
        control_plane_port=args.control_plane_port,
        with_control_plane=args.with_control_plane,
    )
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        print("SA3 Native Lab doctor")
        for check in checks:
            print(format_check(check))
    return 1 if any(check.status == "error" for check in checks) else 0


def run_dev(args: argparse.Namespace) -> int:
    root = resolve_repo_root(args.repo_root)
    artifact_root = args.artifact_root or root / ".sa3_lab"
    checks = collect_checks(
        root,
        host=args.host,
        api_port=args.api_port,
        frontend_port=args.frontend_port,
        control_plane_port=args.control_plane_port,
        with_control_plane=args.with_control_plane,
    )
    blocking = [check for check in checks if check.status == "error"]
    if blocking:
        print("SA3 Native Lab cannot start yet:")
        for check in blocking:
            print(format_check(check))
        print("Run `uv run sa3-lab doctor` for the full readiness report.")
        return 1

    api_url = f"http://{args.host}:{args.api_port}"
    frontend_url = f"http://{args.host}:{args.frontend_port}/"
    control_plane_url = f"http://{args.host}:{args.control_plane_port}"
    children: list[subprocess.Popen[str]] = []

    try:
        if is_port_open(args.host, args.api_port):
            print(f"[api] using existing server at {api_url}")
            wait_for_http(f"{api_url}/health", timeout=args.startup_timeout)
            print_health_summary(api_url)
        else:
            command = build_api_command(
                host=args.host,
                port=args.api_port,
                artifact_root=artifact_root,
                repo_root=root,
                reload_api=args.reload_api,
            )
            children.append(start_process("api", command, cwd=root, env=os.environ.copy()))
            wait_for_http(f"{api_url}/health", timeout=args.startup_timeout)
            print_health_summary(api_url)

        if args.with_control_plane:
            if is_port_open(args.host, args.control_plane_port):
                print(f"[control] using existing server at {control_plane_url}/trpc")
                wait_for_http(f"{control_plane_url}/health", timeout=args.startup_timeout)
            else:
                command = build_control_plane_command(root / "apps" / "control-plane")
                env = build_control_plane_env(
                    os.environ,
                    api_url=api_url,
                    control_plane_port=args.control_plane_port,
                )
                children.append(start_process("control", command, cwd=root, env=env))
                wait_for_http(f"{control_plane_url}/health", timeout=args.startup_timeout)

        if is_port_open(args.host, args.frontend_port):
            print(f"[ui] using existing frontend at {frontend_url}")
            if args.with_control_plane:
                print("[ui] restart the existing frontend if it was not started with this control-plane URL.")
            wait_for_http(frontend_url, timeout=args.startup_timeout)
        else:
            env = build_frontend_env(
                os.environ,
                api_url=api_url,
                control_plane_url=control_plane_url if args.with_control_plane else None,
            )
            command = build_frontend_command(root / "frontend", port=args.frontend_port)
            children.append(start_process("ui", command, cwd=root, env=env))
            wait_for_http(frontend_url, timeout=args.startup_timeout)

        print("")
        print("SA3 Native Lab is running")
        print(f"  API:      {api_url}")
        if args.with_control_plane:
            print(f"  Control:  {control_plane_url}/trpc")
        print(f"  Frontend: {frontend_url}")
        print(f"  Artifacts: {artifact_root}")
        if children:
            print("Press Ctrl-C to stop the processes started by this command.")
            return wait_for_children(children)
        print("Both services were already running; nothing new was started.")
        return 0
    except KeyboardInterrupt:
        print("\nStopping SA3 Native Lab...")
        return 130
    except RuntimeError as exc:
        print(f"Startup failed: {exc}")
        return 1
    finally:
        stop_children(children)


def run_fixture_smoke_command(args: argparse.Namespace) -> int:
    root = resolve_repo_root(args.repo_root)
    artifact_root = args.artifact_root or root / ".sa3_lab" / "smoke-fixture"
    try:
        result = run_fixture_smoke(root=root, artifact_root=artifact_root, timeout=args.timeout)
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "message": str(exc),
                        "artifact_root": str(artifact_root),
                    },
                    indent=2,
                )
            )
        else:
            print(f"SA3 Native Lab fixture smoke failed: {exc}")
        return 1

    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("SA3 Native Lab fixture smoke")
        print(f"[{result.status}] {result.message}")
        print(f"  artifact root: {result.artifact_root}")
        print(f"  session:       {result.session_id}")
        print(f"  job:           {result.job_id}")
        print(f"  source:        {result.source_artifact_id}")
        print(f"  output:        {result.output_artifact_id}")
        print(f"  elapsed:       {result.elapsed_seconds:.2f}s")
    return 0 if result.status == "ok" else 1


def run_fixture_smoke(*, root: Path, artifact_root: Path, timeout: float = 15.0) -> SmokeFixtureResult:
    import numpy as np

    from .jobs import JobManager
    from .runtime import RuntimeDispatcher
    from .storage import ArtifactStore

    start = time.monotonic()
    store = ArtifactStore(artifact_root)
    runtime = RuntimeDispatcher(store, repo_root=root)
    jobs = JobManager(artifact_root / "jobs", max_workers=1)
    try:
        session = store.create_session(
            SessionCreateRequest(
                name="Fixture smoke",
                notes="Fast local smoke that exercises storage, jobs, runtime dispatch, and artifact persistence.",
                metadata={"smoke_fixture": True},
            )
        )
        fixture = np.linspace(-1.0, 1.0, num=64, dtype=np.float32).reshape(16, 4)
        source = store.store_latent_array(
            fixture,
            latent_rate=4.0,
            filename="fixture-source.npy",
            session_id=session.session_id,
            label="fixture source latent",
            metadata={
                "smoke_fixture": True,
                "fixture_role": "source",
                "description": "small deterministic time-major latent used by sa3-lab smoke-fixture",
            },
        )
        recipe = Recipe(
            operator=OperatorName.LATENT_CYCLIC_ROLL,
            backend=BackendName.TORCH_CPU,
            inputs={"source": source.artifact_id},
            params={"shift_frames": 3, "strength": 1.0, "symmetric": False},
            notes="Fixture smoke cyclic roll",
            session_id=session.session_id,
        )
        job = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
        final = wait_for_job_terminal(jobs, job.job_id, timeout=timeout)
        elapsed = time.monotonic() - start
        if final.status != JobStatus.SUCCEEDED:
            detail = final.error or final.message or final.status.value
            raise RuntimeError(f"fixture job ended with {final.status.value}: {detail}")
        if not final.artifact_ids:
            raise RuntimeError("fixture job succeeded without output artifacts")
        output = store.get_artifact(final.artifact_ids[0])
        if output.kind != ArtifactKind.LATENT or output.latent is None:
            raise RuntimeError(f"fixture job produced non-latent artifact: {output.artifact_id}")
        output = store.annotate_artifact(
            output.artifact_id,
            ArtifactAnnotationRequest(
                metadata={
                    "smoke_fixture": True,
                    "fixture_role": "output",
                    "smoke_job_id": final.job_id,
                    "smoke_source_artifact_id": source.artifact_id,
                }
            ),
        )
        return SmokeFixtureResult(
            status="ok",
            message="fixture latent operator completed and persisted an output artifact",
            artifact_root=str(artifact_root),
            session_id=session.session_id,
            job_id=final.job_id,
            source_artifact_id=source.artifact_id,
            output_artifact_id=output.artifact_id,
            phase=final.phase,
            progress=final.progress,
            elapsed_seconds=elapsed,
            metrics=final.metrics,
        )
    finally:
        jobs.shutdown(wait=True)


def wait_for_job_terminal(jobs: Any, job_id: str, *, timeout: float) -> JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = jobs.get(job_id)
        if record.status in TERMINAL_JOB_STATUSES:
            return record
        time.sleep(0.05)
    record = jobs.get(job_id)
    raise TimeoutError(
        f"timed out waiting for job {job_id}; status={record.status.value}, phase={record.phase}, "
        f"message={record.message}"
    )


def resolve_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def collect_checks(
    root: Path,
    *,
    host: str,
    api_port: int,
    frontend_port: int,
    control_plane_port: int = DEFAULT_CONTROL_PLANE_PORT,
    with_control_plane: bool = False,
) -> list[DevCheck]:
    checks: list[DevCheck] = []
    checks.append(
        DevCheck(
            "repo",
            "ok" if (root / "pyproject.toml").exists() else "error",
            f"repo root: {root}",
            None if (root / "pyproject.toml").exists() else "missing pyproject.toml",
        )
    )
    checks.append(
        DevCheck(
            "frontend",
            "ok" if (root / "frontend" / "package.json").exists() else "error",
            "frontend package is present",
            None if (root / "frontend" / "package.json").exists() else "missing frontend/package.json",
        )
    )
    checks.extend(command_checks(root, with_control_plane=with_control_plane))
    checks.extend(import_checks())
    checks.extend(runtime_checks(root))
    checks.append(
        DevCheck(
            "api-port",
            "warn" if is_port_open(host, api_port) else "ok",
            f"API port {host}:{api_port} is {'already in use' if is_port_open(host, api_port) else 'available'}",
        )
    )
    checks.append(
        DevCheck(
            "frontend-port",
            "warn" if is_port_open(host, frontend_port) else "ok",
            f"frontend port {host}:{frontend_port} is {'already in use' if is_port_open(host, frontend_port) else 'available'}",
        )
    )
    if with_control_plane:
        checks.append(
            DevCheck(
                "control-plane-port",
                "warn" if is_port_open(host, control_plane_port) else "ok",
                f"control-plane port {host}:{control_plane_port} is {'already in use' if is_port_open(host, control_plane_port) else 'available'}",
            )
        )
    checks.append(huggingface_auth_check())
    return checks


def command_checks(root: Path, *, with_control_plane: bool = False) -> list[DevCheck]:
    checks = [
        DevCheck(
            "npm",
            "ok" if shutil.which("npm") else "error",
            "npm is available" if shutil.which("npm") else "npm was not found on PATH",
        )
    ]
    vite_bin = root / "frontend" / "node_modules" / ".bin" / "vite"
    checks.append(
        DevCheck(
            "frontend-deps",
            "ok" if vite_bin.exists() else "error",
            "frontend dependencies are installed" if vite_bin.exists() else "frontend dependencies are missing",
            None if vite_bin.exists() else "run `npm install --prefix frontend`",
        )
    )
    if with_control_plane:
        control_plane_dir = root / "apps" / "control-plane"
        checks.append(
            DevCheck(
                "control-plane",
                "ok" if (control_plane_dir / "package.json").exists() else "error",
                "control-plane package is present",
                None if (control_plane_dir / "package.json").exists() else "missing apps/control-plane/package.json",
            )
        )
        tsx_bin = control_plane_dir / "node_modules" / ".bin" / "tsx"
        checks.append(
            DevCheck(
                "control-plane-deps",
                "ok" if tsx_bin.exists() else "error",
                "control-plane dependencies are installed" if tsx_bin.exists() else "control-plane dependencies are missing",
                None if tsx_bin.exists() else "run `npm install --prefix apps/control-plane`",
            )
        )
    return checks


def import_checks() -> list[DevCheck]:
    checks: list[DevCheck] = []
    for module_name in ("fastapi", "uvicorn", "pydantic"):
        try:
            __import__(module_name)
        except ImportError:
            checks.append(
                DevCheck(
                    module_name,
                    "error",
                    f"{module_name} is not importable",
                    "run `uv sync --extra app --extra dev`",
                )
            )
        else:
            checks.append(DevCheck(module_name, "ok", f"{module_name} is importable"))
    return checks


def runtime_checks(root: Path) -> list[DevCheck]:
    checks: list[DevCheck] = []
    try:
        from .runtime import RuntimeDispatcher
        from .storage import ArtifactStore

        runtime = RuntimeDispatcher(ArtifactStore(root / ".sa3_lab"), repo_root=root)
        for status in runtime.backend_statuses():
            level = "ok" if status.available else "warn"
            checks.append(
                DevCheck(
                    f"backend:{status.backend.value}",
                    level,
                    status.message or f"{status.backend.value} backend status checked",
                    json.dumps(status.details, default=str) if status.details else None,
                )
            )
    except Exception as exc:  # pragma: no cover - defensive doctor path.
        checks.append(DevCheck("backends", "warn", "backend status check failed", str(exc)))
    return checks


def huggingface_auth_check() -> DevCheck:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return DevCheck("hf-auth", "ok", "Hugging Face token is configured from the environment")
    try:
        from huggingface_hub import get_token
    except ImportError:
        return DevCheck(
            "hf-auth",
            "warn",
            "Hugging Face auth could not be checked",
            "huggingface_hub is not importable",
        )
    if get_token():
        return DevCheck("hf-auth", "ok", "Hugging Face token is available from the local auth cache")
    return DevCheck(
        "hf-auth",
        "warn",
        "Hugging Face token is not configured",
        "run `hf auth login` or set HF_TOKEN before gated model downloads",
    )


def build_api_command(
    *,
    host: str,
    port: int,
    artifact_root: Path,
    repo_root: Path,
    reload_api: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sa3_native_lab.app.cli",
        "--host",
        host,
        "--port",
        str(port),
        "--artifact-root",
        str(artifact_root),
        "--repo-root",
        str(repo_root),
    ]
    if reload_api:
        command.append("--reload")
    return command


def build_frontend_command(frontend_dir: Path, *, port: int) -> list[str]:
    return ["npm", "run", "dev", "--prefix", str(frontend_dir), "--", "--port", str(port)]


def build_control_plane_command(control_plane_dir: Path) -> list[str]:
    return ["npm", "run", "dev", "--prefix", str(control_plane_dir)]


def build_frontend_env(
    base_env: Mapping[str, str],
    *,
    api_url: str,
    control_plane_url: str | None = None,
) -> dict[str, str]:
    env = dict(base_env)
    env["VITE_SA3_API_BASE"] = api_url
    if control_plane_url:
        env["VITE_SA3_CONTROL_PLANE_URL"] = control_plane_url
    else:
        env.pop("VITE_SA3_CONTROL_PLANE_URL", None)
    return env


def build_control_plane_env(
    base_env: Mapping[str, str],
    *,
    api_url: str,
    control_plane_port: int,
) -> dict[str, str]:
    env = dict(base_env)
    env["SA3_PYTHON_API_BASE"] = api_url
    env["SA3_CONTROL_PLANE_PORT"] = str(control_plane_port)
    return env


def start_process(
    label: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    print(f"[{label}] starting: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    thread = threading.Thread(target=stream_output, args=(label, process), daemon=True)
    thread.start()
    return process


def stream_output(label: str, process: subprocess.Popen[str]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        print(f"[{label}] {line}", end="")


def wait_for_children(children: list[subprocess.Popen[str]]) -> int:
    while True:
        for process in children:
            code = process.poll()
            if code is not None:
                return code
        time.sleep(0.25)


def stop_children(children: list[subprocess.Popen[str]]) -> None:
    for process in children:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in children:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()


def is_port_open(host: str, port: int, *, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_http(url: str, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def fetch_json(url: str, *, timeout: float = 2.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def print_health_summary(api_url: str) -> None:
    try:
        health = fetch_json(f"{api_url}/health")
    except Exception as exc:
        print(f"[api] health summary unavailable: {exc}")
        return
    print(f"[api] artifacts: {health.get('artifact_root')}")
    for backend in health.get("backends", []):
        available = "ready" if backend.get("available") else "unavailable"
        device = backend.get("device") or "unknown"
        print(f"[api] backend {backend.get('backend')}: {available} on {device} - {backend.get('message')}")


def format_check(check: DevCheck) -> str:
    suffix = f" ({check.detail})" if check.detail else ""
    return f"[{check.status}] {check.name}: {check.message}{suffix}"


if __name__ == "__main__":
    main()
