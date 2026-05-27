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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8733
DEFAULT_FRONTEND_PORT = 5173


@dataclass(frozen=True)
class DevCheck:
    name: str
    status: str
    message: str
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != "error"


def main() -> None:
    parser = argparse.ArgumentParser(prog="sa3-lab", description="SA3 Native Lab local app tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check local app/runtime readiness.")
    _add_common_args(doctor)
    doctor.add_argument("--json", action="store_true", help="Print checks as JSON.")

    dev = subparsers.add_parser("dev", help="Run the API daemon and frontend together.")
    _add_common_args(dev)
    dev.add_argument("--reload-api", action="store_true", help="Run the API with uvicorn reload.")
    dev.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for local servers.")

    args = parser.parse_args()
    if args.command == "doctor":
        raise SystemExit(run_doctor(args))
    if args.command == "dev":
        raise SystemExit(run_dev(args))
    parser.error(f"unknown command: {args.command}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--api-port", type=int, default=DEFAULT_API_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)


def run_doctor(args: argparse.Namespace) -> int:
    root = resolve_repo_root(args.repo_root)
    checks = collect_checks(root, host=args.host, api_port=args.api_port, frontend_port=args.frontend_port)
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
    checks = collect_checks(root, host=args.host, api_port=args.api_port, frontend_port=args.frontend_port)
    blocking = [check for check in checks if check.status == "error"]
    if blocking:
        print("SA3 Native Lab cannot start yet:")
        for check in blocking:
            print(format_check(check))
        print("Run `uv run sa3-lab doctor` for the full readiness report.")
        return 1

    api_url = f"http://{args.host}:{args.api_port}"
    frontend_url = f"http://{args.host}:{args.frontend_port}/"
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

        if is_port_open(args.host, args.frontend_port):
            print(f"[ui] using existing frontend at {frontend_url}")
            wait_for_http(frontend_url, timeout=args.startup_timeout)
        else:
            env = os.environ.copy()
            env["VITE_SA3_API_BASE"] = api_url
            command = build_frontend_command(root / "frontend", port=args.frontend_port)
            children.append(start_process("ui", command, cwd=root, env=env))
            wait_for_http(frontend_url, timeout=args.startup_timeout)

        print("")
        print("SA3 Native Lab is running")
        print(f"  API:      {api_url}")
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


def resolve_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def collect_checks(root: Path, *, host: str, api_port: int, frontend_port: int) -> list[DevCheck]:
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
    checks.extend(command_checks(root))
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
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    checks.append(
        DevCheck(
            "hf-token",
            "ok" if token_present else "warn",
            "Hugging Face token is configured" if token_present else "Hugging Face token is not set",
            None if token_present else "set HF_TOKEN or run `hf auth login` before gated model downloads",
        )
    )
    return checks


def command_checks(root: Path) -> list[DevCheck]:
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
