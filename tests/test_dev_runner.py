from __future__ import annotations

import sys
from pathlib import Path

from sa3_native_lab.app.dev import (
    DevCheck,
    build_api_command,
    build_frontend_command,
    format_check,
    is_port_open,
    resolve_repo_root,
)


def test_dev_runner_builds_api_command(tmp_path):
    command = build_api_command(
        host="127.0.0.1",
        port=8733,
        artifact_root=tmp_path / "artifacts",
        repo_root=tmp_path,
        reload_api=True,
    )

    assert command[:3] == [sys.executable, "-m", "sa3_native_lab.app.cli"]
    assert "--host" in command
    assert "127.0.0.1" in command
    assert "--port" in command
    assert "8733" in command
    assert "--artifact-root" in command
    assert str(tmp_path / "artifacts") in command
    assert "--repo-root" in command
    assert str(tmp_path) in command
    assert "--reload" in command


def test_dev_runner_builds_frontend_command(tmp_path):
    command = build_frontend_command(tmp_path / "frontend", port=5174)

    assert command == [
        "npm",
        "run",
        "dev",
        "--prefix",
        str(tmp_path / "frontend"),
        "--",
        "--port",
        "5174",
    ]


def test_dev_runner_formats_checks_with_details():
    line = format_check(DevCheck("hf-token", "warn", "Hugging Face token is not set", "set HF_TOKEN"))

    assert line == "[warn] hf-token: Hugging Face token is not set (set HF_TOKEN)"


def test_dev_runner_repo_root_defaults_to_project_root():
    root = resolve_repo_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "sa3_native_lab").is_dir()


def test_dev_runner_port_probe_returns_false_for_invalid_port():
    assert is_port_open("127.0.0.1", 9, timeout=0.01) is False
