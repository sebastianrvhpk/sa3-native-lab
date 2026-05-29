from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sa3_native_lab.app.contracts import ArtifactKind
from sa3_native_lab.app.dev import (
    DevCheck,
    build_api_command,
    build_control_plane_command,
    build_control_plane_env,
    build_frontend_command,
    build_frontend_env,
    format_check,
    huggingface_auth_check,
    is_port_open,
    resolve_repo_root,
    run_fixture_smoke,
)
from sa3_native_lab.app.storage import ArtifactStore


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


def test_dev_runner_builds_control_plane_command(tmp_path):
    command = build_control_plane_command(tmp_path / "apps" / "control-plane")

    assert command == ["npm", "run", "dev", "--prefix", str(tmp_path / "apps" / "control-plane")]


def test_dev_runner_builds_frontend_env_with_optional_control_plane():
    env = build_frontend_env(
        {"VITE_SA3_CONTROL_PLANE_URL": "http://stale.local/trpc"},
        api_url="http://127.0.0.1:8733",
    )

    assert env["VITE_SA3_API_BASE"] == "http://127.0.0.1:8733"
    assert "VITE_SA3_CONTROL_PLANE_URL" not in env

    env = build_frontend_env({}, api_url="http://127.0.0.1:8733", control_plane_url="http://127.0.0.1:8787")

    assert env["VITE_SA3_CONTROL_PLANE_URL"] == "http://127.0.0.1:8787"


def test_dev_runner_builds_control_plane_env():
    env = build_control_plane_env({}, api_url="http://127.0.0.1:8733", control_plane_port=8788)

    assert env["SA3_PYTHON_API_BASE"] == "http://127.0.0.1:8733"
    assert env["SA3_CONTROL_PLANE_PORT"] == "8788"


def test_dev_runner_formats_checks_with_details():
    line = format_check(DevCheck("hf-auth", "warn", "Hugging Face token is not set", "set HF_TOKEN"))

    assert line == "[warn] hf-auth: Hugging Face token is not set (set HF_TOKEN)"


def test_huggingface_auth_check_accepts_environment_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    check = huggingface_auth_check()

    assert check.name == "hf-auth"
    assert check.status == "ok"
    assert "environment" in check.message


def test_dev_runner_repo_root_defaults_to_project_root():
    root = resolve_repo_root()

    assert (root / "pyproject.toml").exists()
    assert (root / "sa3_native_lab").is_dir()


def test_dev_runner_port_probe_returns_false_for_invalid_port():
    assert is_port_open("127.0.0.1", 9, timeout=0.01) is False


def test_fixture_smoke_runs_local_runtime_job(tmp_path):
    artifact_root = tmp_path / "smoke"

    result = run_fixture_smoke(root=resolve_repo_root(), artifact_root=artifact_root, timeout=10.0)

    assert result.status == "ok"
    assert result.source_artifact_id
    assert result.output_artifact_id
    assert result.output_artifact_id != result.source_artifact_id
    assert result.progress == 1.0
    assert result.phase == "done"

    store = ArtifactStore(artifact_root)
    source = store.get_artifact(result.source_artifact_id)
    output = store.get_artifact(result.output_artifact_id)
    assert source.kind == ArtifactKind.LATENT
    assert output.kind == ArtifactKind.LATENT
    assert output.session_id == result.session_id
    assert output.source_artifact_ids == [source.artifact_id]
    assert output.metadata["smoke_fixture"] is True
    assert output.metadata["smoke_job_id"] == result.job_id
    assert output.metadata["smoke_source_artifact_id"] == source.artifact_id
    assert np.load(source.path, allow_pickle=False).shape == np.load(output.path, allow_pickle=False).shape
