from __future__ import annotations

import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

pytest.importorskip("pydantic")

from sa3_native_lab.app.contracts import (  # noqa: E402
    ArtifactAnnotationRequest,
    BackendName,
    JobStatus,
    LatentDecodeRequest,
    LatentEncodeRequest,
    OperatorName,
    Recipe,
    TextGenerateRequest,
)
from sa3_native_lab.app.jobs import JobManager, JobResult  # noqa: E402
from sa3_native_lab.app.runtime import RuntimeDispatcher  # noqa: E402
from sa3_native_lab.app.storage import ArtifactStore  # noqa: E402


def test_latent_artifact_roundtrip_and_annotation(tmp_path):
    store = ArtifactStore(tmp_path)
    latent = np.arange(12, dtype=np.float32).reshape(4, 3)

    record = store.store_latent_array(latent, latent_rate=2.0, prompt="soft glass loop")
    loaded = store.load_latent_array(record.artifact_id)

    assert record.kind == "latent"
    assert record.latent is not None
    assert record.latent.shape == (4, 3)
    assert record.latent.duration_seconds == 2.0
    np.testing.assert_allclose(loaded, latent)

    annotated = store.annotate_artifact(
        record.artifact_id,
        ArtifactAnnotationRequest(label="keeper", tags=["bright"], metadata={"score": 0.8}),
    )

    assert annotated.label == "keeper"
    assert annotated.tags == ["bright"]
    assert annotated.metadata["score"] == 0.8


def test_audio_peaks_are_derived_from_audio_file(tmp_path):
    store = ArtifactStore(tmp_path)
    audio_path = tmp_path / "shape.wav"
    samples = np.array([0.0, 0.25, -0.5, 0.75, -1.0, 0.5, 0.25, 0.0], dtype=np.float32)
    sf.write(audio_path, samples, 8000, subtype="FLOAT")

    record = store.import_audio_file(audio_path, media_type="application/octet-stream")
    peaks = store.audio_peaks(record.artifact_id, bins=4)

    assert peaks.artifact_id == record.artifact_id
    assert record.file is not None
    assert record.file.media_type == "audio/wav"
    assert peaks.bins == 4
    assert peaks.channels == 1
    assert peaks.sample_rate == 8000
    np.testing.assert_allclose(peaks.peaks, [0.25, 0.75, 1.0, 0.25], atol=1e-6)


def test_app_defaults_target_medium_model_family():
    text_request = TextGenerateRequest()
    encode_request = LatentEncodeRequest(source_artifact_id="art_source")
    decode_request = LatentDecodeRequest(source_artifact_id="art_latent")

    assert text_request.model == "medium"
    assert text_request.cfg_scale == 1.0
    assert text_request.apg_scale == 1.0
    assert encode_request.model == "same-l"
    assert encode_request.chunk_size == 128
    assert encode_request.overlap == 32
    assert decode_request.model == "same-l"
    assert decode_request.chunk_size == 128
    assert decode_request.overlap == 32


def test_operator_specs_cover_typed_request_params(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    specs = {spec.name: spec for spec in RuntimeDispatcher(store, repo_root=tmp_path).operator_specs()}

    text_fields = set(TextGenerateRequest.model_fields) - {"backend"}
    encode_fields = set(LatentEncodeRequest.model_fields) - {"source_artifact_id", "backend"}
    decode_fields = set(LatentDecodeRequest.model_fields) - {"source_artifact_id", "backend"}

    assert text_fields <= set(specs[OperatorName.TEXT_TO_AUDIO].params)
    assert encode_fields <= set(specs[OperatorName.LATENT_ENCODE].params)
    assert decode_fields <= set(specs[OperatorName.LATENT_DECODE].params)
    assert specs[OperatorName.TEXT_TO_AUDIO].backends == [BackendName.MLX]
    assert OperatorName.EXPERIMENT_ALPHA_SWEEP in specs


def test_bundle_artifact_zips_directory_outputs(tmp_path):
    store = ArtifactStore(tmp_path)
    output_dir = tmp_path / "vectors"
    output_dir.mkdir()
    (output_dir / "summary.txt").write_text("best_layer=2\n", encoding="utf-8")
    recipe = Recipe(operator=OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT, backend=BackendName.TORCH_CPU)

    record = store.finalize_bundle_path(artifact_id="art_bundle", path=output_dir, recipe=recipe)

    assert record.kind == "bundle"
    assert record.file is not None
    assert record.file.media_type == "application/zip"
    assert record.path.suffix == ".zip"
    assert record.path.exists()
    assert store.get_recipe(recipe.recipe_id).operator == OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT


def test_job_manager_persists_success(tmp_path):
    manager = JobManager(tmp_path)
    recipe = Recipe(
        operator=OperatorName.LATENT_CYCLIC_ROLL,
        backend=BackendName.TORCH_CPU,
        params={"shift_frames": 1},
    )

    record = manager.submit(recipe, lambda context: JobResult(artifact_ids=["art_test"], metrics={"ok": True}))
    finished = _wait_for_job(manager, record.job_id)

    assert finished.status == JobStatus.SUCCEEDED
    assert finished.artifact_ids == ["art_test"]
    assert finished.metrics["ok"] is True
    assert (tmp_path / f"{record.job_id}.json").exists()


def test_runtime_applies_latent_operator(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    source = store.store_latent_array(np.random.default_rng(0).normal(size=(8, 4)), latent_rate=4.0)
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)
    jobs = JobManager(store.jobs_dir)
    recipe = Recipe(
        operator=OperatorName.LATENT_BLUR,
        backend=BackendName.TORCH_CPU,
        inputs={"source": source.artifact_id},
        params={"mode": "temporal", "temporal_radius": 1, "strength": 0.5},
    )

    record = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
    finished = _wait_for_job(jobs, record.job_id)

    assert finished.status == JobStatus.SUCCEEDED
    assert len(finished.artifact_ids) == 1
    assert store.get_recipe(recipe.recipe_id).operator == OperatorName.LATENT_BLUR
    output = store.get_artifact(finished.artifact_ids[0])
    assert output.latent is not None
    assert output.latent.shape == source.latent.shape
    assert output.source_artifact_ids == [source.artifact_id]


def _wait_for_job(manager: JobManager, job_id: str):
    deadline = time.time() + 10
    while time.time() < deadline:
        record = manager.get(job_id)
        if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return record
        time.sleep(0.05)
    raise AssertionError(f"job did not finish: {job_id}")


def test_fastapi_surface_imports_and_runs_latent_job(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    from fastapi.testclient import TestClient

    from sa3_native_lab.app.server import create_app

    app = create_app(artifact_root=tmp_path / "lab", repo_root=tmp_path)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert any(item["backend"] == "mlx" for item in health.json()["backends"])

    audio_bytes = BytesIO()
    sf.write(audio_bytes, np.linspace(-1.0, 1.0, 16, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
    audio_bytes.seek(0)
    audio = client.post(
        "/audio/import",
        files={"file": ("tone.wav", audio_bytes, "application/octet-stream")},
    )
    assert audio.status_code == 200
    assert audio.json()["file"]["media_type"] == "audio/wav"
    peaks = client.get(f"/artifacts/{audio.json()['artifact_id']}/peaks?bins=8")
    assert peaks.status_code == 200
    assert peaks.json()["bins"] == 8
    assert len(peaks.json()["peaks"]) == 8

    latent_bytes = BytesIO()
    np.save(latent_bytes, np.ones((6, 3), dtype=np.float32))
    latent_bytes.seek(0)
    imported = client.post(
        "/latents/import",
        files={"file": ("latent.npy", latent_bytes, "application/octet-stream")},
        data={"latent_rate": "3.0"},
    )
    assert imported.status_code == 200
    artifact_id = imported.json()["artifact_id"]

    job = client.post(
        "/operators/run",
        json={
            "operator": "latent.cyclic_roll",
            "backend": "torch_cpu",
            "inputs": {"source": artifact_id},
            "params": {"shift_frames": 1},
        },
    )
    assert job.status_code == 200
    finished = _wait_for_api_job(client, job.json()["job_id"])
    assert finished["status"] == "succeeded"
    assert len(finished["artifact_ids"]) == 1
    recipe = client.get(f"/recipes/{finished['recipe']['recipe_id']}")
    assert recipe.status_code == 200
    assert recipe.json()["operator"] == "latent.cyclic_roll"


def test_colab_mode_atlas_covers_numbered_modes(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sa3_native_lab.app.server import create_app

    client = TestClient(create_app(artifact_root=tmp_path / "lab", repo_root=tmp_path))
    response = client.get("/colab/modes")

    assert response.status_code == 200
    modes = response.json()
    mode_ids = {mode["mode_id"] for mode in modes}
    for expected in ["0", "0c", "0e", "0d", "0h", "0f", "0g", "1", "1b", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]:
        assert expected in mode_ids
    assert any(mode["status"] == "native recipe" and "experiment.alpha_sweep" in mode["operators"] for mode in modes)


def test_runtime_same_encode_decode_with_fake_adapter(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, np.zeros((64, 2), dtype=np.float32), 44100)
    audio = store.import_audio_file(audio_path)
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)
    runtime._same_adapter = lambda model_name, backend: FakeSameAdapter()
    jobs = JobManager(store.jobs_dir)

    encode_recipe = Recipe(
        operator=OperatorName.LATENT_ENCODE,
        backend=BackendName.TORCH_CPU,
        inputs={"source": audio.artifact_id},
        params={"model": "same-s", "chunked": False},
        model="same-s",
    )
    encode_record = jobs.submit(encode_recipe, runtime.handler_for_recipe(encode_recipe))
    encode_finished = _wait_for_job(jobs, encode_record.job_id)

    assert encode_finished.status == JobStatus.SUCCEEDED
    latent = store.get_artifact(encode_finished.artifact_ids[0])
    assert latent.latent is not None
    assert latent.latent.shape == (4, 3)
    assert latent.source_artifact_ids == [audio.artifact_id]

    decode_recipe = Recipe(
        operator=OperatorName.LATENT_DECODE,
        backend=BackendName.TORCH_CPU,
        inputs={"source": latent.artifact_id},
        params={"model": "same-s", "chunked": False},
        model="same-s",
    )
    decode_record = jobs.submit(decode_recipe, runtime.handler_for_recipe(decode_recipe))
    decode_finished = _wait_for_job(jobs, decode_record.job_id)

    assert decode_finished.status == JobStatus.SUCCEEDED
    decoded = store.get_artifact(decode_finished.artifact_ids[0])
    assert decoded.audio is not None
    assert decoded.audio.sample_rate == 44100
    assert decoded.source_artifact_ids == [latent.artifact_id]


def test_runtime_wraps_script_experiment_audio_output(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)
    jobs = JobManager(store.jobs_dir)

    def fake_subprocess(command, *, cwd, context):
        output = _command_value(command, "--output")
        sf.write(output, np.zeros((32, 1), dtype=np.float32), 16000, format="WAV")
        context.log("saved fake styled audio")
        return 0

    runtime._run_subprocess = fake_subprocess
    profile_path = tmp_path / "profile.npz"
    profile_path.write_bytes(b"fake")
    recipe = Recipe(
        operator=OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE,
        backend=BackendName.TORCH_CPU,
        params={"profile_path": str(profile_path), "prompt": "soft glass", "steps": 1},
        model="medium",
        seed=3,
    )

    record = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
    finished = _wait_for_job(jobs, record.job_id)

    assert finished.status == JobStatus.SUCCEEDED
    assert len(finished.artifact_ids) == 1
    artifact = store.get_artifact(finished.artifact_ids[0])
    assert artifact.kind == "audio"
    assert artifact.audio is not None
    assert artifact.prompt == "soft glass"
    assert artifact.recipe_id == recipe.recipe_id


def test_runtime_wraps_script_experiment_bundle_output(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)
    jobs = JobManager(store.jobs_dir)

    def fake_subprocess(command, *, cwd, context):
        output = _command_value(command, "--output")
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output) / "steering_vectors.pt").write_bytes(b"fake")
        context.log("saved fake vectors")
        return 0

    runtime._run_subprocess = fake_subprocess
    recipe = Recipe(
        operator=OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT,
        backend=BackendName.TORCH_CPU,
        params={"axis": "valence", "num_pairs": 1, "steps": 1},
        model="medium",
    )

    record = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
    finished = _wait_for_job(jobs, record.job_id)

    assert finished.status == JobStatus.SUCCEEDED
    assert len(finished.artifact_ids) == 1
    artifact = store.get_artifact(finished.artifact_ids[0])
    assert artifact.kind == "bundle"
    assert artifact.file is not None
    assert artifact.file.media_type == "application/zip"
    assert artifact.recipe_id == recipe.recipe_id


def _wait_for_api_job(client, job_id: str):
    deadline = time.time() + 10
    while time.time() < deadline:
        response = client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"API job did not finish: {job_id}")


class FakeLatentItem:
    def __init__(self) -> None:
        self.latent = np.ones((4, 3), dtype=np.float32)
        self.latent_rate = 10.0
        self.sample_rate = 44100
        self.prompt = "fake"
        self.metadata = {"model_name": "fake-same"}


class FakeAutoencoder:
    device = "cpu"


class FakeSameAdapter:
    sample_rate = 44100
    autoencoder = FakeAutoencoder()

    def encode_file(self, *args, **kwargs):
        return FakeLatentItem()

    def decode_latents(self, *args, **kwargs):
        return torch.zeros(1, 2, 64)


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]
