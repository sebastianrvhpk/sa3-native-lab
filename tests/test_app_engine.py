from __future__ import annotations

import json
import re
import time
from io import BytesIO
from io import StringIO
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import soundfile as sf
import torch

pytest.importorskip("pydantic")

from sa3_native_lab.app.contracts import (  # noqa: E402
    ArtifactAnnotationRequest,
    ArtifactKind,
    AudioToAudioRequest,
    BackendName,
    InpaintRequest,
    JobStatus,
    LatentDecodeRequest,
    LatentEncodeRequest,
    OperatorName,
    Recipe,
    SessionCreateRequest,
    SessionStatus,
    SessionUpdateRequest,
    TextGenerateRequest,
)
from sa3_native_lab.app.colab_modes import list_colab_modes  # noqa: E402
from sa3_native_lab.app.jobs import JobManager, JobResult  # noqa: E402
from sa3_native_lab.app.runtime import RuntimeDispatcher, _download_sa3_model_files_with_progress  # noqa: E402
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


def test_artifact_annotation_can_recover_into_active_session(tmp_path):
    store = ArtifactStore(tmp_path)
    source_session = store.create_session(SessionCreateRequest(name="old"))
    target_session = store.create_session(SessionCreateRequest(name="current"))
    record = store.store_latent_array(np.zeros((3, 2), dtype=np.float32), latent_rate=1.5, session_id=source_session.session_id)

    recovered = store.annotate_artifact(
        record.artifact_id,
        ArtifactAnnotationRequest(
            session_id=target_session.session_id,
            metadata={"recovered_from_session_id": source_session.session_id},
        ),
    )

    assert recovered.session_id == target_session.session_id
    assert recovered.metadata["recovered_from_session_id"] == source_session.session_id
    assert store.get_artifact(record.artifact_id).session_id == target_session.session_id


def test_artifact_annotation_can_clear_session_for_archive(tmp_path):
    store = ArtifactStore(tmp_path)
    source_session = store.create_session(SessionCreateRequest(name="current"))
    record = store.store_latent_array(np.zeros((3, 2), dtype=np.float32), latent_rate=1.5, session_id=source_session.session_id)

    archived = store.annotate_artifact(
        record.artifact_id,
        ArtifactAnnotationRequest(
            session_id=None,
            metadata={"archived_from_session_id": source_session.session_id},
        ),
    )

    assert archived.session_id is None
    assert archived.metadata["archived_from_session_id"] == source_session.session_id
    assert store.get_artifact(record.artifact_id).session_id is None


def test_artifact_search_filters_annotation_text_and_tags(tmp_path):
    store = ArtifactStore(tmp_path)
    first = store.store_latent_array(np.ones((4, 3), dtype=np.float32), latent_rate=2.0)
    second = store.store_latent_array(np.zeros((4, 3), dtype=np.float32), latent_rate=2.0)
    store.annotate_artifact(
        first.artifact_id,
        ArtifactAnnotationRequest(label="keeper glass", notes="wide ringing texture", tags=["favorite", "loop"]),
    )
    store.annotate_artifact(second.artifact_id, ArtifactAnnotationRequest(label="discard", tags=["scratch"]))

    by_text = store.list_artifacts(query="ringing")
    by_tag = store.list_artifacts(tags=["favorite"])
    by_text_and_tag = store.list_artifacts(query="glass", tags=["favorite", "loop"])
    missing = store.list_artifacts(query="glass", tags=["scratch"])

    assert [item.artifact_id for item in by_text] == [first.artifact_id]
    assert [item.artifact_id for item in by_tag] == [first.artifact_id]
    assert [item.artifact_id for item in by_text_and_tag] == [first.artifact_id]
    assert missing == []


def test_runtime_memory_query_returns_nearest_latent_artifacts(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    source = store.store_latent_array(np.full((6, 3), 0.0, dtype=np.float32), latent_rate=2.0, label="source")
    near = store.store_latent_array(np.full((6, 3), 0.2, dtype=np.float32), latent_rate=2.0, label="near")
    store.store_latent_array(np.full((6, 3), 4.0, dtype=np.float32), latent_rate=2.0, label="far")
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)
    jobs = JobManager(store.jobs_dir)
    recipe = Recipe(
        operator=OperatorName.MEMORY_QUERY,
        backend=BackendName.CPU,
        inputs={"source": source.artifact_id},
        params={"top_k": 1, "metric": "euclidean", "exclude_self": True},
    )

    record = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
    finished = _wait_for_job(jobs, record.job_id)
    artifact = store.get_artifact(finished.artifact_ids[0])
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))

    assert finished.status == JobStatus.SUCCEEDED
    assert artifact.kind == "bundle"
    assert payload["results"][0]["artifact_id"] == near.artifact_id
    assert artifact.metadata["result_count"] == 1
    inspection = store.inspect_artifact(artifact.artifact_id)
    assert inspection.bundle_preview["result_count"] == 1
    assert inspection.bundle_preview["results"][0]["artifact_id"] == near.artifact_id
    assert inspection.bundle_summary["kind"] == "memory"
    assert inspection.bundle_summary["memory"]["result_count"] == 1
    assert inspection.bundle_summary["memory"]["results"][0]["artifact_id"] == near.artifact_id


def test_runtime_geometry_audit_returns_report_bundle(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    session = store.create_session(SessionCreateRequest(name="geometry"))
    source = store.store_latent_array(np.zeros((8, 3), dtype=np.float32), latent_rate=2.0, session_id=session.session_id)
    store.store_latent_array(np.ones((8, 3), dtype=np.float32), latent_rate=2.0, session_id=session.session_id)
    store.store_latent_array(np.ones((8, 2), dtype=np.float32), latent_rate=2.0, session_id=session.session_id)
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)
    jobs = JobManager(store.jobs_dir)
    recipe = Recipe(
        operator=OperatorName.EXPERIMENT_GEOMETRY_AUDIT,
        backend=BackendName.CPU,
        inputs={"source": source.artifact_id},
        params={"n_components": 2},
        session_id=session.session_id,
    )

    record = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
    finished = _wait_for_job(jobs, record.job_id)
    artifact = store.get_artifact(finished.artifact_ids[0])
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    inspection = store.inspect_artifact(artifact.artifact_id)

    assert finished.status == JobStatus.SUCCEEDED
    assert payload["operator"] == "experiment.geometry_audit"
    assert payload["latent_count"] == 2
    assert payload["report"]["dim"] == 3.0
    assert finished.metrics["latent_count"] == 2
    assert inspection.bundle_summary["kind"] == "geometry"
    assert inspection.bundle_summary["geometry"]["latent_count"] == 2


def test_runtime_prompt_search_returns_probe_bundle(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    audio_path = tmp_path / "warm_granular_loop.wav"
    sf.write(audio_path, np.zeros((800, 1), dtype=np.float32), 8000)
    source = store.import_audio_file(audio_path, prompt="warm granular loop", label="target")
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)
    jobs = JobManager(store.jobs_dir)
    recipe = Recipe(
        operator=OperatorName.EXPERIMENT_PROMPT_SEARCH,
        backend=BackendName.CPU,
        inputs={"source": source.artifact_id},
        params={
            "search_mode": "beam",
            "seed_prompt": "warm loop",
            "vocabulary": "warm, granular, cold, percussive",
            "tokens_generated": 2,
            "beam_width": 2,
            "branch_factor": 4,
            "seed": 7,
        },
    )

    record = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
    finished = _wait_for_job(jobs, record.job_id)
    artifact = store.get_artifact(finished.artifact_ids[0])
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    inspection = store.inspect_artifact(artifact.artifact_id)

    assert finished.status == JobStatus.SUCCEEDED
    assert payload["operator"] == "experiment.prompt_search"
    assert payload["scorer"]["kind"] == "lexical_probe"
    assert payload["scorer"]["model_backed"] is False
    assert "warm" in payload["prompt"]
    assert finished.metrics["scorer"] == "lexical_probe"
    assert inspection.bundle_summary["kind"] == "prompt-search"
    assert inspection.bundle_summary["prompt_search"]["prompt"] == payload["prompt"]
    assert inspection.bundle_summary["prompt_search"]["model_backed"] is False
    assert inspection.bundle_preview["prompt"] == payload["prompt"]


def test_runtime_prompt_search_uses_sa3_flow_probe_scorer(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    audio_path = tmp_path / "target.wav"
    sf.write(audio_path, np.zeros((800, 1), dtype=np.float32), 8000)
    source = store.import_audio_file(audio_path, prompt="cold texture", label="target")
    runtime = RuntimeDispatcher(store, repo_root=tmp_path)

    def fake_flow_scorer(recipe, *, context, params, target_audio_path, source, seed):
        assert recipe.backend == BackendName.TORCH_CPU
        context.set_progress(0.18, "fake SA3 flow scorer ready")
        assert params["model"] == "medium"
        assert Path(target_audio_path).name == "target.wav"
        assert source.prompt == "cold texture"
        assert seed == 11

        def batch(prompts):
            return [10.0 if "warm" in prompt else 0.0 for prompt in prompts]

        return batch, {
            "kind": "sa3_flow_probe",
            "model_backed": True,
            "model": "medium",
            "device": "test",
            "duration_seconds": 1.0,
            "score_samples": 2,
            "timestep_values": [0.25, 0.75],
            "velocity_convention": "noise_minus_data",
        }

    runtime._sa3_flow_prompt_batch_scorer = fake_flow_scorer  # type: ignore[method-assign]
    jobs = JobManager(store.jobs_dir)
    recipe = Recipe(
        operator=OperatorName.EXPERIMENT_PROMPT_SEARCH,
        backend=BackendName.TORCH_CPU,
        inputs={"source": source.artifact_id},
        params={
            "scorer": "sa3_flow_probe",
            "search_mode": "beam",
            "seed_prompt": "target",
            "vocabulary": "warm, cold",
            "tokens_generated": 1,
            "beam_width": 2,
            "branch_factor": 2,
            "model": "medium",
            "score_samples": 2,
            "timestep_values": "0.25,0.75",
            "seed": 11,
        },
    )

    record = jobs.submit(recipe, runtime.handler_for_recipe(recipe))
    finished = _wait_for_job(jobs, record.job_id)
    artifact = store.get_artifact(finished.artifact_ids[0])
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    inspection = store.inspect_artifact(artifact.artifact_id)

    assert finished.status == JobStatus.SUCCEEDED
    assert payload["scorer"]["kind"] == "sa3_flow_probe"
    assert payload["scorer"]["model_backed"] is True
    assert payload["scorer"]["model"] == "medium"
    assert "warm" in payload["prompt"]
    assert finished.metrics["scorer"] == "sa3_flow_probe"
    assert artifact.metadata["metric"] == "sa3_flow_probe"
    assert artifact.metadata["scorer"] == "sa3_flow_probe"
    assert inspection.bundle_summary["prompt_search"]["scorer"] == "sa3_flow_probe"
    assert inspection.bundle_summary["prompt_search"]["model"] == "medium"
    assert inspection.bundle_summary["prompt_search"]["score_samples"] == 2
    assert inspection.bundle_summary["prompt_search"]["families"][0]["prompt"] == payload["prompt"]


def test_sa3_model_download_progress_reports_checkpoint(monkeypatch):
    import huggingface_hub

    updates: list[tuple[float, str]] = []
    downloads: list[tuple[str, str]] = []

    class FakeContext:
        def set_progress(self, progress: float, message: str, *, phase: str | None = None) -> None:
            updates.append((progress, message))

    def fake_hf_hub_download(*, repo_id: str, filename: str, tqdm_class):
        downloads.append((repo_id, filename))
        progress = tqdm_class(total=100, file=StringIO())
        progress.update(25)
        progress.update(75)
        progress.close()
        return f"/fake-cache/{filename}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    _download_sa3_model_files_with_progress(
        "medium",
        context=FakeContext(),  # type: ignore[arg-type]
        progress_start=0.1,
        progress_end=0.2,
    )

    assert downloads == [
        ("stabilityai/stable-audio-3-medium", "model_config.json"),
        ("stabilityai/stable-audio-3-medium", "model.safetensors"),
    ]
    assert any("downloading SA3 medium checkpoint 100%" in message for _, message in updates)
    assert updates[-1][0] == pytest.approx(0.2)
    assert updates[-1][1] == "resolved SA3 medium checkpoint"


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


def test_audio_descriptor_comparison_reports_target_take_delta(tmp_path):
    store = ArtifactStore(tmp_path)
    sample_rate = 8000
    timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
    target_path = tmp_path / "low.wav"
    take_path = tmp_path / "bright.wav"
    sf.write(target_path, 0.3 * np.sin(2 * np.pi * 220 * timeline), sample_rate, subtype="FLOAT")
    sf.write(take_path, 0.3 * np.sin(2 * np.pi * 1760 * timeline), sample_rate, subtype="FLOAT")
    target = store.import_audio_file(target_path, label="target")
    take = store.import_audio_file(take_path, label="take")

    comparison = store.audio_descriptor_comparison(target.artifact_id, take.artifact_id)

    assert comparison.target_artifact_id == target.artifact_id
    assert comparison.take_artifact_id == take.artifact_id
    assert comparison.target["duration_seconds"] == pytest.approx(1.0)
    assert comparison.take["spectral_centroid_hz"] > comparison.target["spectral_centroid_hz"]
    assert comparison.delta["spectral_centroid_hz"] > 1000
    assert "rms_dbfs" in comparison.delta


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

    text_fields = set(TextGenerateRequest.model_fields) - {"source_artifact_id", "backend", "notes", "session_id"}
    audio_to_audio_fields = set(AudioToAudioRequest.model_fields) - {"source_artifact_id", "backend", "notes", "session_id"}
    inpaint_fields = set(InpaintRequest.model_fields) - {"source_artifact_id", "backend", "notes", "session_id"}
    encode_fields = set(LatentEncodeRequest.model_fields) - {"source_artifact_id", "backend", "session_id"}
    decode_fields = set(LatentDecodeRequest.model_fields) - {"source_artifact_id", "backend", "session_id"}

    assert text_fields <= set(specs[OperatorName.TEXT_TO_AUDIO].params)
    assert audio_to_audio_fields <= set(specs[OperatorName.AUDIO_TO_AUDIO].params)
    assert inpaint_fields <= set(specs[OperatorName.INPAINT].params)
    assert encode_fields <= set(specs[OperatorName.LATENT_ENCODE].params)
    assert decode_fields <= set(specs[OperatorName.LATENT_DECODE].params)
    assert specs[OperatorName.TEXT_TO_AUDIO].backends == [BackendName.MLX]
    assert OperatorName.EXPERIMENT_ALPHA_SWEEP in specs

    text_ui_fields = {field.key: field for field in specs[OperatorName.TEXT_TO_AUDIO].ui_fields}
    assert text_ui_fields["duration_seconds"].min == 0.5
    assert text_ui_fields["duration_seconds"].step == 0.5
    assert text_ui_fields["duration_seconds"].description == "Requested output duration in seconds."
    assert text_ui_fields["model"].default == "medium"
    assert [option.value for option in text_ui_fields["model"].options] == ["sm-music", "sm-sfx", "medium"]
    assert text_ui_fields["backend"].default == BackendName.MLX

    alpha_ui_fields = {field.key: field for field in specs[OperatorName.EXPERIMENT_ALPHA_SWEEP].ui_fields}
    assert alpha_ui_fields["vectors_path"].required is True
    assert alpha_ui_fields["vectors_path"].artifact_kinds == [ArtifactKind.BUNDLE]
    assert alpha_ui_fields["vectors_path"].description == "Bundle artifact containing reusable steering vectors."

    memory_ui_fields = {field.key: field for field in specs[OperatorName.MEMORY_QUERY].ui_fields}
    assert [option.value for option in memory_ui_fields["metric"].options] == ["cosine", "euclidean"]

    geometry_ui_fields = {field.key: field for field in specs[OperatorName.EXPERIMENT_GEOMETRY_AUDIT].ui_fields}
    assert geometry_ui_fields["n_components"].default == 8
    assert geometry_ui_fields["n_components"].description == "Number of principal components to keep in the geometry audit."

    prompt_ui_fields = {field.key: field for field in specs[OperatorName.EXPERIMENT_PROMPT_SEARCH].ui_fields}
    assert [option.value for option in prompt_ui_fields["search_mode"].options] == ["beam", "greedy", "coordinate"]
    assert [option.value for option in prompt_ui_fields["scorer"].options] == ["lexical_probe", "sa3_flow_probe", "clap"]
    assert prompt_ui_fields["target_audio_path"].artifact_kinds == [ArtifactKind.AUDIO]
    assert prompt_ui_fields["tokens_generated"].default == 4
    assert prompt_ui_fields["score_samples"].default == 1
    assert specs[OperatorName.EXPERIMENT_PROMPT_SEARCH].backends == [BackendName.CPU, BackendName.TORCH_MPS, BackendName.TORCH_CPU]

    profile_generate_fields = {field.key: field for field in specs[OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE].ui_fields}
    assert {"duration_seconds", "steps", "cfg_scale", "seed", "save_original", "device", "no_half"} <= set(profile_generate_fields)
    assert profile_generate_fields["model"].default == "medium"
    assert profile_generate_fields["duration_seconds"].min == 0.5

    audio_vector_fields = {field.key: field for field in specs[OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS].ui_fields}
    assert {"name", "chunked", "normalize_frame", "device"} <= set(audio_vector_fields)
    assert audio_vector_fields["normalize_frame"].advanced is True

    lora_fields = {field.key: field for field in specs[OperatorName.TRAIN_LORA].ui_fields}
    assert {"model", "base_precision", "svd_bases_path", "lora_checkpoint", "checkpoint_every", "log_every", "demo_every", "num_workers"} <= set(lora_fields)
    assert [option.value for option in lora_fields["adapter_type"].options] == [
        "lora",
        "dora",
        "dora-rows",
        "dora-cols",
        "bora",
        "lora-xs",
        "dora-rows-xs",
        "dora-cols-xs",
        "bora-xs",
    ]


def test_operator_specs_emit_ui_fields_for_all_runtime_params(tmp_path):
    store = ArtifactStore(tmp_path / "lab")
    specs = RuntimeDispatcher(store, repo_root=tmp_path).operator_specs()

    for spec in specs:
        ui_keys = {field.key for field in spec.ui_fields}
        assert set(spec.params) <= ui_keys, f"{spec.name} missing UI fields for {sorted(set(spec.params) - ui_keys)}"
        assert "backend" in ui_keys


def test_colab_mode_map_covers_notebook_sections():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "sa3_same_native_experimental_modes.ipynb"
    notebook = json.loads(notebook_path.read_text())
    markdown = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "markdown")

    notebook_mode_ids = set(re.findall(r"^## Mode ([0-9]+[a-z]?)(?:\.|\s)", markdown, flags=re.MULTILINE))
    if "## Mode 0c Annotation Retrieval" in markdown:
        notebook_mode_ids.add("0c-search")
    if "## Flow Sign Diagnostic" in markdown:
        notebook_mode_ids.add("flow-sign")
    if "## Combined Experiment:" in markdown:
        notebook_mode_ids.add("combined")

    app_mode_ids = {mode.mode_id for mode in list_colab_modes()}

    assert notebook_mode_ids <= app_mode_ids


def test_colab_mode_operators_reference_known_app_surfaces(tmp_path):
    specs = {spec.name for spec in RuntimeDispatcher(ArtifactStore(tmp_path / "lab"), repo_root=tmp_path).operator_specs()}
    non_runtime_operators = {OperatorName.ANNOTATE}

    for mode in list_colab_modes():
        for operator in mode.operators:
            assert operator in specs or operator in non_runtime_operators, f"{mode.mode_id} references unmapped operator {operator}"


def test_colab_mode_runtime_specs_are_interface_complete(tmp_path):
    specs = {spec.name: spec for spec in RuntimeDispatcher(ArtifactStore(tmp_path / "lab"), repo_root=tmp_path).operator_specs()}
    non_runtime_operators = {OperatorName.ANNOTATE}

    for mode in list_colab_modes():
        if "native" in mode.status or "recipe" in mode.status or "chainable" in mode.status:
            assert mode.inputs, f"{mode.mode_id} must describe inputs"
            assert mode.outputs, f"{mode.mode_id} must describe outputs"
            assert mode.operators, f"{mode.mode_id} must list executable operators"
        for operator in mode.operators:
            if operator in non_runtime_operators:
                continue
            spec = specs[operator]
            ui_keys = {field.key for field in spec.ui_fields}
            missing = set(spec.params) - ui_keys
            assert not missing, f"{mode.mode_id} / {operator} missing UI fields for {sorted(missing)}"
            assert spec.backends, f"{mode.mode_id} / {operator} must declare backends"
            assert spec.produces, f"{mode.mode_id} / {operator} must declare artifact kinds"


def test_sessions_persist_and_attach_artifacts(tmp_path):
    store = ArtifactStore(tmp_path)
    session = store.create_session(SessionCreateRequest(name="sketch one"))
    record = store.store_latent_array(np.ones((3, 2), dtype=np.float32), latent_rate=1.0, session_id=session.session_id)
    archived = store.update_session(session.session_id, SessionUpdateRequest(status=SessionStatus.ARCHIVED))

    assert store.get_session(session.session_id).name == "sketch one"
    assert archived.status == SessionStatus.ARCHIVED
    assert archived.archived_at is not None
    assert store.get_artifact(record.artifact_id).session_id == session.session_id
    assert store.list_artifacts(session_id=session.session_id)[0].artifact_id == record.artifact_id


def test_bundle_artifact_zips_directory_outputs(tmp_path):
    store = ArtifactStore(tmp_path)
    output_dir = tmp_path / "vectors"
    output_dir.mkdir()
    (output_dir / "summary.txt").write_text("best_layer=2\n", encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps({"vectors": {"layers": [2, 4], "best_layer": 2, "probe_accuracy": 0.75}, "examples": [{"label": 1}]}),
        encoding="utf-8",
    )
    np.savez(output_dir / "direction.npz", kind="LatentStyleDirection", name="bright", dim=64, mean_delta=np.zeros(64, dtype=np.float32))
    recipe = Recipe(operator=OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT, backend=BackendName.TORCH_CPU)

    record = store.finalize_bundle_path(artifact_id="art_bundle", path=output_dir, recipe=recipe)

    assert record.kind == "bundle"
    assert record.file is not None
    assert record.file.media_type == "application/zip"
    assert record.path.suffix == ".zip"
    assert record.path.exists()
    assert store.get_recipe(recipe.recipe_id).operator == OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT
    inspection = store.inspect_artifact(record.artifact_id)
    assert inspection.recipe is not None
    assert inspection.recipe.recipe_id == recipe.recipe_id
    assert [item.path for item in inspection.bundle_files] == ["direction.npz", "metadata.json", "summary.txt"]
    assert inspection.bundle_summary["kind"] == "vectors"
    assert inspection.bundle_summary["vectors"]["best_layer"] == 2
    assert inspection.bundle_summary["vectors"]["example_count"] == 1
    assert inspection.bundle_summary["npz_files"][0]["scalars"]["kind"] == "LatentStyleDirection"


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
    events = manager.event_history(record.job_id)
    assert [event.job.status for event in events] == [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.SUCCEEDED]
    assert [event.job.phase for event in events] == ["queued", "preflight", "done"]
    assert [event.sequence for event in events] == [1, 2, 3]
    assert (tmp_path / "events" / f"{record.job_id}.jsonl").exists()


def test_job_manager_cancels_running_job(tmp_path):
    manager = JobManager(tmp_path)
    started = Event()
    recipe = Recipe(
        operator=OperatorName.LATENT_CYCLIC_ROLL,
        backend=BackendName.TORCH_CPU,
        params={"shift_frames": 1},
    )

    def handler(context):
        started.set()
        while not context.cancelled():
            time.sleep(0.01)
        return JobResult(artifact_ids=["should_not_commit"])

    record = manager.submit(recipe, handler)
    assert started.wait(timeout=1)

    cancelled = manager.cancel(record.job_id)

    assert cancelled.status == JobStatus.CANCELLED
    assert _wait_for_job(manager, record.job_id).artifact_ids == []


def test_job_manager_caps_logs_to_recent_lines(tmp_path):
    manager = JobManager(tmp_path)
    recipe = Recipe(
        operator=OperatorName.LATENT_CYCLIC_ROLL,
        backend=BackendName.TORCH_CPU,
        params={"shift_frames": 1},
    )
    record = manager.submit(recipe, lambda context: JobResult())
    for index in range(250):
        manager.append_log(record.job_id, f"line {index}")

    updated = manager.get(record.job_id)

    assert len(updated.logs) == 200
    assert updated.logs[0] == "line 50"
    assert updated.logs[-1] == "line 249"


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

    session = client.post("/sessions", json={"name": "api sketch"})
    assert session.status_code == 200
    session_id = session.json()["session_id"]
    assert client.get("/sessions").json()[0]["session_id"] == session_id

    health = client.get("/health")
    assert health.status_code == 200
    assert any(item["backend"] == "mlx" for item in health.json()["backends"])
    readiness = client.get("/readiness")
    assert readiness.status_code == 200
    readiness_names = {item["name"] for item in readiness.json()["checks"]}
    assert {"artifact-root", "hf-auth", "mlx-medium-weights", "same-l-access"} <= readiness_names

    audio_bytes = BytesIO()
    sf.write(audio_bytes, np.linspace(-1.0, 1.0, 16, dtype=np.float32), 16000, format="WAV", subtype="FLOAT")
    audio_bytes.seek(0)
    audio = client.post(
        "/audio/import",
        files={"file": ("tone.wav", audio_bytes, "application/octet-stream")},
        data={"session_id": session_id},
    )
    assert audio.status_code == 200
    assert audio.json()["session_id"] == session_id
    assert audio.json()["file"]["media_type"] == "audio/wav"
    annotated = client.post(
        f"/artifacts/{audio.json()['artifact_id']}/annotate",
        json={"label": "keeper tone", "notes": "wide pulse", "tags": ["favorite", "loop"]},
    )
    assert annotated.status_code == 200
    filtered = client.get("/artifacts?q=pulse&tags=favorite,loop")
    assert filtered.status_code == 200
    assert [item["artifact_id"] for item in filtered.json()] == [audio.json()["artifact_id"]]
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
            "session_id": session_id,
        },
    )
    assert job.status_code == 200
    finished = _wait_for_api_job(client, job.json()["job_id"])
    assert finished["status"] == "succeeded"
    assert finished["recipe"]["session_id"] == session_id
    assert len(finished["artifact_ids"]) == 1
    history = client.get(f"/jobs/{finished['job_id']}/events/history")
    assert history.status_code == 200
    assert history.json()[-1]["job"]["status"] == "succeeded"
    assert history.json()[-1]["sequence"] >= 1
    recipe = client.get(f"/recipes/{finished['recipe']['recipe_id']}")
    assert recipe.status_code == 200
    assert recipe.json()["operator"] == "latent.cyclic_roll"

    replay = client.post(f"/recipes/{finished['recipe']['recipe_id']}/replay")
    assert replay.status_code == 200
    replay_finished = _wait_for_api_job(client, replay.json()["job_id"])
    assert replay_finished["status"] == "succeeded"
    assert replay_finished["recipe"]["recipe_id"] != finished["recipe"]["recipe_id"]

    fork = client.post(f"/recipes/{finished['recipe']['recipe_id']}/fork", json={"params": {"shift_frames": 2}})
    assert fork.status_code == 200
    fork_finished = _wait_for_api_job(client, fork.json()["job_id"])
    assert fork_finished["status"] == "succeeded"
    assert fork_finished["recipe"]["params"]["shift_frames"] == 2

    retry = client.post(f"/jobs/{finished['job_id']}/retry")
    assert retry.status_code == 200
    retry_finished = _wait_for_api_job(client, retry.json()["job_id"])
    assert retry_finished["status"] == "succeeded"

    with client.websocket_connect(f"/jobs/{retry_finished['job_id']}/events") as websocket:
        event = websocket.receive_json()
    assert event["type"] == "snapshot"
    assert event["job"]["job_id"] == retry_finished["job_id"]


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


def test_fastapi_allows_local_dev_port_origins(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sa3_native_lab.app.server import create_app

    client = TestClient(create_app(artifact_root=tmp_path / "lab", repo_root=tmp_path))
    response = client.get("/health", headers={"Origin": "http://127.0.0.1:5174"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5174"


def test_fastapi_compares_audio_descriptors(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sa3_native_lab.app.server import create_app

    app = create_app(artifact_root=tmp_path / "lab", repo_root=tmp_path)
    client = TestClient(app)
    sample_rate = 8000
    timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
    low_path = tmp_path / "low.wav"
    bright_path = tmp_path / "bright.wav"
    sf.write(low_path, 0.3 * np.sin(2 * np.pi * 220 * timeline), sample_rate, subtype="FLOAT")
    sf.write(bright_path, 0.3 * np.sin(2 * np.pi * 1760 * timeline), sample_rate, subtype="FLOAT")
    target = app.state.store.import_audio_file(low_path)
    take = app.state.store.import_audio_file(bright_path)

    response = client.get(f"/artifacts/{target.artifact_id}/descriptor-comparison/{take.artifact_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_artifact_id"] == target.artifact_id
    assert payload["take_artifact_id"] == take.artifact_id
    assert payload["delta"]["spectral_centroid_hz"] > 1000
    assert payload["target"]["duration_seconds"] == pytest.approx(1.0)


def test_fastapi_inspects_bundle_artifact(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sa3_native_lab.app.server import create_app

    app = create_app(artifact_root=tmp_path / "lab", repo_root=tmp_path)
    client = TestClient(app)
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text('{"score": 0.8}\n', encoding="utf-8")
    (output_dir / "plot.png").write_bytes(b"plot")
    sf.write(output_dir / "take.wav", np.zeros((800, 1), dtype=np.float32), 8000)
    recipe = Recipe(operator=OperatorName.EXPERIMENT_ALPHA_SWEEP, backend=BackendName.TORCH_CPU)
    record = app.state.store.finalize_bundle_path(artifact_id="art_bundle", path=output_dir, recipe=recipe)

    response = client.get(f"/artifacts/{record.artifact_id}/inspect")

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact"]["artifact_id"] == record.artifact_id
    assert payload["recipe"]["operator"] == "experiment.alpha_sweep"
    assert payload["bundle_files"][0]["path"] == "metrics.json"
    assert "bundle_preview" in payload
    assert payload["bundle_summary"]["kind"] == "sweep"
    assert payload["bundle_summary"]["json_files"][0]["path"] == "metrics.json"
    assert payload["bundle_summary"]["metrics"]["values"]["score"] == 0.8
    assert payload["bundle_summary"]["plots"]["count"] == 1
    assert payload["bundle_audio_files"][0]["path"] == "take.wav"
    assert payload["bundle_audio_files"][0]["media_type"] == "audio/wav"
    assert payload["bundle_audio_files"][0]["sample_rate"] == 8000
    assert payload["bundle_audio_files"][0]["duration_seconds"] == 0.1

    plot = client.get(f"/artifacts/{record.artifact_id}/bundle-file", params={"path": "plot.png"})
    assert plot.status_code == 200
    assert plot.content == b"plot"
    assert plot.headers["content-type"] == "image/png"

    audio = client.get(f"/artifacts/{record.artifact_id}/bundle-file", params={"path": "take.wav"})
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/wav"

    promoted = client.post(
        f"/artifacts/{record.artifact_id}/bundle-audio/promote",
        json={"path": "take.wav", "label": "keeper take", "session_id": "sess_bundle"},
    )
    assert promoted.status_code == 200
    promoted_payload = promoted.json()
    assert promoted_payload["kind"] == "audio"
    assert promoted_payload["label"] == "keeper take"
    assert promoted_payload["session_id"] == "sess_bundle"
    assert promoted_payload["source_artifact_ids"] == [record.artifact_id]
    assert promoted_payload["metadata"]["operator"] == "artifact.promote_bundle_audio"
    assert promoted_payload["metadata"]["bundle_audio_path"] == "take.wav"
    assert promoted_payload["audio"]["sample_rate"] == 8000

    promoted_inspection = client.get(f"/artifacts/{record.artifact_id}/inspect").json()
    assert promoted_inspection["children"][0]["artifact_id"] == promoted_payload["artifact_id"]
    promoted_recipe = app.state.store.get_recipe(promoted_payload["recipe_id"])
    assert promoted_recipe.operator == OperatorName.ARTIFACT_PROMOTE_BUNDLE_AUDIO
    peaks = client.get(f"/artifacts/{promoted_payload['artifact_id']}/peaks", params={"bins": 8})
    assert peaks.status_code == 200
    assert peaks.json()["bins"] == 8

    rejected = client.post(
        f"/artifacts/{record.artifact_id}/bundle-audio/promote",
        json={"path": "plot.png"},
    )
    assert rejected.status_code == 422


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
