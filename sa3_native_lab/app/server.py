from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from sa3_native_lab import __version__

from .contracts import (
    ArtifactAnnotationRequest,
    ArtifactInspection,
    ArtifactKind,
    ArtifactRecord,
    AudioPeaksResponse,
    AudioToAudioRequest,
    ExperimentRunRequest,
    HealthResponse,
    InpaintRequest,
    JobErrorEvent,
    JobEvent,
    JobJournalEvent,
    JobStatus,
    JobRecord,
    LatentDecodeRequest,
    LatentEncodeRequest,
    NotebookMode,
    OperatorName,
    OperatorRunRequest,
    Recipe,
    RecipeForkRequest,
    ReadinessCheck,
    ReadinessResponse,
    SessionCreateRequest,
    SessionRecord,
    SessionUpdateRequest,
    TextGenerateRequest,
)
from .colab_modes import list_colab_modes
from .jobs import JobManager
from .runtime import RuntimeDispatcher
from .storage import ArtifactStore


def create_app(
    *,
    artifact_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> FastAPI:
    store = ArtifactStore(artifact_root)
    jobs = JobManager(store.jobs_dir)
    runtime = RuntimeDispatcher(store, repo_root=repo_root)

    app = FastAPI(title="SA3 Native Lab API", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        ],
        allow_origin_regex=r"https?://(127\.0\.0\.1|localhost):\d+",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.store = store
    app.state.jobs = jobs
    app.state.runtime = runtime

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            version=__version__,
            artifact_root=store.root,
            backends=runtime.backend_statuses(),
        )

    @app.get("/readiness", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse:
        return _readiness_response(store, runtime)

    @app.get("/models/status")
    def model_statuses():
        return runtime.backend_statuses()

    @app.get("/operators/specs")
    def operator_specs():
        return runtime.operator_specs()

    @app.get("/colab/modes", response_model=list[NotebookMode])
    def colab_modes() -> list[NotebookMode]:
        return list_colab_modes()

    @app.get("/sessions", response_model=list[SessionRecord])
    def list_sessions() -> list[SessionRecord]:
        return store.list_sessions()

    @app.post("/sessions", response_model=SessionRecord)
    def create_session(request: SessionCreateRequest) -> SessionRecord:
        return store.create_session(request)

    @app.get("/sessions/{session_id}", response_model=SessionRecord)
    def get_session(session_id: str) -> SessionRecord:
        try:
            return store.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc

    @app.patch("/sessions/{session_id}", response_model=SessionRecord)
    def update_session(session_id: str, request: SessionUpdateRequest) -> SessionRecord:
        try:
            return store.update_session(session_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc

    @app.get("/artifacts", response_model=list[ArtifactRecord])
    def list_artifacts(
        kind: ArtifactKind | None = None,
        session_id: str | None = None,
        q: str | None = None,
        tags: str | None = None,
    ) -> list[ArtifactRecord]:
        return store.list_artifacts(kind=kind, session_id=session_id, query=q, tags=_split_query_tags(tags))

    @app.get("/artifacts/{artifact_id}", response_model=ArtifactRecord)
    def get_artifact(artifact_id: str) -> ArtifactRecord:
        try:
            return store.get_artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"artifact not found: {artifact_id}") from exc

    @app.get("/artifacts/{artifact_id}/inspect", response_model=ArtifactInspection)
    def inspect_artifact(artifact_id: str) -> ArtifactInspection:
        try:
            return store.inspect_artifact(artifact_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"artifact not found: {artifact_id}") from exc

    @app.get("/artifacts/{artifact_id}/file")
    def get_artifact_file(artifact_id: str):
        record = get_artifact(artifact_id)
        if not record.path.exists():
            raise HTTPException(status_code=404, detail=f"artifact file missing: {artifact_id}")
        media_type = record.file.media_type if record.file else None
        return FileResponse(record.path, media_type=media_type, filename=record.path.name)

    @app.get("/artifacts/{artifact_id}/bundle-file")
    def get_bundle_file(artifact_id: str, path: str):
        try:
            content, media_type, filename = store.bundle_file(artifact_id, path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"bundle file not found: {path}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        headers = {"Content-Disposition": f'inline; filename="{filename}"'}
        return Response(content=content, media_type=media_type or "application/octet-stream", headers=headers)

    @app.get("/artifacts/{artifact_id}/peaks", response_model=AudioPeaksResponse)
    def get_artifact_peaks(artifact_id: str, bins: int = 96) -> AudioPeaksResponse:
        try:
            return store.audio_peaks(artifact_id, bins=bins)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"artifact not found: {artifact_id}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/recipes", response_model=list[Recipe])
    def list_recipes() -> list[Recipe]:
        return store.list_recipes()

    @app.get("/recipes/{recipe_id}", response_model=Recipe)
    def get_recipe(recipe_id: str) -> Recipe:
        try:
            return store.get_recipe(recipe_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"recipe not found: {recipe_id}") from exc

    @app.post("/recipes/{recipe_id}/replay", response_model=JobRecord)
    def replay_recipe(recipe_id: str) -> JobRecord:
        try:
            recipe = store.get_recipe(recipe_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"recipe not found: {recipe_id}") from exc
        return _submit_recipe(jobs, runtime, _fork_recipe(recipe))

    @app.post("/recipes/{recipe_id}/fork", response_model=JobRecord)
    def fork_recipe(recipe_id: str, request: RecipeForkRequest) -> JobRecord:
        try:
            recipe = store.get_recipe(recipe_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"recipe not found: {recipe_id}") from exc
        return _submit_recipe(jobs, runtime, _fork_recipe(recipe, request))

    @app.post("/artifacts/{artifact_id}/annotate", response_model=ArtifactRecord)
    def annotate_artifact(artifact_id: str, request: ArtifactAnnotationRequest) -> ArtifactRecord:
        try:
            return store.annotate_artifact(artifact_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"artifact not found: {artifact_id}") from exc

    @app.post("/audio/import", response_model=ArtifactRecord)
    async def import_audio(
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        label: str | None = Form(None),
        session_id: str | None = Form(None),
    ) -> ArtifactRecord:
        return store.import_audio_stream(
            file.file,
            filename=file.filename or "audio.wav",
            media_type=file.content_type,
            prompt=prompt,
            label=label,
            session_id=session_id,
        )

    @app.post("/latents/import", response_model=ArtifactRecord)
    async def import_latent(
        file: UploadFile = File(...),
        latent_rate: float = Form(...),
        channel_first: bool = Form(False),
        sample_rate: int | None = Form(None),
        prompt: str | None = Form(None),
        label: str | None = Form(None),
        session_id: str | None = Form(None),
    ) -> ArtifactRecord:
        return store.import_latent_stream(
            file.file,
            filename=file.filename or "latent.npy",
            latent_rate=latent_rate,
            channel_first=channel_first,
            sample_rate=sample_rate,
            prompt=prompt,
            label=label,
            session_id=session_id,
        )

    @app.post("/latents/encode", response_model=JobRecord)
    def encode_latent(request: LatentEncodeRequest) -> JobRecord:
        recipe = Recipe(
            operator=OperatorName.LATENT_ENCODE,
            backend=request.backend,
            inputs={"source": request.source_artifact_id},
            params=request.model_dump(mode="json", exclude={"source_artifact_id", "session_id"}),
            model=request.model,
            notes=request.notes,
            session_id=request.session_id,
        )
        return _submit_recipe(jobs, runtime, recipe)

    @app.post("/latents/decode", response_model=JobRecord)
    def decode_latent(request: LatentDecodeRequest) -> JobRecord:
        recipe = Recipe(
            operator=OperatorName.LATENT_DECODE,
            backend=request.backend,
            inputs={"source": request.source_artifact_id},
            params=request.model_dump(mode="json", exclude={"source_artifact_id", "session_id"}),
            model=request.model,
            notes=request.notes,
            session_id=request.session_id,
        )
        return _submit_recipe(jobs, runtime, recipe)

    @app.post("/generate/text", response_model=JobRecord)
    def generate_text(request: TextGenerateRequest) -> JobRecord:
        recipe = Recipe(
            operator=OperatorName.TEXT_TO_AUDIO,
            backend=request.backend,
            params=request.model_dump(mode="json", exclude={"session_id"}),
            model=request.model,
            seed=request.seed,
            session_id=request.session_id,
        )
        return _submit_recipe(jobs, runtime, recipe)

    @app.post("/generate/audio-to-audio", response_model=JobRecord)
    def generate_audio_to_audio(request: AudioToAudioRequest) -> JobRecord:
        recipe = Recipe(
            operator=OperatorName.AUDIO_TO_AUDIO,
            backend=request.backend,
            inputs={"source": request.source_artifact_id},
            params=request.model_dump(mode="json", exclude={"source_artifact_id", "session_id"}),
            model=request.model,
            seed=request.seed,
            session_id=request.session_id,
        )
        return _submit_recipe(jobs, runtime, recipe)

    @app.post("/generate/inpaint", response_model=JobRecord)
    def generate_inpaint(request: InpaintRequest) -> JobRecord:
        recipe = Recipe(
            operator=OperatorName.INPAINT,
            backend=request.backend,
            inputs={"source": request.source_artifact_id},
            params=request.model_dump(mode="json", exclude={"source_artifact_id", "session_id"}),
            model=request.model,
            seed=request.seed,
            session_id=request.session_id,
        )
        return _submit_recipe(jobs, runtime, recipe)

    @app.post("/operators/run", response_model=JobRecord)
    def run_operator(request: OperatorRunRequest) -> JobRecord:
        recipe = Recipe(
            operator=request.operator,
            backend=request.backend,
            inputs=request.inputs,
            params=request.params,
            seed=request.seed,
            notes=request.notes,
            session_id=request.session_id,
        )
        return _submit_recipe(jobs, runtime, recipe)

    @app.post("/experiments/run", response_model=JobRecord)
    def run_experiment(request: ExperimentRunRequest) -> JobRecord:
        recipe = Recipe(
            operator=request.operator,
            backend=request.backend,
            inputs=request.inputs,
            params=request.params,
            model=request.model,
            seed=request.seed,
            notes=request.notes,
            session_id=request.session_id,
        )
        return _submit_recipe(jobs, runtime, recipe)

    @app.get("/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return jobs.list()

    @app.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        try:
            return jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}") from exc

    @app.get("/jobs/{job_id}/events/history", response_model=list[JobJournalEvent])
    def job_event_history(job_id: str, after: int = 0, limit: int = 100) -> list[JobJournalEvent]:
        try:
            return jobs.event_history(job_id, after_sequence=after, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}") from exc

    @app.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        try:
            return jobs.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}") from exc

    @app.post("/jobs/{job_id}/retry", response_model=JobRecord)
    def retry_job(job_id: str) -> JobRecord:
        try:
            record = jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}") from exc
        return _submit_recipe(jobs, runtime, _fork_recipe(record.recipe))

    @app.websocket("/jobs/{job_id}/events")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        previous_payload = None
        while True:
            try:
                record = jobs.get(job_id)
            except KeyError:
                await websocket.send_json(JobErrorEvent(error=f"job not found: {job_id}").model_dump(mode="json"))
                await websocket.close(code=1008)
                return
            payload = JobEvent(job=record).model_dump(mode="json")
            if payload != previous_payload:
                await websocket.send_json(payload)
                previous_payload = payload
            if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
                await websocket.close()
                return
            await asyncio.sleep(0.5)

    return app


def _split_query_tags(tags: str | None) -> list[str] | None:
    if not tags:
        return None
    parsed = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return parsed or None


def _submit_recipe(jobs: JobManager, runtime: RuntimeDispatcher, recipe: Recipe) -> JobRecord:
    try:
        handler = runtime.handler_for_recipe(recipe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return jobs.submit(recipe, handler)


def _fork_recipe(recipe: Recipe, request: RecipeForkRequest | None = None) -> Recipe:
    inputs = dict(recipe.inputs)
    params = dict(recipe.params)
    backend = recipe.backend
    model = recipe.model
    seed = recipe.seed
    notes = recipe.notes
    session_id = recipe.session_id
    if request is not None:
        if request.inputs:
            inputs.update(request.inputs)
        if request.params:
            params.update(request.params)
        backend = request.backend or backend
        model = request.model if request.model is not None else model
        seed = request.seed if request.seed is not None else seed
        notes = request.notes if request.notes is not None else notes
        session_id = request.session_id if request.session_id is not None else session_id
    return Recipe(
        operator=recipe.operator,
        backend=backend,
        inputs=inputs,
        params=params,
        model=model,
        seed=seed,
        notes=notes,
        session_id=session_id,
        version=recipe.version,
    )


def _readiness_response(store: ArtifactStore, runtime: RuntimeDispatcher) -> ReadinessResponse:
    checks = [
        _artifact_root_check(store),
        *_backend_readiness_checks(runtime),
        _huggingface_readiness_check(),
        _mlx_medium_weight_check(runtime.repo_root),
        _same_l_access_check(),
    ]
    errors = sum(1 for check in checks if check.status == "error")
    warnings = sum(1 for check in checks if check.status == "warn")
    return ReadinessResponse(checks=checks, ok=errors == 0, warnings=warnings, errors=errors)


def _artifact_root_check(store: ArtifactStore) -> ReadinessCheck:
    try:
        store.root.mkdir(parents=True, exist_ok=True)
        probe = store.root / ".readiness-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        return ReadinessCheck(
            name="artifact-root",
            status="error",
            message=f"artifact root is not writable: {store.root}",
            detail=str(exc),
        )
    return ReadinessCheck(name="artifact-root", status="ok", message=f"artifact root is writable: {store.root}")


def _backend_readiness_checks(runtime: RuntimeDispatcher) -> list[ReadinessCheck]:
    checks = []
    for status in runtime.backend_statuses():
        checks.append(
            ReadinessCheck(
                name=f"backend:{status.backend.value}",
                status="ok" if status.available else "warn",
                message=status.message or f"{status.backend.value} backend checked",
                detail=str(status.details) if status.details else None,
            )
        )
    return checks


def _huggingface_readiness_check() -> ReadinessCheck:
    try:
        from .dev import huggingface_auth_check

        check = huggingface_auth_check()
        return ReadinessCheck(name=check.name, status=check.status, message=check.message, detail=check.detail)
    except Exception as exc:
        return ReadinessCheck(name="hf-auth", status="warn", message="Hugging Face auth could not be checked", detail=str(exc))


def _mlx_medium_weight_check(repo_root: Path) -> ReadinessCheck:
    required = [
        repo_root / "optimized" / "mlx" / "models" / "mlx" / "dit_medium_f16.npz",
        repo_root / "optimized" / "mlx" / "models" / "mlx" / "same_l_decoder_f32.npz",
        repo_root / "optimized" / "mlx" / "models" / "mlx" / "same_l_encoder_f32.npz",
        repo_root / "optimized" / "mlx" / "models" / "mlx" / "t5gemma_f16.npz",
    ]
    present = [path for path in required if path.exists() or path.is_symlink()]
    missing = [path.name for path in required if path not in present]
    if len(present) == len(required):
        return ReadinessCheck(name="mlx-medium-weights", status="ok", message="Medium MLX weights are present")
    return ReadinessCheck(
        name="mlx-medium-weights",
        status="warn",
        message=f"Medium MLX weights are partially local ({len(present)}/{len(required)})",
        detail=", ".join(missing) or None,
    )


def _same_l_access_check() -> ReadinessCheck:
    hf = _huggingface_readiness_check()
    if hf.status == "ok":
        return ReadinessCheck(name="same-l-access", status="ok", message="SAME-L downloads can use Hugging Face auth")
    return ReadinessCheck(
        name="same-l-access",
        status="warn",
        message="SAME-L may need Hugging Face auth on first load",
        detail=hf.detail,
    )
