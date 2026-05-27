from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from sa3_native_lab import __version__

from .contracts import (
    ArtifactAnnotationRequest,
    ArtifactKind,
    ArtifactRecord,
    AudioPeaksResponse,
    AudioToAudioRequest,
    ExperimentRunRequest,
    HealthResponse,
    InpaintRequest,
    JobStatus,
    JobRecord,
    LatentDecodeRequest,
    LatentEncodeRequest,
    NotebookMode,
    OperatorName,
    OperatorRunRequest,
    Recipe,
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

    @app.get("/artifacts/{artifact_id}/file")
    def get_artifact_file(artifact_id: str):
        record = get_artifact(artifact_id)
        if not record.path.exists():
            raise HTTPException(status_code=404, detail=f"artifact file missing: {artifact_id}")
        media_type = record.file.media_type if record.file else None
        return FileResponse(record.path, media_type=media_type, filename=record.path.name)

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

    @app.websocket("/jobs/{job_id}/events")
    async def job_events(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        previous_payload = None
        while True:
            try:
                record = jobs.get(job_id)
            except KeyError:
                await websocket.send_json({"error": f"job not found: {job_id}"})
                await websocket.close(code=1008)
                return
            payload = record.model_dump(mode="json")
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
