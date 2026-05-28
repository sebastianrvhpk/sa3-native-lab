from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from .contracts import (
    ArtifactAnnotationRequest,
    ArtifactInspection,
    ArtifactKind,
    ArtifactRecord,
    AudioMetadata,
    AudioPeaksResponse,
    BundleFileEntry,
    FileInfo,
    LatentMetadata,
    Recipe,
    SessionCreateRequest,
    SessionRecord,
    SessionStatus,
    SessionUpdateRequest,
)
from .ids import new_id, utc_now


class ArtifactStore:
    def __init__(self, root: str | Path | None = None) -> None:
        env_root = os.environ.get("SA3_LAB_HOME")
        self.root = Path(root) if root is not None else Path(env_root) if env_root else Path.cwd() / ".sa3_lab"
        self.artifacts_dir = self.root / "artifacts"
        self.jobs_dir = self.root / "jobs"
        self.recipes_dir = self.root / "recipes"
        self.sessions_dir = self.root / "sessions"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def artifact_dir(self, artifact_id: str) -> Path:
        return self.artifacts_dir / artifact_id

    def reserve_artifact_path(self, *, artifact_id: str | None = None, filename: str) -> tuple[str, Path]:
        artifact_id = artifact_id or new_id("art")
        directory = self.artifact_dir(artifact_id)
        directory.mkdir(parents=True, exist_ok=True)
        return artifact_id, directory / _safe_filename(filename)

    def import_audio_stream(
        self,
        stream: BinaryIO,
        *,
        filename: str,
        media_type: str | None = None,
        prompt: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        artifact_id, path = self.reserve_artifact_path(filename=filename)
        with path.open("wb") as output:
            shutil.copyfileobj(stream, output)
        return self.finalize_audio_file(
            artifact_id=artifact_id,
            path=path,
            media_type=media_type,
            prompt=prompt,
            label=label,
            metadata=metadata,
            session_id=session_id,
        )

    def import_audio_file(
        self,
        source_path: str | Path,
        *,
        media_type: str | None = None,
        prompt: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        source = Path(source_path)
        artifact_id, path = self.reserve_artifact_path(filename=source.name)
        shutil.copy2(source, path)
        return self.finalize_audio_file(
            artifact_id=artifact_id,
            path=path,
            media_type=media_type,
            prompt=prompt,
            label=label,
            metadata=metadata,
            session_id=session_id,
        )

    def finalize_audio_file(
        self,
        *,
        artifact_id: str,
        path: str | Path,
        media_type: str | None = None,
        recipe: Recipe | None = None,
        source_artifact_ids: list[str] | None = None,
        prompt: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        resolved_session_id = session_id or (recipe.session_id if recipe else None)
        path = Path(path)
        audio = _audio_metadata(path)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind=ArtifactKind.AUDIO,
            path=path,
            file=_file_info(path, media_type=_audio_media_type(path, media_type)),
            audio=audio,
            source_artifact_ids=source_artifact_ids or [],
            recipe_id=recipe.recipe_id if recipe else None,
            prompt=prompt,
            label=label,
            metadata=metadata or {},
            session_id=resolved_session_id,
        )
        self.write_artifact(record)
        if recipe is not None:
            self.write_recipe(recipe)
        return record

    def finalize_bundle_path(
        self,
        *,
        artifact_id: str,
        path: str | Path,
        recipe: Recipe | None = None,
        source_artifact_ids: list[str] | None = None,
        label: str | None = None,
        prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        resolved_session_id = session_id or (recipe.session_id if recipe else None)
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        bundle_metadata = dict(metadata or {})
        if source.is_dir():
            bundle_path = self.artifact_dir(artifact_id) / f"{_safe_filename(source.name)}.zip"
            if bundle_path.exists():
                bundle_path.unlink()
            shutil.make_archive(str(bundle_path.with_suffix("")), "zip", source)
            bundle_metadata["bundle_source_path"] = str(source)
            path_for_record = bundle_path
            media_type = "application/zip"
        else:
            path_for_record = source
            media_type = _guess_media_type(source) or "application/octet-stream"

        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind=ArtifactKind.BUNDLE,
            path=path_for_record,
            file=_file_info(path_for_record, media_type=media_type),
            source_artifact_ids=source_artifact_ids or [],
            recipe_id=recipe.recipe_id if recipe else None,
            prompt=prompt,
            label=label,
            metadata=bundle_metadata,
            session_id=resolved_session_id,
        )
        self.write_artifact(record)
        if recipe is not None:
            self.write_recipe(recipe)
        return record

    def import_latent_stream(
        self,
        stream: BinaryIO,
        *,
        filename: str,
        latent_rate: float,
        channel_first: bool = False,
        sample_rate: int | None = None,
        prompt: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        artifact_id, upload_path = self.reserve_artifact_path(filename=filename)
        with upload_path.open("wb") as output:
            shutil.copyfileobj(stream, output)
        array = np.load(upload_path, allow_pickle=False)
        return self.store_latent_array(
            array,
            artifact_id=artifact_id,
            filename="latent.npy",
            latent_rate=latent_rate,
            channel_first=channel_first,
            sample_rate=sample_rate,
            prompt=prompt,
            label=label,
            metadata=metadata,
            session_id=session_id,
        )

    def store_latent_array(
        self,
        array: Any,
        *,
        latent_rate: float,
        artifact_id: str | None = None,
        filename: str = "latent.npy",
        channel_first: bool = False,
        sample_rate: int | None = None,
        recipe: Recipe | None = None,
        source_artifact_ids: list[str] | None = None,
        prompt: str | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> ArtifactRecord:
        resolved_session_id = session_id or (recipe.session_id if recipe else None)
        latent = np.asarray(array, dtype=np.float32)
        if latent.ndim != 2:
            raise ValueError(f"latent must be 2D, got shape {latent.shape}")
        if channel_first:
            latent = latent.T
        if latent.shape[0] < 1 or latent.shape[1] < 1:
            raise ValueError(f"latent must have non-empty axes, got shape {latent.shape}")
        if not np.isfinite(latent).all():
            raise ValueError("latent contains NaN or infinite values")
        if latent_rate <= 0:
            raise ValueError("latent_rate must be positive")

        artifact_id, path = self.reserve_artifact_path(artifact_id=artifact_id, filename=filename)
        np.save(path, np.ascontiguousarray(latent, dtype=np.float32))
        duration = float(latent.shape[0] / latent_rate)
        record = ArtifactRecord(
            artifact_id=artifact_id,
            kind=ArtifactKind.LATENT,
            path=path,
            file=_file_info(path, media_type="application/x-npy"),
            latent=LatentMetadata(
                shape=(int(latent.shape[0]), int(latent.shape[1])),
                latent_rate=float(latent_rate),
                duration_seconds=duration,
                sample_rate=sample_rate,
                channel_first=False,
            ),
            source_artifact_ids=source_artifact_ids or [],
            recipe_id=recipe.recipe_id if recipe else None,
            prompt=prompt,
            label=label,
            metadata=metadata or {},
            session_id=resolved_session_id,
        )
        self.write_artifact(record)
        if recipe is not None:
            self.write_recipe(recipe)
        return record

    def load_latent_array(self, artifact_id: str) -> np.ndarray:
        record = self.get_artifact(artifact_id)
        if record.kind != ArtifactKind.LATENT:
            raise ValueError(f"artifact {artifact_id} is not a latent artifact")
        return np.load(record.path, allow_pickle=False)

    def write_artifact(self, record: ArtifactRecord) -> None:
        directory = self.artifact_dir(record.artifact_id)
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "artifact.json", record.model_dump(mode="json"))

    def write_recipe(self, recipe: Recipe) -> None:
        _write_json(self.recipes_dir / f"{recipe.recipe_id}.json", recipe.model_dump(mode="json"))

    def get_recipe(self, recipe_id: str) -> Recipe:
        path = self.recipes_dir / f"{recipe_id}.json"
        if not path.exists():
            raise KeyError(recipe_id)
        return Recipe.model_validate(_read_json(path))

    def list_recipes(self) -> list[Recipe]:
        recipes = []
        for path in sorted(self.recipes_dir.glob("*.json")):
            recipes.append(Recipe.model_validate(_read_json(path)))
        return sorted(recipes, key=lambda item: item.created_at, reverse=True)

    def create_session(self, request: SessionCreateRequest | None = None) -> SessionRecord:
        request = request or SessionCreateRequest()
        now = utc_now()
        record = SessionRecord(
            name=request.name or f"Session {now.strftime('%Y-%m-%d %H:%M')}",
            notes=request.notes,
            metadata=request.metadata,
            created_at=now,
            updated_at=now,
        )
        self.write_session(record)
        return record

    def write_session(self, record: SessionRecord) -> None:
        _write_json(self.sessions_dir / f"{record.session_id}.json", record.model_dump(mode="json"))

    def get_session(self, session_id: str) -> SessionRecord:
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise KeyError(session_id)
        return SessionRecord.model_validate(_read_json(path))

    def list_sessions(self) -> list[SessionRecord]:
        records = []
        for path in sorted(self.sessions_dir.glob("*.json")):
            records.append(SessionRecord.model_validate(_read_json(path)))
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def update_session(self, session_id: str, request: SessionUpdateRequest) -> SessionRecord:
        record = self.get_session(session_id)
        status = request.status if request.status is not None else record.status
        archived_at = record.archived_at
        if status == SessionStatus.ARCHIVED and record.status != SessionStatus.ARCHIVED:
            archived_at = utc_now()
        elif status == SessionStatus.ACTIVE:
            archived_at = None
        metadata = dict(record.metadata)
        if request.metadata is not None:
            metadata.update(request.metadata)
        updated = record.model_copy(
            update={
                "name": request.name if request.name is not None else record.name,
                "status": status,
                "notes": request.notes if request.notes is not None else record.notes,
                "metadata": metadata,
                "updated_at": utc_now(),
                "archived_at": archived_at,
            }
        )
        self.write_session(updated)
        return updated

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        path = self.artifact_dir(artifact_id) / "artifact.json"
        if not path.exists():
            raise KeyError(artifact_id)
        return ArtifactRecord.model_validate(_read_json(path))

    def inspect_artifact(self, artifact_id: str) -> ArtifactInspection:
        record = self.get_artifact(artifact_id)
        recipe = None
        if record.recipe_id:
            try:
                recipe = self.get_recipe(record.recipe_id)
            except KeyError:
                recipe = None
        sources = []
        for source_id in record.source_artifact_ids:
            try:
                sources.append(self.get_artifact(source_id))
            except KeyError:
                continue
        children = [
            item
            for item in self.list_artifacts()
            if record.artifact_id in item.source_artifact_ids
        ]
        children = sorted(children, key=lambda item: item.created_at, reverse=True)
        return ArtifactInspection(
            artifact=record,
            recipe=recipe,
            sources=sources,
            children=children,
            bundle_files=_bundle_file_entries(record.path) if record.kind == ArtifactKind.BUNDLE else [],
            bundle_preview=_bundle_preview(record) if record.kind == ArtifactKind.BUNDLE else {},
        )

    def audio_peaks(self, artifact_id: str, *, bins: int = 96) -> AudioPeaksResponse:
        record = self.get_artifact(artifact_id)
        if record.kind != ArtifactKind.AUDIO or record.audio is None:
            raise ValueError(f"artifact {artifact_id} is not an audio artifact")
        bins = max(1, min(int(bins), 2048))
        import soundfile as sf

        audio, sample_rate = sf.read(str(record.path), dtype="float32", always_2d=True)
        frames = int(audio.shape[0])
        if frames == 0:
            peaks = [0.0] * bins
        else:
            edges = np.linspace(0, frames, bins + 1, dtype=np.int64)
            peaks = []
            for start, end in zip(edges[:-1], edges[1:]):
                if end <= start:
                    peaks.append(0.0)
                else:
                    peaks.append(float(np.max(np.abs(audio[start:end]))))
        return AudioPeaksResponse(
            artifact_id=artifact_id,
            bins=bins,
            channels=int(audio.shape[1]) if audio.ndim == 2 else 1,
            sample_rate=int(sample_rate),
            duration_seconds=float(frames / sample_rate) if sample_rate else 0.0,
            peaks=peaks,
        )

    def list_artifacts(
        self,
        *,
        kind: ArtifactKind | None = None,
        session_id: str | None = None,
        query: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ArtifactRecord]:
        records = []
        for manifest in sorted(self.artifacts_dir.glob("*/artifact.json")):
            record = ArtifactRecord.model_validate(_read_json(manifest))
            if (
                (kind is None or record.kind == kind)
                and (session_id is None or record.session_id == session_id)
                and _artifact_matches_search(record, query=query, tags=tags)
            ):
                records.append(record)
        return records

    def annotate_artifact(self, artifact_id: str, request: ArtifactAnnotationRequest) -> ArtifactRecord:
        record = self.get_artifact(artifact_id)
        metadata = dict(record.metadata)
        if request.metadata:
            metadata.update(request.metadata)
        updated = record.model_copy(
            update={
                "label": request.label if request.label is not None else record.label,
                "notes": request.notes if request.notes is not None else record.notes,
                "tags": request.tags if request.tags is not None else record.tags,
                "metadata": metadata,
            }
        )
        self.write_artifact(updated)
        return updated


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
    return safe[:180] or "artifact.bin"


def _artifact_matches_search(record: ArtifactRecord, *, query: str | None, tags: list[str] | None) -> bool:
    query_text = (query or "").strip().lower()
    requested_tags = [tag.strip().lower() for tag in tags or [] if tag.strip()]
    record_tags = {tag.lower() for tag in record.tags}
    if requested_tags and not all(tag in record_tags for tag in requested_tags):
        return False
    if not query_text:
        return True

    haystack = [
        record.artifact_id,
        record.kind.value,
        record.label or "",
        record.prompt or "",
        record.notes or "",
        record.file.filename if record.file else "",
        *record.tags,
    ]
    return query_text in " ".join(haystack).lower()


def _bundle_file_entries(path: Path) -> list[BundleFileEntry]:
    if not path.exists():
        return []
    if path.is_dir():
        entries = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            entries.append(
                BundleFileEntry(
                    path=str(child.relative_to(path)),
                    byte_size=child.stat().st_size,
                )
            )
        return entries
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return [
                BundleFileEntry(
                    path=item.filename,
                    byte_size=int(item.file_size),
                    compressed_size=int(item.compress_size),
                )
                for item in archive.infolist()
                if not item.is_dir()
            ]
    return [BundleFileEntry(path=path.name, byte_size=path.stat().st_size)]


def _bundle_preview(record: ArtifactRecord) -> dict[str, Any]:
    preview: dict[str, Any] = {
        "operator": record.metadata.get("operator"),
        "result_count": record.metadata.get("result_count"),
        "metric": record.metadata.get("metric"),
        "top_k": record.metadata.get("top_k"),
    }
    path = record.path
    if path.exists() and path.is_file() and path.suffix.lower() == ".json" and path.stat().st_size <= 1024 * 1024:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            preview["keys"] = list(payload)[:12]
            for key in ("candidate_count", "metric", "top_k", "source_artifact_id"):
                if key in payload:
                    preview[key] = payload[key]
            results = payload.get("results")
            if isinstance(results, list):
                preview["result_count"] = len(results)
                preview["results"] = results[:5]
    return {key: value for key, value in preview.items() if value is not None}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_info(path: Path, *, media_type: str | None = None) -> FileInfo:
    return FileInfo(
        filename=path.name,
        media_type=media_type,
        byte_size=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guess_media_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".aiff":
        return "audio/aiff"
    if suffix == ".aif":
        return "audio/aiff"
    if suffix == ".ogg":
        return "audio/ogg"
    if suffix == ".m4a":
        return "audio/mp4"
    return None


def _audio_media_type(path: Path, media_type: str | None) -> str | None:
    if media_type and media_type not in {"application/octet-stream", "binary/octet-stream"}:
        return media_type
    return _guess_media_type(path) or media_type


def _audio_metadata(path: Path) -> AudioMetadata:
    import soundfile as sf

    info = sf.info(str(path))
    duration = float(info.frames / info.samplerate) if info.samplerate else 0.0
    return AudioMetadata(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        frames=int(info.frames),
        duration_seconds=duration,
        format=info.format,
    )
