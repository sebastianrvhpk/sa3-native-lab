from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import JobRecord, JobStatus, Recipe
from .ids import utc_now


@dataclass(slots=True)
class JobResult:
    artifact_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class JobContext:
    def __init__(self, manager: "JobManager", job_id: str) -> None:
        self._manager = manager
        self.job_id = job_id

    def set_progress(self, progress: float, message: str | None = None) -> None:
        update: dict[str, Any] = {"progress": progress}
        if message is not None:
            update["message"] = message
        self._manager.update(self.job_id, **update)

    def log(self, line: str) -> None:
        if not line:
            return
        self._manager.append_log(self.job_id, line)

    def cancelled(self) -> bool:
        return self._manager.is_cancelled(self.job_id)


JobHandler = Callable[[JobContext], JobResult]


class JobManager:
    def __init__(self, jobs_dir: str | Path, *, max_workers: int = 1) -> None:
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sa3-job")
        self._lock = Lock()
        self._records: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[None]] = {}
        self._load_existing_jobs()

    def submit(self, recipe: Recipe, handler: JobHandler) -> JobRecord:
        record = JobRecord(recipe=recipe, message="queued")
        with self._lock:
            self._records[record.job_id] = record
            self._persist(record)
        future = self._executor.submit(self._run_job, record.job_id, handler)
        with self._lock:
            self._futures[record.job_id] = future
        return record

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            return self._records[job_id]

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            record = self._records[job_id].model_copy(update=changes)
            self._records[job_id] = record
            self._persist(record)
            return record

    def append_log(self, job_id: str, line: str) -> JobRecord:
        with self._lock:
            record = self._records[job_id]
            logs = [*record.logs, line]
            if len(logs) > 200:
                logs = logs[-200:]
            updated = record.model_copy(update={"logs": logs, "message": line})
            self._records[job_id] = updated
            self._persist(updated)
            return updated

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            record = self._records[job_id]
            if record.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
                return record
            future = self._futures.get(job_id)
            if future is not None:
                future.cancel()
            updated = record.model_copy(
                update={
                    "status": JobStatus.CANCELLED,
                    "finished_at": utc_now(),
                    "message": "cancel requested",
                }
            )
            self._records[job_id] = updated
            self._persist(updated)
            return updated

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return self._records[job_id].status == JobStatus.CANCELLED

    def _run_job(self, job_id: str, handler: JobHandler) -> None:
        context = JobContext(self, job_id)
        if context.cancelled():
            return
        self.update(
            job_id,
            status=JobStatus.RUNNING,
            started_at=utc_now(),
            progress=0.01,
            message="running",
        )
        try:
            result = handler(context)
            if context.cancelled():
                return
            self.update(
                job_id,
                status=JobStatus.SUCCEEDED,
                finished_at=utc_now(),
                progress=1.0,
                message="succeeded",
                artifact_ids=result.artifact_ids,
                metrics=result.metrics,
            )
        except Exception as exc:  # pragma: no cover - traceback shape is not important
            if context.cancelled():
                return
            context.log(traceback.format_exc())
            self.update(
                job_id,
                status=JobStatus.FAILED,
                finished_at=utc_now(),
                progress=1.0,
                message="failed",
                error=str(exc),
            )

    def _load_existing_jobs(self) -> None:
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                record = JobRecord.model_validate(payload)
                if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                    record = record.model_copy(
                        update={
                            "status": JobStatus.FAILED,
                            "finished_at": utc_now(),
                            "message": "interrupted before daemon restart",
                            "error": "job was not running after daemon restart",
                        }
                    )
                self._records[record.job_id] = record
            except Exception:
                continue

    def _persist(self, record: JobRecord) -> None:
        path = self.jobs_dir / f"{record.job_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, indent=2, sort_keys=True)
