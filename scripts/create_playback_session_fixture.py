from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from sa3_native_lab.app.contracts import BackendName, JobRecord, JobStatus, OperatorName, Recipe, SessionCreateRequest
from sa3_native_lab.app.storage import ArtifactStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny audio/session fixture for browser playback smoke tests.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = create_fixture(artifact_root=args.artifact_root, fixtures_dir=args.fixtures_dir)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def create_fixture(*, artifact_root: Path, fixtures_dir: Path) -> dict[str, str]:
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    if fixtures_dir.exists():
        shutil.rmtree(fixtures_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    store = ArtifactStore(artifact_root)
    session = store.create_session(
        SessionCreateRequest(
            name="Playwright Smoke Session",
            notes="Playback/session browser verification fixture.",
            metadata={"smoke": True, "fixture": "playback-session"},
        )
    )

    sample_rate = 8000
    timeline = np.linspace(0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
    source_path = fixtures_dir / "source.wav"
    take_path = fixtures_dir / "take.wav"
    archive_path = fixtures_dir / "archive.wav"
    sf.write(source_path, 0.16 * np.sin(2 * np.pi * 220 * timeline), sample_rate, subtype="FLOAT")
    sf.write(take_path, 0.20 * np.sin(2 * np.pi * 440 * timeline), sample_rate, subtype="FLOAT")
    sf.write(archive_path, 0.10 * np.sin(2 * np.pi * 330 * timeline), sample_rate, subtype="FLOAT")

    source = store.import_audio_file(
        source_path,
        label="Source Pulse",
        prompt="source pulse",
        metadata={"model": "medium", "fixture": "playback-session"},
        session_id=session.session_id,
    )
    recipe = Recipe(
        operator=OperatorName.TEXT_TO_AUDIO,
        backend=BackendName.MLX,
        params={
            "prompt": "warm smoke pulse",
            "duration_seconds": 1.0,
            "steps": 4,
            "model": "medium",
            "decoder": "same-l",
        },
        model="medium",
        seed=17,
        session_id=session.session_id,
    )
    artifact_id, artifact_path = store.reserve_artifact_path(filename="take.wav")
    shutil.copy2(take_path, artifact_path)
    take = store.finalize_audio_file(
        artifact_id=artifact_id,
        path=artifact_path,
        recipe=recipe,
        source_artifact_ids=[source.artifact_id],
        prompt="warm smoke pulse",
        label="Warm Smoke Take",
        metadata={"model": "medium", "result_family_id": "family_smoke", "fixture": "playback-session"},
        session_id=session.session_id,
    )
    archived = store.import_audio_file(
        archive_path,
        label="Archived Smoke Take",
        prompt="archived pulse",
        metadata={"model": "medium", "archived_from_session_id": session.session_id, "fixture": "playback-session"},
        session_id=None,
    )

    job = JobRecord(
        status=JobStatus.SUCCEEDED,
        recipe=recipe,
        progress=1.0,
        phase="done",
        message="succeeded",
        artifact_ids=[take.artifact_id],
        metrics={"duration_seconds": 1.0},
    )
    store.jobs_dir.mkdir(parents=True, exist_ok=True)
    (store.jobs_dir / f"{job.job_id}.json").write_text(json.dumps(job.model_dump(mode="json"), indent=2), encoding="utf-8")

    return {
        "artifact_root": str(artifact_root),
        "session_id": session.session_id,
        "source_artifact_id": source.artifact_id,
        "take_artifact_id": take.artifact_id,
        "archived_artifact_id": archived.artifact_id,
        "job_id": job.job_id,
    }


if __name__ == "__main__":
    main()
