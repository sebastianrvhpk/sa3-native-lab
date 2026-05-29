import type { ArtifactAnnotationPayload } from "./api";
import type { ArtifactRecord } from "./types";

export function artifactRecoveryPayload({
  artifact,
  targetSessionId,
  source,
  now = new Date().toISOString(),
}: {
  artifact: ArtifactRecord;
  targetSessionId: string;
  source: string;
  now?: string;
}): ArtifactAnnotationPayload {
  return {
    session_id: targetSessionId,
    metadata: {
      recovered_from_session_id: artifact.session_id ?? null,
      recovered_into_session_id: targetSessionId,
      recovered_artifact_at: now,
      recovery_source: source,
    },
  };
}

export function recoverableArchiveArtifacts(artifacts: readonly ArtifactRecord[], activeSessionId: string | null | undefined): ArtifactRecord[] {
  if (!activeSessionId) return [];
  return artifacts.filter((artifact) => artifact.session_id !== activeSessionId);
}

export function artifactArchivePayload({
  artifact,
  source,
  now = new Date().toISOString(),
}: {
  artifact: ArtifactRecord;
  source: string;
  now?: string;
}): ArtifactAnnotationPayload {
  return {
    session_id: null,
    metadata: {
      archived_from_session_id: artifact.session_id ?? null,
      archived_artifact_at: now,
      archive_source: source,
    },
  };
}

export function archivableSessionArtifacts(artifacts: readonly ArtifactRecord[], activeSessionId: string | null | undefined): ArtifactRecord[] {
  if (!activeSessionId) return [];
  return artifacts.filter((artifact) => artifact.session_id === activeSessionId);
}
