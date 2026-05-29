import type { ArtifactAnnotationPayload } from "./api";
import type { ArtifactRecord } from "./types";

export interface PlaybackLoopRegion {
  start: number;
  end: number;
}

export interface PlaybackMarker {
  id: string;
  time: number;
  label: string;
}

export interface ArtifactPlaybackState {
  markers: PlaybackMarker[];
  loopRegion: PlaybackLoopRegion | null;
  updatedAt?: string | null;
  source?: string | null;
}

export const PLAYBACK_STATE_METADATA_KEY = "playback_state";

export function playbackStateFromArtifact(artifact: ArtifactRecord): ArtifactPlaybackState {
  const state = objectValue(artifact.metadata[PLAYBACK_STATE_METADATA_KEY]);
  const markers = arrayValue(state?.markers)
    .map((item, index) => playbackMarkerFromMetadata(item, index))
    .filter((marker): marker is PlaybackMarker => Boolean(marker));
  const loopRegion = playbackLoopFromMetadata(state?.loop_region ?? state?.loopRegion);
  return {
    markers,
    loopRegion,
    updatedAt: stringValue(state?.updated_at ?? state?.updatedAt),
    source: stringValue(state?.source),
  };
}

export function playbackAnnotationPayload({
  markers,
  loopRegion,
  source,
  now = new Date().toISOString(),
}: {
  markers: readonly PlaybackMarker[];
  loopRegion: PlaybackLoopRegion | null;
  source: string;
  now?: string;
}): ArtifactAnnotationPayload {
  const safeMarkers = markers
    .map((marker, index) => normalizePlaybackMarker(marker, index))
    .filter((marker): marker is PlaybackMarker => Boolean(marker));
  const safeLoop = loopRegion ? normalizePlaybackLoop(loopRegion) : null;
  return {
    metadata: {
      [PLAYBACK_STATE_METADATA_KEY]: {
        markers: safeMarkers,
        loop_region: safeLoop,
        updated_at: now,
        source,
      },
      playback_marker_count: safeMarkers.length,
      playback_loop_region: safeLoop ? `${safeLoop.start.toFixed(3)}-${safeLoop.end.toFixed(3)}` : null,
    },
  };
}

export function playbackStateSignature(markers: readonly PlaybackMarker[], loopRegion: PlaybackLoopRegion | null): string {
  return JSON.stringify({
    markers: markers.map((marker, index) => normalizePlaybackMarker(marker, index)),
    loopRegion: loopRegion ? normalizePlaybackLoop(loopRegion) : null,
  });
}

function playbackMarkerFromMetadata(value: unknown, index: number): PlaybackMarker | null {
  const item = objectValue(value);
  if (!item) return null;
  const time = numberValue(item.time);
  if (time === null) return null;
  return normalizePlaybackMarker(
    {
      id: stringValue(item.id) || `marker-${index + 1}-${Math.round(time * 1000)}`,
      time,
      label: stringValue(item.label) || `M${index + 1}`,
    },
    index,
  );
}

function playbackLoopFromMetadata(value: unknown): PlaybackLoopRegion | null {
  const item = objectValue(value);
  if (!item) return null;
  const start = numberValue(item.start);
  const end = numberValue(item.end);
  if (start === null || end === null || end <= start) return null;
  return normalizePlaybackLoop({ start, end });
}

function normalizePlaybackMarker(marker: PlaybackMarker, index: number): PlaybackMarker | null {
  const time = finiteNumber(marker.time);
  if (time === null || time < 0) return null;
  return {
    id: marker.id || `marker-${index + 1}-${Math.round(time * 1000)}`,
    label: marker.label || `M${index + 1}`,
    time: roundTime(time),
  };
}

function normalizePlaybackLoop(region: PlaybackLoopRegion): PlaybackLoopRegion {
  return {
    start: roundTime(Math.max(0, finiteNumber(region.start) ?? 0)),
    end: roundTime(Math.max(0, finiteNumber(region.end) ?? 0)),
  };
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function stringValue(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberValue(value: unknown): number | null {
  if (typeof value === "number") return finiteNumber(value);
  if (typeof value === "string" && value.trim()) return finiteNumber(Number(value));
  return null;
}

function finiteNumber(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

function roundTime(value: number): number {
  return Math.round(value * 1000) / 1000;
}
