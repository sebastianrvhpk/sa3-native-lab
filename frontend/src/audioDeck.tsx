import { type CSSProperties, type MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pause, Play, Repeat, SkipBack, SkipForward, X } from "lucide-react";

import { createApi } from "./api";
import { artifactName, formatDuration, formatPlaybackTime } from "./artifactUtils";
import type { ArtifactRecord } from "./types";

interface LoopRegion {
  start: number;
  end: number;
}

export interface PlaybackMarker {
  id: string;
  time: number;
  label: string;
}

export function AudioDeck({ artifact, apiBase, compact = false }: { artifact: ArtifactRecord; apiBase: string; compact?: boolean }) {
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const binCount = compact ? 36 : 128;
  const peaks = useQuery({
    queryKey: ["peaks", apiBase, artifact.artifact_id, binCount],
    queryFn: () => api.audioPeaks(artifact.artifact_id, binCount),
    enabled: artifact.kind === "audio",
    staleTime: Infinity,
  });
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(artifact.audio?.duration_seconds ?? 0);
  const [volume, setVolume] = useState(0.92);
  const [loop, setLoop] = useState(false);
  const [loopRegion, setLoopRegion] = useState<LoopRegion | null>(null);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [markers, setMarkers] = useState<PlaybackMarker[]>([]);
  const fileUrl = `${apiBase}/artifacts/${artifact.artifact_id}/file`;
  const displayDuration = duration || artifact.audio?.duration_seconds || 0;
  const progress = displayDuration > 0 ? currentTime / displayDuration : 0;
  const loopStartFraction = displayDuration && loopRegion ? loopRegion.start / displayDuration : 0;
  const loopEndFraction = displayDuration && loopRegion ? loopRegion.end / displayDuration : 1;

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(artifact.audio?.duration_seconds ?? 0);
    setLoopRegion(null);
    setMarkers([]);
  }, [artifact.artifact_id, artifact.audio?.duration_seconds]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  const seekTo = (fraction: number) => {
    const audio = audioRef.current;
    const targetDuration = duration || artifact.audio?.duration_seconds || audio?.duration || 0;
    if (!audio || !targetDuration) return;
    const nextTime = clamp(fraction, 0, 1) * targetDuration;
    audio.currentTime = nextTime;
    setCurrentTime(nextTime);
  };

  const seekBy = (seconds: number) => {
    const targetDuration = duration || artifact.audio?.duration_seconds || audioRef.current?.duration || 0;
    if (!targetDuration) return;
    seekTo((currentTime + seconds) / targetDuration);
  };

  const setRegionEdge = (edge: "start" | "end") => {
    const targetDuration = duration || artifact.audio?.duration_seconds || audioRef.current?.duration || 0;
    if (!targetDuration) return;
    const fallbackStart = loopRegion?.start ?? 0;
    const fallbackEnd = loopRegion?.end ?? targetDuration;
    const next = clampLoopRegion(
      edge === "start" ? currentTime : fallbackStart,
      edge === "end" ? currentTime : fallbackEnd,
      targetDuration,
    );
    setLoopRegion(next);
    setLoop(true);
  };

  const addMarker = () => {
    const targetDuration = duration || artifact.audio?.duration_seconds || audioRef.current?.duration || 0;
    setMarkers((current) => addPlaybackMarker(current, currentTime, targetDuration));
  };

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      audio.pause();
      return;
    }
    try {
      await audio.play();
    } catch {
      setPlaying(false);
    }
  };

  return (
    <div className={`audio-deck ${compact ? "compact" : ""}`}>
      <audio
        ref={audioRef}
        src={fileUrl}
        preload="metadata"
        loop={loop}
        onLoadedMetadata={(event) => {
          const loadedDuration = event.currentTarget.duration;
          if (Number.isFinite(loadedDuration)) setDuration(loadedDuration);
        }}
        onTimeUpdate={(event) => {
          const audio = event.currentTarget;
          if (loop && loopRegion && audio.currentTime >= loopRegion.end) {
            audio.currentTime = loopRegion.start;
          }
          setCurrentTime(audio.currentTime);
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />
      <div className="deck-head">
        <button type="button" className="transport-wheel" onClick={togglePlay} title={playing ? "Pause" : "Play"}>
          {playing ? <Pause size={compact ? 18 : 24} /> : <Play size={compact ? 18 : 24} />}
        </button>
        <div className="deck-readout">
          <strong>{artifactName(artifact)}</strong>
          <span>
            {formatPlaybackTime(currentTime)} / {formatPlaybackTime(displayDuration)} · {artifact.audio?.sample_rate ?? 0} Hz
          </span>
        </div>
        <button type="button" className={`deck-icon ${loop ? "active" : ""}`} onClick={() => setLoop((value) => !value)} title="Loop">
          <Repeat size={compact ? 15 : 17} />
        </button>
      </div>
      <PlayableWave
        peaks={peaks.data?.peaks ?? placeholderPeaks(artifact.artifact_id, binCount)}
        progress={progress}
        loading={peaks.isLoading}
        compact={compact}
        loopActive={loop && Boolean(loopRegion)}
        loopStart={loopStartFraction}
        loopEnd={loopEndFraction}
        markers={markerFractions(markers, displayDuration)}
        onSeek={seekTo}
      />
      {!compact ? (
        <div className="deck-controls">
          <button type="button" className="deck-icon" onClick={() => seekBy(-2)} title="Back 2 seconds">
            <SkipBack size={16} />
          </button>
          <button type="button" className="deck-icon" onClick={() => seekBy(2)} title="Forward 2 seconds">
            <SkipForward size={16} />
          </button>
          <button type="button" className="deck-chip" onClick={() => setRegionEdge("start")} title="Set loop start">
            In
          </button>
          <button type="button" className="deck-chip" onClick={() => setRegionEdge("end")} title="Set loop end">
            Out
          </button>
          <button type="button" className="deck-chip" onClick={addMarker} title="Add marker at current playback position">
            Mark
          </button>
          {loopRegion ? (
            <button type="button" className="deck-chip" onClick={() => setLoopRegion(null)} title="Clear loop region">
              {formatLoopRegion(loopRegion)}
            </button>
          ) : null}
          {loopRegion ? (
            <span className="deck-region-editor" aria-label="Loop region editor">
              <button type="button" onClick={() => setLoopRegion(nudgeLoopRegion(loopRegion, "start", -0.25, displayDuration))} title="Move loop start earlier">
                In-
              </button>
              <button type="button" onClick={() => setLoopRegion(nudgeLoopRegion(loopRegion, "start", 0.25, displayDuration))} title="Move loop start later">
                In+
              </button>
              <button type="button" onClick={() => setLoopRegion(nudgeLoopRegion(loopRegion, "end", -0.25, displayDuration))} title="Move loop end earlier">
                Out-
              </button>
              <button type="button" onClick={() => setLoopRegion(nudgeLoopRegion(loopRegion, "end", 0.25, displayDuration))} title="Move loop end later">
                Out+
              </button>
            </span>
          ) : null}
          {markers.map((marker) => (
            <span key={marker.id} className="deck-marker">
              <button type="button" className="deck-chip marker-chip" onClick={() => seekTo(displayDuration ? marker.time / displayDuration : 0)} title={`Jump to ${formatPlaybackTime(marker.time)}`}>
                {marker.label} {formatPlaybackTime(marker.time)}
              </button>
              <button type="button" className="deck-marker-delete" aria-label={`Delete ${marker.label}`} onClick={() => setMarkers((current) => removePlaybackMarker(current, marker.id))}>
                <X size={12} />
              </button>
            </span>
          ))}
          {markers.length ? (
            <button type="button" className="deck-chip" onClick={() => setMarkers([])} title="Clear local markers">
              Clear marks
            </button>
          ) : null}
          <label className="deck-volume">
            Volume
            <input type="range" min={0} max={1} step={0.01} value={volume} onChange={(event) => setVolume(Number(event.target.value))} />
          </label>
          <label className="deck-volume rate">
            Rate
            <select value={playbackRate} onChange={(event) => setPlaybackRate(Number(event.target.value))}>
              {[0.5, 0.75, 1, 1.25, 1.5].map((rate) => (
                <option key={rate} value={rate}>
                  {rate}x
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}
    </div>
  );
}

export function TinyWave({ artifact, apiBase }: { artifact: ArtifactRecord; apiBase: string }) {
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const peaks = useQuery({
    queryKey: ["peaks", apiBase, artifact.artifact_id, 18],
    queryFn: () => api.audioPeaks(artifact.artifact_id, 18),
    enabled: artifact.kind === "audio",
    staleTime: Infinity,
  });

  return (
    <WaveBars
      className="tiny-wave"
      peaks={peaks.data?.peaks ?? placeholderPeaks(artifact.artifact_id, 18)}
      loading={peaks.isLoading}
    />
  );
}

function PlayableWave({
  peaks,
  progress,
  loading,
  compact,
  loopActive,
  loopStart,
  loopEnd,
  markers = [],
  onSeek,
}: {
  peaks: number[];
  progress: number;
  loading?: boolean;
  compact?: boolean;
  loopActive?: boolean;
  loopStart?: number;
  loopEnd?: number;
  markers?: number[];
  onSeek: (fraction: number) => void;
}) {
  const maxPeak = Math.max(...peaks, 0.0001);
  const progressPercent = clamp(progress, 0, 1) * 100;
  const onPointerSeek = (event: MouseEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    onSeek((event.clientX - bounds.left) / bounds.width);
  };

  return (
    <div
      className={`playable-wave ${compact ? "compact" : ""} ${loading ? "loading" : ""}`}
      role="slider"
      aria-label="Audio position"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(progressPercent)}
      tabIndex={0}
      onClick={onPointerSeek}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft") onSeek(progress - 0.04);
        if (event.key === "ArrowRight") onSeek(progress + 0.04);
        if (event.key === "Home") onSeek(0);
        if (event.key === "End") onSeek(1);
      }}
      style={
        {
          "--play-progress": `${progressPercent}%`,
          "--loop-start": `${clamp(loopStart ?? 0, 0, 1) * 100}%`,
          "--loop-end": `${clamp(loopEnd ?? 1, 0, 1) * 100}%`,
        } as CSSProperties
      }
    >
      {loopActive ? <b aria-hidden="true" /> : null}
      {markers.map((marker, index) => (
        <em
          key={`${marker}-${index}`}
          className="wave-marker"
          aria-hidden="true"
          style={{ left: `${clamp(marker, 0, 1) * 100}%` }}
        />
      ))}
      {peaks.map((peak, index) => {
        const played = peaks.length <= 1 ? false : index / (peaks.length - 1) <= progress;
        return (
          <span
            key={index}
            className={played ? "played" : ""}
            style={{ height: `${Math.max(6, (Math.abs(peak) / maxPeak) * 100)}%` }}
          />
        );
      })}
      <i aria-hidden="true" />
    </div>
  );
}

function WaveBars({ className, peaks, loading }: { className: string; peaks: number[]; loading?: boolean }) {
  const maxPeak = Math.max(...peaks, 0.0001);
  return (
    <div className={`${className} ${loading ? "loading" : ""}`} aria-hidden="true">
      {peaks.map((peak, index) => (
        <span key={index} style={{ height: `${Math.max(5, (Math.abs(peak) / maxPeak) * 100)}%` }} />
      ))}
    </div>
  );
}

function placeholderPeaks(seed: string, count: number) {
  let value = 0;
  for (const char of seed) value = (value * 31 + char.charCodeAt(0)) >>> 0;
  return Array.from({ length: count }, (_, index) => {
    value = (value * 1664525 + 1013904223 + index) >>> 0;
    return 0.18 + (value % 72) / 100;
  });
}

export function clampLoopRegion(start: number, end: number, duration: number): LoopRegion {
  const boundedDuration = Math.max(0, duration);
  if (!boundedDuration) return { start: 0, end: 0 };
  const safeStart = clamp(Number.isFinite(start) ? start : 0, 0, boundedDuration);
  const safeEnd = clamp(Number.isFinite(end) ? end : boundedDuration, 0, boundedDuration);
  if (safeEnd - safeStart >= 0.05) return { start: safeStart, end: safeEnd };
  const widenedEnd = clamp(safeStart + 0.5, 0, boundedDuration);
  if (widenedEnd - safeStart >= 0.05) return { start: safeStart, end: widenedEnd };
  return { start: Math.max(0, boundedDuration - 0.5), end: boundedDuration };
}

export function formatLoopRegion(region: LoopRegion) {
  return `${formatPlaybackTime(region.start)}-${formatPlaybackTime(region.end)}`;
}

export function nudgeLoopRegion(region: LoopRegion, edge: "start" | "end", deltaSeconds: number, duration: number): LoopRegion {
  if (edge === "start") return clampLoopRegion(region.start + deltaSeconds, region.end, duration);
  return clampLoopRegion(region.start, region.end + deltaSeconds, duration);
}

export function addPlaybackMarker(markers: readonly PlaybackMarker[], currentTime: number, duration: number, limit = 8): PlaybackMarker[] {
  const boundedDuration = Math.max(0, duration);
  if (!boundedDuration) return [...markers];
  const time = clamp(Number.isFinite(currentTime) ? currentTime : 0, 0, boundedDuration);
  const withoutNearDuplicate = markers.filter((marker) => Math.abs(marker.time - time) > 0.05);
  const nextIndex = markers.length + 1;
  const next = [
    ...withoutNearDuplicate,
    {
      id: `marker-${nextIndex}-${Math.round(time * 1000)}`,
      time,
      label: `M${nextIndex}`,
    },
  ]
    .sort((left, right) => left.time - right.time)
    .slice(-limit);
  return next.map((marker, index) => ({ ...marker, label: `M${index + 1}` }));
}

export function removePlaybackMarker(markers: readonly PlaybackMarker[], markerId: string): PlaybackMarker[] {
  return markers
    .filter((marker) => marker.id !== markerId)
    .map((marker, index) => ({ ...marker, label: `M${index + 1}` }));
}

export function markerFractions(markers: readonly PlaybackMarker[], duration: number): number[] {
  if (!duration) return [];
  return markers.map((marker) => Math.round(clamp(marker.time / duration, 0, 1) * 1_000_000) / 1_000_000);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
