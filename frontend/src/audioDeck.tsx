import { type CSSProperties, type MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pause, Play, Repeat, SkipBack, SkipForward } from "lucide-react";

import { createApi } from "./api";
import { artifactName, formatDuration, formatPlaybackTime } from "./artifactUtils";
import type { ArtifactRecord } from "./types";

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
  const fileUrl = `${apiBase}/artifacts/${artifact.artifact_id}/file`;
  const displayDuration = duration || artifact.audio?.duration_seconds || 0;
  const progress = displayDuration > 0 ? currentTime / displayDuration : 0;

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(artifact.audio?.duration_seconds ?? 0);
  }, [artifact.artifact_id, artifact.audio?.duration_seconds]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

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
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
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
          <label className="deck-volume">
            Volume
            <input type="range" min={0} max={1} step={0.01} value={volume} onChange={(event) => setVolume(Number(event.target.value))} />
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
  onSeek,
}: {
  peaks: number[];
  progress: number;
  loading?: boolean;
  compact?: boolean;
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
      style={{ "--play-progress": `${progressPercent}%` } as CSSProperties}
    >
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

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
