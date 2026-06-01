import { type CSSProperties, type MouseEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pause, Play, Repeat, SkipBack, SkipForward, X } from "lucide-react";
import type WaveSurfer from "wavesurfer.js";
import type RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js";
import type { Region } from "wavesurfer.js/dist/plugins/regions.js";

import { createApi, type ArtifactAnnotationPayload } from "./api";
import { artifactName, formatPlaybackTime } from "./artifactUtils";
import {
  playbackAnnotationPayload,
  playbackStateFromArtifact,
  playbackStateSignature,
  type PlaybackLoopRegion,
  type PlaybackMarker,
} from "./playbackState";
import type { ArtifactRecord } from "./types";

export function AudioDeck({
  artifact,
  apiBase,
  compact = false,
  onAnnotate,
  autoPlay = false,
  onEnded,
  persisting = false,
}: {
  artifact: ArtifactRecord;
  apiBase: string;
  compact?: boolean;
  onAnnotate?: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
  autoPlay?: boolean;
  onEnded?: () => void;
  persisting?: boolean;
}) {
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const binCount = compact ? 36 : 128;
  const peaks = useQuery({
    queryKey: ["peaks", apiBase, artifact.artifact_id, binCount],
    queryFn: () => api.audioPeaks(artifact.artifact_id, binCount),
    enabled: artifact.kind === "audio",
    staleTime: Infinity,
  });
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const autoPlayKeyRef = useRef("");
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(artifact.audio?.duration_seconds ?? 0);
  const [volume, setVolume] = useState(0.92);
  const [loop, setLoop] = useState(false);
  const persistedPlaybackState = useMemo(() => playbackStateFromArtifact(artifact), [artifact]);
  const persistedSignature = useMemo(
    () => playbackStateSignature(persistedPlaybackState.markers, persistedPlaybackState.loopRegion),
    [persistedPlaybackState],
  );
  const [loopRegion, setLoopRegion] = useState<PlaybackLoopRegion | null>(persistedPlaybackState.loopRegion);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [markers, setMarkers] = useState<PlaybackMarker[]>(persistedPlaybackState.markers);
  const [savedSignature, setSavedSignature] = useState(persistedSignature);
  const [zoom, setZoom] = useState(0);
  const fileUrl = `${apiBase}/artifacts/${artifact.artifact_id}/file`;
  const wavePeaks = useMemo(
    () => peaks.data?.peaks ?? placeholderPeaks(artifact.artifact_id, binCount),
    [artifact.artifact_id, binCount, peaks.data?.peaks],
  );
  const displayDuration = duration || artifact.audio?.duration_seconds || 0;
  const progress = displayDuration > 0 ? currentTime / displayDuration : 0;
  const loopStartFraction = displayDuration && loopRegion ? loopRegion.start / displayDuration : 0;
  const loopEndFraction = displayDuration && loopRegion ? loopRegion.end / displayDuration : 1;
  const localSignature = useMemo(() => playbackStateSignature(markers, loopRegion), [markers, loopRegion]);
  const cuesDirty = localSignature !== savedSignature;

  const setAudioNode = useCallback((node: HTMLAudioElement | null) => {
    audioRef.current = node;
    setAudioElement(node);
  }, []);

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(artifact.audio?.duration_seconds ?? 0);
    setLoopRegion(persistedPlaybackState.loopRegion);
    setMarkers(persistedPlaybackState.markers);
    setSavedSignature(persistedSignature);
  }, [artifact.artifact_id, artifact.audio?.duration_seconds, persistedPlaybackState, persistedSignature]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.volume = volume;
  }, [volume]);

  useEffect(() => {
    if (audioRef.current) audioRef.current.playbackRate = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    const audio = audioRef.current;
    const autoPlayKey = `${artifact.artifact_id}:${autoPlay}`;
    if (!autoPlay) {
      autoPlayKeyRef.current = "";
      return;
    }
    if (!audio || autoPlayKeyRef.current === autoPlayKey) return;
    autoPlayKeyRef.current = autoPlayKey;
    audio.play().catch(() => setPlaying(false));
  }, [artifact.artifact_id, autoPlay]);

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

  const handleLoopRegionChange = useCallback((region: PlaybackLoopRegion) => {
    setLoopRegion(region);
    setLoop(true);
  }, []);

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

  const savePlaybackState = () => {
    if (!onAnnotate) return;
    onAnnotate(
      artifact.artifact_id,
      playbackAnnotationPayload({
        markers,
        loopRegion,
        source: "audio_deck",
      }),
    );
    setSavedSignature(localSignature);
  };

  return (
    <div className={`audio-deck ${compact ? "compact" : ""}`}>
      <audio
        ref={setAudioNode}
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
        onEnded={() => {
          setPlaying(false);
          onEnded?.();
        }}
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
      {compact ? (
        <PlayableWave
          peaks={wavePeaks}
          progress={progress}
          loading={peaks.isLoading}
          compact={compact}
          loopActive={loop && Boolean(loopRegion)}
          loopStart={loopStartFraction}
          loopEnd={loopEndFraction}
          markers={markerFractions(markers, displayDuration)}
          onSeek={seekTo}
        />
      ) : (
        <WaveSurferWave
          media={audioElement}
          fileUrl={fileUrl}
          peaks={wavePeaks}
          loading={peaks.isLoading}
          duration={displayDuration}
          progress={progress}
          loopActive={loop && Boolean(loopRegion)}
          loopRegion={loopRegion}
          markers={markerFractions(markers, displayDuration)}
          zoom={zoom}
          onSeek={seekTo}
          onTimeUpdate={setCurrentTime}
          onLoopRegionChange={handleLoopRegionChange}
        />
      )}
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
              <input
                aria-label={`Note for ${marker.label}`}
                className="deck-marker-note"
                value={marker.note ?? ""}
                placeholder="note"
                maxLength={160}
                onChange={(event) => setMarkers((current) => updatePlaybackMarkerNote(current, marker.id, event.target.value))}
              />
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
          {onAnnotate ? (
            <button type="button" className={`deck-chip persist ${cuesDirty ? "dirty" : ""}`} onClick={savePlaybackState} disabled={!cuesDirty || persisting} title="Save playback cues to artifact metadata">
              {persisting ? "Saving cues" : cuesDirty ? "Save cues" : "Cues saved"}
            </button>
          ) : null}
          <label className="deck-volume">
            Volume
            <input type="range" min={0} max={1} step={0.01} value={volume} onChange={(event) => setVolume(Number(event.target.value))} />
          </label>
          <label className="deck-volume zoom">
            Zoom
            <input type="range" min={0} max={180} step={1} value={zoom} onChange={(event) => setZoom(Number(event.target.value))} />
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

function WaveSurferWave({
  media,
  fileUrl,
  peaks,
  loading,
  duration,
  progress,
  loopActive,
  loopRegion,
  markers,
  zoom,
  onSeek,
  onTimeUpdate,
  onLoopRegionChange,
}: {
  media: HTMLAudioElement | null;
  fileUrl: string;
  peaks: number[];
  loading?: boolean;
  duration: number;
  progress: number;
  loopActive?: boolean;
  loopRegion: PlaybackLoopRegion | null;
  markers: number[];
  zoom: number;
  onSeek: (fraction: number) => void;
  onTimeUpdate: (time: number) => void;
  onLoopRegionChange: (region: PlaybackLoopRegion) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const waveRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const waveCleanupRef = useRef<(() => void) | null>(null);
  const syncingRegionRef = useRef(false);
  const loopActiveRef = useRef(loopActive);
  const loopRegionRef = useRef(loopRegion);
  const safeDuration = Math.max(0, duration);

  useEffect(() => {
    loopActiveRef.current = loopActive;
    loopRegionRef.current = loopRegion;
  }, [loopActive, loopRegion]);

  useEffect(() => {
    if (!containerRef.current || !media || !safeDuration) return;
    let cancelled = false;
    waveCleanupRef.current?.();
    waveCleanupRef.current = null;

    void Promise.all([
      import("wavesurfer.js"),
      import("wavesurfer.js/dist/plugins/regions.js"),
      import("wavesurfer.js/dist/plugins/zoom.js"),
    ]).then(([waveModule, regionsModule, zoomModule]) => {
      if (cancelled || !containerRef.current) return;
      const regions = regionsModule.default.create();
      const wave = waveModule.default.create({
        container: containerRef.current,
        media,
        url: fileUrl,
        peaks: [peaks],
        duration: safeDuration,
        height: 96,
        normalize: true,
        minPxPerSec: 0,
        cursorWidth: 2,
        cursorColor: "rgba(45, 51, 52, 0.75)",
        waveColor: ["rgba(143, 91, 210, 0.72)", "rgba(85, 198, 199, 0.62)", "rgba(243, 168, 60, 0.54)"],
        progressColor: ["rgba(234, 74, 162, 0.78)", "rgba(201, 238, 72, 0.62)", "rgba(65, 108, 199, 0.72)"],
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        dragToSeek: true,
        plugins: [
          regions,
          zoomModule.default.create({
            maxZoom: 180,
            scale: 0.35,
            deltaThreshold: 8,
          }),
        ],
      });
      waveRef.current = wave;
      regionsRef.current = regions;

      const offTime = wave.on("timeupdate", onTimeUpdate);
      const offInteraction = wave.on("interaction", (time) => onTimeUpdate(time));
      const offError = wave.on("error", () => {});
      const offCreated = regions.on("region-created", (region) => {
        if (syncingRegionRef.current) return;
        for (const item of regions.getRegions()) {
          if (item.id !== region.id) item.remove();
        }
        onLoopRegionChange(clampLoopRegion(region.start, region.end, safeDuration));
      });
      const offUpdated = regions.on("region-updated", (region) => {
        if (syncingRegionRef.current) return;
        onLoopRegionChange(clampLoopRegion(region.start, region.end, safeDuration));
      });
      const disableDragSelection = regions.enableDragSelection(
        {
          color: "rgba(85, 198, 199, 0.24)",
          drag: true,
          resize: true,
          minLength: 0.05,
        },
        5,
      );
      syncingRegionRef.current = true;
      if (loopActiveRef.current && loopRegionRef.current) {
        addWaveRegion(regions, loopRegionRef.current, safeDuration);
      }
      queueMicrotask(() => {
        syncingRegionRef.current = false;
      });

      waveCleanupRef.current = () => {
        offTime();
        offInteraction();
        offError();
        offCreated();
        offUpdated();
        disableDragSelection();
        wave.destroy();
        waveRef.current = null;
        regionsRef.current = null;
      };
    }).catch(() => {});

    return () => {
      cancelled = true;
      waveCleanupRef.current?.();
      waveCleanupRef.current = null;
    };
  }, [fileUrl, media, onLoopRegionChange, onTimeUpdate, peaks, safeDuration]);

  useEffect(() => {
    const wave = waveRef.current;
    if (!wave || !duration) return;
    try {
      wave.setOptions({ peaks: [peaks], duration });
    } catch {
      // The plain bar fallback still reflects peaks if WaveSurfer cannot re-render this browser's media.
    }
  }, [duration, peaks]);

  useEffect(() => {
    const wave = waveRef.current;
    if (!wave) return;
    try {
      wave.zoom(zoom);
    } catch {
      // Zoom is unavailable until WaveSurfer has decoded or accepted precomputed peaks.
    }
  }, [zoom]);

  useEffect(() => {
    const regions = regionsRef.current;
    if (!regions || !safeDuration) return;
    syncingRegionRef.current = true;
    regions.clearRegions();
    if (loopActive && loopRegion) {
      addWaveRegion(regions, loopRegion, safeDuration);
    }
    queueMicrotask(() => {
      syncingRegionRef.current = false;
    });
  }, [loopActive, loopRegion, safeDuration]);

  return (
    <div
      className={`wave-surfer-stage ${loading ? "loading" : ""}`}
      role="slider"
      aria-label="Audio position"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(clamp(progress, 0, 1) * 100)}
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "ArrowLeft") onSeek(progress - 0.04);
        if (event.key === "ArrowRight") onSeek(progress + 0.04);
        if (event.key === "Home") onSeek(0);
        if (event.key === "End") onSeek(1);
      }}
    >
      <div ref={containerRef} className="wave-surfer-canvas" />
      <div className="surfer-marker-layer" aria-hidden="true">
        {markers.map((marker, index) => (
          <i key={`${marker}-${index}`} className="wave-marker" style={{ left: `${clamp(marker, 0, 1) * 100}%` }} />
        ))}
      </div>
    </div>
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

export function clampLoopRegion(start: number, end: number, duration: number): PlaybackLoopRegion {
  const boundedDuration = Math.max(0, duration);
  if (!boundedDuration) return { start: 0, end: 0 };
  const safeStart = clamp(Number.isFinite(start) ? start : 0, 0, boundedDuration);
  const safeEnd = clamp(Number.isFinite(end) ? end : boundedDuration, 0, boundedDuration);
  if (safeEnd - safeStart >= 0.05) return { start: safeStart, end: safeEnd };
  const widenedEnd = clamp(safeStart + 0.5, 0, boundedDuration);
  if (widenedEnd - safeStart >= 0.05) return { start: safeStart, end: widenedEnd };
  return { start: Math.max(0, boundedDuration - 0.5), end: boundedDuration };
}

export function formatLoopRegion(region: PlaybackLoopRegion) {
  return `${formatPlaybackTime(region.start)}-${formatPlaybackTime(region.end)}`;
}

export function nudgeLoopRegion(region: PlaybackLoopRegion, edge: "start" | "end", deltaSeconds: number, duration: number): PlaybackLoopRegion {
  if (edge === "start") return clampLoopRegion(region.start + deltaSeconds, region.end, duration);
  return clampLoopRegion(region.start, region.end + deltaSeconds, duration);
}

function addWaveRegion(regions: RegionsPlugin, region: PlaybackLoopRegion, duration: number): Region {
  const next = clampLoopRegion(region.start, region.end, duration);
  return regions.addRegion({
    id: "active-loop-region",
    start: next.start,
    end: next.end,
    color: "rgba(85, 198, 199, 0.24)",
    drag: true,
    resize: true,
    minLength: 0.05,
  });
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

export function updatePlaybackMarkerNote(markers: readonly PlaybackMarker[], markerId: string, note: string): PlaybackMarker[] {
  const trimmed = note.trim();
  return markers.map((marker) => {
    if (marker.id !== markerId) return marker;
    const next = { ...marker };
    if (trimmed) {
      next.note = trimmed.slice(0, 160);
    } else {
      delete next.note;
    }
    return next;
  });
}

export function markerFractions(markers: readonly PlaybackMarker[], duration: number): number[] {
  if (!duration) return [];
  return markers.map((marker) => Math.round(clamp(marker.time / duration, 0, 1) * 1_000_000) / 1_000_000);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
