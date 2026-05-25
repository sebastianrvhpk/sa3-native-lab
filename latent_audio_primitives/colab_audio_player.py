from __future__ import annotations

import base64
import html
import json
import mimetypes
import uuid
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def audio_player_html(
    paths: str | Path | Iterable[str | Path],
    *,
    labels: Iterable[str] | None = None,
    title: str = "Audio Player",
    peak_count: int = 900,
    max_embed_mb: float = 96.0,
) -> str:
    """Return a self-contained Colab-friendly audio player.

    The player embeds audio as base64 data URLs so it works in Colab outputs
    without a local web server. It also precomputes waveform peaks in Python so
    the browser only draws a compact canvas representation.
    """

    path_list = _as_paths(paths)
    if not path_list:
        raise ValueError("paths must contain at least one audio file")
    label_list = list(labels) if labels is not None else [path.name for path in path_list]
    if len(label_list) != len(path_list):
        raise ValueError("labels length must match paths length")

    max_bytes = int(max_embed_mb * 1024 * 1024)
    tracks = []
    total_bytes = 0
    for path, label in zip(path_list, label_list):
        raw = path.read_bytes()
        total_bytes += len(raw)
        if total_bytes > max_bytes:
            raise ValueError(
                f"embedded audio would be {total_bytes / 1024 / 1024:.1f} MB, "
                f"above max_embed_mb={max_embed_mb}. Use fewer/shorter files."
            )
        mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
        tracks.append(
            {
                "label": str(label),
                "path": str(path),
                "src": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
                "mime": mime,
                "duration": _audio_duration_seconds(path),
                "peaks": _waveform_peaks(path, peak_count=peak_count),
            }
        )

    player_id = f"lap_{uuid.uuid4().hex}"
    payload = json.dumps(
        {
            "title": title,
            "tracks": tracks,
        },
        ensure_ascii=True,
    ).replace("</", "<\\/")
    safe_title = html.escape(title)

    return f"""
<div id="{player_id}" class="lap-root">
  <div class="lap-fallback">Loading {safe_title}...</div>
</div>
<style>
  #{player_id}.lap-root {{
    --lap-bg: #111318;
    --lap-panel: #1b1f27;
    --lap-soft: #252b35;
    --lap-text: #edf1f7;
    --lap-muted: #99a3b3;
    --lap-accent: #8bd3ff;
    --lap-accent-2: #f6c177;
    --lap-border: #343c4a;
    background: var(--lap-bg);
    color: var(--lap-text);
    border: 1px solid var(--lap-border);
    border-radius: 8px;
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 100%;
    overflow: hidden;
  }}
  #{player_id} * {{ box-sizing: border-box; }}
  #{player_id} .lap-shell {{ display: grid; grid-template-columns: minmax(0, 1fr) 280px; min-height: 280px; }}
  #{player_id} .lap-main {{ padding: 14px; min-width: 0; }}
  #{player_id} .lap-side {{ border-left: 1px solid var(--lap-border); background: #151922; max-height: 420px; overflow: auto; }}
  #{player_id} .lap-title {{ font-size: 14px; font-weight: 700; margin-bottom: 2px; }}
  #{player_id} .lap-track-title {{ font-size: 13px; color: var(--lap-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  #{player_id} canvas {{ display: block; width: 100%; height: 128px; background: #0c0f14; border: 1px solid var(--lap-border); border-radius: 6px; margin: 12px 0; cursor: crosshair; }}
  #{player_id} .lap-controls, #{player_id} .lap-loop {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
  #{player_id} button {{
    background: var(--lap-soft);
    color: var(--lap-text);
    border: 1px solid var(--lap-border);
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 12px;
    cursor: pointer;
  }}
  #{player_id} button:hover {{ border-color: var(--lap-accent); }}
  #{player_id} button.lap-active {{ background: #163246; border-color: var(--lap-accent); }}
  #{player_id} .lap-readout {{ color: var(--lap-muted); font-variant-numeric: tabular-nums; font-size: 12px; min-width: 108px; }}
  #{player_id} label {{ color: var(--lap-muted); font-size: 12px; display: inline-flex; gap: 5px; align-items: center; }}
  #{player_id} input[type="range"] {{ width: 96px; }}
  #{player_id} input[type="number"] {{
    width: 72px;
    background: #10141b;
    color: var(--lap-text);
    border: 1px solid var(--lap-border);
    border-radius: 5px;
    padding: 5px;
    font-size: 12px;
  }}
  #{player_id} .lap-loop {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--lap-border); }}
  #{player_id} .lap-row {{
    padding: 10px 12px;
    border-bottom: 1px solid var(--lap-border);
    cursor: pointer;
  }}
  #{player_id} .lap-row:hover {{ background: #202633; }}
  #{player_id} .lap-row.lap-selected {{ background: #173246; }}
  #{player_id} .lap-row-name {{ font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  #{player_id} .lap-row-meta {{ color: var(--lap-muted); font-size: 11px; margin-top: 3px; font-variant-numeric: tabular-nums; }}
  #{player_id} .lap-help {{ color: var(--lap-muted); font-size: 11px; margin-top: 10px; line-height: 1.4; }}
  @media (max-width: 760px) {{
    #{player_id} .lap-shell {{ grid-template-columns: 1fr; }}
    #{player_id} .lap-side {{ border-left: 0; border-top: 1px solid var(--lap-border); max-height: 220px; }}
  }}
</style>
<script>
(() => {{
  const root = document.getElementById({json.dumps(player_id)});
  const data = {payload};
  const audio = new Audio();
  audio.preload = "auto";
  let index = 0;
  let loopRegion = false;
  let rafId = null;

  root.innerHTML = `
    <div class="lap-shell">
      <div class="lap-main">
        <div class="lap-title"></div>
        <div class="lap-track-title"></div>
        <canvas width="1200" height="180"></canvas>
        <div class="lap-controls">
          <button data-action="prev" title="Previous track">Prev</button>
          <button data-action="play" title="Play or pause">Play</button>
          <button data-action="stop" title="Stop">Stop</button>
          <button data-action="next" title="Next track">Next</button>
          <button data-action="loop-track" title="Loop current track">Loop track</button>
          <span class="lap-readout">0:00 / 0:00</span>
          <label>Vol <input class="lap-volume" type="range" min="0" max="1" step="0.01" value="1"></label>
          <label>Speed <input class="lap-rate" type="range" min="0.5" max="1.5" step="0.01" value="1"></label>
        </div>
        <div class="lap-loop">
          <button data-action="mark-in" title="Set loop start at current time">Set in</button>
          <button data-action="mark-out" title="Set loop end at current time">Set out</button>
          <button data-action="loop-region" title="Toggle loop region">Loop region</button>
          <button data-action="clear-region" title="Clear loop region">Clear</button>
          <label>In <input class="lap-in" type="number" min="0" step="0.01" value="0"></label>
          <label>Out <input class="lap-out" type="number" min="0" step="0.01" value="0"></label>
        </div>
        <div class="lap-help">Shortcuts while this player is focused/clicked: space play/pause, arrows seek, J/K previous/next.</div>
      </div>
      <div class="lap-side"></div>
    </div>`;

  root.tabIndex = 0;
  const title = root.querySelector(".lap-title");
  const trackTitle = root.querySelector(".lap-track-title");
  const canvas = root.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const readout = root.querySelector(".lap-readout");
  const playButton = root.querySelector('[data-action="play"]');
  const loopTrackButton = root.querySelector('[data-action="loop-track"]');
  const loopRegionButton = root.querySelector('[data-action="loop-region"]');
  const volume = root.querySelector(".lap-volume");
  const rate = root.querySelector(".lap-rate");
  const inInput = root.querySelector(".lap-in");
  const outInput = root.querySelector(".lap-out");
  const side = root.querySelector(".lap-side");

  title.textContent = data.title;
  side.innerHTML = data.tracks.map((track, i) => `
    <div class="lap-row" data-index="${{i}}">
      <div class="lap-row-name">${{escapeHtml(track.label)}}</div>
      <div class="lap-row-meta">${{fmt(track.duration)}} · ${{escapeHtml(track.mime)}}</div>
    </div>`).join("");

  function escapeHtml(value) {{
    return String(value).replace(/[&<>"']/g, (ch) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
  }}
  function fmt(value) {{
    if (!Number.isFinite(value) || value <= 0) return "0:00";
    const minutes = Math.floor(value / 60);
    const seconds = Math.floor(value % 60).toString().padStart(2, "0");
    return `${{minutes}}:${{seconds}}`;
  }}
  function currentTrack() {{ return data.tracks[index]; }}
  function setTrack(nextIndex, autoplay=false) {{
    index = (nextIndex + data.tracks.length) % data.tracks.length;
    const track = currentTrack();
    audio.src = track.src;
    audio.currentTime = 0;
    trackTitle.textContent = track.label;
    inInput.value = "0";
    outInput.value = track.duration ? track.duration.toFixed(2) : "0";
    root.querySelectorAll(".lap-row").forEach((row, i) => row.classList.toggle("lap-selected", i === index));
    draw();
    updateReadout();
    if (autoplay) audio.play();
  }}
  function draw() {{
    const track = currentTrack();
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c0f14";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#33404f";
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    const peaks = track.peaks || [];
    if (peaks.length) {{
      ctx.strokeStyle = "#8bd3ff";
      ctx.lineWidth = 1;
      const step = width / peaks.length;
      for (let i = 0; i < peaks.length; i++) {{
        const peak = Math.max(0, Math.min(1, Math.abs(peaks[i])));
        const x = i * step;
        const y = peak * (height * 0.46);
        ctx.beginPath();
        ctx.moveTo(x, height / 2 - y);
        ctx.lineTo(x, height / 2 + y);
        ctx.stroke();
      }}
    }}
    const duration = audio.duration || track.duration || 0;
    const progress = duration ? audio.currentTime / duration : 0;
    ctx.fillStyle = "rgba(246, 193, 119, 0.92)";
    ctx.fillRect(progress * width, 0, 2, height);
    if (loopRegion) {{
      const start = Number(inInput.value || 0);
      const end = Number(outInput.value || 0);
      if (duration && end > start) {{
        ctx.fillStyle = "rgba(246, 193, 119, 0.14)";
        ctx.fillRect((start / duration) * width, 0, ((end - start) / duration) * width, height);
      }}
    }}
  }}
  function updateReadout() {{
    const duration = audio.duration || currentTrack().duration || 0;
    readout.textContent = `${{fmt(audio.currentTime)}} / ${{fmt(duration)}}`;
    playButton.textContent = audio.paused ? "Play" : "Pause";
    loopTrackButton.classList.toggle("lap-active", audio.loop);
    loopRegionButton.classList.toggle("lap-active", loopRegion);
  }}
  function tick() {{
    draw();
    updateReadout();
    rafId = requestAnimationFrame(tick);
  }}
  function seekBy(seconds) {{
    const duration = audio.duration || currentTrack().duration || 0;
    audio.currentTime = Math.max(0, Math.min(duration, audio.currentTime + seconds));
  }}

  root.addEventListener("click", (event) => {{
    root.focus();
    const row = event.target.closest(".lap-row");
    if (row) {{
      setTrack(Number(row.dataset.index), !audio.paused);
      return;
    }}
    const button = event.target.closest("button");
    if (!button) return;
    const action = button.dataset.action;
    if (action === "play") audio.paused ? audio.play() : audio.pause();
    if (action === "stop") {{ audio.pause(); audio.currentTime = 0; }}
    if (action === "prev") setTrack(index - 1, !audio.paused);
    if (action === "next") setTrack(index + 1, !audio.paused);
    if (action === "loop-track") audio.loop = !audio.loop;
    if (action === "mark-in") inInput.value = audio.currentTime.toFixed(2);
    if (action === "mark-out") outInput.value = audio.currentTime.toFixed(2);
    if (action === "loop-region") {{ loopRegion = !loopRegion; if (loopRegion) audio.loop = false; }}
    if (action === "clear-region") {{ loopRegion = false; inInput.value = "0"; outInput.value = (audio.duration || currentTrack().duration || 0).toFixed(2); }}
    draw();
    updateReadout();
  }});
  canvas.addEventListener("click", (event) => {{
    const rect = canvas.getBoundingClientRect();
    const duration = audio.duration || currentTrack().duration || 0;
    if (duration) audio.currentTime = ((event.clientX - rect.left) / rect.width) * duration;
    draw();
  }});
  volume.addEventListener("input", () => audio.volume = Number(volume.value));
  rate.addEventListener("input", () => audio.playbackRate = Number(rate.value));
  audio.addEventListener("timeupdate", () => {{
    if (loopRegion) {{
      const start = Math.max(0, Number(inInput.value || 0));
      const end = Number(outInput.value || 0);
      if (end > start && audio.currentTime >= end) audio.currentTime = start;
    }}
    updateReadout();
  }});
  audio.addEventListener("play", () => {{ if (rafId === null) tick(); updateReadout(); }});
  audio.addEventListener("pause", () => {{ if (rafId !== null) {{ cancelAnimationFrame(rafId); rafId = null; }} draw(); updateReadout(); }});
  audio.addEventListener("loadedmetadata", () => {{ outInput.value = (audio.duration || currentTrack().duration || 0).toFixed(2); updateReadout(); draw(); }});
  root.addEventListener("keydown", (event) => {{
    if (event.target.tagName === "INPUT") return;
    if (event.code === "Space") {{ event.preventDefault(); audio.paused ? audio.play() : audio.pause(); }}
    if (event.key === "ArrowLeft") seekBy(-2);
    if (event.key === "ArrowRight") seekBy(2);
    if (event.key.toLowerCase() === "j") setTrack(index - 1, !audio.paused);
    if (event.key.toLowerCase() === "k") setTrack(index + 1, !audio.paused);
  }});

  setTrack(0, false);
}})();
</script>
"""


def display_audio_player(
    paths: str | Path | Iterable[str | Path],
    *,
    labels: Iterable[str] | None = None,
    title: str = "Audio Player",
    peak_count: int = 900,
    max_embed_mb: float = 96.0,
) -> Any:
    """Display the custom player in a notebook and return the HTML object."""

    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise RuntimeError("IPython is required to display the Colab audio player.") from exc
    html_obj = HTML(
        audio_player_html(
            paths,
            labels=labels,
            title=title,
            peak_count=peak_count,
            max_embed_mb=max_embed_mb,
        )
    )
    display(html_obj)
    return html_obj


def _as_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(path) for path in paths]
    for path in path_list:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"not a file: {path}")
    return path_list


def _audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate and info.frames:
            return float(info.frames / info.samplerate)
    except Exception:
        pass

    try:
        with wave.open(str(path), "rb") as handle:
            return float(handle.getnframes() / handle.getframerate())
    except Exception:
        return 0.0


def _waveform_peaks(path: Path, *, peak_count: int = 900) -> list[float]:
    if peak_count <= 0:
        raise ValueError("peak_count must be positive")
    audio = _read_audio_mono(path)
    if audio.size == 0:
        return []
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 0:
        audio = audio / max_abs
    bucket_count = min(peak_count, audio.shape[0])
    edges = np.linspace(0, audio.shape[0], bucket_count + 1, dtype=np.int64)
    peaks = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            peaks.append(0.0)
        else:
            chunk = audio[start:end]
            positive = float(np.max(chunk))
            negative = float(np.min(chunk))
            peaks.append(positive if abs(positive) >= abs(negative) else negative)
    return peaks


def _read_audio_mono(path: Path) -> np.ndarray:
    try:
        import soundfile as sf

        data, _sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        return data.mean(axis=1)
    except Exception:
        pass

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
            if sample_width == 1:
                data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                data = (data - 128.0) / 128.0
            elif sample_width == 2:
                data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            elif sample_width == 4:
                data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                return np.zeros(0, dtype=np.float32)
            if channels > 1:
                data = data.reshape(-1, channels).mean(axis=1)
            return data
    except Exception:
        return np.zeros(0, dtype=np.float32)
