from __future__ import annotations

import base64
from datetime import datetime, timezone
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
    metadata: Iterable[dict[str, Any]] | None = None,
    title: str = "Audio Player",
    peak_count: int = 900,
    max_embed_mb: float = 96.0,
    annotation_callback: str | None = None,
    existing_annotations: dict[str, Any] | None = None,
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
    metadata_list = list(metadata) if metadata is not None else [{} for _path in path_list]
    if len(metadata_list) != len(path_list):
        raise ValueError("metadata length must match paths length")
    annotation_lookup = _annotation_lookup(existing_annotations or {})

    max_bytes = int(max_embed_mb * 1024 * 1024)
    tracks = []
    total_bytes = 0
    for path, label, track_metadata in zip(path_list, label_list, metadata_list):
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
                "metadata": track_metadata,
                "annotation": annotation_lookup.get(str(path), {}),
            }
        )

    player_id = f"lap_{uuid.uuid4().hex}"
    payload = json.dumps(
        {
            "title": title,
            "tracks": tracks,
            "annotationCallback": annotation_callback,
            "annotationsEnabled": bool(annotation_callback),
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
  #{player_id} .lap-annotations {{
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid var(--lap-border);
    display: grid;
    grid-template-columns: 90px minmax(120px, 1fr) minmax(160px, 2fr);
    gap: 8px;
    align-items: start;
  }}
  #{player_id} .lap-annotations[hidden] {{ display: none; }}
  #{player_id} .lap-annotations input, #{player_id} .lap-annotations textarea {{
    width: 100%;
    background: #10141b;
    color: var(--lap-text);
    border: 1px solid var(--lap-border);
    border-radius: 5px;
    padding: 6px;
    font-size: 12px;
    font-family: inherit;
  }}
  #{player_id} .lap-annotations textarea {{ min-height: 70px; resize: vertical; grid-column: 1 / -1; }}
  #{player_id} .lap-annotation-status {{ color: var(--lap-muted); font-size: 11px; align-self: center; }}
  @media (max-width: 760px) {{
    #{player_id} .lap-shell {{ grid-template-columns: 1fr; }}
    #{player_id} .lap-side {{ border-left: 0; border-top: 1px solid var(--lap-border); max-height: 220px; }}
    #{player_id} .lap-annotations {{ grid-template-columns: 1fr; }}
    #{player_id} .lap-annotations textarea {{ grid-column: auto; }}
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
        <div class="lap-annotations" hidden>
          <label>Rating <input class="lap-rating" type="number" min="-5" max="5" step="0.5" placeholder="0"></label>
          <label>Tags <input class="lap-tags" type="text" placeholder="drums, bright, keeper"></label>
          <label>Use/value <input class="lap-value" type="text" placeholder="loop, transition, texture"></label>
          <textarea class="lap-description" placeholder="Describe what this variation changed, preserved, broke, or made useful."></textarea>
          <button data-action="save-annotation" title="Save annotation to the Colab JSON store">Save note</button>
          <span class="lap-annotation-status"></span>
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
  const annotations = root.querySelector(".lap-annotations");
  const ratingInput = root.querySelector(".lap-rating");
  const tagsInput = root.querySelector(".lap-tags");
  const valueInput = root.querySelector(".lap-value");
  const descriptionInput = root.querySelector(".lap-description");
  const annotationStatus = root.querySelector(".lap-annotation-status");

  title.textContent = data.title;
  annotations.hidden = !data.annotationsEnabled;
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
    loadAnnotation(track.annotation || {{}});
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
  function tagsFromText(value) {{
    return String(value || "").split(",").map((tag) => tag.trim()).filter(Boolean);
  }}
  function loadAnnotation(annotation) {{
    if (!data.annotationsEnabled) return;
    ratingInput.value = annotation.rating ?? "";
    tagsInput.value = Array.isArray(annotation.tags) ? annotation.tags.join(", ") : (annotation.tags || "");
    valueInput.value = annotation.value ?? "";
    descriptionInput.value = annotation.description ?? "";
    annotationStatus.textContent = annotation.updated_at ? `loaded ${{annotation.updated_at}}` : "";
  }}
  function currentAnnotationPayload() {{
    const track = currentTrack();
    const ratingRaw = ratingInput.value;
    return {{
      path: track.path,
      label: track.label,
      track_index: index,
      player_title: data.title,
      rating: ratingRaw === "" ? null : Number(ratingRaw),
      tags: tagsFromText(tagsInput.value),
      value: valueInput.value,
      description: descriptionInput.value,
      metadata: track.metadata || {{}},
    }};
  }}
  async function saveAnnotation() {{
    if (!data.annotationCallback || !(window.google && google.colab && google.colab.kernel)) {{
      annotationStatus.textContent = "Colab callback unavailable";
      return;
    }}
    const payload = currentAnnotationPayload();
    annotationStatus.textContent = "saving...";
    try {{
      const result = await google.colab.kernel.invokeFunction(data.annotationCallback, [payload], {{}});
      currentTrack().annotation = payload;
      annotationStatus.textContent = result && result.data && result.data["application/json"]
        ? "saved"
        : "saved";
    }} catch (error) {{
      annotationStatus.textContent = `save failed: ${{error.message || error}}`;
    }}
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
    if (action === "save-annotation") saveAnnotation();
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
    if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
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
    metadata: Iterable[dict[str, Any]] | None = None,
    title: str = "Audio Player",
    peak_count: int = 900,
    max_embed_mb: float = 96.0,
    annotation_path: str | Path | None = None,
) -> Any:
    """Display the custom player in a notebook and return the HTML object."""

    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise RuntimeError("IPython is required to display the Colab audio player.") from exc

    callback_name = None
    existing_annotations = None
    if annotation_path is not None:
        annotation_path = Path(annotation_path)
        existing_annotations = load_audio_annotations(annotation_path)
        try:
            from google.colab import output  # type: ignore

            callback_name = f"latent_audio_primitives.save_annotation_{uuid.uuid4().hex}"

            def _save_annotation(payload):
                return save_audio_annotation(annotation_path, payload)

            output.register_callback(callback_name, _save_annotation)
        except Exception:
            callback_name = None

    html_obj = HTML(
        audio_player_html(
            paths,
            labels=labels,
            metadata=metadata,
            title=title,
            peak_count=peak_count,
            max_embed_mb=max_embed_mb,
            annotation_callback=callback_name,
            existing_annotations=existing_annotations,
        )
    )
    display(html_obj)
    return html_obj


def load_audio_annotations(path: str | Path) -> dict[str, Any]:
    """Load a JSON annotation store produced by the Colab player."""

    annotation_path = Path(path)
    if not annotation_path.exists():
        return {"annotation_version": 1, "items": []}
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"annotation_version": 1, "items": payload}
    payload.setdefault("annotation_version", 1)
    payload.setdefault("items", [])
    return payload


def save_audio_annotation(path: str | Path, annotation: dict[str, Any]) -> dict[str, Any]:
    """Upsert one track annotation into a JSON store."""

    annotation_path = Path(path)
    store = load_audio_annotations(annotation_path)
    items = list(store.get("items", []))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    normalized = _normalize_annotation(annotation, updated_at=now)
    key = normalized["path"]
    replaced = False
    for index, item in enumerate(items):
        if str(item.get("path", "")) == key:
            normalized["created_at"] = item.get("created_at", now)
            items[index] = normalized
            replaced = True
            break
    if not replaced:
        normalized["created_at"] = now
        items.append(normalized)

    store["annotation_version"] = 1
    store["items"] = items
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    with annotation_path.open("w", encoding="utf-8") as handle:
        json.dump(store, handle, indent=2, ensure_ascii=True)
    return {"ok": True, "path": str(annotation_path), "count": len(items), "item": normalized}


def search_audio_annotations(
    path: str | Path,
    *,
    query: str = "",
    tags: Iterable[str] | None = None,
    min_rating: float | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Search annotation text/tags and return matching tracks."""

    store = load_audio_annotations(path)
    query = query.strip().lower()
    required_tags = {tag.strip().lower() for tag in (tags or []) if tag.strip()}
    matches = []
    for item in store.get("items", []):
        item_tags = {str(tag).lower() for tag in item.get("tags", [])}
        if required_tags and not required_tags.issubset(item_tags):
            continue
        rating = item.get("rating")
        if min_rating is not None and (rating is None or float(rating) < min_rating):
            continue
        searchable = " ".join(
            [
                str(item.get("label", "")),
                str(item.get("value", "")),
                str(item.get("description", "")),
                " ".join(str(tag) for tag in item.get("tags", [])),
            ]
        ).lower()
        if query and query not in searchable:
            continue
        matches.append(item)

    matches.sort(key=lambda item: (float(item.get("rating") or 0.0), item.get("updated_at", "")), reverse=True)
    return matches[:limit] if limit is not None else matches


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


def _annotation_lookup(store: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = store.get("items", []) if isinstance(store, dict) else []
    lookup = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_path = str(item.get("path", ""))
        if item_path:
            lookup[item_path] = item
    return lookup


def _normalize_annotation(annotation: dict[str, Any], *, updated_at: str) -> dict[str, Any]:
    path = str(annotation.get("path", ""))
    if not path:
        raise ValueError("annotation must include a path")
    return {
        "path": path,
        "label": str(annotation.get("label", "")),
        "track_index": _optional_int(annotation.get("track_index")),
        "player_title": str(annotation.get("player_title", "")),
        "rating": _optional_float(annotation.get("rating")),
        "tags": _normalize_tags(annotation.get("tags", [])),
        "value": str(annotation.get("value", "")),
        "description": str(annotation.get("description", "")),
        "metadata": annotation.get("metadata", {}) if isinstance(annotation.get("metadata", {}), dict) else {},
        "updated_at": updated_at,
    }


def _normalize_tags(tags: Any) -> list[str]:
    if isinstance(tags, str):
        raw_tags = tags.split(",")
    elif isinstance(tags, Iterable):
        raw_tags = tags
    else:
        raw_tags = []
    normalized = []
    seen = set()
    for tag in raw_tags:
        value = str(tag).strip()
        key = value.lower()
        if value and key not in seen:
            normalized.append(value)
            seen.add(key)
    return normalized


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


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
