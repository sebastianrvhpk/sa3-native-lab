"""SA3 residual-timestep cartography rows, schedules, and probe converters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TrajectoryCell:
    """One measured SA3 internal-trajectory coordinate.

    A cell is usually produced from a residual layer/timestep probe row. It is a
    microscope/selector object: high score means a contrast is visible at this
    layer and sampler coordinate, not that steering there is already proven.
    """

    layer_index: int
    score: float
    rank: int = 0
    source: str = "residual_timestep"
    method: str = ""
    status: str = "ok"
    step_index: int | None = None
    window_index: int | None = None
    timestep: float | None = None
    sigma: float | None = None
    logsnr: float | None = None
    sampler_type: str = ""
    calls_per_step: int | None = None
    call_start: int | None = None
    call_end: int | None = None
    call_count: int | None = None
    mapping_status: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": int(self.layer_index),
            "score": float(self.score),
            "rank": int(self.rank),
            "source": self.source,
            "method": self.method,
            "status": self.status,
            "step_index": _optional_int(self.step_index),
            "window_index": _optional_int(self.window_index),
            "timestep": _optional_float(self.timestep),
            "sigma": _optional_float(self.sigma),
            "logsnr": _optional_float(self.logsnr),
            "sampler_type": self.sampler_type,
            "calls_per_step": _optional_int(self.calls_per_step),
            "call_start": _optional_int(self.call_start),
            "call_end": _optional_int(self.call_end),
            "call_count": _optional_int(self.call_count),
            "mapping_status": self.mapping_status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryMap:
    """Ranked collection of SA3 internal-trajectory cells."""

    cells: Sequence[TrajectoryCell]
    source: str = "residual_timestep"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))

    def ranked(
        self,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        mapping_statuses: Sequence[str] | str | None = None,
        require_ok: bool = True,
    ) -> list[TrajectoryCell]:
        return rank_trajectory_cells(
            self.cells,
            top_k=top_k,
            min_score=min_score,
            mapping_statuses=mapping_statuses,
            require_ok=require_ok,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "metadata": dict(self.metadata),
            "cell_count": len(self.cells),
            "cells": [cell.to_dict() for cell in self.cells],
        }


@dataclass(frozen=True, slots=True)
class TrajectoryAlphaEntry:
    """One scheduled residual-steering entry over hook-call coordinates."""

    layer_index: int
    alpha: float
    score: float
    rank: int = 0
    step_index: int | None = None
    call_start: int | None = None
    call_end: int | None = None
    calls_per_step: int | None = None
    mapping_status: str = ""

    def matches_call(self, layer_index: int, call_index: int) -> bool:
        if int(layer_index) != int(self.layer_index):
            return False
        if self.call_start is not None and self.call_end is not None:
            return int(self.call_start) <= int(call_index) < int(self.call_end)
        if self.step_index is not None and self.calls_per_step:
            start = int(self.step_index) * int(self.calls_per_step)
            end = start + int(self.calls_per_step)
            return start <= int(call_index) < end
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": int(self.layer_index),
            "alpha": float(self.alpha),
            "score": float(self.score),
            "rank": int(self.rank),
            "step_index": _optional_int(self.step_index),
            "call_start": _optional_int(self.call_start),
            "call_end": _optional_int(self.call_end),
            "calls_per_step": _optional_int(self.calls_per_step),
            "mapping_status": self.mapping_status,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryAlphaSchedule:
    """Callable residual-steering schedule produced from trajectory cells."""

    entries: Sequence[TrajectoryAlphaEntry]
    combine: str = "max_abs"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.combine not in {"max_abs", "sum"}:
            raise ValueError("combine must be 'max_abs' or 'sum'")
        object.__setattr__(self, "entries", tuple(self.entries))

    def alpha_for(self, layer_index: int, call_index: int, base_alpha: float = 1.0) -> float:
        matches = [
            float(entry.alpha)
            for entry in self.entries
            if entry.matches_call(int(layer_index), int(call_index))
        ]
        if not matches:
            return 0.0
        if self.combine == "sum":
            return float(base_alpha) * sum(matches)
        strongest = max(matches, key=lambda value: abs(value))
        return float(base_alpha) * strongest

    def __call__(self, layer_index: int, call_index: int, base_alpha: float = 1.0) -> float:
        return self.alpha_for(layer_index, call_index, base_alpha=base_alpha)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "combine": self.combine,
            "metadata": dict(self.metadata),
            "entry_count": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
        }


def trajectory_map_from_probe_rows(
    rows: Sequence[Any],
    *,
    source: str = "residual_timestep",
    score_key: str = "accuracy_mean",
    metadata: dict[str, Any] | None = None,
) -> TrajectoryMap:
    """Convert residual probe rows or dictionaries into a trajectory map."""

    cells = [trajectory_cell_from_probe_row(row, source=source, score_key=score_key) for row in rows]
    cells = rank_trajectory_cells(cells, require_ok=False)
    ranked = [
        TrajectoryCell(**{**cell.to_dict(), "rank": rank})
        for rank, cell in enumerate(cells, start=1)
    ]
    return TrajectoryMap(cells=ranked, source=source, metadata=dict(metadata or {}))


def trajectory_cell_from_probe_row(
    row: Any,
    *,
    source: str = "residual_timestep",
    score_key: str = "accuracy_mean",
) -> TrajectoryCell:
    """Convert one ``LayerProbeRow``-like object into a trajectory cell."""

    metadata = {}
    for key in (
        "accuracy_std",
        "fold_count",
        "sample_count",
        "positive_count",
        "negative_count",
        "error",
        "window_label",
        "window_start_fraction",
        "window_end_fraction",
        "window_count",
    ):
        value = _row_get(row, key)
        if value is not None:
            metadata[key] = value

    return TrajectoryCell(
        layer_index=int(_row_get(row, "layer_index")),
        score=float(_row_get(row, score_key, 0.0) or 0.0),
        rank=int(_row_get(row, "rank", 0) or 0),
        source=source,
        method=str(_row_get(row, "method", "")),
        status=str(_row_get(row, "status", "ok") or "ok"),
        step_index=_optional_int(_row_get(row, "step_index")),
        window_index=_optional_int(_row_get(row, "window_index")),
        timestep=_optional_float(_row_get(row, "timestep")),
        sigma=_optional_float(_row_get(row, "sigma")),
        logsnr=_optional_float(_row_get(row, "logsnr")),
        sampler_type=str(_row_get(row, "sampler_type", "") or ""),
        calls_per_step=_optional_int(_row_get(row, "calls_per_step")),
        call_start=_optional_int(_row_get(row, "call_start")),
        call_end=_optional_int(_row_get(row, "call_end")),
        call_count=_optional_int(_row_get(row, "call_count")),
        mapping_status=str(_row_get(row, "mapping_status", "") or ""),
        metadata=metadata,
    )


def rank_trajectory_cells(
    cells: Sequence[TrajectoryCell],
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    mapping_statuses: Sequence[str] | str | None = None,
    require_ok: bool = True,
) -> list[TrajectoryCell]:
    """Return cells sorted by score with optional evidence filters."""

    allowed_mapping = _as_status_set(mapping_statuses)
    out = []
    for cell in cells:
        if require_ok and cell.status not in {"", "ok"}:
            continue
        if min_score is not None and cell.score < float(min_score):
            continue
        if allowed_mapping is not None and cell.mapping_status not in allowed_mapping:
            continue
        out.append(cell)
    out.sort(
        key=lambda cell: (
            -float(cell.score),
            int(cell.rank or 10**9),
            int(cell.layer_index),
            -1 if cell.step_index is None else int(cell.step_index),
            -1 if cell.window_index is None else int(cell.window_index),
        )
    )
    if top_k is not None:
        out = out[: max(0, int(top_k))]
    return out


def summarize_trajectory_bands(
    cells: Sequence[TrajectoryCell],
    *,
    step_count: int | None = None,
    min_score: float | None = None,
    require_ok: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate trajectory cells into noisy/mid/clean or early/mid/late bands."""

    ranked = rank_trajectory_cells(cells, min_score=min_score, require_ok=require_ok)
    grouped: dict[str, list[TrajectoryCell]] = {}
    for cell in ranked:
        grouped.setdefault(trajectory_band(cell, step_count=step_count), []).append(cell)

    rows: list[dict[str, Any]] = []
    for band, band_cells in grouped.items():
        scores = [float(cell.score) for cell in band_cells]
        top = band_cells[0]
        rows.append(
            {
                "band": band,
                "cell_count": len(band_cells),
                "score_mean": float(sum(scores) / max(len(scores), 1)),
                "score_max": float(max(scores)),
                "top_layer": int(top.layer_index),
                "top_step": _optional_int(top.step_index),
                "top_timestep": _optional_float(top.timestep),
                "top_sigma": _optional_float(top.sigma),
                "top_logsnr": _optional_float(top.logsnr),
                "top_mapping_status": top.mapping_status,
            }
        )
    rows.sort(key=lambda row: -float(row["score_max"]))
    return rows


def trajectory_band(cell: TrajectoryCell, *, step_count: int | None = None) -> str:
    """Name the sampler/flow region for a trajectory cell."""

    if cell.logsnr is not None:
        if cell.logsnr <= -1.0:
            return "noisy_high_sigma"
        if cell.logsnr >= 1.0:
            return "clean_low_sigma"
        return "middle_logsnr"
    if step_count and cell.step_index is not None:
        frac = (float(cell.step_index) + 0.5) / max(float(step_count), 1.0)
        if frac < 1.0 / 3.0:
            return "early_steps"
        if frac < 2.0 / 3.0:
            return "middle_steps"
        return "late_steps"
    return "unknown"


def trajectory_cells_to_table(
    cells: Sequence[TrajectoryCell],
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    mapping_statuses: Sequence[str] | str | None = None,
    require_ok: bool = True,
) -> list[dict[str, Any]]:
    """Return compact JSON-friendly rows for notebook tables and heatmaps."""

    ranked = rank_trajectory_cells(
        cells,
        top_k=top_k,
        min_score=min_score,
        mapping_statuses=mapping_statuses,
        require_ok=require_ok,
    )
    return [
        {
            "rank": index,
            "layer": int(cell.layer_index),
            "step": _optional_int(cell.step_index),
            "window": _optional_int(cell.window_index),
            "score": float(cell.score),
            "band": trajectory_band(cell),
            "timestep": _optional_float(cell.timestep),
            "sigma": _optional_float(cell.sigma),
            "logsnr": _optional_float(cell.logsnr),
            "calls_per_step": _optional_int(cell.calls_per_step),
            "mapping_status": cell.mapping_status,
            "status": cell.status,
        }
        for index, cell in enumerate(ranked, start=1)
    ]


def trajectory_cells_to_alpha_schedule(
    cells: Sequence[TrajectoryCell],
    *,
    top_k: int | None = 8,
    min_score: float | None = None,
    mapping_statuses: Sequence[str] | str | None = ("exact_one_call_per_step", "grouped_calls_per_step"),
    normalize: bool = True,
    alpha: float = 1.0,
    combine: str = "max_abs",
) -> TrajectoryAlphaSchedule:
    """Convert high-scoring cells into a call-index residual steering schedule."""

    ranked = rank_trajectory_cells(
        cells,
        top_k=top_k,
        min_score=min_score,
        mapping_statuses=mapping_statuses,
    )
    if normalize and ranked:
        scale = max(abs(float(cell.score)) for cell in ranked) or 1.0
    else:
        scale = 1.0

    entries = []
    skipped = 0
    for cell in ranked:
        has_call_window = cell.call_start is not None and cell.call_end is not None
        has_step_window = cell.step_index is not None and cell.calls_per_step is not None
        if not has_call_window and not has_step_window:
            skipped += 1
            continue
        entries.append(
            TrajectoryAlphaEntry(
                layer_index=cell.layer_index,
                alpha=float(alpha) * float(cell.score) / scale,
                score=float(cell.score),
                rank=int(cell.rank),
                step_index=cell.step_index,
                call_start=cell.call_start,
                call_end=cell.call_end,
                calls_per_step=cell.calls_per_step,
                mapping_status=cell.mapping_status,
            )
        )

    return TrajectoryAlphaSchedule(
        entries=entries,
        combine=combine,
        metadata={
            "top_k": top_k,
            "min_score": min_score,
            "mapping_statuses": None if mapping_statuses is None else sorted(_as_status_set(mapping_statuses) or []),
            "normalize": normalize,
            "alpha": float(alpha),
            "skipped_cells_without_call_window": skipped,
        },
    )


def trajectory_cells_to_flow_probe_bank(
    cells: Sequence[TrajectoryCell],
    *,
    top_k: int | None = 8,
    min_score: float | None = None,
    velocity_convention: str = "noise_minus_data",
    seed: int = 0,
    shared_noise: bool = True,
    antithetic_noise: bool = False,
    metadata: dict[str, Any] | None = None,
):
    """Build a flow probe bank from the timesteps of high-scoring cells."""

    from latent_audio_primitives.flow_prompt import FlowProbeBank, FlowProbeSpec

    ranked = rank_trajectory_cells(cells, top_k=top_k, min_score=min_score)
    signs = (1.0, -1.0) if antithetic_noise else (1.0,)
    probes = []
    seen: set[tuple[float, float]] = set()
    probe_index = 0
    value_index = 0
    for cell in ranked:
        if cell.timestep is None:
            continue
        timestep_seen = False
        for sign in signs:
            key = (round(float(cell.timestep), 8), float(sign))
            if key in seen:
                continue
            seen.add(key)
            timestep_seen = True
            probes.append(
                FlowProbeSpec(
                    probe_index=probe_index,
                    timestep=float(cell.timestep),
                    logsnr=cell.logsnr,
                    noise_seed=int(seed) + value_index * 1009,
                    noise_sign=float(sign),
                )
            )
            probe_index += 1
        if timestep_seen:
            value_index += 1
    if not probes:
        raise ValueError("no trajectory cells with timesteps were available for a flow probe bank")
    return FlowProbeBank(
        probes=probes,
        velocity_convention=velocity_convention,
        shared_noise=shared_noise,
        antithetic_noise=antithetic_noise,
        seed=int(seed),
        metadata={
            **dict(metadata or {}),
            "source": "trajectory_cells_to_flow_probe_bank",
            "cell_count": len(ranked),
            "top_k": top_k,
            "min_score": min_score,
        },
    )


def trajectory_cells_to_cyclic_mix_schedule(
    cells: Sequence[TrajectoryCell],
    *,
    steps: int,
    base_mix: float,
    top_k: int | None = 8,
    min_score: float | None = None,
    normalize: bool = True,
    default_mix: float = 0.0,
) -> list[float]:
    """Create per-denoising-step cyclic projection strengths from trajectory cells."""

    step_count = max(1, int(steps))
    values = [float(default_mix) for _ in range(step_count)]
    ranked = [
        cell
        for cell in rank_trajectory_cells(cells, top_k=top_k, min_score=min_score)
        if cell.step_index is not None and 0 <= int(cell.step_index) < step_count
    ]
    if not ranked:
        return values
    scale = max(abs(float(cell.score)) for cell in ranked) if normalize else 1.0
    if scale <= 0:
        scale = 1.0
    for cell in ranked:
        step = int(cell.step_index)
        mix = float(base_mix) * abs(float(cell.score)) / scale
        values[step] = max(values[step], mix)
    return values


def sampler_timestep_recorder(
    records: list[dict[str, Any]],
    *,
    user_callback: Any = None,
    sampler_type: str | None = None,
):
    """Return an SA3 sampler callback that records JSON-friendly step metadata."""

    def callback(info: dict[str, Any]) -> None:
        record = normalize_sampler_step_record(info, sampler_type=sampler_type)
        records.append(record)
        if user_callback is not None:
            user_callback(info)

    return callback


def normalize_sampler_step_record(info: dict[str, Any], *, sampler_type: str | None = None) -> dict[str, Any]:
    """Summarize one SA3 sampler callback payload without retaining tensors."""

    timestep = _tensor_scalar_summary(info.get("t"))
    sigma = _tensor_scalar_summary(info.get("sigma", info.get("sigma_hat", info.get("t"))))
    sigma_value = sigma.get("mean")
    return {
        "sampler_index": _optional_int(info.get("i")),
        "timestep": timestep.get("mean"),
        "timestep_min": timestep.get("min"),
        "timestep_max": timestep.get("max"),
        "sigma": sigma_value,
        "sigma_min": sigma.get("min"),
        "sigma_max": sigma.get("max"),
        "logsnr": logsnr_from_sigma(sigma_value),
        "sampler_type": sampler_type or "",
    }


def logsnr_from_sigma(sigma: float | None) -> float | None:
    """Convert a unit-interval sigma/timestep to the local amplitude logSNR."""

    if sigma is None:
        return None
    eps = 1e-8
    clipped = min(max(float(sigma), eps), 1.0 - eps)
    return float(math.log((1.0 - clipped) / clipped))


def _tensor_scalar_summary(value: Any) -> dict[str, float | None]:
    if value is None:
        return {"mean": None, "min": None, "max": None}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().cpu().reshape(-1)
            if tensor.numel() == 0:
                return {"mean": None, "min": None, "max": None}
            return {
                "mean": float(tensor.mean().item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
            }
    except Exception:
        pass
    try:
        scalar = float(value)
    except Exception:
        return {"mean": None, "min": None, "max": None}
    return {"mean": scalar, "min": scalar, "max": scalar}


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _as_status_set(value: Sequence[str] | str | None) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}
