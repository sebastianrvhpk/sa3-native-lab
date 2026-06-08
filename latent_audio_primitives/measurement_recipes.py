"""Measurement recipes for the SA3/SAME notebook lab.

The helpers in this module keep the notebook cells small without turning the
project into an app framework. They describe evidence rows, perturbation
recipes, and lightweight measurements that compose lower-level primitives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .latent_blur import LatentBlurSpec, apply_latent_blur
from .latent_dsp import LatentDSPSpec, apply_latent_dsp, latent_change_report


@dataclass(frozen=True, slots=True)
class BottleneckPerturbationSpec:
    """A SAME-latent perturbation used for tomography and survival tests."""

    name: str
    family: str
    mode: str
    strength: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["params"] = dict(self.params)
        return row


def default_bottleneck_specs() -> list[BottleneckPerturbationSpec]:
    """Structured perturbations that test what the SAME latent bottleneck carries."""

    return [
        BottleneckPerturbationSpec(
            name="temporal_blur_r2",
            family="temporal_detail",
            mode="latent_blur",
            strength=0.35,
            params={"mode": "temporal", "temporal_radius": 2},
        ),
        BottleneckPerturbationSpec(
            name="temporal_blur_r8",
            family="temporal_detail",
            mode="latent_blur",
            strength=0.65,
            params={"mode": "temporal", "temporal_radius": 8},
        ),
        BottleneckPerturbationSpec(
            name="channel_blur_r2",
            family="channel_topology",
            mode="latent_blur",
            strength=0.45,
            params={"mode": "channel", "channel_radius": 2},
        ),
        BottleneckPerturbationSpec(
            name="low_rank_8",
            family="latent_basis",
            mode="latent_blur",
            strength=0.6,
            params={"mode": "low_rank", "rank": 8},
        ),
        BottleneckPerturbationSpec(
            name="fft_lowpass_025",
            family="trajectory_bandwidth",
            mode="latent_blur",
            strength=0.55,
            params={"mode": "fft_lowpass", "filter_cutoff": 0.25},
        ),
        BottleneckPerturbationSpec(
            name="fft_highpass_025",
            family="trajectory_bandwidth",
            mode="latent_blur",
            strength=0.55,
            params={"mode": "fft_highpass", "filter_cutoff": 0.25},
        ),
        BottleneckPerturbationSpec(
            name="channel_dropout_10",
            family="channel_topology",
            mode="channel_dropout",
            strength=0.4,
            params={"fraction": 0.10, "seed": 0, "fill": "mean"},
        ),
        BottleneckPerturbationSpec(
            name="channel_dropout_25",
            family="channel_topology",
            mode="channel_dropout",
            strength=0.8,
            params={"fraction": 0.25, "seed": 1, "fill": "mean"},
        ),
        BottleneckPerturbationSpec(
            name="latent_noise_002",
            family="noise_stability",
            mode="gaussian_noise",
            strength=0.2,
            params={"amount": 0.02, "seed": 0},
        ),
        BottleneckPerturbationSpec(
            name="latent_noise_010",
            family="noise_stability",
            mode="gaussian_noise",
            strength=0.7,
            params={"amount": 0.10, "seed": 1},
        ),
        BottleneckPerturbationSpec(
            name="motion_softclip_08",
            family="amplitude_geometry",
            mode="latent_dsp",
            strength=0.45,
            params={"mode": "softclip", "drive": 0.8},
        ),
    ]


def apply_bottleneck_perturbation(latents: Any, spec: BottleneckPerturbationSpec) -> Any:
    """Apply a tomography perturbation to a latent tensor.

    The operation intentionally stays in SAME-latent space. Any SA3 polish should
    happen in a separate survival cell so prior invention is measurable.
    """

    if spec.mode == "latent_blur":
        blur_spec = LatentBlurSpec(name=spec.name, strength=spec.strength, **dict(spec.params))
        return apply_latent_blur(latents, blur_spec)
    if spec.mode == "latent_dsp":
        dsp_spec = LatentDSPSpec(name=spec.name, strength=spec.strength, **dict(spec.params))
        return apply_latent_dsp(latents, dsp_spec)
    if spec.mode == "channel_dropout":
        return _apply_channel_dropout(latents, **dict(spec.params))
    if spec.mode == "gaussian_noise":
        return _apply_gaussian_noise(latents, **dict(spec.params))
    raise ValueError(f"Unknown bottleneck perturbation mode: {spec.mode!r}")


def bottleneck_row(
    reference_latents: Any,
    edited_latents: Any,
    spec: BottleneckPerturbationSpec,
    *,
    descriptor_delta_norm: float | None = None,
) -> dict[str, Any]:
    """Measure a bottleneck perturbation against its reference latent."""

    report = dict(latent_change_report(reference_latents, edited_latents))
    row: dict[str, Any] = {
        "name": spec.name,
        "family": spec.family,
        "mode": spec.mode,
        "strength": spec.strength,
        "descriptor_delta_norm": descriptor_delta_norm,
        **report,
    }
    row["severity_hint"] = _severity_hint(row.get("delta_rms", 0.0), row.get("cosine_similarity", 1.0))
    return row


def classify_edit_survival(
    source_latents: Any,
    direct_edited_latents: Any,
    polished_latents: Any,
    *,
    plain_polished_latents: Any | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Classify whether an edit survives a SAME->SA3->SAME round trip."""

    direct_delta = float(latent_change_report(source_latents, direct_edited_latents)["delta_rms"])
    polished_delta = float(latent_change_report(source_latents, polished_latents)["delta_rms"])
    edit_survival = float(
        latent_change_report(direct_edited_latents, polished_latents)["delta_rms"]
    )
    survival_ratio = polished_delta / max(direct_delta, eps)
    plain_polish_delta = None
    invention_ratio = None
    if plain_polished_latents is not None:
        plain_polish_delta = float(latent_change_report(source_latents, plain_polished_latents)["delta_rms"])
        invention_ratio = plain_polish_delta / max(polished_delta, eps)

    if direct_delta < eps * 10:
        label = "no_op_or_failed_edit"
    elif survival_ratio < 0.25:
        label = "erased"
    elif survival_ratio <= 1.25:
        label = "preserved"
    elif survival_ratio <= 2.0:
        label = "amplified"
    else:
        label = "prior_dominated_or_unstable"

    if invention_ratio is not None and invention_ratio > 0.75 and survival_ratio > 1.25:
        label = "prior_invention_or_plain_polish"

    return {
        "direct_delta": direct_delta,
        "polished_delta": polished_delta,
        "direct_to_polished_delta": edit_survival,
        "survival_ratio": survival_ratio,
        "plain_polish_delta": plain_polish_delta,
        "plain_polish_ratio": invention_ratio,
        "survival_label": label,
    }


def source_cartography_row(
    source_latents: Any,
    donor_latents: Any,
    edited_latents: Any,
    *,
    name: str,
    mask_name: str,
    selected_channel_count: int | None = None,
) -> dict[str, Any]:
    """Measure how a masked edit sits between source and donor latent evidence."""

    source_report = dict(latent_change_report(source_latents, edited_latents))
    donor_report = dict(latent_change_report(donor_latents, edited_latents))
    source_delta = float(source_report["delta_rms"])
    donor_delta = float(donor_report["delta_rms"])
    donor_pull = source_delta / max(source_delta + donor_delta, 1e-8)
    return {
        "name": name,
        "mask_name": mask_name,
        "selected_channel_count": selected_channel_count,
        "source_delta": source_delta,
        "donor_delta": donor_delta,
        "donor_pull": donor_pull,
        "source_cosine": source_report.get("cosine_similarity"),
        "donor_cosine": donor_report.get("cosine_similarity"),
        "leakage_flag": bool(donor_pull > 0.65),
    }


def control_identification_row(
    *,
    control: str,
    probe: Any | None = None,
    intervention: Mapping[str, Any] | None = None,
    lane_score: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """Compact row for latent control system-identification experiments."""

    train_r2 = _optional_float(getattr(probe, "r2_train", None))
    item_count = getattr(probe, "item_count", None)
    delta = None if intervention is None else _optional_float(intervention.get("delta"))
    if train_r2 is None:
        maturity = "unmeasured"
    elif train_r2 < 0.1:
        maturity = "not_observable"
    elif delta is None:
        maturity = "observable_needs_intervention"
    elif abs(delta) < 1e-4:
        maturity = "observable_but_not_moved"
    else:
        maturity = "observable_and_moved"
    return {
        "control": control,
        "train_r2": train_r2,
        "item_count": item_count,
        "intervention_delta": delta,
        "lane_score": lane_score,
        "maturity_hint": maturity,
        "notes": notes,
    }


def composition_plan_rows(
    *,
    continuations: Sequence[tuple[Any, float]] | None = None,
    bridges: Sequence[tuple[Any, float]] | None = None,
    path: Sequence[Any] | None = None,
    path_cost: float | None = None,
) -> list[dict[str, Any]]:
    """Normalize continuation, bridge, and path-ranking outputs into rows."""

    rows: list[dict[str, Any]] = []
    for rank, (item, cost) in enumerate(continuations or [], start=1):
        rows.append(
            {
                "kind": "continuation",
                "rank": rank,
                "item_id": getattr(item, "item_id", str(item)),
                "cost": float(cost),
            }
        )
    for rank, (item, cost) in enumerate(bridges or [], start=1):
        rows.append(
            {
                "kind": "bridge",
                "rank": rank,
                "item_id": getattr(item, "item_id", str(item)),
                "cost": float(cost),
            }
        )
    if path:
        rows.append(
            {
                "kind": "path",
                "rank": 1,
                "item_id": " -> ".join(getattr(item, "item_id", str(item)) for item in path),
                "cost": None if path_cost is None else float(path_cost),
            }
        )
    return rows


def flow_semantic_band_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    semantic_axis_by_prompt: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate flow prompt rows by prompt, semantic axis, and logSNR band."""

    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        prompt = str(_row_get(row, "prompt", ""))
        if not prompt:
            continue
        axis = str((semantic_axis_by_prompt or {}).get(prompt, _infer_prompt_axis(prompt)))
        band = _logsnr_band(_optional_float(_row_get(row, "logsnr", None)))
        buckets.setdefault((prompt, axis, band), []).append(row)

    out: list[dict[str, Any]] = []
    for (prompt, axis, band), group in sorted(buckets.items()):
        losses = [_optional_float(_row_get(row, "loss", _row_get(row, "mse", None))) for row in group]
        cosines = [_optional_float(_row_get(row, "cosine", _row_get(row, "cosine_direction", None))) for row in group]
        losses = [value for value in losses if value is not None]
        cosines = [value for value in cosines if value is not None]
        out.append(
            {
                "prompt": prompt,
                "semantic_axis": axis,
                "logsnr_band": band,
                "probe_count": len(group),
                "loss_mean": _mean_or_none(losses),
                "loss_min": min(losses) if losses else None,
                "loss_max": max(losses) if losses else None,
                "cosine_mean": _mean_or_none(cosines),
            }
        )
    return out


def default_factor_axes() -> dict[str, list[str]]:
    """Prompt families for a first melody/rhythm/timbre factor atlas."""

    return {
        "rhythm": [
            "steady pulse, clear rhythm, tight groove",
            "syncopated rhythm, offbeat accents, percussive motion",
            "slow sparse rhythm, wide gaps, minimal percussion",
        ],
        "timbre": [
            "bright metallic timbre, sharp attacks",
            "warm muted timbre, rounded transients",
            "noisy textured timbre, grain and air",
        ],
        "melody_harmony": [
            "simple rising melody, tonal center",
            "drifting atonal melody, unstable harmony",
            "static drone, no melodic contour",
        ],
        "density": [
            "dense layered arrangement, many overlapping voices",
            "sparse arrangement, isolated events, silence",
            "single focused source, no accompaniment",
        ],
    }


def factor_atlas_rows(
    *,
    factor_axes: Mapping[str, Sequence[str]] | None = None,
    flow_rows: Iterable[Mapping[str, Any]] | None = None,
    same_rows: Iterable[Mapping[str, Any]] | None = None,
    trajectory_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Summarize convergent evidence for musical factor hypotheses."""

    axes = factor_axes or default_factor_axes()
    flow_rows = list(flow_rows or [])
    same_rows = list(same_rows or [])
    trajectory_rows = list(trajectory_rows or [])
    out: list[dict[str, Any]] = []
    for factor, prompts in axes.items():
        prompt_set = set(prompts)
        factor_flow = [row for row in flow_rows if str(_row_get(row, "prompt", "")) in prompt_set]
        flow_loss_values = [
            _optional_float(_row_get(row, "loss_mean", _row_get(row, "loss", None))) for row in factor_flow
        ]
        flow_loss_values = [value for value in flow_loss_values if value is not None]
        same_values = [
            _optional_float(_row_get(row, "descriptor_delta_norm", _row_get(row, "delta_rms", None)))
            for row in same_rows
            if factor in str(_row_get(row, "family", "")) or factor in str(_row_get(row, "name", ""))
        ]
        same_values = [value for value in same_values if value is not None]
        trajectory_values = [
            _optional_float(_row_get(row, "salience", _row_get(row, "score", None)))
            for row in trajectory_rows
            if factor in str(_row_get(row, "label", "")) or factor in str(_row_get(row, "band", ""))
        ]
        trajectory_values = [value for value in trajectory_values if value is not None]
        evidence_count = len(factor_flow) + len(same_values) + len(trajectory_values)
        out.append(
            {
                "factor": factor,
                "prompt_count": len(prompts),
                "flow_loss_mean": _mean_or_none(flow_loss_values),
                "same_delta_mean": _mean_or_none(same_values),
                "trajectory_score_mean": _mean_or_none(trajectory_values),
                "evidence_count": evidence_count,
                "maturity_hint": "candidate" if evidence_count >= 2 else "needs_more_evidence",
            }
        )
    return out


def tensor_tree_distance(a: Any, b: Any) -> dict[str, float | int | None]:
    """Flatten tensor-like condition objects and compute simple geometry."""

    va = _flatten_tensor_tree(a)
    vb = _flatten_tensor_tree(b)
    n = min(va.size, vb.size)
    if n == 0:
        return {
            "size": 0,
            "distance": None,
            "normalized_distance": None,
            "cosine_similarity": None,
        }
    va = va[:n].astype(np.float64, copy=False)
    vb = vb[:n].astype(np.float64, copy=False)
    diff = va - vb
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    distance = float(np.linalg.norm(diff))
    cosine = float(np.dot(va, vb) / max(norm_a * norm_b, 1e-12))
    return {
        "size": int(n),
        "distance": distance,
        "normalized_distance": distance / max((norm_a + norm_b) * 0.5, 1e-12),
        "cosine_similarity": cosine,
    }


def condition_geometry_rows(named_conditions: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pairwise geometry for prompt-condition tensors or soft-prompt states."""

    rows: list[dict[str, Any]] = []
    for name_a, name_b in combinations(named_conditions.keys(), 2):
        metrics = tensor_tree_distance(named_conditions[name_a], named_conditions[name_b])
        rows.append({"name_a": name_a, "name_b": name_b, **metrics})
    return rows


def sampler_step_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize recorded sampler-path metadata."""

    rows = list(records)
    sigmas = [_optional_float(_row_get(row, "sigma", None)) for row in rows]
    logsnrs = [_optional_float(_row_get(row, "logsnr", None)) for row in rows]
    timesteps = [_optional_float(_row_get(row, "timestep", _row_get(row, "t", None))) for row in rows]
    sigmas = [value for value in sigmas if value is not None]
    logsnrs = [value for value in logsnrs if value is not None]
    timesteps = [value for value in timesteps if value is not None]
    return {
        "step_record_count": len(rows),
        "sigma_min": min(sigmas) if sigmas else None,
        "sigma_max": max(sigmas) if sigmas else None,
        "logsnr_min": min(logsnrs) if logsnrs else None,
        "logsnr_max": max(logsnrs) if logsnrs else None,
        "timestep_min": min(timesteps) if timesteps else None,
        "timestep_max": max(timesteps) if timesteps else None,
    }


def sampler_physiology_row(
    *,
    name: str,
    sampler_type: str,
    steps: int,
    init_noise_level: float | None = None,
    cfg_scale: float | None = None,
    step_records: Iterable[Mapping[str, Any]] | None = None,
    source_latents: Any | None = None,
    output_latents: Any | None = None,
) -> dict[str, Any]:
    """Build one sampler physiology comparison row."""

    row: dict[str, Any] = {
        "name": name,
        "sampler_type": sampler_type,
        "steps": steps,
        "init_noise_level": init_noise_level,
        "cfg_scale": cfg_scale,
    }
    row.update(sampler_step_summary(step_records or []))
    if source_latents is not None and output_latents is not None:
        report = dict(latent_change_report(source_latents, output_latents))
        row["output_delta_rms"] = report.get("delta_rms")
        row["output_cosine_similarity"] = report.get("cosine_similarity")
    return row


def _apply_channel_dropout(latents: Any, *, fraction: float = 0.1, seed: int = 0, fill: str = "mean") -> Any:
    import torch

    x = latents.clone()
    channel_dim = -2 if x.ndim >= 2 else 0
    channels = x.shape[channel_dim]
    count = max(1, min(channels, int(round(channels * float(fraction)))))
    generator = torch.Generator(device=x.device)
    generator.manual_seed(int(seed))
    idx = torch.randperm(channels, generator=generator, device=x.device)[:count]
    if fill == "zero":
        fill_values = torch.zeros_like(x.index_select(channel_dim, idx))
    elif fill == "mean":
        fill_values = x.mean(dim=channel_dim, keepdim=True).expand_as(x.index_select(channel_dim, idx))
    else:
        raise ValueError(f"Unknown channel dropout fill mode: {fill!r}")
    return x.index_copy(channel_dim, idx, fill_values)


def _apply_gaussian_noise(latents: Any, *, amount: float = 0.02, seed: int = 0) -> Any:
    import torch

    generator = torch.Generator(device=latents.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
    return latents + float(amount) * noise


def _flatten_tensor_tree(obj: Any) -> np.ndarray:
    if obj is None:
        return np.empty((0,), dtype=np.float64)
    if hasattr(obj, "detach") and hasattr(obj, "reshape"):
        return obj.detach().float().cpu().reshape(-1).numpy()
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float64, copy=False).reshape(-1)
    if isinstance(obj, Mapping):
        parts = [_flatten_tensor_tree(obj[key]) for key in sorted(obj.keys(), key=str)]
        return _concat_flat(parts)
    if isinstance(obj, (list, tuple)):
        parts = [_flatten_tensor_tree(value) for value in obj]
        return _concat_flat(parts)
    if isinstance(obj, (int, float, np.number)):
        return np.array([float(obj)], dtype=np.float64)
    return np.empty((0,), dtype=np.float64)


def _concat_flat(parts: Sequence[np.ndarray]) -> np.ndarray:
    parts = [part.reshape(-1) for part in parts if part.size]
    if not parts:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(parts).astype(np.float64, copy=False)


def _row_get(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    return row.get(key, default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(result):
        return None
    return result


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _severity_hint(rms_delta: Any, cosine_similarity: Any) -> str:
    rms = _optional_float(rms_delta) or 0.0
    cosine = _optional_float(cosine_similarity)
    if cosine is None:
        cosine = 1.0
    if rms < 0.02 and cosine > 0.995:
        return "tiny"
    if rms < 0.10 and cosine > 0.95:
        return "small"
    if rms < 0.30 and cosine > 0.80:
        return "medium"
    return "large"


def _infer_prompt_axis(prompt: str) -> str:
    text = prompt.lower()
    if any(token in text for token in ("rhythm", "pulse", "groove", "syncop")):
        return "rhythm"
    if any(token in text for token in ("timbre", "bright", "warm", "metallic", "noisy")):
        return "timbre"
    if any(token in text for token in ("melody", "harmony", "tonal", "atonal", "drone")):
        return "melody_harmony"
    if any(token in text for token in ("dense", "sparse", "layer", "single")):
        return "density"
    return "general"


def _logsnr_band(logsnr: float | None) -> str:
    if logsnr is None:
        return "unknown"
    if logsnr < -2.0:
        return "high_noise"
    if logsnr <= 2.0:
        return "mid_trajectory"
    return "low_noise"
