from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from latent_audio_primitives.latent_blur import LatentBlurSpec, apply_latent_blur
from latent_audio_primitives.latent_dsp import LatentDSPSpec, apply_latent_dsp
from latent_audio_primitives.looping import cyclic_mix_latents, cyclic_roll_latents
from latent_audio_primitives.selective_renoise import (
    LatentMaskSpec,
    graft_latent_channels,
    masked_latent_noise,
    select_latent_channels,
)

from .contracts import (
    ArtifactKind,
    BackendName,
    ModelStatus,
    OperatorName,
    OperatorSpec,
    Recipe,
)
from .ids import new_id
from .jobs import JobContext, JobResult
from .storage import ArtifactStore


SCRIPT_EXPERIMENT_OPERATORS = {
    OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS,
    OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE,
    OperatorName.EXPERIMENT_STYLE_PROFILE_BUILD,
    OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE,
    OperatorName.EXPERIMENT_STYLE_DIRECTION_GENERATE,
    OperatorName.EXPERIMENT_AUDIO_DIRECTION_GENERATE,
    OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT,
    OperatorName.EXPERIMENT_AUDIO_RESIDUAL_VECTORS_EXTRACT,
    OperatorName.EXPERIMENT_ALPHA_SWEEP,
    OperatorName.EXPERIMENT_SOFT_PROMPT_OPTIMIZE,
    OperatorName.EXPERIMENT_SOFT_PROMPT_GENERATE,
    OperatorName.DATASET_PRE_ENCODE,
    OperatorName.TRAIN_LORA,
}

PROGRESS_PERCENT_RE = re.compile(r"(?<![\d.])(\d{1,3})(?:\.\d+)?%")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


class RuntimeDispatcher:
    def __init__(self, store: ArtifactStore, *, repo_root: str | Path | None = None) -> None:
        self.store = store
        env_repo_root = os.environ.get("SA3_REPO_ROOT")
        self.repo_root = (
            Path(repo_root)
            if repo_root is not None
            else Path(env_repo_root)
            if env_repo_root
            else Path(__file__).resolve().parents[2]
        )
        self.mlx_dir = self.repo_root / "optimized" / "mlx"
        self.mlx_wrapper = self.mlx_dir / "sa3"
        self._same_adapters: dict[tuple[str, str], Any] = {}

    def backend_statuses(self) -> list[ModelStatus]:
        return [self._mlx_status(), self._torch_status(), self._cpu_status()]

    def operator_specs(self) -> list[OperatorSpec]:
        return [
            OperatorSpec(
                name=OperatorName.TEXT_TO_AUDIO,
                maturity="core",
                backends=[BackendName.MLX],
                inputs=[],
                params={
                    "prompt": "str",
                    "negative_prompt": "str|null",
                    "duration_seconds": "float",
                    "steps": "int",
                    "seed": "int|null",
                    "cfg_scale": "float",
                    "apg_scale": "float",
                    "model": "sm-music|sm-sfx|medium",
                    "decoder": "same-s|same-l|null",
                },
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.AUDIO_TO_AUDIO,
                maturity="core",
                backends=[BackendName.MLX],
                inputs=["source"],
                params={"prompt": "str", "init_noise_level": "float"},
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.INPAINT,
                maturity="core",
                backends=[BackendName.MLX],
                inputs=["source"],
                params={"prompt": "str", "inpaint_start_seconds": "float", "inpaint_end_seconds": "float"},
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_BLUR,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source"],
                params={"mode": "temporal|channel|low_rank|sharpen|fft_lowpass|...", "strength": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_DSP,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source", "donor?"],
                params={"mode": "gain|compress|softclip|fft_eq|phase_blend|...", "strength": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_GRAFT,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source", "donor"],
                params={"mode": "random_channels|high_variance|...", "fraction": "float", "amount": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_RENOISE,
                maturity="lab",
                backends=[BackendName.TORCH_CPU],
                inputs=["source"],
                params={"mode": "random_channels|high_variance|...", "fraction": "float", "sigma": "float"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_CYCLIC_ROLL,
                maturity="lab",
                backends=[BackendName.TORCH_CPU, BackendName.TORCH_MPS],
                inputs=["source"],
                params={"shift_frames": "int", "strength": "float|null", "symmetric": "bool"},
                produces=[ArtifactKind.LATENT],
                status="implemented for .npy latent artifacts",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_ENCODE,
                maturity="core",
                backends=[BackendName.TORCH_MPS],
                inputs=["source"],
                params={
                    "model": "same-s|same-l",
                    "chunked": "bool",
                    "chunk_size": "int",
                    "overlap": "int",
                    "prompt": "str|null",
                    "notes": "str|null",
                },
                produces=[ArtifactKind.LATENT],
                status="implemented",
            ),
            OperatorSpec(
                name=OperatorName.LATENT_DECODE,
                maturity="core",
                backends=[BackendName.TORCH_MPS],
                inputs=["source"],
                params={
                    "model": "same-s|same-l",
                    "chunked": "bool",
                    "chunk_size": "int",
                    "overlap": "int",
                    "notes": "str|null",
                },
                produces=[ArtifactKind.AUDIO],
                status="implemented",
            ),
            *self._script_operator_specs(),
        ]

    def _script_operator_specs(self) -> list[OperatorSpec]:
        return [
            OperatorSpec(
                name=OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"positive_path": "folder", "negative_path": "folder", "model": "same-s|same-l", "limit": "int"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"input_path": "folder", "model": "same-s|same-l", "name": "str"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_STYLE_PROFILE_BUILD,
                maturity="lab",
                backends=[BackendName.CPU],
                inputs=[],
                params={"target_memory_path": "folder", "reference_memory_path": "folder|null", "name": "str"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"profile_path": "file", "prompt": "str", "alpha": "float", "duration_seconds": "float"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_STYLE_DIRECTION_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"direction_path": "file", "prompt": "str", "alpha": "float", "std_alpha": "float"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_AUDIO_DIRECTION_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"direction_path": "file", "prompt": "str", "alpha": "float"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT,
                maturity="probe",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"axis": "str|all", "num_pairs": "int", "layers": "csv"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_AUDIO_RESIDUAL_VECTORS_EXTRACT,
                maturity="probe",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"positive_path": "folder", "negative_path": "folder|null", "baseline": "prompt|negative_audio"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_ALPHA_SWEEP,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"vectors_path": "folder|file", "prompt": "str", "alphas": "csv", "layer": "int"},
                produces=[ArtifactKind.AUDIO, ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_SOFT_PROMPT_OPTIMIZE,
                maturity="probe",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=["source?"],
                params={"target_audio_path": "file|null", "seed_prompt": "str", "optimization_steps": "int"},
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.EXPERIMENT_SOFT_PROMPT_GENERATE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={"soft_prompt_path": "file", "steps": "int", "seed": "int"},
                produces=[ArtifactKind.AUDIO],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.DATASET_PRE_ENCODE,
                maturity="lab",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={
                    "data_dir": "folder",
                    "model": "same-s|same-l",
                    "batch_size": "int",
                    "sample_size": "int",
                    "pad": "bool",
                    "model_half": "bool",
                },
                produces=[ArtifactKind.BUNDLE],
                status="script adapter",
            ),
            OperatorSpec(
                name=OperatorName.TRAIN_LORA,
                maturity="danger",
                backends=[BackendName.TORCH_MPS, BackendName.TORCH_CPU],
                inputs=[],
                params={
                    "encoded_dir": "folder",
                    "data_dir": "folder|null",
                    "steps": "int",
                    "rank": "int",
                    "lora_alpha": "int",
                    "adapter_type": "lora|lora-xs",
                    "dropout": "float",
                    "logger": "wandb|comet|csv|none",
                    "include": "patterns",
                    "exclude": "patterns",
                },
                produces=[ArtifactKind.BUNDLE],
                status="script adapter; long-running training",
            ),
        ]

    def handler_for_recipe(self, recipe: Recipe):
        if recipe.operator in {
            OperatorName.TEXT_TO_AUDIO,
            OperatorName.AUDIO_TO_AUDIO,
            OperatorName.INPAINT,
        }:
            return lambda context: self._run_mlx_generation(recipe, context)
        if recipe.operator in {
            OperatorName.LATENT_BLUR,
            OperatorName.LATENT_DSP,
            OperatorName.LATENT_GRAFT,
            OperatorName.LATENT_RENOISE,
            OperatorName.LATENT_CYCLIC_ROLL,
        }:
            return lambda context: self._run_latent_operator(recipe, context)
        if recipe.operator == OperatorName.LATENT_ENCODE:
            return lambda context: self._run_same_encode(recipe, context)
        if recipe.operator == OperatorName.LATENT_DECODE:
            return lambda context: self._run_same_decode(recipe, context)
        if recipe.operator in SCRIPT_EXPERIMENT_OPERATORS:
            return lambda context: self._run_script_experiment(recipe, context)
        raise ValueError(f"operator is not implemented yet: {recipe.operator}")

    def _mlx_status(self) -> ModelStatus:
        apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        venv_python = self.mlx_dir / ".venv" / "bin" / "python"
        available = apple_silicon and self.mlx_wrapper.exists() and venv_python.exists()
        if not apple_silicon:
            message = "MLX backend requires Apple Silicon"
        elif not self.mlx_wrapper.exists():
            message = "optimized/mlx/sa3 wrapper is missing"
        elif not venv_python.exists():
            message = "optimized/mlx/.venv is missing; run optimized/mlx/install.sh"
        else:
            message = "MLX wrapper is ready"
        weights_dir = self.mlx_dir / "models" / "mlx"
        return ModelStatus(
            backend=BackendName.MLX,
            available=available,
            loaded=False,
            device="metal" if apple_silicon else None,
            message=message,
            details={
                "wrapper": str(self.mlx_wrapper),
                "venv_python": str(venv_python),
                "weights_dir": str(weights_dir),
                "weights_dir_exists": weights_dir.exists(),
            },
        )

    def _torch_status(self) -> ModelStatus:
        try:
            import torch

            mps_available = bool(torch.backends.mps.is_available())
            cuda_available = bool(torch.cuda.is_available())
            if mps_available:
                device = "mps"
                available = True
            elif cuda_available:
                device = "cuda"
                available = True
            else:
                device = "cpu"
                available = True
            return ModelStatus(
                backend=BackendName.TORCH_MPS,
                available=available,
                loaded=False,
                device=device,
                message="Torch is importable",
                details={
                    "torch_version": torch.__version__,
                    "mps_available": mps_available,
                    "cuda_available": cuda_available,
                },
            )
        except Exception as exc:
            return ModelStatus(
                backend=BackendName.TORCH_MPS,
                available=False,
                loaded=False,
                message=str(exc),
            )

    def _cpu_status(self) -> ModelStatus:
        return ModelStatus(
            backend=BackendName.CPU,
            available=True,
            loaded=True,
            device="cpu",
            message="CPU artifact, metadata, and latent-file operations are available",
        )

    def _run_mlx_generation(self, recipe: Recipe, context: JobContext) -> JobResult:
        status = self._mlx_status()
        if not status.available:
            raise RuntimeError(status.message or "MLX backend is not available")

        params = recipe.params
        model = str(recipe.model or params.get("model") or "medium")
        decoder = params.get("decoder")
        if decoder is None:
            decoder = "same-l" if model == "medium" else "same-s"
        prompt = str(params.get("prompt", ""))
        duration = float(params.get("duration_seconds", 30.0))
        steps = int(params.get("steps", 8))
        output_id, output_path = self.store.reserve_artifact_path(filename=f"{_operator_value(recipe.operator)}.wav")

        command = [
            str(self.mlx_wrapper),
            "--prompt",
            prompt,
            "--dit",
            model,
            "--decoder",
            str(decoder),
            "--seconds",
            str(duration),
            "--steps",
            str(steps),
            "--cfg",
            str(float(params.get("cfg_scale", 1.0))),
            "--apg",
            str(float(params.get("apg_scale", 1.0))),
            "--out",
            str(output_path),
        ]
        seed = recipe.seed if recipe.seed is not None else params.get("seed")
        if seed is not None:
            command.extend(["--seed", str(int(seed))])
        if params.get("negative_prompt"):
            command.extend(["--negative-prompt", str(params["negative_prompt"])])
        if recipe.operator in {OperatorName.AUDIO_TO_AUDIO, OperatorName.INPAINT}:
            source_id = recipe.inputs.get("source")
            if not source_id:
                raise ValueError("audio generation recipe requires inputs.source")
            source = self.store.get_artifact(source_id)
            command.extend(["--init-audio", str(source.path)])
            command.extend(["--init-noise-level", str(float(params.get("init_noise_level", 0.7)))])
        if recipe.operator == OperatorName.INPAINT:
            start = float(params["inpaint_start_seconds"])
            end = float(params["inpaint_end_seconds"])
            command.extend(["--inpaint-range", f"{start},{end}"])

        context.set_progress(0.05, "starting MLX subprocess")
        process = subprocess.Popen(
            command,
            cwd=str(self.mlx_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            _update_subprocess_progress(context, clean)
            context.log(clean)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"MLX generation failed with exit code {return_code}")
        if not output_path.exists():
            raise RuntimeError(f"MLX generation did not create output file: {output_path}")

        source_ids = [recipe.inputs["source"]] if "source" in recipe.inputs else []
        artifact = self.store.finalize_audio_file(
            artifact_id=output_id,
            path=output_path,
            recipe=recipe,
            source_artifact_ids=source_ids,
            prompt=prompt,
            metadata={
                "backend": BackendName.MLX.value,
                "model": model,
                "decoder": decoder,
                "steps": steps,
                "duration_seconds": duration,
            },
        )
        return JobResult(artifact_ids=[artifact.artifact_id], metrics={"return_code": return_code})

    def _run_latent_operator(self, recipe: Recipe, context: JobContext) -> JobResult:
        source_id = recipe.inputs.get("source")
        if not source_id:
            raise ValueError("latent operator recipe requires inputs.source")
        source = self.store.get_artifact(source_id)
        if source.latent is None:
            raise ValueError(f"source artifact {source_id} does not have latent metadata")
        source_array = self.store.load_latent_array(source_id)
        x = _time_major_to_bct(source_array)
        params = dict(recipe.params)
        metadata: dict[str, Any] = {
            "backend": recipe.backend.value if isinstance(recipe.backend, BackendName) else str(recipe.backend),
            "operator": _operator_value(recipe.operator),
        }

        context.set_progress(0.15, f"running {recipe.operator}")
        if recipe.operator == OperatorName.LATENT_BLUR:
            spec = LatentBlurSpec(**_dataclass_kwargs(LatentBlurSpec, {"name": "api", **params}))
            result = apply_latent_blur(x, spec)
            metadata["spec"] = _clean_params(params)
        elif recipe.operator == OperatorName.LATENT_DSP:
            spec = LatentDSPSpec(**_dataclass_kwargs(LatentDSPSpec, {"name": "api", **params}))
            donor = self._load_optional_donor(recipe)
            result = apply_latent_dsp(x, spec, donor_latents=donor)
            metadata["spec"] = _clean_params(params)
        elif recipe.operator == OperatorName.LATENT_GRAFT:
            donor = self._load_required_donor(recipe)
            mask = LatentMaskSpec(**_dataclass_kwargs(LatentMaskSpec, {"name": "api", **params}))
            channels = select_latent_channels(x, mask)
            result = graft_latent_channels(x, donor, channels, amount=float(params.get("amount", 1.0)))
            metadata["selected_channels"] = channels
        elif recipe.operator == OperatorName.LATENT_RENOISE:
            mask = LatentMaskSpec(**_dataclass_kwargs(LatentMaskSpec, {"name": "api", **params}))
            channels = select_latent_channels(x, mask)
            result = masked_latent_noise(
                x,
                channels,
                sigma=float(params.get("sigma", params.get("amount", 0.4))),
                seed=int(recipe.seed if recipe.seed is not None else params.get("seed", 0)),
            )
            metadata["selected_channels"] = channels
        elif recipe.operator == OperatorName.LATENT_CYCLIC_ROLL:
            shift_frames = int(params.get("shift_frames", 0))
            strength = params.get("strength")
            if strength is None:
                result = cyclic_roll_latents(x, shift_frames)
            else:
                result = cyclic_mix_latents(
                    x,
                    shift_frames,
                    strength=float(strength),
                    symmetric=bool(params.get("symmetric", True)),
                )
        else:
            raise ValueError(f"unsupported latent operator: {recipe.operator}")

        output = _bct_to_time_major(result)
        context.set_progress(0.75, "saving latent artifact")
        artifact = self.store.store_latent_array(
            output,
            latent_rate=source.latent.latent_rate,
            recipe=recipe,
            source_artifact_ids=[item for item in [source_id, recipe.inputs.get("donor")] if item],
            prompt=source.prompt,
            metadata=metadata,
        )
        return JobResult(artifact_ids=[artifact.artifact_id])

    def _run_same_encode(self, recipe: Recipe, context: JobContext) -> JobResult:
        source_id = recipe.inputs.get("source")
        if not source_id:
            raise ValueError("latent encode recipe requires inputs.source")
        source = self.store.get_artifact(source_id)
        if source.kind != ArtifactKind.AUDIO or source.audio is None:
            raise ValueError(f"source artifact {source_id} is not an audio artifact")

        params = dict(recipe.params)
        model_name = str(recipe.model or params.get("model") or "same-l")
        adapter = self._same_adapter(model_name, recipe.backend)
        context.set_progress(0.20, f"loaded SAME encoder {model_name}")

        item = adapter.encode_file(
            source.path,
            item_id=source.artifact_id,
            prompt=params.get("prompt") or source.prompt,
            chunked=bool(params.get("chunked", False)),
            metadata={
                "source_artifact_id": source_id,
                "source_path": str(source.path),
                "backend": _backend_value(recipe.backend),
            },
            chunk_size=int(params.get("chunk_size", 128)),
            overlap=int(params.get("overlap", 32)),
        )
        context.set_progress(0.75, "saving encoded latent")
        artifact = self.store.store_latent_array(
            item.latent,
            latent_rate=item.latent_rate,
            sample_rate=item.sample_rate,
            recipe=recipe,
            source_artifact_ids=[source_id],
            prompt=item.prompt,
            metadata=item.metadata,
        )
        return JobResult(artifact_ids=[artifact.artifact_id])

    def _run_same_decode(self, recipe: Recipe, context: JobContext) -> JobResult:
        source_id = recipe.inputs.get("source")
        if not source_id:
            raise ValueError("latent decode recipe requires inputs.source")
        source = self.store.get_artifact(source_id)
        if source.latent is None:
            raise ValueError(f"source artifact {source_id} does not have latent metadata")

        params = dict(recipe.params)
        model_name = str(recipe.model or params.get("model") or "same-l")
        adapter = self._same_adapter(model_name, recipe.backend)
        context.set_progress(0.20, f"loaded SAME decoder {model_name}")

        latents = _time_major_to_bct(self.store.load_latent_array(source_id))
        device = getattr(adapter.autoencoder, "device", None)
        if device is not None:
            latents = latents.to(str(device))
        audio = adapter.decode_latents(
            latents,
            chunked=bool(params.get("chunked", False)),
            chunk_size=int(params.get("chunk_size", 128)),
            overlap=int(params.get("overlap", 32)),
        )
        output_id, output_path = self.store.reserve_artifact_path(filename=f"{source_id}.decoded.wav")
        _write_audio_tensor(output_path, audio, sample_rate=int(adapter.sample_rate))
        context.set_progress(0.80, "saving decoded audio")
        artifact = self.store.finalize_audio_file(
            artifact_id=output_id,
            path=output_path,
            recipe=recipe,
            source_artifact_ids=[source_id],
            prompt=source.prompt,
            metadata={
                "backend": _backend_value(recipe.backend),
                "model": model_name,
                "source_artifact_id": source_id,
            },
        )
        return JobResult(artifact_ids=[artifact.artifact_id])

    def _run_script_experiment(self, recipe: Recipe, context: JobContext) -> JobResult:
        output_id, marker_path = self.store.reserve_artifact_path(filename=f"{_operator_slug(recipe.operator)}.zip")
        output_root = marker_path.parent / "script_output"
        output_root.mkdir(parents=True, exist_ok=True)
        command, primary_output = self._script_experiment_command(recipe, output_root)

        context.set_progress(0.05, f"starting {_operator_value(recipe.operator)}")
        return_code = self._run_subprocess(command, cwd=self.repo_root, context=context)
        if return_code != 0:
            raise RuntimeError(f"{recipe.operator} failed with exit code {return_code}")
        if not primary_output.exists():
            raise RuntimeError(f"{recipe.operator} did not create expected output: {primary_output}")

        context.set_progress(0.85, "registering experiment artifacts")
        artifact_ids = self._register_script_outputs(
            recipe,
            primary_output=primary_output,
            bundle_artifact_id=output_id,
        )
        return JobResult(
            artifact_ids=artifact_ids,
            metrics={
                "return_code": return_code,
                "primary_output": str(primary_output),
                "command": _redacted_command(command),
            },
        )

    def _script_experiment_command(self, recipe: Recipe, output_root: Path) -> tuple[list[str], Path]:
        params = dict(recipe.params)
        model = str(recipe.model or params.get("model") or _default_script_model(recipe.operator))
        seed = recipe.seed if recipe.seed is not None else params.get("seed")

        def script(name: str) -> list[str]:
            return [sys.executable, str(self.repo_root / "scripts" / name)]

        def torch_common(command: list[str]) -> None:
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            if bool(params.get("no_half", False)):
                command.append("--no-half")

        if recipe.operator == OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS:
            output = output_root / "audio_style_vectors"
            command = script("extract_audio_style_vectors.py")
            _append_required(command, "--positive", params, "positive_path")
            _append_required(command, "--negative", params, "negative_path")
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            _append_bool(command, "--chunked", params.get("chunked"))
            _append_int(command, "--limit", params.get("limit"))
            _append_option(command, "--name", params.get("name"))
            _append_bool(command, "--normalize-frame", params.get("normalize_frame"))
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE:
            output = output_root / "positive_style"
            memory = output / "memory"
            profile = output / "profile.npz"
            command = script("build_positive_audio_style_profile.py")
            _append_required(command, "--input", params, "input_path")
            command.extend(["--memory-output", str(memory), "--profile-output", str(profile), "--model", model])
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            _append_bool(command, "--chunked", params.get("chunked"))
            _append_int(command, "--limit", params.get("limit"))
            _append_option(command, "--name", params.get("name"))
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_STYLE_PROFILE_BUILD:
            output = output_root / "style_profile"
            profile = output / "profile.npz"
            direction = output / "direction.npz"
            command = script("build_same_style_profile.py")
            _append_required(command, "--target", params, "target_memory_path")
            command.extend(["--output", str(profile)])
            _append_option(command, "--name", params.get("name"))
            if params.get("reference_memory_path"):
                command.extend(["--reference", str(params["reference_memory_path"]), "--direction-output", str(direction)])
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_STYLE_PROFILE_GENERATE:
            output = output_root / "styled_profile.wav"
            command = script("generate_sa3_with_style_profile.py")
            _append_required(command, "--profile", params, "profile_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--alpha", params.get("alpha"))
            if params.get("match_std") is False or params.get("no_std"):
                command.append("--no-std")
            _append_bool(command, "--save-original", params.get("save_original"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_STYLE_DIRECTION_GENERATE:
            output = output_root / "styled_direction.wav"
            command = script("generate_sa3_with_style_direction.py")
            _append_required(command, "--direction", params, "direction_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--alpha", params.get("alpha"))
            _append_float(command, "--std-alpha", params.get("std_alpha"))
            _append_bool(command, "--save-original", params.get("save_original"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_AUDIO_DIRECTION_GENERATE:
            output = output_root / "audio_direction.wav"
            command = script("generate_sa3_with_audio_direction.py")
            _append_required(command, "--direction", params, "direction_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--alpha", params.get("alpha"))
            _append_bool(command, "--save-original", params.get("save_original"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_SA3_VECTORS_EXTRACT:
            output = output_root / "sa3_vectors"
            command = script("extract_sa3_vectors.py")
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--axis", params.get("axis"))
            _append_int(command, "--num-pairs", params.get("num_pairs"))
            _append_generation_params(command, params, seed=seed)
            _append_option(command, "--layers", params.get("layers"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_AUDIO_RESIDUAL_VECTORS_EXTRACT:
            output = output_root / "audio_residual_vectors"
            command = script("extract_audio_residual_vectors.py")
            _append_required(command, "--positive", params, "positive_path")
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--negative", params.get("negative_path"))
            _append_option(command, "--baseline", params.get("baseline"))
            _append_prompt(command, params)
            _append_generation_params(command, params, seed=seed)
            _append_float(command, "--init-noise-level", params.get("init_noise_level"))
            _append_option(command, "--layers", params.get("layers"))
            _append_int(command, "--limit", params.get("limit"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_ALPHA_SWEEP:
            output = output_root / "alpha_sweep"
            command = script("run_sa3_alpha_sweep.py")
            _append_required(command, "--vectors", params, "vectors_path")
            _append_prompt(command, params)
            command.extend(["--output", str(output), "--model", model])
            _append_option(command, "--alphas", params.get("alphas"))
            _append_generation_params(command, params, seed=seed)
            _append_int(command, "--layer", params.get("layer"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_SOFT_PROMPT_OPTIMIZE:
            output = output_root / "soft_prompt.pt"
            target_audio = params.get("target_audio_path")
            if not target_audio and recipe.inputs.get("source"):
                target_audio = str(self.store.get_artifact(recipe.inputs["source"]).path)
            if not target_audio:
                raise ValueError("soft prompt optimization requires params.target_audio_path or inputs.source")
            command = script("optimize_sa3_soft_prompt.py")
            command.extend(["--target-audio", str(target_audio), "--output", str(output), "--model", model])
            _append_option(command, "--seed-prompt", params.get("seed_prompt"))
            _append_float(command, "--duration", params.get("duration_seconds", params.get("duration")))
            _append_int(command, "--optimization-steps", params.get("optimization_steps"))
            _append_float(command, "--lr", params.get("lr"))
            _append_float(command, "--reg-weight", params.get("reg_weight"))
            _append_int(command, "--seed", seed)
            _append_option(command, "--train-keys", params.get("train_keys"))
            _append_option(command, "--velocity-convention", params.get("velocity_convention"))
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.EXPERIMENT_SOFT_PROMPT_GENERATE:
            output = output_root / "soft_prompt.wav"
            command = script("generate_sa3_with_soft_prompt.py")
            _append_required(command, "--soft-prompt", params, "soft_prompt_path")
            command.extend(["--output", str(output), "--model", model])
            _append_int(command, "--steps", params.get("steps"))
            _append_float(command, "--cfg-scale", params.get("cfg_scale"))
            _append_int(command, "--seed", seed)
            torch_common(command)
            return command, output

        if recipe.operator == OperatorName.DATASET_PRE_ENCODE:
            output = output_root / "encoded_dataset"
            command = script("pre_encode_dataset.py")
            _append_required(command, "--data_dir", params, "data_dir")
            command.extend(["--output_path", str(output), "--model", model])
            _append_int(command, "--batch_size", params.get("batch_size"))
            _append_int(command, "--sample_size", params.get("sample_size"))
            _append_bool(command, "--model_half", params.get("model_half"))
            _append_option(command, "--device", params.get("device") or _device_for_backend(recipe.backend))
            _append_bool(command, "--pad", params.get("pad"))
            return command, output

        if recipe.operator == OperatorName.TRAIN_LORA:
            output = output_root / "lora_checkpoints"
            command = script("train_lora.py")
            command.extend(["--save_dir", str(output), "--model", model])
            if params.get("encoded_dir"):
                command.extend(["--encoded_dir", str(params["encoded_dir"])])
            elif params.get("data_dir"):
                command.extend(["--data_dir", str(params["data_dir"])])
            else:
                raise ValueError("LoRA training requires params.encoded_dir or params.data_dir")
            for flag, key in [
                ("--rank", "rank"),
                ("--lora_alpha", "lora_alpha"),
                ("--adapter_type", "adapter_type"),
                ("--dropout", "dropout"),
                ("--svd_bases_path", "svd_bases_path"),
                ("--base_precision", "base_precision"),
                ("--lora_checkpoint", "lora_checkpoint"),
                ("--lr", "lr"),
                ("--steps", "steps"),
                ("--batch_size", "batch_size"),
                ("--duration", "duration_seconds"),
                ("--device", "device"),
                ("--logger", "logger"),
                ("--name", "name"),
                ("--checkpoint_every", "checkpoint_every"),
                ("--log_every", "log_every"),
                ("--demo_every", "demo_every"),
                ("--num_workers", "num_workers"),
            ]:
                _append_option(command, flag, params.get(key))
            _append_multi_option(command, "--include", params.get("include"))
            _append_multi_option(command, "--exclude", params.get("exclude"))
            _append_int(command, "--seed", seed)
            if not params.get("device"):
                _append_option(command, "--device", _device_for_backend(recipe.backend))
            return command, output

        raise ValueError(f"unsupported script experiment operator: {recipe.operator}")

    def _run_subprocess(self, command: list[str], *, cwd: Path, context: JobContext) -> int:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip()
            _update_subprocess_progress(context, clean)
            context.log(clean)
        return process.wait()

    def _register_script_outputs(self, recipe: Recipe, *, primary_output: Path, bundle_artifact_id: str) -> list[str]:
        source_ids = [value for value in recipe.inputs.values() if value]
        audio_paths = []
        if primary_output.is_dir():
            audio_paths = [path for path in sorted(primary_output.rglob("*")) if path.suffix.lower() in AUDIO_EXTENSIONS]
        elif primary_output.suffix.lower() in AUDIO_EXTENSIONS:
            audio_paths = [primary_output]

        audio_artifact_ids: list[str] = []
        for audio_path in audio_paths:
            artifact = self.store.finalize_audio_file(
                artifact_id=new_id("art"),
                path=audio_path,
                recipe=recipe,
                source_artifact_ids=source_ids,
                prompt=str(recipe.params.get("prompt", "")) or None,
                label=audio_path.stem,
                metadata={
                    "backend": _backend_value(recipe.backend),
                    "operator": _operator_value(recipe.operator),
                    "script_output_path": str(audio_path),
                },
            )
            audio_artifact_ids.append(artifact.artifact_id)

        if primary_output.is_dir() or not audio_paths:
            bundle = self.store.finalize_bundle_path(
                artifact_id=bundle_artifact_id,
                path=primary_output,
                recipe=recipe,
                source_artifact_ids=source_ids,
                label=_operator_value(recipe.operator),
                prompt=str(recipe.params.get("prompt", "")) or None,
                metadata={
                    "backend": _backend_value(recipe.backend),
                    "operator": _operator_value(recipe.operator),
                    "script_output_path": str(primary_output),
                },
            )
            return [*audio_artifact_ids, bundle.artifact_id]

        return audio_artifact_ids

    def _load_optional_donor(self, recipe: Recipe):
        donor_id = recipe.inputs.get("donor")
        if not donor_id:
            return None
        return _time_major_to_bct(self.store.load_latent_array(donor_id))

    def _load_required_donor(self, recipe: Recipe):
        donor = self._load_optional_donor(recipe)
        if donor is None:
            raise ValueError("operator requires inputs.donor")
        return donor

    def _same_adapter(self, model_name: str, backend: BackendName | str):
        from latent_audio_primitives.adapters.stable_audio3 import SAMEAutoencoderAdapter

        device = _device_for_backend(backend)
        key = (model_name, device)
        if key not in self._same_adapters:
            self._same_adapters[key] = SAMEAutoencoderAdapter.from_pretrained(model_name, device=device)
        return self._same_adapters[key]


def _time_major_to_bct(array: np.ndarray):
    import torch

    latent = np.asarray(array, dtype=np.float32)
    if latent.ndim != 2:
        raise ValueError(f"latent must be 2D time-major, got shape {latent.shape}")
    return torch.from_numpy(latent.T.copy()).unsqueeze(0)


def _bct_to_time_major(tensor: Any) -> np.ndarray:
    detached = tensor.detach().cpu().float()
    if detached.ndim == 3:
        detached = detached.squeeze(0)
    if detached.ndim != 2:
        raise ValueError(f"operator produced invalid latent shape {tuple(detached.shape)}")
    return detached.numpy().T.astype(np.float32, copy=False)


def _dataclass_kwargs(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in params.items() if key in valid}


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in params.items()}


def _operator_value(operator: OperatorName | str) -> str:
    return operator.value if isinstance(operator, OperatorName) else str(operator)


def _operator_slug(operator: OperatorName | str) -> str:
    return _operator_value(operator).replace(".", "_").replace("/", "_")


def _backend_value(backend: BackendName | str) -> str:
    return backend.value if isinstance(backend, BackendName) else str(backend)


def _device_for_backend(backend: BackendName | str) -> str:
    value = _backend_value(backend)
    if value == BackendName.TORCH_CPU.value:
        return "cpu"
    if value == BackendName.TORCH_MPS.value:
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
    return "cpu"


def _write_audio_tensor(path: Path, audio: Any, *, sample_rate: int) -> None:
    import soundfile as sf

    arr = audio.detach().float().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"decoded audio must be shaped channels x samples, got {arr.shape}")
    arr = np.clip(arr, -1.0, 1.0).T
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sample_rate)


def _update_subprocess_progress(context: JobContext, line: str) -> None:
    match = PROGRESS_PERCENT_RE.search(line)
    if not match:
        return
    percent = float(match.group(1))
    if not 0 <= percent <= 100:
        return
    progress = 0.05 + (percent / 100.0) * 0.80
    context.set_progress(min(0.84, max(0.05, progress)), line[:240])


def _default_script_model(operator: OperatorName | str) -> str:
    if operator in {
        OperatorName.EXPERIMENT_AUDIO_STYLE_VECTORS,
        OperatorName.EXPERIMENT_POSITIVE_STYLE_PROFILE,
        OperatorName.DATASET_PRE_ENCODE,
    }:
        return "same-l"
    if operator == OperatorName.TRAIN_LORA:
        return "medium-base"
    return "medium"


def _append_required(command: list[str], flag: str, params: dict[str, Any], key: str) -> None:
    value = params.get(key)
    if value in {None, ""}:
        raise ValueError(f"missing required parameter: {key}")
    command.extend([flag, str(value)])


def _append_option(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def _append_multi_option(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value if str(item).strip()]
    else:
        parts = [part.strip() for part in str(value).replace(",", " ").split() if part.strip()]
    if parts:
        command.append(flag)
        command.extend(parts)


def _append_bool(command: list[str], flag: str, value: Any) -> None:
    if bool(value):
        command.append(flag)


def _append_int(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(int(value))])


def _append_float(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(float(value))])


def _append_prompt(command: list[str], params: dict[str, Any]) -> None:
    command.extend(["--prompt", str(params.get("prompt", ""))])


def _append_generation_params(command: list[str], params: dict[str, Any], *, seed: Any) -> None:
    _append_float(command, "--duration", params.get("duration_seconds", params.get("duration")))
    _append_int(command, "--steps", params.get("steps"))
    _append_float(command, "--cfg-scale", params.get("cfg_scale"))
    _append_int(command, "--seed", seed)


def _redacted_command(command: list[str]) -> list[str]:
    redacted: list[str] = []
    skip_next = False
    sensitive_flags = {"--token", "--hf-token", "--huggingface-token"}
    for part in command:
        if skip_next:
            redacted.append("<redacted>")
            skip_next = False
            continue
        redacted.append(part)
        if part in sensitive_flags:
            skip_next = True
    return redacted
