from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ids import new_id, utc_now


class ArtifactKind(str, Enum):
    AUDIO = "audio"
    LATENT = "latent"
    BUNDLE = "bundle"
    RECIPE = "recipe"
    TEXT = "text"


class BackendName(str, Enum):
    MLX = "mlx"
    TORCH_MPS = "torch_mps"
    TORCH_CPU = "torch_cpu"
    CPU = "cpu"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobEventType(str, Enum):
    SNAPSHOT = "snapshot"
    ERROR = "error"


class OperatorName(str, Enum):
    TEXT_TO_AUDIO = "generate.text_to_audio"
    AUDIO_TO_AUDIO = "generate.audio_to_audio"
    INPAINT = "generate.inpaint"
    LATENT_ENCODE = "latent.encode"
    LATENT_DECODE = "latent.decode"
    LATENT_BLUR = "latent.blur"
    LATENT_DSP = "latent.dsp"
    LATENT_GRAFT = "latent.graft"
    LATENT_RENOISE = "latent.renoise"
    LATENT_CYCLIC_ROLL = "latent.cyclic_roll"
    EXPERIMENT_AUDIO_STYLE_VECTORS = "experiment.audio_style_vectors"
    EXPERIMENT_POSITIVE_STYLE_PROFILE = "experiment.positive_style_profile"
    EXPERIMENT_STYLE_PROFILE_BUILD = "experiment.style_profile.build"
    EXPERIMENT_STYLE_PROFILE_GENERATE = "experiment.style_profile.generate"
    EXPERIMENT_STYLE_DIRECTION_GENERATE = "experiment.style_direction.generate"
    EXPERIMENT_AUDIO_DIRECTION_GENERATE = "experiment.audio_direction.generate"
    EXPERIMENT_SA3_VECTORS_EXTRACT = "experiment.sa3_vectors.extract"
    EXPERIMENT_AUDIO_RESIDUAL_VECTORS_EXTRACT = "experiment.audio_residual_vectors.extract"
    EXPERIMENT_ALPHA_SWEEP = "experiment.alpha_sweep"
    EXPERIMENT_GEOMETRY_AUDIT = "experiment.geometry_audit"
    EXPERIMENT_SOFT_PROMPT_OPTIMIZE = "experiment.soft_prompt.optimize"
    EXPERIMENT_SOFT_PROMPT_GENERATE = "experiment.soft_prompt.generate"
    DATASET_PRE_ENCODE = "dataset.pre_encode"
    TRAIN_LORA = "training.lora"
    MEMORY_QUERY = "memory.query"
    ARTIFACT_PROMOTE_BUNDLE_AUDIO = "artifact.promote_bundle_audio"
    ANNOTATE = "artifact.annotate"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileInfo(ContractModel):
    filename: str
    media_type: str | None = None
    byte_size: int = 0
    sha256: str | None = None


class AudioMetadata(ContractModel):
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    format: str | None = None


class AudioPeaksResponse(ContractModel):
    artifact_id: str
    bins: int
    channels: int
    sample_rate: int
    duration_seconds: float
    peaks: list[float]


class BundleFileEntry(ContractModel):
    path: str
    byte_size: int
    compressed_size: int | None = None


class BundleAudioEntry(ContractModel):
    path: str
    byte_size: int
    media_type: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    frames: int | None = None
    duration_seconds: float | None = None
    format: str | None = None


class LatentMetadata(ContractModel):
    shape: tuple[int, int]
    latent_rate: float
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channel_first: bool = False

    @field_validator("latent_rate")
    @classmethod
    def latent_rate_must_be_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("latent_rate must be positive")
        return value


class ArtifactRecord(ContractModel):
    artifact_id: str = Field(default_factory=lambda: new_id("art"))
    kind: ArtifactKind
    path: Path
    file: FileInfo | None = None
    audio: AudioMetadata | None = None
    latent: LatentMetadata | None = None
    source_artifact_ids: list[str] = Field(default_factory=list)
    recipe_id: str | None = None
    label: str | None = None
    prompt: str | None = None
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Recipe(ContractModel):
    recipe_id: str = Field(default_factory=lambda: new_id("recipe"))
    operator: OperatorName
    backend: BackendName
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    seed: int | None = None
    notes: str | None = None
    session_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    version: int = 1


class ArtifactInspection(ContractModel):
    artifact: ArtifactRecord
    recipe: Recipe | None = None
    sources: list[ArtifactRecord] = Field(default_factory=list)
    children: list[ArtifactRecord] = Field(default_factory=list)
    bundle_files: list[BundleFileEntry] = Field(default_factory=list)
    bundle_audio_files: list[BundleAudioEntry] = Field(default_factory=list)
    bundle_preview: dict[str, Any] = Field(default_factory=dict)
    bundle_summary: dict[str, Any] = Field(default_factory=dict)


class BundleAudioPromotionRequest(ContractModel):
    path: str
    label: str | None = None
    prompt: str | None = None
    session_id: str | None = None


class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class SessionRecord(ContractModel):
    session_id: str = Field(default_factory=lambda: new_id("sess"))
    name: str
    status: SessionStatus = SessionStatus.ACTIVE
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    archived_at: datetime | None = None


class SessionCreateRequest(ContractModel):
    name: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionUpdateRequest(ContractModel):
    name: str | None = None
    status: SessionStatus | None = None
    notes: str | None = None
    metadata: dict[str, Any] | None = None


class JobRecord(ContractModel):
    job_id: str = Field(default_factory=lambda: new_id("job"))
    status: JobStatus = JobStatus.QUEUED
    recipe: Recipe
    progress: float = 0.0
    message: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("progress")
    @classmethod
    def progress_is_unit_interval(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("progress must be in [0, 1]")
        return value


class JobEvent(ContractModel):
    type: JobEventType = JobEventType.SNAPSHOT
    job: JobRecord


class JobErrorEvent(ContractModel):
    type: JobEventType = JobEventType.ERROR
    error: str


class JobJournalEvent(ContractModel):
    sequence: int
    type: JobEventType = JobEventType.SNAPSHOT
    created_at: datetime = Field(default_factory=utc_now)
    job: JobRecord


class ModelStatus(ContractModel):
    backend: BackendName
    available: bool
    loaded: bool = False
    device: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(ContractModel):
    app: str = "sa3-native-lab"
    version: str
    artifact_root: Path
    backends: list[ModelStatus]


class ReadinessCheck(ContractModel):
    name: str
    status: str
    message: str
    detail: str | None = None


class ReadinessResponse(ContractModel):
    checks: list[ReadinessCheck]
    ok: bool
    warnings: int = 0
    errors: int = 0


class OperatorFieldOption(ContractModel):
    value: str
    label: str | None = None


class OperatorFieldSpec(ContractModel):
    key: str
    label: str
    type: str
    default: Any = None
    required: bool = False
    advanced: bool = False
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[OperatorFieldOption] = Field(default_factory=list)
    artifact_kinds: list[ArtifactKind] = Field(default_factory=list)
    placeholder: str | None = None
    description: str | None = None


class OperatorSpec(ContractModel):
    name: OperatorName
    maturity: str
    backends: list[BackendName]
    inputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    ui_fields: list[OperatorFieldSpec] = Field(default_factory=list)
    produces: list[ArtifactKind] = Field(default_factory=list)
    status: str = "available"


class NotebookMode(ContractModel):
    mode_id: str
    title: str
    priority: str
    maturity: str
    status: str
    native_surface: str
    operators: list[OperatorName] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    notes: str | None = None


class TextGenerateRequest(ContractModel):
    prompt: str = ""
    negative_prompt: str | None = None
    duration_seconds: float = Field(default=30.0, gt=0.0)
    steps: int = Field(default=8, ge=1)
    seed: int | None = None
    cfg_scale: float = 1.0
    apg_scale: float = 1.0
    model: str = "medium"
    decoder: str | None = None
    backend: BackendName = BackendName.MLX
    session_id: str | None = None


class AudioToAudioRequest(TextGenerateRequest):
    source_artifact_id: str
    init_noise_level: float = Field(default=0.7, gt=0.0)


class InpaintRequest(AudioToAudioRequest):
    inpaint_start_seconds: float = Field(ge=0.0)
    inpaint_end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def inpaint_range_must_be_valid(self) -> "InpaintRequest":
        if self.inpaint_start_seconds >= self.inpaint_end_seconds:
            raise ValueError("inpaint_start_seconds must be less than inpaint_end_seconds")
        if self.inpaint_end_seconds > self.duration_seconds:
            raise ValueError("inpaint_end_seconds cannot exceed duration_seconds")
        return self


class LatentEncodeRequest(ContractModel):
    source_artifact_id: str
    model: str = "same-l"
    backend: BackendName = BackendName.TORCH_MPS
    chunked: bool = False
    chunk_size: int = Field(default=128, ge=1)
    overlap: int = Field(default=32, ge=0)
    prompt: str | None = None
    notes: str | None = None
    session_id: str | None = None


class LatentDecodeRequest(ContractModel):
    source_artifact_id: str
    model: str = "same-l"
    backend: BackendName = BackendName.TORCH_MPS
    chunked: bool = False
    chunk_size: int = Field(default=128, ge=1)
    overlap: int = Field(default=32, ge=0)
    notes: str | None = None
    session_id: str | None = None


class OperatorRunRequest(ContractModel):
    operator: OperatorName
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    backend: BackendName = BackendName.TORCH_CPU
    seed: int | None = None
    notes: str | None = None
    session_id: str | None = None


class ExperimentRunRequest(ContractModel):
    operator: OperatorName
    inputs: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    backend: BackendName = BackendName.TORCH_MPS
    model: str | None = None
    seed: int | None = None
    notes: str | None = None
    session_id: str | None = None


class RecipeForkRequest(ContractModel):
    inputs: dict[str, str] | None = None
    params: dict[str, Any] | None = None
    backend: BackendName | None = None
    model: str | None = None
    seed: int | None = None
    notes: str | None = None
    session_id: str | None = None


class ArtifactAnnotationRequest(ContractModel):
    label: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
