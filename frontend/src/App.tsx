import { Fragment, Suspense, lazy, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  Archive,
  AudioLines,
  Box,
  Braces,
  Check,
  CircleAlert,
  CircleDot,
  Database,
  Download,
  FileAudio,
  FlaskConical,
  Gauge,
  GitFork,
  LoaderCircle,
  Play,
  Plus,
  Repeat,
  Search,
  SkipBack,
  SkipForward,
  SlidersHorizontal,
  Upload,
  Wand2,
  Waves,
  X,
} from "lucide-react";

import modelImage from "../../stable-audio-3.png";
import {
  createApi,
  type ArtifactAnnotationPayload,
  type AudioToAudioPayload,
  type ExperimentPayload,
  type GenerateTextPayload,
  type InpaintPayload,
  type RecipeForkPayload,
} from "./api";
import {
  artifactFilterOptions,
  artifactFiltersActive,
  emptyArtifactFilters,
  filterArtifacts,
  type ArtifactDecisionFilter,
  type ArtifactFilterOption,
  type ArtifactFilterState,
  type ArtifactKindFilter,
  type ArtifactLineageFilter,
} from "./artifactFilters";
import { ArtifactBadge, ArtifactIcon } from "./artifactDisplay";
import { artifactMeta, artifactName, artifactShape, formatDuration, sortNewest, sortNewestJobs } from "./artifactUtils";
import { auditionCursor, auditionKeyboardTarget, auditionPositionLabel, auditionSequenceOptions, auditionStackRows, type AuditionSequenceMode } from "./auditionStack";
import { AudioDeck, TinyWave } from "./audioDeck";
import { createControlPlaneClient, DEFAULT_CONTROL_PLANE_URL, type ResultFamily, type WorkbenchState } from "./controlPlane";
import { ForkRecipePanel } from "./forkRecipePanel";
import { JobProgress, type JobActionHandlers } from "./jobProgress";
import { isJobActive, landingArtifactId, shortOperatorName } from "./jobUtils";
import { artifactLineageModel, type CompareSlots } from "./lineageModel";
import { ListeningDecisionBadge } from "./listeningDecision";
import { specCoverageSummary, specPairCoverageSummary } from "./operatorSpecCoverage";
import {
  createOperatorPreset,
  deleteOperatorPreset,
  loadOperatorPresets,
  operatorPresetDiffRows,
  persistOperatorPresets,
  upsertOperatorPreset,
  type OperatorPreset,
  type OperatorPresetDiffRow,
} from "./operatorPresets";
import {
  applyPromptSearchAxisSet,
  applyPromptSearchPreset,
  applyPromptSearchVocabularySet,
  promptSearchAxisSets,
  promptSearchHistoryRows,
  promptSearchPresets,
  promptSearchScorerNote,
  promptSearchVocabularySets,
  type PromptSearchAxisSet,
  type PromptSearchHistoryRow,
  type PromptSearchPreset,
  type PromptSearchVocabularySet,
} from "./promptSearchPresets";
import { RecipeFields } from "./RecipeFields";
import { FamilyDetailPanel, ResultFamilyPanel } from "./resultFamilies";
import {
  buildExperimentPayload,
  buildGenerationPayload,
  buildLatentDecodePayload,
  buildLatentEncodePayload,
  buildOperatorParams,
  defaultFieldForm,
  defaultDecoderForGenerationModel,
  experimentReady,
  fieldKeys,
  fillMissingFieldDefaults,
  generationReady,
  operatorBackend,
  operatorReady,
  operatorSeed,
  operatorUsesDonor,
  withOperatorSpecFields,
  type FieldConfig,
  type GenerationMode,
  type RecipeField,
  type RecipeValue,
} from "./recipeFormModel";
import { archivableSessionArtifacts, artifactArchivePayload, artifactRecoveryPayload, recoverableArchiveArtifacts } from "./sessionRecovery";
import { summarizeSessionWorkspace, workspaceFocus, workspacePulseRows, type WorkspaceFocus, type WorkspacePulseRow } from "./sessionWorkspace";
import { useBenchStore } from "./store";
import type {
  ArtifactRecord,
  HealthResponse,
  JobRecord,
  ModelStatus,
  NotebookMode,
  OperatorName,
  OperatorSpec,
  ReadinessCheck,
  Recipe,
  SessionRecord,
} from "./types";
import type { PromptCandidateGenerationRequest } from "./bundleInspector";

const BundleField = lazy(() => import("./bundleInspector").then((module) => ({ default: module.BundleField })));

const audioModels = [
  { value: "medium", label: "medium", decoder: "same-l" },
  { value: "sm-music", label: "sm-music", decoder: "same-s" },
  { value: "sm-sfx", label: "sm-sfx", decoder: "same-s" },
] as const;

type ExperimentMode =
  | "experiment.audio_style_vectors"
  | "experiment.positive_style_profile"
  | "experiment.style_profile.build"
  | "experiment.style_profile.generate"
  | "experiment.style_direction.generate"
  | "experiment.audio_direction.generate"
  | "experiment.sa3_vectors.extract"
  | "experiment.audio_residual_vectors.extract"
  | "experiment.alpha_sweep"
  | "experiment.geometry_audit"
  | "experiment.prompt_search"
  | "experiment.soft_prompt.optimize"
  | "experiment.soft_prompt.generate"
  | "dataset.pre_encode"
  | "memory.query"
  | "training.lora";

interface ExperimentConfig {
  value: ExperimentMode;
  label: string;
  family: "Style" | "Residual" | "Prompt" | "Soft Prompt" | "Dataset" | "Geometry" | "Training";
  maturity: "lab" | "probe" | "danger";
  backend: ExperimentPayload["backend"];
  modelDefault?: string;
  produces: readonly ArtifactRecord["kind"][];
  fields: readonly RecipeField[];
  selectedAudioFallback?: string;
  selectedLatentFallback?: boolean;
}

type LatentOperatorMode =
  | "latent.cyclic_roll"
  | "latent.blur"
  | "latent.dsp"
  | "latent.graft"
  | "latent.renoise";

interface OperatorConfig extends FieldConfig {
  value: LatentOperatorMode;
  label: string;
  family: "Loop" | "Blur" | "DSP" | "Graft" | "Renoise";
  maturity: "lab" | "probe";
  defaultBackend: "torch_cpu" | "torch_mps";
  requiresDonor?: boolean;
}

interface GenerationConfig extends FieldConfig {
  value: GenerationMode;
  label: string;
}

const sameModelOptions = [
  { value: "same-s", label: "same-s" },
  { value: "same-l", label: "same-l" },
] as const;

const sa3ModelOptions = [
  { value: "medium", label: "medium" },
  { value: "small-music", label: "small-music" },
  { value: "small-sfx", label: "small-sfx" },
] as const;

const backendOptions = [
  { value: "torch_mps", label: "torch_mps" },
  { value: "torch_cpu", label: "torch_cpu" },
  { value: "cpu", label: "cpu" },
] as const;

const promptSearchModeOptions = [
  { value: "beam", label: "beam" },
  { value: "greedy", label: "greedy" },
  { value: "coordinate", label: "coordinate" },
] as const;

const promptSearchScorerOptions = [
  { value: "lexical_probe", label: "lexical probe" },
  { value: "sa3_flow_probe", label: "SA3 flow probe" },
  { value: "clap", label: "CLAP queued" },
] as const;

const velocityConventionOptions = [
  { value: "noise_minus_data", label: "noise minus data" },
  { value: "data_minus_noise", label: "data minus noise" },
] as const;

const promptSearchVocabulary = [
  "warm",
  "cold",
  "bright",
  "dark",
  "muted",
  "shimmering",
  "granular",
  "textured",
  "pulsing",
  "drifting",
  "wide stereo",
  "reverberant",
  "ambient",
  "cinematic",
  "electronic",
  "acoustic",
  "percussive",
  "sustained",
  "metallic",
  "soft",
  "dense",
  "sparse",
  "loop",
  "rhythm",
  "tone",
  "noise",
  "gesture",
  "texture",
].join(", ");

const operatorBackendOptions = [
  { value: "torch_mps", label: "torch_mps" },
  { value: "torch_cpu", label: "torch_cpu" },
] as const;

const torchCpuOperatorOptions = [
  { value: "torch_cpu", label: "torch_cpu" },
] as const;

const maskModeOptions = [
  { value: "random_channels", label: "random channels" },
  { value: "high_variance", label: "high variance" },
  { value: "low_variance", label: "low variance" },
  { value: "high_activity", label: "high activity" },
  { value: "low_activity", label: "low activity" },
  { value: "channel_block", label: "channel block" },
  { value: "every_n", label: "comb / every n" },
] as const;

const latentBlurModeOptions = [
  { value: "temporal", label: "temporal blur" },
  { value: "channel", label: "channel blur" },
  { value: "temporal_channel", label: "temporal + channel" },
  { value: "low_rank", label: "low rank" },
  { value: "detail_attenuate", label: "detail attenuate" },
  { value: "sharpen", label: "temporal sharpen" },
  { value: "channel_sharpen", label: "channel sharpen" },
  { value: "fft_lowpass", label: "FFT lowpass" },
  { value: "fft_highpass", label: "FFT highpass" },
  { value: "fft_bandpass", label: "FFT bandpass" },
  { value: "mean_blend", label: "mean blend" },
] as const;

const temporalKernelOptions = [
  { value: "gaussian", label: "gaussian" },
  { value: "box", label: "box" },
] as const;

const temporalDirectionOptions = [
  { value: "centered", label: "centered" },
  { value: "past", label: "past" },
  { value: "future", label: "future" },
] as const;

const latentDspModeOptions = [
  { value: "gain", label: "gain" },
  { value: "compress", label: "compress" },
  { value: "expand", label: "expand" },
  { value: "softclip", label: "soft clip" },
  { value: "fft_eq", label: "FFT EQ" },
  { value: "fft_phase_shift", label: "phase shift" },
  { value: "fft_phase_randomize", label: "phase randomize" },
  { value: "fft_phase_blend", label: "donor phase blend" },
  { value: "fft_mag_phase_graft", label: "donor magnitude graft" },
  { value: "fft_phase_from_donor", label: "donor phase" },
  { value: "pca_gain", label: "PCA component gain" },
] as const;

const dspCenterOptions = [
  { value: "channel_mean", label: "channel mean" },
  { value: "global_mean", label: "global mean" },
  { value: "zero", label: "zero" },
] as const;

const operatorModes = [
  { value: "latent.cyclic_roll", label: "cyclic roll" },
  { value: "latent.blur", label: "latent blur" },
  { value: "latent.dsp", label: "latent DSP" },
  { value: "latent.graft", label: "latent graft" },
  { value: "latent.renoise", label: "latent renoise" },
] as const;

const operatorCatalog: readonly OperatorConfig[] = [
  {
    value: "latent.cyclic_roll",
    label: "Cyclic roll",
    family: "Loop",
    maturity: "lab",
    defaultBackend: "torch_mps",
    fields: [
      { key: "shift_frames", label: "Shift frames", type: "number", defaultValue: 1, min: -4096, max: 4096, step: 1 },
      { key: "strength", label: "Mix strength", type: "range", defaultValue: 1, min: 0, max: 1, step: 0.01 },
      { key: "symmetric", label: "Symmetric mix", type: "checkbox", defaultValue: true },
      { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: operatorBackendOptions, advanced: true },
      { key: "seed", label: "Seed", type: "number", defaultValue: 7, step: 1, advanced: true },
    ],
  },
  {
    value: "latent.blur",
    label: "Latent blur",
    family: "Blur",
    maturity: "lab",
    defaultBackend: "torch_mps",
    fields: [
      { key: "mode", label: "Mode", type: "select", defaultValue: "temporal", options: latentBlurModeOptions },
      { key: "strength", label: "Strength", type: "range", defaultValue: 0.5, min: 0, max: 1.5, step: 0.01 },
      { key: "temporal_radius", label: "Temporal radius", type: "number", defaultValue: 4, min: 0, step: 1 },
      { key: "channel_radius", label: "Channel radius", type: "number", defaultValue: 2, min: 0, step: 1 },
      { key: "temporal_sigma", label: "Temporal sigma", type: "number", min: 0, step: 0.05, advanced: true },
      { key: "temporal_kernel", label: "Temporal kernel", type: "select", defaultValue: "gaussian", options: temporalKernelOptions, advanced: true },
      { key: "temporal_direction", label: "Temporal direction", type: "select", defaultValue: "centered", options: temporalDirectionOptions, advanced: true },
      { key: "channel_sigma", label: "Channel sigma", type: "number", min: 0, step: 0.05, advanced: true },
      { key: "rank", label: "Low-rank keep", type: "number", defaultValue: 16, min: 1, step: 1, advanced: true },
      { key: "detail_gain", label: "Detail gain", type: "range", defaultValue: 0.25, min: 0, max: 2, step: 0.01, advanced: true },
      { key: "sharpen_amount", label: "Sharpen amount", type: "range", defaultValue: 0.5, min: 0, max: 2, step: 0.01, advanced: true },
      { key: "filter_cutoff", label: "Filter cutoff", type: "range", defaultValue: 0.5, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "filter_low_cutoff", label: "Band low cutoff", type: "range", defaultValue: 0.1, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "filter_high_cutoff", label: "Band high cutoff", type: "range", defaultValue: 0.6, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "filter_low_gain", label: "Low gain", type: "range", defaultValue: 0, min: 0, max: 2, step: 0.01, advanced: true },
      { key: "filter_mid_gain", label: "Mid gain", type: "range", defaultValue: 1, min: 0, max: 2, step: 0.01, advanced: true },
      { key: "filter_high_gain", label: "High gain", type: "range", defaultValue: 0, min: 0, max: 2, step: 0.01, advanced: true },
      { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: operatorBackendOptions, advanced: true },
      { key: "seed", label: "Seed", type: "number", defaultValue: 7, step: 1, advanced: true },
    ],
  },
  {
    value: "latent.dsp",
    label: "Latent DSP",
    family: "DSP",
    maturity: "lab",
    defaultBackend: "torch_mps",
    fields: [
      { key: "mode", label: "Mode", type: "select", defaultValue: "gain", options: latentDspModeOptions },
      { key: "strength", label: "Blend strength", type: "range", defaultValue: 1, min: 0, max: 1.5, step: 0.01 },
      { key: "gain", label: "Gain", type: "range", defaultValue: 1.2, min: 0, max: 3, step: 0.01 },
      { key: "center", label: "Center", type: "select", defaultValue: "channel_mean", options: dspCenterOptions },
      { key: "threshold", label: "Threshold", type: "range", defaultValue: 1, min: 0.01, max: 4, step: 0.01, advanced: true },
      { key: "ratio", label: "Ratio", type: "range", defaultValue: 4, min: 0.1, max: 12, step: 0.1, advanced: true },
      { key: "makeup_gain", label: "Makeup gain", type: "range", defaultValue: 1, min: 0, max: 3, step: 0.01, advanced: true },
      { key: "drive", label: "Drive", type: "range", defaultValue: 1, min: 0.01, max: 6, step: 0.01, advanced: true },
      { key: "ceiling", label: "Ceiling", type: "range", defaultValue: 2, min: 0.01, max: 8, step: 0.01, advanced: true },
      { key: "fft_low_cutoff", label: "FFT low cutoff", type: "range", defaultValue: 0.15, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "fft_high_cutoff", label: "FFT high cutoff", type: "range", defaultValue: 0.65, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "fft_low_gain", label: "FFT low gain", type: "range", defaultValue: 1, min: 0, max: 3, step: 0.01, advanced: true },
      { key: "fft_mid_gain", label: "FFT mid gain", type: "range", defaultValue: 1, min: 0, max: 3, step: 0.01, advanced: true },
      { key: "fft_high_gain", label: "FFT high gain", type: "range", defaultValue: 1, min: 0, max: 3, step: 0.01, advanced: true },
      { key: "phase_shift_fraction", label: "Phase shift", type: "range", defaultValue: 0, min: -1, max: 1, step: 0.01, advanced: true },
      { key: "phase_random_amount", label: "Random phase", type: "range", defaultValue: 1, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "phase_blend_amount", label: "Donor phase blend", type: "range", defaultValue: 1, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "magnitude_amount", label: "Donor magnitude", type: "range", defaultValue: 1, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "pca_rank", label: "PCA rank", type: "number", min: 1, step: 1, advanced: true },
      { key: "pca_component_gains", label: "PCA gains", type: "text", placeholder: "1,0.8,1.2", advanced: true },
      { key: "seed", label: "Seed", type: "number", defaultValue: 7, step: 1, advanced: true },
      { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: operatorBackendOptions, advanced: true },
    ],
  },
  {
    value: "latent.graft",
    label: "Latent graft",
    family: "Graft",
    maturity: "lab",
    defaultBackend: "torch_mps",
    requiresDonor: true,
    fields: [
      { key: "mode", label: "Mask mode", type: "select", defaultValue: "random_channels", options: maskModeOptions },
      { key: "fraction", label: "Channel fraction", type: "range", defaultValue: 0.25, min: 0.01, max: 1, step: 0.01 },
      { key: "amount", label: "Graft amount", type: "range", defaultValue: 1, min: 0, max: 1, step: 0.01 },
      { key: "seed", label: "Mask seed", type: "number", defaultValue: 7, step: 1 },
      { key: "channels", label: "Explicit channels", type: "text", placeholder: "0,3,12", advanced: true },
      { key: "start_channel", label: "Block start", type: "number", min: 0, step: 1, advanced: true },
      { key: "block_size", label: "Block size", type: "number", min: 1, step: 1, advanced: true },
      { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: operatorBackendOptions, advanced: true },
    ],
  },
  {
    value: "latent.renoise",
    label: "Latent renoise",
    family: "Renoise",
    maturity: "lab",
    defaultBackend: "torch_cpu",
    fields: [
      { key: "mode", label: "Mask mode", type: "select", defaultValue: "random_channels", options: maskModeOptions },
      { key: "fraction", label: "Channel fraction", type: "range", defaultValue: 0.25, min: 0.01, max: 1, step: 0.01 },
      { key: "sigma", label: "Noise sigma", type: "range", defaultValue: 0.4, min: 0, max: 1.5, step: 0.01 },
      { key: "seed", label: "Noise seed", type: "number", defaultValue: 7, step: 1 },
      { key: "channels", label: "Explicit channels", type: "text", placeholder: "0,3,12", advanced: true },
      { key: "start_channel", label: "Block start", type: "number", min: 0, step: 1, advanced: true },
      { key: "block_size", label: "Block size", type: "number", min: 1, step: 1, advanced: true },
      { key: "backend", label: "Backend", type: "select", defaultValue: "torch_cpu", options: torchCpuOperatorOptions, advanced: true },
    ],
  },
];

const experimentCatalog: readonly ExperimentConfig[] = [
  {
    value: "experiment.audio_style_vectors",
    label: "Audio style vectors",
    family: "Style",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "same-l",
    produces: ["bundle"],
    fields: [
      { key: "positive_path", label: "Positive folder", type: "path", required: true },
      { key: "negative_path", label: "Negative folder", type: "path", required: true },
      { key: "name", label: "Name", type: "text", defaultValue: "audio_direction" },
      { key: "model", label: "SAME model", type: "select", defaultValue: "same-l", options: sameModelOptions },
      { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: backendOptions },
      { key: "limit", label: "Limit", type: "number", defaultValue: 0, min: 0, step: 1, advanced: true },
      { key: "chunked", label: "Chunked encode", type: "checkbox", defaultValue: false, advanced: true },
      { key: "normalize_frame", label: "Normalize frame", type: "checkbox", defaultValue: false, advanced: true },
      { key: "device", label: "Device", type: "text", advanced: true },
    ],
  },
  {
    value: "experiment.positive_style_profile",
    label: "Positive style profile",
    family: "Style",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "same-l",
    produces: ["bundle"],
    fields: [
      { key: "input_path", label: "Audio folder", type: "path", required: true },
      { key: "name", label: "Name", type: "text", defaultValue: "positive_style" },
      { key: "model", label: "SAME model", type: "select", defaultValue: "same-l", options: sameModelOptions },
      { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: backendOptions },
      { key: "limit", label: "Limit", type: "number", defaultValue: 0, min: 0, step: 1, advanced: true },
      { key: "chunked", label: "Chunked encode", type: "checkbox", defaultValue: false, advanced: true },
      { key: "device", label: "Device", type: "text", advanced: true },
    ],
  },
  {
    value: "experiment.style_profile.build",
    label: "Build style profile",
    family: "Style",
    maturity: "lab",
    backend: "cpu",
    produces: ["bundle"],
    fields: [
      { key: "target_memory_path", label: "Target memory", type: "artifact-path", required: true, artifactKinds: ["bundle"] },
      { key: "reference_memory_path", label: "Reference memory", type: "artifact-path", artifactKinds: ["bundle"] },
      { key: "name", label: "Name", type: "text", defaultValue: "target_style" },
      { key: "backend", label: "Backend", type: "select", defaultValue: "cpu", options: backendOptions, advanced: true },
    ],
  },
  {
    value: "experiment.style_profile.generate",
    label: "Generate from profile",
    family: "Style",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["audio"],
    fields: [
      { key: "profile_path", label: "Profile .npz", type: "artifact-path", required: true, artifactKinds: ["bundle"] },
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "audio texture", required: true },
      { key: "alpha", label: "Alpha", type: "number", defaultValue: 0.6, step: 0.05 },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      ...generationFields(),
      { key: "match_std", label: "Match std", type: "checkbox", defaultValue: true, advanced: true },
      { key: "save_original", label: "Save original", type: "checkbox", defaultValue: false, advanced: true },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "experiment.style_direction.generate",
    label: "Generate from style direction",
    family: "Style",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["audio"],
    fields: [
      { key: "direction_path", label: "Direction .npz", type: "artifact-path", required: true, artifactKinds: ["bundle"] },
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "audio texture", required: true },
      { key: "alpha", label: "Alpha", type: "number", defaultValue: 0.6, step: 0.05 },
      { key: "std_alpha", label: "Std alpha", type: "number", defaultValue: 0, step: 0.05 },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      ...generationFields(),
      { key: "save_original", label: "Save original", type: "checkbox", defaultValue: false, advanced: true },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "experiment.audio_direction.generate",
    label: "Generate from audio direction",
    family: "Style",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["audio"],
    fields: [
      { key: "direction_path", label: "Frame direction .npz", type: "artifact-path", required: true, artifactKinds: ["bundle"] },
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "audio texture", required: true },
      { key: "alpha", label: "Alpha", type: "number", defaultValue: 0.6, step: 0.05 },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      ...generationFields(),
      { key: "save_original", label: "Save original", type: "checkbox", defaultValue: false, advanced: true },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "experiment.sa3_vectors.extract",
    label: "Prompt residual vectors",
    family: "Residual",
    maturity: "probe",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["bundle"],
    fields: [
      { key: "axis", label: "Axis", type: "text", defaultValue: "valence", required: true },
      { key: "num_pairs", label: "Prompt pairs", type: "number", defaultValue: 2, min: 1, step: 1 },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      ...generationFields(),
      { key: "layers", label: "Layers", type: "text", placeholder: "blank or 1,4,8", advanced: true },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "experiment.audio_residual_vectors.extract",
    label: "Audio residual vectors",
    family: "Residual",
    maturity: "probe",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["bundle"],
    fields: [
      { key: "positive_path", label: "Positive folder", type: "path", required: true },
      { key: "negative_path", label: "Negative folder", type: "path" },
      { key: "baseline", label: "Baseline", type: "select", defaultValue: "prompt", options: [{ value: "prompt", label: "prompt" }, { value: "negative_audio", label: "negative_audio" }] },
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "audio texture" },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      ...generationFields(),
      { key: "init_noise_level", label: "Init noise", type: "number", defaultValue: 0.35, min: 0, step: 0.05, advanced: true },
      { key: "layers", label: "Layers", type: "text", advanced: true },
      { key: "limit", label: "Limit", type: "number", defaultValue: 0, min: 0, step: 1, advanced: true },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "experiment.alpha_sweep",
    label: "Alpha sweep",
    family: "Residual",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["audio", "bundle"],
    fields: [
      { key: "vectors_path", label: "Vectors", type: "artifact-path", required: true, artifactKinds: ["bundle"] },
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "audio texture", required: true },
      { key: "alphas", label: "Alphas", type: "text", defaultValue: "-8,-4,0,4,8" },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      ...generationFields(),
      { key: "layer", label: "Layer", type: "number", defaultValue: -1, step: 1, advanced: true },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "experiment.prompt_search",
    label: "Prompt search",
    family: "Prompt",
    maturity: "probe",
    backend: "cpu",
    produces: ["bundle"],
    selectedAudioFallback: "target_audio_path",
    fields: [
      { key: "target_audio_path", label: "Target audio", type: "artifact-path", artifactKinds: ["audio"] },
      { key: "seed_prompt", label: "Seed prompt", type: "text", defaultValue: "audio texture" },
      { key: "search_mode", label: "Search mode", type: "select", defaultValue: "beam", options: promptSearchModeOptions },
      { key: "scorer", label: "Scorer", type: "select", defaultValue: "lexical_probe", options: promptSearchScorerOptions },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      {
        key: "duration_seconds",
        label: "Duration",
        type: "number",
        min: 0,
        step: 0.5,
        description: "Leave empty or zero to infer the target audio duration.",
      },
      { key: "tokens_generated", label: "Tokens", type: "number", defaultValue: 4, min: 1, step: 1 },
      { key: "beam_width", label: "Beam width", type: "number", defaultValue: 4, min: 1, step: 1 },
      { key: "score_samples", label: "Score samples", type: "number", defaultValue: 1, min: 1, step: 1 },
      { key: "seed", label: "Seed", type: "number", defaultValue: 0, step: 1 },
      { key: "vocabulary", label: "Vocabulary", type: "text", defaultValue: promptSearchVocabulary, advanced: true },
      { key: "branch_factor", label: "Branch factor", type: "number", defaultValue: 64, min: 0, step: 1, advanced: true },
      { key: "runs", label: "Greedy runs", type: "number", defaultValue: 4, min: 1, step: 1, advanced: true },
      { key: "rounds", label: "Coordinate rounds", type: "number", defaultValue: 2, min: 1, step: 1, advanced: true },
      { key: "candidate_batch_size", label: "Batch size", type: "number", defaultValue: 0, min: 0, step: 1, advanced: true },
      { key: "timestep_values", label: "Timesteps", type: "text", placeholder: "0.12,0.5,0.88", advanced: true },
      { key: "logsnr_values", label: "LogSNR probes", type: "text", placeholder: "2,0,-2", advanced: true },
      { key: "min_t", label: "Min t", type: "number", defaultValue: 0.05, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "max_t", label: "Max t", type: "number", defaultValue: 0.95, min: 0, max: 1, step: 0.01, advanced: true },
      { key: "shared_noise", label: "Shared noise", type: "checkbox", defaultValue: true, advanced: true },
      { key: "antithetic_noise", label: "Antithetic noise", type: "checkbox", defaultValue: false, advanced: true },
      { key: "normalize_mse", label: "Normalize MSE", type: "checkbox", defaultValue: true, advanced: true },
      { key: "cosine_weight", label: "Cosine weight", type: "number", defaultValue: 0, min: 0, step: 0.05, advanced: true },
      { key: "conditional_delta_weight", label: "Delta weight", type: "number", defaultValue: 0, min: 0, step: 0.05, advanced: true },
      {
        key: "velocity_convention",
        label: "Velocity",
        type: "select",
        defaultValue: "noise_minus_data",
        options: velocityConventionOptions,
        advanced: true,
      },
      { key: "prefix", label: "Prefix", type: "text", advanced: true },
      { key: "suffix", label: "Suffix", type: "text", advanced: true },
      { key: "separator", label: "Separator", type: "text", defaultValue: " ", advanced: true },
      { key: "modifier_axes", label: "Modifier axes", type: "text", placeholder: "bright|dark|warm; sparse|dense", advanced: true },
      { key: "device", label: "Device", type: "text", placeholder: "auto", advanced: true },
      { key: "model_half", label: "Half precision", type: "checkbox", defaultValue: false, advanced: true },
      { key: "backend", label: "Backend", type: "select", defaultValue: "cpu", options: backendOptions, advanced: true },
    ],
  },
  {
    value: "experiment.soft_prompt.optimize",
    label: "Optimize soft prompt",
    family: "Soft Prompt",
    maturity: "probe",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["bundle"],
    selectedAudioFallback: "target_audio_path",
    fields: [
      { key: "target_audio_path", label: "Target audio", type: "artifact-path", artifactKinds: ["audio"] },
      { key: "seed_prompt", label: "Seed prompt", type: "text", defaultValue: "audio texture" },
      { key: "optimization_steps", label: "Opt steps", type: "number", defaultValue: 100, min: 1, step: 1 },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      { key: "duration_seconds", label: "Seconds", type: "number", defaultValue: 0, min: 0, step: 0.5, advanced: true },
      { key: "lr", label: "LR", type: "number", defaultValue: 0.01, step: 0.001, advanced: true },
      { key: "reg_weight", label: "Reg weight", type: "number", defaultValue: 0.0001, step: 0.0001, advanced: true },
      { key: "seed", label: "Seed", type: "number", defaultValue: 0, step: 1, advanced: true },
      { key: "train_keys", label: "Train keys", type: "text", defaultValue: "prompt", advanced: true },
      { key: "velocity_convention", label: "Velocity", type: "select", defaultValue: "noise_minus_data", advanced: true, options: [{ value: "noise_minus_data", label: "noise_minus_data" }, { value: "data_minus_noise", label: "data_minus_noise" }] },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "experiment.soft_prompt.generate",
    label: "Generate soft prompt",
    family: "Soft Prompt",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "medium",
    produces: ["audio"],
    fields: [
      { key: "soft_prompt_path", label: "Soft prompt .pt", type: "artifact-path", required: true, artifactKinds: ["bundle"] },
      { key: "model", label: "SA3 model", type: "select", defaultValue: "medium", options: sa3ModelOptions },
      { key: "steps", label: "Steps", type: "number", defaultValue: 8, min: 1, step: 1 },
      { key: "cfg_scale", label: "CFG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
      { key: "seed", label: "Seed", type: "number", defaultValue: 42, step: 1 },
      ...torchAdvancedFields(),
    ],
  },
  {
    value: "dataset.pre_encode",
    label: "Pre-encode dataset",
    family: "Dataset",
    maturity: "lab",
    backend: "torch_mps",
    modelDefault: "same-l",
    produces: ["bundle"],
    fields: [
      { key: "data_dir", label: "Dataset folder", type: "path", required: true },
      { key: "model", label: "SAME model", type: "select", defaultValue: "same-l", options: sameModelOptions },
      { key: "batch_size", label: "Batch size", type: "number", defaultValue: 1, min: 1, step: 1 },
      { key: "sample_size", label: "Sample size", type: "number", defaultValue: 12582912, min: 1, step: 1, advanced: true },
      { key: "model_half", label: "Model half", type: "checkbox", defaultValue: false, advanced: true },
      { key: "pad", label: "Pad", type: "checkbox", defaultValue: false, advanced: true },
      { key: "device", label: "Device", type: "text", advanced: true },
    ],
  },
  {
    value: "memory.query",
    label: "Memory query",
    family: "Dataset",
    maturity: "lab",
    backend: "cpu",
    produces: ["bundle"],
    selectedLatentFallback: true,
    fields: [
      { key: "top_k", label: "Top K", type: "number", defaultValue: 5, min: 1, step: 1 },
      { key: "metric", label: "Metric", type: "select", defaultValue: "cosine", options: [{ value: "cosine", label: "cosine" }, { value: "euclidean", label: "euclidean" }] },
      { key: "exclude_self", label: "Exclude selected", type: "checkbox", defaultValue: true },
    ],
  },
  {
    value: "experiment.geometry_audit",
    label: "Geometry audit",
    family: "Geometry",
    maturity: "probe",
    backend: "cpu",
    produces: ["bundle"],
    selectedLatentFallback: true,
    fields: [
      { key: "n_components", label: "Components", type: "number", defaultValue: 8, min: 1, step: 1 },
      { key: "limit", label: "Limit", type: "number", defaultValue: 0, min: 0, step: 1, advanced: true },
    ],
  },
  {
    value: "training.lora",
    label: "Train LoRA",
    family: "Training",
    maturity: "danger",
    backend: "torch_mps",
    modelDefault: "medium-base",
    produces: ["bundle"],
    fields: [
      { key: "encoded_dir", label: "Encoded dataset", type: "artifact-path", artifactKinds: ["bundle"] },
      { key: "data_dir", label: "Raw dataset", type: "path" },
      { key: "model", label: "Base model", type: "select", defaultValue: "medium-base", options: [{ value: "medium-base", label: "medium-base" }] },
      { key: "steps", label: "Steps", type: "number", defaultValue: 10000, min: 1, step: 100 },
      { key: "rank", label: "Rank", type: "number", defaultValue: 16, min: 1, step: 1 },
      { key: "adapter_type", label: "Adapter", type: "select", defaultValue: "dora-rows", advanced: true, options: ["lora", "dora", "dora-rows", "dora-cols", "bora", "lora-xs", "dora-rows-xs", "dora-cols-xs", "bora-xs"].map((value) => ({ value, label: value })) },
      { key: "lora_alpha", label: "LoRA alpha", type: "number", step: 1, advanced: true },
      { key: "dropout", label: "Dropout", type: "number", defaultValue: 0, min: 0, max: 1, step: 0.05, advanced: true },
      { key: "include", label: "Include", type: "text", advanced: true },
      { key: "exclude", label: "Exclude", type: "text", advanced: true },
      { key: "svd_bases_path", label: "SVD bases", type: "artifact-path", artifactKinds: ["bundle"], advanced: true },
      { key: "base_precision", label: "Base precision", type: "select", defaultValue: "bf16", advanced: true, options: ["bf16", "bfloat16", "fp16", "float16"].map((value) => ({ value, label: value })) },
      { key: "lora_checkpoint", label: "Resume ckpt", type: "artifact-path", artifactKinds: ["bundle"], advanced: true },
      { key: "lr", label: "LR", type: "number", defaultValue: 0.0001, step: 0.00001, advanced: true },
      { key: "batch_size", label: "Batch size", type: "number", defaultValue: 1, min: 1, step: 1, advanced: true },
      { key: "duration_seconds", label: "Seconds", type: "number", defaultValue: 380, min: 1, step: 1, advanced: true },
      { key: "seed", label: "Seed", type: "number", defaultValue: 42, step: 1, advanced: true },
      { key: "device", label: "Device", type: "text", advanced: true },
      { key: "logger", label: "Logger", type: "select", defaultValue: "csv", advanced: true, options: ["wandb", "comet", "csv", "none"].map((value) => ({ value, label: value })) },
      { key: "name", label: "Run name", type: "text", defaultValue: "lora-finetune", advanced: true },
      { key: "checkpoint_every", label: "Ckpt every", type: "number", defaultValue: 500, min: 1, step: 1, advanced: true },
      { key: "log_every", label: "Log every", type: "number", defaultValue: 100, min: 1, step: 1, advanced: true },
      { key: "demo_every", label: "Demo every", type: "number", defaultValue: 500, min: 1, step: 1, advanced: true },
      { key: "num_workers", label: "Workers", type: "number", defaultValue: 8, min: 0, step: 1, advanced: true },
    ],
  },
];

const experimentModes = experimentCatalog.map(({ value, label }) => ({ value, label }));
const generationModes = [
  { value: "generate.text_to_audio", label: "Text" },
  { value: "generate.audio_to_audio", label: "A2A" },
  { value: "generate.inpaint", label: "Inpaint" },
] as const;
const generateControlKeys = ["prompt", "negative_prompt", "duration_seconds", "steps", "seed", "cfg_scale", "apg_scale", "model", "decoder"];
const audioToAudioControlKeys = [...generateControlKeys, "init_noise_level"];
const inpaintControlKeys = [...audioToAudioControlKeys, "inpaint_start_seconds", "inpaint_end_seconds"];
const sameEncodeControlKeys = ["model", "chunked", "chunk_size", "overlap", "prompt", "notes"];
const sameDecodeControlKeys = ["model", "chunked", "chunk_size", "overlap", "notes"];

const generationCatalog: readonly GenerationConfig[] = [
  {
    value: "generate.text_to_audio",
    label: "Text",
    fields: [
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "short soft percussive click", required: true },
      { key: "model", label: "Model", type: "select", defaultValue: "medium", options: audioModels.map(({ value, label }) => ({ value, label })) },
      { key: "duration_seconds", label: "Seconds", type: "number", defaultValue: 5, min: 0.5, max: 120, step: 0.5 },
      { key: "steps", label: "Steps", type: "number", defaultValue: 8, min: 1, max: 256, step: 1 },
      { key: "seed", label: "Seed", type: "number", defaultValue: 7, step: 1 },
      { key: "cfg_scale", label: "CFG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
      { key: "apg_scale", label: "APG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
      { key: "negative_prompt", label: "Negative prompt", type: "text", placeholder: "optional", advanced: true },
      { key: "decoder", label: "Decoder", type: "select", defaultValue: "same-l", options: sameModelOptions, advanced: true },
    ],
  },
  {
    value: "generate.audio_to_audio",
    label: "A2A",
    fields: [
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "short soft percussive click", required: true },
      { key: "model", label: "Model", type: "select", defaultValue: "medium", options: audioModels.map(({ value, label }) => ({ value, label })) },
      { key: "duration_seconds", label: "Seconds", type: "number", defaultValue: 5, min: 0.5, max: 120, step: 0.5 },
      { key: "steps", label: "Steps", type: "number", defaultValue: 8, min: 1, max: 256, step: 1 },
      { key: "init_noise_level", label: "Init noise", type: "number", defaultValue: 0.7, min: 0, max: 1, step: 0.05 },
      { key: "seed", label: "Seed", type: "number", defaultValue: 7, step: 1 },
      { key: "cfg_scale", label: "CFG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
      { key: "apg_scale", label: "APG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
      { key: "negative_prompt", label: "Negative prompt", type: "text", placeholder: "optional", advanced: true },
      { key: "decoder", label: "Decoder", type: "select", defaultValue: "same-l", options: sameModelOptions, advanced: true },
    ],
  },
  {
    value: "generate.inpaint",
    label: "Inpaint",
    fields: [
      { key: "prompt", label: "Prompt", type: "text", defaultValue: "short soft percussive click", required: true },
      { key: "model", label: "Model", type: "select", defaultValue: "medium", options: audioModels.map(({ value, label }) => ({ value, label })) },
      { key: "duration_seconds", label: "Seconds", type: "number", defaultValue: 5, min: 0.5, max: 120, step: 0.5 },
      { key: "steps", label: "Steps", type: "number", defaultValue: 8, min: 1, max: 256, step: 1 },
      { key: "init_noise_level", label: "Init noise", type: "number", defaultValue: 0.7, min: 0, max: 1, step: 0.05 },
      { key: "inpaint_start_seconds", label: "Inpaint start", type: "number", defaultValue: 0, min: 0, step: 0.1 },
      { key: "inpaint_end_seconds", label: "Inpaint end", type: "number", defaultValue: 2, min: 0.1, step: 0.1 },
      { key: "seed", label: "Seed", type: "number", defaultValue: 7, step: 1 },
      { key: "cfg_scale", label: "CFG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
      { key: "apg_scale", label: "APG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
      { key: "negative_prompt", label: "Negative prompt", type: "text", placeholder: "optional", advanced: true },
      { key: "decoder", label: "Decoder", type: "select", defaultValue: "same-l", options: sameModelOptions, advanced: true },
    ],
  },
];

const sameConfig: FieldConfig = {
  fields: [
    { key: "model", label: "SAME model", type: "select", defaultValue: "same-l", options: sameModelOptions },
    { key: "chunked", label: "Chunked encode/decode", type: "checkbox", defaultValue: false },
    { key: "chunk_size", label: "Chunk size", type: "number", defaultValue: 128, min: 1, step: 1, advanced: true },
    { key: "overlap", label: "Overlap", type: "number", defaultValue: 32, min: 0, step: 1, advanced: true },
    { key: "prompt", label: "Encode prompt", type: "text", placeholder: "optional", advanced: true },
    { key: "notes", label: "Notes", type: "text", placeholder: "optional", advanced: true },
  ],
};

export function App() {
  const queryClient = useQueryClient();
  const { apiBase, setApiBase, selectedArtifactId, selectArtifact, sessionId, sessionStartedAt, setSession, compare, setCompare } = useBenchStore();
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const controlPlaneUrl = DEFAULT_CONTROL_PLANE_URL.trim();
  const controlPlane = useMemo(() => (controlPlaneUrl ? createControlPlaneClient(controlPlaneUrl) : null), [controlPlaneUrl]);
  const useControlPlane = Boolean(controlPlane);

  const health = useQuery({ queryKey: ["health", apiBase], queryFn: api.health, refetchInterval: 3000, enabled: !useControlPlane });
  const readiness = useQuery({ queryKey: ["readiness", apiBase], queryFn: api.readiness, refetchInterval: 30000 });
  const operatorSpecs = useQuery({ queryKey: ["operator-specs", apiBase], queryFn: api.operatorSpecs, staleTime: 30000, enabled: !useControlPlane });
  const sessions = useQuery({ queryKey: ["sessions", apiBase], queryFn: api.sessions, refetchInterval: 3000, enabled: !useControlPlane });
  const modeAtlas = useQuery({ queryKey: ["colab-modes", apiBase], queryFn: api.colabModes, staleTime: Infinity, enabled: !useControlPlane });
  const artifacts = useQuery({ queryKey: ["artifacts", apiBase], queryFn: () => api.artifacts(), refetchInterval: 1500, enabled: !useControlPlane });
  const jobs = useQuery({ queryKey: ["jobs", apiBase], queryFn: api.jobs, refetchInterval: 1000, enabled: !useControlPlane });
  const workbench = useQuery({
    queryKey: ["workbench", controlPlaneUrl, apiBase, sessionId, sessionStartedAt, selectedArtifactId],
    queryFn: () =>
      controlPlane!.workbench.load.query({
        apiBase,
        sessionId,
        sessionStartedAt,
        selectedArtifactId,
      }),
    enabled: useControlPlane,
    refetchInterval: 1500,
  });

  const [generationMode, setGenerationMode] = useState<GenerationMode>("generate.text_to_audio");
  const [generationForm, setGenerationForm] = useState<Record<string, RecipeValue>>(() => defaultGenerationForm("generate.text_to_audio"));
  const [sameForm, setSameForm] = useState<Record<string, RecipeValue>>(() => defaultFieldForm(sameConfig));
  const [operator, setOperator] = useState<LatentOperatorMode>("latent.cyclic_roll");
  const [operatorForm, setOperatorForm] = useState<Record<string, RecipeValue>>(() => defaultOperatorForm("latent.cyclic_roll"));
  const [donorArtifactId, setDonorArtifactId] = useState("");
  const [operatorPresets, setOperatorPresets] = useState<OperatorPreset[]>(() => loadOperatorPresets());
  const [operatorPresetName, setOperatorPresetName] = useState("");
  const [selectedOperatorPresetId, setSelectedOperatorPresetId] = useState("");
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>("experiment.audio_style_vectors");
  const [experimentForm, setExperimentForm] = useState<Record<string, RecipeValue>>(() => defaultExperimentForm("experiment.audio_style_vectors"));
  const [liveJobsById, setLiveJobsById] = useState<Record<string, JobRecord>>({});
  const [forkTarget, setForkTarget] = useState<Recipe | null>(null);
  const [inspectedFamilyId, setInspectedFamilyId] = useState<string | null>(null);

  const workbenchState = workbench.data;
  const allArtifacts = workbenchState?.artifacts ?? artifacts.data ?? [];
  const allSessions = workbenchState?.sessions ?? sessions.data ?? [];
  const activeSession = workbenchState?.activeSession ?? findActiveSession(allSessions, sessionId);
  const activeSessionId = workbenchState?.activeSessionId ?? activeSession?.session_id ?? sessionId;
  const sessionArtifacts = workbenchState?.sessionArtifacts ?? (activeSessionId
    ? allArtifacts.filter((item) => item.session_id === activeSessionId)
    : allArtifacts.filter((item) => createdAfter(item.created_at, sessionStartedAt)));
  const visibleArtifacts = sessionArtifacts;
  const selectedArtifact = allArtifacts.find((item) => item.artifact_id === selectedArtifactId) ?? workbenchState?.selectedArtifact ?? visibleArtifacts[0] ?? null;
  const audioArtifacts = allArtifacts.filter((item) => item.kind === "audio");
  const latentArtifacts = allArtifacts.filter((item) => item.kind === "latent");
  const bundleArtifacts = allArtifacts.filter((item) => item.kind === "bundle");
  const serverJobs = workbenchState?.jobs ?? jobs.data ?? [];
  const allJobs = mergeJobRecords(serverJobs, Object.values(liveJobsById));
  const sessionJobs = workbenchState?.sessionJobs ?? (activeSessionId
    ? allJobs.filter((item) => item.recipe.session_id === activeSessionId)
    : allJobs.filter((item) => createdAfter(item.created_at, sessionStartedAt)));
  const archiveJobs = workbenchState?.archiveJobs ?? (activeSessionId
    ? allJobs.filter((item) => item.recipe.session_id !== activeSessionId)
    : allJobs.filter((item) => !createdAfter(item.created_at, sessionStartedAt)));
  const archiveArtifacts = workbenchState?.archiveArtifacts ?? (activeSessionId
    ? allArtifacts.filter((item) => item.session_id !== activeSessionId)
    : allArtifacts.filter((item) => !createdAfter(item.created_at, sessionStartedAt)));
  const runningJobs = workbenchState?.runningJobs ?? allJobs.filter(isJobActive);
  const resultFamilies = workbenchState?.resultFamilies ?? buildResultFamilies(allArtifacts, allJobs);
  const sessionResultFamilies = workbenchState?.sessionResultFamilies ?? filterFamiliesForWork(resultFamilies, sessionArtifacts, sessionJobs);
  const inspectedFamily = sessionResultFamilies.find((family) => family.familyId === inspectedFamilyId) ?? sessionResultFamilies[0] ?? null;
  const latestJobs = workbenchState?.latestJob ? [workbenchState.latestJob, ...allJobs.filter((job) => job.job_id !== workbenchState.latestJob?.job_id)].slice(0, 8) : allJobs.slice(0, 8);
  const compareA = allArtifacts.find((item) => item.artifact_id === compare.a) ?? null;
  const compareB = allArtifacts.find((item) => item.artifact_id === compare.b) ?? null;
  const baseExperiment = experimentCatalog.find((item) => item.value === experimentMode) ?? experimentCatalog[0];
  const baseOperatorConfig = operatorCatalog.find((item) => item.value === operator) ?? operatorCatalog[0];
  const operatorSpecRows = workbenchState?.operatorSpecs ?? operatorSpecs.data ?? [];
  const modeAtlasRows = workbenchState?.modeAtlas ?? modeAtlas.data ?? [];
  const healthData = workbenchState?.health ?? health.data;
  const readinessChecks = readiness.data?.checks ?? readinessChecksFromHealth(healthData);
  const specMap = useMemo(() => new Map(operatorSpecRows.map((spec) => [spec.name, spec])), [operatorSpecRows]);
  const activeGenerateSpec = specMap.get(generationMode);
  const sameEncodeSpec = specMap.get("latent.encode");
  const sameDecodeSpec = specMap.get("latent.decode");
  const activeOperatorSpec = specMap.get(operator);
  const activeExperimentSpec = specMap.get(baseExperiment.value);
  const activeGenerationBase = generationCatalog.find((item) => item.value === generationMode) ?? generationCatalog[0];
  const activeGenerationConfig = useMemo(
    () => withOperatorSpecFields(activeGenerationBase, filteredOperatorSpec(activeGenerateSpec, generationControlKeys(generationMode))),
    [activeGenerateSpec, activeGenerationBase, generationMode],
  );
  const activeSameConfig = useMemo(
    () => withOperatorSpecFields(sameConfig, filteredOperatorSpec(sameEncodeSpec, sameEncodeControlKeys)),
    [sameEncodeSpec],
  );
  const activeOperatorConfig = useMemo(() => withOperatorSpecFields(baseOperatorConfig, activeOperatorSpec), [baseOperatorConfig, activeOperatorSpec]);
  const activeExperiment = useMemo(() => withOperatorSpecFields(baseExperiment, activeExperimentSpec), [baseExperiment, activeExperimentSpec]);
  const activeOperatorPresets = useMemo(() => operatorPresets.filter((preset) => preset.operator === operator), [operatorPresets, operator]);
  const selectedOperatorPreset = useMemo(
    () => operatorPresets.find((preset) => preset.id === selectedOperatorPresetId) ?? null,
    [operatorPresets, selectedOperatorPresetId],
  );
  const operatorPresetDiffs = useMemo(
    () => selectedOperatorPreset ? operatorPresetDiffRows(selectedOperatorPreset, operatorForm, donorArtifactId) : [],
    [donorArtifactId, operatorForm, selectedOperatorPreset],
  );
  const generationNeedsSource = generationMode !== "generate.text_to_audio";
  const generationSource = selectedArtifact?.kind === "audio" ? selectedArtifact : null;
  const canGenerate = generationReady({ form: generationForm, needsSource: generationNeedsSource, sourceArtifact: generationSource });
  const canRunOperator = operatorReady(activeOperatorConfig, operatorForm, selectedArtifact, donorArtifactId);
  const generateJob = activeJobForOperator(runningJobs, generationMode);
  const encodeJob = activeJobForOperator(runningJobs, "latent.encode");
  const decodeJob = activeJobForOperator(runningJobs, "latent.decode");
  const operatorJob = activeJobForOperator(runningJobs, operator);
  const experimentJob = activeJobForOperator(runningJobs, activeExperiment.value);
  const runningJobIds = runningJobs.map((job) => job.job_id).sort().join("|");
  const liveEventing = Object.values(liveJobsById).some((job) => isJobActive(job));

  useEffect(() => {
    if (!selectedArtifactId && visibleArtifacts[0]) {
      selectArtifact(visibleArtifacts[0].artifact_id);
    }
  }, [selectArtifact, selectedArtifactId, visibleArtifacts]);

  useEffect(() => {
    if (!allSessions.length) return;
    if (sessionId && allSessions.some((session) => session.session_id === sessionId)) return;
    const latestActive = allSessions.find((session) => session.status === "active") ?? allSessions[0];
    setSession(latestActive.session_id, latestActive.created_at);
  }, [sessionId, allSessions, setSession]);

  useEffect(() => {
    setOperatorForm((current) => fillMissingFieldDefaults(activeOperatorConfig, current));
  }, [activeOperatorConfig]);

  useEffect(() => {
    setGenerationForm((current) => fillMissingFieldDefaults(activeGenerationConfig, current));
  }, [activeGenerationConfig]);

  useEffect(() => {
    setSameForm((current) => fillMissingFieldDefaults(activeSameConfig, current));
  }, [activeSameConfig]);

  useEffect(() => {
    if (!selectedOperatorPresetId) return;
    if (activeOperatorPresets.some((preset) => preset.id === selectedOperatorPresetId)) return;
    setSelectedOperatorPresetId("");
    setOperatorPresetName("");
  }, [activeOperatorPresets, selectedOperatorPresetId]);

  useEffect(() => {
    setExperimentForm((current) => fillMissingFieldDefaults(activeExperiment, current));
  }, [activeExperiment]);

  useEffect(() => {
    if (!sessionResultFamilies.length) {
      setInspectedFamilyId(null);
      return;
    }
    if (!inspectedFamilyId || !sessionResultFamilies.some((family) => family.familyId === inspectedFamilyId)) {
      setInspectedFamilyId(sessionResultFamilies[0].familyId);
    }
  }, [inspectedFamilyId, sessionResultFamilies]);

  const invalidate = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["workbench"] }),
      queryClient.invalidateQueries({ queryKey: ["artifacts", apiBase] }),
      queryClient.invalidateQueries({ queryKey: ["jobs", apiBase] }),
      queryClient.invalidateQueries({ queryKey: ["health", apiBase] }),
      queryClient.invalidateQueries({ queryKey: ["sessions", apiBase] }),
    ]);
  };

  useEffect(() => {
    if (!runningJobIds) return;
    const landTerminalJob = (job: JobRecord) => {
      const artifactId = landingArtifactId(job);
      if (artifactId) selectArtifact(artifactId);
    };
    if (controlPlane) {
      const subscriptions = runningJobIds.split("|").map((jobId) =>
        controlPlane.jobs.events.subscribe(
          { jobId },
          {
            onData: (event) => {
              const job = jobFromJobEvent(event);
              if (!job) return;
              setLiveJobsById((current) => ({ ...current, [job.job_id]: job }));
              if (!isJobActive(job)) {
                landTerminalJob(job);
                void invalidate();
              }
            },
            onError: () => {
              void invalidate();
            },
          },
        ),
      );
      return () => {
        subscriptions.forEach((subscription) => subscription.unsubscribe());
      };
    }
    const sockets = runningJobIds.split("|").map((jobId) => {
      const socket = new WebSocket(api.jobEventsUrl(jobId));
      socket.onmessage = (event) => {
        const job = parseJobEvent(event.data);
        if (!job) return;
        setLiveJobsById((current) => ({ ...current, [job.job_id]: job }));
        if (!isJobActive(job)) {
          landTerminalJob(job);
          void invalidate();
        }
      };
      return socket;
    });
    return () => {
      sockets.forEach((socket) => socket.close());
    };
  }, [api, controlPlane, runningJobIds, selectArtifact]);

  const importAudio = useMutation({
    mutationFn: (file: File) => api.importAudio(file, file.name, activeSessionId),
    onSuccess: async (artifact) => {
      selectArtifact(artifact.artifact_id);
      await invalidate();
    },
  });

  const createSession = useMutation({
    mutationFn: () => api.createSession({ name: `Session ${new Date().toLocaleString()}` }),
    onSuccess: async (session) => {
      setSession(session.session_id, session.created_at);
      await invalidate();
    },
  });

  const archiveSession = useMutation({
    mutationFn: async (session: SessionRecord) => {
      await api.updateSession(session.session_id, { status: "archived" });
      return api.createSession({ name: `Session ${new Date().toLocaleString()}` });
    },
    onSuccess: async (session) => {
      setSession(session.session_id, session.created_at);
      selectArtifact(null);
      await invalidate();
    },
  });

  const recoverArtifact = useMutation({
    mutationFn: (artifact: ArtifactRecord) => {
      if (!activeSessionId) {
        throw new Error("No active session available for recovery");
      }
      return api.annotateArtifact(
        artifact.artifact_id,
        artifactRecoveryPayload({
          artifact,
          targetSessionId: activeSessionId,
          source: "archive_tray",
        }),
      );
    },
    onSuccess: async (artifact) => {
      selectArtifact(artifact.artifact_id);
      await invalidate();
    },
  });

  const archiveArtifact = useMutation({
    mutationFn: ({ artifact, source }: { artifact: ArtifactRecord; source: string }) =>
      api.annotateArtifact(
        artifact.artifact_id,
        artifactArchivePayload({
          artifact,
          source,
        }),
      ),
    onSuccess: async (artifact) => {
      if (selectedArtifactId === artifact.artifact_id) selectArtifact(null);
      await invalidate();
    },
  });

  const cancelJobMutation = useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onSuccess: invalidate,
  });

  const retryJobMutation = useMutation({
    mutationFn: (jobId: string) => api.retryJob(jobId),
    onSuccess: invalidate,
  });

  const replayRecipeMutation = useMutation({
    mutationFn: (recipeId: string) => api.replayRecipe(recipeId),
    onSuccess: invalidate,
  });

  const forkRecipeMutation = useMutation({
    mutationFn: ({ recipeId, payload }: { recipeId: string; payload: RecipeForkPayload }) => api.forkRecipe(recipeId, payload),
    onSuccess: async () => {
      setForkTarget(null);
      await invalidate();
    },
  });

  const annotateArtifact = useMutation({
    mutationFn: ({ artifactId, payload }: { artifactId: string; payload: ArtifactAnnotationPayload }) =>
      api.annotateArtifact(artifactId, payload),
    onSuccess: async () => {
      await invalidate();
    },
  });

  const generate = useMutation({
    mutationFn: () => {
      const payload = buildGenerationPayload({
        mode: generationMode,
        form: generationForm,
        sourceArtifact: generationSource,
        sessionId: activeSessionId,
      });
      if (generationMode === "generate.audio_to_audio") {
        return api.generateAudioToAudio(payload as AudioToAudioPayload);
      }
      if (generationMode === "generate.inpaint") {
        return api.generateInpaint(payload as InpaintPayload);
      }
      return api.generateText(payload as GenerateTextPayload);
    },
    onSuccess: invalidate,
  });

  const generatePromptCandidate = useMutation({
    mutationFn: (candidate: PromptCandidateGenerationRequest) => {
      const payload = buildGenerationPayload({
        mode: "generate.text_to_audio",
        form: generationForm,
        promptOverride: candidate.prompt,
        sourceArtifactId: candidate.bundleArtifactId,
        notes: `Prompt candidate ${candidate.rank ? `#${candidate.rank} ` : ""}from prompt-search bundle ${candidate.bundleArtifactId}`,
        metadata: {
          generation_origin: "prompt_search_candidate",
          prompt_search_bundle_id: candidate.bundleArtifactId,
          prompt_candidate_rank: candidate.rank ?? null,
          prompt_candidate_score: primitiveMetadataValue(candidate.score),
          prompt_candidate_source: candidate.source ?? null,
          prompt_search_scorer: candidate.scorer ?? null,
          prompt_search_mode: candidate.searchMode ?? null,
          prompt_search_model: candidate.searchModel ?? null,
          prompt_search_duration_seconds: candidate.searchDurationSeconds ?? null,
        },
        sessionId: activeSessionId,
      });
      return api.generateText(payload as GenerateTextPayload);
    },
    onSuccess: invalidate,
  });

  const encode = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.encodeLatent(buildLatentEncodePayload({ form: sameForm, artifact, sessionId: activeSessionId })),
    onSuccess: invalidate,
  });

  const decode = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.decodeLatent(buildLatentDecodePayload({ form: sameForm, artifact, sessionId: activeSessionId })),
    onSuccess: invalidate,
  });

  const runOperator = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.runOperator({
        operator,
        backend: operatorBackend(operatorForm, activeOperatorConfig.defaultBackend),
        inputs: {
          source: artifact.artifact_id,
          ...(donorArtifactId && operatorUsesDonor(activeOperatorConfig, operatorForm) ? { donor: donorArtifactId } : {}),
        },
        params: buildOperatorParams(activeOperatorConfig, operatorForm),
        seed: operatorSeed(operatorForm, generationSeedFallback(generationForm)),
        session_id: activeSessionId,
      }),
    onSuccess: invalidate,
  });

  const runExperiment = useMutation({
    mutationFn: () =>
      api.runExperiment(
        buildExperimentPayload({
          config: activeExperiment,
          form: experimentForm,
          selectedArtifact,
          sessionId: activeSessionId,
        }),
      ),
    onSuccess: invalidate,
  });

  const canRunExperiment = experimentReady(activeExperiment, experimentForm, selectedArtifact);

  const setExperimentField = (key: string, value: RecipeValue) => {
    setExperimentForm((current) => ({ ...current, [key]: value }));
  };

  const setGenerationField = (key: string, value: RecipeValue) => {
    setGenerationForm((current) => {
      const next = { ...current, [key]: value };
      if (key === "model") next.decoder = defaultDecoderForGenerationModel(value);
      return next;
    });
  };

  const setSameField = (key: string, value: RecipeValue) => {
    setSameForm((current) => ({ ...current, [key]: value }));
  };

  const setOperatorField = (key: string, value: RecipeValue) => {
    setOperatorForm((current) => ({ ...current, [key]: value }));
  };

  const updateOperatorPresets = (next: OperatorPreset[]) => {
    persistOperatorPresets(next);
    setOperatorPresets(next);
  };

  const saveOperatorPreset = () => {
    const existing = operatorPresets.find((preset) => preset.id === selectedOperatorPresetId);
    const preset = createOperatorPreset({
      id: existing?.id,
      createdAt: existing?.createdAt,
      name: operatorPresetName.trim() || existing?.name || `${activeOperatorConfig.label} sketch`,
      operator,
      form: operatorForm,
      donorArtifactId: donorArtifactId || null,
    });
    updateOperatorPresets(upsertOperatorPreset(operatorPresets, preset));
    setSelectedOperatorPresetId(preset.id);
    setOperatorPresetName(preset.name);
  };

  const applyOperatorPreset = (presetId: string) => {
    const preset = operatorPresets.find((item) => item.id === presetId);
    if (!preset || !isLatentOperatorMode(preset.operator)) return;
    setOperator(preset.operator);
    setOperatorForm({
      ...defaultOperatorForm(preset.operator),
      ...preset.form,
    });
    setDonorArtifactId(preset.donorArtifactId ?? "");
    setSelectedOperatorPresetId(preset.id);
    setOperatorPresetName(preset.name);
  };

  const removeOperatorPreset = () => {
    if (!selectedOperatorPresetId) return;
    updateOperatorPresets(deleteOperatorPreset(operatorPresets, selectedOperatorPresetId));
    setSelectedOperatorPresetId("");
    setOperatorPresetName("");
  };

  const selectOperatorMode = (mode: LatentOperatorMode) => {
    setOperator(mode);
    setOperatorForm(defaultOperatorForm(mode));
    setDonorArtifactId("");
    setSelectedOperatorPresetId("");
    setOperatorPresetName("");
  };

  const selectExperimentMode = (mode: ExperimentMode) => {
    setExperimentMode(mode);
    setExperimentForm(defaultExperimentForm(mode));
  };

  const useArtifactAsDonor = (artifactId: string) => {
    const artifact = allArtifacts.find((item) => item.artifact_id === artifactId);
    if (!artifact || artifact.kind !== "latent") return;
    const currentConfig = operatorCatalog.find((item) => item.value === operator) ?? operatorCatalog[0];
    if (!operatorUsesDonor(currentConfig, operatorForm)) {
      setOperator("latent.graft");
      setOperatorForm(defaultOperatorForm("latent.graft"));
    }
    setDonorArtifactId(artifactId);
  };

  const useBundleInRecipe = (fieldKey: string, path: string, mode: string) => {
    if (!isExperimentMode(mode)) return;
    setExperimentMode(mode);
    setExperimentForm({
      ...defaultExperimentForm(mode),
      [fieldKey]: path,
    });
  };

  const usePromptCandidate = (candidatePrompt: string) => {
    setGenerationMode("generate.text_to_audio");
    setGenerationForm((current) => ({ ...current, prompt: candidatePrompt }));
  };

  const runPromptCandidate = (candidate: PromptCandidateGenerationRequest) => {
    usePromptCandidate(candidate.prompt);
    generatePromptCandidate.mutate(candidate);
  };

  return (
    <main className="app-shell">
      <header className="top-strip">
        <div className="brand-mark">
          <img src={modelImage} alt="" />
          <div>
            <strong>SA3 Native Lab</strong>
            <span>{healthData?.artifact_root ?? ".sa3_lab"}</span>
          </div>
        </div>
        <div className="api-field">
          <label htmlFor="api-base">API</label>
          <input id="api-base" value={apiBase} onChange={(event) => setApiBase(event.target.value)} />
        </div>
        <BackendPills backends={healthData?.backends ?? []} />
      </header>

      <section className="bench-grid">
        <aside className="source-rail">
          <div className="rail-head">
            <div>
              <span className="eyebrow">Source</span>
              <strong>{visibleArtifacts.length} session artifacts</strong>
            </div>
            <label className="icon-button" title="Import audio">
              <Upload size={18} />
              <input
                type="file"
                accept="audio/*"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) importAudio.mutate(file);
                  event.currentTarget.value = "";
                }}
              />
            </label>
          </div>
          <ArtifactStack
            artifacts={visibleArtifacts}
            selectedId={selectedArtifact?.artifact_id ?? null}
            onSelect={selectArtifact}
            apiBase={apiBase}
          />
        </aside>

        <section className={`operator-surface ${selectedArtifact ? "has-selection" : "idle"}`}>
          <div className="surface-head">
            <div>
              <span className="eyebrow">Listening Bench</span>
              <h1>{selectedArtifact ? artifactName(selectedArtifact) : "No artifact selected"}</h1>
            </div>
            {selectedArtifact ? <ArtifactBadge artifact={selectedArtifact} /> : null}
          </div>

          <Specimen
            artifact={selectedArtifact}
            artifacts={allArtifacts}
            jobs={allJobs}
            families={resultFamilies}
            compare={compare}
            apiBase={apiBase}
            annotating={annotateArtifact.isPending}
            activeSessionId={activeSessionId}
            archivingArtifactId={archiveArtifact.isPending ? archiveArtifact.variables?.artifact.artifact_id ?? null : null}
            onAnnotate={(artifactId, payload) => annotateArtifact.mutate({ artifactId, payload })}
            onCompare={setCompare}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onForkRecipe={setForkTarget}
            onArchiveArtifact={(artifact) => archiveArtifact.mutate({ artifact, source: "specimen" })}
            onSelectArtifact={selectArtifact}
            onUseAsDonor={useArtifactAsDonor}
            onUseInRecipe={useBundleInRecipe}
            onUsePrompt={usePromptCandidate}
            onGeneratePrompt={runPromptCandidate}
            getArtifactPath={artifactPathForField}
          />
          <RunMonitor
            runningJobs={runningJobs}
            latestJob={latestJobs[0] ?? null}
            eventing={liveEventing}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />

          <div className="action-bands">
            <div className="band">
              <div className="band-title">
                <Wand2 size={18} />
                <span>Generate</span>
              </div>
              <SpecCoverage spec={activeGenerateSpec} controlledKeys={generationControlKeys(generationMode)} />
              <div className="segmented">
                {generationModes.map((mode) => (
                  <button key={mode.value} className={generationMode === mode.value ? "active" : ""} onClick={() => setGenerationMode(mode.value)}>
                    {mode.label}
                  </button>
                ))}
              </div>
              <RecipeFields
                config={activeGenerationConfig}
                form={generationForm}
                artifacts={allArtifacts}
                selectedArtifact={selectedArtifact}
                onChange={setGenerationField}
                getArtifactLabel={artifactName}
                getArtifactPath={artifactPathForField}
              />
              {generationNeedsSource && !generationSource ? <div className="quiet-panel compact">Select an audio artifact to use this mode.</div> : null}
              <button className="primary-action" onClick={() => generate.mutate()} disabled={!canGenerate || generate.isPending || Boolean(generateJob)}>
                {generate.isPending || generateJob ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
                {generateJob ? "MLX running" : generate.isPending ? "Queueing" : "Run MLX"}
              </button>
              <InlineJobStatus
                job={generateJob}
                onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
                onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
              />
            </div>

            <div className="band">
              <div className="band-title">
                <AudioLines size={18} />
                <span>SAME</span>
              </div>
              <SpecCoveragePair specs={[sameEncodeSpec, sameDecodeSpec]} controlledKeys={[sameEncodeControlKeys, sameDecodeControlKeys]} />
              <RecipeFields
                config={activeSameConfig}
                form={sameForm}
                artifacts={allArtifacts}
                selectedArtifact={selectedArtifact}
                onChange={setSameField}
                getArtifactLabel={artifactName}
                getArtifactPath={artifactPathForField}
              />
              <div className="two-actions">
                <button disabled={!selectedArtifact || selectedArtifact.kind !== "audio" || encode.isPending || Boolean(encodeJob)} onClick={() => selectedArtifact && encode.mutate(selectedArtifact)}>
                  {encode.isPending || encodeJob ? <LoaderCircle className="spin" size={17} /> : <Box size={17} />}
                  {encodeJob ? "Encoding" : "Encode"}
                </button>
                <button disabled={!selectedArtifact || selectedArtifact.kind !== "latent" || decode.isPending || Boolean(decodeJob)} onClick={() => selectedArtifact && decode.mutate(selectedArtifact)}>
                  {decode.isPending || decodeJob ? <LoaderCircle className="spin" size={17} /> : <Waves size={17} />}
                  {decodeJob ? "Decoding" : "Decode"}
                </button>
              </div>
              <InlineJobStatus
                job={encodeJob ?? decodeJob}
                onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
                onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
              />
            </div>

            <div className="band operator-band">
              <div className="band-title">
                <SlidersHorizontal size={18} />
                <span>Operator Studio</span>
              </div>
              <div className="recipe-mode-grid operator-mode-grid">
                <label>
                  Transform
                  <select value={operator} onChange={(event) => selectOperatorMode(event.target.value as LatentOperatorMode)}>
                    {operatorModes.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <span className={`recipe-chip ${activeOperatorConfig.maturity}`}>{activeOperatorConfig.family}</span>
                <span className={`recipe-chip ${activeOperatorConfig.maturity}`}>{activeOperatorConfig.maturity}</span>
              </div>
              <SpecCoverage spec={activeOperatorSpec} controlledKeys={fieldKeys(activeOperatorConfig)} />
              <OperatorPresetRack
                presets={activeOperatorPresets}
                selectedPreset={selectedOperatorPreset}
                selectedPresetId={selectedOperatorPresetId}
                presetName={operatorPresetName}
                diffRows={operatorPresetDiffs}
                onSelectPreset={applyOperatorPreset}
                onChangePresetName={setOperatorPresetName}
                onSavePreset={saveOperatorPreset}
                onResetPreset={() => selectedOperatorPreset && applyOperatorPreset(selectedOperatorPreset.id)}
                onDeletePreset={removeOperatorPreset}
              />
              {operatorUsesDonor(activeOperatorConfig, operatorForm) ? (
                <label className="control-cell donor-cell">
                  Donor latent
                  <select value={donorArtifactId} onChange={(event) => setDonorArtifactId(event.target.value)}>
                    <option value="">Select latent</option>
                    {latentArtifacts
                      .filter((artifact) => artifact.artifact_id !== selectedArtifact?.artifact_id)
                      .map((artifact) => (
                        <option key={artifact.artifact_id} value={artifact.artifact_id}>
                          {artifactName(artifact)}
                        </option>
                      ))}
                  </select>
                </label>
              ) : null}
              <RecipeFields
                config={activeOperatorConfig}
                form={operatorForm}
                artifacts={allArtifacts}
                selectedArtifact={selectedArtifact}
                onChange={setOperatorField}
                getArtifactPath={artifactPathForField}
                getArtifactLabel={artifactName}
              />
              <button className="primary-action" disabled={!canRunOperator || runOperator.isPending || Boolean(operatorJob)} onClick={() => selectedArtifact && runOperator.mutate(selectedArtifact)}>
                {runOperator.isPending || operatorJob ? <LoaderCircle className="spin" size={18} /> : <GitFork size={17} />}
                {operatorJob ? "Fork running" : "Fork latent"}
              </button>
              <InlineJobStatus
                job={operatorJob}
                onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
                onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
              />
            </div>

            <div className="band experiment-band">
              <div className="band-title">
                <FlaskConical size={18} />
                <span>Recipe Studio</span>
              </div>
              <div className="recipe-mode-grid">
                <label>
                  Mode
                  <select value={experimentMode} onChange={(event) => selectExperimentMode(event.target.value as ExperimentMode)}>
                    {experimentModes.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <span className={`recipe-chip ${activeExperiment.maturity}`}>{activeExperiment.family}</span>
                <span className={`recipe-chip ${activeExperiment.maturity}`}>{activeExperiment.maturity}</span>
              </div>
              <SpecCoverage spec={activeExperimentSpec} controlledKeys={fieldKeys(activeExperiment)} />
              {activeExperiment.value === "experiment.prompt_search" ? (
                <PromptSearchPresetRack
                  presets={promptSearchPresets}
                  vocabularySets={promptSearchVocabularySets}
                  axisSets={promptSearchAxisSets}
                  historyRows={promptSearchHistoryRows(allArtifacts)}
                  scorer={experimentForm.scorer}
                  onApply={(presetId) => setExperimentForm((current) => applyPromptSearchPreset(current, presetId))}
                  onApplyVocabulary={(setId) => setExperimentForm((current) => applyPromptSearchVocabularySet(current, setId))}
                  onApplyAxis={(setId) => setExperimentForm((current) => applyPromptSearchAxisSet(current, setId))}
                  onUseHistoryPrompt={(prompt) => setExperimentForm((current) => ({ ...current, seed_prompt: prompt }))}
                />
              ) : null}
              <RecipeFields
                config={activeExperiment}
                form={experimentForm}
                artifacts={allArtifacts}
                selectedArtifact={selectedArtifact}
                onChange={setExperimentField}
                getArtifactPath={artifactPathForField}
                getArtifactLabel={artifactName}
              />
              <button className="primary-action" disabled={!canRunExperiment || runExperiment.isPending || Boolean(experimentJob)} onClick={() => runExperiment.mutate()}>
                {runExperiment.isPending || experimentJob ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
                {experimentJob ? "Recipe running" : "Run recipe"}
              </button>
              <InlineJobStatus
                job={experimentJob}
                onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
                onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
              />
              <ModeAtlas modes={modeAtlasRows} activeOperator={activeExperiment.value} />
            </div>
          </div>
        </section>

        <aside className="result-rail">
          <div className="rail-head">
            <div>
              <span className="eyebrow">Result Family</span>
              <strong>{sessionResultFamilies.length} families</strong>
            </div>
            <Activity size={19} />
          </div>
          <ReadinessPanel checks={readinessChecks} />
          {forkTarget ? (
            <ForkRecipePanel
              recipe={forkTarget}
              submitting={forkRecipeMutation.isPending}
              onClose={() => setForkTarget(null)}
              onSubmit={(payload) =>
                forkRecipeMutation.mutate({
                  recipeId: forkTarget.recipe_id,
                  payload: { ...payload, session_id: activeSessionId },
                })
              }
            />
          ) : null}
          <ResultFamilyPanel
            families={sessionResultFamilies}
            artifacts={allArtifacts}
            selectedId={selectedArtifact?.artifact_id ?? null}
            inspectedFamilyId={inspectedFamily?.familyId ?? null}
            onSelect={selectArtifact}
            onInspectFamily={setInspectedFamilyId}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onForkRecipe={setForkTarget}
          />
          <FamilyDetailPanel
            family={inspectedFamily}
            families={sessionResultFamilies}
            artifacts={allArtifacts}
            jobs={allJobs}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            activeSessionId={activeSessionId}
            archivingArtifactId={archiveArtifact.isPending ? archiveArtifact.variables?.artifact.artifact_id ?? null : null}
            onSelect={selectArtifact}
            onInspectFamily={setInspectedFamilyId}
            onCompare={setCompare}
            onAnnotate={(artifactId, payload) => annotateArtifact.mutate({ artifactId, payload })}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onForkRecipe={setForkTarget}
            onArchiveArtifact={(artifact) => archiveArtifact.mutate({ artifact, source: "family_detail" })}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
          <SessionTray
            artifacts={sessionArtifacts}
            archivedArtifacts={archiveArtifacts}
            jobs={sessionJobs}
            archivedJobs={archiveJobs}
            families={resultFamilies}
            runningJobs={runningJobs}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            activeSessionId={activeSessionId}
            session={activeSession}
            sessionStartedAt={activeSession?.created_at ?? sessionStartedAt}
            creatingSession={createSession.isPending}
            archivingSession={archiveSession.isPending}
            recoveringArtifactId={recoverArtifact.isPending ? recoverArtifact.variables?.artifact_id ?? null : null}
            archivingArtifactId={archiveArtifact.isPending ? archiveArtifact.variables?.artifact.artifact_id ?? null : null}
            onSelect={selectArtifact}
            onStartSession={() => createSession.mutate()}
            onArchiveSession={() => activeSession && archiveSession.mutate(activeSession)}
            onRecoverArtifact={(artifact) => recoverArtifact.mutate(artifact)}
            onArchiveArtifact={(artifact) => archiveArtifact.mutate({ artifact, source: "session_tray" })}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
          <ComparePanel a={compareA} b={compareB} apiBase={apiBase} />
          <AuditionStackPanel
            artifacts={sessionArtifacts}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            onSelect={selectArtifact}
            onCompare={setCompare}
          />
          <div className="mini-counts">
            <span><FileAudio size={15} /> {audioArtifacts.length}</span>
            <span><Braces size={15} /> {latentArtifacts.length}</span>
            <span><Box size={15} /> {bundleArtifacts.length}</span>
          </div>
        </aside>
      </section>
    </main>
  );
}

function PromptSearchPresetRack({
  presets,
  vocabularySets,
  axisSets,
  historyRows,
  scorer,
  onApply,
  onApplyVocabulary,
  onApplyAxis,
  onUseHistoryPrompt,
}: {
  presets: readonly PromptSearchPreset[];
  vocabularySets: readonly PromptSearchVocabularySet[];
  axisSets: readonly PromptSearchAxisSet[];
  historyRows: readonly PromptSearchHistoryRow[];
  scorer: RecipeValue | undefined;
  onApply: (presetId: string) => void;
  onApplyVocabulary: (setId: string) => void;
  onApplyAxis: (setId: string) => void;
  onUseHistoryPrompt: (prompt: string) => void;
}) {
  const note = promptSearchScorerNote(scorer);
  return (
    <div className="prompt-search-guide">
      <div className="prompt-search-preset-rack" aria-label="Prompt search presets">
        {presets.map((preset) => (
          <button key={preset.id} type="button" onClick={() => onApply(preset.id)} title={preset.intent}>
            <Search aria-hidden="true" size={13} />
            <span>{preset.label}</span>
            <small>{preset.modeLabel} · {preset.cost}</small>
          </button>
        ))}
      </div>
      <div className={`prompt-search-scorer-note ${note.maturity}`}>
        <strong>{note.label}</strong>
        <span>{note.cost}</span>
        <p>{note.guidance}</p>
      </div>
      <div className="prompt-search-token-tools" aria-label="Prompt search vocabulary tools">
        <div>
          <strong>Vocabulary</strong>
          <span>{vocabularySets.length} sets</span>
        </div>
        <div className="prompt-search-tool-buttons">
          {vocabularySets.map((set) => (
            <button key={set.id} type="button" onClick={() => onApplyVocabulary(set.id)} title={set.terms}>
              <Wand2 aria-hidden="true" size={13} />
              <span>{set.label}</span>
              <small>{set.focus}</small>
            </button>
          ))}
        </div>
        <div>
          <strong>Axes</strong>
          <span>Mode 3</span>
        </div>
        <div className="prompt-search-tool-buttons">
          {axisSets.map((set) => (
            <button key={set.id} type="button" onClick={() => onApplyAxis(set.id)} title={set.axes}>
              <SlidersHorizontal aria-hidden="true" size={13} />
              <span>{set.label}</span>
              <small>{set.focus}</small>
            </button>
          ))}
        </div>
      </div>
      {historyRows.length ? (
        <div className="prompt-search-history" aria-label="Prompt search history">
          <div>
            <strong>Prompt history</strong>
            <span>{historyRows.length} prompts</span>
          </div>
          {historyRows.slice(0, 4).map((row) => (
            <button key={row.prompt} type="button" onClick={() => onUseHistoryPrompt(row.prompt)} title={row.latestNote ?? row.prompt}>
              <span>{row.prompt}</span>
              <small>{row.keeper}K · {row.maybe}M · {row.rejected}R · {row.total} take{row.total === 1 ? "" : "s"}</small>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function AuditionStackPanel({
  artifacts,
  selectedId,
  apiBase,
  onSelect,
  onCompare,
}: {
  artifacts: ArtifactRecord[];
  selectedId: string | null;
  apiBase: string;
  onSelect: (artifactId: string) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
}) {
  const [sequenceMode, setSequenceMode] = useState<AuditionSequenceMode>("recent");
  const rows = auditionStackRows(artifacts, 8, sequenceMode, selectedId);
  const cursor = auditionCursor(artifacts, selectedId, 8, sequenceMode);
  const position = auditionPositionLabel(artifacts, selectedId, 8, sequenceMode);
  if (!rows.length) return null;
  const moveSelection = (key: string) => {
    const target = auditionKeyboardTarget(artifacts, selectedId, key, 8, sequenceMode);
    if (!target) return false;
    onSelect(target.artifact_id);
    return true;
  };
  return (
    <div
      className="audition-stack"
      aria-label="Session audition stack"
      tabIndex={0}
      onKeyDown={(event) => {
        if (!moveSelection(event.key)) return;
        event.preventDefault();
      }}
    >
      <div className="audition-stack-head">
        <div>
          <span className="eyebrow">Audition</span>
          <strong>{position}</strong>
        </div>
        <label className="audition-sequence">
          <span>Sequence</span>
          <select value={sequenceMode} onChange={(event) => setSequenceMode(event.target.value as AuditionSequenceMode)}>
            {auditionSequenceOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="audition-transport">
          <button type="button" disabled={!cursor.previous} onClick={() => cursor.previous && onSelect(cursor.previous.artifact_id)} title="Previous take">
            <SkipBack size={14} />
          </button>
          <button type="button" disabled={!cursor.next} onClick={() => cursor.next && onSelect(cursor.next.artifact_id)} title="Next take">
            <SkipForward size={14} />
          </button>
          <button type="button" disabled={!cursor.selected} onClick={() => cursor.selected && onCompare("a", cursor.selected.artifact_id)} title="Send selected take to A">
            A
          </button>
          <button type="button" disabled={!cursor.selected} onClick={() => cursor.selected && onCompare("b", cursor.selected.artifact_id)} title="Send selected take to B">
            B
          </button>
        </div>
      </div>
      {rows.map((row) => {
        const artifact = artifacts.find((item) => item.artifact_id === row.artifactId);
        if (!artifact) return null;
        return (
          <article key={row.artifactId} className={selectedId === row.artifactId ? "selected" : ""}>
            <button type="button" onClick={() => onSelect(row.artifactId)} title={row.prompt ?? row.label}>
              <span>{row.label}</span>
              <small>{row.sequence} · {row.origin} · {row.meta}</small>
              <ListeningDecisionBadge artifact={artifact} />
            </button>
            <AudioDeck artifact={artifact} apiBase={apiBase} compact />
            <div className="audition-stack-actions">
              <button type="button" onClick={() => onCompare("a", row.artifactId)} title="Send take to comparison slot A">A</button>
              <button type="button" onClick={() => onCompare("b", row.artifactId)} title="Send take to comparison slot B">B</button>
            </div>
          </article>
        );
      })}
    </div>
  );
}

function BackendPills({ backends }: { backends: ModelStatus[] }) {
  return (
    <div className="backend-pills">
      {backends.map((backend) => (
        <span key={backend.backend} className={backend.available ? "ready" : "offline"} title={backend.message ?? backend.backend}>
          {backend.available ? <Check size={14} /> : <CircleAlert size={14} />}
          {backend.backend}
        </span>
      ))}
    </div>
  );
}

function ArtifactStack({
  artifacts,
  selectedId,
  onSelect,
  apiBase,
}: {
  artifacts: ArtifactRecord[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  apiBase: string;
}) {
  if (!artifacts.length) {
    return (
      <div className="empty-panel">
        <Upload size={22} />
        <strong>Import audio</strong>
      </div>
    );
  }
  return (
    <div className="artifact-stack">
      {artifacts.map((artifact) => (
        <button
          key={artifact.artifact_id}
          className={`artifact-row ${selectedId === artifact.artifact_id ? "selected" : ""}`}
          onClick={() => onSelect(artifact.artifact_id)}
        >
          <ArtifactIcon artifact={artifact} />
          <div>
            <strong>{artifactName(artifact)}</strong>
            <span>{artifactMeta(artifact)}</span>
          </div>
          {artifact.kind === "audio" ? <TinyWave artifact={artifact} apiBase={apiBase} /> : null}
        </button>
      ))}
    </div>
  );
}

function BundleFieldFallback({ artifact }: { artifact: ArtifactRecord }) {
  const cells = artifact.recipe_id ? 36 : 18;
  return (
    <div className="bundle-field bundle-field-loading" aria-label={`Experiment bundle ${artifactName(artifact)}`}>
      {Array.from({ length: cells }, (_, index) => (
        <span key={index} />
      ))}
      <div className="bundle-readout">
        <strong>{artifact.file?.filename ?? artifactName(artifact)}</strong>
        <span>Loading bundle inspector</span>
      </div>
    </div>
  );
}

function Specimen({
  artifact,
  artifacts,
  jobs,
  families,
  compare,
  apiBase,
  annotating,
  activeSessionId,
  archivingArtifactId,
  onAnnotate,
  onCompare,
  onReplayRecipe,
  onForkRecipe,
  onArchiveArtifact,
  onSelectArtifact,
  onUseAsDonor,
  onUseInRecipe,
  onUsePrompt,
  onGeneratePrompt,
  getArtifactPath,
}: {
  artifact: ArtifactRecord | null;
  artifacts: ArtifactRecord[];
  jobs: JobRecord[];
  families: ResultFamily[];
  compare: CompareSlots;
  apiBase: string;
  annotating: boolean;
  activeSessionId: string | null;
  archivingArtifactId: string | null;
  onAnnotate: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onReplayRecipe: (recipeId: string) => void;
  onForkRecipe: (recipe: Recipe) => void;
  onArchiveArtifact: (artifact: ArtifactRecord) => void;
  onSelectArtifact: (artifactId: string | null) => void;
  onUseAsDonor: (artifactId: string) => void;
  onUseInRecipe: (fieldKey: string, path: string, mode: string) => void;
  onUsePrompt: (prompt: string) => void;
  onGeneratePrompt: (request: PromptCandidateGenerationRequest) => void;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
}) {
  if (!artifact) {
    return (
      <div className="specimen empty">
        <CircleDot size={24} />
      </div>
    );
  }
  const fileUrl = `${apiBase}/artifacts/${artifact.artifact_id}/file`;
  const sourceArtifacts = artifact.source_artifact_ids
    .map((artifactId) => artifacts.find((item) => item.artifact_id === artifactId))
    .filter((item): item is ArtifactRecord => Boolean(item));
  const artifactRecipe = artifact.recipe_id ? jobs.find((job) => job.recipe.recipe_id === artifact.recipe_id)?.recipe ?? null : null;
  const canArchiveArtifact = Boolean(activeSessionId && artifact.session_id === activeSessionId);
  return (
    <div className="specimen">
      <div className={`wave-bus ${waveBusStateClass({ artifact, sources: sourceArtifacts, jobs, families, compare })}`}>
        {artifact.kind === "audio" ? (
          <AudioDeck artifact={artifact} apiBase={apiBase} onAnnotate={onAnnotate} persisting={annotating} />
        ) : artifact.kind === "latent" ? (
          <LatentField artifact={artifact} />
        ) : (
          <Suspense fallback={<BundleFieldFallback artifact={artifact} />}>
            <BundleField
              artifact={artifact}
              artifacts={artifacts}
              apiBase={apiBase}
              onCompare={onCompare}
              onSelectArtifact={onSelectArtifact}
              onUseAsDonor={onUseAsDonor}
              onUseInRecipe={onUseInRecipe}
              onUsePrompt={onUsePrompt}
              onGeneratePrompt={onGeneratePrompt}
              onAnnotate={onAnnotate}
              getArtifactPath={getArtifactPath}
            />
          </Suspense>
        )}
        <LineageThread
          artifact={artifact}
          sources={sourceArtifacts}
          jobs={jobs}
          families={families}
          compare={compare}
          onSelectArtifact={onSelectArtifact}
        />
      </div>
      <div className="specimen-info">
        <dl>
          <div>
            <dt>ID</dt>
            <dd>{artifact.artifact_id}</dd>
          </div>
          <div>
            <dt>Lineage</dt>
            <dd>{artifact.source_artifact_ids.length || 0}</dd>
          </div>
          <div>
            <dt>Recipe</dt>
            <dd>{artifact.recipe_id ?? "source"}</dd>
          </div>
          <div>
            <dt>Shape</dt>
            <dd>{artifactShape(artifact)}</dd>
          </div>
        </dl>
        <div className="specimen-actions">
          <button disabled={artifact.kind !== "audio"} onClick={() => onCompare("a", artifact.artifact_id)}>
            A
          </button>
          <button disabled={artifact.kind !== "audio"} onClick={() => onCompare("b", artifact.artifact_id)}>
            B
          </button>
          <button
            type="button"
            aria-label="Replay recipe"
            disabled={!artifact.recipe_id}
            onClick={() => {
              if (artifact.recipe_id) onReplayRecipe(artifact.recipe_id);
            }}
            title="Replay recipe"
          >
            <Repeat size={17} />
          </button>
          <button
            type="button"
            aria-label="Fork recipe"
            disabled={!artifactRecipe}
            onClick={() => {
              if (artifactRecipe) onForkRecipe(artifactRecipe);
            }}
            title="Fork recipe"
          >
            <GitFork size={17} />
          </button>
          <button
            type="button"
            aria-label="Archive artifact"
            disabled={!canArchiveArtifact || archivingArtifactId === artifact.artifact_id}
            onClick={() => onArchiveArtifact(artifact)}
            title="Archive artifact"
          >
            <Archive size={17} />
          </button>
          <a className="icon-link" href={fileUrl} download title="Download artifact">
            <Download size={17} />
          </a>
        </div>
        <ArtifactVitals artifact={artifact} />
        <ArtifactAnnotationPanel artifact={artifact} saving={annotating} onSave={onAnnotate} />
      </div>
    </div>
  );
}

function ArtifactVitals({ artifact }: { artifact: ArtifactRecord }) {
  const rows = artifactVitalRows(artifact);
  if (!rows.length) return null;
  return (
    <div className={`artifact-vitals ${artifact.kind}`} aria-label="Artifact inspector">
      {rows.map(([label, value]) => (
        <span key={label}>
          <small>{label}</small>
          <strong>{value}</strong>
        </span>
      ))}
    </div>
  );
}

function artifactVitalRows(artifact: ArtifactRecord): [string, string][] {
  if (artifact.kind === "audio" && artifact.audio) {
    return [
      ["duration", artifact.audio.duration_seconds ? formatDuration(artifact.audio.duration_seconds) : "unknown"],
      ["rate", `${artifact.audio.sample_rate} Hz`],
      ["channels", String(artifact.audio.channels)],
      ["frames", artifact.audio.frames.toLocaleString()],
    ];
  }
  if (artifact.kind === "latent" && artifact.latent) {
    return [
      ["shape", artifact.latent.shape.join(" x ")],
      ["latent rate", `${artifact.latent.latent_rate.toFixed(2)} Hz`],
      ["duration", artifact.latent.duration_seconds ? formatDuration(artifact.latent.duration_seconds) : "unknown"],
      ["layout", artifact.latent.channel_first ? "channel-first" : "time-first"],
    ];
  }
  if (artifact.kind === "bundle") {
    const operator = typeof artifact.metadata.operator === "string" ? artifact.metadata.operator : "bundle";
    const resultCount = typeof artifact.metadata.result_count === "number" ? String(artifact.metadata.result_count) : null;
    return [
      ["kind", "bundle"],
      ["size", artifact.file ? artifactMeta(artifact).replace(" bundle", "") : "unknown"],
      ["operator", shortOperatorName(operator as OperatorName)],
      ["results", resultCount ?? "inspect"],
    ];
  }
  return [
    ["kind", artifact.kind],
    ["sources", String(artifact.source_artifact_ids.length)],
    ["recipe", artifact.recipe_id ?? "source"],
  ];
}

function ArtifactAnnotationPanel({
  artifact,
  saving,
  onSave,
}: {
  artifact: ArtifactRecord;
  saving: boolean;
  onSave: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
}) {
  const [label, setLabel] = useState(artifact.label ?? "");
  const [tags, setTags] = useState(artifact.tags.join(", "));
  const [notes, setNotes] = useState(artifact.notes ?? "");

  useEffect(() => {
    setLabel(artifact.label ?? "");
    setTags(artifact.tags.join(", "));
    setNotes(artifact.notes ?? "");
  }, [artifact.artifact_id, artifact.label, artifact.notes, artifact.tags]);

  return (
    <div className="annotation-panel">
      <label>
        Label
        <input value={label} onChange={(event) => setLabel(event.target.value)} placeholder="keeper, brittle, needs graft..." />
      </label>
      <label>
        Tags
        <input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="favorite, loop, noisy" />
      </label>
      <label>
        Notes
        <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="What should future-you remember?" />
      </label>
      <button
        type="button"
        className="annotation-save"
        disabled={saving}
        onClick={() =>
          onSave(artifact.artifact_id, {
            label: label.trim() || null,
            notes: notes.trim() || null,
            tags: parseTags(tags),
          })
        }
      >
        {saving ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}
        Save annotation
      </button>
    </div>
  );
}

function RunMonitor({
  runningJobs,
  latestJob,
  eventing = false,
  onCancelJob,
  onRetryJob,
}: {
  runningJobs: JobRecord[];
  latestJob: JobRecord | null;
  eventing?: boolean;
} & JobActionHandlers) {
  const monitorJobs = runningJobs.length ? runningJobs.slice(0, 3) : latestJob ? [latestJob] : [];
  if (!monitorJobs.length) {
    return (
      <div className="run-monitor idle">
        <div>
          <span className="eyebrow">Run Monitor</span>
          <strong>Ready</strong>
        </div>
        <span className="monitor-state">idle</span>
      </div>
    );
  }

  const busy = runningJobs.length > 0;
  return (
    <div className={`run-monitor ${busy ? "busy" : "idle"}`}>
      <div className="run-monitor-head">
        <div>
          <span className="eyebrow">Run Monitor</span>
          <strong>{busy ? `${runningJobs.length} active job${runningJobs.length === 1 ? "" : "s"}` : "Last run"}</strong>
        </div>
        <span className={`monitor-state ${eventing ? "live" : ""}`}>{busy ? (eventing ? "live events" : "running") : latestJob?.status ?? "idle"}</span>
      </div>
      <div className="monitor-jobs">
        {monitorJobs.map((job) => (
          <JobProgress key={job.job_id} job={job} compact={monitorJobs.length > 1} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
        ))}
      </div>
    </div>
  );
}

function InlineJobStatus({ job, onCancelJob, onRetryJob }: { job: JobRecord | null | undefined } & JobActionHandlers) {
  if (!job) return null;
  return (
    <div className="inline-job-status">
      <JobProgress job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
    </div>
  );
}

function ReadinessPanel({ checks }: { checks: ReadinessCheck[] }) {
  const rows = priorityReadinessChecks(checks);
  const errorCount = checks.filter((check) => check.status === "error").length;
  const warnCount = checks.filter((check) => check.status === "warn").length;
  const state = errorCount ? "error" : warnCount ? "warn" : "ok";
  return (
    <details className={`readiness-panel ${state}`}>
      <summary>
        <span>
          <Gauge size={15} />
          Readiness
        </span>
        <strong>{state}</strong>
      </summary>
      <div className="readiness-list">
        {rows.map((check) => (
          <div key={check.name} className={`readiness-row ${check.status}`}>
            <span>{readinessLabel(check.name)}</span>
            <strong>{check.status}</strong>
            <small title={check.detail ?? check.message}>{check.message}</small>
          </div>
        ))}
      </div>
    </details>
  );
}

function SessionTray({
  artifacts,
  archivedArtifacts,
  jobs,
  archivedJobs,
  families,
  runningJobs,
  selectedId,
  apiBase,
  activeSessionId,
  session,
  sessionStartedAt,
  creatingSession,
  archivingSession,
  recoveringArtifactId,
  archivingArtifactId,
  onSelect,
  onStartSession,
  onArchiveSession,
  onRecoverArtifact,
  onArchiveArtifact,
  onCancelJob,
  onRetryJob,
}: {
  artifacts: ArtifactRecord[];
  archivedArtifacts: ArtifactRecord[];
  jobs: JobRecord[];
  archivedJobs: JobRecord[];
  families: ResultFamily[];
  runningJobs: JobRecord[];
  selectedId: string | null;
  apiBase: string;
  activeSessionId: string | null;
  session: SessionRecord | null;
  sessionStartedAt: string;
  creatingSession: boolean;
  archivingSession: boolean;
  recoveringArtifactId: string | null;
  archivingArtifactId: string | null;
  onSelect: (artifactId: string | null) => void;
  onStartSession: () => void;
  onArchiveSession: () => void;
  onRecoverArtifact: (artifact: ArtifactRecord) => void;
  onArchiveArtifact: (artifact: ArtifactRecord) => void;
} & JobActionHandlers) {
  const [artifactFilters, setArtifactFilters] = useState<ArtifactFilterState>(emptyArtifactFilters);
  const filterContext = useMemo(() => ({ jobs: [...jobs, ...archivedJobs], families }), [jobs, archivedJobs, families]);
  const filterCorpus = useMemo(() => [...artifacts, ...archivedArtifacts], [artifacts, archivedArtifacts]);
  const filterOptions = useMemo(() => artifactFilterOptions(filterCorpus, filterContext), [filterCorpus, filterContext]);
  const filterActive = artifactFiltersActive(artifactFilters);
  const filteredSessionArtifacts = useMemo(() => filterArtifacts(artifacts, artifactFilters, filterContext), [artifacts, artifactFilters, filterContext]);
  const filteredArchiveArtifacts = useMemo(() => filterArtifacts(archivedArtifacts, artifactFilters, filterContext), [archivedArtifacts, artifactFilters, filterContext]);
  const sessionArtifactRows = sortNewest(filteredSessionArtifacts).slice(0, 8);
  const sessionJobs = sortNewestJobs(jobs).slice(0, 4);
  const archiveArtifactRows = sortNewest(filteredArchiveArtifacts).slice(0, 10);
  const archiveJobRows = filterActive ? [] : sortNewestJobs(archivedJobs).slice(0, 10);
  const activeJobs = runningJobs.slice(0, 3);
  const archivableIds = useMemo(
    () => new Set(archivableSessionArtifacts(sessionArtifactRows, activeSessionId).map((artifact) => artifact.artifact_id)),
    [sessionArtifactRows, activeSessionId],
  );
  const recoverableIds = useMemo(
    () => new Set(recoverableArchiveArtifacts(archiveArtifactRows, activeSessionId).map((artifact) => artifact.artifact_id)),
    [archiveArtifactRows, activeSessionId],
  );
  const workspaceSummary = useMemo(
    () => summarizeSessionWorkspace({ artifacts, archivedArtifacts, jobs, archivedJobs, families, runningJobs, selectedId }),
    [artifacts, archivedArtifacts, jobs, archivedJobs, families, runningJobs, selectedId],
  );
  const pulseRows = useMemo(() => workspacePulseRows(workspaceSummary), [workspaceSummary]);
  const focus = useMemo(() => workspaceFocus(workspaceSummary), [workspaceSummary]);

  return (
    <div className="session-tray">
      <div className="session-head">
        <div>
          <span className="eyebrow">Session</span>
          <strong>{session?.name ?? formatSessionStamp(sessionStartedAt)}</strong>
        </div>
        <div className="session-actions">
          <button type="button" className="session-new" onClick={onStartSession} title="New session" disabled={creatingSession || archivingSession}>
            {creatingSession ? <LoaderCircle className="spin" size={16} /> : <Plus size={16} />}
            {creatingSession ? "Creating" : "New"}
          </button>
          <button
            type="button"
            className="session-new"
            onClick={onArchiveSession}
            title="Archive this session and start a clean one"
            disabled={!session || session.status === "archived" || creatingSession || archivingSession || activeJobs.length > 0}
          >
            {archivingSession ? <LoaderCircle className="spin" size={16} /> : <Archive size={16} />}
            {archivingSession ? "Archiving" : "Archive"}
          </button>
        </div>
      </div>

      <WorkspacePulse rows={pulseRows} focus={focus} onSelect={onSelect} />

      {activeJobs.length ? (
        <div className="session-block">
          <span className="session-label">Running</span>
          {activeJobs.map((job) => (
            <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
          ))}
        </div>
      ) : null}

      <ArtifactFilterPanel
        filters={artifactFilters}
        options={filterOptions}
        active={filterActive}
        onChange={setArtifactFilters}
      />

      <div className="session-block">
        <span className="session-label">
          Takes
          {filterActive ? <i>{filteredSessionArtifacts.length}/{artifacts.length}</i> : null}
        </span>
        {sessionArtifactRows.length ? (
          <div className="session-artifacts">
            {sessionArtifactRows.map((artifact) => (
              <SessionArtifactRow
                key={artifact.artifact_id}
                artifact={artifact}
                selected={artifact.artifact_id === selectedId}
                apiBase={apiBase}
                onSelect={onSelect}
                action={
                  archivableIds.has(artifact.artifact_id)
                    ? {
                        label: archivingArtifactId === artifact.artifact_id ? "Archiving" : "Archive",
                        disabled: Boolean(archivingArtifactId),
                        onAction: () => onArchiveArtifact(artifact),
                      }
                    : undefined
                }
              />
            ))}
          </div>
        ) : (
          <div className="quiet-panel compact">{filterActive ? "No matching takes" : "Fresh session"}</div>
        )}
      </div>

      {sessionJobs.length ? (
        <details className="archive-drawer">
          <summary>
            <Gauge size={15} />
            Session jobs
            <span>{jobs.length}</span>
          </summary>
          <div className="archive-list">
            {sessionJobs.map((job) => (
              <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
            ))}
          </div>
        </details>
      ) : null}

      <details className="archive-drawer">
        <summary>
          <Database size={15} />
          Archive
          <span>{filterActive ? `${filteredArchiveArtifacts.length}/${archivedArtifacts.length}` : archivedArtifacts.length + archivedJobs.length}</span>
        </summary>
        <div className="archive-list">
          {archiveArtifactRows.map((artifact) => (
            <SessionArtifactRow
              key={artifact.artifact_id}
              artifact={artifact}
              selected={artifact.artifact_id === selectedId}
              apiBase={apiBase}
              onSelect={onSelect}
              action={
                recoverableIds.has(artifact.artifact_id)
                  ? {
                      label: recoveringArtifactId === artifact.artifact_id ? "Recovering" : "Recover",
                      disabled: Boolean(recoveringArtifactId),
                      onAction: () => onRecoverArtifact(artifact),
                    }
                  : undefined
              }
            />
          ))}
          {archiveJobRows.map((job) => (
            <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
          ))}
          {!archiveArtifactRows.length && !archiveJobRows.length ? <div className="quiet-panel compact">{filterActive ? "No matching takes" : "Archive empty"}</div> : null}
        </div>
      </details>
    </div>
  );
}

function WorkspacePulse({
  rows,
  focus,
  onSelect,
}: {
  rows: WorkspacePulseRow[];
  focus: WorkspaceFocus;
  onSelect: (artifactId: string | null) => void;
}) {
  return (
    <section className="workspace-pulse" aria-label="Workspace pulse">
      <div className={`workspace-focus ${focus.tone}`}>
        <span>
          <CircleDot size={14} />
          {focus.label}
        </span>
        <strong>{focus.detail}</strong>
        {focus.artifactId ? (
          <button type="button" onClick={() => onSelect(focus.artifactId ?? null)}>
            Open
          </button>
        ) : null}
      </div>
      <div className="workspace-pulse-grid">
        {rows.map((row) => (
          <div key={row.key} className={`workspace-pulse-node ${row.tone}`}>
            <span>{row.label}</span>
            <strong>{row.value}</strong>
            <small>{row.detail}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

const decisionFilters: { value: ArtifactDecisionFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "keeper", label: "Keep" },
  { value: "maybe", label: "Maybe" },
  { value: "rejected", label: "Reject" },
  { value: "undecided", label: "Open" },
];

function ArtifactFilterPanel({
  filters,
  options,
  active,
  onChange,
}: {
  filters: ArtifactFilterState;
  options: ReturnType<typeof artifactFilterOptions>;
  active: boolean;
  onChange: (filters: ArtifactFilterState) => void;
}) {
  const update = (patch: Partial<ArtifactFilterState>) => onChange({ ...filters, ...patch });
  return (
    <div className="artifact-filter-panel" aria-label="Take filters">
      <div className="artifact-filter-head">
        <span className="session-label">
          <SlidersHorizontal size={13} />
          Find takes
        </span>
        {active ? (
          <button type="button" className="archive-clear" aria-label="Clear take filters" onClick={() => onChange(emptyArtifactFilters)}>
            <X size={14} />
          </button>
        ) : null}
      </div>
      <label className="artifact-filter-search">
        <Search size={14} />
        <input
          type="search"
          aria-label="Filter takes"
          value={filters.query}
          onChange={(event) => update({ query: event.target.value })}
          placeholder="label, notes, prompt, tags"
        />
      </label>
      <div className="artifact-decision-filter" aria-label="Listening decision filters">
        {decisionFilters.map((item) => (
          <button
            key={item.value}
            type="button"
            aria-pressed={filters.decision === item.value}
            className={filters.decision === item.value ? `selected ${item.value}` : item.value}
            onClick={() => update({ decision: item.value })}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="artifact-filter-selects">
        <FilterSelect
          label="Kind"
          value={filters.kind === "all" ? "" : filters.kind}
          allLabel="all kinds"
          options={options.kinds}
          onChange={(value) => update({ kind: (value || "all") as ArtifactKindFilter })}
        />
        <FilterSelect
          label="Model"
          value={filters.model}
          allLabel="any model"
          options={options.models}
          onChange={(model) => update({ model })}
        />
        <FilterSelect
          label="Operator"
          value={filters.operator}
          allLabel="any operator"
          options={options.operators.map((option) => ({ ...option, label: shortOperatorName(option.value as OperatorName) }))}
          onChange={(operator) => update({ operator })}
        />
        <FilterSelect
          label="Family"
          value={filters.familyId}
          allLabel="any family"
          options={options.families}
          onChange={(familyId) => update({ familyId })}
        />
        <FilterSelect
          label="Lineage"
          value={filters.lineage === "all" ? "" : filters.lineage}
          allLabel="any lineage"
          options={[
            { value: "source", label: "source", count: 0 },
            { value: "derived", label: "derived", count: 0 },
            { value: "has_sources", label: "has source", count: 0 },
          ]}
          onChange={(lineage) => update({ lineage: (lineage || "all") as ArtifactLineageFilter })}
        />
        <FilterSelect
          label="Tag"
          value={filters.tag}
          allLabel="any tag"
          options={options.tags}
          onChange={(tag) => update({ tag })}
        />
      </div>
    </div>
  );
}

function FilterSelect({
  label,
  value,
  allLabel,
  options,
  onChange,
}: {
  label: string;
  value: string;
  allLabel: string;
  options: ArtifactFilterOption[];
  onChange: (value: string) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <select aria-label={`Filter ${label.toLowerCase()}`} value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{allLabel}</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.count ? `${option.label} (${option.count})` : option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function SessionArtifactRow({
  artifact,
  selected,
  apiBase,
  onSelect,
  action,
}: {
  artifact: ArtifactRecord;
  selected: boolean;
  apiBase: string;
  onSelect: (artifactId: string | null) => void;
  action?: {
    label: string;
    disabled?: boolean;
    onAction: () => void;
  };
}) {
  return (
    <article className={`session-artifact ${selected ? "selected" : ""}`}>
      <button type="button" className="session-artifact-main" onClick={() => onSelect(artifact.artifact_id)}>
        <ArtifactIcon artifact={artifact} />
        <div>
          <strong>{artifactName(artifact)}</strong>
          <span>{artifactMeta(artifact)}</span>
          <ListeningDecisionBadge artifact={artifact} />
          {artifact.tags.length ? (
            <span className="artifact-tags">
              {artifact.tags.slice(0, 3).map((tag) => (
                <i key={tag}>#{tag}</i>
              ))}
            </span>
          ) : null}
        </div>
        {artifact.kind === "audio" ? <TinyWave artifact={artifact} apiBase={apiBase} /> : null}
      </button>
      {action ? (
        <button type="button" className="session-artifact-action" disabled={action.disabled} onClick={action.onAction}>
          {action.label}
        </button>
      ) : null}
    </article>
  );
}

function ComparePanel({ a, b, apiBase }: { a: ArtifactRecord | null; b: ArtifactRecord | null; apiBase: string }) {
  return (
    <div className="compare-panel">
      <div className="band-title">
        <FlaskConical size={18} />
        <span>A/B</span>
      </div>
      <CompareSlot label="A" artifact={a} apiBase={apiBase} />
      <CompareSlot label="B" artifact={b} apiBase={apiBase} />
    </div>
  );
}

function CompareSlot({ label, artifact, apiBase }: { label: string; artifact: ArtifactRecord | null; apiBase: string }) {
  return (
    <div className="compare-slot">
      <strong>{label}</strong>
      {artifact ? (
        <div>
          <span>{artifactName(artifact)}</span>
          <AudioDeck artifact={artifact} apiBase={apiBase} compact />
        </div>
      ) : (
        <span className="muted">empty</span>
      )}
    </div>
  );
}

function ModeAtlas({ modes, activeOperator }: { modes: NotebookMode[]; activeOperator: OperatorName }) {
  if (!modes.length) {
    return null;
  }
  return (
    <div className="mode-atlas">
      <div className="mode-atlas-head">
        <span>
          <Database size={16} />
          Colab Mode Atlas
        </span>
        <strong>{modes.length}</strong>
      </div>
      <div className="mode-atlas-list">
        {modes.map((mode) => {
          const active = mode.operators.includes(activeOperator);
          return (
            <article key={mode.mode_id} className={`mode-card ${active ? "active" : ""}`}>
              <div>
                <strong>
                  {mode.mode_id}. {mode.title}
                </strong>
                <span>{mode.native_surface}</span>
              </div>
              <span className={`mode-status ${statusClass(mode.status)}`}>{mode.status}</span>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function LatentField({ artifact }: { artifact: ArtifactRecord }) {
  const shape = artifact.latent?.shape ?? [1, 1];
  const rows = Math.min(6, Math.max(2, shape[1]));
  const columns = Math.min(18, Math.max(6, shape[0]));
  return (
    <div className="latent-field" aria-label={`Latent tensor ${shape.join(" by ")}`}>
      {Array.from({ length: rows * columns }, (_, index) => (
        <span key={index} style={{ animationDelay: `${(index % columns) * 28}ms` }} />
      ))}
      <div className="latent-readout">
        <span>{shape.join(" x ")}</span>
        <span>{artifact.latent?.latent_rate.toFixed(2) ?? "0"} Hz latent</span>
      </div>
    </div>
  );
}

function LineageThread({
  artifact,
  sources,
  jobs,
  families,
  compare,
  onSelectArtifact,
}: {
  artifact: ArtifactRecord;
  sources: ArtifactRecord[];
  jobs: JobRecord[];
  families: ResultFamily[];
  compare: CompareSlots;
  onSelectArtifact: (artifactId: string | null) => void;
}) {
  const nodes = artifactLineageModel({ artifact, sources, jobs, families, compare });
  return (
    <div className="lineage-thread" aria-label="Artifact lineage">
      {nodes.map((node, index) => (
        <Fragment key={node.id}>
          {index ? <span className="thread-route" /> : null}
          {node.artifactId ? (
            <button type="button" className={`thread-node ${node.kind} ${node.kind === "current" ? artifact.kind : ""}`} title={node.title} onClick={() => onSelectArtifact(node.artifactId ?? null)}>
              {node.label}
            </button>
          ) : (
            <span className={`thread-node ${node.kind}`} title={node.title}>
              {node.label}
            </span>
          )}
        </Fragment>
      ))}
    </div>
  );
}

function waveBusStateClass({
  artifact,
  sources,
  jobs,
  families,
  compare,
}: {
  artifact: ArtifactRecord;
  sources: ArtifactRecord[];
  jobs: JobRecord[];
  families: ResultFamily[];
  compare: CompareSlots;
}) {
  const hasJob = jobs.some((job) => job.recipe.recipe_id === artifact.recipe_id || job.artifact_ids.includes(artifact.artifact_id));
  const hasFamily = families.some((family) => family.artifactIds.includes(artifact.artifact_id) || family.recipeId === artifact.recipe_id);
  const hasCompare = compare.a === artifact.artifact_id || compare.b === artifact.artifact_id;
  return [
    sources.length ? "has-sources" : "root-source",
    hasJob ? "has-job" : "no-job",
    hasFamily ? "has-family" : "no-family",
    hasCompare ? "has-compare" : "no-compare",
    sources.length || hasJob || hasFamily || hasCompare ? "has-flow" : "quiet-flow",
  ].join(" ");
}

function OperatorPresetRack({
  presets,
  selectedPreset,
  selectedPresetId,
  presetName,
  diffRows,
  onSelectPreset,
  onChangePresetName,
  onSavePreset,
  onResetPreset,
  onDeletePreset,
}: {
  presets: OperatorPreset[];
  selectedPreset: OperatorPreset | null;
  selectedPresetId: string;
  presetName: string;
  diffRows: OperatorPresetDiffRow[];
  onSelectPreset: (presetId: string) => void;
  onChangePresetName: (name: string) => void;
  onSavePreset: () => void;
  onResetPreset: () => void;
  onDeletePreset: () => void;
}) {
  const diffStatus = !selectedPreset ? "empty" : diffRows.length ? "changed" : "clean";
  return (
    <div className="operator-preset-stack">
      <div className="operator-preset-rack" aria-label="Operator presets">
        <label>
          Preset
          <select value={selectedPresetId} onChange={(event) => onSelectPreset(event.target.value)}>
            <option value="">New preset</option>
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Name
          <input
            value={presetName}
            onChange={(event) => onChangePresetName(event.target.value)}
            placeholder={presets.length ? "save current params" : "name this setting"}
          />
        </label>
        <button type="button" onClick={onSavePreset} title="Save current operator parameters">
          <Plus aria-hidden="true" size={13} />
          Save
        </button>
        <button type="button" onClick={onResetPreset} disabled={!selectedPreset || !diffRows.length} title="Revert current parameters to the selected preset">
          <Repeat aria-hidden="true" size={13} />
          Revert
        </button>
        <button type="button" onClick={onDeletePreset} disabled={!selectedPresetId} title="Delete selected operator preset">
          <X aria-hidden="true" size={13} />
          Delete
        </button>
      </div>
      <div className={`operator-preset-diff ${diffStatus}`} aria-label="Operator preset diff">
        <div className="operator-preset-diff-head">
          <strong>{!selectedPreset ? "Preset diff" : diffRows.length ? `${diffRows.length} unsaved change${diffRows.length === 1 ? "" : "s"}` : "Matches preset"}</strong>
          <small>{selectedPreset ? selectedPreset.name : "save or select a preset to compare params"}</small>
        </div>
        {diffRows.length ? (
          <div className="operator-preset-diff-list">
            {diffRows.slice(0, 4).map((row) => (
              <span key={row.key} className={row.status} title={`${formatPresetValue(row.presetValue)} -> ${formatPresetValue(row.currentValue)}`}>
                <b>{row.label}</b>
                <i>{formatPresetValue(row.presetValue)}</i>
                <em>{formatPresetValue(row.currentValue)}</em>
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function formatPresetValue(value: RecipeValue | null): string {
  if (value === null || value === "") return "none";
  if (typeof value === "number") return Number.isInteger(value) ? value.toString() : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  if (typeof value === "boolean") return value ? "on" : "off";
  if (value.length > 18) return `${value.slice(0, 15)}...`;
  return value;
}

function SpecCoverage({ spec, controlledKeys }: { spec: OperatorSpec | undefined; controlledKeys: readonly string[] }) {
  const coverage = specCoverageSummary(spec, controlledKeys);
  return (
    <div className={`spec-coverage ${coverage.status}`}>
      <span>{!spec ? "Spec pending" : coverage.missing.length ? `${coverage.missing.length} missing params` : "Spec covered"}</span>
      <small>
        {spec ? `${coverage.paramCount} params · ${spec.backends.join(", ")} · ${spec.status}` : "waiting for /operators/specs"}
      </small>
      {coverage.missing.length ? <em title={coverage.missing.join(", ")}>{coverage.missing.slice(0, 4).join(", ")}</em> : null}
    </div>
  );
}

function SpecCoveragePair({
  specs,
  controlledKeys,
}: {
  specs: readonly (OperatorSpec | undefined)[];
  controlledKeys: readonly (readonly string[])[];
}) {
  const coverage = specPairCoverageSummary(specs, controlledKeys);
  const readySpecs = specs.filter(Boolean) as OperatorSpec[];
  return (
    <div className={`spec-coverage ${coverage.status}`}>
      <span>{readySpecs.length !== specs.length ? "Spec pending" : coverage.missing.length ? `${coverage.missing.length} missing params` : "Spec covered"}</span>
      <small>{readySpecs.length ? `${coverage.paramCount} params · encode/decode` : "waiting for /operators/specs"}</small>
      {coverage.missing.length ? <em title={coverage.missing.join(", ")}>{coverage.missing.slice(0, 4).join(", ")}</em> : null}
    </div>
  );
}

function defaultExperimentForm(mode: ExperimentMode): Record<string, RecipeValue> {
  const config = experimentCatalog.find((item) => item.value === mode) ?? experimentCatalog[0];
  const form = defaultFieldForm(config);
  if (!form.backend) form.backend = config.backend;
  if (config.modelDefault && !form.model) form.model = config.modelDefault;
  return form;
}

function defaultOperatorForm(mode: LatentOperatorMode): Record<string, RecipeValue> {
  const config = operatorCatalog.find((item) => item.value === mode) ?? operatorCatalog[0];
  const form = defaultFieldForm(config);
  if (!form.backend) form.backend = config.defaultBackend;
  return form;
}

function defaultGenerationForm(mode: GenerationMode): Record<string, RecipeValue> {
  const config = generationCatalog.find((item) => item.value === mode) ?? generationCatalog[0];
  return defaultFieldForm(config);
}

function isLatentOperatorMode(value: string): value is LatentOperatorMode {
  return operatorCatalog.some((item) => item.value === value);
}

function isExperimentMode(value: string): value is ExperimentMode {
  return experimentCatalog.some((item) => item.value === value);
}

function generationControlKeys(mode: GenerationMode): readonly string[] {
  if (mode === "generate.inpaint") return inpaintControlKeys;
  if (mode === "generate.audio_to_audio") return audioToAudioControlKeys;
  return generateControlKeys;
}

function filteredOperatorSpec(spec: OperatorSpec | undefined, fieldKeysToKeep: readonly string[]): OperatorSpec | undefined {
  if (!spec) return undefined;
  const allowed = new Set(fieldKeysToKeep);
  return {
    ...spec,
    ui_fields: spec.ui_fields.filter((field) => allowed.has(field.key)),
  };
}

function generationSeedFallback(form: Record<string, RecipeValue>) {
  const seed = typeof form.seed === "number" ? form.seed : Number(form.seed);
  return Number.isFinite(seed) ? seed : 7;
}

function findActiveSession(sessions: SessionRecord[], sessionId: string | null): SessionRecord | null {
  if (sessionId) {
    return sessions.find((session) => session.session_id === sessionId) ?? null;
  }
  return sessions.find((session) => session.status === "active") ?? null;
}

function generationFields(): RecipeField[] {
  return [
    { key: "duration_seconds", label: "Seconds", type: "number", defaultValue: 8, min: 0.5, step: 0.5 },
    { key: "steps", label: "Steps", type: "number", defaultValue: 8, min: 1, step: 1 },
    { key: "cfg_scale", label: "CFG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
    { key: "seed", label: "Seed", type: "number", defaultValue: 42, step: 1, advanced: true },
  ];
}

function torchAdvancedFields(): RecipeField[] {
  return [
    { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: backendOptions, advanced: true },
    { key: "device", label: "Device", type: "text", advanced: true },
    { key: "no_half", label: "No half", type: "checkbox", defaultValue: false, advanced: true },
  ];
}

function artifactPathForField(artifact: ArtifactRecord, fieldKey: string) {
  const rawScriptOutput = artifact.metadata?.script_output_path;
  const scriptOutput = typeof rawScriptOutput === "string" ? rawScriptOutput : "";
  const source = scriptOutput || artifact.path;
  if (!scriptOutput) return source;
  if (fieldKey === "profile_path") return `${source}/profile.npz`;
  if (fieldKey === "direction_path") {
    return artifact.metadata.operator === "experiment.audio_style_vectors" ? `${source}/frame_direction.npz` : `${source}/direction.npz`;
  }
  if (fieldKey === "target_memory_path" || fieldKey === "reference_memory_path") return `${source}/memory`;
  return source;
}

function parseTags(value: string) {
  const seen = new Set<string>();
  return value
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .filter((item) => {
      const key = item.toLocaleLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function formatSessionStamp(value: string) {
  const started = Date.parse(value);
  if (!Number.isFinite(started) || started <= 0) return "All work";
  const date = new Date(started);
  return `Since ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}

function createdAfter(createdAt: string, startedAt: string) {
  const created = Date.parse(createdAt);
  const started = Date.parse(startedAt);
  if (!Number.isFinite(created) || !Number.isFinite(started)) return true;
  return created >= started;
}

function readinessChecksFromHealth(health: HealthResponse | undefined): ReadinessCheck[] {
  if (!health) return [];
  return [
    { name: "artifact-root", status: "ok", message: health.artifact_root },
    ...health.backends.map((backend) => ({
      name: `backend:${backend.backend}`,
      status: backend.available ? "ok" : "warn",
      message: backend.message ?? backend.device ?? backend.backend,
    })),
  ];
}

function priorityReadinessChecks(checks: ReadinessCheck[]) {
  const priority = ["artifact-root", "hf-auth", "mlx-medium-weights", "same-l-access", "backend:mlx", "backend:torch_mps"];
  const byName = new Map(checks.map((check) => [check.name, check]));
  const selected = priority.map((name) => byName.get(name)).filter((check): check is ReadinessCheck => Boolean(check));
  const urgent = checks.filter((check) => (check.status === "error" || check.status === "warn") && !priority.includes(check.name));
  return [...selected, ...urgent].slice(0, 7);
}

function readinessLabel(name: string) {
  return name
    .replace("backend:", "")
    .replace("hf-auth", "HF auth")
    .replace("mlx-medium-weights", "MLX medium")
    .replace("same-l-access", "SAME-L")
    .replace("artifact-root", "Artifacts");
}

function mergeJobRecords(baseJobs: JobRecord[], overlayJobs: JobRecord[]) {
  const byId = new Map(baseJobs.map((job) => [job.job_id, job]));
  for (const job of overlayJobs) {
    byId.set(job.job_id, job);
  }
  return sortNewestJobs([...byId.values()]);
}

function parseJobEvent(raw: string): JobRecord | null {
  try {
    const payload = JSON.parse(raw) as unknown;
    return jobFromJobEvent(payload);
  } catch {
    return null;
  }
}

function jobFromJobEvent(payload: unknown): JobRecord | null {
  if (!payload || typeof payload !== "object") return null;
  const event = "data" in payload && payload.data && typeof payload.data === "object" ? payload.data : payload;
  if ("type" in event) {
    return event.type === "snapshot" && "job" in event ? (event.job as JobRecord) : null;
  }
  return "job_id" in event ? (event as JobRecord) : null;
}

function buildResultFamilies(artifacts: ArtifactRecord[], jobs: JobRecord[]): ResultFamily[] {
  const artifactsByRecipe = groupArtifactsByRecipe(artifacts);
  const families = jobs.reduce<Map<string, { familyId: string; recipe: JobRecord["recipe"]; jobs: JobRecord[]; artifacts: ArtifactRecord[] }>>(
    (items, job) => {
      const recipeId = job.recipe.recipe_id;
      const familyId = resultFamilyIdForRecipe(job.recipe);
      const recipeArtifacts = artifactsByRecipe.get(recipeId) ?? [];
      const family = items.get(familyId);
      if (family) {
        family.jobs.push(job);
        family.artifacts = uniqueArtifacts([...family.artifacts, ...recipeArtifacts]);
      } else {
        items.set(familyId, {
          familyId,
          recipe: job.recipe,
          jobs: [job],
          artifacts: recipeArtifacts,
        });
      }
      return items;
    },
    new Map(),
  );
  return [...families.values()]
    .map((family) => {
      const sortedArtifacts = sortNewest(family.artifacts);
      const artifactIds = unique([
        ...family.jobs.flatMap((job) => job.artifact_ids),
        ...family.artifacts.map((artifact) => artifact.artifact_id),
      ]);
      const timestamps = [
        ...family.jobs.map((job) => job.finished_at ?? job.started_at ?? job.created_at),
        ...family.artifacts.map((artifact) => artifact.created_at),
      ];
      const sortedJobs = sortNewestJobs(family.jobs);
      const recipe = sortedJobs[0]?.recipe ?? family.recipe;
      return {
        familyId: family.familyId,
        recipeId: recipe.recipe_id,
        recipe,
        operator: recipe.operator,
        sessionId: recipe.session_id ?? null,
        status: familyStatus(family.jobs),
        jobIds: family.jobs.map((job) => job.job_id),
        artifactIds,
        artifactKinds: unique(family.artifacts.map((artifact) => artifact.kind)),
        metrics: sortedJobs[0]?.metrics ?? {},
        latestArtifactId: sortedArtifacts[0]?.artifact_id ?? artifactIds[0] ?? null,
        createdAt: oldestTimestamp(timestamps) ?? recipe.created_at,
        updatedAt: newestTimestamp(timestamps) ?? recipe.created_at,
      };
    })
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
}

function resultFamilyIdForRecipe(recipe: Recipe) {
  const sourceId = typeof recipe.inputs.source === "string" ? recipe.inputs.source : "";
  const metadata = recordValue(recipe.params.metadata);
  if (recipe.operator === "generate.text_to_audio" && sourceId && metadata?.generation_origin === "prompt_search_candidate") {
    return `prompt-candidates:${sourceId}`;
  }
  return recipe.recipe_id;
}

function filterFamiliesForWork(families: ResultFamily[], artifacts: ArtifactRecord[], jobs: JobRecord[]) {
  const recipeIds = new Set([
    ...jobs.map((job) => job.recipe.recipe_id),
    ...artifacts.map((artifact) => artifact.recipe_id).filter((recipeId): recipeId is string => Boolean(recipeId)),
  ]);
  return families.filter((family) => recipeIds.has(family.recipeId));
}

function groupArtifactsByRecipe(artifacts: ArtifactRecord[]) {
  const groups = new Map<string, ArtifactRecord[]>();
  for (const artifact of artifacts) {
    if (!artifact.recipe_id) continue;
    const group = groups.get(artifact.recipe_id) ?? [];
    group.push(artifact);
    groups.set(artifact.recipe_id, group);
  }
  return groups;
}

function familyStatus(jobs: JobRecord[]): ResultFamily["status"] {
  if (jobs.some((job) => job.status === "running")) return "running";
  if (jobs.some((job) => job.status === "queued")) return "queued";
  if (jobs.some((job) => job.status === "failed")) return "failed";
  if (jobs.length && jobs.every((job) => job.status === "cancelled")) return "cancelled";
  if (jobs.length && jobs.every((job) => job.status === "succeeded")) return "succeeded";
  return "mixed";
}

function unique<T>(items: T[]) {
  return [...new Set(items)];
}

function uniqueArtifacts(artifacts: ArtifactRecord[]) {
  const seen = new Set<string>();
  return artifacts.filter((artifact) => {
    if (seen.has(artifact.artifact_id)) return false;
    seen.add(artifact.artifact_id);
    return true;
  });
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function primitiveMetadataValue(value: unknown) {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean" ? value : null;
}

function newestTimestamp(timestamps: string[]) {
  return timestamps.reduce<string | null>((latest, item) => (!latest || Date.parse(item) > Date.parse(latest) ? item : latest), null);
}

function oldestTimestamp(timestamps: string[]) {
  return timestamps.reduce<string | null>((oldest, item) => (!oldest || Date.parse(item) < Date.parse(oldest) ? item : oldest), null);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function activeJobForOperator(jobs: JobRecord[], operator: OperatorName) {
  return jobs.find((job) => job.recipe.operator === operator) ?? null;
}

function statusClass(status: string) {
  if (status.includes("native")) return "native";
  if (status.includes("partial") || status.includes("chainable")) return "partial";
  return "scaffold";
}
