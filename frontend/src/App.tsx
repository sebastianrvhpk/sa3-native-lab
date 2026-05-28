import { useEffect, useMemo, useState } from "react";
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
  SlidersHorizontal,
  Upload,
  Wand2,
  Waves,
  X,
} from "lucide-react";

import modelImage from "../../stable-audio-3.png";
import { createApi, type ArtifactAnnotationPayload, type ExperimentPayload, type RecipeForkPayload } from "./api";
import { ArtifactBadge, ArtifactIcon } from "./artifactDisplay";
import { artifactMeta, artifactName, artifactShape, formatDuration, sortNewest, sortNewestJobs } from "./artifactUtils";
import { AudioDeck, TinyWave } from "./audioDeck";
import { BundleField } from "./bundleInspector";
import { createControlPlaneClient, DEFAULT_CONTROL_PLANE_URL, type ResultFamily, type WorkbenchState } from "./controlPlane";
import { ForkRecipePanel } from "./forkRecipePanel";
import { JobProgress, type JobActionHandlers } from "./jobProgress";
import { isJobActive, shortOperatorName } from "./jobUtils";
import { RecipeFields } from "./RecipeFields";
import { FamilyDetailPanel, ResultFamilyPanel } from "./resultFamilies";
import {
  buildExperimentPayload,
  buildOperatorParams,
  defaultFieldForm,
  experimentReady,
  fieldKeys,
  fillMissingFieldDefaults,
  operatorBackend,
  operatorReady,
  operatorSeed,
  operatorUsesDonor,
  withOperatorSpecFields,
  type FieldConfig,
  type RecipeField,
  type RecipeValue,
} from "./recipeFormModel";
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
  | "experiment.soft_prompt.optimize"
  | "experiment.soft_prompt.generate"
  | "dataset.pre_encode"
  | "memory.query"
  | "training.lora";

type GenerationMode = "generate.text_to_audio" | "generate.audio_to_audio" | "generate.inpaint";

interface ExperimentConfig {
  value: ExperimentMode;
  label: string;
  family: "Style" | "Residual" | "Soft Prompt" | "Dataset" | "Training";
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

  const [prompt, setPrompt] = useState("short soft percussive click");
  const [generationMode, setGenerationMode] = useState<GenerationMode>("generate.text_to_audio");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [duration, setDuration] = useState(5);
  const [steps, setSteps] = useState(8);
  const [seed, setSeed] = useState(7);
  const [cfgScale, setCfgScale] = useState(1);
  const [apgScale, setApgScale] = useState(1);
  const [initNoiseLevel, setInitNoiseLevel] = useState(0.7);
  const [inpaintStart, setInpaintStart] = useState(0);
  const [inpaintEnd, setInpaintEnd] = useState(2);
  const [audioModel, setAudioModel] = useState<(typeof audioModels)[number]["value"]>("medium");
  const [audioDecoder, setAudioDecoder] = useState<"same-s" | "same-l">("same-l");
  const [sameModel, setSameModel] = useState<"same-s" | "same-l">("same-l");
  const [sameChunked, setSameChunked] = useState(false);
  const [sameChunkSize, setSameChunkSize] = useState(128);
  const [sameOverlap, setSameOverlap] = useState(32);
  const [samePrompt, setSamePrompt] = useState("");
  const [sameNotes, setSameNotes] = useState("");
  const [operator, setOperator] = useState<LatentOperatorMode>("latent.cyclic_roll");
  const [operatorForm, setOperatorForm] = useState<Record<string, RecipeValue>>(() => defaultOperatorForm("latent.cyclic_roll"));
  const [donorArtifactId, setDonorArtifactId] = useState("");
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
  const activeOperatorConfig = useMemo(() => withOperatorSpecFields(baseOperatorConfig, activeOperatorSpec), [baseOperatorConfig, activeOperatorSpec]);
  const activeExperiment = useMemo(() => withOperatorSpecFields(baseExperiment, activeExperimentSpec), [baseExperiment, activeExperimentSpec]);
  const generationNeedsSource = generationMode !== "generate.text_to_audio";
  const generationSource = selectedArtifact?.kind === "audio" ? selectedArtifact : null;
  const canGenerate = !generationNeedsSource || Boolean(generationSource);
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
          void invalidate();
        }
      };
      return socket;
    });
    return () => {
      sockets.forEach((socket) => socket.close());
    };
  }, [api, controlPlane, runningJobIds]);

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
    onSuccess: async (artifact) => {
      selectArtifact(artifact.artifact_id);
      await invalidate();
    },
  });

  const generate = useMutation({
    mutationFn: () => {
      const selected = audioModels.find((item) => item.value === audioModel)!;
      const basePayload = {
        prompt,
        negative_prompt: negativePrompt.trim() || null,
        duration_seconds: duration,
        steps,
        seed,
        cfg_scale: cfgScale,
        apg_scale: apgScale,
        model: selected.value,
        decoder: audioDecoder,
        backend: "mlx",
        session_id: activeSessionId,
      } as const;
      if (generationMode === "generate.audio_to_audio") {
        if (!generationSource) throw new Error("Audio-to-audio requires a selected audio artifact.");
        return api.generateAudioToAudio({
          ...basePayload,
          source_artifact_id: generationSource.artifact_id,
          init_noise_level: initNoiseLevel,
        });
      }
      if (generationMode === "generate.inpaint") {
        if (!generationSource) throw new Error("Inpaint requires a selected audio artifact.");
        return api.generateInpaint({
          ...basePayload,
          source_artifact_id: generationSource.artifact_id,
          init_noise_level: initNoiseLevel,
          inpaint_start_seconds: inpaintStart,
          inpaint_end_seconds: inpaintEnd,
        });
      }
      return api.generateText(basePayload);
    },
    onSuccess: invalidate,
  });

  const encode = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.encodeLatent({
        source_artifact_id: artifact.artifact_id,
        model: sameModel,
        backend: "torch_mps",
        chunked: sameChunked,
        chunk_size: sameChunkSize,
        overlap: sameOverlap,
        prompt: samePrompt.trim() || null,
        notes: sameNotes.trim() || null,
        session_id: activeSessionId,
      }),
    onSuccess: invalidate,
  });

  const decode = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.decodeLatent({
        source_artifact_id: artifact.artifact_id,
        model: sameModel,
        backend: "torch_mps",
        chunked: sameChunked,
        chunk_size: sameChunkSize,
        overlap: sameOverlap,
        notes: sameNotes.trim() || null,
        session_id: activeSessionId,
      }),
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
        seed: operatorSeed(operatorForm, seed),
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

  const setOperatorField = (key: string, value: RecipeValue) => {
    setOperatorForm((current) => ({ ...current, [key]: value }));
  };

  const selectOperatorMode = (mode: LatentOperatorMode) => {
    setOperator(mode);
    setOperatorForm(defaultOperatorForm(mode));
    setDonorArtifactId("");
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

        <section className="operator-surface">
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
            apiBase={apiBase}
            annotating={annotateArtifact.isPending}
            onAnnotate={(artifactId, payload) => annotateArtifact.mutate({ artifactId, payload })}
            onCompare={setCompare}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onSelectArtifact={selectArtifact}
            onUseAsDonor={useArtifactAsDonor}
            onUseInRecipe={useBundleInRecipe}
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
              <label className="control-cell">
                Prompt
                <input value={prompt} onChange={(event) => setPrompt(event.target.value)} />
              </label>
              <label className="control-cell">
                Negative prompt
                <input value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} placeholder="optional" />
              </label>
              <div className="generation-grid">
                <label className="control-cell">
                  Model
                  <select
                    value={audioModel}
                    onChange={(event) => {
                      const value = event.target.value as typeof audioModel;
                      setAudioModel(value);
                      setAudioDecoder(audioModels.find((item) => item.value === value)?.decoder ?? "same-l");
                    }}
                  >
                    {audioModels.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="control-cell">
                  Decoder
                  <select value={audioDecoder} onChange={(event) => setAudioDecoder(event.target.value as typeof audioDecoder)}>
                    {sameModelOptions.map((item) => (
                      <option key={item.value} value={item.value}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="control-cell">
                  Seconds
                  <input type="number" min={0.5} max={120} step={0.5} value={duration} onChange={(event) => setDuration(Number(event.target.value))} />
                </label>
                <label className="control-cell">
                  Steps
                  <input type="number" min={1} max={64} value={steps} onChange={(event) => setSteps(Number(event.target.value))} />
                </label>
                <label className="control-cell">
                  Seed
                  <input type="number" step={1} value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
                </label>
                <label className="control-cell">
                  CFG
                  <input type="number" min={0} step={0.1} value={cfgScale} onChange={(event) => setCfgScale(Number(event.target.value))} />
                </label>
                <label className="control-cell">
                  APG
                  <input type="number" min={0} step={0.1} value={apgScale} onChange={(event) => setApgScale(Number(event.target.value))} />
                </label>
                {generationNeedsSource ? (
                  <label className="control-cell">
                    Init noise
                    <input type="number" min={0} max={1} step={0.05} value={initNoiseLevel} onChange={(event) => setInitNoiseLevel(Number(event.target.value))} />
                  </label>
                ) : null}
                {generationMode === "generate.inpaint" ? (
                  <>
                    <label className="control-cell">
                      Inpaint start
                      <input type="number" min={0} step={0.1} value={inpaintStart} onChange={(event) => setInpaintStart(Number(event.target.value))} />
                    </label>
                    <label className="control-cell">
                      Inpaint end
                      <input type="number" min={0.1} step={0.1} value={inpaintEnd} onChange={(event) => setInpaintEnd(Number(event.target.value))} />
                    </label>
                  </>
                ) : null}
              </div>
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
              <div className="segmented">
                <button className={sameModel === "same-s" ? "active" : ""} onClick={() => setSameModel("same-s")}>
                  same-s
                </button>
                <button className={sameModel === "same-l" ? "active" : ""} onClick={() => setSameModel("same-l")}>
                  same-l
                </button>
              </div>
              <details className="recipe-advanced same-advanced" open>
                <summary>
                  <SlidersHorizontal size={15} />
                  Parameters
                </summary>
                <div className="recipe-fields advanced same-fields">
                  <label className="field-checkbox">
                    <input type="checkbox" checked={sameChunked} onChange={(event) => setSameChunked(event.target.checked)} />
                    <span>Chunked encode/decode</span>
                  </label>
                  <label className="control-cell">
                    Chunk size
                    <input type="number" min={1} step={1} value={sameChunkSize} onChange={(event) => setSameChunkSize(Number(event.target.value))} />
                  </label>
                  <label className="control-cell">
                    Overlap
                    <input type="number" min={0} step={1} value={sameOverlap} onChange={(event) => setSameOverlap(Number(event.target.value))} />
                  </label>
                  <label className="control-cell">
                    Encode prompt
                    <input value={samePrompt} onChange={(event) => setSamePrompt(event.target.value)} placeholder="optional" />
                  </label>
                  <label className="control-cell">
                    Notes
                    <input value={sameNotes} onChange={(event) => setSameNotes(event.target.value)} placeholder="optional" />
                  </label>
                </div>
              </details>
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
            artifacts={allArtifacts}
            jobs={allJobs}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            onSelect={selectArtifact}
            onCompare={setCompare}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onForkRecipe={setForkTarget}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
          <SessionTray
            artifacts={sessionArtifacts}
            archivedArtifacts={archiveArtifacts}
            jobs={sessionJobs}
            archivedJobs={archiveJobs}
            runningJobs={runningJobs}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            session={activeSession}
            sessionStartedAt={activeSession?.created_at ?? sessionStartedAt}
            creatingSession={createSession.isPending}
            archivingSession={archiveSession.isPending}
            onSelect={selectArtifact}
            onStartSession={() => createSession.mutate()}
            onArchiveSession={() => activeSession && archiveSession.mutate(activeSession)}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
          <ComparePanel a={compareA} b={compareB} apiBase={apiBase} />
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

function Specimen({
  artifact,
  artifacts,
  apiBase,
  annotating,
  onAnnotate,
  onCompare,
  onReplayRecipe,
  onSelectArtifact,
  onUseAsDonor,
  onUseInRecipe,
  getArtifactPath,
}: {
  artifact: ArtifactRecord | null;
  artifacts: ArtifactRecord[];
  apiBase: string;
  annotating: boolean;
  onAnnotate: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onReplayRecipe: (recipeId: string) => void;
  onSelectArtifact: (artifactId: string | null) => void;
  onUseAsDonor: (artifactId: string) => void;
  onUseInRecipe: (fieldKey: string, path: string, mode: string) => void;
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
  return (
    <div className="specimen">
      <div className="wave-bus">
        {artifact.kind === "audio" ? (
          <AudioDeck artifact={artifact} apiBase={apiBase} />
        ) : artifact.kind === "latent" ? (
          <LatentField artifact={artifact} />
        ) : (
          <BundleField
            artifact={artifact}
            artifacts={artifacts}
            apiBase={apiBase}
            onCompare={onCompare}
            onSelectArtifact={onSelectArtifact}
            onUseAsDonor={onUseAsDonor}
            onUseInRecipe={onUseInRecipe}
            getArtifactPath={getArtifactPath}
          />
        )}
        <LineageThread artifact={artifact} sources={sourceArtifacts} />
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
            disabled={!artifact.recipe_id}
            onClick={() => {
              if (artifact.recipe_id) onReplayRecipe(artifact.recipe_id);
            }}
            title="Replay recipe"
          >
            <Repeat size={17} />
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
  runningJobs,
  selectedId,
  apiBase,
  session,
  sessionStartedAt,
  creatingSession,
  archivingSession,
  onSelect,
  onStartSession,
  onArchiveSession,
  onCancelJob,
  onRetryJob,
}: {
  artifacts: ArtifactRecord[];
  archivedArtifacts: ArtifactRecord[];
  jobs: JobRecord[];
  archivedJobs: JobRecord[];
  runningJobs: JobRecord[];
  selectedId: string | null;
  apiBase: string;
  session: SessionRecord | null;
  sessionStartedAt: string;
  creatingSession: boolean;
  archivingSession: boolean;
  onSelect: (artifactId: string | null) => void;
  onStartSession: () => void;
  onArchiveSession: () => void;
} & JobActionHandlers) {
  const [archiveQuery, setArchiveQuery] = useState("");
  const [archiveTag, setArchiveTag] = useState("");
  const sessionArtifacts = sortNewest(artifacts).slice(0, 8);
  const sessionJobs = sortNewestJobs(jobs).slice(0, 4);
  const activeArchiveTags = archiveTag ? [archiveTag] : [];
  const archiveSearching = Boolean(archiveQuery.trim() || archiveTag);
  const archiveTags = archiveTagOptions(archivedArtifacts);
  const filteredArchiveArtifacts = archivedArtifacts.filter((artifact) => artifactMatchesSearch(artifact, archiveQuery, activeArchiveTags));
  const archiveArtifactRows = sortNewest(filteredArchiveArtifacts).slice(0, 10);
  const archiveJobRows = archiveSearching ? [] : sortNewestJobs(archivedJobs).slice(0, 10);
  const activeJobs = runningJobs.slice(0, 3);

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

      {activeJobs.length ? (
        <div className="session-block">
          <span className="session-label">Running</span>
          {activeJobs.map((job) => (
            <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
          ))}
        </div>
      ) : null}

      <div className="session-block">
        <span className="session-label">Takes</span>
        {sessionArtifacts.length ? (
          <div className="session-artifacts">
            {sessionArtifacts.map((artifact) => (
              <SessionArtifactRow
                key={artifact.artifact_id}
                artifact={artifact}
                selected={artifact.artifact_id === selectedId}
                apiBase={apiBase}
                onSelect={onSelect}
              />
            ))}
          </div>
        ) : (
          <div className="quiet-panel compact">Fresh session</div>
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
          <span>{archiveSearching ? `${filteredArchiveArtifacts.length}/${archivedArtifacts.length}` : archivedArtifacts.length + archivedJobs.length}</span>
        </summary>
        <div className="archive-search">
          <label>
            <Search size={14} />
            <input
              type="search"
              aria-label="Search archive"
              value={archiveQuery}
              onChange={(event) => setArchiveQuery(event.target.value)}
              placeholder="label, notes, tags"
            />
          </label>
          {archiveTags.length ? (
            <div className="archive-tags" aria-label="Archive tags">
              {archiveTags.slice(0, 10).map((tag) => (
                <button
                  key={tag}
                  type="button"
                  aria-pressed={archiveTag === tag}
                  className={archiveTag === tag ? "selected" : ""}
                  onClick={() => setArchiveTag(archiveTag === tag ? "" : tag)}
                >
                  #{tag}
                </button>
              ))}
            </div>
          ) : null}
          {archiveSearching ? (
            <button type="button" className="archive-clear" aria-label="Clear archive search" onClick={() => { setArchiveQuery(""); setArchiveTag(""); }}>
              <X size={14} />
            </button>
          ) : null}
        </div>
        <div className="archive-list">
          {archiveArtifactRows.map((artifact) => (
            <SessionArtifactRow
              key={artifact.artifact_id}
              artifact={artifact}
              selected={artifact.artifact_id === selectedId}
              apiBase={apiBase}
              onSelect={onSelect}
            />
          ))}
          {archiveJobRows.map((job) => (
            <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
          ))}
          {!archiveArtifactRows.length && !archiveJobRows.length ? <div className="quiet-panel compact">{archiveSearching ? "No matching takes" : "Archive empty"}</div> : null}
        </div>
      </details>
    </div>
  );
}

function SessionArtifactRow({
  artifact,
  selected,
  apiBase,
  onSelect,
}: {
  artifact: ArtifactRecord;
  selected: boolean;
  apiBase: string;
  onSelect: (artifactId: string | null) => void;
}) {
  return (
    <button type="button" className={`session-artifact ${selected ? "selected" : ""}`} onClick={() => onSelect(artifact.artifact_id)}>
      <ArtifactIcon artifact={artifact} />
      <div>
        <strong>{artifactName(artifact)}</strong>
        <span>{artifactMeta(artifact)}</span>
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

function LineageThread({ artifact, sources }: { artifact: ArtifactRecord; sources: ArtifactRecord[] }) {
  const sourceLabels = sources.length ? sources.slice(0, 3).map(artifactName) : ["origin"];
  return (
    <div className="lineage-thread" aria-label="Artifact lineage">
      <div className="thread-sources">
        {sourceLabels.map((label, index) => (
          <span key={`${label}-${index}`} className="thread-node source" title={label}>
            {label}
          </span>
        ))}
      </div>
      <span className="thread-route" />
      <span className={`thread-node current ${artifact.kind}`}>{artifact.kind}</span>
    </div>
  );
}

function SpecCoverage({ spec, controlledKeys }: { spec: OperatorSpec | undefined; controlledKeys: readonly string[] }) {
  const params = specParamKeys(spec);
  const missing = missingParamKeys(spec, controlledKeys);
  const status = !spec ? "waiting" : missing.length ? "partial" : "covered";
  return (
    <div className={`spec-coverage ${status}`}>
      <span>{!spec ? "Spec pending" : missing.length ? `${missing.length} missing params` : "Spec covered"}</span>
      <small>
        {spec ? `${params.length} params · ${spec.backends.join(", ")} · ${spec.status}` : "waiting for /operators/specs"}
      </small>
      {missing.length ? <em title={missing.join(", ")}>{missing.slice(0, 4).join(", ")}</em> : null}
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
  const mergedMissing = specs.flatMap((spec, index) => missingParamKeys(spec, controlledKeys[index] ?? []));
  const readySpecs = specs.filter(Boolean) as OperatorSpec[];
  const status = readySpecs.length !== specs.length ? "waiting" : mergedMissing.length ? "partial" : "covered";
  const paramCount = readySpecs.reduce((count, spec) => count + specParamKeys(spec).length, 0);
  return (
    <div className={`spec-coverage ${status}`}>
      <span>{readySpecs.length !== specs.length ? "Spec pending" : mergedMissing.length ? `${mergedMissing.length} missing params` : "Spec covered"}</span>
      <small>{readySpecs.length ? `${paramCount} params · encode/decode` : "waiting for /operators/specs"}</small>
      {mergedMissing.length ? <em title={mergedMissing.join(", ")}>{mergedMissing.slice(0, 4).join(", ")}</em> : null}
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

function isExperimentMode(value: string): value is ExperimentMode {
  return experimentCatalog.some((item) => item.value === value);
}

function generationControlKeys(mode: GenerationMode): readonly string[] {
  if (mode === "generate.inpaint") return inpaintControlKeys;
  if (mode === "generate.audio_to_audio") return audioToAudioControlKeys;
  return generateControlKeys;
}

function specParamKeys(spec: OperatorSpec | undefined): string[] {
  return Object.keys(spec?.params ?? {});
}

function missingParamKeys(spec: OperatorSpec | undefined, controlledKeys: readonly string[]): string[] {
  const controlled = new Set(controlledKeys);
  return specParamKeys(spec).filter((key) => !controlled.has(key));
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

function archiveTagOptions(artifacts: ArtifactRecord[]) {
  return Array.from(new Set(artifacts.flatMap((artifact) => artifact.tags))).sort((a, b) => a.localeCompare(b));
}

function artifactMatchesSearch(artifact: ArtifactRecord, query: string, tags: string[]) {
  const recordTags = new Set(artifact.tags.map((tag) => tag.toLowerCase()));
  if (tags.some((tag) => !recordTags.has(tag.toLowerCase()))) return false;
  const needle = query.trim().toLowerCase();
  if (!needle) return true;
  return artifactSearchText(artifact).includes(needle);
}

function artifactSearchText(artifact: ArtifactRecord) {
  return [
    artifact.artifact_id,
    artifact.kind,
    artifact.label,
    artifact.prompt,
    artifact.notes,
    artifact.file?.filename,
    ...artifact.tags,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
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
  const families = jobs.reduce<Map<string, { recipe: JobRecord["recipe"]; jobs: JobRecord[]; artifacts: ArtifactRecord[] }>>(
    (items, job) => {
      const recipeId = job.recipe.recipe_id;
      const family = items.get(recipeId);
      if (family) {
        family.jobs.push(job);
      } else {
        items.set(recipeId, {
          recipe: job.recipe,
          jobs: [job],
          artifacts: artifactsByRecipe.get(recipeId) ?? [],
        });
      }
      return items;
    },
    new Map(),
  );
  return [...families.values()]
    .map((family) => {
      const recipeId = family.recipe.recipe_id;
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
      return {
        familyId: recipeId,
        recipeId,
        recipe: family.recipe,
        operator: family.recipe.operator,
        sessionId: family.recipe.session_id ?? null,
        status: familyStatus(family.jobs),
        jobIds: family.jobs.map((job) => job.job_id),
        artifactIds,
        artifactKinds: unique(family.artifacts.map((artifact) => artifact.kind)),
        metrics: sortedJobs[0]?.metrics ?? {},
        latestArtifactId: sortedArtifacts[0]?.artifact_id ?? artifactIds[0] ?? null,
        createdAt: oldestTimestamp(timestamps) ?? family.recipe.created_at,
        updatedAt: newestTimestamp(timestamps) ?? family.recipe.created_at,
      };
    })
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
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
