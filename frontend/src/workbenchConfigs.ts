import type { ExperimentPayload } from "./api";
import type { ArtifactRecord, OperatorSpec } from "./types";
import {
  defaultFieldForm,
  type FieldConfig,
  type GenerationMode,
  type RecipeField,
  type RecipeValue,
} from "./recipeFormModel";

export const audioModels = [
  { value: "medium", label: "medium", decoder: "same-l" },
  { value: "sm-music", label: "sm-music", decoder: "same-s" },
  { value: "sm-sfx", label: "sm-sfx", decoder: "same-s" },
] as const;

export type ExperimentMode =
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

export interface ExperimentConfig {
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

export type LatentOperatorMode =
  | "latent.cyclic_roll"
  | "latent.blur"
  | "latent.dsp"
  | "latent.graft"
  | "latent.renoise";

export interface OperatorConfig extends FieldConfig {
  value: LatentOperatorMode;
  label: string;
  family: "Loop" | "Blur" | "DSP" | "Graft" | "Renoise";
  maturity: "lab" | "probe";
  defaultBackend: "torch_cpu" | "torch_mps";
  requiresDonor?: boolean;
}

export interface GenerationConfig extends FieldConfig {
  value: GenerationMode;
  label: string;
}

export const sameModelOptions = [
  { value: "same-s", label: "same-s" },
  { value: "same-l", label: "same-l" },
] as const;

export const sa3ModelOptions = [
  { value: "medium", label: "medium" },
  { value: "small-music", label: "small-music" },
  { value: "small-sfx", label: "small-sfx" },
] as const;

export const backendOptions = [
  { value: "torch_mps", label: "torch_mps" },
  { value: "torch_cpu", label: "torch_cpu" },
  { value: "cpu", label: "cpu" },
] as const;

export const promptSearchModeOptions = [
  { value: "beam", label: "beam" },
  { value: "greedy", label: "greedy" },
  { value: "coordinate", label: "coordinate" },
] as const;

export const promptSearchScorerOptions = [
  { value: "lexical_probe", label: "lexical probe" },
  { value: "sa3_flow_probe", label: "SA3 flow probe" },
  { value: "clap", label: "CLAP queued" },
] as const;

export const velocityConventionOptions = [
  { value: "noise_minus_data", label: "noise minus data" },
  { value: "data_minus_noise", label: "data minus noise" },
] as const;

export const promptSearchVocabulary = [
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

export const operatorBackendOptions = [
  { value: "torch_mps", label: "torch_mps" },
  { value: "torch_cpu", label: "torch_cpu" },
] as const;

export const torchCpuOperatorOptions = [
  { value: "torch_cpu", label: "torch_cpu" },
] as const;

export const maskModeOptions = [
  { value: "random_channels", label: "random channels" },
  { value: "high_variance", label: "high variance" },
  { value: "low_variance", label: "low variance" },
  { value: "high_activity", label: "high activity" },
  { value: "low_activity", label: "low activity" },
  { value: "channel_block", label: "channel block" },
  { value: "every_n", label: "comb / every n" },
] as const;

export const latentBlurModeOptions = [
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

export const temporalKernelOptions = [
  { value: "gaussian", label: "gaussian" },
  { value: "box", label: "box" },
] as const;

export const temporalDirectionOptions = [
  { value: "centered", label: "centered" },
  { value: "past", label: "past" },
  { value: "future", label: "future" },
] as const;

export const latentDspModeOptions = [
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

export const dspCenterOptions = [
  { value: "channel_mean", label: "channel mean" },
  { value: "global_mean", label: "global mean" },
  { value: "zero", label: "zero" },
] as const;

export const operatorModes = [
  { value: "latent.cyclic_roll", label: "cyclic roll" },
  { value: "latent.blur", label: "latent blur" },
  { value: "latent.dsp", label: "latent DSP" },
  { value: "latent.graft", label: "latent graft" },
  { value: "latent.renoise", label: "latent renoise" },
] as const;

export const operatorCatalog: readonly OperatorConfig[] = [
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

export const experimentCatalog: readonly ExperimentConfig[] = [
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

export const experimentModes = experimentCatalog.map(({ value, label }) => ({ value, label }));
export const generationModes = [
  { value: "generate.text_to_audio", label: "Text" },
  { value: "generate.audio_to_audio", label: "A2A" },
  { value: "generate.inpaint", label: "Inpaint" },
] as const;
export const generateControlKeys = ["prompt", "negative_prompt", "duration_seconds", "steps", "seed", "cfg_scale", "apg_scale", "model", "decoder"];
export const audioToAudioControlKeys = [...generateControlKeys, "init_noise_level"];
export const inpaintControlKeys = [...audioToAudioControlKeys, "inpaint_start_seconds", "inpaint_end_seconds"];
export const sameEncodeControlKeys = ["model", "chunked", "chunk_size", "overlap", "prompt", "notes"];
export const sameDecodeControlKeys = ["model", "chunked", "chunk_size", "overlap", "notes"];

export const generationCatalog: readonly GenerationConfig[] = [
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

export const sameConfig: FieldConfig = {
  fields: [
    { key: "model", label: "SAME model", type: "select", defaultValue: "same-l", options: sameModelOptions },
    { key: "chunked", label: "Chunked encode/decode", type: "checkbox", defaultValue: false },
    { key: "chunk_size", label: "Chunk size", type: "number", defaultValue: 128, min: 1, step: 1, advanced: true },
    { key: "overlap", label: "Overlap", type: "number", defaultValue: 32, min: 0, step: 1, advanced: true },
    { key: "prompt", label: "Encode prompt", type: "text", placeholder: "optional", advanced: true },
    { key: "notes", label: "Notes", type: "text", placeholder: "optional", advanced: true },
  ],
};

export function defaultExperimentForm(mode: ExperimentMode): Record<string, RecipeValue> {
  const config = experimentCatalog.find((item) => item.value === mode) ?? experimentCatalog[0];
  return defaultFieldForm(config);
}

export function defaultOperatorForm(mode: LatentOperatorMode): Record<string, RecipeValue> {
  const config = operatorCatalog.find((item) => item.value === mode) ?? operatorCatalog[0];
  return defaultFieldForm(config);
}

export function defaultGenerationForm(mode: GenerationMode): Record<string, RecipeValue> {
  const config = generationCatalog.find((item) => item.value === mode) ?? generationCatalog[0];
  return defaultFieldForm(config);
}

export function isLatentOperatorMode(value: string): value is LatentOperatorMode {
  return operatorCatalog.some((item) => item.value === value);
}

export function isExperimentMode(value: string): value is ExperimentMode {
  return experimentCatalog.some((item) => item.value === value);
}

export function generationControlKeys(mode: GenerationMode): readonly string[] {
  if (mode === "generate.inpaint") return inpaintControlKeys;
  if (mode === "generate.audio_to_audio") return audioToAudioControlKeys;
  return generateControlKeys;
}

export function filteredOperatorSpec(spec: OperatorSpec | undefined, fieldKeysToKeep: readonly string[]): OperatorSpec | undefined {
  if (!spec) return undefined;
  const keep = new Set(fieldKeysToKeep);
  return {
    ...spec,
    ui_fields: spec.ui_fields.filter((field) => keep.has(field.key)),
  };
}

export function generationSeedFallback(form: Record<string, RecipeValue>) {
  const seed = form.seed;
  return typeof seed === "number" ? seed : 7;
}

function generationFields(): RecipeField[] {
  return [
    { key: "duration_seconds", label: "Seconds", type: "number", defaultValue: 8, min: 0.5, step: 0.5 },
    { key: "steps", label: "Steps", type: "number", defaultValue: 8, min: 1, step: 1 },
    { key: "cfg_scale", label: "CFG", type: "number", defaultValue: 1, min: 0, step: 0.1 },
    { key: "seed", label: "Seed", type: "number", defaultValue: 42, step: 1 },
  ];
}

function torchAdvancedFields(): RecipeField[] {
  return [
    { key: "backend", label: "Backend", type: "select", defaultValue: "torch_mps", options: backendOptions, advanced: true },
    { key: "device", label: "Device", type: "text", placeholder: "auto", advanced: true },
    { key: "no_half", label: "No half", type: "checkbox", defaultValue: false, advanced: true },
  ];
}
