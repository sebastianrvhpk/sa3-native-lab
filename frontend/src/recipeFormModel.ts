import { z } from "zod";

import type {
  AudioToAudioPayload,
  ExperimentPayload,
  GenerateTextPayload,
  InpaintPayload,
  LatentDecodePayload,
  LatentEncodePayload,
} from "./api";
import type { ArtifactKind, ArtifactRecord, OperatorFieldSpec, OperatorSpec } from "./types";

export const recipeValueSchema = z.union([z.string(), z.number(), z.boolean()]);

export type RecipeValue = z.infer<typeof recipeValueSchema>;
export type RecipeFieldType = "text" | "number" | "range" | "select" | "checkbox" | "path" | "artifact-path";

export interface RecipeField {
  key: string;
  label: string;
  type: RecipeFieldType;
  defaultValue?: RecipeValue;
  required?: boolean;
  advanced?: boolean;
  min?: number;
  max?: number;
  step?: number;
  options?: readonly { value: string; label: string }[];
  artifactKinds?: readonly ArtifactRecord["kind"][];
  placeholder?: string;
  description?: string;
}

export interface FieldConfig {
  fields: readonly RecipeField[];
}

export interface ExperimentPayloadConfig<TOperator extends ExperimentPayload["operator"] = ExperimentPayload["operator"]>
  extends FieldConfig {
  value: TOperator;
  backend: ExperimentPayload["backend"];
  modelDefault?: string;
  selectedAudioFallback?: string;
  selectedLatentFallback?: boolean;
}

export interface OperatorPayloadConfig<TMode extends string = string> extends FieldConfig {
  value: TMode;
  defaultBackend: "torch_cpu" | "torch_mps";
  requiresDonor?: boolean;
}

export type GenerationMode = "generate.text_to_audio" | "generate.audio_to_audio" | "generate.inpaint";
export type GenerationPayload = GenerateTextPayload | AudioToAudioPayload | InpaintPayload;

export function defaultFieldForm(config: FieldConfig): Record<string, RecipeValue> {
  const form: Record<string, RecipeValue> = {};
  for (const field of config.fields) {
    if (field.defaultValue !== undefined) {
      form[field.key] = field.defaultValue;
    } else if (field.type === "checkbox") {
      form[field.key] = false;
    } else if (field.type === "select") {
      form[field.key] = field.options?.[0]?.value ?? "";
    } else {
      form[field.key] = "";
    }
  }
  return form;
}

export function fieldKeys(config: FieldConfig): string[] {
  return config.fields.map((field) => field.key).filter((key) => key !== "backend");
}

export function fillMissingFieldDefaults(config: FieldConfig, form: Record<string, RecipeValue>): Record<string, RecipeValue> {
  const defaults = defaultFieldForm(config);
  let next = form;
  for (const [key, value] of Object.entries(defaults)) {
    if (next[key] !== undefined) continue;
    if (next === form) next = { ...form };
    next[key] = value;
  }
  return next;
}

export function withOperatorSpecFields<TConfig extends FieldConfig>(config: TConfig, spec?: OperatorSpec): TConfig {
  if (!spec?.ui_fields.length) return config;
  const byKey = new Map(spec.ui_fields.map((field) => [field.key, field]));
  const seen = new Set<string>();
  const fields: RecipeField[] = config.fields.map((field) => {
    seen.add(field.key);
    const specField = byKey.get(field.key);
    return specField ? mergeRecipeField(field, specField) : field;
  });

  for (const specField of spec.ui_fields) {
    if (seen.has(specField.key)) continue;
    fields.push(recipeFieldFromSpec(specField));
  }

  return { ...config, fields } as TConfig;
}

export function validateRecipeField(field: RecipeField, value: RecipeValue | undefined): string | undefined {
  if (field.required && !stringValue(value)) return `${field.label} is required`;
  if (field.type === "select" && stringValue(value) && field.options?.length) {
    const allowed = new Set(field.options.map((option) => option.value));
    if (!allowed.has(stringValue(value))) return `${field.label} must be one of ${field.options.map((option) => option.label).join(", ")}`;
  }
  if ((field.type === "number" || field.type === "range") && stringValue(value)) {
    const numberValue = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(numberValue)) return `${field.label} must be a number`;
    if (field.min !== undefined && numberValue < field.min) return `${field.label} must be at least ${field.min}`;
    if (field.max !== undefined && numberValue > field.max) return `${field.label} must be at most ${field.max}`;
    if (field.step === 1 && !Number.isInteger(numberValue)) return `${field.label} must be a whole number`;
  }
  if (field.key === "alphas" && stringValue(value) && !parseNumberList(value ?? "").length) return `${field.label} must include at least one number`;
  return undefined;
}

export function experimentReady(
  config: ExperimentPayloadConfig,
  form: Record<string, RecipeValue>,
  selectedArtifact: ArtifactRecord | null,
) {
  if (config.selectedLatentFallback && selectedArtifact?.kind !== "latent") {
    return false;
  }
  if (config.selectedAudioFallback && selectedArtifact?.kind === "audio" && !stringValue(form[config.selectedAudioFallback])) {
    return true;
  }
  return config.fields.every((field) => !field.required || Boolean(stringValue(form[field.key])));
}

export function buildExperimentPayload({
  config,
  form,
  selectedArtifact,
  sessionId,
}: {
  config: ExperimentPayloadConfig;
  form: Record<string, RecipeValue>;
  selectedArtifact: ArtifactRecord | null;
  sessionId?: string | null;
}): ExperimentPayload {
  const params: Record<string, unknown> = {};
  const inputs: Record<string, string> = {};

  for (const field of config.fields) {
    if (field.key === "backend" || field.key === "model" || field.key === "seed") continue;
    const value = form[field.key];
    if (field.type === "checkbox") {
      params[field.key] = Boolean(value);
    } else if (value !== undefined && stringValue(value) !== "") {
      params[field.key] = value;
    }
  }

  if (config.selectedAudioFallback && !params[config.selectedAudioFallback] && selectedArtifact?.kind === "audio") {
    inputs.source = selectedArtifact.artifact_id;
  }
  if (config.selectedLatentFallback && selectedArtifact?.kind === "latent") {
    inputs.source = selectedArtifact.artifact_id;
  }

  const backend = (stringValue(form.backend) || config.backend) as ExperimentPayload["backend"];
  const seedValue = form.seed;
  const seed = typeof seedValue === "number" && Number.isFinite(seedValue) ? seedValue : undefined;
  const model = stringValue(form.model) || config.modelDefault || null;

  return {
    operator: config.value,
    backend,
    inputs,
    model,
    seed,
    session_id: sessionId,
    params,
  };
}

export function generationReady({
  form,
  needsSource,
  sourceArtifact,
}: {
  form: Record<string, RecipeValue>;
  needsSource: boolean;
  sourceArtifact: ArtifactRecord | null;
}) {
  if (!stringValue(form.prompt)) return false;
  if (needsSource && sourceArtifact?.kind !== "audio") return false;
  const duration = numberValue(form.duration_seconds, 0);
  const steps = numberValue(form.steps, 0);
  if (duration <= 0 || steps < 1) return false;
  const inpaintEnd = numberValue(form.inpaint_end_seconds, 0);
  const inpaintStart = numberValue(form.inpaint_start_seconds, 0);
  if (inpaintEnd && inpaintStart >= inpaintEnd) return false;
  if (inpaintEnd && inpaintEnd > duration) return false;
  return true;
}

export function buildGenerationPayload({
  mode,
  form,
  sourceArtifact,
  sessionId,
  promptOverride,
  sourceArtifactId,
  notes,
  metadata,
}: {
  mode: GenerationMode;
  form: Record<string, RecipeValue>;
  sourceArtifact?: ArtifactRecord | null;
  sessionId?: string | null;
  promptOverride?: string;
  sourceArtifactId?: string | null;
  notes?: string | null;
  metadata?: Record<string, unknown> | null;
}): GenerationPayload {
  const base: GenerateTextPayload = {
    prompt: promptOverride ?? stringValue(form.prompt),
    negative_prompt: optionalString(form.negative_prompt),
    duration_seconds: numberValue(form.duration_seconds, 30),
    steps: Math.max(1, Math.trunc(numberValue(form.steps, 8))),
    seed: optionalInteger(form.seed),
    cfg_scale: numberValue(form.cfg_scale, 1),
    apg_scale: numberValue(form.apg_scale, 1),
    model: generationModelValue(form.model),
    decoder: sameModelValue(form.decoder, defaultDecoderForGenerationModel(form.model)),
    backend: "mlx",
    session_id: sessionId,
    notes,
    metadata,
  };
  if (sourceArtifactId) {
    base.source_artifact_id = sourceArtifactId;
  }
  if (mode === "generate.audio_to_audio" || mode === "generate.inpaint") {
    if (sourceArtifact?.kind !== "audio") throw new Error("Generation mode requires a selected audio artifact.");
    const audioPayload: AudioToAudioPayload = {
      ...base,
      source_artifact_id: sourceArtifact.artifact_id,
      init_noise_level: numberValue(form.init_noise_level, 0.7),
    };
    if (mode === "generate.audio_to_audio") return audioPayload;
    return {
      ...audioPayload,
      inpaint_start_seconds: numberValue(form.inpaint_start_seconds, 0),
      inpaint_end_seconds: numberValue(form.inpaint_end_seconds, 2),
    };
  }
  return base;
}

export function buildLatentEncodePayload({
  form,
  artifact,
  sessionId,
}: {
  form: Record<string, RecipeValue>;
  artifact: ArtifactRecord;
  sessionId?: string | null;
}): LatentEncodePayload {
  return {
    source_artifact_id: artifact.artifact_id,
    model: sameModelValue(form.model, "same-l"),
    backend: torchBackendValue(form.backend),
    chunked: Boolean(form.chunked),
    chunk_size: Math.max(1, Math.trunc(numberValue(form.chunk_size, 128))),
    overlap: Math.max(0, Math.trunc(numberValue(form.overlap, 32))),
    prompt: optionalString(form.prompt),
    notes: optionalString(form.notes),
    session_id: sessionId,
  };
}

export function buildLatentDecodePayload({
  form,
  artifact,
  sessionId,
}: {
  form: Record<string, RecipeValue>;
  artifact: ArtifactRecord;
  sessionId?: string | null;
}): LatentDecodePayload {
  return {
    source_artifact_id: artifact.artifact_id,
    model: sameModelValue(form.model, "same-l"),
    backend: torchBackendValue(form.backend),
    chunked: Boolean(form.chunked),
    chunk_size: Math.max(1, Math.trunc(numberValue(form.chunk_size, 128))),
    overlap: Math.max(0, Math.trunc(numberValue(form.overlap, 32))),
    notes: optionalString(form.notes),
    session_id: sessionId,
  };
}

export function operatorReady(
  config: OperatorPayloadConfig,
  form: Record<string, RecipeValue>,
  selectedArtifact: ArtifactRecord | null,
  donorArtifactId: string,
) {
  if (selectedArtifact?.kind !== "latent") return false;
  if (operatorUsesDonor(config, form) && !donorArtifactId) return false;
  return config.fields.every((field) => !field.required || Boolean(stringValue(form[field.key])));
}

export function buildOperatorParams(config: OperatorPayloadConfig, form: Record<string, RecipeValue>) {
  const params: Record<string, unknown> = {};
  for (const field of config.fields) {
    if (field.key === "backend") continue;
    const value = form[field.key];
    if (field.type === "checkbox") {
      params[field.key] = Boolean(value);
      continue;
    }
    if (value === undefined || stringValue(value) === "") continue;
    if (field.key === "channels") {
      params[field.key] = parseNumberList(value).map((item) => Math.trunc(item));
    } else if (field.key === "pca_component_gains") {
      params[field.key] = parseNumberList(value);
    } else {
      params[field.key] = value;
    }
  }
  return params;
}

export function operatorBackend(form: Record<string, RecipeValue>, fallback: OperatorPayloadConfig["defaultBackend"]) {
  const value = stringValue(form.backend);
  return value === "torch_cpu" || value === "torch_mps" ? value : fallback;
}

export function operatorSeed(form: Record<string, RecipeValue>, fallback: number) {
  const value = form.seed;
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const parsed = Number(stringValue(value));
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function operatorUsesDonor(config: OperatorPayloadConfig, form: Record<string, RecipeValue>) {
  if (config.requiresDonor) return true;
  if (config.value !== "latent.dsp") return false;
  const mode = stringValue(form.mode);
  return mode === "fft_phase_blend" || mode === "fft_mag_phase_graft" || mode === "fft_phase_from_donor";
}

export function stringValue(value: RecipeValue | undefined) {
  if (value === undefined || value === null) return "";
  return String(value).trim();
}

export function parseNumberList(value: RecipeValue) {
  return stringValue(value)
    .split(/[,\s]+/)
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));
}

function mergeRecipeField(field: RecipeField, specField: OperatorFieldSpec): RecipeField {
  const converted = recipeFieldFromSpec(specField);
  return {
    ...field,
    defaultValue: field.defaultValue ?? converted.defaultValue,
    required: Boolean(field.required || converted.required),
    advanced: Boolean(field.advanced || converted.advanced),
    min: converted.min ?? field.min,
    max: converted.max ?? field.max,
    step: converted.step ?? field.step,
    options: converted.options?.length ? converted.options : field.options,
    artifactKinds: converted.artifactKinds?.length ? converted.artifactKinds : field.artifactKinds,
    placeholder: field.placeholder ?? converted.placeholder,
    description: field.description ?? converted.description,
    type: convergedFieldType(field, converted),
  };
}

function recipeFieldFromSpec(field: OperatorFieldSpec): RecipeField {
  return {
    key: field.key,
    label: field.label,
    type: recipeFieldTypeFromSpec(field.type),
    defaultValue: recipeValueFromUnknown(field.default),
    required: field.required,
    advanced: field.advanced,
    min: nullToUndefined(field.min),
    max: nullToUndefined(field.max),
    step: nullToUndefined(field.step),
    options: field.options.map((option) => ({ value: option.value, label: option.label ?? option.value })),
    artifactKinds: field.artifact_kinds.filter(isArtifactKind),
    placeholder: field.placeholder ?? undefined,
    description: field.description ?? undefined,
  };
}

function convergedFieldType(field: RecipeField, converted: RecipeField): RecipeFieldType {
  if (converted.type === "artifact-path") return "artifact-path";
  if (field.type === "text" && converted.type !== "text") return converted.type;
  if (field.type === "path" && converted.type === "select") return converted.type;
  return field.type;
}

function recipeFieldTypeFromSpec(type: string): RecipeFieldType {
  if (
    type === "text" ||
    type === "number" ||
    type === "range" ||
    type === "select" ||
    type === "checkbox" ||
    type === "path" ||
    type === "artifact-path"
  ) {
    return type;
  }
  return "text";
}

function recipeValueFromUnknown(value: unknown): RecipeValue | undefined {
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") return value;
  return undefined;
}

function nullToUndefined(value: number | null | undefined): number | undefined {
  return value ?? undefined;
}

function isArtifactKind(value: string): value is ArtifactKind {
  return value === "audio" || value === "latent" || value === "bundle" || value === "recipe" || value === "text";
}

function optionalString(value: RecipeValue | undefined) {
  const normalized = stringValue(value);
  return normalized ? normalized : null;
}

function numberValue(value: RecipeValue | undefined, fallback: number) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const parsed = Number(stringValue(value));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function optionalInteger(value: RecipeValue | undefined) {
  const parsed = numberValue(value, Number.NaN);
  return Number.isFinite(parsed) ? Math.trunc(parsed) : null;
}

export function defaultDecoderForGenerationModel(value: RecipeValue | undefined): GenerateTextPayload["decoder"] {
  return generationModelValue(value) === "medium" ? "same-l" : "same-s";
}

function generationModelValue(value: RecipeValue | undefined): GenerateTextPayload["model"] {
  const normalized = stringValue(value);
  if (normalized === "sm-music" || normalized === "sm-sfx" || normalized === "medium") return normalized;
  return "medium";
}

function sameModelValue(value: RecipeValue | undefined, fallback: GenerateTextPayload["decoder"] = "same-l"): "same-s" | "same-l" {
  const normalized = stringValue(value);
  if (normalized === "same-s" || normalized === "same-l") return normalized;
  return fallback;
}

function torchBackendValue(value: RecipeValue | undefined): "torch_mps" | "torch_cpu" {
  return stringValue(value) === "torch_cpu" ? "torch_cpu" : "torch_mps";
}
