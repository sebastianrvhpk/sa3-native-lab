import { z } from "zod";

import type { ExperimentPayload } from "./api";
import type { ArtifactRecord } from "./types";

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

export function validateRecipeField(field: RecipeField, value: RecipeValue | undefined): string | undefined {
  if (field.required && !stringValue(value)) return `${field.label} is required`;
  if ((field.type === "number" || field.type === "range") && stringValue(value)) {
    const numberValue = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(numberValue)) return `${field.label} must be a number`;
    if (field.min !== undefined && numberValue < field.min) return `${field.label} must be at least ${field.min}`;
    if (field.max !== undefined && numberValue > field.max) return `${field.label} must be at most ${field.max}`;
  }
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
