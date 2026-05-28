import type { RecipeValue } from "./recipeFormModel";

const STORAGE_KEY = "sa3_operator_presets";

export interface OperatorPreset {
  id: string;
  name: string;
  operator: string;
  form: Record<string, RecipeValue>;
  donorArtifactId?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface OperatorPresetInput {
  id?: string;
  name: string;
  operator: string;
  form: Record<string, RecipeValue>;
  donorArtifactId?: string | null;
  createdAt?: string;
  now?: string;
}

export function loadOperatorPresets(storage: Storage = localStorage): OperatorPreset[] {
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isOperatorPreset).sort(sortPresets);
  } catch {
    return [];
  }
}

export function persistOperatorPresets(presets: OperatorPreset[], storage: Storage = localStorage): void {
  storage.setItem(STORAGE_KEY, JSON.stringify(presets.filter(isOperatorPreset).sort(sortPresets)));
}

export function createOperatorPreset(input: OperatorPresetInput): OperatorPreset {
  const now = input.now ?? new Date().toISOString();
  return {
    id: input.id ?? newPresetId(),
    name: input.name.trim() || "Untitled preset",
    operator: input.operator,
    form: sanitizeForm(input.form),
    donorArtifactId: input.donorArtifactId || null,
    createdAt: input.createdAt ?? now,
    updatedAt: now,
  };
}

export function upsertOperatorPreset(presets: OperatorPreset[], preset: OperatorPreset): OperatorPreset[] {
  return [preset, ...presets.filter((item) => item.id !== preset.id)].sort(sortPresets);
}

export function deleteOperatorPreset(presets: OperatorPreset[], presetId: string): OperatorPreset[] {
  return presets.filter((preset) => preset.id !== presetId).sort(sortPresets);
}

function sanitizeForm(form: Record<string, RecipeValue>): Record<string, RecipeValue> {
  return Object.fromEntries(
    Object.entries(form).filter(([, value]) => (
      typeof value === "string" || typeof value === "number" || typeof value === "boolean"
    )),
  );
}

function isOperatorPreset(value: unknown): value is OperatorPreset {
  const record = value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
  if (!record) return false;
  return (
    typeof record.id === "string"
    && typeof record.name === "string"
    && typeof record.operator === "string"
    && record.form !== null
    && typeof record.form === "object"
    && !Array.isArray(record.form)
    && typeof record.createdAt === "string"
    && typeof record.updatedAt === "string"
  );
}

function sortPresets(a: OperatorPreset, b: OperatorPreset): number {
  return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
}

function newPresetId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `preset_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
