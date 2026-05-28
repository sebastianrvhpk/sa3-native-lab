import type { RecipeValue } from "./recipeFormModel";
import type { ArtifactRecord } from "./types";

export interface PromptSearchPreset {
  id: string;
  label: string;
  modeLabel: string;
  cost: string;
  intent: string;
  fields: Record<string, RecipeValue>;
}

export interface PromptSearchScorerNote {
  scorer: string;
  label: string;
  cost: string;
  guidance: string;
  maturity: "ready" | "probe" | "queued";
}

export interface PromptSearchVocabularySet {
  id: string;
  label: string;
  focus: string;
  terms: string;
}

export interface PromptSearchAxisSet {
  id: string;
  label: string;
  focus: string;
  axes: string;
}

export interface PromptSearchHistoryRow {
  prompt: string;
  total: number;
  keeper: number;
  maybe: number;
  rejected: number;
  latestAt: string;
  latestNote?: string | null;
}

export const promptSearchPresets: readonly PromptSearchPreset[] = [
  {
    id: "mode2-hard-token",
    label: "Hard-token probe",
    modeLabel: "Mode 2",
    cost: "fast CPU",
    intent: "Cheap babble search over explicit vocabulary before spending Medium cycles.",
    fields: {
      search_mode: "beam",
      scorer: "lexical_probe",
      backend: "cpu",
      model: "medium",
      duration_seconds: 0,
      tokens_generated: 4,
      beam_width: 4,
      branch_factor: 64,
      score_samples: 1,
      seed: 0,
    },
  },
  {
    id: "mode3-readable",
    label: "Readable prompt",
    modeLabel: "Mode 3",
    cost: "fast CPU",
    intent: "Constrain the search toward interpretable modifier families.",
    fields: {
      search_mode: "coordinate",
      scorer: "lexical_probe",
      backend: "cpu",
      model: "medium",
      duration_seconds: 0,
      tokens_generated: 5,
      rounds: 2,
      runs: 4,
      score_samples: 1,
      modifier_axes: "bright|dark|warm|cold; sparse|dense|percussive|sustained; clean|noisy|wide|narrow",
      seed: 0,
    },
  },
  {
    id: "medium-flow-check",
    label: "Medium flow check",
    modeLabel: "Flow scorer",
    cost: "slow MPS",
    intent: "Small model-backed score probe for promising targets.",
    fields: {
      search_mode: "beam",
      scorer: "sa3_flow_probe",
      backend: "torch_mps",
      model: "medium",
      duration_seconds: 6,
      tokens_generated: 3,
      beam_width: 3,
      branch_factor: 24,
      score_samples: 2,
      candidate_batch_size: 8,
      timestep_values: "0.18,0.5,0.82",
      shared_noise: true,
      antithetic_noise: false,
      normalize_mse: true,
      seed: 0,
    },
  },
];

export const promptSearchScorerNotes: readonly PromptSearchScorerNote[] = [
  {
    scorer: "lexical_probe",
    label: "Lexical probe",
    cost: "fast CPU",
    guidance: "Use for wiring, vocabulary shape, and cheap prompt-family iteration. It is not an audio-text inversion score.",
    maturity: "ready",
  },
  {
    scorer: "sa3_flow_probe",
    label: "SA3 flow probe",
    cost: "slow MPS",
    guidance: "Use on short targets or promising candidates. Keep score samples low until the take comparison has enough evidence.",
    maturity: "probe",
  },
  {
    scorer: "clap",
    label: "CLAP",
    cost: "queued",
    guidance: "Reserved behind the scorer contract. Add only after Medium flow comparisons produce enough listened examples.",
    maturity: "queued",
  },
];

export const promptSearchVocabularySets: readonly PromptSearchVocabularySet[] = [
  {
    id: "texture-color",
    label: "Texture color",
    focus: "timbre",
    terms: "warm, cold, bright, dark, muted, glassy, metallic, soft, harsh, noisy, clean, saturated, airy, dusty, crystalline",
  },
  {
    id: "motion-rhythm",
    label: "Motion rhythm",
    focus: "movement",
    terms: "pulsing, drifting, swelling, stuttering, rolling, granular, shimmering, percussive, sustained, sparse, dense, syncopated, looping, fractured",
  },
  {
    id: "space-scene",
    label: "Space scene",
    focus: "place",
    terms: "wide stereo, narrow, distant, intimate, reverberant, dry, cinematic, ambient, underwater, industrial, acoustic, electronic, cavernous",
  },
];

export const promptSearchAxisSets: readonly PromptSearchAxisSet[] = [
  {
    id: "readable-timbre",
    label: "Readable timbre",
    focus: "Mode 3",
    axes: "bright|dark|warm|cold; clean|noisy|glassy|metallic; soft|harsh|muted|saturated",
  },
  {
    id: "readable-motion",
    label: "Readable motion",
    focus: "Mode 3",
    axes: "sparse|dense|percussive|sustained; pulsing|drifting|swelling|stuttering; steady|fractured|looping|syncopated",
  },
];

export function promptSearchPresetById(presetId: string): PromptSearchPreset | undefined {
  return promptSearchPresets.find((preset) => preset.id === presetId);
}

export function promptSearchScorerNote(scorer: RecipeValue | undefined): PromptSearchScorerNote {
  const value = typeof scorer === "string" ? scorer : "";
  return promptSearchScorerNotes.find((note) => note.scorer === value) ?? promptSearchScorerNotes[0];
}

export function applyPromptSearchPreset(
  form: Record<string, RecipeValue>,
  presetId: string,
): Record<string, RecipeValue> {
  const preset = promptSearchPresetById(presetId);
  return preset ? { ...form, ...preset.fields } : form;
}

export function applyPromptSearchVocabularySet(
  form: Record<string, RecipeValue>,
  setId: string,
): Record<string, RecipeValue> {
  const set = promptSearchVocabularySets.find((item) => item.id === setId);
  return set ? { ...form, vocabulary: set.terms } : form;
}

export function applyPromptSearchAxisSet(
  form: Record<string, RecipeValue>,
  setId: string,
): Record<string, RecipeValue> {
  const set = promptSearchAxisSets.find((item) => item.id === setId);
  return set ? { ...form, modifier_axes: set.axes, search_mode: "coordinate" } : form;
}

export function promptSearchHistoryRows(artifacts: readonly ArtifactRecord[]): PromptSearchHistoryRow[] {
  const rows = new Map<string, PromptSearchHistoryRow>();
  for (const artifact of artifacts) {
    if (!isPromptSearchTake(artifact)) continue;
    const prompt = artifact.prompt?.trim();
    if (!prompt) continue;
    const decision = artifact.metadata.listening_decision;
    const existing = rows.get(prompt) ?? {
      prompt,
      total: 0,
      keeper: 0,
      maybe: 0,
      rejected: 0,
      latestAt: artifact.created_at,
      latestNote: null,
    };
    existing.total += 1;
    if (decision === "keeper") existing.keeper += 1;
    if (decision === "maybe") existing.maybe += 1;
    if (decision === "rejected") existing.rejected += 1;
    if (Date.parse(artifact.created_at) >= Date.parse(existing.latestAt)) {
      existing.latestAt = artifact.created_at;
      existing.latestNote = promptSearchTakeNote(artifact);
    }
    rows.set(prompt, existing);
  }
  return [...rows.values()].sort((a, b) =>
    b.keeper - a.keeper
    || b.maybe - a.maybe
    || b.total - a.total
    || Date.parse(b.latestAt) - Date.parse(a.latestAt),
  );
}

function isPromptSearchTake(artifact: ArtifactRecord): boolean {
  return artifact.kind === "audio" && artifact.metadata.generation_origin === "prompt_search_candidate";
}

function promptSearchTakeNote(artifact: ArtifactRecord): string | null {
  if (typeof artifact.metadata.listening_decision_note === "string" && artifact.metadata.listening_decision_note.trim()) {
    return artifact.metadata.listening_decision_note.trim();
  }
  return artifact.notes?.trim() || null;
}
