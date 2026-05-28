import type { RecipeValue } from "./recipeFormModel";

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
