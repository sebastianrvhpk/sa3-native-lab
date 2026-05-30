import type { GestureId } from "./gestureModel";
import type { FieldConfig, GenerationMode, RecipeField } from "./recipeFormModel";
import type { ExperimentMode, LatentOperatorMode } from "./workbenchConfigs";

export interface TuneFieldGroupContext {
  gestureId: GestureId;
  generationMode?: GenerationMode;
  operatorMode?: LatentOperatorMode;
  experimentMode?: ExperimentMode;
}

export function withTuneFieldGroups<TConfig extends FieldConfig>(config: TConfig, context: TuneFieldGroupContext): TConfig {
  const primary = new Set(primaryFieldKeysForGesture(context));
  const inspectOnly = new Set(inspectOnlyFieldKeys());
  const fields: RecipeField[] = config.fields.map((field) => {
    const isPrimary = primary.has(field.key) || Boolean(field.required && !inspectOnly.has(field.key));
    return {
      ...field,
      advanced: inspectOnly.has(field.key) ? true : !isPrimary,
    };
  });
  return { ...config, fields } as TConfig;
}

export function primaryFieldKeysForGesture(context: TuneFieldGroupContext): readonly string[] {
  if (context.gestureId === "make") return ["prompt", "duration_seconds", "seed", "model"];
  if (context.gestureId === "continue" || context.gestureId === "vary") {
    return ["prompt", "init_noise_level", "duration_seconds", "seed", "model"];
  }
  if (context.gestureId === "encode") return ["model", "prompt"];
  if (context.gestureId === "decode") return ["model"];
  if (context.gestureId === "borrow_texture") return ["mode", "fraction", "amount", "strength", "seed"];
  if (context.gestureId === "morph") return morphPrimaryKeys(context.operatorMode);
  if (context.gestureId === "steer") return steerPrimaryKeys(context.experimentMode);
  return ["label", "tags", "notes"];
}

export function advancedFieldKeysForGesture(context: TuneFieldGroupContext, config: FieldConfig): readonly string[] {
  const primary = new Set(primaryFieldKeysForGesture(context));
  return config.fields.map((field) => field.key).filter((key) => !primary.has(key));
}

export function inspectOnlyFieldKeys(): readonly string[] {
  return [
    "backend",
    "device",
    "no_half",
    "model_half",
    "logger",
    "checkpoint_every",
    "log_every",
    "demo_every",
    "num_workers",
    "return_code",
  ];
}

function morphPrimaryKeys(operatorMode?: LatentOperatorMode): readonly string[] {
  if (operatorMode === "latent.blur") return ["mode", "strength", "temporal_radius", "channel_radius", "seed"];
  if (operatorMode === "latent.dsp") return ["mode", "strength", "gain", "center", "seed"];
  if (operatorMode === "latent.renoise") return ["mode", "fraction", "sigma", "seed"];
  if (operatorMode === "latent.graft") return ["mode", "fraction", "amount", "seed"];
  return ["shift_frames", "strength", "symmetric", "seed"];
}

function steerPrimaryKeys(experimentMode?: ExperimentMode): readonly string[] {
  if (experimentMode === "experiment.style_profile.generate") return ["profile_path", "prompt", "alpha", "model", "duration_seconds", "seed"];
  if (experimentMode === "experiment.style_direction.generate" || experimentMode === "experiment.audio_direction.generate") {
    return ["direction_path", "prompt", "alpha", "std_alpha", "model", "duration_seconds", "seed"];
  }
  if (experimentMode === "experiment.alpha_sweep") return ["vectors_path", "prompt", "alphas", "model", "duration_seconds", "seed"];
  if (experimentMode === "experiment.prompt_search") return ["target_audio_path", "seed_prompt", "search_mode", "scorer", "model"];
  if (experimentMode === "memory.query") return ["top_k", "metric", "exclude_self"];
  if (experimentMode === "dataset.pre_encode") return ["data_dir", "model", "batch_size"];
  if (experimentMode === "training.lora") return ["encoded_dir", "data_dir", "model", "steps", "rank"];
  if (experimentMode === "experiment.soft_prompt.optimize") return ["target_audio_path", "seed_prompt", "optimization_steps", "model"];
  if (experimentMode === "experiment.soft_prompt.generate") return ["soft_prompt_path", "model", "steps", "cfg_scale", "seed"];
  if (experimentMode === "experiment.geometry_audit") return ["n_components"];
  if (experimentMode === "experiment.audio_residual_vectors.extract") return ["positive_path", "negative_path", "baseline", "prompt", "model"];
  if (experimentMode === "experiment.sa3_vectors.extract") return ["axis", "num_pairs", "model", "duration_seconds", "seed"];
  return ["positive_path", "negative_path", "input_path", "name", "model"];
}
