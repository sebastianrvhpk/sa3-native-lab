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
      ...productFieldCopy(field, context),
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
  if (context.gestureId === "borrow_texture") return ["mode", "fraction", "amount", "channels", "start_channel", "block_size", "strength", "seed"];
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
    "return_code",
  ];
}

function morphPrimaryKeys(operatorMode?: LatentOperatorMode): readonly string[] {
  if (operatorMode === "latent.blur") return ["mode", "strength", "temporal_radius", "channel_radius", "seed"];
  if (operatorMode === "latent.dsp") return ["mode", "strength", "gain", "center", "seed"];
  if (operatorMode === "latent.graft") return ["mode", "fraction", "amount", "channels", "start_channel", "block_size", "seed"];
  if (operatorMode === "latent.renoise") return ["mode", "fraction", "sigma", "channels", "start_channel", "block_size", "seed"];
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
  if (experimentMode === "experiment.soft_prompt.optimize") return ["target_audio_path", "seed_prompt", "optimization_steps", "model"];
  if (experimentMode === "experiment.soft_prompt.generate") return ["soft_prompt_path", "model", "steps", "cfg_scale", "seed"];
  if (experimentMode === "experiment.geometry_audit") return ["n_components"];
  if (experimentMode === "experiment.audio_residual_vectors.extract") return ["positive_path", "negative_path", "baseline", "prompt", "model"];
  if (experimentMode === "experiment.sa3_vectors.extract") return ["axis", "num_pairs", "model", "duration_seconds", "seed"];
  return ["positive_path", "negative_path", "input_path", "name", "model"];
}

function productFieldCopy(field: RecipeField, context: TuneFieldGroupContext): Partial<RecipeField> {
  const label = productFieldLabel(field.key, context);
  const description = productFieldDescription(field.key, context);
  return {
    ...(label ? { label } : {}),
    ...(description && !field.description ? { description } : {}),
  };
}

function productFieldLabel(key: string, context: TuneFieldGroupContext): string | null {
  if (key === "duration_seconds") return "Length";
  if (key === "init_noise_level") return context.generationMode === "generate.audio_to_audio" ? "Variation amount" : "Rewrite amount";
  if (key === "inpaint_start_seconds") return "Edit starts";
  if (key === "inpaint_end_seconds") return "Edit ends";
  if (key === "profile_path") return "Profile source";
  if (key === "direction_path") return "Direction source";
  if (key === "vectors_path") return "Sweep source";
  if (key === "target_audio_path") return "Target sound";
  if (key === "source_audio_path") return "Source sound";
  if (key === "soft_prompt_path") return "Soft prompt source";
  if (key === "data_dir") return "Audio folder";
  if (key === "encoded_dataset_path") return "Encoded dataset";
  if (key === "checkpoint_path") return "Checkpoint";
  if (key === "search_mode") return "Candidate strategy";
  if (key === "scorer") return "Prompt probe";
  if (key === "channels") return "Exact channels";
  if (key === "start_channel") return "Channel start";
  if (key === "block_size") return "Channel span";
  if (key === "temporal_radius") return "Time smear";
  if (key === "channel_radius") return "Channel smear";
  if (key === "sigma") return "Noise amount";
  if (key === "amount" && context.operatorMode === "latent.graft") return "Borrow amount";
  return null;
}

function productFieldDescription(key: string, context: TuneFieldGroupContext): string | null {
  if (key === "init_noise_level") return "How much the current sound can move while keeping its source identity nearby.";
  if (key === "target_audio_path") return "Audio whose qualities the probe or optimization should listen against.";
  if (key === "vectors_path") return "Reusable vector bundle for sweep listening.";
  if (key === "direction_path") return "Reusable direction bundle for steering generation.";
  if (key === "profile_path") return "Reusable profile bundle for profile-guided generation.";
  if (key === "channels") return "Comma-separated latent channel numbers; exact submitted values remain visible.";
  if (key === "start_channel" || key === "block_size") return "Contiguous latent-channel block for channel-mask gestures.";
  if (key === "temporal_radius" && context.operatorMode === "latent.blur") return "Radius of the latent-time smear, measured in latent frames.";
  if (key === "scorer") return "Probe used to rank candidate prompts; Medium flow probe is slower and model-backed.";
  return null;
}
