export interface BundleReuseAction {
  label: string;
  fieldKey: string;
  mode: string;
  value?: string;
}

export interface BundleReuseContext {
  kind?: string | null;
  operator?: string | null;
  prompt?: string | null;
}

export function bundleReuseActionsForContext(context: BundleReuseContext): BundleReuseAction[] {
  const kind = normalize(context.kind);
  const operator = normalize(context.operator);
  const prompt = context.prompt?.trim() || undefined;
  const actions: BundleReuseAction[] = [];
  if (kind === "profile" || operator.includes("style_profile") || operator.includes("positive_style_profile")) {
    actions.push({ label: "Use as profile", fieldKey: "profile_path", mode: "experiment.style_profile.generate" });
    actions.push({ label: "Use memory", fieldKey: "target_memory_path", mode: "experiment.style_profile.build" });
  }
  if (kind === "vectors" || operator.includes("vectors") || operator.includes("direction")) {
    actions.push({ label: "Sweep vectors", fieldKey: "vectors_path", mode: "experiment.alpha_sweep" });
    actions.push({ label: "Use direction", fieldKey: "direction_path", mode: "experiment.style_direction.generate" });
  }
  if (kind === "soft-prompt" || operator.includes("soft_prompt")) {
    actions.push({ label: "Use soft prompt", fieldKey: "soft_prompt_path", mode: "experiment.soft_prompt.generate" });
  }
  if (kind === "prompt-search" || operator.includes("prompt_search")) {
    actions.push({
      label: "Use prompt in sweep",
      fieldKey: "prompt",
      mode: "experiment.alpha_sweep",
      value: prompt,
    });
  }
  if (kind === "memory" || operator.includes("memory")) {
    actions.push({ label: "Use as target memory", fieldKey: "target_memory_path", mode: "experiment.style_profile.build" });
    actions.push({ label: "Use as reference", fieldKey: "reference_memory_path", mode: "experiment.style_profile.build" });
  }
  if (kind === "dataset" || operator.includes("dataset.pre_encode")) {
    actions.push({ label: "Use encoded dataset", fieldKey: "encoded_dir", mode: "training.lora" });
  }
  if (kind === "training" || operator.includes("lora")) {
    actions.push({ label: "Use checkpoint", fieldKey: "lora_checkpoint", mode: "training.lora" });
  }
  return dedupeReuseActions(actions);
}

function normalize(value: string | null | undefined) {
  return value?.trim().toLowerCase() ?? "";
}

function dedupeReuseActions(actions: BundleReuseAction[]) {
  const seen = new Set<string>();
  return actions.filter((action) => {
    const key = `${action.mode}:${action.fieldKey}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
