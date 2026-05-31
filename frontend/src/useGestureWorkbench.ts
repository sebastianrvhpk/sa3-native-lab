import { useState } from "react";

import { gestureById, type GestureId } from "./gestureModel";
import type { MemoryReuseAction } from "./memoryModel";
import type { ProductNextAction } from "./nextActionModel";
import {
  defaultFieldForm,
  defaultDecoderForGenerationModel,
  operatorUsesDonor,
  type GenerationMode,
  type RecipeValue,
} from "./recipeFormModel";
import type { ArtifactRecord } from "./types";
import {
  defaultExperimentForm,
  defaultGenerationForm,
  defaultOperatorForm,
  isExperimentMode,
  operatorCatalog,
  sameConfig,
  type ExperimentMode,
  type LatentOperatorMode,
} from "./workbenchConfigs";

/*
 * Boundary: this hook owns semantic workbench state only: active gesture, Tune
 * forms, donor/source reuse, next-action routing, prompt seeding, and bundle
 * reuse. React Query mutations, job-event landing, archive/recover, pending-take
 * selection, and other side effects stay in App.tsx. A future descriptor helper
 * may shape labels/readiness/disabled reasons, but it should not move mutations
 * into this hook.
 */

export interface UseGestureWorkbenchInput {
  allArtifacts: readonly ArtifactRecord[];
  selectedArtifact: ArtifactRecord | null;
  selectArtifact: (artifactId: string | null) => void;
  setCompare: (slot: "a" | "b", artifactId: string | null) => void;
  artifactPathForField: (artifact: ArtifactRecord, fieldKey: string) => string;
}

export function useGestureWorkbench({
  allArtifacts,
  selectedArtifact,
  selectArtifact,
  setCompare,
  artifactPathForField,
}: UseGestureWorkbenchInput) {
  const [generationMode, setGenerationMode] = useState<GenerationMode>("generate.text_to_audio");
  const [generationForm, setGenerationForm] = useState<Record<string, RecipeValue>>(() => defaultGenerationForm("generate.text_to_audio"));
  const [sameForm, setSameForm] = useState<Record<string, RecipeValue>>(() => defaultFieldForm(sameConfig));
  const [operator, setOperator] = useState<LatentOperatorMode>("latent.cyclic_roll");
  const [operatorForm, setOperatorForm] = useState<Record<string, RecipeValue>>(() => defaultOperatorForm("latent.cyclic_roll"));
  const [activeGestureId, setActiveGestureId] = useState<GestureId>("make");
  const [donorArtifactId, setDonorArtifactId] = useState("");
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>("experiment.audio_style_vectors");
  const [experimentForm, setExperimentForm] = useState<Record<string, RecipeValue>>(() => defaultExperimentForm("experiment.audio_style_vectors"));

  const setExperimentField = (key: string, value: RecipeValue) => {
    setExperimentForm((current) => ({ ...current, [key]: value }));
  };

  const setGenerationField = (key: string, value: RecipeValue) => {
    setGenerationForm((current) => {
      const next = { ...current, [key]: value };
      if (key === "model") next.decoder = defaultDecoderForGenerationModel(value);
      return next;
    });
  };

  const setSameField = (key: string, value: RecipeValue) => {
    setSameForm((current) => ({ ...current, [key]: value }));
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

  const selectGesture = (gestureId: GestureId) => {
    setActiveGestureId(gestureId);
    const gesture = gestureById(gestureId);
    if (gesture.defaultGenerationMode) {
      setGenerationMode(gesture.defaultGenerationMode);
      setGenerationForm(defaultGenerationForm(gesture.defaultGenerationMode));
    }
    if (gesture.defaultOperatorMode) {
      selectOperatorMode(gesture.defaultOperatorMode);
    }
    if (gesture.defaultExperimentMode) {
      selectExperimentMode(gesture.defaultExperimentMode);
    }
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

  const usePromptCandidate = (candidatePrompt: string) => {
    setGenerationMode("generate.text_to_audio");
    setGenerationForm((current) => ({ ...current, prompt: candidatePrompt }));
  };

  const applyNextAction = (action: ProductNextAction) => {
    if (!action.available) return;
    if (action.gestureId) selectGesture(action.gestureId);
    if (action.generationMode) {
      setGenerationMode(action.generationMode);
      setGenerationForm(defaultGenerationForm(action.generationMode));
    }
    if (action.operatorMode) {
      selectOperatorMode(action.operatorMode);
    }
    if (action.experimentMode && isExperimentMode(action.experimentMode)) {
      setExperimentMode(action.experimentMode);
      setExperimentForm({
        ...defaultExperimentForm(action.experimentMode),
        ...(selectedArtifact && action.fieldKey
          ? { [action.fieldKey]: action.value ?? artifactPathForField(selectedArtifact, action.fieldKey) }
          : {}),
      });
      setActiveGestureId("steer");
    }
    if (action.kind === "remember") {
      setActiveGestureId("remember");
    }
  };

  const applyMemoryAction = (artifact: ArtifactRecord, action: MemoryReuseAction) => {
    if (!action.available || action.intent === "recover") return false;
    if (action.intent === "source") {
      selectArtifact(artifact.artifact_id);
      if (artifact.kind === "audio") setCompare("b", artifact.artifact_id);
      selectGesture(artifact.kind === "latent" ? "morph" : "continue");
      return true;
    }
    if (action.intent === "anchor") {
      setCompare("a", artifact.artifact_id);
      selectArtifact(artifact.artifact_id);
      return true;
    }
    if (action.intent === "donor") {
      useArtifactAsDonor(artifact.artifact_id);
      setActiveGestureId("borrow_texture");
      return true;
    }
    if (action.intent === "prompt_seed") {
      usePromptCandidate(action.value ?? artifact.prompt ?? artifact.label ?? artifact.notes ?? "");
      setActiveGestureId("make");
      return true;
    }
    if (action.intent === "advanced_gesture" && action.fieldKey && action.mode && isExperimentMode(action.mode)) {
      useBundleInRecipe(action.fieldKey, action.value ?? artifactPathForField(artifact, action.fieldKey), action.mode);
      setActiveGestureId("steer");
      return true;
    }
    return false;
  };

  return {
    activeGestureId,
    setActiveGestureId,
    generationMode,
    setGenerationMode,
    generationForm,
    setGenerationForm,
    sameForm,
    setSameForm,
    operator,
    setOperator,
    operatorForm,
    setOperatorForm,
    donorArtifactId,
    setDonorArtifactId,
    experimentMode,
    setExperimentMode,
    experimentForm,
    setExperimentForm,
    selectGesture,
    selectOperatorMode,
    selectExperimentMode,
    setExperimentField,
    setGenerationField,
    setSameField,
    setOperatorField,
    applyNextAction,
    applyMemoryAction,
    useArtifactAsDonor,
    useBundleInRecipe,
    usePromptCandidate,
  };
}
