import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  Box,
  Braces,
  FileAudio,
  GitFork,
  LoaderCircle,
  Play,
  Settings,
  Upload,
  Waves,
} from "lucide-react";

import modelImage from "../../stable-audio-3.png";
import {
  createApi,
  type ArtifactAnnotationPayload,
  type AudioToAudioPayload,
  type GenerateTextPayload,
  type InpaintPayload,
  type RecipeForkPayload,
} from "./api";
import { ArtifactBadge } from "./artifactDisplay";
import { artifactName } from "./artifactUtils";
import { AuditionStackPanel } from "./auditionStackPanel";
import { ComparePanel } from "./comparePanel";
import { createControlPlaneClient, DEFAULT_CONTROL_PLANE_URL, type ResultFamily, type WorkbenchState } from "./controlPlane";
import { ForkRecipePanel } from "./forkRecipePanel";
import { describeGestureAction } from "./gestureActionDescriptor";
import { buildGestureOptions, gestureById, type GestureId } from "./gestureModel";
import { GestureStrip } from "./gestureStrip";
import { isJobActive, landingArtifactId } from "./jobUtils";
import { type MemoryReuseAction } from "./memoryModel";
import { ModeAtlas } from "./modeAtlas";
import { nextActionsForArtifact, type ProductNextAction } from "./nextActionModel";
import { NextActionsPanel } from "./nextActionsPanel";
import { OperatorPresetRack } from "./operatorPresetRack";
import {
  createOperatorPreset,
  deleteOperatorPreset,
  loadOperatorPresets,
  operatorPresetDiffRows,
  persistOperatorPresets,
  upsertOperatorPreset,
  type OperatorPreset,
} from "./operatorPresets";
import {
  applyPromptSearchAxisSet,
  applyPromptSearchPreset,
  applyPromptSearchVocabularySet,
  promptSearchAxisSets,
  promptSearchHistoryRows,
  promptSearchPresets,
  promptSearchVocabularySets,
} from "./promptSearchPresets";
import { PromptSearchPresetRack } from "./promptSearchRack";
import { RecipeFields } from "./RecipeFields";
import { FamilyDetailPanel, ResultFamilyPanel } from "./resultFamilies";
import { pendingTakesFromJobs } from "./pendingTakeModel";
import { PendingTakesPanel } from "./pendingTakesPanel";
import {
  buildExperimentPayload,
  buildGenerationPayload,
  buildLatentDecodePayload,
  buildLatentEncodePayload,
  buildOperatorParams,
  experimentReady,
  fieldKeys,
  fillMissingFieldDefaults,
  generationReady,
  operatorBackend,
  operatorReady,
  operatorSeed,
  operatorUsesDonor,
  withOperatorSpecFields,
  type GenerationMode,
} from "./recipeFormModel";
import { artifactArchivePayload, artifactRecoveryPayload } from "./sessionRecovery";
import { SessionTray } from "./sessionPanel";
import { buildProductSources } from "./sourceModel";
import { SourcePicker } from "./SourceField";
import { SourceShelf } from "./sourceShelf";
import { SpecCoverage, SpecCoveragePair } from "./specCoverage";
import { Specimen } from "./specimenPanel";
import { BackendPills, InlineJobStatus, ReadinessPanel, RunMonitor } from "./statusPanels";
import { useBenchStore } from "./store";
import { TuneDrawer } from "./tuneDrawer";
import { withTuneFieldGroups } from "./tuneFieldGroups";
import { useGestureWorkbench } from "./useGestureWorkbench";
import type {
  ArtifactRecord,
  HealthResponse,
  JobRecord,
  OperatorName,
  ReadinessCheck,
  Recipe,
  SessionRecord,
} from "./types";
import type { PromptCandidateGenerationRequest } from "./bundleInspector";
import {
  defaultOperatorForm,
  experimentCatalog,
  experimentModes,
  filteredOperatorSpec,
  generationCatalog,
  generationControlKeys,
  generationModes,
  generationSeedFallback,
  isLatentOperatorMode,
  operatorCatalog,
  operatorModes,
  sameConfig,
  sameDecodeControlKeys,
  sameEncodeControlKeys,
  type ExperimentMode,
  type LatentOperatorMode,
} from "./workbenchConfigs";
import {
  activeJobForOperator,
  buildResultFamilies,
  createdAfter,
  filterFamiliesForWork,
  jobFromJobEvent,
  mergeJobRecords,
  parseJobEvent,
  primitiveMetadataValue,
} from "./workbenchModel";

export function App() {
  const queryClient = useQueryClient();
  const { apiBase, setApiBase, selectedArtifactId, selectArtifact, sessionId, sessionStartedAt, setSession, compare, setCompare } = useBenchStore();
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const controlPlaneUrl = DEFAULT_CONTROL_PLANE_URL.trim();
  const controlPlane = useMemo(() => (controlPlaneUrl ? createControlPlaneClient(controlPlaneUrl) : null), [controlPlaneUrl]);
  const useControlPlane = Boolean(controlPlane);

  const health = useQuery({ queryKey: ["health", apiBase], queryFn: api.health, refetchInterval: 3000, enabled: !useControlPlane });
  const readiness = useQuery({ queryKey: ["readiness", apiBase], queryFn: api.readiness, refetchInterval: 30000 });
  const operatorSpecs = useQuery({ queryKey: ["operator-specs", apiBase], queryFn: api.operatorSpecs, staleTime: 30000, enabled: !useControlPlane });
  const sessions = useQuery({ queryKey: ["sessions", apiBase], queryFn: api.sessions, refetchInterval: 3000, enabled: !useControlPlane });
  const modeAtlas = useQuery({ queryKey: ["colab-modes", apiBase], queryFn: api.colabModes, staleTime: Infinity, enabled: !useControlPlane });
  const artifacts = useQuery({ queryKey: ["artifacts", apiBase], queryFn: () => api.artifacts(), refetchInterval: 1500, enabled: !useControlPlane });
  const jobs = useQuery({ queryKey: ["jobs", apiBase], queryFn: api.jobs, refetchInterval: 1000, enabled: !useControlPlane });
  const workbench = useQuery({
    queryKey: ["workbench", controlPlaneUrl, apiBase, sessionId, sessionStartedAt, selectedArtifactId],
    queryFn: () =>
      controlPlane!.workbench.load.query({
        apiBase,
        sessionId,
        sessionStartedAt,
        selectedArtifactId,
      }),
    enabled: useControlPlane,
    refetchInterval: 1500,
  });

  const [operatorPresets, setOperatorPresets] = useState<OperatorPreset[]>(() => loadOperatorPresets());
  const [operatorPresetName, setOperatorPresetName] = useState("");
  const [selectedOperatorPresetId, setSelectedOperatorPresetId] = useState("");
  const [liveJobsById, setLiveJobsById] = useState<Record<string, JobRecord>>({});
  const [forkTarget, setForkTarget] = useState<Recipe | null>(null);
  const [inspectedFamilyId, setInspectedFamilyId] = useState<string | null>(null);

  const workbenchState = workbench.data;
  const allArtifacts = workbenchState?.artifacts ?? artifacts.data ?? [];
  const allSessions = workbenchState?.sessions ?? sessions.data ?? [];
  const activeSession = workbenchState?.activeSession ?? findActiveSession(allSessions, sessionId);
  const activeSessionId = workbenchState?.activeSessionId ?? activeSession?.session_id ?? sessionId;
  const sessionArtifacts = workbenchState?.sessionArtifacts ?? (activeSessionId
    ? allArtifacts.filter((item) => item.session_id === activeSessionId)
    : allArtifacts.filter((item) => createdAfter(item.created_at, sessionStartedAt)));
  const visibleArtifacts = sessionArtifacts;
  const archiveArtifacts = workbenchState?.archiveArtifacts ?? (activeSessionId
    ? allArtifacts.filter((item) => item.session_id !== activeSessionId)
    : allArtifacts.filter((item) => !createdAfter(item.created_at, sessionStartedAt)));
  const selectedArtifact = allArtifacts.find((item) => item.artifact_id === selectedArtifactId) ?? workbenchState?.selectedArtifact ?? visibleArtifacts[0] ?? null;
  const audioArtifacts = allArtifacts.filter((item) => item.kind === "audio");
  const latentArtifacts = allArtifacts.filter((item) => item.kind === "latent");
  const bundleArtifacts = allArtifacts.filter((item) => item.kind === "bundle");
  const gestureWorkbench = useGestureWorkbench({
    allArtifacts,
    selectedArtifact,
    selectArtifact,
    setCompare,
    artifactPathForField,
  });
  const {
    activeGestureId,
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
    experimentForm,
    setExperimentForm,
    selectGesture: selectWorkbenchGesture,
    selectOperatorMode: selectWorkbenchOperatorMode,
    selectExperimentMode,
    setExperimentField,
    setGenerationField,
    setSameField,
    setOperatorField,
    applyNextAction: applyWorkbenchNextAction,
    useArtifactAsDonor: useWorkbenchArtifactAsDonor,
    useBundleInRecipe,
    usePromptCandidate,
  } = gestureWorkbench;
  const sourceRows = useMemo(
    () =>
      buildProductSources([...visibleArtifacts, ...archiveArtifacts], {
        activeSessionId,
        currentArtifactId: selectedArtifact?.artifact_id ?? null,
        anchorArtifactId: compare.a,
        sourceArtifactId: compare.b,
        donorArtifactId,
      }),
    [activeSessionId, archiveArtifacts, compare.a, compare.b, donorArtifactId, selectedArtifact?.artifact_id, visibleArtifacts],
  );
  const serverJobs = workbenchState?.jobs ?? jobs.data ?? [];
  const allJobs = mergeJobRecords(serverJobs, Object.values(liveJobsById));
  const sessionJobs = workbenchState?.sessionJobs ?? (activeSessionId
    ? allJobs.filter((item) => item.recipe.session_id === activeSessionId)
    : allJobs.filter((item) => createdAfter(item.created_at, sessionStartedAt)));
  const archiveJobs = workbenchState?.archiveJobs ?? (activeSessionId
    ? allJobs.filter((item) => item.recipe.session_id !== activeSessionId)
    : allJobs.filter((item) => !createdAfter(item.created_at, sessionStartedAt)));
  const runningJobs = workbenchState?.runningJobs ?? allJobs.filter(isJobActive);
  const pendingTakes = useMemo(
    () =>
      pendingTakesFromJobs(
        sessionJobs
          .filter((job) => isJobActive(job) || ((job.status === "failed" || job.status === "cancelled") && !job.artifact_ids.length))
          .slice(0, 4),
      ),
    [sessionJobs],
  );
  const resultFamilies = workbenchState?.resultFamilies ?? buildResultFamilies(allArtifacts, allJobs);
  const sessionResultFamilies = workbenchState?.sessionResultFamilies ?? filterFamiliesForWork(resultFamilies, sessionArtifacts, sessionJobs);
  const inspectedFamily = sessionResultFamilies.find((family) => family.familyId === inspectedFamilyId) ?? sessionResultFamilies[0] ?? null;
  const latestJobs = workbenchState?.latestJob ? [workbenchState.latestJob, ...allJobs.filter((job) => job.job_id !== workbenchState.latestJob?.job_id)].slice(0, 8) : allJobs.slice(0, 8);
  const compareA = allArtifacts.find((item) => item.artifact_id === compare.a) ?? null;
  const compareB = allArtifacts.find((item) => item.artifact_id === compare.b) ?? null;
  const baseExperiment = experimentCatalog.find((item) => item.value === experimentMode) ?? experimentCatalog[0];
  const baseOperatorConfig = operatorCatalog.find((item) => item.value === operator) ?? operatorCatalog[0];
  const operatorSpecRows = workbenchState?.operatorSpecs ?? operatorSpecs.data ?? [];
  const modeAtlasRows = workbenchState?.modeAtlas ?? modeAtlas.data ?? [];
  const healthData = workbenchState?.health ?? health.data;
  const readinessChecks = readiness.data?.checks ?? readinessChecksFromHealth(healthData);
  const specMap = useMemo(() => new Map(operatorSpecRows.map((spec) => [spec.name, spec])), [operatorSpecRows]);
  const activeGenerateSpec = specMap.get(generationMode);
  const sameEncodeSpec = specMap.get("latent.encode");
  const sameDecodeSpec = specMap.get("latent.decode");
  const activeOperatorSpec = specMap.get(operator);
  const activeExperimentSpec = specMap.get(baseExperiment.value);
  const activeGenerationBase = generationCatalog.find((item) => item.value === generationMode) ?? generationCatalog[0];
  const activeGenerationConfig = useMemo(
    () => withOperatorSpecFields(activeGenerationBase, filteredOperatorSpec(activeGenerateSpec, generationControlKeys(generationMode))),
    [activeGenerateSpec, activeGenerationBase, generationMode],
  );
  const activeSameConfig = useMemo(
    () => withOperatorSpecFields(sameConfig, filteredOperatorSpec(sameEncodeSpec, sameEncodeControlKeys)),
    [sameEncodeSpec],
  );
  const activeOperatorConfig = useMemo(() => withOperatorSpecFields(baseOperatorConfig, activeOperatorSpec), [baseOperatorConfig, activeOperatorSpec]);
  const activeExperiment = useMemo(() => withOperatorSpecFields(baseExperiment, activeExperimentSpec), [baseExperiment, activeExperimentSpec]);
  const activeOperatorPresets = useMemo(() => operatorPresets.filter((preset) => preset.operator === operator), [operatorPresets, operator]);
  const selectedOperatorPreset = useMemo(
    () => operatorPresets.find((preset) => preset.id === selectedOperatorPresetId) ?? null,
    [operatorPresets, selectedOperatorPresetId],
  );
  const operatorPresetDiffs = useMemo(
    () => selectedOperatorPreset ? operatorPresetDiffRows(selectedOperatorPreset, operatorForm, donorArtifactId) : [],
    [donorArtifactId, operatorForm, selectedOperatorPreset],
  );
  const generationNeedsSource = generationMode !== "generate.text_to_audio";
  const generationSource = selectedArtifact?.kind === "audio" ? selectedArtifact : null;
  const canGenerate = generationReady({ form: generationForm, needsSource: generationNeedsSource, sourceArtifact: generationSource });
  const operatorNeedsDonor = operatorUsesDonor(activeOperatorConfig, operatorForm);
  const canRunOperator = operatorReady(activeOperatorConfig, operatorForm, selectedArtifact, donorArtifactId);
  const generateJob = activeJobForOperator(runningJobs, generationMode);
  const encodeJob = activeJobForOperator(runningJobs, "latent.encode");
  const decodeJob = activeJobForOperator(runningJobs, "latent.decode");
  const operatorJob = activeJobForOperator(runningJobs, operator);
  const experimentJob = activeJobForOperator(runningJobs, activeExperiment.value);
  const runningJobIds = runningJobs.map((job) => job.job_id).sort().join("|");
  const liveEventing = Object.values(liveJobsById).some((job) => isJobActive(job));

  useEffect(() => {
    if (!selectedArtifactId && visibleArtifacts[0]) {
      selectArtifact(visibleArtifacts[0].artifact_id);
    }
  }, [selectArtifact, selectedArtifactId, visibleArtifacts]);

  useEffect(() => {
    if (!allSessions.length) return;
    if (sessionId && allSessions.some((session) => session.session_id === sessionId)) return;
    const latestActive = allSessions.find((session) => session.status === "active") ?? allSessions[0];
    setSession(latestActive.session_id, latestActive.created_at);
  }, [sessionId, allSessions, setSession]);

  useEffect(() => {
    setOperatorForm((current) => fillMissingFieldDefaults(activeOperatorConfig, current));
  }, [activeOperatorConfig]);

  useEffect(() => {
    setGenerationForm((current) => fillMissingFieldDefaults(activeGenerationConfig, current));
  }, [activeGenerationConfig]);

  useEffect(() => {
    setSameForm((current) => fillMissingFieldDefaults(activeSameConfig, current));
  }, [activeSameConfig]);

  useEffect(() => {
    if (!selectedOperatorPresetId) return;
    if (activeOperatorPresets.some((preset) => preset.id === selectedOperatorPresetId)) return;
    setSelectedOperatorPresetId("");
    setOperatorPresetName("");
  }, [activeOperatorPresets, selectedOperatorPresetId]);

  useEffect(() => {
    setExperimentForm((current) => fillMissingFieldDefaults(activeExperiment, current));
  }, [activeExperiment]);

  useEffect(() => {
    if (!sessionResultFamilies.length) {
      setInspectedFamilyId(null);
      return;
    }
    if (!inspectedFamilyId || !sessionResultFamilies.some((family) => family.familyId === inspectedFamilyId)) {
      setInspectedFamilyId(sessionResultFamilies[0].familyId);
    }
  }, [inspectedFamilyId, sessionResultFamilies]);

  const invalidate = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["workbench"] }),
      queryClient.invalidateQueries({ queryKey: ["artifacts", apiBase] }),
      queryClient.invalidateQueries({ queryKey: ["jobs", apiBase] }),
      queryClient.invalidateQueries({ queryKey: ["health", apiBase] }),
      queryClient.invalidateQueries({ queryKey: ["sessions", apiBase] }),
    ]);
  };

  useEffect(() => {
    if (!runningJobIds) return;
    const landTerminalJob = (job: JobRecord) => {
      const artifactId = landingArtifactId(job);
      if (artifactId) selectArtifact(artifactId);
    };
    if (controlPlane) {
      const subscriptions = runningJobIds.split("|").map((jobId) =>
        controlPlane.jobs.events.subscribe(
          { jobId },
          {
            onData: (event) => {
              const job = jobFromJobEvent(event);
              if (!job) return;
              setLiveJobsById((current) => ({ ...current, [job.job_id]: job }));
              if (!isJobActive(job)) {
                landTerminalJob(job);
                void invalidate();
              }
            },
            onError: () => {
              void invalidate();
            },
          },
        ),
      );
      return () => {
        subscriptions.forEach((subscription) => subscription.unsubscribe());
      };
    }
    const sockets = runningJobIds.split("|").map((jobId) => {
      const socket = new WebSocket(api.jobEventsUrl(jobId));
      socket.onmessage = (event) => {
        const job = parseJobEvent(event.data);
        if (!job) return;
        setLiveJobsById((current) => ({ ...current, [job.job_id]: job }));
        if (!isJobActive(job)) {
          landTerminalJob(job);
          void invalidate();
        }
      };
      return socket;
    });
    return () => {
      sockets.forEach((socket) => socket.close());
    };
  }, [api, controlPlane, runningJobIds, selectArtifact]);

  const importAudio = useMutation({
    mutationFn: (file: File) => api.importAudio(file, file.name, activeSessionId),
    onSuccess: async (artifact) => {
      selectArtifact(artifact.artifact_id);
      await invalidate();
    },
  });

  const createSession = useMutation({
    mutationFn: () => api.createSession({ name: `Session ${new Date().toLocaleString()}` }),
    onSuccess: async (session) => {
      setSession(session.session_id, session.created_at);
      await invalidate();
    },
  });

  const archiveSession = useMutation({
    mutationFn: async (session: SessionRecord) => {
      await api.updateSession(session.session_id, { status: "archived" });
      return api.createSession({ name: `Session ${new Date().toLocaleString()}` });
    },
    onSuccess: async (session) => {
      setSession(session.session_id, session.created_at);
      selectArtifact(null);
      await invalidate();
    },
  });

  const recoverArtifact = useMutation({
    mutationFn: (artifact: ArtifactRecord) => {
      if (!activeSessionId) {
        throw new Error("No active session available for recovery");
      }
      return api.annotateArtifact(
        artifact.artifact_id,
        artifactRecoveryPayload({
          artifact,
          targetSessionId: activeSessionId,
          source: "archive_tray",
        }),
      );
    },
    onSuccess: async (artifact) => {
      selectArtifact(artifact.artifact_id);
      await invalidate();
    },
  });

  const archiveArtifact = useMutation({
    mutationFn: ({ artifact, source }: { artifact: ArtifactRecord; source: string }) =>
      api.annotateArtifact(
        artifact.artifact_id,
        artifactArchivePayload({
          artifact,
          source,
        }),
      ),
    onSuccess: async (artifact) => {
      if (selectedArtifactId === artifact.artifact_id) selectArtifact(null);
      await invalidate();
    },
  });

  const cancelJobMutation = useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onSuccess: invalidate,
  });

  const retryJobMutation = useMutation({
    mutationFn: (jobId: string) => api.retryJob(jobId),
    onSuccess: invalidate,
  });

  const replayRecipeMutation = useMutation({
    mutationFn: (recipeId: string) => api.replayRecipe(recipeId),
    onSuccess: invalidate,
  });

  const forkRecipeMutation = useMutation({
    mutationFn: ({ recipeId, payload }: { recipeId: string; payload: RecipeForkPayload }) => api.forkRecipe(recipeId, payload),
    onSuccess: async () => {
      setForkTarget(null);
      await invalidate();
    },
  });

  const annotateArtifact = useMutation({
    mutationFn: ({ artifactId, payload }: { artifactId: string; payload: ArtifactAnnotationPayload }) =>
      api.annotateArtifact(artifactId, payload),
    onSuccess: async () => {
      await invalidate();
    },
  });

  const generate = useMutation({
    mutationFn: () => {
      const payload = buildGenerationPayload({
        mode: generationMode,
        form: generationForm,
        sourceArtifact: generationSource,
        sessionId: activeSessionId,
      });
      if (generationMode === "generate.audio_to_audio") {
        return api.generateAudioToAudio(payload as AudioToAudioPayload);
      }
      if (generationMode === "generate.inpaint") {
        return api.generateInpaint(payload as InpaintPayload);
      }
      return api.generateText(payload as GenerateTextPayload);
    },
    onSuccess: invalidate,
  });

  const generatePromptCandidate = useMutation({
    mutationFn: (candidate: PromptCandidateGenerationRequest) => {
      const payload = buildGenerationPayload({
        mode: "generate.text_to_audio",
        form: generationForm,
        promptOverride: candidate.prompt,
        sourceArtifactId: candidate.bundleArtifactId,
        notes: `Prompt candidate ${candidate.rank ? `#${candidate.rank} ` : ""}from prompt-search bundle ${candidate.bundleArtifactId}`,
        metadata: {
          generation_origin: "prompt_search_candidate",
          prompt_search_bundle_id: candidate.bundleArtifactId,
          prompt_candidate_rank: candidate.rank ?? null,
          prompt_candidate_score: primitiveMetadataValue(candidate.score),
          prompt_candidate_source: candidate.source ?? null,
          prompt_search_scorer: candidate.scorer ?? null,
          prompt_search_mode: candidate.searchMode ?? null,
          prompt_search_model: candidate.searchModel ?? null,
          prompt_search_duration_seconds: candidate.searchDurationSeconds ?? null,
        },
        sessionId: activeSessionId,
      });
      return api.generateText(payload as GenerateTextPayload);
    },
    onSuccess: invalidate,
  });

  const encode = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.encodeLatent(buildLatentEncodePayload({ form: sameForm, artifact, sessionId: activeSessionId })),
    onSuccess: invalidate,
  });

  const decode = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.decodeLatent(buildLatentDecodePayload({ form: sameForm, artifact, sessionId: activeSessionId })),
    onSuccess: invalidate,
  });

  const runOperator = useMutation({
    mutationFn: (artifact: ArtifactRecord) =>
      api.runOperator({
        operator,
        backend: operatorBackend(operatorForm, activeOperatorConfig.defaultBackend),
        inputs: {
          source: artifact.artifact_id,
          ...(donorArtifactId && operatorUsesDonor(activeOperatorConfig, operatorForm) ? { donor: donorArtifactId } : {}),
        },
        params: buildOperatorParams(activeOperatorConfig, operatorForm),
        seed: operatorSeed(operatorForm, generationSeedFallback(generationForm)),
        session_id: activeSessionId,
      }),
    onSuccess: invalidate,
  });

  const runExperiment = useMutation({
    mutationFn: () =>
      api.runExperiment(
        buildExperimentPayload({
          config: activeExperiment,
          form: experimentForm,
          selectedArtifact,
          sessionId: activeSessionId,
        }),
      ),
    onSuccess: invalidate,
  });

  const canRunExperiment = experimentReady(activeExperiment, experimentForm, selectedArtifact);
  const gestureOptions = useMemo(() => buildGestureOptions(selectedArtifact), [selectedArtifact]);
  const activeGesture = gestureOptions.find((gesture) => gesture.id === activeGestureId) ?? gestureOptions[0] ?? buildGestureOptions(null)[0]!;
  const gestureAction = describeGestureAction({
    gesture: activeGesture,
    selectedArtifact,
    generationMode,
    generationForm,
    generationNeedsSource,
    generationSource,
    generationReady: canGenerate,
    generationBusy: generate.isPending || Boolean(generateJob),
    encodeReady: Boolean(activeGesture.available && selectedArtifact?.kind === "audio"),
    encodeBusy: encode.isPending || Boolean(encodeJob),
    decodeReady: Boolean(activeGesture.available && selectedArtifact?.kind === "latent"),
    decodeBusy: decode.isPending || Boolean(decodeJob),
    operatorMode: operator,
    operatorLabel: activeOperatorConfig.label,
    operatorReady: canRunOperator,
    operatorBusy: runOperator.isPending || Boolean(operatorJob),
    operatorNeedsDonor,
    donorArtifactId,
    donorArtifactLabel: latentArtifacts.find((artifact) => artifact.artifact_id === donorArtifactId)?.label ?? null,
    experimentLabel: activeExperiment.label,
    experimentReady: canRunExperiment,
    experimentBusy: runExperiment.isPending || Boolean(experimentJob),
    rememberBusy: archiveArtifact.isPending,
  });
  const currentNextActions = useMemo(
    () => nextActionsForArtifact(selectedArtifact, { donorLatents: latentArtifacts }),
    [latentArtifacts, selectedArtifact],
  );

  const updateOperatorPresets = (next: OperatorPreset[]) => {
    persistOperatorPresets(next);
    setOperatorPresets(next);
  };

  const saveOperatorPreset = () => {
    const existing = operatorPresets.find((preset) => preset.id === selectedOperatorPresetId);
    const preset = createOperatorPreset({
      id: existing?.id,
      createdAt: existing?.createdAt,
      name: operatorPresetName.trim() || existing?.name || `${activeOperatorConfig.label} sketch`,
      operator,
      form: operatorForm,
      donorArtifactId: donorArtifactId || null,
    });
    updateOperatorPresets(upsertOperatorPreset(operatorPresets, preset));
    setSelectedOperatorPresetId(preset.id);
    setOperatorPresetName(preset.name);
  };

  const applyOperatorPreset = (presetId: string) => {
    const preset = operatorPresets.find((item) => item.id === presetId);
    if (!preset || !isLatentOperatorMode(preset.operator)) return;
    setOperator(preset.operator);
    setOperatorForm({
      ...defaultOperatorForm(preset.operator),
      ...preset.form,
    });
    setDonorArtifactId(preset.donorArtifactId ?? "");
    setSelectedOperatorPresetId(preset.id);
    setOperatorPresetName(preset.name);
  };

  const removeOperatorPreset = () => {
    if (!selectedOperatorPresetId) return;
    updateOperatorPresets(deleteOperatorPreset(operatorPresets, selectedOperatorPresetId));
    setSelectedOperatorPresetId("");
    setOperatorPresetName("");
  };

  const selectOperatorMode = (mode: LatentOperatorMode) => {
    selectWorkbenchOperatorMode(mode);
    setSelectedOperatorPresetId("");
    setOperatorPresetName("");
  };

  const selectGesture = (gestureId: GestureId) => {
    const gesture = gestureById(gestureId);
    selectWorkbenchGesture(gestureId);
    if (gesture.defaultOperatorMode) {
      setSelectedOperatorPresetId("");
      setOperatorPresetName("");
    }
  };

  const applyNextAction = (action: ProductNextAction) => {
    applyWorkbenchNextAction(action);
    if (action.operatorMode || (action.gestureId && gestureById(action.gestureId).defaultOperatorMode)) {
      setSelectedOperatorPresetId("");
      setOperatorPresetName("");
    }
  };

  const useArtifactAsDonor = (artifactId: string) => {
    useWorkbenchArtifactAsDonor(artifactId);
    setSelectedOperatorPresetId("");
    setOperatorPresetName("");
  };

  const applyMemoryAction = (artifact: ArtifactRecord, action: MemoryReuseAction) => {
    if (!action.available) return;
    if (action.intent === "recover") {
      recoverArtifact.mutate(artifact);
      return;
    }
    const handled = gestureWorkbench.applyMemoryAction(artifact, action);
    if (handled && action.intent === "donor") {
      setSelectedOperatorPresetId("");
      setOperatorPresetName("");
    }
  };

  const branchFromArtifact = (artifact: ArtifactRecord) => {
    const recipe = artifact.recipe_id ? allJobs.find((job) => job.recipe.recipe_id === artifact.recipe_id)?.recipe ?? null : null;
    if (recipe) setForkTarget(recipe);
  };

  const continueFromArtifact = (artifact: ArtifactRecord) => {
    selectArtifact(artifact.artifact_id);
    if (artifact.kind === "audio") setCompare("b", artifact.artifact_id);
    selectGesture(artifact.kind === "latent" ? "morph" : "continue");
  };

  const runPromptCandidate = (candidate: PromptCandidateGenerationRequest) => {
    usePromptCandidate(candidate.prompt);
    generatePromptCandidate.mutate(candidate);
  };

  const renderGestureTune = () => {
    if (activeGesture.tuneSource === "generation") {
      return (
        <>
          <div className="gesture-mode-grid">
            <label>
              Generation move
              <select value={generationMode} onChange={(event) => setGenerationMode(event.target.value as GenerationMode)}>
                {generationModes.map((mode) => (
                  <option key={mode.value} value={mode.value}>
                    {mode.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {generationNeedsSource ? (
            <SourcePicker
              className="control-cell source-audio-cell"
              label="Source sound"
              description="Choose the audio source that this gesture will continue or inpaint."
              value={generationSource?.artifact_id ?? ""}
              sources={sourceRows}
              selectedArtifact={selectedArtifact}
              artifactKinds={["audio"]}
              fieldKey="source_artifact_id"
              valueMode="artifact-id"
              getArtifactPath={artifactPathForField}
              onChange={(artifactId) => {
                selectArtifact(artifactId);
                setCompare("b", artifactId);
              }}
              emptyLabel="Choose sound"
            />
          ) : null}
          <RecipeFields
            config={withTuneFieldGroups(activeGenerationConfig, { gestureId: activeGesture.id, generationMode })}
            form={generationForm}
            artifacts={allArtifacts}
            sources={sourceRows}
            selectedArtifact={selectedArtifact}
            onChange={setGenerationField}
            getArtifactLabel={artifactName}
            getArtifactPath={artifactPathForField}
          />
          {generationNeedsSource && !generationSource ? <div className="quiet-panel compact">Select an audio sound for this gesture.</div> : null}
        </>
      );
    }

    if (activeGesture.tuneSource === "same") {
      return (
        <RecipeFields
          config={withTuneFieldGroups(activeSameConfig, { gestureId: activeGesture.id })}
          form={sameForm}
          artifacts={allArtifacts}
          sources={sourceRows}
          selectedArtifact={selectedArtifact}
          onChange={setSameField}
          getArtifactLabel={artifactName}
          getArtifactPath={artifactPathForField}
        />
      );
    }

    if (activeGesture.tuneSource === "operator") {
      return (
        <>
          <div className="recipe-mode-grid operator-mode-grid">
            <label>
              Latent move
              <select value={operator} onChange={(event) => selectOperatorMode(event.target.value as LatentOperatorMode)}>
                {operatorModes.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <span className={`recipe-chip ${activeOperatorConfig.maturity}`}>{activeOperatorConfig.family}</span>
            <span className={`recipe-chip ${activeOperatorConfig.maturity}`}>{activeOperatorConfig.maturity}</span>
          </div>
          <OperatorPresetRack
            presets={activeOperatorPresets}
            selectedPreset={selectedOperatorPreset}
            selectedPresetId={selectedOperatorPresetId}
            presetName={operatorPresetName}
            diffRows={operatorPresetDiffs}
            onSelectPreset={applyOperatorPreset}
            onChangePresetName={setOperatorPresetName}
            onSavePreset={saveOperatorPreset}
            onResetPreset={() => selectedOperatorPreset && applyOperatorPreset(selectedOperatorPreset.id)}
            onDeletePreset={removeOperatorPreset}
          />
          {operatorNeedsDonor ? (
            <div className="control-cell donor-cell">
              <SourcePicker
                label="Texture donor"
                description="Choose a recovered or encoded latent as the donor texture."
                value={donorArtifactId}
                sources={sourceRows.filter((source) => source.artifactId !== selectedArtifact?.artifact_id)}
                selectedArtifact={null}
                artifactKinds={["latent"]}
                fieldKey="donor"
                valueMode="artifact-id"
                getArtifactPath={artifactPathForField}
                onChange={setDonorArtifactId}
                emptyLabel="Choose latent donor"
                showSelectedButton={false}
              />
              <small>
                {latentArtifacts.filter((artifact) => artifact.artifact_id !== selectedArtifact?.artifact_id).length
                  ? "Use a recovered or encoded latent as the donor texture."
                  : "Encode or recover another latent before borrowing texture."}
              </small>
            </div>
          ) : null}
          <RecipeFields
            config={withTuneFieldGroups(activeOperatorConfig, { gestureId: activeGesture.id, operatorMode: operator })}
            form={operatorForm}
            artifacts={allArtifacts}
            sources={sourceRows}
            selectedArtifact={selectedArtifact}
            onChange={setOperatorField}
            getArtifactPath={artifactPathForField}
            getArtifactLabel={artifactName}
            operatorMode={operator}
          />
        </>
      );
    }

    if (activeGesture.tuneSource === "experiment") {
      return (
        <>
          <div className="recipe-mode-grid">
            <label>
              Experiment
              <select value={experimentMode} onChange={(event) => selectExperimentMode(event.target.value as ExperimentMode)}>
                {experimentModes.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <span className={`recipe-chip ${activeExperiment.maturity}`}>{activeExperiment.family}</span>
            <span className={`recipe-chip ${activeExperiment.maturity}`}>{activeExperiment.maturity}</span>
          </div>
          {activeExperiment.value === "experiment.prompt_search" ? (
            <PromptSearchPresetRack
              presets={promptSearchPresets}
              vocabularySets={promptSearchVocabularySets}
              axisSets={promptSearchAxisSets}
              historyRows={promptSearchHistoryRows(allArtifacts)}
              scorer={experimentForm.scorer}
              onApply={(presetId) => setExperimentForm((current) => applyPromptSearchPreset(current, presetId))}
              onApplyVocabulary={(setId) => setExperimentForm((current) => applyPromptSearchVocabularySet(current, setId))}
              onApplyAxis={(setId) => setExperimentForm((current) => applyPromptSearchAxisSet(current, setId))}
              onUseHistoryPrompt={(prompt) => setExperimentForm((current) => ({ ...current, seed_prompt: prompt }))}
            />
          ) : null}
          <RecipeFields
            config={withTuneFieldGroups(activeExperiment, { gestureId: activeGesture.id, experimentMode })}
            form={experimentForm}
            artifacts={allArtifacts}
            sources={sourceRows}
            selectedArtifact={selectedArtifact}
            onChange={setExperimentField}
            getArtifactPath={artifactPathForField}
            getArtifactLabel={artifactName}
          />
        </>
      );
    }

    return (
      <div className="remember-gesture-panel">
        <strong>{selectedArtifact ? artifactName(selectedArtifact) : "No current material"}</strong>
        <p>Remember saves the current material into Memory so it can be recovered, used as a source, or kept out of the active session.</p>
      </div>
    );
  };

  const renderGestureInspect = () => {
    if (activeGesture.tuneSource === "generation") {
      return <SpecCoverage spec={activeGenerateSpec} controlledKeys={generationControlKeys(generationMode)} />;
    }
    if (activeGesture.tuneSource === "same") {
      return <SpecCoveragePair specs={[sameEncodeSpec, sameDecodeSpec]} controlledKeys={[sameEncodeControlKeys, sameDecodeControlKeys]} />;
    }
    if (activeGesture.tuneSource === "operator") {
      return <SpecCoverage spec={activeOperatorSpec} controlledKeys={fieldKeys(activeOperatorConfig)} />;
    }
    if (activeGesture.tuneSource === "experiment") {
      return (
        <>
          <SpecCoverage spec={activeExperimentSpec} controlledKeys={fieldKeys(activeExperiment)} />
          <details className="dev-drawer nested">
            <summary>Developer mode map</summary>
            <ModeAtlas modes={modeAtlasRows} activeOperator={activeExperiment.value} />
          </details>
        </>
      );
    }
    return null;
  };

  const renderGestureAction = () => {
    if (activeGesture.tuneSource === "generation") {
      return (
        <>
          <button className="primary-action" onClick={() => generate.mutate()} disabled={!gestureAction.ready} title={gestureAction.disabledReason ?? gestureAction.intentCopy}>
            {generate.isPending || generateJob ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
            {gestureAction.label}
          </button>
          <InlineJobStatus
            job={generateJob}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
        </>
      );
    }

    if (activeGesture.id === "encode") {
      return (
        <>
          <button disabled={!gestureAction.ready} title={gestureAction.disabledReason ?? gestureAction.intentCopy} onClick={() => selectedArtifact && encode.mutate(selectedArtifact)}>
            {encode.isPending || encodeJob ? <LoaderCircle className="spin" size={17} /> : <Box size={17} />}
            {gestureAction.label}
          </button>
          <InlineJobStatus
            job={encodeJob}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
        </>
      );
    }

    if (activeGesture.id === "decode") {
      return (
        <>
          <button disabled={!gestureAction.ready} title={gestureAction.disabledReason ?? gestureAction.intentCopy} onClick={() => selectedArtifact && decode.mutate(selectedArtifact)}>
            {decode.isPending || decodeJob ? <LoaderCircle className="spin" size={17} /> : <Waves size={17} />}
            {gestureAction.label}
          </button>
          <InlineJobStatus
            job={decodeJob}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
        </>
      );
    }

    if (activeGesture.tuneSource === "operator") {
      return (
        <>
          <button className="primary-action" disabled={!gestureAction.ready} title={gestureAction.disabledReason ?? gestureAction.intentCopy} onClick={() => selectedArtifact && runOperator.mutate(selectedArtifact)}>
            {runOperator.isPending || operatorJob ? <LoaderCircle className="spin" size={18} /> : <GitFork size={17} />}
            {gestureAction.label}
          </button>
          <InlineJobStatus
            job={operatorJob}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
        </>
      );
    }

    if (activeGesture.tuneSource === "experiment") {
      return (
        <>
          <button className="primary-action" disabled={!gestureAction.ready} title={gestureAction.disabledReason ?? gestureAction.intentCopy} onClick={() => runExperiment.mutate()}>
            {runExperiment.isPending || experimentJob ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
            {gestureAction.label}
          </button>
          <InlineJobStatus
            job={experimentJob}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
        </>
      );
    }

    return (
      <button
        className="primary-action"
        disabled={!gestureAction.ready}
        title={gestureAction.disabledReason ?? gestureAction.intentCopy}
        onClick={() => selectedArtifact && archiveArtifact.mutate({ artifact: selectedArtifact, source: "remember_gesture" })}
      >
        {archiveArtifact.isPending ? <LoaderCircle className="spin" size={18} /> : <Box size={17} />}
        {gestureAction.label}
      </button>
    );
  };

  return (
    <main className="app-shell">
      <header className="top-strip">
        <div className="brand-mark">
          <img src={modelImage} alt="" />
          <div>
            <strong>SA3 Native Lab</strong>
            <span>{healthData?.artifact_root ?? ".sa3_lab"}</span>
          </div>
        </div>
        <div className="instrument-question">What do you want to do with this sound next?</div>
        <details className="settings-panel">
          <summary>
            <Settings size={17} />
            Settings
          </summary>
          <div className="settings-body">
            <div className="api-field">
              <label htmlFor="api-base">API</label>
              <input id="api-base" value={apiBase} onChange={(event) => setApiBase(event.target.value)} />
            </div>
            <BackendPills backends={healthData?.backends ?? []} />
            <ReadinessPanel checks={readinessChecks} />
          </div>
        </details>
      </header>

      <section className="bench-grid">
        <aside className="source-rail">
          <div className="rail-head">
            <div>
              <span className="eyebrow">Sources</span>
              <strong>{sourceRows.length} source options</strong>
            </div>
            <label className="icon-button" title="Import audio">
              <Upload size={18} />
              <input
                type="file"
                accept="audio/*"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) importAudio.mutate(file);
                  event.currentTarget.value = "";
                }}
              />
            </label>
          </div>
          <SourceShelf
            sources={sourceRows}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            onSelect={selectArtifact}
            onUseAction={(source, action) => applyMemoryAction(source.artifact, action)}
          />
        </aside>

        <section className={`operator-surface ${selectedArtifact ? "has-selection" : "idle"}`}>
          <div className="surface-head">
            <div>
              <span className="eyebrow">Current Sound</span>
              <h1>{selectedArtifact ? artifactName(selectedArtifact) : "No sound selected"}</h1>
            </div>
            {selectedArtifact ? <ArtifactBadge artifact={selectedArtifact} /> : null}
          </div>

          <GestureStrip gestures={gestureOptions} activeId={activeGesture.id} onSelect={selectGesture} />

          <Specimen
            artifact={selectedArtifact}
            artifacts={allArtifacts}
            jobs={allJobs}
            families={resultFamilies}
            compare={compare}
            apiBase={apiBase}
            annotating={annotateArtifact.isPending}
            activeSessionId={activeSessionId}
            archivingArtifactId={archiveArtifact.isPending ? archiveArtifact.variables?.artifact.artifact_id ?? null : null}
            onAnnotate={(artifactId, payload) => annotateArtifact.mutate({ artifactId, payload })}
            onCompare={setCompare}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onForkRecipe={setForkTarget}
            onArchiveArtifact={(artifact) => archiveArtifact.mutate({ artifact, source: "specimen" })}
            onSelectArtifact={selectArtifact}
            onUseAsDonor={useArtifactAsDonor}
            onUseInRecipe={useBundleInRecipe}
            onUsePrompt={usePromptCandidate}
            onGeneratePrompt={runPromptCandidate}
            getArtifactPath={artifactPathForField}
          />
          <NextActionsPanel actions={currentNextActions} onAction={applyNextAction} />
          <details className="dev-drawer activity-drawer">
            <summary>Inspect activity</summary>
            <RunMonitor
              runningJobs={runningJobs}
              latestJob={latestJobs[0] ?? null}
              eventing={liveEventing}
              onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
              onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
            />
          </details>

          <div className="gesture-workbench">
            <TuneDrawer gesture={activeGesture} inspect={renderGestureInspect()} action={renderGestureAction()} actionDescriptor={gestureAction}>
              {renderGestureTune()}
            </TuneDrawer>
          </div>
        </section>

        <aside className="result-rail">
          <div className="rail-head">
            <div>
              <span className="eyebrow">Takes / Branches</span>
              <strong>{sessionResultFamilies.length} branches</strong>
            </div>
            <Activity size={19} />
          </div>
          <AuditionStackPanel
            artifacts={sessionArtifacts}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            activeSessionId={activeSessionId}
            archivingArtifactId={archiveArtifact.isPending ? archiveArtifact.variables?.artifact.artifact_id ?? null : null}
            onSelect={selectArtifact}
            onCompare={setCompare}
            onAnnotate={(artifactId, payload) => annotateArtifact.mutate({ artifactId, payload })}
            onRemember={(artifact) => archiveArtifact.mutate({ artifact, source: "take_strip" })}
            onContinue={continueFromArtifact}
            onBranch={branchFromArtifact}
          />
          <PendingTakesPanel
            takes={pendingTakes}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
          {forkTarget ? (
            <ForkRecipePanel
              recipe={forkTarget}
              submitting={forkRecipeMutation.isPending}
              onClose={() => setForkTarget(null)}
              onSubmit={(payload) =>
                forkRecipeMutation.mutate({
                  recipeId: forkTarget.recipe_id,
                  payload: { ...payload, session_id: activeSessionId },
                })
              }
            />
          ) : null}
          <ResultFamilyPanel
            families={sessionResultFamilies}
            artifacts={allArtifacts}
            selectedId={selectedArtifact?.artifact_id ?? null}
            inspectedFamilyId={inspectedFamily?.familyId ?? null}
            onSelect={selectArtifact}
            onInspectFamily={setInspectedFamilyId}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onForkRecipe={setForkTarget}
          />
          <FamilyDetailPanel
            family={inspectedFamily}
            families={sessionResultFamilies}
            artifacts={allArtifacts}
            jobs={allJobs}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            activeSessionId={activeSessionId}
            archivingArtifactId={archiveArtifact.isPending ? archiveArtifact.variables?.artifact.artifact_id ?? null : null}
            onSelect={selectArtifact}
            onInspectFamily={setInspectedFamilyId}
            onCompare={setCompare}
            onAnnotate={(artifactId, payload) => annotateArtifact.mutate({ artifactId, payload })}
            onReplayRecipe={(recipeId) => replayRecipeMutation.mutate(recipeId)}
            onForkRecipe={setForkTarget}
            onContinueArtifact={continueFromArtifact}
            onBranchArtifact={branchFromArtifact}
            onArchiveArtifact={(artifact) => archiveArtifact.mutate({ artifact, source: "family_detail" })}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
          <SessionTray
            artifacts={sessionArtifacts}
            archivedArtifacts={archiveArtifacts}
            jobs={sessionJobs}
            archivedJobs={archiveJobs}
            families={resultFamilies}
            runningJobs={runningJobs}
            selectedId={selectedArtifact?.artifact_id ?? null}
            apiBase={apiBase}
            activeSessionId={activeSessionId}
            session={activeSession}
            sessionStartedAt={activeSession?.created_at ?? sessionStartedAt}
            creatingSession={createSession.isPending}
            archivingSession={archiveSession.isPending}
            recoveringArtifactId={recoverArtifact.isPending ? recoverArtifact.variables?.artifact_id ?? null : null}
            archivingArtifactId={archiveArtifact.isPending ? archiveArtifact.variables?.artifact.artifact_id ?? null : null}
            onSelect={selectArtifact}
            onStartSession={() => createSession.mutate()}
            onArchiveSession={() => activeSession && archiveSession.mutate(activeSession)}
            onRecoverArtifact={(artifact) => recoverArtifact.mutate(artifact)}
            onArchiveArtifact={(artifact) => archiveArtifact.mutate({ artifact, source: "session_tray" })}
            onUseMemoryAction={applyMemoryAction}
            onCancelJob={(job) => cancelJobMutation.mutate(job.job_id)}
            onRetryJob={(job) => retryJobMutation.mutate(job.job_id)}
          />
          <ComparePanel a={compareA} b={compareB} apiBase={apiBase} />
          <details className="dev-drawer">
            <summary>Material counts</summary>
            <div className="mini-counts">
              <span><FileAudio size={15} /> {audioArtifacts.length}</span>
              <span><Braces size={15} /> {latentArtifacts.length}</span>
              <span><Box size={15} /> {bundleArtifacts.length}</span>
            </div>
          </details>
        </aside>
      </section>
    </main>
  );
}

function findActiveSession(sessions: SessionRecord[], sessionId: string | null): SessionRecord | null {
  if (sessionId) {
    return sessions.find((session) => session.session_id === sessionId) ?? null;
  }
  return sessions.find((session) => session.status === "active") ?? null;
}

function artifactPathForField(artifact: ArtifactRecord, fieldKey: string) {
  const rawScriptOutput = artifact.metadata?.script_output_path;
  const scriptOutput = typeof rawScriptOutput === "string" ? rawScriptOutput : "";
  const source = scriptOutput || artifact.path;
  if (!scriptOutput) return source;
  if (fieldKey === "profile_path") return `${source}/profile.npz`;
  if (fieldKey === "direction_path") {
    return artifact.metadata.operator === "experiment.audio_style_vectors" ? `${source}/frame_direction.npz` : `${source}/direction.npz`;
  }
  if (fieldKey === "target_memory_path" || fieldKey === "reference_memory_path") return `${source}/memory`;
  return source;
}

function readinessChecksFromHealth(health: HealthResponse | undefined): ReadinessCheck[] {
  if (!health) return [];
  return [
    { name: "artifact-root", status: "ok", message: health.artifact_root },
    ...health.backends.map((backend) => ({
      name: `backend:${backend.backend}`,
      status: backend.available ? "ok" : "warn",
      message: backend.message ?? backend.device ?? backend.backend,
    })),
  ];
}
