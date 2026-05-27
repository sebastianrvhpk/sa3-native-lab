import { create } from "zustand";

import { CONFIGURED_API_BASE, DEFAULT_API_BASE } from "./api";

type CompareSlot = "a" | "b";

interface BenchState {
  apiBase: string;
  selectedArtifactId: string | null;
  sessionId: string | null;
  sessionStartedAt: string;
  compare: Record<CompareSlot, string | null>;
  setApiBase: (value: string) => void;
  selectArtifact: (artifactId: string | null) => void;
  setSession: (sessionId: string | null, startedAt: string) => void;
  setCompare: (slot: CompareSlot, artifactId: string | null) => void;
}

export const useBenchStore = create<BenchState>((set) => ({
  apiBase: CONFIGURED_API_BASE ?? localStorage.getItem("sa3_api_base") ?? DEFAULT_API_BASE,
  selectedArtifactId: null,
  sessionId: localStorage.getItem("sa3_session_id"),
  sessionStartedAt: localStorage.getItem("sa3_session_started_at") ?? "1970-01-01T00:00:00.000Z",
  compare: { a: null, b: null },
  setApiBase: (value) => {
    localStorage.setItem("sa3_api_base", value);
    set({ apiBase: value });
  },
  selectArtifact: (artifactId) => set({ selectedArtifactId: artifactId }),
  setSession: (sessionId, startedAt) => {
    if (sessionId) {
      localStorage.setItem("sa3_session_id", sessionId);
    } else {
      localStorage.removeItem("sa3_session_id");
    }
    localStorage.setItem("sa3_session_started_at", startedAt);
    set({ sessionId, sessionStartedAt: startedAt, selectedArtifactId: null, compare: { a: null, b: null } });
  },
  setCompare: (slot, artifactId) =>
    set((state) => ({ compare: { ...state.compare, [slot]: artifactId } })),
}));
