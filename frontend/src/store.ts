import { create } from "zustand";

type CompareSlot = "a" | "b";

interface BenchState {
  apiBase: string;
  selectedArtifactId: string | null;
  sessionStartedAt: string;
  compare: Record<CompareSlot, string | null>;
  setApiBase: (value: string) => void;
  selectArtifact: (artifactId: string | null) => void;
  startSession: () => void;
  setCompare: (slot: CompareSlot, artifactId: string | null) => void;
}

export const useBenchStore = create<BenchState>((set) => ({
  apiBase: localStorage.getItem("sa3_api_base") ?? "http://127.0.0.1:8733",
  selectedArtifactId: null,
  sessionStartedAt: localStorage.getItem("sa3_session_started_at") ?? "1970-01-01T00:00:00.000Z",
  compare: { a: null, b: null },
  setApiBase: (value) => {
    localStorage.setItem("sa3_api_base", value);
    set({ apiBase: value });
  },
  selectArtifact: (artifactId) => set({ selectedArtifactId: artifactId }),
  startSession: () => {
    const startedAt = new Date().toISOString();
    localStorage.setItem("sa3_session_started_at", startedAt);
    set({ sessionStartedAt: startedAt, selectedArtifactId: null, compare: { a: null, b: null } });
  },
  setCompare: (slot, artifactId) =>
    set((state) => ({ compare: { ...state.compare, [slot]: artifactId } })),
}));
