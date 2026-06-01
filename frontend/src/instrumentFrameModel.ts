export interface InstrumentFlowInput {
  currentLabel: string | null;
  activeGestureLabel: string;
  pendingCount: number;
  takeCount: number;
  branchCount: number;
  memoryCount: number;
}

export interface InstrumentFlowNode {
  id: string;
  label: string;
  value: string;
  tone: "sound" | "gesture" | "pending" | "take" | "branch" | "memory";
}

export function instrumentFlowNodes(input: InstrumentFlowInput): InstrumentFlowNode[] {
  return [
    {
      id: "current-sound",
      label: "Current Sound",
      value: input.currentLabel ?? "No sound selected",
      tone: "sound",
    },
    {
      id: "active-gesture",
      label: "Gesture",
      value: input.activeGestureLabel,
      tone: "gesture",
    },
    {
      id: "pending-takes",
      label: "Pending",
      value: countLabel(input.pendingCount, "take"),
      tone: "pending",
    },
    {
      id: "take-field",
      label: "Takes",
      value: countLabel(input.takeCount, "sound"),
      tone: "take",
    },
    {
      id: "branches",
      label: "Branches",
      value: countLabel(input.branchCount, "path"),
      tone: "branch",
    },
    {
      id: "memory",
      label: "Memory",
      value: countLabel(input.memoryCount, "material"),
      tone: "memory",
    },
  ];
}

export function instrumentLoopPrompt({
  pendingCount,
  takeCount,
  branchCount,
}: Pick<InstrumentFlowInput, "pendingCount" | "takeCount" | "branchCount">) {
  if (pendingCount > 0) return "Listen for the landing take";
  if (takeCount > 0) return "Listen, branch, remember, or tune";
  if (branchCount > 0) return "Open a branch and continue the sound";
  return "Make the first take";
}

function countLabel(count: number, singular: string) {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}
