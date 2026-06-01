import { describe, expect, it } from "vitest";

import { instrumentFlowNodes, instrumentLoopPrompt } from "./instrumentFrameModel";

describe("instrument frame model", () => {
  it("builds a real loop strip from current workbench counts", () => {
    const nodes = instrumentFlowNodes({
      currentLabel: "Warm Smoke Take",
      activeGestureLabel: "Continue",
      pendingCount: 2,
      takeCount: 5,
      branchCount: 3,
      memoryCount: 4,
    });

    expect(nodes.map((node) => `${node.label}:${node.value}`)).toEqual([
      "Current Sound:Warm Smoke Take",
      "Gesture:Continue",
      "Pending:2 takes",
      "Takes:5 sounds",
      "Branches:3 paths",
      "Memory:4 materials",
    ]);
    expect(nodes.map((node) => node.tone)).toEqual(["sound", "gesture", "pending", "take", "branch", "memory"]);
  });

  it("keeps the loop prompt tied to real workbench state", () => {
    expect(instrumentLoopPrompt({ pendingCount: 1, takeCount: 0, branchCount: 0 })).toBe("Listen for the landing take");
    expect(instrumentLoopPrompt({ pendingCount: 0, takeCount: 2, branchCount: 0 })).toBe("Listen, branch, remember, or tune");
    expect(instrumentLoopPrompt({ pendingCount: 0, takeCount: 0, branchCount: 2 })).toBe("Open a branch and continue the sound");
    expect(instrumentLoopPrompt({ pendingCount: 0, takeCount: 0, branchCount: 0 })).toBe("Make the first take");
  });
});
