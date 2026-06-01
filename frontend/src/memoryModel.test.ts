import { describe, expect, it } from "vitest";

import { memoryActionsForArtifact, memoryItemFromArtifact, promptSeedFromMemory } from "./memoryModel";
import { testArtifact } from "./test/fixtures";

describe("memoryModel", () => {
  it("offers active reuse actions for remembered audio", () => {
    const item = memoryItemFromArtifact(
      testArtifact({ kind: "audio", session_id: null, prompt: "warm glass loop", metadata: { listening_decision: "keeper" } }),
      { activeSessionId: "sess_1" },
    );

    expect(item.role).toBe("keeper");
    expect(item.promptSeed).toBe("warm glass loop");
    expect(item.actions.find((action) => action.id === "source")).toMatchObject({ available: true, gestureId: "continue" });
    expect(item.actions.find((action) => action.id === "anchor")).toMatchObject({ available: true, compareSlot: "a" });
    expect(item.actions.find((action) => action.id === "recover")).toMatchObject({ available: true });
  });

  it("uses remembered latents as Borrow Texture donors", () => {
    const actions = memoryActionsForArtifact(testArtifact({ kind: "latent", session_id: null }), { activeSessionId: "sess_1" });

    expect(actions.find((action) => action.id === "donor")).toMatchObject({
      available: true,
      gestureId: "borrow_texture",
    });
    expect(actions.find((action) => action.id === "find_similar")).toMatchObject({
      available: true,
      gestureId: "steer",
      mode: "memory.query",
    });
    expect(actions.find((action) => action.id === "anchor")?.disabledReason).toMatch(/Only audio/);
  });

  it("maps remembered bundles into existing advanced gesture use paths", () => {
    const actions = memoryActionsForArtifact(testArtifact({
      kind: "bundle",
      metadata: { operator: "experiment.style_direction.generate" },
    }));

    expect(actions).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "direction_path", label: "Use direction", mode: "experiment.style_direction.generate" }),
      expect.objectContaining({ id: "vectors_path", label: "Sweep vectors", mode: "experiment.alpha_sweep" }),
    ]));
  });

  it("falls back to label and notes when seeding a prompt", () => {
    expect(promptSeedFromMemory(testArtifact({ prompt: null, label: "Label Seed", notes: "Note Seed" }))).toBe("Label Seed");
    expect(promptSeedFromMemory(testArtifact({ prompt: null, label: null, notes: "Note Seed" }))).toBe("Note Seed");
  });
});
