import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useGestureWorkbench } from "./useGestureWorkbench";
import { testArtifact } from "./test/fixtures";

describe("useGestureWorkbench", () => {
  it("routes next actions into gesture and Tune state", () => {
    const selected = testArtifact({ kind: "latent", artifact_id: "art_latent" });
    const { result } = renderHook(() =>
      useGestureWorkbench({
        allArtifacts: [selected],
        selectedArtifact: selected,
        selectArtifact: vi.fn(),
        setCompare: vi.fn(),
        artifactPathForField: (artifact, fieldKey) => `${artifact.path}/${fieldKey}`,
      }),
    );

    act(() => {
      result.current.applyNextAction({
        id: "borrow_texture",
        label: "Borrow Texture",
        description: "Use a donor",
        kind: "gesture",
        available: true,
        disabledReason: null,
        gestureId: "borrow_texture",
        operatorMode: "latent.graft",
      });
    });

    expect(result.current.activeGestureId).toBe("borrow_texture");
    expect(result.current.operator).toBe("latent.graft");
  });

  it("applies memory source and prompt-seed actions without app-level plumbing", () => {
    const selectArtifact = vi.fn();
    const setCompare = vi.fn();
    const remembered = testArtifact({ kind: "audio", artifact_id: "art_memory", prompt: "soft remembered loop" });
    const { result } = renderHook(() =>
      useGestureWorkbench({
        allArtifacts: [remembered],
        selectedArtifact: null,
        selectArtifact,
        setCompare,
        artifactPathForField: (artifact) => artifact.path,
      }),
    );

    act(() => {
      result.current.applyMemoryAction(remembered, {
        id: "source",
        label: "Use as Source",
        description: "Use source",
        intent: "source",
        available: true,
        disabledReason: null,
      });
    });

    expect(selectArtifact).toHaveBeenCalledWith("art_memory");
    expect(setCompare).toHaveBeenCalledWith("b", "art_memory");
    expect(result.current.activeGestureId).toBe("continue");

    act(() => {
      result.current.applyMemoryAction(remembered, {
        id: "prompt_seed",
        label: "Seed Prompt",
        description: "Seed prompt",
        intent: "prompt_seed",
        available: true,
        disabledReason: null,
        value: "soft remembered loop",
      });
    });

    expect(result.current.activeGestureId).toBe("make");
    expect(result.current.generationForm.prompt).toBe("soft remembered loop");
  });
});
