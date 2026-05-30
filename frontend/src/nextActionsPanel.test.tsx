import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { nextActionsForArtifact } from "./nextActionModel";
import { NextActionsPanel } from "./nextActionsPanel";
import { testArtifact } from "./test/fixtures";

describe("NextActionsPanel", () => {
  it("routes a suggested action when clicked", async () => {
    const user = userEvent.setup();
    const onAction = vi.fn();
    const actions = nextActionsForArtifact(testArtifact({ kind: "audio" }));

    render(<NextActionsPanel actions={actions} onAction={onAction} />);

    await user.click(screen.getByRole("button", { name: /Continue/ }));

    expect(onAction).toHaveBeenCalledWith(expect.objectContaining({
      id: "continue",
      gestureId: "continue",
      generationMode: "generate.audio_to_audio",
    }));
  });
});
