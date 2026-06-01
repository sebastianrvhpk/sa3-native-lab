import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SoundInstrumentSurface, ImportAudioButton } from "./SoundInstrumentSurface";
import { instrumentFlowNodes } from "./instrumentFrameModel";

describe("SoundInstrumentSurface", () => {
  it("keeps the first surface focused on sound, gesture, tune, and the take lane", () => {
    render(
      <SoundInstrumentSurface
        selected
        brand={<div>SA3 Native Lab</div>}
        question="What do you want to do with this sound next?"
        loopPrompt="Listen, branch, remember, or tune"
        settings={<button type="button">Settings</button>}
        currentHeader={<h1>Warm Smoke Take</h1>}
        currentSound={<div>current sound slot</div>}
        flowNodes={instrumentFlowNodes({
          currentLabel: "Warm Smoke Take",
          activeGestureLabel: "Continue",
          pendingCount: 1,
          takeCount: 3,
          branchCount: 2,
          memoryCount: 4,
        })}
        sourceCount={5}
        branchCount={2}
        importControl={<button type="button">Import</button>}
        materialBay={<div>source shelf</div>}
        gestureRack={<section aria-label="Gestures">gesture rack</section>}
        nextActions={<section aria-label="Do next">next actions</section>}
        tuneBank={<div>tune bank</div>}
        takeQueue={<div>take queue</div>}
        pendingTakes={<div>pending takes</div>}
        forkPanel={<div>fork panel</div>}
        branchList={<div>branch list</div>}
        branchDetail={<div>branch detail</div>}
        sessionMemory={<div>session memory</div>}
        evidenceDock={<details><summary>Inspect activity</summary></details>}
        utilityDock={<div>utility dock</div>}
      />,
    );

    expect(screen.getByLabelText("SA3 Native Lab sound instrument")).toBeInTheDocument();
    expect(screen.getByLabelText("Prompt and gesture").textContent).toContain("tune bank");
    expect(screen.getByLabelText("Take lane").textContent).toContain("take queue");
    expect(screen.getByLabelText("Secondary instrument trays").textContent).toContain("5 sources");
    expect(screen.getByLabelText("Secondary instrument trays").textContent).toContain("2 paths");
    expect(screen.queryByLabelText("Material bay")).not.toBeInTheDocument();

    const loop = screen.getByLabelText("Instrument loop");
    expect(within(loop).getByText("Current Sound")).toBeInTheDocument();
    expect(within(loop).getByText("Warm Smoke Take")).toBeInTheDocument();
    expect(within(loop).getByText("1 take")).toBeInTheDocument();
  });

  it("routes imported audio files through the import control", async () => {
    const onFile = vi.fn();
    const { container } = render(<ImportAudioButton onFile={onFile} />);
    const input = container.querySelector("input");
    expect(input).not.toBeNull();

    const file = new File(["riff"], "texture.wav", { type: "audio/wav" });
    Object.defineProperty(input, "files", { value: [file], configurable: true });
    input!.dispatchEvent(new Event("change", { bubbles: true }));

    expect(onFile).toHaveBeenCalledWith(file);
  });
});
