import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { OperatorPresetRack } from "./operatorPresetRack";
import type { OperatorPreset, OperatorPresetDiffRow } from "./operatorPresets";

describe("OperatorPresetRack", () => {
  it("selects, renames, saves, reverts, deletes, and explains unsaved changes", async () => {
    const user = userEvent.setup();
    const onSelectPreset = vi.fn();
    const onChangePresetName = vi.fn();
    const onSavePreset = vi.fn();
    const onResetPreset = vi.fn();
    const onDeletePreset = vi.fn();
    const preset = operatorPreset();
    const diffRows: OperatorPresetDiffRow[] = [
      { key: "shift_frames", label: "shift frames", presetValue: 12, currentValue: 24, status: "changed" },
      { key: "wrap", label: "wrap", presetValue: false, currentValue: true, status: "changed" },
    ];

    render(<ControlledRack />);

    expect(screen.getByText("2 unsaved changes")).toBeInTheDocument();
    expect(screen.getByText("shift frames")).toBeInTheDocument();
    expect(screen.getByText("wrap")).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Preset"), "preset_roll");
    await user.clear(screen.getByLabelText("Name"));
    await user.type(screen.getByLabelText("Name"), "Tighter roll");
    await user.click(screen.getByRole("button", { name: /Save/i }));
    await user.click(screen.getByRole("button", { name: /Revert/i }));
    await user.click(screen.getByRole("button", { name: /Delete/i }));

    expect(onSelectPreset).toHaveBeenCalledWith("preset_roll");
    expect(onChangePresetName).toHaveBeenLastCalledWith("Tighter roll");
    expect(onSavePreset).toHaveBeenCalledOnce();
    expect(onResetPreset).toHaveBeenCalledOnce();
    expect(onDeletePreset).toHaveBeenCalledOnce();

    function ControlledRack() {
      const [name, setName] = useState("Wide roll");
      return (
        <OperatorPresetRack
          presets={[preset]}
          selectedPreset={preset}
          selectedPresetId={preset.id}
          presetName={name}
          diffRows={diffRows}
          onSelectPreset={onSelectPreset}
          onChangePresetName={(value) => {
            setName(value);
            onChangePresetName(value);
          }}
          onSavePreset={onSavePreset}
          onResetPreset={onResetPreset}
          onDeletePreset={onDeletePreset}
        />
      );
    }
  });

  it("disables destructive preset actions until there is a selected preset", () => {
    render(
      <OperatorPresetRack
        presets={[]}
        selectedPreset={null}
        selectedPresetId=""
        presetName=""
        diffRows={[]}
        onSelectPreset={vi.fn()}
        onChangePresetName={vi.fn()}
        onSavePreset={vi.fn()}
        onResetPreset={vi.fn()}
        onDeletePreset={vi.fn()}
      />,
    );

    expect(screen.getByText("Preset diff")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Revert/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /Delete/i })).toBeDisabled();
  });
});

function operatorPreset(): OperatorPreset {
  return {
    id: "preset_roll",
    name: "Wide roll",
    operator: "latent.cyclic_roll",
    form: { shift_frames: 12, wrap: false },
    donorArtifactId: "art_latent",
    createdAt: "2026-05-28T15:00:00.000Z",
    updatedAt: "2026-05-28T15:00:00.000Z",
  };
}
