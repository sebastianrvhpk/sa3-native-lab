import { describe, expect, it } from "vitest";

import {
  createOperatorPreset,
  deleteOperatorPreset,
  loadOperatorPresets,
  persistOperatorPresets,
  upsertOperatorPreset,
} from "./operatorPresets";

describe("operator presets", () => {
  it("persists sanitized operator presets in newest-first order", () => {
    const storage = new MemoryStorage();
    const first = createOperatorPreset({
      id: "preset_a",
      name: "soft roll",
      operator: "latent.cyclic_roll",
      form: { shift_frames: 4, symmetric: true, ignored: {} as never },
      now: "2026-05-28T12:00:00.000Z",
    });
    const second = createOperatorPreset({
      id: "preset_b",
      name: "phase graft",
      operator: "latent.dsp",
      form: { mode: "fft_phase_blend", mix: 0.4 },
      donorArtifactId: "art_donor",
      now: "2026-05-28T12:05:00.000Z",
    });

    persistOperatorPresets([first, second], storage as Storage);

    expect(loadOperatorPresets(storage as Storage)).toEqual([
      expect.objectContaining({ id: "preset_b", donorArtifactId: "art_donor" }),
      expect.objectContaining({ id: "preset_a", form: { shift_frames: 4, symmetric: true } }),
    ]);
  });

  it("upserts and deletes presets by id", () => {
    const original = createOperatorPreset({
      id: "preset_a",
      name: "soft roll",
      operator: "latent.cyclic_roll",
      form: { shift_frames: 4 },
      now: "2026-05-28T12:00:00.000Z",
    });
    const updated = createOperatorPreset({
      id: "preset_a",
      name: "brighter roll",
      operator: "latent.cyclic_roll",
      form: { shift_frames: 8 },
      createdAt: original.createdAt,
      now: "2026-05-28T12:10:00.000Z",
    });

    const next = upsertOperatorPreset([original], updated);

    expect(next).toEqual([expect.objectContaining({ id: "preset_a", name: "brighter roll", form: { shift_frames: 8 } })]);
    expect(deleteOperatorPreset(next, "preset_a")).toEqual([]);
  });
});

class MemoryStorage {
  private values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }

  clear(): void {
    this.values.clear();
  }

  key(index: number): string | null {
    return [...this.values.keys()][index] ?? null;
  }

  get length(): number {
    return this.values.size;
  }
}
