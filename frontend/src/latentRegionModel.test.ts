import { describe, expect, it } from "vitest";

import { latentRegionDescriptor } from "./latentRegionModel";
import { testArtifact } from "./test/fixtures";

describe("latentRegionModel", () => {
  it("describes backend-supported channel masks for graft and renoise", () => {
    const latent = testArtifact({ kind: "latent", latent: { shape: [256, 64], latent_rate: 10.77, duration_seconds: 6, channel_first: true } });
    const descriptor = latentRegionDescriptor({
      operatorMode: "latent.graft",
      selectedArtifact: latent,
      form: { mode: "channel_block", fraction: 0.25, start_channel: 8, block_size: 12, amount: 0.7 },
    });

    expect(descriptor?.title).toMatch(/channel mask/i);
    expect(descriptor?.detail).toMatch(/block 8 \+ 12/);
    expect(descriptor?.detail).toMatch(/256ch x 64 frames/);
    expect(descriptor?.warning).toMatch(/not bounded time-region/i);
  });

  it("keeps blur and DSP honest about global latent-time controls", () => {
    const latent = testArtifact({ kind: "latent", latent: { shape: [256, 64], latent_rate: 10, duration_seconds: 6.4, channel_first: true } });

    expect(latentRegionDescriptor({
      operatorMode: "latent.blur",
      selectedArtifact: latent,
      form: { mode: "temporal", temporal_radius: 5, temporal_direction: "past" },
    })?.detail).toContain("~0.50s");

    expect(latentRegionDescriptor({
      operatorMode: "latent.dsp",
      selectedArtifact: latent,
      form: { mode: "fft_eq", strength: 0.8 },
    })?.warning).toMatch(/not as waveform EQ/i);
  });
});
