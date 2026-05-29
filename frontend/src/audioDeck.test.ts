import { describe, expect, it } from "vitest";

import { addPlaybackMarker, clampLoopRegion, formatLoopRegion, markerFractions } from "./audioDeck";

describe("audio deck loop regions", () => {
  it("keeps loop regions inside the loaded duration", () => {
    expect(clampLoopRegion(-2, 12, 10)).toEqual({ start: 0, end: 10 });
  });

  it("widens inverted loop regions around the selected time", () => {
    expect(clampLoopRegion(4, 2, 10)).toEqual({ start: 4, end: 4.5 });
  });

  it("formats loop regions as playback time", () => {
    expect(formatLoopRegion({ start: 1.2, end: 4.8 })).toBe("0:01-0:04");
  });

  it("keeps local playback markers sorted and de-duplicates nearby marks", () => {
    const markers = addPlaybackMarker([], 2, 10);
    const updated = addPlaybackMarker(markers, 2.03, 10);
    const next = addPlaybackMarker(updated, 7, 10);

    expect(updated).toHaveLength(1);
    expect(next.map((marker) => marker.time)).toEqual([2.03, 7]);
    expect(markerFractions(next, 10)).toEqual([0.203, 0.7]);
  });
});
