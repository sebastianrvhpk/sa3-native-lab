import { describe, expect, it } from "vitest";

import { latestSafeRuntimeLine, safeRuntimeLogLines, sanitizeRuntimeText } from "./runtimeTrustModel";

describe("runtimeTrustModel", () => {
  it("redacts common command and log secrets", () => {
    expect(sanitizeRuntimeText("python run.py --token hf_abcdefghijklmnopqrstuvwxyz --api-key=sk-secret-value")).toBe(
      "python run.py --token [redacted] --api-key=[redacted]",
    );
    expect(sanitizeRuntimeText("Authorization: Bearer abcdefghijklmnop")).toBe("Authorization: Bearer [redacted]");
    expect(sanitizeRuntimeText("HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz")).toBe("HF_TOKEN=[redacted]");
    expect(sanitizeRuntimeText('{"api_key":"super-secret-value"}')).toBe('{"api_key":"[redacted]"}');
  });

  it("keeps the useful tail while sanitizing each line", () => {
    const lines = safeRuntimeLogLines(["a", "b", "download hf_abcdefghijklmnopqrstuvwxyz", "done"], 2);

    expect(lines).toEqual(["download [redacted-token]", "done"]);
  });

  it("finds the latest non-heartbeat runtime line safely", () => {
    expect(latestSafeRuntimeLine(["loading", "[heartbeat] ok", "failed with token=hf_abcdefghijklmnopqrstuvwxyz"])).toBe(
      "failed with token=[redacted]",
    );
  });
});
