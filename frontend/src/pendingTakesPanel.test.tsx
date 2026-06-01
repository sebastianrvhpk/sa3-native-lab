import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { pendingTakeFromJob } from "./pendingTakeModel";
import { PendingTakesPanel } from "./pendingTakesPanel";
import { testJob, testRecipe } from "./test/fixtures";

describe("PendingTakesPanel", () => {
  it("keeps runtime trust details inspectable but redacted", () => {
    const take = pendingTakeFromJob(testJob({
      status: "failed",
      error: "failed with HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz",
      metrics: { command: "python run.py --token hf_abcdefghijklmnopqrstuvwxyz" },
      logs: ["loading", "stderr Authorization: Bearer abcdefghijklmnop"],
      recipe: testRecipe({
        params: { prompt: "warm", token: "hf_abcdefghijklmnopqrstuvwxyz", alternates: ["hf_zyxwvutsrqponmlkjihgfedcba"] },
      }),
    }));

    render(<PendingTakesPanel takes={[take]} />);

    expect(screen.getByText(/failed with HF_TOKEN=\[redacted\]/)).toBeInTheDocument();
    expect(screen.getByText("python run.py --token [redacted]")).toBeInTheDocument();
    expect(screen.getByText(/Authorization: Bearer \[redacted\]/)).toBeInTheDocument();
    expect(screen.getByText(/token: \[redacted-token\]/)).toBeInTheDocument();
    expect(screen.getByText(/alternates: \[\[redacted-token\]\]/)).toBeInTheDocument();
    expect(screen.queryByText(/hf_abcdefghijklmnopqrstuvwxyz/)).not.toBeInTheDocument();
    expect(screen.queryByText(/abcdefghijklmnop/)).not.toBeInTheDocument();
  });
});
