import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { BackendPills, InlineJobStatus, ReadinessPanel, RunMonitor } from "./statusPanels";
import { testBackend, testJob, testReadiness } from "./test/fixtures";

describe("status panels", () => {
  it("surfaces backend availability and readiness priority checks", () => {
    const { container } = render(
      <>
        <BackendPills backends={[testBackend("mlx", true), testBackend("torch_mps", false)]} />
        <ReadinessPanel
          checks={[
            testReadiness("artifact-root", "ok", "artifact root ready"),
            testReadiness("hf-auth", "warn", "HF token missing"),
            testReadiness("backend:mlx", "ok", "MLX ready"),
          ]}
        />
      </>,
    );

    const backendPills = container.querySelector(".backend-pills");
    expect(backendPills).not.toBeNull();
    expect(within(backendPills as HTMLElement).getByText("mlx")).toBeInTheDocument();
    expect(within(backendPills as HTMLElement).getByText("torch_mps")).toHaveAttribute("title", "torch_mps offline");
    expect(screen.getByText("Readiness")).toBeInTheDocument();
    expect(screen.getAllByText("warn").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Artifacts")).toBeInTheDocument();
    expect(screen.getByText("HF auth")).toBeInTheDocument();
    expect(screen.getByText("HF token missing · hf-auth detail")).toBeInTheDocument();
  });

  it("shows active run state and forwards job actions", async () => {
    const user = userEvent.setup();
    const onCancelJob = vi.fn();
    const job = testJob({ job_id: "job_running", status: "running", phase: "diffusing", progress: 0.4 });

    render(<RunMonitor runningJobs={[job]} latestJob={job} eventing onCancelJob={onCancelJob} onRetryJob={vi.fn()} />);

    expect(screen.getByText("1 pending take")).toBeInTheDocument();
    expect(screen.getByText("live progress")).toBeInTheDocument();
    expect(screen.getByText("diffusing")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onCancelJob).toHaveBeenCalledWith(job);
  });

  it("renders the latest failed job inline with retry affordance", async () => {
    const user = userEvent.setup();
    const onRetryJob = vi.fn();
    const failed = testJob({ job_id: "job_failed", status: "failed", error: "shape mismatch" });

    const { container } = render(<InlineJobStatus job={failed} onCancelJob={vi.fn()} onRetryJob={onRetryJob} />);

    const status = container.querySelector(".job-progress.failed");
    expect(status).not.toBeNull();
    expect(within(status as HTMLElement).getByText("Review log tail")).toBeInTheDocument();

    await user.click(within(status as HTMLElement).getByRole("button", { name: "Retry" }));
    expect(onRetryJob).toHaveBeenCalledWith(failed);
  });
});
