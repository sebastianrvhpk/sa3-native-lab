import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SessionTray } from "./sessionPanel";
import { testArtifact, testFamily, testJob, testRecipe, testSession } from "./test/fixtures";

vi.mock("./audioDeck", () => ({
  TinyWave: ({ artifact }: { artifact: { artifact_id: string } }) => <span data-testid={`tiny-wave-${artifact.artifact_id}`} />,
}));

vi.mock("./jobProgress", () => ({
  JobProgress: ({ job }: { job: { job_id: string; status: string } }) => (
    <div data-testid={`job-${job.job_id}`}>{job.status}</div>
  ),
}));

describe("SessionTray", () => {
  it("filters active takes by text and listening decision", async () => {
    const user = userEvent.setup();
    const keeper = testArtifact({
      artifact_id: "art_keeper",
      label: "Keeper Take",
      tags: ["bright"],
      metadata: { listening_decision: "keeper", model: "medium" },
    });
    const rejected = testArtifact({
      artifact_id: "art_rejected",
      label: "Rejected Take",
      tags: ["dark"],
      metadata: { listening_decision: "rejected", model: "medium" },
      created_at: "2026-05-28T14:00:00.000Z",
    });
    const { container } = renderSessionTray({ artifacts: [keeper, rejected] });

    await user.type(screen.getByLabelText("Filter takes"), "keeper");

    expect(activeTakes(container)).toHaveTextContent("Keeper Take");
    expect(activeTakes(container)).not.toHaveTextContent("Rejected Take");

    await user.click(screen.getByRole("button", { name: "Reject" }));

    expect(activeTakes(container)).toHaveTextContent("No matching takes");
  });

  it("archives active takes and recovers archived takes through row actions", async () => {
    const user = userEvent.setup();
    const onArchiveArtifact = vi.fn();
    const onRecoverArtifact = vi.fn();
    const active = testArtifact({ artifact_id: "art_active", label: "Active Take" });
    const archived = testArtifact({ artifact_id: "art_archived", label: "Archived Take", session_id: null });

    renderSessionTray({
      artifacts: [active],
      archivedArtifacts: [archived],
      onArchiveArtifact,
      onRecoverArtifact,
    });

    const activeRow = screen.getByText("Active Take").closest("article");
    expect(activeRow).not.toBeNull();
    await user.click(within(activeRow as HTMLElement).getByRole("button", { name: "Remember" }));

    const archivedRow = screen.getByText("Archived Take").closest("article");
    expect(archivedRow).not.toBeNull();
    await user.click(within(archivedRow as HTMLElement).getByRole("button", { name: "Recover" }));

    expect(onArchiveArtifact).toHaveBeenCalledWith(active);
    expect(onRecoverArtifact).toHaveBeenCalledWith(archived);
  });

  it("disables session archive while a job is active", () => {
    renderSessionTray({
      runningJobs: [testJob({ job_id: "job_running", status: "running", progress: 0.3 })],
      jobs: [testJob({ job_id: "job_running", status: "running", progress: 0.3 })],
    });

    expect(screen.getByTitle("Remember this session and start a clean one")).toBeDisabled();
    expect(screen.getAllByTestId("job-job_running")).toHaveLength(2);
  });

  it("opens the workspace focus artifact", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const latest = testArtifact({ artifact_id: "art_latest", label: "Latest Take", created_at: "2026-05-28T16:00:00.000Z" });

    renderSessionTray({ artifacts: [latest], selectedId: null, onSelect });

    expect(screen.getByText("Listen next")).toBeInTheDocument();
    await user.click(within(screen.getByLabelText("Workspace pulse")).getByRole("button", { name: "Open" }));

    expect(onSelect).toHaveBeenCalledWith("art_latest");
  });
});

function renderSessionTray({
  artifacts = [testArtifact()],
  archivedArtifacts = [],
  jobs = [testJob()],
  archivedJobs = [],
  families = [testFamily()],
  runningJobs = [],
  selectedId = artifacts[0]?.artifact_id ?? null,
  onSelect = vi.fn(),
  onArchiveArtifact = vi.fn(),
  onRecoverArtifact = vi.fn(),
}: Partial<Parameters<typeof SessionTray>[0]> = {}) {
  return render(
    <SessionTray
      artifacts={artifacts}
      archivedArtifacts={archivedArtifacts}
      jobs={jobs}
      archivedJobs={archivedJobs}
      families={families}
      runningJobs={runningJobs}
      selectedId={selectedId}
      apiBase="http://api.test"
      activeSessionId="sess_1"
      session={testSession()}
      sessionStartedAt="2026-05-28T15:00:00.000Z"
      creatingSession={false}
      archivingSession={false}
      recoveringArtifactId={null}
      archivingArtifactId={null}
      onSelect={onSelect}
      onStartSession={vi.fn()}
      onArchiveSession={vi.fn()}
      onRecoverArtifact={onRecoverArtifact}
      onArchiveArtifact={onArchiveArtifact}
      onCancelJob={vi.fn()}
      onRetryJob={vi.fn()}
    />,
  );
}

function activeTakes(container: HTMLElement) {
  const block = [...container.querySelectorAll(".session-block")].find((item) => item.querySelector(".session-label")?.textContent?.includes("Takes"));
  if (!block) throw new Error("active takes block not found");
  return block as HTMLElement;
}
