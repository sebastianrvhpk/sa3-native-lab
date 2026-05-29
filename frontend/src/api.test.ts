import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

import { createApi } from "./api";

const server = setupServer(
  http.get("http://api.test/health", () =>
    HttpResponse.json({
      app: "sa3-native-lab",
      version: "test",
      artifact_root: "/tmp/lab",
      backends: [{ backend: "mlx", available: true, loaded: false, details: {} }],
    }),
  ),
  http.get("http://api.test/artifacts/art_bundle/inspect", () =>
    HttpResponse.json({
      artifact: {
        artifact_id: "art_bundle",
        kind: "bundle",
        path: "/tmp/art_bundle/bundle.zip",
        file: { filename: "bundle.zip", media_type: "application/zip", byte_size: 32 },
        source_artifact_ids: [],
        recipe_id: "recipe_1",
        tags: [],
        metadata: {},
        created_at: "2026-05-27T15:00:00.000Z",
      },
      recipe: null,
      sources: [],
      children: [],
      bundle_files: [{ path: "metrics.json", byte_size: 16, compressed_size: 12 }],
      bundle_audio_files: [{ path: "take.wav", byte_size: 3200, media_type: "audio/wav", duration_seconds: 0.1 }],
      bundle_preview: { result_count: 2 },
      bundle_summary: { kind: "sweep", file_count: 1 },
    }),
  ),
  http.get("http://api.test/artifacts/art_target/descriptor-comparison/art_take", () =>
    HttpResponse.json({
      target_artifact_id: "art_target",
      take_artifact_id: "art_take",
      target: { duration_seconds: 8, rms_dbfs: -18, spectral_centroid_hz: 440 },
      take: { duration_seconds: 8, rms_dbfs: -16, spectral_centroid_hz: 900 },
      delta: { rms_dbfs: 2, spectral_centroid_hz: 460, spectral_flux: 0.02, spectral_flatness: 0.1, stereo_width: 0.05 },
    }),
  ),
  http.get("http://api.test/readiness", () =>
    HttpResponse.json({
      ok: true,
      warnings: 0,
      errors: 0,
      checks: [{ name: "artifact-root", status: "ok", message: "/tmp/lab" }],
    }),
  ),
  http.post("http://api.test/artifacts/art_bundle/bundle-audio/promote", async ({ request }) => {
    const payload = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      artifact_id: "art_promoted",
      kind: "audio",
      path: "/tmp/art_promoted/take.wav",
      file: { filename: "take.wav", media_type: "audio/wav", byte_size: 3200 },
      audio: { sample_rate: 8000, channels: 1, frames: 800, duration_seconds: 0.1, format: "WAV" },
      source_artifact_ids: ["art_bundle"],
      recipe_id: "recipe_promote",
      label: payload.label ?? "take",
      prompt: payload.prompt ?? null,
      tags: [],
      metadata: { operator: "artifact.promote_bundle_audio", bundle_audio_path: payload.path },
      session_id: payload.session_id ?? null,
      created_at: "2026-05-27T15:00:00.000Z",
    });
  }),
  http.post("http://api.test/recipes/recipe_1/fork", async ({ request }) => {
    const payload = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      job_id: "job_fork",
      status: "queued",
      recipe: {
        recipe_id: "recipe_fork",
        operator: "latent.cyclic_roll",
        backend: payload.backend ?? "torch_mps",
        inputs: {},
        params: payload.params ?? {},
        seed: payload.seed ?? null,
        session_id: payload.session_id ?? null,
        created_at: "2026-05-27T15:00:00.000Z",
        version: 1,
      },
      progress: 0,
      artifact_ids: [],
      metrics: {},
      logs: [],
      created_at: "2026-05-27T15:00:00.000Z",
    });
  }),
  http.patch("http://api.test/sessions/sess_1", async ({ request }) => {
    const payload = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      session_id: "sess_1",
      name: payload.name ?? "session one",
      status: payload.status ?? "active",
      notes: payload.notes ?? null,
      metadata: {},
      created_at: "2026-05-27T15:00:00.000Z",
      updated_at: "2026-05-27T15:05:00.000Z",
      archived_at: payload.status === "archived" ? "2026-05-27T15:05:00.000Z" : null,
    });
  }),
  http.post("http://api.test/artifacts/art_archive/annotate", async ({ request }) => {
    const payload = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      artifact_id: "art_archive",
      kind: "audio",
      path: "/tmp/art_archive.wav",
      file: { filename: "art_archive.wav", media_type: "audio/wav", byte_size: 3200 },
      audio: { sample_rate: 8000, channels: 1, frames: 800, duration_seconds: 0.1, format: "WAV" },
      source_artifact_ids: [],
      recipe_id: "recipe_old",
      label: "archive take",
      prompt: null,
      notes: null,
      tags: ["maybe"],
      metadata: payload.metadata ?? {},
      session_id: payload.session_id ?? null,
      created_at: "2026-05-27T15:00:00.000Z",
    });
  }),
);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("createApi", () => {
  it("reads mocked API responses through MSW", async () => {
    await expect(createApi("http://api.test").health()).resolves.toMatchObject({
      artifact_root: "/tmp/lab",
      backends: [{ backend: "mlx", available: true }],
    });
  });

  it("inspects bundle file inventories", async () => {
    const api = createApi("http://api.test");
    await expect(api.inspectArtifact("art_bundle")).resolves.toMatchObject({
      artifact: { artifact_id: "art_bundle" },
      bundle_files: [{ path: "metrics.json", byte_size: 16 }],
      bundle_audio_files: [{ path: "take.wav", media_type: "audio/wav" }],
      bundle_preview: { result_count: 2 },
    });
    expect(api.bundleFileUrl("art_bundle", "plots/main plot.png")).toBe(
      "http://api.test/artifacts/art_bundle/bundle-file?path=plots%2Fmain%20plot.png",
    );
  });

  it("promotes audio from a bundle into a first-class artifact", async () => {
    await expect(
      createApi("http://api.test").promoteBundleAudio("art_bundle", {
        path: "take.wav",
        label: "keeper take",
        session_id: "sess_1",
      }),
    ).resolves.toMatchObject({
      artifact_id: "art_promoted",
      kind: "audio",
      label: "keeper take",
      session_id: "sess_1",
      metadata: { operator: "artifact.promote_bundle_audio", bundle_audio_path: "take.wav" },
    });
  });

  it("compares descriptors between a target and generated take", async () => {
    await expect(createApi("http://api.test").audioDescriptorComparison("art_target", "art_take")).resolves.toMatchObject({
      target_artifact_id: "art_target",
      take_artifact_id: "art_take",
      delta: { rms_dbfs: 2, spectral_centroid_hz: 460 },
    });
  });

  it("reads readiness checks", async () => {
    await expect(createApi("http://api.test").readiness()).resolves.toMatchObject({
      ok: true,
      checks: [{ name: "artifact-root" }],
    });
  });

  it("forks recipes with edited params", async () => {
    await expect(
      createApi("http://api.test").forkRecipe("recipe_1", {
        backend: "torch_mps",
        params: { shift_frames: 4 },
        seed: 11,
      }),
    ).resolves.toMatchObject({
      job_id: "job_fork",
      recipe: { params: { shift_frames: 4 }, seed: 11 },
    });
  });

  it("updates session archive state", async () => {
    await expect(createApi("http://api.test").updateSession("sess_1", { status: "archived" })).resolves.toMatchObject({
      session_id: "sess_1",
      status: "archived",
      archived_at: "2026-05-27T15:05:00.000Z",
    });
  });

  it("recovers an archived artifact into a target session through annotation", async () => {
    await expect(
      createApi("http://api.test").annotateArtifact("art_archive", {
        session_id: "sess_1",
        metadata: { recovered_from_session_id: "sess_old" },
      }),
    ).resolves.toMatchObject({
      artifact_id: "art_archive",
      session_id: "sess_1",
      metadata: { recovered_from_session_id: "sess_old" },
    });
  });

  it("archives an active artifact by sending an explicit null session id", async () => {
    await expect(
      createApi("http://api.test").annotateArtifact("art_archive", {
        session_id: null,
        metadata: { archived_from_session_id: "sess_1" },
      }),
    ).resolves.toMatchObject({
      artifact_id: "art_archive",
      session_id: null,
      metadata: { archived_from_session_id: "sess_1" },
    });
  });
});
