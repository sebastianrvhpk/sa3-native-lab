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
});
