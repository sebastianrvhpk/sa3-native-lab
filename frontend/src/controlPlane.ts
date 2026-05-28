import { createTRPCClient, httpBatchLink } from "@trpc/client";

import type { AppRouter, ResultFamily, WorkbenchState } from "../../apps/control-plane/src";

export const DEFAULT_CONTROL_PLANE_URL = import.meta.env.VITE_SA3_CONTROL_PLANE_URL ?? "";

export function createControlPlaneClient(baseUrl: string) {
  return createTRPCClient<AppRouter>({
    links: [
      httpBatchLink({
        url: controlPlaneEndpoint(baseUrl),
      }),
    ],
  });
}

export function controlPlaneEndpoint(baseUrl: string) {
  const trimmed = baseUrl.trim().replace(/\/$/, "");
  if (!trimmed) return "";
  return trimmed.endsWith("/trpc") ? trimmed : `${trimmed}/trpc`;
}

export type { ResultFamily, WorkbenchState };
