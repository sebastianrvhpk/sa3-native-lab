import { createHTTPServer } from "@trpc/server/adapters/standalone";

import { appRouter } from "./router.js";

const port = Number(process.env.SA3_CONTROL_PLANE_PORT ?? 8787);
const pythonBaseUrl = process.env.SA3_PYTHON_API_BASE ?? "http://127.0.0.1:8733";

const server = createHTTPServer({
  router: appRouter,
  createContext: () => ({
    baseUrl: pythonBaseUrl,
  }),
});

server.listen(port);
console.log(`SA3 control plane listening on http://127.0.0.1:${port}/trpc`);
console.log(`Python runtime worker: ${pythonBaseUrl}`);
