import { createHTTPServer } from "@trpc/server/adapters/standalone";

import { appRouter } from "./router.js";

const port = Number(process.env.SA3_CONTROL_PLANE_PORT ?? 8787);
const pythonBaseUrl = process.env.SA3_PYTHON_API_BASE ?? "http://127.0.0.1:8733";

const server = createHTTPServer({
  router: appRouter,
  basePath: "/trpc/",
  createContext: () => ({
    baseUrl: pythonBaseUrl,
  }),
  middleware: (_req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, trpc-accept");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    if (_req.url === "/health") {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ ok: true, pythonBaseUrl }));
      return;
    }
    if (_req.method === "OPTIONS") {
      res.statusCode = 204;
      res.end();
      return;
    }
    next();
  },
});

server.listen(port);
console.log(`SA3 control plane listening on http://127.0.0.1:${port}/trpc`);
console.log(`Python runtime worker: ${pythonBaseUrl}`);
