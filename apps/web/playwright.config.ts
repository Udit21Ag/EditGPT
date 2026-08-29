import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";

import { loadRepoEnv } from "./repo-env";

// The runner decides whether the signed-in specs skip, and it is a different process from
// the server it starts — so it needs the repository's `.env` as much as Next does.
loadRepoEnv();

const here = path.dirname(fileURLToPath(import.meta.url));
// `localhost`, not `127.0.0.1`. They are different origins for cookies, and Clerk's
// session cookie is set on the one the browser visited — sign in against one and reload
// the other and the server sees a signed-out visitor, which is exactly what happened.
const BASE_URL = process.env.EDITGPT_E2E_URL ?? "http://localhost:3210";

/**
 * Full-stack browser tests: a real Next server, a real gateway, a real worker.
 *
 * Distinct from the Vitest `browser` project, which drives the same components with
 * `fetch` replaced. That one proves the client works; this one proves the *deployment*
 * does — Clerk really issuing a session, the gateway really accepting it, the worker
 * really loading models. They fail for different reasons, which is why both exist.
 *
 * The Next server is started here. Everything behind it is not: models take minutes to
 * warm and Postgres and Redis are containers, so `make e2e` brings those up and this
 * assumes they are there. A run against something already deployed sets `EDITGPT_E2E_URL`.
 */
export default defineConfig({
  testDir: path.join(here, "e2e"),
  fullyParallel: false, // one worker, one Celery concurrency; see apps/worker/AGENTS.md
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: process.env.CI === undefined ? "list" : [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer:
    process.env.EDITGPT_E2E_URL === undefined
      ? {
          // The binary directly, not through `pnpm`: it is commonly installed where a
          // login shell can find it and a spawned process cannot, which the Makefile
          // documents and works around the same way.
          command: "./node_modules/.bin/next dev --port 3210",
          url: BASE_URL,
          cwd: here,
          reuseExistingServer: true,
          timeout: 120_000,
        }
      : undefined,
});
