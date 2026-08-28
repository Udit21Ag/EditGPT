import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineWorkspace } from "vitest/config";

const here = path.dirname(fileURLToPath(import.meta.url));

/**
 * Two tiers, because canvas work cannot be verified where there is no canvas.
 *
 * `unit` is jsdom and fast, and covers the logic. `browser` is a real Chromium and covers
 * what jsdom stubs out: the crop arithmetic, the mask overlay, image decoding. The picker
 * shipped with fifteen passing tests and its drawing code had never executed anywhere —
 * every one of them ran against a `getContext` that returns null.
 *
 * The split is by filename rather than by directory so a module's browser tests sit next
 * to its unit tests, which is where the next person will look for them.
 */
const alias = { "@": here };
const BROWSER = "**/*.browser.test.{ts,tsx}";

export default defineWorkspace([
  {
    plugins: [react()],
    resolve: { alias },
    test: {
      name: "unit",
      environment: "jsdom",
      setupFiles: ["./vitest.setup.ts", "./vitest.setup.jsdom.ts"],
      include: ["**/*.test.{ts,tsx}"],
      exclude: ["node_modules/**", ".next/**", BROWSER],
    },
  },
  {
    plugins: [react()],
    resolve: { alias },
    test: {
      name: "browser",
      setupFiles: ["./vitest.setup.ts", "./vitest.setup.browser.ts"],
      include: [BROWSER],
      exclude: ["node_modules/**", ".next/**"],
      browser: {
        enabled: true,
        provider: "playwright",
        name: "chromium",
        headless: true,
        screenshotFailures: false,
      },
    },
  },
]);
