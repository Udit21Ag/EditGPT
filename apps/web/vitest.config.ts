import { defineConfig } from "vitest/config";

/**
 * Root-level settings only. The tiers live in `vitest.workspace.ts`.
 *
 * Coverage cannot be configured per project in a workspace, so it stays here — where it
 * was before the browser tier existed, and with the same provider and reporters.
 */
export default defineConfig({
  test: {
    coverage: { provider: "v8", reporter: ["text", "lcov"] },
  },
});
