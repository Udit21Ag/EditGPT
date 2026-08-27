import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  // The `@/` alias comes from tsconfig `paths`, which only teaches the *type checker*
  // about it. Without the same mapping here, a component importing `@/lib/...` type-checks
  // and then fails to resolve the moment a test renders it.
  resolve: { alias: { "@": here } },
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    include: ["**/*.test.ts", "**/*.test.tsx"],
    exclude: ["node_modules/**", ".next/**"],
    coverage: { provider: "v8", reporter: ["text", "lcov"] },
  },
});
