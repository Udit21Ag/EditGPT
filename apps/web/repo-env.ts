import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Load the repository's single `.env` into `process.env`.
 *
 * Next reads `.env` relative to its own working directory, and this app runs from
 * `apps/web` while the project keeps one `.env` at the root — the file `.env.example`
 * documents, `make harness` validates, and the gateway and worker both read. Without this
 * the app compiled, type-checked, tested and then answered 500 with "Missing
 * publishableKey", for as long as Clerk had been configured.
 *
 * Shared by `next.config.ts` and `playwright.config.ts` because both need it and for the
 * same reason: the Playwright *runner* decides whether the signed-in specs skip, and it is
 * a different process from the server it starts.
 *
 * Parsed here rather than through `@next/env`, which pnpm's isolated `node_modules` does
 * not expose to this package, and rather than through `dotenv`, which would be a
 * dependency for a dozen lines. The format is already constrained: `make harness` rejects
 * anything in `.env` that is not a single-line `KEY=VALUE`.
 *
 * A variable already present in the real environment always wins, so CI and a deployment
 * override the file rather than fighting it.
 */
export function loadRepoEnv(): void {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
  let contents: string;
  try {
    contents = readFileSync(path.join(root, ".env"), "utf8");
  } catch {
    return; // no `.env` is normal: a deployment supplies real environment variables
  }

  for (const line of contents.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith("#")) continue;
    const at = trimmed.indexOf("=");
    if (at <= 0) continue;
    const key = trimmed.slice(0, at).trim();
    const value = trimmed
      .slice(at + 1)
      .trim()
      .replace(/^["']|["']$/g, "");
    if (value.length > 0 && process.env[key] === undefined) process.env[key] = value;
  }
}
