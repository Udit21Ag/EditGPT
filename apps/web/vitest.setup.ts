/**
 * Unmount rendered components between tests. Shared by both tiers.
 *
 * Testing Library registers this itself, but only when Vitest runs with `globals: true` —
 * and this project imports `describe`/`it` explicitly. Without it every render stays in
 * the document and queries accumulate across the file: the symptom is a test that asks
 * for two options and is handed fifty.
 */
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);
