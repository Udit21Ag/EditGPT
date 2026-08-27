/**
 * Unmount rendered components between tests.
 *
 * Testing Library registers this itself, but only when Vitest runs with `globals: true` —
 * and this project imports `describe`/`it` explicitly instead. Without it every render
 * stays in the document and queries accumulate across the file: the symptom is a test
 * that asks for two options and is handed fifty.
 */
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);

/**
 * jsdom has no 2-D canvas context and logs a stack trace every time one is asked for.
 *
 * Returning null is what it means, and what a browser with canvas disabled does — so
 * this replaces a page of stderr per render with the degradation path the components are
 * written to take. The drawing itself is not covered here by design: `lib/rle.ts` keeps
 * the pixel logic in `tintPixels`, which is pure, precisely so that testing it does not
 * require a native module.
 */
HTMLCanvasElement.prototype.getContext = (() => null) as typeof HTMLCanvasElement.prototype.getContext;
