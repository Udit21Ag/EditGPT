/**
 * Rasterising brush strokes, in a real Chromium.
 *
 * `strokes.ts` draws through a canvas deliberately — round caps and joins are what make a
 * dragged stroke read as a brush, and reimplementing capsule rasterisation to keep it pure
 * would be more code, slower and worse looking. The cost is that jsdom cannot run any of
 * it, so this is where it is verified.
 *
 * Every figure below was measured before it was asserted, and each one matches closed-form
 * geometry, which is what makes them worth asserting rather than recording:
 *
 * | | measured | geometry |
 * | --- | ---: | ---: |
 * | a 0.6-long, 0.1-wide stroke on 200x160 | 0.0850 | (120x20 + two r=10 caps) / 32000 = 0.0848 |
 * | a single tap at width 0.1 | 0.00988 | pi x 10^2 / 32000 = 0.00982 |
 * | the same stroke after a 0.05-wide erase through it | 0.0453 | ~0.085 - (120x10)/32000 = 0.0475 |
 */

import { describe, expect, it } from "vitest";
import { maskFromStrokes, maskSize, rasterise } from "./strokes";
import { decodeMask } from "./rle";
import type { Stroke } from "./mask-history";

const SIZE = { width: 200, height: 160 };

const line = (mode: "paint" | "erase", width: number): Stroke => ({
  mode,
  points: [
    [0.2, 0.5],
    [0.8, 0.5],
  ],
  width,
});

function coverage(pixels: Uint8Array): number {
  let set = 0;
  for (const value of pixels) if (value === 1) set += 1;
  return set / pixels.length;
}

function at(pixels: Uint8Array, x: number, y: number): number {
  return pixels[y * SIZE.width + x]!;
}

describe("painting", () => {
  it("covers the area the stroke sweeps, to within a percent of the geometry", () => {
    expect(coverage(rasterise([line("paint", 0.1)], SIZE))).toBeCloseTo(0.085, 2);
  });

  it("puts the stroke where it was drawn", () => {
    const pixels = rasterise([line("paint", 0.1)], SIZE);
    expect(at(pixels, 100, 80)).toBe(1); // on the line
    expect(at(pixels, 100, 20)).toBe(0); // well above it
    expect(at(pixels, 10, 80)).toBe(0); // before it starts
  });

  it("draws a tap as a disc rather than nothing at all", () => {
    // Stroking a zero-length path draws nothing, so a single tap silently did nothing
    // until this was special-cased. On a phone a tap is the most natural gesture there is.
    const dot = rasterise([{ mode: "paint", points: [[0.5, 0.5]], width: 0.1 }], SIZE);
    expect(coverage(dot)).toBeCloseTo(0.0098, 3);
    expect(at(dot, 100, 80)).toBe(1);
  });

  it("scales the brush with the picture, not in pixels", () => {
    // The same rule as the eraser's dilation: a width that is right at 1024 px is a
    // hairline at 15.9 MP.
    const small = coverage(rasterise([line("paint", 0.05)], SIZE));
    const large = coverage(rasterise([line("paint", 0.1)], SIZE));
    expect(large).toBeGreaterThan(small * 1.6);
  });

  it("ignores a stroke with no points instead of throwing", () => {
    expect(coverage(rasterise([{ mode: "paint", points: [], width: 0.1 }], SIZE))).toBe(0);
  });
});

describe("erasing", () => {
  it("lifts what is already painted, in stroke order", () => {
    const before = coverage(rasterise([line("paint", 0.1)], SIZE));
    const after = coverage(rasterise([line("paint", 0.1), line("erase", 0.05)], SIZE));
    expect(after).toBeLessThan(before * 0.6);
    expect(after).toBeGreaterThan(0); // it thins the stroke, it does not delete it
  });

  it("does not paint when it is the only stroke", () => {
    expect(coverage(rasterise([line("erase", 0.1)], SIZE))).toBe(0);
  });
});

describe("handing the mask over", () => {
  it("round-trips through the wire format the gateway reads", () => {
    const strokes = [line("paint", 0.1)];
    const mask = maskFromStrokes(strokes, SIZE)!;
    expect(Array.from(decodeMask(mask))).toEqual(Array.from(rasterise(strokes, SIZE)));
  });

  it("stays compact: a stroke is hundreds of runs, not tens of thousands of pixels", () => {
    const mask = maskFromStrokes([line("paint", 0.1)], SIZE)!;
    expect(mask.counts.length).toBeLessThan(600);
    expect(mask.counts.reduce((a, b) => a + b, 0)).toBe(SIZE.width * SIZE.height);
  });

  it("returns null when nothing survives, so the caller can disable the button", () => {
    // An all-erase history is a real outcome, not a request to reject.
    expect(maskFromStrokes([], SIZE)).toBeNull();
    expect(maskFromStrokes([line("erase", 0.1)], SIZE)).toBeNull();
  });
});

describe("the size a mask is rasterised at", () => {
  it("leaves a picture smaller than the bound alone", () => {
    expect(maskSize(669, 446)).toEqual({ width: 669, height: 446 });
  });

  it("bounds a large one without changing its shape", () => {
    // `EditSpec` accepts a mask at any size whose aspect matches the image, and rejects
    // one that is a few pixels out of shape.
    const size = maskSize(3456, 4608);
    expect(Math.max(size.width, size.height)).toBe(2048);
    expect(Math.abs(size.width * 4608 - size.height * 3456)).toBeLessThanOrEqual(3456 + 4608);
  });
});
