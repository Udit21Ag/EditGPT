/**
 * Decoding the run-length masks the gateway sends with each grounding candidate.
 *
 * The encoding is COCO's: **column-major**, always opening with a run of zeros, so odd
 * runs are the set ones. Column-major is the trap — reading the runs straight into a
 * row-major buffer produces a mask that looks plausibly object-shaped and is transposed,
 * which is a bug you notice on a portrait photograph and not on a square test fixture.
 *
 * Masks arrive at the resolution grounding ran at, bounded to 2048 on the longest side,
 * not at the upload's. Nothing here needs to know that: a mask is scaled to whatever it
 * is being drawn onto, and the gateway accepts it back at the size it was sent.
 */

export interface MaskPayload {
  readonly width: number;
  readonly height: number;
  readonly counts: readonly number[];
}

/** Set pixels, without decoding. The encoding opens with zeros, so odd runs are ones. */
export function areaPx(mask: MaskPayload): number {
  let total = 0;
  for (let i = 1; i < mask.counts.length; i += 2) total += mask.counts[i]!;
  return total;
}

/**
 * A row-major `Uint8Array` of 0 and 1, `width * height` long.
 *
 * Transposes on the way out so callers can index it the way a canvas does, `y * width + x`.
 */
export function decodeMask(mask: MaskPayload): Uint8Array {
  const { width, height, counts } = mask;
  const out = new Uint8Array(width * height);

  let index = 0; // position along the column-major flattening
  for (let run = 0; run < counts.length; run += 1) {
    const length = counts[run]!;
    if (run % 2 === 1) {
      for (let k = index; k < index + length; k += 1) {
        // Column-major: k counts down a column before moving right.
        const x = (k / height) | 0;
        const y = k - x * height;
        if (x < width) out[y * width + x] = 1;
      }
    }
    index += length;
  }
  return out;
}

/**
 * A binary mask back to the wire format, column-major, opening with a run of zeros.
 *
 * The inverse of `decodeMask`, and the half the brush needs: a drawn region has to reach
 * the gateway as RLE like any other. Mirrors `editgpt_core.rle.encode` exactly — the
 * opening zero run is what makes odd runs the set ones, and a mask whose very first pixel
 * is set therefore begins with a zero-length run rather than skipping it.
 *
 * `rle.test.ts` asserts this against the same Python-generated fixtures the decoder is
 * checked with, so the two languages cannot drift apart in either direction.
 */
export function encodeMask(pixels: Uint8Array, width: number, height: number): MaskPayload {
  const counts: number[] = [];
  let value = 0; // the encoding always opens with a run of zeros, even an empty one
  let run = 0;

  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      const bit = pixels[y * width + x] === 0 ? 0 : 1;
      if (bit === value) {
        run += 1;
      } else {
        counts.push(run);
        value = bit;
        run = 1;
      }
    }
  }
  counts.push(run);
  return { width, height, counts };
}

export interface Box {
  readonly x0: number;
  readonly y0: number;
  readonly x1: number;
  readonly y1: number;
}

/**
 * The tightest box around the set pixels, in fractions of the image.
 *
 * The candidate carries the *detector's* box, which is what the phrase matched; this is
 * what SAM actually selected. They differ, and the mask is the thing being edited, so it
 * is the honest thing to frame a thumbnail around. Returns null for an empty mask.
 *
 * `decoded` lets a caller that already has the pixels avoid a second pass. A picker with
 * five candidates at 2048 px otherwise decodes 15 million pixels three times over — once
 * here, once for the tint, once for the decode itself.
 */
export function maskBounds(mask: MaskPayload, decoded?: Uint8Array): Box | null {
  const pixels = decoded ?? decodeMask(mask);
  const { width, height } = mask;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      if (pixels[row + x] === 0) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;
  return {
    x0: minX / width,
    y0: minY / height,
    x1: (maxX + 1) / width,
    y1: (maxY + 1) / height,
  };
}

/**
 * A square window around `box`, in fractions, with `context` times its extent around it.
 *
 * Square because the thumbnails sit in a row and a ragged row is harder to compare than
 * the objects in it. Clamped to the image, so a candidate against an edge stays inside
 * the picture instead of framing empty space.
 */
export function windowAround(box: Box, aspect: number, context = 1.8): Box {
  const centreX = (box.x0 + box.x1) / 2;
  const centreY = (box.y0 + box.y1) / 2;

  // In fractions the axes have different scales, so extents are compared in *pixels* —
  // `aspect` (width / height) converts one to the other.
  const extentX = (box.x1 - box.x0) * aspect;
  const extentY = box.y1 - box.y0;
  const side = Math.max(extentX, extentY) * context;

  const halfX = Math.min(side / aspect, 1) / 2;
  const halfY = Math.min(side, 1) / 2;
  const x0 = Math.min(Math.max(centreX - halfX, 0), 1 - halfX * 2);
  const y0 = Math.min(Math.max(centreY - halfY, 0), 1 - halfY * 2);
  return { x0, y0, x1: x0 + halfX * 2, y1: y0 + halfY * 2 };
}

/**
 * RGBA for a mask: a translucent tint over the region, opaque along its boundary.
 *
 * Separated from the canvas on purpose. This is the part with judgement in it — which
 * pixels count as an edge, how much alpha each gets — and a canvas would make it
 * untestable without a native module, so the DOM stays in `tintCanvas` below.
 *
 * The boundary is drawn solid because a flat tint turns to mush at thumbnail size, and
 * the boundary is exactly what the user is being asked to compare.
 */
export function tintPixels(
  mask: MaskPayload,
  colour: readonly [number, number, number],
  alpha = 0.45,
  decoded?: Uint8Array,
): Uint8ClampedArray {
  const pixels = decoded ?? decodeMask(mask);
  const { width, height } = mask;
  const data = new Uint8ClampedArray(width * height * 4);
  const [r, g, b] = colour;
  const edgeAlpha = Math.round(255 * Math.min(alpha * 2.1, 1));
  const fillAlpha = Math.round(255 * alpha);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const at = y * width + x;
      if (pixels[at] === 0) continue;
      const edge =
        x === 0 ||
        y === 0 ||
        x === width - 1 ||
        y === height - 1 ||
        pixels[at - 1] === 0 ||
        pixels[at + 1] === 0 ||
        pixels[at - width] === 0 ||
        pixels[at + width] === 0;
      const offset = at * 4;
      data[offset] = r;
      data[offset + 1] = g;
      data[offset + 2] = b;
      data[offset + 3] = edge ? edgeAlpha : fillAlpha;
    }
  }
  return data;
}

/**
 * The same tint, on a canvas at the mask's own resolution.
 *
 * At the mask's resolution rather than the thumbnail's so the *browser* does the
 * rescaling when it is drawn. Sampling per destination pixel instead would drop a thin
 * object — a bat, a railing — between samples, which is the case the picker exists for.
 */
export function tintCanvas(
  mask: MaskPayload,
  colour: readonly [number, number, number],
  alpha = 0.45,
  decoded?: Uint8Array,
): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = mask.width;
  canvas.height = mask.height;

  const context = canvas.getContext("2d");
  if (context === null) return canvas; // headless: the caller draws nothing, not garbage

  const image = context.createImageData(mask.width, mask.height);
  image.data.set(tintPixels(mask, colour, alpha, decoded));
  context.putImageData(image, 0, 0);
  return canvas;
}
