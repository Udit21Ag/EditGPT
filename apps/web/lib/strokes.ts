/**
 * Turning brush strokes into the mask the gateway takes.
 *
 * `mask-history.ts` keeps strokes rather than bitmaps — a stroke is a few hundred bytes
 * where a 15.9 MP mask is ~16 MB — so something has to rasterise them at the moment a job
 * is submitted. That is here.
 *
 * **Drawn with a canvas, on purpose.** Round caps and joins are what make a dragged
 * stroke read as a brush rather than a chain of squares, and the browser already does
 * that correctly. Reimplementing capsule rasterisation to keep this pure would be more
 * code, slower, and worse looking. The cost is that it cannot be tested under jsdom,
 * which is what `strokes.browser.test.ts` and the browser tier exist for.
 */

import { decodeMask, encodeMask, type MaskPayload } from "./rle";
import type { Stroke } from "./mask-history";

export const MAX_MASK_SIDE = 2048;
/** Longest side a drawn mask is rasterised at.
 *
 * The worker edits at this size too, so nothing the user draws is lost to the client
 * being coarser than the pipeline. Going finer would only send bytes that the worker's
 * own downscale immediately discards. An image already smaller than this is rasterised
 * at its own size rather than blown up.
 */

export interface Size {
  readonly width: number;
  readonly height: number;
}

export function maskSize(width: number, height: number): Size {
  const longest = Math.max(width, height);
  if (longest <= MAX_MASK_SIDE) return { width, height };
  const scale = MAX_MASK_SIDE / longest;
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

/**
 * Paint `strokes` onto a 2-D context sized to it.
 *
 * Exported so the live canvas and the rasteriser draw through exactly the same code: a
 * preview that disagreed with what was submitted would be the worst kind of bug here,
 * because the user approved the preview.
 */
export function paint(
  context: CanvasRenderingContext2D,
  strokes: readonly Stroke[],
  size: Size,
  base: MaskPayload | null = null,
): void {
  const longest = Math.max(size.width, size.height);
  context.clearRect(0, 0, size.width, size.height);
  // A region a tap selected, painted first so strokes add to and erase from it. Composing
  // them is what lets magic select be a *starting point* rather than a separate answer:
  // tap the object, then brush the bit SAM took with it.
  if (base !== null) drawBase(context, base, size);
  context.lineCap = "round";
  context.lineJoin = "round";

  for (const stroke of strokes) {
    if (stroke.points.length === 0) continue;
    // `destination-out` is why erasing works on strokes rather than needing a second
    // buffer: it removes from what is already painted, in stroke order.
    context.globalCompositeOperation = stroke.mode === "erase" ? "destination-out" : "source-over";
    context.strokeStyle = "#ffffff";
    context.fillStyle = "#ffffff";
    const width = Math.max(stroke.width * longest, 1);
    context.lineWidth = width;

    if (stroke.points.length === 1) {
      // A tap is a dot. Stroking a zero-length path draws nothing at all, which made a
      // single tap silently do nothing.
      const [x, y] = stroke.points[0]!;
      context.beginPath();
      context.arc(x * size.width, y * size.height, width / 2, 0, Math.PI * 2);
      context.fill();
      continue;
    }

    context.beginPath();
    stroke.points.forEach(([x, y], index) => {
      const px = x * size.width;
      const py = y * size.height;
      if (index === 0) context.moveTo(px, py);
      else context.lineTo(px, py);
    });
    context.stroke();
  }
  context.globalCompositeOperation = "source-over";
}

/**
 * A mask onto a context, scaled to fit.
 *
 * Scaled rather than assumed to match: the server produces a mask at the resolution
 * grounding ran at, which is bounded the same way but need not be identical, and a
 * one-pixel disagreement would offset the whole selection.
 */
function drawBase(context: CanvasRenderingContext2D, base: MaskPayload, size: Size): void {
  const pixels = decodeMask(base);
  const source = document.createElement("canvas");
  source.width = base.width;
  source.height = base.height;
  const into = source.getContext("2d");
  if (into === null) return;

  const image = into.createImageData(base.width, base.height);
  for (let i = 0; i < pixels.length; i += 1) {
    if (pixels[i] === 0) continue;
    const at = i * 4;
    image.data[at] = 255;
    image.data[at + 1] = 255;
    image.data[at + 2] = 255;
    image.data[at + 3] = 255;
  }
  into.putImageData(image, 0, 0);
  context.drawImage(source, 0, 0, size.width, size.height);
}

/** The painted pixels as a row-major `Uint8Array` of 0 and 1. */
export function rasterise(
  strokes: readonly Stroke[],
  size: Size,
  base: MaskPayload | null = null,
): Uint8Array {
  const pixels = new Uint8Array(size.width * size.height);
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext("2d");
  if (context === null) return pixels;

  paint(context, strokes, size, base);
  const data = context.getImageData(0, 0, size.width, size.height).data;
  for (let i = 0; i < pixels.length; i += 1) {
    // Alpha, not colour: `destination-out` erases by clearing alpha, and the midpoint
    // keeps a round cap's antialiased rim out of the mask rather than fringing it.
    pixels[i] = data[i * 4 + 3]! > 127 ? 1 : 0;
  }
  return pixels;
}

/**
 * Strokes to a mask ready for the wire, or null when nothing survives.
 *
 * Null rather than an empty mask because an all-erase history is a real outcome the
 * caller has to handle — the button should be disabled, not the request rejected.
 */
export function maskFromStrokes(
  strokes: readonly Stroke[],
  size: Size,
  base: MaskPayload | null = null,
): MaskPayload | null {
  const pixels = rasterise(strokes, size, base);
  if (!pixels.some((value) => value === 1)) return null;
  return encodeMask(pixels, size.width, size.height);
}
