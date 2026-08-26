/**
 * Undo/redo for brush strokes on the mask canvas.
 *
 * Strokes are kept rather than bitmaps: a stroke is a few hundred bytes where a
 * 15.9 MP mask is ~16 MB, so a 50-step history costs kilobytes instead of gigabytes.
 * The canvas is re-rasterised from the stroke list, which also makes a stroke
 * resolution-independent — the same history replays correctly after a zoom.
 */

export type StrokeMode = "paint" | "erase";

export interface Stroke {
  readonly mode: StrokeMode;
  /** Points in image space, normalised to 0..1 so they survive a resize. */
  readonly points: ReadonlyArray<readonly [number, number]>;
  /** Brush diameter as a fraction of the image's longest side. */
  readonly width: number;
}

export interface MaskHistory {
  readonly strokes: readonly Stroke[];
  readonly undone: readonly Stroke[];
}

export const EMPTY_HISTORY: MaskHistory = { strokes: [], undone: [] };

/** Maximum retained strokes. Beyond this the oldest are folded away permanently. */
export const MAX_DEPTH = 50;

export function push(history: MaskHistory, stroke: Stroke): MaskHistory {
  if (stroke.points.length === 0) return history;
  const strokes = [...history.strokes, stroke].slice(-MAX_DEPTH);
  // A new stroke invalidates the redo branch, the same as every editor.
  return { strokes, undone: [] };
}

export function undo(history: MaskHistory): MaskHistory {
  if (history.strokes.length === 0) return history;
  const strokes = history.strokes.slice(0, -1);
  const last = history.strokes[history.strokes.length - 1]!;
  return { strokes, undone: [...history.undone, last] };
}

export function redo(history: MaskHistory): MaskHistory {
  if (history.undone.length === 0) return history;
  const last = history.undone[history.undone.length - 1]!;
  return { strokes: [...history.strokes, last], undone: history.undone.slice(0, -1) };
}

export function clear(): MaskHistory {
  return EMPTY_HISTORY;
}

export function canUndo(history: MaskHistory): boolean {
  return history.strokes.length > 0;
}

export function canRedo(history: MaskHistory): boolean {
  return history.undone.length > 0;
}

/** Whether any paint stroke survives; an all-erase history yields no mask at all. */
export function hasMask(history: MaskHistory): boolean {
  return history.strokes.some((stroke) => stroke.mode === "paint");
}
