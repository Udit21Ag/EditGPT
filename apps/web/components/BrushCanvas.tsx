"use client";

/**
 * Painting the region by hand, for when describing it does not work.
 *
 * This is the other half of the chooser. ADR-0003 measured the right region as present in
 * the top five candidates 83.2% of the time, which leaves about one edit in six with no
 * way forward — "None of these" was a dead end saying "describe it again". A phrase that
 * the detector cannot ground is not going to ground on the second attempt either.
 *
 * It is also the way past two defects that have no fix under this project's memory
 * budget: the mask swallowing whatever occludes the target (TD-004) and grounding that
 * does not generalise to relational phrases (TD-012). Neither is repaired by a brush.
 * Both stop being dead ends.
 *
 * **Strokes, never bitmaps.** `mask-history.ts` holds the history; a stroke is a few
 * hundred bytes where a 15.9 MP mask is ~16 MB, and normalised points replay correctly at
 * any size. Nothing here keeps a pixel buffer between renders.
 *
 * **Pointer events, not mouse events.** One code path for a mouse, a finger and a stylus,
 * which is what `apps/web/AGENTS.md` means by not bolting a phone layout onto a desktop
 * tool. `touch-action: none` is load-bearing: without it a drag scrolls the page instead
 * of drawing.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  EMPTY_HISTORY,
  canRedo,
  canUndo,
  hasMask,
  push,
  redo,
  undo,
  type MaskHistory,
  type Stroke,
  type StrokeMode,
} from "@/lib/mask-history";
import { paint, type Size } from "@/lib/strokes";

const TINT = "#3884ff";
const CANVAS_SIDE = 1024;

/** Brush diameter as a fraction of the image's longest side. */
export const BRUSH_SIZES = [0.02, 0.05, 0.1, 0.18] as const;
export const DEFAULT_BRUSH = 0.05;

export interface BrushCanvasProps {
  imageUrl: string;
  /** Width divided by height of the picture, so the canvas matches it. */
  aspect: number;
  history: MaskHistory;
  onChange: (history: MaskHistory) => void;
}

function displaySize(aspect: number): Size {
  return aspect >= 1
    ? { width: CANVAS_SIDE, height: Math.round(CANVAS_SIDE / aspect) }
    : { width: Math.round(CANVAS_SIDE * aspect), height: CANVAS_SIDE };
}

export function BrushCanvas({ imageUrl, aspect, history, onChange }: BrushCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [mode, setMode] = useState<StrokeMode>("paint");
  const [width, setWidth] = useState<number>(DEFAULT_BRUSH);
  const [drawing, setDrawing] = useState<Stroke | null>(null);

  // Memoised because it is an effect dependency: rebuilt every render it would redraw the
  // canvas every render, which drops frames mid-stroke on a phone.
  const size = useMemo(() => displaySize(aspect), [aspect]);

  useEffect(() => {
    if (typeof Image === "undefined") return;
    const element = new Image();
    element.src = imageUrl;
    let live = true;
    element.onload = () => {
      if (live) setImage(element);
    };
    return () => {
      live = false;
    };
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const context = canvas.getContext("2d");
    if (context === null) return;

    context.clearRect(0, 0, size.width, size.height);
    if (image !== null) context.drawImage(image, 0, 0, size.width, size.height);

    // The in-progress stroke is drawn with the committed ones rather than separately, so
    // an erase in flight lifts what is already there instead of floating above it.
    const strokes = drawing === null ? history.strokes : [...history.strokes, drawing];
    if (strokes.length === 0) return;

    const overlay = document.createElement("canvas");
    overlay.width = size.width;
    overlay.height = size.height;
    const layer = overlay.getContext("2d");
    if (layer === null) return;

    paint(layer, strokes, size);
    // Recolour whatever was painted, leaving the untouched pixels transparent.
    layer.globalCompositeOperation = "source-in";
    layer.fillStyle = TINT;
    layer.fillRect(0, 0, size.width, size.height);

    context.globalAlpha = 0.45;
    context.drawImage(overlay, 0, 0);
    context.globalAlpha = 1;
  }, [image, history, drawing, size]);

  const at = useCallback((event: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const box = event.currentTarget.getBoundingClientRect();
    // Normalised to 0..1 against the *displayed* box, so the same stroke replays
    // correctly whatever size the canvas is shown at.
    return [
      Math.min(Math.max((event.clientX - box.left) / box.width, 0), 1),
      Math.min(Math.max((event.clientY - box.top) / box.height, 0), 1),
    ];
  }, []);

  const start = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrawing({ mode, points: [at(event)], width });
  };

  const extend = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (drawing === null) return;
    setDrawing({ ...drawing, points: [...drawing.points, at(event)] });
  };

  const finish = () => {
    if (drawing === null) return;
    onChange(push(history, drawing));
    setDrawing(null);
  };

  const button =
    "rounded-md border px-3 py-2 text-sm disabled:opacity-40 border-neutral-300 dark:border-neutral-700";

  return (
    <div className="flex flex-col gap-3">
      <canvas
        ref={canvasRef}
        width={size.width}
        height={size.height}
        aria-label="Paint the region to change"
        role="img"
        onPointerDown={start}
        onPointerMove={extend}
        onPointerUp={finish}
        onPointerCancel={finish}
        className="w-full touch-none rounded-lg bg-neutral-100 dark:bg-neutral-900"
      />

      {/* Ordered for a thumb: the two that get used constantly sit at the near edge. */}
      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Brush tools">
        <button
          type="button"
          aria-pressed={mode === "paint"}
          onClick={() => setMode("paint")}
          className={`${button} ${mode === "paint" ? "bg-blue-50 dark:bg-blue-950/40" : ""}`}
        >
          Paint
        </button>
        <button
          type="button"
          aria-pressed={mode === "erase"}
          onClick={() => setMode("erase")}
          className={`${button} ${mode === "erase" ? "bg-blue-50 dark:bg-blue-950/40" : ""}`}
        >
          Erase
        </button>

        <span className="flex items-center gap-1" role="group" aria-label="Brush size">
          {BRUSH_SIZES.map((size_) => (
            <button
              key={size_}
              type="button"
              aria-pressed={width === size_}
              aria-label={`Brush ${Math.round(size_ * 100)}%`}
              onClick={() => setWidth(size_)}
              className={`flex h-10 w-10 items-center justify-center rounded-md border ${
                width === size_
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
                  : "border-neutral-300 dark:border-neutral-700"
              }`}
            >
              <span
                className="rounded-full bg-current"
                style={{ width: `${8 + size_ * 60}px`, height: `${8 + size_ * 60}px` }}
              />
            </button>
          ))}
        </span>

        <button type="button" onClick={() => onChange(undo(history))} disabled={!canUndo(history)} className={button}>
          Undo
        </button>
        <button type="button" onClick={() => onChange(redo(history))} disabled={!canRedo(history)} className={button}>
          Redo
        </button>
        <button
          type="button"
          onClick={() => onChange(EMPTY_HISTORY)}
          disabled={history.strokes.length === 0}
          className={button}
        >
          Clear
        </button>
      </div>

      {!hasMask(history) ? (
        <p className="text-sm text-neutral-600 dark:text-neutral-400">
          Paint over what you want changed.
        </p>
      ) : null}
    </div>
  );
}
