/**
 * Drawing with a pointer, in a real Chromium.
 *
 * jsdom cannot host this at all: there is no canvas to paint on and no layout, so
 * `getBoundingClientRect` returns zeros and every normalised coordinate collapses to the
 * same point. What is verified here is the part a person actually does — press, drag,
 * release — and that the result is a stroke in the place they dragged.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BrushCanvas } from "./BrushCanvas";
import { EMPTY_HISTORY, push, type MaskHistory } from "@/lib/mask-history";
import type { PointPrompt } from "@/lib/api";

/**
 * Rendered at an offset and at a size that is not its own.
 *
 * Both matter, and neither happens by default: Tailwind is not loaded in this tier, so
 * `w-full` does nothing and the canvas lays out at its intrinsic 1024 px at the origin.
 * That geometry hides bugs. With `left` at zero, code that forgets to subtract it passes;
 * with a box 1024 wide, code that divides by a constant near 1000 passes too. Both were
 * confirmed against the real thing — a mutation replacing the box arithmetic with
 * `clientX / 1000` survived this suite until these numbers changed.
 */
const OFFSET = { left: 137, top: 61 };
const DISPLAY = { width: 300, height: 240 };

function show(history: MaskHistory = EMPTY_HISTORY, onTap?: (p: PointPrompt) => void) {
  const onChange = vi.fn<(next: MaskHistory) => void>();
  const view = render(
    <div style={{ marginLeft: `${OFFSET.left}px`, marginTop: `${OFFSET.top}px` }}>
      <BrushCanvas
        imageUrl={blank()}
        aspect={200 / 160}
        history={history}
        onChange={onChange}
        onTap={onTap}
      />
    </div>,
  );
  const canvas = view.container.querySelector("canvas") as HTMLCanvasElement;
  canvas.style.width = `${DISPLAY.width}px`;
  canvas.style.height = `${DISPLAY.height}px`;
  return { canvas, onChange, view };
}

function blank(): string {
  const canvas = document.createElement("canvas");
  canvas.width = 200;
  canvas.height = 160;
  const c = canvas.getContext("2d")!;
  c.fillStyle = "#808080";
  c.fillRect(0, 0, 200, 160);
  return canvas.toDataURL("image/png");
}

/** Drag across the canvas in its own coordinate space, as a finger would. */
function drag(canvas: HTMLCanvasElement, from: [number, number], to: [number, number]) {
  const box = canvas.getBoundingClientRect();
  const point = ([x, y]: [number, number]) => ({
    clientX: box.left + x * box.width,
    clientY: box.top + y * box.height,
    pointerId: 1,
  });
  fireEvent.pointerDown(canvas, point(from));
  fireEvent.pointerMove(canvas, point([(from[0] + to[0]) / 2, (from[1] + to[1]) / 2]));
  fireEvent.pointerMove(canvas, point(to));
  fireEvent.pointerUp(canvas, point(to));
}

const painted = (canvas: HTMLCanvasElement) =>
  canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);

describe("drawing", () => {
  it("commits a stroke on release, with the points it was dragged through", () => {
    const { canvas, onChange } = show();
    drag(canvas, [0.2, 0.5], [0.8, 0.5]);

    expect(onChange).toHaveBeenCalledTimes(1);
    const next = onChange.mock.calls[0]![0];
    expect(next.strokes).toHaveLength(1);
    expect(next.strokes[0]!.points.length).toBeGreaterThanOrEqual(3);
    expect(next.strokes[0]!.mode).toBe("paint");
  });

  it("records points normalised to the canvas, so a stroke survives a resize", () => {
    const { canvas, onChange } = show();
    const box = canvas.getBoundingClientRect();
    expect(box.left).toBeGreaterThan(0); // the offset is the point; see `show`
    expect(box.width).toBe(DISPLAY.width);

    drag(canvas, [0.25, 0.5], [0.75, 0.5]);

    const points = onChange.mock.calls[0]![0].strokes[0]!.points;
    expect(points[0]![0]).toBeCloseTo(0.25, 2);
    expect(points[0]![1]).toBeCloseTo(0.5, 2);
    expect(points[points.length - 1]![0]).toBeCloseTo(0.75, 2);
    for (const [x, y] of points) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
    }
  });

  it("commits nothing on a release that never started", () => {
    const { canvas, onChange } = show();
    fireEvent.pointerUp(canvas, { clientX: 10, clientY: 10, pointerId: 1 });
    expect(onChange).not.toHaveBeenCalled();
  });

  it("shows the paint where it was dragged and nowhere else", async () => {
    // The overlay is the only feedback the user gets, and it is drawn from the same
    // `paint` the submitted mask is rasterised with — a preview that disagreed with what
    // was sent would be the worst bug available here, because the user approved it.
    const history = push(EMPTY_HISTORY, {
      mode: "paint",
      points: [
        [0.2, 0.5],
        [0.8, 0.5],
      ],
      width: 0.1,
    });
    const { canvas } = show(history);

    const data = await vi.waitFor(() => {
      const image = painted(canvas);
      if (!image.data.some((v, i) => i % 4 === 3 && v > 0)) throw new Error("blank");
      return image;
    });

    const blue = (x: number, y: number) => data.data[(y * canvas.width + x) * 4 + 2]!;
    // The tint is #3884ff over flat grey, so blue rises sharply where paint landed.
    expect(blue(Math.round(canvas.width / 2), Math.round(canvas.height / 2))).toBeGreaterThan(150);
    expect(blue(Math.round(canvas.width / 2), Math.round(canvas.height * 0.1))).toBeLessThan(150);
  });
});

describe("the tools", () => {
  it("switches to removing, and the next stroke lifts rather than paints", () => {
    // The control is "Remove" rather than "Erase" because it now applies to tapping too:
    // one word for taking something out, whichever tool is in hand.
    const { canvas, onChange } = show();
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    drag(canvas, [0.3, 0.5], [0.6, 0.5]);
    expect(onChange.mock.calls[0]![0].strokes[0]!.mode).toBe("erase");
  });

  it("changes the brush size, and the next stroke carries it", () => {
    const { canvas, onChange } = show();
    fireEvent.click(screen.getByRole("button", { name: "Brush 18%" }));
    drag(canvas, [0.3, 0.5], [0.6, 0.5]);
    expect(onChange.mock.calls[0]![0].strokes[0]!.width).toBeCloseTo(0.18, 3);
  });

  it("offers undo only once there is something to undo", () => {
    const { onChange } = show();
    const undoButton = screen.getByRole("button", { name: "Undo" }) as HTMLButtonElement;
    expect(undoButton.disabled).toBe(true);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("undoes the last stroke", () => {
    const history = push(EMPTY_HISTORY, { mode: "paint", points: [[0.5, 0.5]], width: 0.1 });
    const { onChange } = show(history);
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(onChange.mock.calls[0]![0].strokes).toHaveLength(0);
  });

  it("clears everything at once", () => {
    const history = push(EMPTY_HISTORY, { mode: "paint", points: [[0.5, 0.5]], width: 0.1 });
    const { onChange } = show(history);
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(onChange.mock.calls[0]![0]).toEqual(EMPTY_HISTORY);
  });

  it("prompts while the canvas is still empty", () => {
    show();
    expect(screen.getByText("Paint over what you want changed.")).toBeDefined();
  });
});

describe("tapping to select", () => {
  function tap(canvas: HTMLCanvasElement, [x, y]: [number, number], drift = 0) {
    const box = canvas.getBoundingClientRect();
    const point = (dx: number) => ({
      clientX: box.left + (x + dx) * box.width,
      clientY: box.top + y * box.height,
      pointerId: 1,
    });
    fireEvent.pointerDown(canvas, point(0));
    fireEvent.pointerUp(canvas, point(drift));
  }

  it("reports where the user tapped, in fractions of the picture", () => {
    const onTap = vi.fn<(p: PointPrompt) => void>();
    const { canvas } = show(EMPTY_HISTORY, onTap);

    tap(canvas, [0.4, 0.6]);
    expect(onTap).toHaveBeenCalledTimes(1);
    const point = onTap.mock.calls[0]![0];
    expect(point.x).toBeCloseTo(0.4, 2);
    expect(point.y).toBeCloseTo(0.6, 2);
    expect(point.include).toBe(true);
  });

  it("marks a tap as an exclusion once Remove is chosen", () => {
    // The second half of the interaction: tap the thing, then tap what came along.
    const onTap = vi.fn<(p: PointPrompt) => void>();
    const { canvas } = show(EMPTY_HISTORY, onTap);

    fireEvent.click(screen.getByRole("button", { name: "Remove" }));
    tap(canvas, [0.4, 0.6]);
    expect(onTap.mock.calls[0]![0].include).toBe(false);
  });

  it("ignores a press that travelled, which is a scroll and not a tap", () => {
    // A finger never lands perfectly still; without the slop every stray pixel selects.
    const onTap = vi.fn<(p: PointPrompt) => void>();
    const { canvas } = show(EMPTY_HISTORY, onTap);

    tap(canvas, [0.3, 0.5], 0.2);
    expect(onTap).not.toHaveBeenCalled();
  });

  it("does not paint while the tap tool is in hand", () => {
    const onTap = vi.fn<(p: PointPrompt) => void>();
    const { canvas, onChange } = show(EMPTY_HISTORY, onTap);

    drag(canvas, [0.2, 0.5], [0.8, 0.5]);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("paints again once the brush is chosen", () => {
    const onTap = vi.fn<(p: PointPrompt) => void>();
    const { canvas, onChange } = show(EMPTY_HISTORY, onTap);

    fireEvent.click(screen.getByRole("button", { name: "Brush" }));
    drag(canvas, [0.2, 0.5], [0.8, 0.5]);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onTap).not.toHaveBeenCalled();
  });

  it("offers no tap tool at all when the caller cannot resolve one", () => {
    show();
    expect(screen.queryByRole("button", { name: "Tap" })).toBeNull();
  });
});
