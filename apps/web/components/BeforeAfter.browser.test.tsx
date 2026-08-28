/**
 * The wipe, in a real Chromium.
 *
 * Layout is the whole mechanism here — a clip percentage means nothing without a box to
 * clip — so jsdom cannot verify any of it.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BeforeAfter } from "./BeforeAfter";

function flat(colour: string): string {
  const canvas = document.createElement("canvas");
  canvas.width = 40;
  canvas.height = 30;
  const c = canvas.getContext("2d")!;
  c.fillStyle = colour;
  c.fillRect(0, 0, 40, 30);
  return canvas.toDataURL("image/png");
}

/** The clipped-away percentage. Read out rather than string-matched because the browser
 * normalises `0` to `0px` — it is the authority on its own computed style, not us. */
function clippedAt(image: HTMLImageElement): number {
  const found = /([\d.]+)%/.exec(image.style.clipPath);
  expect(found, `no percentage in clipPath: ${image.style.clipPath}`).not.toBeNull();
  return Number(found![1]);
}

function show() {
  const view = render(<BeforeAfter before={flat("#ff0000")} after={flat("#0000ff")} />);
  const slider = screen.getByRole("slider") as HTMLInputElement;
  const after = view.container.querySelectorAll("img")[1] as HTMLImageElement;
  return { slider, after, view };
}

describe("wiping between the two", () => {
  it("starts halfway, showing some of each", () => {
    const { slider, after } = show();
    expect(slider.value).toBe("50");
    expect(clippedAt(after)).toBe(50);
  });

  it("moves the clip with the handle", () => {
    const { slider, after } = show();
    fireEvent.change(slider, { target: { value: "80" } });
    expect(clippedAt(after)).toBe(80);
  });

  it("hides the result entirely at one end and shows it whole at the other", () => {
    const { slider, after } = show();
    fireEvent.change(slider, { target: { value: "100" } });
    expect(clippedAt(after)).toBe(100);
    fireEvent.change(slider, { target: { value: "0" } });
    expect(clippedAt(after)).toBe(0);
  });

  it("overlays the two exactly, so the wipe compares the same pixels", () => {
    // The failure this catches: the clipped image laid out below the first instead of on
    // top of it, which looks like a wipe that does nothing.
    const { view } = show();
    const images = view.container.querySelectorAll("img");
    const first = images[0]!.getBoundingClientRect();
    const second = images[1]!.getBoundingClientRect();
    expect(second.top).toBeCloseTo(first.top, 0);
    expect(second.left).toBeCloseTo(first.left, 0);
    expect(second.width).toBeCloseTo(first.width, 0);
    expect(first.width).toBeGreaterThan(0);
  });

  it("is reachable by keyboard, because the handle is a real slider", () => {
    const { slider } = show();
    expect(slider.getAttribute("aria-label")).toMatch(/wipe/i);
    expect(slider.type).toBe("range");
  });
});
