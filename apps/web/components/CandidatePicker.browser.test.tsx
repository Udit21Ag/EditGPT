/**
 * The picker's drawing, in a real Chromium.
 *
 * The unit tier covers what a person can do with this component; it cannot cover what
 * they *see*, because jsdom has no 2-D context and `vitest.setup.jsdom.ts` makes
 * `getContext` return null. The picker shipped with fifteen passing tests and its crop
 * arithmetic, mask overlay and image decoding had never executed anywhere.
 *
 * The fixture is four flat quadrants, so which part of the picture reached a thumbnail is
 * legible from its pixels. Measured before these assertions were written, on the
 * geometry below:
 *
 * | | yellow | red/green/blue | tinted |
 * | --- | ---: | ---: | ---: |
 * | thumbnail, mask in the yellow quadrant | 0.975 | 0.000 | 0.226 |
 * | the same picture drawn uncropped | 0.250 | 0.750 | — |
 *
 * That gap is the point: a thumbnail that ignored its candidate and drew the whole frame
 * would still satisfy every test in the unit tier.
 */

import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CandidatePicker } from "./CandidatePicker";
import { MaskPreview } from "./MaskPreview";
import type { Candidate } from "@/lib/api";

const W = 200;
const H = 160;

const RED = "#ff0000";
const GREEN = "#00ff00";
const BLUE = "#0000ff";
const YELLOW = "#ffff00";

function quadrantImage(): string {
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const c = canvas.getContext("2d")!;
  c.fillStyle = RED;
  c.fillRect(0, 0, W / 2, H / 2);
  c.fillStyle = GREEN;
  c.fillRect(W / 2, 0, W / 2, H / 2);
  c.fillStyle = BLUE;
  c.fillRect(0, H / 2, W / 2, H / 2);
  c.fillStyle = YELLOW;
  c.fillRect(W / 2, H / 2, W / 2, H / 2);
  return canvas.toDataURL("image/png");
}

/** Column-major RLE for a rectangle, matching `editgpt_core.rle` on the wire. */
function rleRect(x0: number, y0: number, x1: number, y1: number) {
  const counts: number[] = [];
  let run = 0;
  let previous = 0;
  for (let x = 0; x < W; x += 1) {
    for (let y = 0; y < H; y += 1) {
      const bit = x >= x0 && x < x1 && y >= y0 && y < y1 ? 1 : 0;
      if (bit === previous) {
        run += 1;
      } else {
        counts.push(run);
        run = 1;
        previous = bit;
      }
    }
  }
  counts.push(run);
  if (counts.length % 2 === 0) counts.push(0);
  return { width: W, height: H, counts };
}

function candidate(
  box: [number, number, number, number],
  mask: ReturnType<typeof rleRect>,
  label = "",
): Candidate {
  return { box, score: 0.9, mask, label };
}

/** Wait until something has actually been painted, then read the pixels back. */
async function drawn(canvas: HTMLCanvasElement): Promise<ImageData> {
  return vi.waitFor(
    () => {
      const data = canvas.getContext("2d")!.getImageData(0, 0, canvas.width, canvas.height);
      if (!data.data.some((v, i) => i % 4 === 3 && v > 0)) throw new Error("canvas is blank");
      return data;
    },
    { timeout: 5000, interval: 50 },
  );
}

type Shares = { red: number; green: number; blue: number; yellow: number; other: number };

function tally(data: ImageData): Shares {
  const counts: Shares = { red: 0, green: 0, blue: 0, yellow: 0, other: 0 };
  for (let i = 0; i < data.data.length; i += 4) {
    const r = data.data[i]!;
    const g = data.data[i + 1]!;
    const b = data.data[i + 2]!;
    if (r > 120 && g < 110 && b < 110) counts.red += 1;
    else if (g > 120 && r < 110 && b < 110) counts.green += 1;
    else if (b > 120 && r < 110 && g < 110) counts.blue += 1;
    else if (r > 120 && g > 120 && b < 140) counts.yellow += 1;
    else counts.other += 1;
  }
  const total = data.data.length / 4;
  for (const key of Object.keys(counts) as (keyof Shares)[]) counts[key] /= total;
  return counts;
}

/** Pure yellow carries no blue at all; the tint is (56,132,255) at 0.4, which lifts it. */
function tintedShare(data: ImageData): number {
  let tinted = 0;
  for (let i = 0; i < data.data.length; i += 4) {
    if (data.data[i]! > 120 && data.data[i + 1]! > 120 && data.data[i + 2]! > 40) tinted += 1;
  }
  return tinted / (data.data.length / 4);
}

/** Bottom-right (yellow) and top-left (red), so the two thumbnails cannot be confused. */
const IN_YELLOW = candidate([0.75, 0.75, 0.95, 0.94], rleRect(150, 120, 190, 150), "corner");
const IN_RED = candidate([0.05, 0.06, 0.25, 0.25], rleRect(10, 10, 50, 40), "far side");

function show(candidates: Candidate[], url: string) {
  return render(
    <CandidatePicker
      candidates={candidates}
      imageUrl={url}
      aspect={W / H}
      selected={0}
      onSelect={() => {}}
      phrase="the thing"
    />,
  );
}

describe("what a thumbnail shows", () => {
  it("frames the region the candidate selects, not the whole picture", async () => {
    const { container } = show([IN_YELLOW], quadrantImage());
    const shares = tally(await drawn(container.querySelector("canvas")!));

    expect(shares.yellow).toBeGreaterThan(0.9);
    // Uncropped these three would total 0.75. Measured: 0.000.
    expect(shares.red + shares.green + shares.blue).toBeLessThan(0.05);
  });

  it("tints the selected region, over part of the thumbnail rather than all of it", async () => {
    const { container } = show([IN_YELLOW], quadrantImage());
    const share = tintedShare(await drawn(container.querySelector("canvas")!));

    expect(share).toBeGreaterThan(0.1); // measured 0.226 — the overlay drew
    expect(share).toBeLessThan(0.6); // and it is a region, not a wash over everything
  });

  it("draws each candidate on its own region", async () => {
    // The assertion that makes this a chooser. Thumbnails that all rendered the same
    // frame would leave the user picking between identical pictures, and every test in
    // the unit tier would still pass.
    const { container } = show([IN_YELLOW, IN_RED], quadrantImage());
    const canvases = container.querySelectorAll("canvas");
    expect(canvases).toHaveLength(2);

    const first = tally(await drawn(canvases[0] as HTMLCanvasElement));
    const second = tally(await drawn(canvases[1] as HTMLCanvasElement));

    expect(first.yellow).toBeGreaterThan(0.9);
    expect(second.red).toBeGreaterThan(0.9);
  });
});

describe("the preview of what will change", () => {
  it("keeps the picture outside the mask and tints inside it", async () => {
    const { container } = render(
      <MaskPreview imageUrl={quadrantImage()} mask={IN_YELLOW.mask} alt="preview" />,
    );
    const data = await drawn(container.querySelector("canvas")!);

    // All four quadrants are still there: this is the whole frame, not a crop.
    const shares = tally(data);
    for (const share of [shares.red, shares.green, shares.blue]) {
      expect(share).toBeGreaterThan(0.2);
    }

    const at = (x: number, y: number) => {
      const i = (y * W + x) * 4;
      return [data.data[i]!, data.data[i + 1]!, data.data[i + 2]!];
    };
    expect(at(20, 20)).toEqual([255, 0, 0]); // untouched, far from the mask
    expect(at(170, 135)[2]).toBeGreaterThan(40); // inside the mask, lifted by the tint
  });

  it("draws the picture unchanged when there is no region yet", async () => {
    const { container } = render(<MaskPreview imageUrl={quadrantImage()} mask={null} alt="p" />);
    const data = await drawn(container.querySelector("canvas")!);
    expect(tintedShare(data)).toBe(0);
  });
});
