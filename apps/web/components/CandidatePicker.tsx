"use client";

/**
 * "Which one did you mean?" — the chooser ADR-0003 measured and nothing could reach.
 *
 * The measurement it exists to collect: answering with the top detection is right 51.6%
 * of the time on held-out RefCOCOg, and the right answer is somewhere in the top five
 * 83.2% of the time. That gap is a *UI* gap, not a model gap — every candidate below was
 * already computed, ranked and thrown away.
 *
 * Three things follow from that, and they are why this looks the way it does.
 *
 * **The mask is the thumbnail.** A box tells the user which object; the mask tells them
 * what will actually be erased, which is the thing they are being asked to approve. Each
 * option is cropped to what SAM selected rather than to what the detector boxed.
 *
 * **"None of these" is a first-class answer.** The right region is missing one time in
 * six. A chooser without an exit turns those into a wrong edit rather than a brush stroke.
 *
 * **It has to work with a thumb.** `apps/web/AGENTS.md` is explicit that this is not a
 * desktop tool with a phone layout bolted on, so the options are a scrollable row of
 * real touch targets — and a keyboard user gets arrows, digits and roving focus.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Candidate } from "@/lib/api";
import { decodeMask, maskBounds, tintCanvas, windowAround, type Box } from "@/lib/rle";

/** The selection tint. Chosen to sit on foliage, skin and sky without reading as part of
 * the photograph — which a red or a green does. */
const TINT: readonly [number, number, number] = [56, 132, 255];

export const REJECTED = -1;

export interface CandidatePickerProps {
  candidates: readonly Candidate[];
  /** Object URL of the picture the candidates were grounded against. */
  imageUrl: string;
  /** Width divided by height of that picture, for framing the crops squarely. */
  aspect: number;
  selected: number;
  onSelect: (index: number) => void;
  /** The phrase that produced these, quoted back so the user can see what was searched. */
  phrase: string;
}

interface Framed {
  readonly candidate: Candidate;
  readonly tint: HTMLCanvasElement | null;
  readonly window: Box;
}

/** Decode each mask exactly once, and derive both the crop and the tint from it. */
function useFramed(candidates: readonly Candidate[], aspect: number): Framed[] {
  return useMemo(
    () =>
      candidates.map((candidate) => {
        const pixels = decodeMask(candidate.mask);
        const bounds = maskBounds(candidate.mask, pixels);
        // An empty mask should not reach here — the gateway drops those — but framing the
        // detector's box is a better answer than a crash if one ever does.
        const [bx0, by0, bx1, by1] = candidate.box;
        const box = bounds ?? { x0: bx0, y0: by0, x1: bx1, y1: by1 };
        return {
          candidate,
          window: windowAround(box, aspect),
          tint: typeof document === "undefined" ? null : tintCanvas(candidate.mask, TINT, 0.4, pixels),
        };
      }),
    [candidates, aspect],
  );
}

function Thumbnail({ framed, image }: { framed: Framed; image: HTMLImageElement | null }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (canvas === null || image === null) return;
    const context = canvas.getContext("2d");
    if (context === null) return; // headless, or a browser with canvas disabled

    const { window: crop, tint } = framed;
    const sx = crop.x0 * image.naturalWidth;
    const sy = crop.y0 * image.naturalHeight;
    const sw = (crop.x1 - crop.x0) * image.naturalWidth;
    const sh = (crop.y1 - crop.y0) * image.naturalHeight;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
    if (tint !== null) {
      // The tint canvas is at the mask's resolution, so the same fractional window maps
      // onto it directly — and the browser resamples, which keeps a thin object from
      // vanishing between destination samples.
      context.drawImage(
        tint,
        crop.x0 * tint.width,
        crop.y0 * tint.height,
        (crop.x1 - crop.x0) * tint.width,
        (crop.y1 - crop.y0) * tint.height,
        0,
        0,
        canvas.width,
        canvas.height,
      );
    }
  }, [framed, image]);

  return (
    <canvas
      ref={ref}
      width={224}
      height={224}
      className="h-28 w-28 rounded-md bg-neutral-100 object-cover dark:bg-neutral-800"
      aria-hidden
    />
  );
}

export function CandidatePicker({
  candidates,
  imageUrl,
  aspect,
  selected,
  onSelect,
  phrase,
}: CandidatePickerProps) {
  const framed = useFramed(candidates, aspect);
  const [image, setImage] = useState<HTMLImageElement | null>(null);

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

  const move = useCallback(
    (delta: number) => {
      // Wraps, including onto "None of these", so every option is reachable without
      // knowing how many there are.
      const options = candidates.length + 1;
      const current = selected === REJECTED ? candidates.length : selected;
      const next = (current + delta + options) % options;
      onSelect(next === candidates.length ? REJECTED : next);
    },
    [candidates.length, selected, onSelect],
  );

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      move(1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      move(-1);
    } else if (/^[1-9]$/.test(event.key)) {
      const index = Number(event.key) - 1;
      if (index < candidates.length) {
        event.preventDefault();
        onSelect(index);
      }
    }
  };

  return (
    <div
      role="radiogroup"
      aria-label={`Which “${phrase}” did you mean?`}
      onKeyDown={onKeyDown}
      className="flex flex-col gap-3"
    >
      <p className="text-sm text-neutral-600 dark:text-neutral-400">
        More than one thing matches <span className="font-medium">“{phrase}”</span>. Pick the
        one you meant.
      </p>

      <div className="-mx-1 flex snap-x gap-3 overflow-x-auto px-1 pb-2">
        {framed.map((item, index) => {
          const checked = index === selected;
          return (
            <button
              key={index}
              type="button"
              role="radio"
              aria-checked={checked}
              tabIndex={checked ? 0 : -1}
              onClick={() => onSelect(index)}
              className={`flex shrink-0 snap-start flex-col items-center gap-1.5 rounded-lg border-2 p-2 transition ${
                checked
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
                  : "border-transparent hover:border-neutral-300 dark:hover:border-neutral-700"
              }`}
            >
              <Thumbnail framed={item} image={image} />
              <span className="text-xs font-medium">
                {item.candidate.label || `Option ${index + 1}`}
              </span>
              {index === 0 ? (
                <span className="text-[11px] text-neutral-500 dark:text-neutral-400">
                  Best match
                </span>
              ) : null}
            </button>
          );
        })}

        <button
          type="button"
          role="radio"
          aria-checked={selected === REJECTED}
          tabIndex={selected === REJECTED ? 0 : -1}
          onClick={() => onSelect(REJECTED)}
          className={`flex h-[9.5rem] w-28 shrink-0 snap-start flex-col items-center justify-center gap-1.5 rounded-lg border-2 border-dashed p-2 text-xs transition ${
            selected === REJECTED
              ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
              : "border-neutral-300 hover:border-neutral-400 dark:border-neutral-700"
          }`}
        >
          <span className="text-lg leading-none" aria-hidden>
            ✎
          </span>
          <span className="font-medium">None of these</span>
          <span className="text-[11px] text-neutral-500 dark:text-neutral-400">
            Describe it again
          </span>
        </button>
      </div>
    </div>
  );
}
