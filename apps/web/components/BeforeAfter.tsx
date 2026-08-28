"use client";

/**
 * The result over the original, with a handle to wipe between them.
 *
 * Two images stacked with the top one clipped, rather than a canvas: the browser's own
 * `clip-path` is exact at any size, costs no redraw, and keeps both pictures as real
 * `<img>` elements a viewer can zoom or save. A canvas would resample both.
 *
 * The handle is a range input under a transparent layer, so a keyboard gets arrow keys and
 * a screen reader gets a labelled slider for free — the things a hand-rolled drag handle
 * has to reimplement, badly.
 *
 * The overlay geometry is an inline style rather than a utility class, unlike everything
 * cosmetic here. Exact overlap is the *mechanism*: a clip percentage means nothing unless
 * the two pictures occupy the same box, and pairing an inline `clip-path` with positioning
 * that lives in a stylesheet makes the component correct only when that sheet has loaded.
 */

import { useState } from "react";

export function BeforeAfter({ before, after }: { before: string; after: string }) {
  const [at, setAt] = useState(50);

  return (
    <figure
      className="w-full select-none overflow-hidden rounded-lg"
      style={{ position: "relative" }}
    >
      {/* eslint-disable @next/next/no-img-element */}
      {/* The first picture sets the box; the second is clipped inside it. Both are
          mechanism, so both are sized here rather than by a utility class. */}
      <img src={before} alt="Before the edit" style={{ display: "block", width: "100%" }} />
      <img
        src={after}
        alt="After the edit"
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          clipPath: `inset(0 0 0 ${at}%)`,
        }}
      />
      {/* eslint-enable @next/next/no-img-element */}

      <div
        aria-hidden
        className="pointer-events-none bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
        style={{ position: "absolute", top: 0, bottom: 0, width: "2px", left: `${at}%` }}
      />

      <input
        type="range"
        min={0}
        max={100}
        value={at}
        onChange={(event) => setAt(Number(event.target.value))}
        aria-label="Wipe between the original and the result"
        className="cursor-ew-resize opacity-0"
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
      />
      <figcaption className="sr-only">
        Original on the left, result on the right. Drag or use the arrow keys.
      </figcaption>
    </figure>
  );
}
