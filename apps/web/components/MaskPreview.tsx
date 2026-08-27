"use client";

/**
 * The uploaded picture with the selected region drawn on it.
 *
 * The thumbnails in the picker answer "which object"; this answers "how much of it",
 * which is the question that matters once a choice is made — a mask that clips a limb or
 * swallows the thing in front of it looks fine as a 112 px crop and obvious here.
 *
 * An erase is destructive and the multi-pass loop is seconds of model time, so showing
 * the region before committing is cheaper than undoing (ADR-0003 rejected "let them undo"
 * for exactly that reason).
 */

import { useEffect, useRef } from "react";
import { tintCanvas, type MaskPayload } from "@/lib/rle";

const TINT: readonly [number, number, number] = [56, 132, 255];

export function MaskPreview({
  imageUrl,
  mask,
  alt,
}: {
  imageUrl: string;
  mask: MaskPayload | null;
  alt: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (canvas === null || typeof Image === "undefined") return;

    let live = true;
    const image = new Image();
    image.src = imageUrl;
    image.onload = () => {
      if (!live) return;
      // The canvas is sized to the image so the mask lands on it one-to-one; CSS scales
      // it down to whatever the layout allows.
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d");
      if (context === null) return;
      context.drawImage(image, 0, 0);
      if (mask !== null) {
        const tint = tintCanvas(mask, TINT, 0.4);
        context.drawImage(tint, 0, 0, canvas.width, canvas.height);
      }
    };
    return () => {
      live = false;
    };
  }, [imageUrl, mask]);

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={alt}
      className="max-h-[60vh] w-full rounded-lg bg-neutral-100 object-contain dark:bg-neutral-900"
    />
  );
}
