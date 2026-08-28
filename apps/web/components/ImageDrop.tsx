"use client";

/**
 * Getting a picture in: choose, drop, paste, or the camera.
 *
 * Four routes because they are four different situations. Choosing is the fallback that
 * always works; dropping is what a desktop user does with a file already on screen;
 * pasting is what anyone does with a screenshot, and it is the one people miss most when
 * it is absent; `capture` turns the file input into the camera on a phone, which
 * `apps/web/AGENTS.md` asks for and costs one attribute.
 *
 * The paste listener is on the document rather than an element, because a paste has no
 * meaningful focus target — the user presses the shortcut looking at the page, not at a
 * particular box.
 */

import { useEffect, useState } from "react";
import { pickFile, pickImage } from "@/lib/pick-image";

export function ImageDrop({
  onFile,
  busy,
  replacing,
}: {
  onFile: (file: File) => void;
  busy: boolean;
  replacing: boolean;
}) {
  const [over, setOver] = useState(false);

  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const file = pickImage(event.clipboardData?.items);
      if (file !== null) {
        event.preventDefault();
        onFile(file);
      }
    };
    document.addEventListener("paste", onPaste);
    return () => document.removeEventListener("paste", onPaste);
  }, [onFile]);

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        // `items` carries the markup a browser attaches alongside a dragged picture;
        // `files` is what a file manager sends. Either can be the one with the image.
        const file = pickImage(event.dataTransfer.items) ?? pickFile(event.dataTransfer.files);
        if (file !== null) onFile(file);
      }}
      data-testid="drop-zone"
      className={`flex flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 text-center transition ${
        over
          ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
          : "border-neutral-300 dark:border-neutral-700"
      }`}
    >
      <label className="cursor-pointer rounded-md border border-neutral-300 px-3 py-2 text-sm dark:border-neutral-700">
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp,image/avif"
          className="sr-only"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file !== undefined) onFile(file);
            // Cleared so choosing the same file twice fires again, which it otherwise
            // does not — the value is unchanged and the browser stays quiet.
            event.target.value = "";
          }}
        />
        {busy ? "Uploading…" : replacing ? "Choose a different picture" : "Choose a picture"}
      </label>

      <p className="text-xs text-neutral-500 dark:text-neutral-400">
        or drop one here, or paste with <kbd>⌘V</kbd>
      </p>

      {/* Its own input: `capture` on the one above would send every desktop user to a
          webcam instead of their files. */}
      <label className="cursor-pointer text-xs underline underline-offset-2 sm:hidden">
        <input
          type="file"
          accept="image/*"
          capture="environment"
          className="sr-only"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file !== undefined) onFile(file);
            event.target.value = "";
          }}
        />
        Take a photo
      </label>
    </div>
  );
}
