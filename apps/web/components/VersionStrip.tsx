"use client";

/**
 * Every result this session produced, and going back to one.
 *
 * Choosing an earlier version makes it the *input* to the next edit — the branch in
 * "version history and branch-from-version". Nothing is overwritten to make that work:
 * assets are content-addressed, so each result is simply another digest and stepping back
 * is choosing which one to point at.
 *
 * A radio group rather than buttons, because exactly one version is current at a time and
 * that is what a radio group means to a screen reader.
 */

import type { Version } from "@/lib/versions";

export function VersionStrip({
  versions,
  current,
  urls,
  onPick,
}: {
  versions: readonly Version[];
  current: number;
  urls: ReadonlyMap<string, string>;
  onPick: (index: number) => void;
}) {
  if (versions.length < 2) return null; // nothing to go back to yet

  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-xs font-medium text-neutral-600 dark:text-neutral-400">History</h2>
      <div
        role="radiogroup"
        aria-label="Version history"
        className="flex gap-2 overflow-x-auto pb-1"
      >
        {versions.map((version, index) => (
          <button
            key={version.sha256}
            type="button"
            role="radio"
            aria-checked={index === current}
            tabIndex={index === current ? 0 : -1}
            onClick={() => onPick(index)}
            title={version.label}
            className={`flex w-24 shrink-0 flex-col items-center gap-1 rounded-md border-2 p-1 ${
              index === current
                ? "border-blue-500 bg-blue-50 dark:bg-blue-950/40"
                : "border-transparent hover:border-neutral-300 dark:hover:border-neutral-700"
            }`}
          >
            {/* An empty `src` makes a browser re-request the page itself, so a version
                with no link yet renders a placeholder instead. */}
            {urls.get(version.sha256) !== undefined ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={urls.get(version.sha256)}
                alt=""
                className="h-16 w-full rounded object-cover"
              />
            ) : (
              <span className="h-16 w-full rounded bg-neutral-200 dark:bg-neutral-800" />
            )}
            <span className="w-full truncate text-[11px]">{version.label}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
