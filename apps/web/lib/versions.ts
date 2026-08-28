/**
 * The chain of results, and going back to one of them.
 *
 * Every finished job writes a new content-addressed image; nothing overwrites anything.
 * So a history is just the list of digests this session has produced, and "branch from
 * version" is choosing one as the input to the next edit — no server support, no undo
 * stack, and no way to lose work by editing the wrong thing.
 *
 * Kept in the client because it is a *session's* narrative rather than a fact about the
 * image. Two people editing the same upload have different histories and the stored
 * assets are identical; persisting it would be recording the difference between them.
 */

export interface Version {
  /** The stored image. The original upload is the first entry. */
  readonly sha256: string;
  readonly width: number;
  readonly height: number;
  /** What produced it — "the car, removed" — or "original" for the upload. */
  readonly label: string;
}

export const ORIGINAL = "original";

/**
 * Add a result, branching from `current` when the user has stepped back.
 *
 * Everything after the point being edited is dropped, the same way an editor's undo
 * history behaves: a branch the user can no longer reach through the interface is a
 * promise to keep something they cannot see.
 */
export function record(
  history: readonly Version[],
  current: number,
  version: Version,
): readonly Version[] {
  const kept = history.slice(0, Math.max(current, 0) + 1);
  return [...kept, version];
}

/** A short description of an edit, for the history strip. */
export function describe(op: string, target: string, content: string): string {
  const what = target.trim();
  const into = content.trim();
  if (op === "remove") return what.length > 0 ? `removed ${what}` : "removed a region";
  if (op === "replace") return what.length > 0 ? `${what} → ${into}` : `replaced with ${into}`;
  if (op === "add") return `added ${into}`;
  if (op === "background") return "new background";
  if (op === "upscale") return "upscaled";
  return op;
}
