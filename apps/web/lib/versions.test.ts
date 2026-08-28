/**
 * The session's chain of results.
 *
 * The behaviour worth pinning is what happens after stepping back: recording a new result
 * from an earlier point drops what came after it, because a branch nothing in the
 * interface can reach is a promise to keep something the user cannot see.
 */

import { describe, expect, it } from "vitest";
import { ORIGINAL, describe as label, record, type Version } from "./versions";

const version = (sha: string, text = "edit"): Version => ({
  sha256: sha.repeat(64).slice(0, 64),
  width: 100,
  height: 80,
  label: text,
});

const original = version("a", ORIGINAL);

describe("recording a result", () => {
  it("appends to the end when nothing was stepped back to", () => {
    const history = record([original], 0, version("b"));
    expect(history.map((v) => v.label)).toEqual([ORIGINAL, "edit"]);
  });

  it("drops what came after the version being edited", () => {
    const three = record(record([original], 0, version("b", "first")), 1, version("c", "second"));
    expect(three).toHaveLength(3);

    const branched = record(three, 1, version("d", "other"));
    expect(branched.map((v) => v.label)).toEqual([ORIGINAL, "first", "other"]);
  });

  it("keeps the original when branching from it", () => {
    const three = record(record([original], 0, version("b")), 1, version("c"));
    expect(record(three, 0, version("d", "fresh")).map((v) => v.label)).toEqual([
      ORIGINAL,
      "fresh",
    ]);
  });

  it("treats a negative index as the start rather than slicing from the end", () => {
    // `slice(0, -1 + 1)` is `slice(0, 0)` only by accident; a stray negative from an empty
    // selection would otherwise silently discard the original.
    expect(record([original], -1, version("b"))).toHaveLength(2);
  });
});

describe("describing an edit", () => {
  it("says what was done, in the user's own words where there are any", () => {
    expect(label("remove", "the car", "")).toBe("removed the car");
    expect(label("replace", "the horse", "a sheep")).toBe("the horse → a sheep");
    expect(label("add", "", "a moustache")).toBe("added a moustache");
  });

  it("still says something when there were no words", () => {
    // A brushed or tapped region has no phrase, and "removed" alone is not a label.
    expect(label("remove", "", "")).toBe("removed a region");
    expect(label("upscale", "", "")).toBe("upscaled");
    expect(label("background", "", "")).toBe("new background");
  });
});
