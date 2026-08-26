import { describe, expect, it } from "vitest";

import {
  EMPTY_HISTORY,
  MAX_DEPTH,
  type Stroke,
  canRedo,
  canUndo,
  clear,
  hasMask,
  push,
  redo,
  undo,
} from "./mask-history";

const stroke = (mode: Stroke["mode"] = "paint", n = 3): Stroke => ({
  mode,
  points: Array.from({ length: n }, (_, i) => [i / 10, i / 10] as const),
  width: 0.05,
});

describe("mask history", () => {
  it("starts empty and offers nothing to undo or redo", () => {
    expect(canUndo(EMPTY_HISTORY)).toBe(false);
    expect(canRedo(EMPTY_HISTORY)).toBe(false);
    expect(hasMask(EMPTY_HISTORY)).toBe(false);
  });

  it("records a stroke", () => {
    const history = push(EMPTY_HISTORY, stroke());
    expect(history.strokes).toHaveLength(1);
    expect(canUndo(history)).toBe(true);
  });

  it("ignores an empty stroke rather than polluting the history", () => {
    const empty: Stroke = { mode: "paint", points: [], width: 0.05 };
    expect(push(EMPTY_HISTORY, empty)).toBe(EMPTY_HISTORY);
  });

  it("undoes and redoes back to the same state", () => {
    const one = push(EMPTY_HISTORY, stroke());
    const two = push(one, stroke("erase"));
    const undone = undo(two);

    expect(undone.strokes).toHaveLength(1);
    expect(canRedo(undone)).toBe(true);
    expect(redo(undone).strokes).toEqual(two.strokes);
  });

  it("discards the redo branch once a new stroke is drawn", () => {
    const two = push(push(EMPTY_HISTORY, stroke()), stroke());
    const branched = push(undo(two), stroke("erase"));

    expect(canRedo(branched)).toBe(false);
    expect(branched.strokes).toHaveLength(2);
  });

  it("undo on an empty history is a no-op, not a crash", () => {
    expect(undo(EMPTY_HISTORY)).toBe(EMPTY_HISTORY);
    expect(redo(EMPTY_HISTORY)).toBe(EMPTY_HISTORY);
  });

  it("caps depth so a long session cannot grow without bound", () => {
    let history = EMPTY_HISTORY;
    for (let i = 0; i < MAX_DEPTH + 20; i += 1) history = push(history, stroke());
    expect(history.strokes).toHaveLength(MAX_DEPTH);
  });

  it("treats an all-erase history as having no mask", () => {
    const history = push(push(EMPTY_HISTORY, stroke("erase")), stroke("erase"));
    expect(hasMask(history)).toBe(false);
  });

  it("never mutates the history it was given", () => {
    const original = push(EMPTY_HISTORY, stroke());
    const snapshot = JSON.stringify(original);
    push(original, stroke("erase"));
    undo(original);
    expect(JSON.stringify(original)).toBe(snapshot);
  });

  it("clears back to empty", () => {
    expect(clear()).toEqual(EMPTY_HISTORY);
  });
});
