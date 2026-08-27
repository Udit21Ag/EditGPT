/**
 * The chooser, asserted on what a person can actually do with it.
 *
 * No canvas here: jsdom has no 2-D context, and the thumbnails are the one part of this
 * component with no decisions in it — `lib/rle.test.ts` covers the pixels. What is worth
 * testing is the part that decides whether the 0.516 -> 0.832 in ADR-0003 is reachable:
 * that every candidate is offered, in rank order, selectable by touch and by keyboard,
 * and that a user who recognises none of them has somewhere to go.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CandidatePicker, REJECTED } from "./CandidatePicker";
import type { Candidate } from "@/lib/api";

/** A 4x3 mask with a single set pixel, which is enough to frame a crop around. */
function candidate(score: number, label = ""): Candidate {
  return {
    box: [0.1, 0.1, 0.4, 0.4],
    score,
    mask: { width: 4, height: 3, counts: [4, 1, 7] },
    label,
  };
}

function show(candidates: Candidate[], selected = 0, onSelect = vi.fn()) {
  render(
    <CandidatePicker
      candidates={candidates}
      imageUrl="blob:test"
      aspect={4 / 3}
      selected={selected}
      onSelect={onSelect}
      phrase="the zebra"
    />,
  );
  return onSelect;
}

function options() {
  return screen.getAllByRole("radio");
}

describe("offering the candidates", () => {
  it("shows every candidate the gateway ranked, plus a way out", () => {
    // The whole premise of ADR-0003: the right answer is in the top five 83.2% of the
    // time, and every one of them was already computed and thrown away.
    show([candidate(0.9), candidate(0.6), candidate(0.5), candidate(0.4), candidate(0.3)]);
    expect(options()).toHaveLength(6);
    expect(screen.getByText("None of these")).toBeDefined();
  });

  it("keeps the gateway's ranking rather than re-sorting", () => {
    show([candidate(0.9, "zebra on the left"), candidate(0.6, "zebra on the right")]);
    const labels = options().map((option) => option.textContent);
    expect(labels[0]).toContain("zebra on the left");
    expect(labels[1]).toContain("zebra on the right");
  });

  it("marks the first as the best match, so answering is still the default", () => {
    show([candidate(0.9), candidate(0.6)]);
    expect(screen.getByText("Best match")).toBeDefined();
  });

  it("names an unlabelled option by position rather than leaving it blank", () => {
    show([candidate(0.9), candidate(0.6)]);
    expect(screen.getByText("Option 2")).toBeDefined();
  });

  it("quotes the phrase back so the user can see what was searched for", () => {
    show([candidate(0.9)]);
    expect(screen.getByRole("radiogroup").getAttribute("aria-label")).toContain("the zebra");
  });
});

describe("choosing", () => {
  it("reports the index that was tapped", () => {
    const onSelect = show([candidate(0.9), candidate(0.6), candidate(0.5)]);
    fireEvent.click(options()[2]!);
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  it("marks exactly one option as chosen", () => {
    show([candidate(0.9), candidate(0.6)], 1);
    const checked = options().filter((o) => o.getAttribute("aria-checked") === "true");
    expect(checked).toHaveLength(1);
  });

  it("reports a rejection distinctly from a choice", () => {
    // Not the same as picking nothing: the right region is missing one time in six, and
    // a chooser with no exit turns those into a wrong edit.
    const onSelect = show([candidate(0.9), candidate(0.6)]);
    fireEvent.click(screen.getByText("None of these"));
    expect(onSelect).toHaveBeenCalledWith(REJECTED);
  });
});

describe("keyboard", () => {
  it("moves with the arrow keys", () => {
    const onSelect = show([candidate(0.9), candidate(0.6), candidate(0.5)], 0);
    fireEvent.keyDown(screen.getByRole("radiogroup"), { key: "ArrowRight" });
    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it("wraps past the last option onto the way out, so nothing is unreachable", () => {
    const onSelect = show([candidate(0.9), candidate(0.6)], 1);
    fireEvent.keyDown(screen.getByRole("radiogroup"), { key: "ArrowRight" });
    expect(onSelect).toHaveBeenCalledWith(REJECTED);
  });

  it("wraps backwards from the first option too", () => {
    const onSelect = show([candidate(0.9), candidate(0.6)], 0);
    fireEvent.keyDown(screen.getByRole("radiogroup"), { key: "ArrowLeft" });
    expect(onSelect).toHaveBeenCalledWith(REJECTED);
  });

  it("selects directly by number", () => {
    const onSelect = show([candidate(0.9), candidate(0.6), candidate(0.5)], 0);
    fireEvent.keyDown(screen.getByRole("radiogroup"), { key: "3" });
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  it("ignores a number with no option behind it", () => {
    const onSelect = show([candidate(0.9), candidate(0.6)], 0);
    fireEvent.keyDown(screen.getByRole("radiogroup"), { key: "7" });
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("puts the tab stop on the chosen option, not on all of them", () => {
    // Roving tabindex: a chooser with six tab stops is six presses to get past.
    show([candidate(0.9), candidate(0.6), candidate(0.5)], 1);
    const stops = options().filter((o) => o.getAttribute("tabindex") === "0");
    expect(stops).toHaveLength(1);
    expect(stops[0]!.getAttribute("aria-checked")).toBe("true");
  });
});

describe("degrading", () => {
  it("renders without a canvas context, which is all jsdom and some browsers have", () => {
    expect(() => show([candidate(0.9)])).not.toThrow();
    expect(options()).toHaveLength(2);
  });
});
