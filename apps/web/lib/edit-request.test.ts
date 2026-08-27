/**
 * The rules the form uses to decide what to ask for and what to send.
 *
 * These mirror `EditSpec` in `packages/core`, which is the authority. Testing them here
 * is not duplication: the point of the mirror is that the button greys out instead of
 * the user being handed a 422, so what is asserted is that the two agree.
 */

import { describe, expect, it } from "vitest";
import { OPERATIONS, buildJob, groundable, ready, specFor, type Draft } from "./edit-request";

const draft = (over: Partial<Draft> = {}): Draft => ({
  op: "remove",
  imageSha256: "a".repeat(64),
  target: "the car",
  content: "",
  colour: "#2ea043",
  ...over,
});

describe("what each operation needs", () => {
  it("refuses a remove with no phrase, the way EditSpec does", () => {
    expect(ready(draft({ target: "  " }))).toBe(false);
  });

  it("refuses a generative operation with nothing to put there", () => {
    expect(ready(draft({ op: "replace", content: "" }))).toBe(false);
    expect(ready(draft({ op: "replace", content: "a sheep" }))).toBe(true);
  });

  it("lets background run with no phrase at all", () => {
    // It floods inward from the border, so it needs no region.
    expect(ready(draft({ op: "background", target: "", content: "" }))).toBe(true);
  });

  it("refuses a backdrop whose colour is not a colour", () => {
    expect(ready(draft({ op: "background", target: "", colour: "cornflower" }))).toBe(false);
  });

  it("lets upscale run with neither", () => {
    expect(ready(draft({ op: "upscale", target: "", content: "" }))).toBe(true);
  });

  it("refuses anything before an image exists", () => {
    expect(ready(draft({ imageSha256: "" }))).toBe(false);
  });
});

describe("when to ground", () => {
  it("grounds a phrase for the operations that act on a region", () => {
    expect(groundable("remove", "the car")).toBe(true);
    expect(groundable("replace", "the horse")).toBe(true);
  });

  it("grounds an optional phrase too", () => {
    // Background accepts a subject as the fallback for a border that is not a uniform
    // backdrop, and a phrase that is going to be used deserves the same confirmation.
    expect(groundable("background", "the person")).toBe(true);
    expect(groundable("background", "")).toBe(false);
  });

  it("never grounds for an operation with no region", () => {
    expect(groundable("upscale", "anything")).toBe(false);
  });
});

describe("building the request", () => {
  const mask = { width: 4, height: 3, counts: [4, 1, 7] };

  it("carries the region the user approved", () => {
    expect(buildJob(draft(), mask).mask).toEqual(mask);
  });

  it("calls a grounded region 'text', not 'brush'", () => {
    // The distinction is load-bearing downstream: the worker does not second-guess a
    // brushed mask, and this one came from a model.
    expect(buildJob(draft(), mask).mask_source).toBe("text");
  });

  it("calls it 'whole' when no phrase was given", () => {
    const request = buildJob(draft({ op: "background", target: "", content: "" }), null);
    expect(request.mask_source).toBe("whole");
    expect(request.target).toBeUndefined();
  });

  it("sends the backdrop colour as an exact value, not a description", () => {
    // TD-020: "change the background to blue" returned green and reported success,
    // because nothing downstream read the description.
    const request = buildJob(draft({ op: "background", target: "", colour: "#3366FF" }), null);
    expect(request.colour).toBe("#3366ff");
  });

  it("does not attach a colour to an operation that does not paint one", () => {
    expect(buildJob(draft({ op: "remove" }), null).colour).toBeUndefined();
  });

  it("does not label a grounded region 'whole' just because the phrase was optional", () => {
    // The incoherence this guards against: background does not *require* a target, so
    // keying the source off the requirement would attach a mask and call it the whole
    // image.
    const request = buildJob(draft({ op: "background", target: "the person", content: "green" }), mask);
    expect(request.mask_source).toBe("text");
    expect(request.mask).toEqual(mask);
  });

  it("trims what the user typed rather than sending their whitespace to a model", () => {
    const request = buildJob(draft({ target: "  the car  ", content: " " }), null);
    expect(request.target).toBe("the car");
    expect(request.content).toBeUndefined();
  });

  it("omits the mask when there is none, leaving the worker to ground the phrase", () => {
    expect(buildJob(draft(), null).mask).toBeUndefined();
  });
});

describe("the operation list", () => {
  it("offers every operation the gateway implements and none it does not", () => {
    // `/capabilities` advertises these five; restyle and retouch are explicitly
    // unsupported, and offering one would be a lie the user acts on.
    expect(OPERATIONS.map((o) => o.op).sort()).toEqual(
      ["add", "background", "remove", "replace", "upscale"].sort(),
    );
  });

  it("gives every operation a hint for the fields it actually shows", () => {
    for (const spec of OPERATIONS) {
      if (spec.acceptsTarget) expect(spec.targetHint.length).toBeGreaterThan(0);
      if (spec.needsContent && !spec.picksColour) {
        expect(spec.contentHint.length).toBeGreaterThan(0);
      }
    }
  });

  it("raises on an operation it does not know, rather than silently doing nothing", () => {
    expect(() => specFor("restyle" as never)).toThrow(/unknown operation/);
  });
});
