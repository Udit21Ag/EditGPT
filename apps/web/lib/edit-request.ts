/**
 * What each operation needs before it can be run, and how a request is assembled.
 *
 * Pure, and separate from the component, for the same reason `mask-history.ts` is: this
 * is where the rules live, and rules are worth testing without a DOM. The rules
 * themselves are not invented here — `EditSpec` in `packages/core` already refuses a
 * remove with no target and a generative op with no content. This mirrors those so the
 * form can grey out a button instead of showing the user a 422.
 *
 * Mirroring means it can drift, so `test_capabilities_match_the_web_client` asserts the
 * operation list against what `/capabilities` advertises.
 */

import type { CreateJob, MaskPayload } from "./api";

export type Operation = "remove" | "add" | "replace" | "background" | "upscale";

export interface OperationSpec {
  readonly op: Operation;
  readonly label: string;
  /** Whether a phrase naming a region is required before the job can run. */
  readonly requiresTarget: boolean;
  /**
   * Whether a phrase naming a region is *useful* at all, which is not the same thing.
   * `background` floods inward from the border and needs no region, but accepts one as
   * the fallback for a border that is not a uniform backdrop (TD-005) — so it offers the
   * field, grounds what is typed there, and runs fine with it left empty.
   */
  readonly acceptsTarget: boolean;
  /** Whether a phrase describing what to put there is required. */
  readonly needsContent: boolean;
  /**
   * Whether what goes there is a *colour* rather than a description.
   *
   * `background` composites a flat backdrop rather than generating one (TD-005), so a
   * colour input says exactly what it will do. Asking for free text here was worse than
   * imprecise: until `EditSpec.colour` existed nothing read the answer, and typing
   * "a blue background" produced green and reported success (TD-020).
   */
  readonly picksColour: boolean;
  /** Placeholder for the target field, which differs a lot by operation. */
  readonly targetHint: string;
  readonly contentHint: string;
}

export const OPERATIONS: readonly OperationSpec[] = [
  {
    op: "remove",
    label: "Remove",
    picksColour: false,
    requiresTarget: true,
    acceptsTarget: true,
    needsContent: false,
    targetHint: "the car",
    contentHint: "",
  },
  {
    op: "replace",
    label: "Replace",
    picksColour: false,
    requiresTarget: true,
    acceptsTarget: true,
    needsContent: true,
    targetHint: "the horse",
    contentHint: "a white sheep grazing",
  },
  {
    op: "add",
    label: "Add",
    picksColour: false,
    requiresTarget: true,
    acceptsTarget: true,
    needsContent: true,
    targetHint: "the upper lip",
    contentHint: "a realistic moustache",
  },
  {
    // Flood-fills inward from the border, so it needs no region at all. Naming the
    // subject is the fallback for when the border is not a uniform backdrop (TD-005).
    op: "background",
    label: "Background",
    picksColour: true,
    requiresTarget: false,
    acceptsTarget: true,
    needsContent: true,
    targetHint: "the person (optional)",
    contentHint: "a solid green background",
  },
  {
    op: "upscale",
    label: "Upscale",
    picksColour: false,
    requiresTarget: false,
    acceptsTarget: false,
    needsContent: false,
    targetHint: "",
    contentHint: "",
  },
];

export function specFor(op: Operation): OperationSpec {
  const found = OPERATIONS.find((candidate) => candidate.op === op);
  if (found === undefined) throw new Error(`unknown operation ${op}`);
  return found;
}

/** Whether grounding is worth asking for: only a phrase can be grounded. */
export function groundable(op: Operation, target: string): boolean {
  return specFor(op).acceptsTarget && target.trim().length > 0;
}

/**
 * Where the region came from, which the server treats three different ways.
 *
 * Explicit rather than a nullable mask plus an inferred source: a grounded phrase, a
 * candidate the user picked and a region the user drew are three different claims, and
 * the worker acts on the difference — it will not second-guess a brushed mask, and it
 * grounds a phrase only when no mask arrives with it.
 */
export type Region =
  | { readonly kind: "phrase" }
  | { readonly kind: "chosen"; readonly mask: MaskPayload }
  | { readonly kind: "drawn"; readonly mask: MaskPayload };

export const PHRASE: Region = { kind: "phrase" };

export interface Draft {
  readonly op: Operation;
  readonly imageSha256: string;
  readonly target: string;
  readonly content: string;
  /** `#rrggbb`, for the operations that paint one. */
  readonly colour: string;
}

/** Whether the form is complete enough to send. Mirrors `EditSpec`'s own rules. */
export function ready(draft: Draft, region: Region = PHRASE): boolean {
  const spec = specFor(draft.op);
  if (draft.imageSha256.length === 0) return false;
  // A drawn region *is* the answer to "what to change", so it stands in for the phrase.
  // `EditSpec` says the same: remove needs either a target or an explicit mask.
  if (spec.requiresTarget && region.kind !== "drawn" && draft.target.trim().length === 0) {
    return false;
  }
  // A colour is a complete answer to "what goes there", so it satisfies the same rule.
  if (spec.needsContent && !spec.picksColour && draft.content.trim().length === 0) return false;
  if (spec.picksColour && !/^#[0-9a-fA-F]{6}$/.test(draft.colour)) return false;
  return true;
}

/**
 * The job to create, carrying the region the user approved.
 *
 * Sending the chosen mask rather than the phrase alone is the point of the picker twice
 * over. It is the only way the user's choice survives — re-grounding server-side would
 * discard it — and it saves the worker grounding a phrase that has already been grounded,
 * which is the detector and the SAM encoder, about two seconds and 2 GB.
 *
 * `mask_source` records where the region came from, and the difference is load-bearing:
 * `text` is a model's guess the user approved, `brush` is the user's own and the worker
 * will not second-guess it.
 */
export function buildJob(draft: Draft, region: Region = PHRASE): CreateJob {
  const target = draft.target.trim();
  const request: CreateJob = {
    op: draft.op,
    image_sha256: draft.imageSha256,
    // Otherwise keyed off whether a phrase was actually given, not off whether the
    // operation demands one. `background` accepts a target without requiring it, so
    // anything else would label a grounded region `whole` — a request contradicting
    // itself.
    mask_source: region.kind === "drawn" ? "brush" : target.length > 0 ? "text" : "whole",
  };
  // A drawn region answers for itself. Sending the phrase alongside would invite a
  // reading of the mask as something a model produced from those words.
  if (target.length > 0 && region.kind !== "drawn") request.target = target;
  if (draft.content.trim().length > 0) request.content = draft.content.trim();
  if (specFor(draft.op).picksColour) request.colour = draft.colour.toLowerCase();
  if (region.kind !== "phrase") request.mask = region.mask;
  return request;
}
