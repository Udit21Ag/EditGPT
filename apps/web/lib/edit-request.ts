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
  /** Placeholder for the target field, which differs a lot by operation. */
  readonly targetHint: string;
  readonly contentHint: string;
}

export const OPERATIONS: readonly OperationSpec[] = [
  {
    op: "remove",
    label: "Remove",
    requiresTarget: true,
    acceptsTarget: true,
    needsContent: false,
    targetHint: "the car",
    contentHint: "",
  },
  {
    op: "replace",
    label: "Replace",
    requiresTarget: true,
    acceptsTarget: true,
    needsContent: true,
    targetHint: "the horse",
    contentHint: "a white sheep grazing",
  },
  {
    op: "add",
    label: "Add",
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
    requiresTarget: false,
    acceptsTarget: true,
    needsContent: true,
    targetHint: "the person (optional)",
    contentHint: "a solid green background",
  },
  {
    op: "upscale",
    label: "Upscale",
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

export interface Draft {
  readonly op: Operation;
  readonly imageSha256: string;
  readonly target: string;
  readonly content: string;
}

/** Whether the form is complete enough to send. Mirrors `EditSpec`'s own rules. */
export function ready(draft: Draft): boolean {
  const spec = specFor(draft.op);
  if (draft.imageSha256.length === 0) return false;
  if (spec.requiresTarget && draft.target.trim().length === 0) return false;
  if (spec.needsContent && draft.content.trim().length === 0) return false;
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
 * `mask_source` stays `text` even with a mask attached, because that is where the region
 * came from. A brushed mask means something different downstream: the worker will not
 * second-guess it.
 */
export function buildJob(draft: Draft, mask: MaskPayload | null): CreateJob {
  const target = draft.target.trim();
  const request: CreateJob = {
    op: draft.op,
    image_sha256: draft.imageSha256,
    // Keyed off whether a phrase was actually given, not off whether the operation
    // demands one. `background` accepts a target without requiring it, so anything else
    // would label a grounded region `whole` — a request that contradicts itself.
    mask_source: target.length > 0 ? "text" : "whole",
  };
  if (target.length > 0) request.target = target;
  if (draft.content.trim().length > 0) request.content = draft.content.trim();
  if (mask !== null) request.mask = mask;
  return request;
}
