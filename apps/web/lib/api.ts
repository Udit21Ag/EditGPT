/**
 * Calling the gateway as a signed-in user.
 *
 * Every `/v1` endpoint requires a session, so every call needs a Clerk token on it. The
 * token is fetched per request rather than held in a module: Clerk rotates short-lived
 * session tokens, and a cached one is a request that starts failing after a minute.
 *
 * `getToken` is passed in rather than imported. In a React component it comes from
 * `useAuth()`, on the server from `auth()` — a module-level import would bind this file
 * to one of those and make it untestable in either.
 */

import type { MaskPayload } from "./rle";

export type { MaskPayload };

export type GetToken = () => Promise<string | null>;

/**
 * Where the gateway lives.
 *
 * `||`, not `??`. An unset variable is `undefined`, but a variable present-and-empty in
 * `.env` — which is exactly what a half-filled template produces — is `""`, and `??`
 * happily accepts that. The result is a relative URL, so every call quietly goes to the
 * Next.js server instead of the gateway and 404s with no clue why.
 */
export const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function authorized(getToken: GetToken, init: RequestInit = {}): Promise<RequestInit> {
  const token = await getToken();
  if (!token) {
    // Fail here rather than sending an unauthenticated request: the gateway would answer
    // 401 and the user would see "unauthorized" when the real cause is a lapsed session.
    throw new ApiError(401, "not signed in");
  }
  const headers = new Headers(init.headers);
  headers.set("authorization", `Bearer ${token}`);
  return { ...init, headers };
}

async function decode(response: Response): Promise<never> {
  let detail = response.statusText;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    // A non-JSON error body is still an error; the status carries the meaning.
  }
  throw new ApiError(response.status, detail);
}

export interface UploadedImage {
  sha256: string;
  width: number;
  height: number;
  content_type: string;
  megapixels: number;
}

export interface JobStep {
  state: string;
  at: string;
  detail: string;
  progress: number;
}

export interface Job {
  id: string;
  state: string;
  progress: number;
  op: string;
  result_sha256: string | null;
  error: string | null;
  steps: JobStep[];
}

/** One region a phrase might have meant. `box` is fractions of the image, not pixels. */
export interface Candidate {
  box: [number, number, number, number];
  score: number;
  mask: MaskPayload;
  label: string;
}

export interface Grounding {
  candidates: Candidate[];
  /** Whether the client should ask before editing. See ADR-0003. */
  ambiguous: boolean;
  /** How far the best candidate beat the runner-up. Zero with fewer than two. */
  margin: number;
}

/**
 * Ask what a phrase refers to, without editing anything.
 *
 * Deliberately its own call rather than part of creating a job. Grounding is cheap and
 * reversible and an erase is neither, so the user can be shown what is about to change —
 * and when the answer is shaky, the difference between one extra click and erasing the
 * wrong object.
 *
 * Each candidate carries its mask, at the resolution grounding ran at rather than the
 * upload's. Send the chosen one straight back on the job: the gateway takes a mask at any
 * size that matches the image's shape, so there is nothing to rescale here.
 */
export async function groundPhrase(
  getToken: GetToken,
  imageSha256: string,
  target: string,
): Promise<Grounding> {
  const response = await fetch(
    `${GATEWAY_URL}/v1/masks`,
    await authorized(getToken, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ image_sha256: imageSha256, target }),
    }),
  );
  return response.ok ? response.json() : decode(response);
}

export async function uploadImage(getToken: GetToken, file: File): Promise<UploadedImage> {
  const body = new FormData();
  body.append("file", file);
  // No content-type header: the browser must set the multipart boundary itself, and
  // setting it by hand produces a body the server cannot parse.
  const response = await fetch(
    `${GATEWAY_URL}/v1/images`,
    await authorized(getToken, { method: "POST", body }),
  );
  return response.ok ? response.json() : decode(response);
}

export interface CreateJob {
  op: string;
  image_sha256: string;
  target?: string;
  content?: string;
  mask_source?: string;
  mask?: MaskPayload;
  editor?: string;
}

export async function createJob(
  getToken: GetToken,
  request: CreateJob,
  idempotencyKey: string = crypto.randomUUID(),
): Promise<Job> {
  const response = await fetch(
    `${GATEWAY_URL}/v1/jobs`,
    await authorized(getToken, {
      method: "POST",
      headers: { "content-type": "application/json", "idempotency-key": idempotencyKey },
      body: JSON.stringify(request),
    }),
  );
  return response.ok ? response.json() : decode(response);
}

export async function getJob(getToken: GetToken, id: string): Promise<Job> {
  const response = await fetch(`${GATEWAY_URL}/v1/jobs/${id}`, await authorized(getToken));
  return response.ok ? response.json() : decode(response);
}

export async function cancelJob(getToken: GetToken, id: string): Promise<Job> {
  const response = await fetch(
    `${GATEWAY_URL}/v1/jobs/${id}/cancel`,
    await authorized(getToken, { method: "POST" }),
  );
  return response.ok ? response.json() : decode(response);
}

/**
 * A blob URL for a stored image.
 *
 * Fetched rather than pointed at with `<img src>`, because the endpoint needs an
 * `Authorization` header and a plain `<img>` cannot send one. The caller owns the
 * returned URL and must `URL.revokeObjectURL` it, or every viewed image stays in memory
 * for the life of the tab.
 *
 * The longer-term answer is a short-lived signed URL, which `<img>` *can* use — planned
 * for Phase 9 alongside object expiry.
 */
export async function imageObjectUrl(getToken: GetToken, digest: string): Promise<string> {
  const response = await fetch(`${GATEWAY_URL}/v1/images/${digest}`, await authorized(getToken));
  if (!response.ok) await decode(response);
  return URL.createObjectURL(await response.blob());
}

/**
 * Follow a job's progress.
 *
 * `EventSource` cannot carry an `Authorization` header, so this reads the SSE body off a
 * `fetch` stream and parses the frames. Returns a cancel function; call it when the
 * component unmounts or the connection outlives the job.
 */
export function streamJob(
  getToken: GetToken,
  id: string,
  onEvent: (event: { state: string; progress: number; detail: string; terminal: boolean }) => void,
): () => void {
  const controller = new AbortController();

  void (async () => {
    const response = await fetch(
      `${GATEWAY_URL}/v1/jobs/${id}/events`,
      await authorized(getToken, { signal: controller.signal }),
    );
    if (!response.ok || !response.body) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Frames are separated by a blank line; a partial frame stays in the buffer.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const data = frame
          .split("\n")
          .find((line) => line.startsWith("data: "))
          ?.slice(6);
        if (data) onEvent(JSON.parse(data));
      }
    }
  })().catch(() => {
    // An aborted stream is the normal way this ends. A real failure surfaces through the
    // job's own state, which the caller is polling or will fetch on completion.
  });

  return () => controller.abort();
}
