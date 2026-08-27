import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createJob, imageObjectUrl, streamJob, uploadImage } from "./api";

const token = async () => "session-token";
const noToken = async () => null;

afterEach(() => vi.unstubAllGlobals());

type FetchMock = ReturnType<typeof vi.fn<typeof fetch>>;

function respondWith(body: unknown, init: ResponseInit = { status: 200 }): FetchMock {
  const fetchMock = vi.fn<typeof fetch>(
    async () =>
      new Response(typeof body === "string" ? body : JSON.stringify(body), {
        headers: { "content-type": "application/json" },
        ...init,
      }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The `RequestInit` of the first call, asserted rather than assumed. */
function sentHeaders(fetchMock: FetchMock): Headers {
  const call = fetchMock.mock.calls[0];
  expect(call, "fetch was never called").toBeDefined();
  return new Headers(call![1]?.headers);
}

describe("attaching the session", () => {
  it("sends the token as a bearer credential", async () => {
    const fetchMock = respondWith({ id: "job-1" });
    await createJob(token, { op: "remove", image_sha256: "a".repeat(64) });

    expect(sentHeaders(fetchMock).get("authorization")).toBe("Bearer session-token");
  });

  it("fails before sending when there is no session", async () => {
    const fetchMock = respondWith({});
    await expect(uploadImage(noToken, new File(["x"], "a.png"))).rejects.toThrow(ApiError);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("does not set a content-type on an upload", async () => {
    // The browser must choose the multipart boundary; setting it by hand produces a body
    // the server cannot parse.
    const fetchMock = respondWith({ sha256: "a".repeat(64) });
    await uploadImage(token, new File(["x"], "a.png", { type: "image/png" }));

    expect(sentHeaders(fetchMock).get("content-type")).toBeNull();
  });

  it("sends an idempotency key so a retry is not a second job", async () => {
    const fetchMock = respondWith({ id: "job-1" });
    await createJob(token, { op: "remove", image_sha256: "a".repeat(64) }, "key-123");

    expect(sentHeaders(fetchMock).get("idempotency-key")).toBe("key-123");
  });
});

describe("errors", () => {
  it("surfaces the gateway's reason rather than the status text", async () => {
    respondWith(
      { detail: "remove needs either a `target` phrase or an explicit mask" },
      {
        status: 422,
      },
    );
    await expect(createJob(token, { op: "remove", image_sha256: "a".repeat(64) })).rejects.toThrow(
      /target/,
    );
  });

  it("still throws when the body is not JSON", async () => {
    respondWith("<html>gateway down</html>", { status: 502 });
    await expect(
      createJob(token, { op: "remove", image_sha256: "a".repeat(64) }),
    ).rejects.toMatchObject({ status: 502 });
  });
});

describe("images", () => {
  it("fetches with the header and hands back an object url", async () => {
    // `<img src>` cannot send an Authorization header, which is the whole reason this
    // goes through fetch and a blob.
    respondWith("bytes");
    vi.stubGlobal("URL", { ...URL, createObjectURL: () => "blob:fake" });

    await expect(imageObjectUrl(token, "a".repeat(64))).resolves.toBe("blob:fake");
  });
});

describe("progress stream", () => {
  it("parses sse frames and ignores a partial one", async () => {
    const frames =
      'event: progress\ndata: {"state":"running","progress":0.3,"detail":"","terminal":false}\n\n' +
      'event: progress\ndata: {"state":"done","progress":1,"detail":"","terminal":true}\n\n' +
      'event: progress\ndata: {"state":"trunc';

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(new TextEncoder().encode(frames))),
    );

    const seen: string[] = [];
    streamJob(token, "job-1", (event) => seen.push(event.state));
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(seen).toEqual(["running", "done"]);
  });
});

describe("gateway url", () => {
  it("falls back when the variable is present but empty", async () => {
    // A half-filled `.env` gives `""`, not `undefined`. `??` would accept it and every
    // call would go to the Next server as a relative URL, 404ing with no clue why.
    //
    // `resetModules` rather than a query-string import: the constant is evaluated once at
    // module load, so the module has to be re-evaluated to see the stubbed value — and a
    // `./api?empty` specifier, while Vitest resolves it, is not a module `tsc` can find.
    vi.stubEnv("NEXT_PUBLIC_GATEWAY_URL", "");
    vi.resetModules();
    try {
      const { GATEWAY_URL } = await import("./api");
      expect(GATEWAY_URL).toBe("http://localhost:8000");
    } finally {
      vi.unstubAllEnvs();
      vi.resetModules();
    }
  });
});
