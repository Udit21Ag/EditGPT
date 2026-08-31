/**
 * The whole client flow, in a real Chromium, with the gateway replaced at the network
 * boundary and nothing else.
 *
 * This is the test the project did not have: every piece of the flow was covered and the
 * *flow* was not, so a picker that never received its candidates, or a job whose result
 * never reached the screen, would have passed everything. `harness/testing.md` calls for
 * replacing the transport and never the logic — `fetch` is the transport here, and the
 * components, canvases, pointer events and state machine are all the real ones.
 *
 * What it deliberately does not cover: Clerk, and the models behind the gateway. Those are
 * `e2e/` and the golden set respectively.
 */

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Workspace } from "./Workspace";

const SHA = "a".repeat(64);
const RESULT = "b".repeat(64);
const token = async () => "test-token";

function png(colour = "#3366ff"): string {
  const canvas = document.createElement("canvas");
  canvas.width = 80;
  canvas.height = 60;
  const c = canvas.getContext("2d")!;
  c.fillStyle = colour;
  c.fillRect(0, 0, 80, 60);
  return canvas.toDataURL("image/png");
}

async function blob(): Promise<Blob> {
  return await (await fetch(png("#20c060"))).blob();
}

/** Column-major RLE for a rectangle over an 80x60 image. */
function rect(x0: number, y0: number, x1: number, y1: number) {
  const counts: number[] = [];
  let run = 0;
  let previous = 0;
  for (let x = 0; x < 80; x += 1) {
    for (let y = 0; y < 60; y += 1) {
      const bit = x >= x0 && x < x1 && y >= y0 && y < y1 ? 1 : 0;
      if (bit === previous) run += 1;
      else { counts.push(run); run = 1; previous = bit; }
    }
  }
  counts.push(run);
  if (counts.length % 2 === 0) counts.push(0);
  return { width: 80, height: 60, counts };
}

interface Server {
  /** Every request body the workspace sent, by path. */
  sent: { path: string; body: unknown }[];
  candidates: number;
  ambiguous: boolean;
}

/** A gateway that answers, recording what it was asked. */
function serve(over: Partial<Server> = {}) {
  const server: Server = { sent: [], candidates: 1, ambiguous: false, ...over };
  const real = globalThis.fetch;

  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    // Data URLs are how the fixtures make pictures; they are not the gateway.
    if (url.startsWith("data:") || url.startsWith("blob:")) return real(input, init);

    const body = init?.body === undefined ? undefined : init.body;
    const parsed =
      typeof body === "string" ? JSON.parse(body) : body instanceof FormData ? "form" : undefined;
    const path = new URL(url).pathname;
    server.sent.push({ path, body: parsed });

    const json = (value: unknown, status = 200) =>
      new Response(JSON.stringify(value), {
        status,
        headers: { "content-type": "application/json" },
      });

    if (path === "/v1/images" && init?.method === "POST") {
      return json(
        {
          sha256: SHA,
          width: 80,
          height: 60,
          content_type: "image/png",
          megapixels: 0.005,
          url: `/v1/images/${SHA}?expires=1&signature=x`,
          url_expires_at: 1,
        },
        201,
      );
    }
    if (path === "/v1/masks") {
      const boxes = [rect(10, 10, 30, 30), rect(50, 20, 70, 45)].slice(0, server.candidates);
      return json({
        candidates: boxes.map((mask, i) => ({
          box: [0.1, 0.1, 0.4, 0.5],
          score: 0.9 - i * 0.05,
          mask,
          label: "",
        })),
        ambiguous: server.ambiguous,
        margin: server.ambiguous ? 0.05 : 1,
      });
    }
    if (path === "/v1/jobs" && init?.method === "POST") {
      return json(
        { id: "job-1", state: "queued", progress: 0, op: "remove", result_sha256: null, result_url: "", error: null, steps: [] },
        202,
      );
    }
    if (path === "/v1/jobs/job-1/events") {
      // One frame, terminal, in the SSE shape `streamJob` parses.
      const frame = `data: ${JSON.stringify({ state: "done", progress: 1, detail: "erased", terminal: true })}\n\n`;
      return new Response(frame, { headers: { "content-type": "text/event-stream" } });
    }
    if (path === "/v1/jobs/job-1") {
      return json({
        id: "job-1", state: "done", progress: 1, op: "remove",
        result_sha256: RESULT,
        result_url: png("#20c060"),
        error: null,
        steps: [
          { state: "queued", at: "2026-08-29T10:00:00Z", detail: "", progress: 0 },
          { state: "running", at: "2026-08-29T10:00:02Z", detail: "migan", progress: 0.5 },
          { state: "done", at: "2026-08-29T10:00:09Z", detail: "erased", progress: 1 },
        ],
      });
    }
    if (path === `/v1/images/${RESULT}`) return new Response(await blob());
    return json({ detail: `unexpected ${path}` }, 500);
  });

  return server;
}

afterEach(() => vi.unstubAllGlobals());

async function upload() {
  render(<Workspace getToken={token} />);
  const file = new File([await (await fetch(png())).arrayBuffer()], "p.png", { type: "image/png" });
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
  await screen.findByText(/80×60/);
}

describe("the flow", () => {
  it("uploads, grounds, runs and shows the result", async () => {
    const server = serve();
    await upload();

    fireEvent.change(screen.getByLabelText("What to change"), { target: { value: "the car" } });
    fireEvent.click(screen.getByRole("button", { name: "Find it" }));
    await screen.findByText(/region that will change/i);

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("slider", { name: /wipe/i }, { timeout: 5000 });

    const paths = server.sent.map((s) => s.path);
    expect(paths).toEqual([
      "/v1/images",
      "/v1/masks",
      "/v1/jobs",
      "/v1/jobs/job-1/events",
      "/v1/jobs/job-1",
    ]);
  });

  it("never fetches the result image itself", async () => {
    // The signed link is the point: `<img src>` loads it, so the bytes never pass through
    // JavaScript and never sit in an object URL waiting to be revoked.
    const server = serve();
    await upload();
    fireEvent.change(screen.getByLabelText("What to change"), { target: { value: "the car" } });
    fireEvent.click(screen.getByRole("button", { name: "Find it" }));
    await screen.findByText(/region that will change/i);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("slider", { name: /wipe/i }, { timeout: 5000 });

    expect(server.sent.filter((s) => s.path.startsWith("/v1/images/"))).toEqual([]);
  });

  it("sends the grounded mask with the job rather than the phrase alone", async () => {
    // The point of the picker, asserted where it actually has to survive.
    const server = serve();
    await upload();
    fireEvent.change(screen.getByLabelText("What to change"), { target: { value: "the car" } });
    fireEvent.click(screen.getByRole("button", { name: "Find it" }));
    await screen.findByText(/region that will change/i);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("slider", { name: /wipe/i }, { timeout: 5000 });

    const job = server.sent.find((s) => s.path === "/v1/jobs")!.body as Record<string, unknown>;
    expect(job.mask).toBeDefined();
    expect(job.mask_source).toBe("text");
    expect(job.target).toBe("the car");
  });

  it("shows the chooser only when the gateway says the answer is shaky", async () => {
    serve({ candidates: 2, ambiguous: true });
    await upload();
    fireEvent.change(screen.getByLabelText("What to change"), { target: { value: "the zebra" } });
    fireEvent.click(screen.getByRole("button", { name: "Find it" }));

    await screen.findByRole("radiogroup", { name: /which/i });
    expect(screen.getAllByRole("radio").length).toBeGreaterThan(2);
  });

  it("does not put the chooser in front of a clear answer", async () => {
    serve({ candidates: 1, ambiguous: false });
    await upload();
    fireEvent.change(screen.getByLabelText("What to change"), { target: { value: "the car" } });
    fireEvent.click(screen.getByRole("button", { name: "Find it" }));

    await screen.findByText(/region that will change/i);
    expect(screen.queryByRole("radiogroup", { name: /which/i })).toBeNull();
  });

  it("shows what the job did, step by step", async () => {
    serve();
    await upload();
    fireEvent.change(screen.getByLabelText("What to change"), { target: { value: "the car" } });
    fireEvent.click(screen.getByRole("button", { name: "Find it" }));
    await screen.findByText(/region that will change/i);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    // Waiting on the *content*, not on the list appearing: a list shows as soon as the
    // first streamed event lands, and the job's own steps — which carry the pass detail
    // and the timestamps — replace it a moment later.
    await waitFor(
      () => {
        const timeline = screen.getByRole("list", { name: /what the job did/i });
        expect(timeline.textContent).toContain("migan");
      },
      { timeout: 5000 },
    );
  });

  it("records the result in the history once there is more than one version", async () => {
    serve();
    await upload();
    expect(screen.queryByRole("radiogroup", { name: /version history/i })).toBeNull();

    fireEvent.change(screen.getByLabelText("What to change"), { target: { value: "the car" } });
    fireEvent.click(screen.getByRole("button", { name: "Find it" }));
    await screen.findByText(/region that will change/i);
    fireEvent.click(screen.getByRole("button", { name: "Run" }));

    const history = await screen.findByRole("radiogroup", { name: /version history/i }, { timeout: 5000 });
    expect(history.querySelectorAll('[role="radio"]')).toHaveLength(2);
    expect(history.textContent).toContain("removed the car");
  });
});

describe("tapping instead of describing", () => {
  it("sends taps to the gateway and uses what comes back", async () => {
    const server = serve();
    await upload();

    fireEvent.click(screen.getByRole("button", { name: "Draw it" }));
    const canvas = document.querySelector("canvas") as HTMLCanvasElement;
    canvas.style.width = "300px";
    canvas.style.height = "225px";

    const box = canvas.getBoundingClientRect();
    const point = { clientX: box.left + box.width * 0.3, clientY: box.top + box.height * 0.3, pointerId: 1 };
    fireEvent.pointerDown(canvas, point);
    fireEvent.pointerUp(canvas, point);

    await waitFor(() => {
      const masks = server.sent.filter((s) => s.path === "/v1/masks");
      expect(masks).toHaveLength(1);
      expect((masks[0]!.body as Record<string, unknown>).points).toBeDefined();
    });

    fireEvent.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByRole("slider", { name: /wipe/i }, { timeout: 5000 });

    const job = server.sent.find((s) => s.path === "/v1/jobs")!.body as Record<string, unknown>;
    expect(job.mask_source).toBe("brush");
    expect(job.target).toBeUndefined();
  });
});

describe("when the gateway cannot be reached", () => {
  it("says so, instead of 'something went wrong'", async () => {
    // The message that hid a gateway with no CORS middleware. `fetch` rejects with a bare
    // TypeError when the request never left the browser: no status, no body, and the
    // reason only in a console nobody was reading.
    vi.stubGlobal("fetch", async (input: RequestInfo | URL) => {
      if (String(input).startsWith("data:")) return new Response(new Blob());
      throw new TypeError("Failed to fetch");
    });

    render(<Workspace getToken={token} />);
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [new File(["x"], "p.png", { type: "image/png" })] } });

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/could not reach the gateway/i);
    expect(alert.textContent).toContain("gateway.test");
  });
});
