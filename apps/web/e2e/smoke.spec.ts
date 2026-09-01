/**
 * The deployment answers, and the pieces can see each other.
 *
 * Runs with no credentials, on every checkout, because everything it asserts is true
 * before anyone signs in — and because the two failures it would have caught were both
 * of exactly this kind. The site did not build for as long as Clerk had been installed
 * (Core 3 replaced the control components with stubs that throw at *render*, which lint,
 * typecheck and unit tests all pass through), and then did not run, because Next reads
 * `.env` from `apps/web` while the project keeps one at the repository root.
 */

import { expect, test } from "@playwright/test";

const GATEWAY = process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

test("the landing page renders and offers a way in", async ({ page }) => {
  const problems: string[] = [];
  page.on("pageerror", (error) => problems.push(String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(message.text());
  });

  const response = await page.goto("/");
  expect(response?.status(), "the page did not serve").toBe(200);

  await expect(page.getByRole("heading", { name: "EditGPT" })).toBeVisible();
  await expect(page.getByRole("button", { name: /sign in/i }).first()).toBeVisible();
  expect(problems, `console and page errors: ${problems.join(" | ")}`).toEqual([]);
});

test("the workspace is behind a session, not merely hidden", async ({ page }) => {
  // A signed-out visitor must not be able to reach the editor by any route the page
  // renders. `<Show when="signed-in">` decides this on the server, so the markup should
  // not contain it at all rather than hiding it with CSS.
  await page.goto("/");
  await expect(page.getByRole("group", { name: "What to do" })).toHaveCount(0);
  await expect(page.getByText(/choose a picture/i)).toHaveCount(0);
});

/**
 * Degradations the gateway is allowed to report, each because a deployment may legitimately
 * lack that piece. Anything outside this set is new and worth failing over — which is the
 * assertion, rather than "nothing is degraded": CI runs without Clerk keys on purpose, and
 * a check that only passes on a fully configured stack cannot run where it is needed most.
 */
const EXPLICABLE = [
  "no authentication: every request acts as the shared account",
  "no redis: progress streaming and rate limiting are off",
  "no queue: jobs are accepted but never executed",
  "jobs are in memory and will not survive a restart",
  "image links are signed with a per-process key: set EDITGPT_URL_SIGNING_KEY",
  "cors still allows localhost: set EDITGPT_CORS_ORIGINS",
  "no planner model: an instruction must name the operation plainly (set GEMINI_API_KEY)",
];

// Adding a line here is meant to be a deliberate act. The list caught its own first new
// entry — signed links arrived, `/ready` grew a degradation, and this failed until
// somebody acknowledged it, which is exactly the point.

test("the gateway is reachable and reports nothing unexpected", async ({ request }) => {
  const health = await request.get(`${GATEWAY}/ready`);
  expect(health.ok(), `${GATEWAY}/ready answered ${health.status()}`).toBe(true);

  const body = await health.json();
  expect(["ready", "degraded"]).toContain(body.status);

  const surprises = (body.degraded as string[]).filter((d) => !EXPLICABLE.includes(d));
  expect(surprises, `unexplained degradation: ${JSON.stringify(surprises)}`).toEqual([]);
});

test("the gateway never claims an authentication it is not enforcing", async ({ request }) => {
  // The property worth asserting in every configuration, and the one that would actually
  // hurt: an open API that reports itself protected. Asserting a flat 401 instead would
  // only hold on a fully configured stack, and would pass vacuously everywhere else.
  const body = await (await request.get(`${GATEWAY}/ready`)).json();
  const enforcing = !(body.degraded as string[]).some((d) => d.startsWith("no authentication"));

  const response = await request.post(`${GATEWAY}/v1/jobs`, {
    data: { op: "remove", image_sha256: "a".repeat(64), target: "the car" },
  });

  if (enforcing) {
    expect(response.status(), "auth is configured, so this must be refused").toBe(401);
  } else {
    expect(
      response.status(),
      "the gateway declares no authentication, so it must not answer 401 either",
    ).not.toBe(401);
  }
});
