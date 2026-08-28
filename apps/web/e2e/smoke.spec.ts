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

test("the gateway is reachable and reports itself ready", async ({ request }) => {
  const health = await request.get(`${GATEWAY}/ready`);
  expect(health.ok(), `${GATEWAY}/ready answered ${health.status()}`).toBe(true);

  const body = await health.json();
  expect(body.status, `degraded: ${JSON.stringify(body.degraded)}`).toBe("ready");
  expect(body.degraded).toEqual([]);
});

test("the gateway refuses an unauthenticated edit", async ({ request }) => {
  // The fail-closed check. A 401 here is the whole reason the flow needs Clerk at all.
  const response = await request.post(`${GATEWAY}/v1/jobs`, {
    data: { op: "remove", image_sha256: "a".repeat(64), target: "the car" },
  });
  expect(response.status()).toBe(401);
});
