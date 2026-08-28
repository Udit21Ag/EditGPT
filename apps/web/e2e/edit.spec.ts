/**
 * The headline flows, signed in, against the real stack.
 *
 * **Skips without credentials**, the same way `service`-marked Python tests skip without
 * Postgres: a fresh checkout stays green, and where the credentials exist it genuinely
 * runs. `harness/testing.md` has the reasoning — the two incidents that bought that rule
 * were both "verified by review, never actually applied".
 *
 * What it needs, and why each one:
 *
 * | Variable | Why |
 * | --- | --- |
 * | `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY` | already in `.env`; `clerkSetup` exchanges them for a Testing Token, which is what lets a script past bot protection |
 * | `CLERK_E2E_EMAIL`, `CLERK_E2E_PASSWORD` | a user to *be*. Clerk has no way to conjure a session without one, and creating one changes somebody's account, so it is asked for rather than assumed |
 *
 * The gateway and worker are assumed up: `make e2e` starts them.
 */

import { clerk, clerkSetup } from "@clerk/testing/playwright";
import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const EMAIL = process.env.CLERK_E2E_EMAIL;
const PASSWORD = process.env.CLERK_E2E_PASSWORD;
const PHOTO = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "..",
  "evals",
  "photos",
  "i2.jpg",
);

test.skip(
  EMAIL === undefined || PASSWORD === undefined,
  "set CLERK_E2E_EMAIL and CLERK_E2E_PASSWORD to run the signed-in flows",
);

test.beforeAll(async () => {
  await clerkSetup();
});

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await clerk.signIn({
    page,
    signInParams: { strategy: "password", identifier: EMAIL!, password: PASSWORD! },
  });
  await page.goto("/");
  await expect(page.getByText(/choose a picture/i)).toBeVisible();
});

async function upload(page: import("@playwright/test").Page) {
  await page.locator('input[type="file"]').first().setInputFiles(PHOTO);
  await expect(page.getByText(/669×446/)).toBeVisible({ timeout: 30_000 });
}

test("describe a region, confirm it, and get an edited picture back", async ({ page }) => {
  await upload(page);

  await page.getByLabel("What to change").fill("the woman");
  await page.getByRole("button", { name: "Find it" }).click();
  // Grounding runs the detector and the SAM encoder — seconds, not milliseconds.
  await expect(page.getByText(/region that will change/i)).toBeVisible({ timeout: 60_000 });

  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.getByRole("slider", { name: /wipe/i })).toBeVisible({ timeout: 120_000 });

  // The result is downloadable and the history now has something to go back to.
  await expect(page.getByRole("link", { name: "Download" })).toBeVisible();
  await expect(page.getByRole("radiogroup", { name: /version history/i })).toBeVisible();
});

test("tap a region and erase it without typing anything", async ({ page }) => {
  await upload(page);

  await page.getByRole("button", { name: "Draw it" }).click();
  const canvas = page.getByRole("img", { name: /paint the region/i });
  await expect(canvas).toBeVisible();

  // The dome, in the middle of the frame. SAM alone, so this is fast.
  const box = (await canvas.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.2);
  await expect(page.getByText(/looking at what you tapped/i)).toBeVisible();
  await expect(page.getByText(/looking at what you tapped/i)).toBeHidden({ timeout: 60_000 });

  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.getByRole("slider", { name: /wipe/i })).toBeVisible({ timeout: 120_000 });
});
