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
 * | `CLERK_E2E_EMAIL` | a user to *be*. Clerk cannot conjure a session without one, and creating one changes somebody's account, so it is asked for rather than assumed |
 * | `CLERK_E2E_PASSWORD` | only when the address is an ordinary one |
 *
 * **Prefer a `+clerk_test` address.** Clerk treats those as test accounts and accepts a
 * fixed verification code, which its helper supplies, so no password is involved and none
 * is stored. It also
 * sidesteps a real refusal seen here: Clerk rejects sign-in with a password that appears
 * in a breach corpus, which is a sensible policy and an awkward one to satisfy with a
 * password that has to live in a file.
 *
 * The gateway and worker are assumed up: see `docs/RUNBOOK.md`.
 */

import { clerk, clerkSetup, setupClerkTestingToken } from "@clerk/testing/playwright";
import { expect, test } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const EMAIL = process.env.CLERK_E2E_EMAIL;
const PASSWORD = process.env.CLERK_E2E_PASSWORD;

/** Clerk's helper supplies the fixed verification code for a `+clerk_test` address
 * itself — its `email_code` params take only the identifier. */
const TEST_ACCOUNT = EMAIL !== undefined && EMAIL.includes("+clerk_test");
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
  EMAIL === undefined || (!TEST_ACCOUNT && PASSWORD === undefined),
  "set CLERK_E2E_EMAIL (a +clerk_test address, or an ordinary one with CLERK_E2E_PASSWORD)",
);

test.beforeAll(async () => {
  await clerkSetup();
});

test.beforeEach(async ({ page }) => {
  // Two halves, and both are needed. `clerkSetup` exchanges the instance keys for a
  // Testing Token once; this attaches it to *this page's* requests. Without it the
  // sign-in is accepted and then parks at `needs_client_trust` — Clerk's device-trust
  // check, which a script has no way to satisfy — leaving `Clerk.session` null and the
  // server rendering a signed-out page with no error anywhere to explain it.
  await setupClerkTestingToken({ page });
  await page.goto("/");
  await clerk.signIn({
    page,
    signInParams: TEST_ACCOUNT
      ? { strategy: "email_code", identifier: EMAIL! }
      : { strategy: "password", identifier: EMAIL!, password: PASSWORD! },
  });
  // Fail here, naming the cause, rather than twenty lines later on a missing button.
  // A sign-in that parks at `needs_client_trust` leaves `Clerk.session` null and no error
  // anywhere, and the only symptom is a page that renders as though nobody signed in.
  const signedIn = await page.evaluate(() => {
    const clerkJs = (window as unknown as { Clerk?: { session?: unknown } }).Clerk;
    return clerkJs?.session !== null && clerkJs?.session !== undefined;
  });
  if (!signedIn) {
    throw new Error(
      `Clerk accepted the credentials but created no session for ${EMAIL}. On this ` +
        "instance an ordinary address needs a device-trust code posted to its mailbox, " +
        "which no script can read. Use an address containing '+clerk_test' — Clerk " +
        "accepts a fixed code for those — and leave CLERK_E2E_PASSWORD empty.",
    );
  }

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
