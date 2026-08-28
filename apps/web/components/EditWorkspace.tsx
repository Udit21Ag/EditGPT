"use client";

/**
 * The workspace, bound to the signed-in session.
 *
 * Ten lines on purpose. `useAuth` is a hook and hooks cannot be opted out of, so a
 * component that calls one can only run inside a `ClerkProvider` — which would put the
 * entire flow behind a session in every context including a test, and Clerk's package
 * expects Node globals that a browser test environment does not have.
 *
 * `lib/api.ts` already argues the general form of this: "`getToken` is passed in rather
 * than imported ... a module-level import would bind this file to one of those and make it
 * untestable in either." `Workspace` is the same reasoning one level up.
 */

import { useAuth } from "@clerk/nextjs";
import { Workspace } from "./Workspace";

export function EditWorkspace() {
  const { getToken } = useAuth();
  return <Workspace getToken={getToken} />;
}
