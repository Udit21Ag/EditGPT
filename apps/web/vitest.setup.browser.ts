/**
 * Stand in for the build-time substitution Next would have done.
 *
 * `lib/api.ts` reads `process.env.NEXT_PUBLIC_GATEWAY_URL`, which is correct: Next inlines
 * every `NEXT_PUBLIC_*` at build time, so the shipped bundle contains a string literal and
 * a browser never evaluates `process`. Vitest serves the module unbuilt, so it does — and
 * without this the whole tier fails to import before a single test runs.
 *
 * An absolute URL, because the tests intercept `fetch` and read `new URL(...).pathname`
 * off what the client sent. A relative one would resolve against the test server and hide
 * exactly the mistake the `||` in `GATEWAY_URL` exists to prevent.
 */
const shim = { env: { NEXT_PUBLIC_GATEWAY_URL: "http://gateway.test" } };
(globalThis as { process?: unknown }).process ??= shim;
