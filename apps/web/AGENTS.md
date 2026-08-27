# apps/web

Next.js 15 App Router, TypeScript strict, Tailwind v4.

- `pnpm lint` · `pnpm typecheck` · `pnpm test` — all three run in `make check`.
- ESLint 9 flat config bridges `eslint-config-next` through `FlatCompat`, because Next 15
  still ships eslintrc-format configs. Revisit when it ships a flat config.

## Masks

Store **strokes, not bitmaps**. A stroke is a few hundred bytes; a 15.9 MP mask is ~16 MB,
so a 50-step history costs kilobytes instead of gigabytes. Points are normalised to 0..1
so a stroke survives a resize and replays correctly after a zoom.

## Mobile matters

The brush must work with a finger: touch targets, pinch-zoom, bottom-sheet tools. This is
not a desktop tool with a phone layout bolted on.

Never send image bytes through the app. The gateway owns storage; the browser talks to it
with references.

## Clerk

`middleware.ts` uses an allowlist of **public** routes, not a list of protected ones. A
page added later is protected by default; a list of protected routes silently leaves
anything nobody remembered to add wide open.

**Every `/v1` call needs a bearer token**, fetched per request in `lib/api.ts`. Do not
cache one in a module: Clerk's session tokens are short-lived and a cached token is a
request that starts failing a minute later.

**`<img src>` cannot send an `Authorization` header**, which is why `imageObjectUrl`
fetches and hands back a blob URL — and why the caller must `revokeObjectURL` it. Signed
URLs (Phase 9) are the real fix.

**`EventSource` cannot send one either**, so `streamJob` reads the SSE body off a `fetch`
stream and parses the frames by hand. It returns a cancel function; call it on unmount.
