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
