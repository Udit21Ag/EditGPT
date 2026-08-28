/**
 * The jsdom tier only: make `canvas.getContext` return null.
 *
 * jsdom has no 2-D context and logs a stack trace every time one is asked for. Null is
 * what it means, and what a browser with canvas disabled does, so this replaces a page of
 * stderr per render with the degradation path the components are written to take.
 *
 * It is also why the browser tier exists. Everything this stub switches off — the crop
 * arithmetic, the mask overlay, image decoding — is unverified under jsdom no matter how
 * many tests run, and `*.browser.test.tsx` is where it actually executes.
 */
HTMLCanvasElement.prototype.getContext = (() =>
  null) as typeof HTMLCanvasElement.prototype.getContext;
