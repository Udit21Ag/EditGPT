"""External benchmarks: held-out evaluation on datasets we did not curate.

`evals/` is the **dev set** — 18 hand-built cases whose reference boxes were drawn by
hand and against which the pipeline's thresholds were tuned. That makes it unsuitable for
answering "does this generalise": the thresholds were fitted to it.

This package answers that question instead, using datasets that carry their own ground
truth so no manual labelling is involved and no threshold has been fitted to them.
"""
