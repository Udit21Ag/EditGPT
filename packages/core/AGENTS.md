# packages/core

Contracts only. Every agent in the system speaks these types.

**Keep this package light.** `__init__` re-exports the pure contract layer; `metrics` and
`rle` are imported explicitly. A process that only handles `EditSpec` must never load
OpenCV because it imported `editgpt_core`. Adding a convenience re-export breaks that.

- `spec.py` — `EditSpec` and friends. Pydantic only. Validation here rejects unactionable
  work at construction rather than three agents downstream, so **add the rule here**, not
  in a caller.
- `rle.py` — mask codec, COCO convention, column-major, opens with a zero run. numpy only.
- `jobs.py` — the job lifecycle. The transition table is **data**, not a chain of `if`s,
  because it is the thing most likely to be got wrong in three places at once. A `Job` is
  immutable: a transition returns a new one, constructed rather than `model_copy`'d, so
  the invariants actually run.
- `metrics.py` — quality scoring. Needs `[metrics]` for OpenCV.
- `errors.py` — every deliberate failure is one of these.

## Before changing metrics.py

Read `docs/EVALUATION.md` first. `compare()` exists because raw fill cost picked the
visually worse image three times; it charges for area erased beyond the base mask. If you
are tempted to compare two masks with `.cost`, that is the bug.

Constants here are calibrated, not chosen. `DEFAULT_GROWTH_PENALTY = 25` is the value at
which the desk case is correctly rejected (+0.7) and the car case correctly accepted
(−2.7). Change it and both tests should tell you.

It is a **default**, not the value the system runs on: `editgpt_models.config.Thresholds`
owns the fitted one and passes it to `compare()`. This package cannot read that file —
`core` imports nothing of ours, and that one-way edge is what lets the orchestrator speak
`EditSpec` without an imaging stack. Do not add a config read here to "fix" it.

**`fill_metrics(...).cost` measures plausibility, not fidelity** (TD-013). It is valid for
comparing two fills of the same mask and invalid as a stand-in for quality.
