# evals

The golden set. 14 runnable cases plus one deferred, each with a reference box and a note.

`make eval` runs them and prints status, seconds, cost, IoU and the pass chain.
`evals/out/report.json` is the machine-readable form. Compare against `main` before
claiming an improvement — see `docs/EVALUATION.md`.

## Rules for a case

- Every case carries a **note explaining what it tests**. A case nobody can interpret
  when it regresses is worse than no case. `test_every_case_carries_a_rationale` enforces
  a floor on this.
- Removals need a `target` phrase and a reference box to score against.
- Additions need `content` and a `fill` prompt phrased for a mask+prompt model.
- Background cases name the **subject to keep**, not the background to replace.
- A case that cannot run yet is `"deferred": true` with the reason, so the gap stays
  visible instead of quietly absent.

The tests in `evals/tests/` validate the case file itself. They run in `make check` and
need no models, so a broken `cases.json` is caught in seconds rather than minutes.

## Fixtures

`evals/photos/` is committed — 8 MB. An eval set you cannot reproduce is worthless, and
that is a fair price. Do not add large fixtures without asking whether the case earns it.
