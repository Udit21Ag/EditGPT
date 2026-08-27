# Testing

**Read when:** writing tests, changing tests, or deciding whether a change is verified.
**Solves:** proving a change works, and preventing the same defect twice.
**Authority:** binding. Test integrity rules are not negotiable.

Testing answers two different questions, and both matter:

1. **Does the code satisfy its contract?**
2. **Does the system actually perform the intended task?**

A suite that answers only the first passes while the product is broken.

## Discover before writing

Inspect existing tests for the module first. Match their fixtures, naming and structure.
A second, differently-shaped suite for the same module is a maintenance tax.

## Unit tests

Cover normal behaviour, boundaries, invalid input, error handling, and the invariants
that matter. Skip getters, framework glue, and anything whose test would restate the
implementation — a test that mirrors the code catches typos and doubles the cost of every
future change.

## Integration tests

Where components interact in ways unit tests cannot reach: a pipeline across modules, a
provider chain with failover, a request through the gateway. Use stubs at the network
boundary, real objects everywhere inside it.

**A test that spans two apps lives in `tests/`, not beside either of them.** The gateway
and the worker do not depend on each other, deliberately — the web tier must never pull
in the worker's model stack — and a test placed in either directory would quietly create
that dependency. Everything one package can prove alone stays next to that package.

Replace the _transport_, never the logic. The end-to-end test runs the real FastAPI app,
the real repository and the real Celery task function, with an in-process queue and an
in-memory Redis. Stubbing the task instead would have produced a test of the stub.

## End-to-end and acceptance

For user-facing behaviour, verify the actual workflow, not its parts. For this project
the end-to-end check is the golden set (`make eval`), which exercises grounding, routing,
editing, compositing and scoring together.

**Diff it; do not read it.** `make eval-diff` compares a run against `evals/baseline.json`
and CI does the same on every pull request. Record a new baseline with `make eval-baseline`
only after looking at the pictures — the baseline is an assertion that the current output
is acceptable, so updating it to make a diff go away is the one way to defeat the check.

**Trust the images over the numbers.** `cost` is a photometric proxy this project has
measured as unreliable (TD-017), and TD-004 is the worked example: a change that visibly
ruined `i8` moved cost 3.2% and moved the result thumbnail 0.400. A clean diff is not
evidence the pictures are fine; a flagged case is an instruction to open one.

## Regression tests

Every meaningful bug fix gets a test that fails before the fix. Name the incident:

```python
def test_compare_rejects_a_bigger_erase_that_looks_better_on_raw_cost() -> None:
    """The i6 regression, in miniature."""
```

## Every test must be able to fail

Before finishing: mentally delete the implementation. If the test still passes it is not
a test. This is the most common defect in generated suites.

## Calibrate fixtures against real numbers

A fixture that is too clean passes for the wrong reason and proves nothing.

Real incident: a synthetic "flat fill" was written to prove a scoring penalty was needed,
but it was _so_ flat that a different term rejected it independently. The test passed
while testing nothing. The fix was to compute the real numbers first in a scratch script,
then build the assertion around what was measured.

**Measure, then assert.** If you cannot state what the numbers are, you cannot write the
test.

## ML and data testing

Distinguish these; conflating them is how ML projects end up unverified.

| Kind              | Question                                                | Deterministic?           |
| ----------------- | ------------------------------------------------------- | ------------------------ |
| unit              | does this function meet its contract                    | yes                      |
| data validation   | is the input what we assume                             | yes                      |
| model behaviour   | does the model respond correctly to a controlled change | mostly                   |
| metric evaluation | does quality clear a threshold on a fixed set           | yes, given seeds         |
| end-to-end task   | does the system do the job                              | yes, given fixtures      |
| qualitative       | does it look right                                      | no — human, and required |

**Model output being plausible is not evidence.** Numbers rank candidates; a human
confirms them. In this project a photometric score picked the visually worse image three
separate times — see `docs/EVALUATION.md`, which is authoritative on which metrics are
valid here and why several common ones are not.

**A metric is not validated until it holds on a second dataset.** The proxy that drove
the router was adopted on one benchmark and turned out to correlate the wrong way. Before
any score is allowed to make a decision, report its correlation with ground truth on two
independent sets — a correlation that appears on one and not the other is a property of
that dataset, not of the metric.

**When a fitted value loses to the default, do not ship it.** `benchmarks/tune.py` has
declined to write one twice, and both refusals were the correct result. A tuner that
always writes something is a tuner that launders overfitting into a config file.

## Determinism

Seed every RNG. Freeze time. No wall-clock dependence except explicit sleeps in
lifetime tests. A flaky test is worse than no test: it teaches people to re-run instead
of read.

**And no dependence on the developer's machine.** A run must give the same answer on a
laptop with every credential configured, on a fresh checkout with none, and in CI. The
root `conftest.py` enforces that by unhooking `.env` from the settings classes and
clearing credential variables before collection.

That rule was bought: `Settings` reads `.env`, which is right for the application and
wrong for tests, and it did no harm for as long as no setting changed behaviour. The day
real Clerk keys were added, twenty tests expecting unauthenticated mode began returning 401. Nothing was broken — the suite had been reading the machine all along, and the
machine had only just acquired an opinion. **A test that needs a credential sets it
itself.**

## No network

Sockets are disabled in the default run. Stub providers. A test that genuinely needs the
network is marked `live` and runs nightly.

## Markers

| Marker    | Meaning                                         | In `make check`?   |
| --------- | ----------------------------------------------- | ------------------ |
| _(none)_  | fast, hermetic                                  | yes                |
| `service` | needs a live Postgres or MinIO; skips otherwise | yes, when it is up |
| `slow`    | more than a few seconds                         | no                 |
| `memory`  | asserts peak RSS, own process                   | no — `make memory` |
| `live`    | real network                                    | no — nightly       |

`service` is in the default tier on purpose, and it earns the exception to "no network".
Two things forced it, both the same shape:

- The initial migration was generated against SQLite, reviewed, merged and never applied.
  Its first real run failed because Postgres rejected an untyped bind for a `uuid` column,
  while every SQLite test stayed green throughout. **A migration is not verified by
  review, only by applying it and rolling it back on the dialect it will meet.**
- `S3AssetStore` was covered entirely by a hand-written stub, which proves the arithmetic
  around the calls and nothing about the calls. **A stub cannot tell you which exception
  the real client raises**, and the caller's `except` is built on that answer.

These skip cleanly with no containers, so a fresh checkout stays green, and CI runs the
services so they genuinely execute. `make compose-up` and `make compose-s3` locally.

Each test builds a throwaway database or bucket. Never point a test that drops everything
at one somebody is using, and never share one between tests that delete — two parallel
runs will delete each other's.

Resource assertions run **over repeated iterations**. Measured resource use varies
run-to-run; a single sample once set a ceiling 20% too low.

## Coverage

The gate is 80%. It is a floor, not a target. Coverage measures which lines ran, not
whether they were checked — a module at 100% whose assertions are all "is not None" is
untested. **State what you could not test and why.** A named gap is worth more than a
percentage.

## Integrity

Never weaken a test to achieve green. See `harness/code_generation.md`, "no verification
evasion". If a test is genuinely wrong, fix it deliberately and say why in the commit.
