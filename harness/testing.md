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

## End-to-end and acceptance

For user-facing behaviour, verify the actual workflow, not its parts. For this project
the end-to-end check is the golden set (`make eval`), which exercises grounding, routing,
editing, compositing and scoring together.

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

## Determinism

Seed every RNG. Freeze time. No wall-clock dependence except explicit sleeps in
lifetime tests. A flaky test is worse than no test: it teaches people to re-run instead
of read.

## No network

Sockets are disabled in the default run. Stub providers. A test that genuinely needs the
network is marked `live` and runs nightly.

## Markers

| Marker   | Meaning                       | In `make check`?   |
| -------- | ----------------------------- | ------------------ |
| _(none)_ | fast, hermetic                | yes                |
| `slow`   | more than a few seconds       | no                 |
| `memory` | asserts peak RSS, own process | no — `make memory` |
| `live`   | real network                  | no — nightly       |

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
