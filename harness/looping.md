# Looping

**Read when:** verification failed, or you are about to try a fix for the second time.
**Solves:** converging on a correct change instead of thrashing.
**Authority:** binding on iteration and escalation.

## The loop

```
UNDERSTAND ─► PLAN ─► IMPLEMENT ─► FORMAT ─► LINT ─► TYPE CHECK
                                                        │
                                                        ▼
                                                   UNIT TESTS
                                                        │
                                                        ▼
                                            INTEGRATION / E2E
                                                        │
                                                        ▼
                                            RUNTIME VERIFICATION
                                                        │
                                                        ▼
                                        TASK ACCEPTANCE CHECK
                                                        │
                                                  ALL GREEN?
                                                   ╱        ╲
                                                 NO          YES
                                                 │            │
                                            DIAGNOSE       REPORT
                                                 │
                                             REPAIR
                                                 │
                                             RE-RUN
```

In this repository the middle band is one command: `make check` runs format check, lint,
types, tests and the web tier in that order. `make check-fast` is the Python-only inner
loop. The **task acceptance check** is whatever proves _this_ change did what was asked —
often `make eval`, sometimes a manual run, occasionally a browser check.

Order matters. A type error frequently explains the test failure below it, so read
upward from the first failure rather than starting with the last.

## Iterate automatically when the failure is actionable

Actionable failures include: compilation and import errors, type errors, lint and format
failures, unit and integration test failures, runtime errors, incorrect behaviour,
acceptance failures, and observable regressions.

Fix them and re-run without asking.

## Never repeat a failed approach

After each failed iteration:

1. **Read the failure.** The actual message, the actual line — not the summary.
2. **Identify the root cause**, not the symptom.
3. **Decide what is wrong**: the implementation, the test, the environment, or your
   assumption about the requirement. These need different fixes and confusing them is
   how loops become infinite.
4. **Make the smallest coherent correction.**
5. **Re-run the narrowest verification that covers it**, then widen.

If your second attempt is a variation of the first, stop and re-diagnose. Two similar
failures mean the model of the problem is wrong, not that the fix needs tuning.

## Signs the loop is not converging

- the same test fails three times with different patches
- a fix for one failure creates another, repeatedly
- you are considering weakening a check
- you cannot state, in one sentence, why the last attempt failed

Any of these: stop and escalate. See `human_in_the_loop.md`.

## Diagnosing beyond the message

- **Test fails, implementation looks right** — check the fixture. A fixture that is too
  clean or too extreme passes and fails for the wrong reasons.
- **Passes locally, fails in CI** — environment. Check extras, lockfiles, and whether
  something reaches the network.
- **Intermittent** — an unseeded RNG, wall-clock dependence, or shared state. Fix the
  determinism; do not re-run until it passes.
- **Runtime error only under load or at size** — resource limits. Check
  `harness/observability.md`.
- **A container is involved** — `harness/docker_log_analysis.md`.

## Stop and ask when

- the requirement is ambiguous and readings differ materially
- verification cannot establish correctness at all
- the fix requires a product decision
- the change is security-sensitive or destructive
- external access you need is unavailable
- repeated attempts point at an architectural problem
- **you would have to weaken verification to proceed**

Escalating with a clear diagnosis is a good outcome. Grinding is not.
