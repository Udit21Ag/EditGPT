# Code generation

**Read when:** you are about to write or change code.
**Solves:** producing changes that fit the codebase and survive review.
**Authority:** binding. Where it conflicts with a directory's `AGENTS.md`, the narrower
scope wins.

## Understand before editing

Before changing a file, class or function, inspect:

- its **callers** — what depends on current behaviour
- its **callees** — what it relies on
- its **tests** — the contract as encoded
- **nearby patterns** — how this codebase already solves this shape of problem
- related **configuration** — thresholds, settings, feature gates

A component existing is not evidence it works. Trace the data flow rather than inferring
behaviour from a name or a docstring.

## Write for the unit, not the line

Code must satisfy the responsibility of the whole module, class or function — not merely
the line you were sent to change. Patchwork fixes that satisfy one call site while
leaving the abstraction incoherent are worse than the bug.

## Minimal change

Change the **smallest coherent architectural unit** required to solve the problem
correctly. This is not "touch the fewest lines". A three-line patch that entrenches a
broken abstraction is not minimal; it is deferred cost.

## Reuse existing abstractions

Search before you add. Do not introduce a second utility, service, model wrapper or
helper that overlaps an existing one without saying why the existing one does not fit.
This repository has already grown near-duplicate mask utilities twice.

Equally: do not build an abstraction for a single caller. A protocol with one
implementation is a guess about the future, and the guess is usually wrong. If you are
writing a base class while implementing the first subclass, stop.

## Quality bar

Generated code must be readable, maintainable, conventional for this repository, and free
of dead code, hidden side effects, unnecessary abstraction and unnecessary dependencies.
Type safety is preserved: this repository runs mypy strict.

## Constants carry their provenance

A threshold derived from measurement records the measurement:

```python
RESIDUAL_MAX_GROWTH = 0.50
"""Cap on how much the residual pass may grow the mask.

Measured: +35% of the object's area cleared a car's cast shadow; +78% and +119% meant the
detector had latched onto scene content and the second erase destroyed it.
"""
```

Not `# max growth`. The next reader must know whether the value is considered or
arbitrary, and therefore whether they may change it. **A magic number with no provenance
is technical debt** — record it (`tech_debt_tracker.md`) if you cannot justify it.

## Errors name the fix

An error message should tell a human or an agent what to do next without opening the
source. Include the observed value and the expectation.

## Silence is a bug

Returning the input unchanged _looks_ like success and _reports_ as success. This project
lost real time to a mask so small the output was identical to the input while every
numeric check passed. Raise instead.

A genuinely empty result is different: a text prompt matching nothing is a real outcome,
so return an empty result with zero confidence and let the caller decide. The distinction
is whether the caller can tell.

## Validate at boundaries

External input — uploads, prompts, request bodies, provider responses — is validated
where it enters the system, once. Downstream code may then assume it is well-formed.
Do not re-validate defensively at every layer; do not skip it at the boundary.

## Comment the surprising

`# increment i` is noise. "this mask convention is inverted — 255 means keep" saves an
afternoon. If you had to work something out, write down what you worked out.

## Tests

Never modify a test to make a failing implementation pass.

A test may change when the requirement changed, the behaviour intentionally changed, the
test was wrong, the test was brittle, or it no longer reflects the contract. **Say which,
in the commit body.**

## No verification evasion

Never disable a lint rule, suppress an error, remove an assertion, weaken an expectation,
skip a failing test, or alter CI in order to get green. If verification is genuinely
wrong, fix verification deliberately and say so. If you find yourself wanting to weaken a
check to finish, that is an escalation trigger — see `human_in_the_loop.md`.

## When you are unsure

State the assumption in the code and in your report, and continue. Do not silently pick
and hope. If the choice changes the _shape_ of the work rather than a detail, stop and
ask.
