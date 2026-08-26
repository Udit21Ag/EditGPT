# Execution plans

**Read when:** work looks larger than a single sitting.
**Solves:** carrying intent, decisions and progress across sessions and agents.
**Authority:** an active plan is authoritative on _that_ work's scope and decisions.

## When to write one

Any of these:

- spans multiple components or packages
- spans multiple sessions
- has meaningful ordering dependencies
- has architectural consequences
- carries significant risk of breaking working behaviour
- needs several distinct verification stages

If none apply, do not write one. A plan for a two-file change is overhead that will go
stale and mislead someone.

## Lifecycle

```
active/NNN-short-name.md  ──►  completed/NNN-short-name.md
```

Update **Progress** as you go, in the same session as the work. A plan updated
retrospectively records what you remember, not what happened.

When finished, move it to `completed/` with the outcome filled in. Completed plans are
kept for their decisions and their dead ends — "we tried X and it did not work, here is
the measurement" is the most valuable thing in the directory, and the reason these are
not disposable notes.

## Template

```markdown
# NNN — Title

## Goal

What is true when this is done. One paragraph, testable.

## Context

Why now. What prompted it. Links to issues, ADRs, debt IDs.

## Current state

What exists today, honestly — including what is broken or missing.

## Proposed approach

The shape of the solution and why this one. Name the alternatives rejected.

## Constraints

Memory, compute, cost, compatibility, deadline.

## Tasks

- [ ] task — change class (LOCAL / CROSS_COMPONENT / ...) — verification

## Verification

How we will know it worked. Commands, metrics, acceptance criteria.

## Risks

What could go wrong, and the early signal for each.

## Decisions

Decisions taken during the work, dated, with reasoning. Promote significant ones to ADRs.

## Progress

Dated entries. What was done, what was learned, what changed about the plan.

## Deferred work

What was consciously left out, with tech-debt IDs where recorded.
```

## Relationship to other records

| Record             | Holds                                                     |
| ------------------ | --------------------------------------------------------- |
| execution plan     | how a piece of work will be and was carried out           |
| ADR (`docs/adr/`)  | a decision that closed off alternatives, and its evidence |
| tech debt register | a problem deliberately deferred                           |
| commit history     | what changed                                              |

A plan is not a substitute for an ADR. If a decision inside a plan will outlive it,
promote it.
