# Harness maintenance

**Read when:** you learned something durable, or a harness file contradicts the repository.
**Solves:** keeping the harness true, which is the only thing that makes it worth reading.
**Authority:** binding on changes to `AGENTS.md` and `harness/`.

A harness that has drifted is worse than none: it is confidently wrong, and it is trusted.

## Update when enduring operating knowledge changes

| Change                                      | Update                                                         |
| ------------------------------------------- | -------------------------------------------------------------- |
| architecture, boundary or invariant changed | `harness/architecture.md`                                      |
| build, lint, test or run command changed    | `AGENTS.md` command table                                      |
| new testing methodology or category         | `harness/testing.md`                                           |
| new security boundary or input surface      | `harness/architecture.md` (security section)                   |
| new operational diagnostic procedure        | `harness/observability.md` or `harness/docker_log_analysis.md` |
| a recurring engineering problem appeared    | consider a rule, a check, or a lint — not just prose           |
| a decision closed off alternatives          | an ADR in `docs/adr/`, not the harness                         |
| a problem was deliberately deferred         | `harness/tech_debt_tracker.md`                                 |

## Do not update merely because code changed

The harness encodes **durable operating knowledge**, not the current state of the code.
A new function, a refactor, a renamed variable — none of these are harness events. If you
find yourself editing the harness on every task, the harness has become documentation and
has stopped being an operating system.

Ask: _would this still be true in six months, after two refactors?_ If no, it belongs in
code comments, project docs, or the debt tracker.

## Do not overfit

Never encode into the harness: temporary implementation details, one-off debugging steps,
individual task instructions, arbitrary stylistic preferences, model preferences without a
measured reason, or anything that belongs in ordinary project documentation.

## Conflict resolution

When two sources disagree, resolve in this order, and **say that you did**:

1. **Executable behaviour beats stale documentation.** What runs is what is true.
2. **Explicit project rules beat generic assumptions.** A repository convention outranks
   a general best practice.
3. **Tests and contracts are authoritative on intended behaviour** — where they are
   themselves current. A test asserting old behaviour is a conflict, not an authority.
4. **Name the contradiction explicitly** in your report. Silently picking a side hides
   the defect and it recurs.
5. **Never silently change product semantics** to make a document true. Change the
   document, or escalate.
6. **Escalate when authority is genuinely ambiguous** — for example, code and test
   disagree and neither is obviously stale.

The full source-of-truth hierarchy is in `AGENTS.md`.

## Verify a harness change

A harness edit is a change to production infrastructure for every future agent. Before
finishing:

- **No contradictions** — does this disagree with `AGENTS.md`, another harness file, or a
  directory-level `AGENTS.md`?
- **Every referenced path exists.**
- **Every referenced command exists** and does what is claimed. Run it.
- **The rule is enforceable.** A rule nobody can check is a wish. Prefer one a hook, a
  lint or a test can enforce, and say which does.
- **`AGENTS.md` stayed a map.** If it grew a section of detail, that detail belongs in
  `harness/`.
- **A directory `AGENTS.md` stayed scoped** — it must hold only what applies to that
  directory, not a second copy of the root file.
- **No secrets, no credential values, no internal hostnames.**

## Harness integrity

Not permitted, ever: contradictory rules, obsolete commands, broken references,
duplicate conflicting instructions, invented architecture, secret values, or rules that
cannot be verified.

If you find one, fixing it is in scope for whatever task you were doing. A known-wrong
harness left in place is the failure mode that makes every later session slower.
