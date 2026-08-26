# Human in the loop

**Read when:** you are unsure whether to proceed on your own.
**Solves:** knowing the boundary between autonomous work and a decision that is not yours.
**Authority:** binding. When in doubt, the answer is ask.

Ordinary, reversible development work is autonomous. Write the code, run the checks, fix
the failures, report the evidence. Asking permission for routine work wastes the human's
attention and trains them to stop reading.

## Always ask before

**Security-sensitive changes** — authentication, authorization, secrets handling,
encryption, permission boundaries, anything touching production credentials. Also: any
change that would widen what untrusted input can reach.

**Destructive operations** — deleting data, destructive migrations, dropping volumes,
irreversible infrastructure changes, rewriting shared history.

**External side effects** — deploying, publishing, sending anything outward, modifying an
external service, any irreversible API call. Spending quota or money counts: a run of the
golden set consumes a real free-tier budget.

**Cross-repository changes** — anything outside this repository, unless explicitly
authorised for this task.

**Architectural decisions** — when several significant options have real tradeoffs and no
existing source of truth settles it. `AGENTS.md` lists the invariants; changing one is
this category.

**Ambiguous requirements** — when readings differ _materially_. Two interpretations that
produce the same work are not ambiguity; pick one and note it.

**Weakening verification** — if finishing appears to require disabling a check, deleting
an assertion, or skipping a test, stop. That is never a judgement call to make alone.

## Do not ask about

- which of two equivalent implementations to use — pick, and say which
- formatting, naming, or file layout the conventions already settle
- whether to write a test — yes
- anything the repository already answers; read it instead
- permission to run a read-only command

Excessive escalation is its own failure. It converts an assistant into a queue.

## How to ask

State, briefly:

1. **the decision** — what needs deciding, in one sentence
2. **why it is not yours** — which category above it falls into
3. **the options** — two or three, concretely
4. **the tradeoffs** — what each costs and forecloses
5. **your recommendation**, if you have one, and why

Then stop. Do not implement the recommendation while awaiting an answer.

**Do everything that does not depend on the answer first.** Arriving with the
independent work complete and one crisp question is far better than arriving with nothing
and a question.

## When the human overrides you

If you raised a concern and it is reaffirmed, that is the decision. Say you understand,
then implement the full request. Do not re-litigate, and do not implement a quiet
compromise between what was asked and what you preferred.
