# AGENTS.md

Universal entrypoint for any coding agent working in this repository. This file is a
**map**, not a manual. Detailed guidance lives in `harness/`; load only what your task
needs.

## Project identity

**EditGPT** — prompt-driven image editing. A user uploads an image, optionally brushes or
boxes a region, and gives an instruction ("remove the car", "add a moustache"). The system
grounds the instruction to a mask, routes to a local or remote editing model, composites
the result, and scores it.

Stack: Python 3.12 (uv workspace) · FastAPI · ONNX Runtime · PyTorch (one model) ·
TypeScript · Next.js 15 · Tailwind v4 · pnpm workspace · Docker Compose · GitHub Actions.

```
packages/core        contracts: EditSpec, Job, AssetRef, MaskRef, RLE codec, metrics
packages/models      ModelSlot, grounding, erasers, multi-pass pipeline, thresholds
packages/providers   remote provider protocol, circuit breaker, failover
packages/store       content-addressed assets, the schema, job persistence, progress
apps/gateway         FastAPI: upload, job intake, SSE progress, rate limits
apps/worker          Celery: the job lifecycle
apps/web             Next.js frontend
evals/               golden image set and its runner
benchmarks/          held-out datasets and the threshold fitting
tests/               integration across apps; everything else lives beside its code
harness/             this operating system
docs/                project documentation and ADRs
spike/               frozen Phase 0 feasibility work — history, not a dependency
```

**Hard constraint: 8 GB RAM, CPU only, free-tier services.** Every design decision is
downstream of this. A proposal that violates it is not a proposal.

## First actions

Before modifying anything:

1. Read this file.
2. Read the `AGENTS.md` in the directory you are changing, if one exists. Read only that
   one — not all of them.
3. Read the harness file for your task (table below).
4. `git status` and `git diff` — know what is already in flight.
5. Inspect the code you intend to change **and its callers, callees and tests**.

## Source of truth

When sources disagree, authority runs in this order:

1. **Executable configuration** — `Makefile`, `pyproject.toml`, `package.json`,
   `.github/workflows/`, `docker-compose.yml`
2. **Source code** — what actually runs
3. **Tests** — the encoded contract
4. **Generated artifacts** — `uv.lock`, `pnpm-lock.yaml`, `evals/out/report.json`, OpenAPI
5. **Project documentation** — `docs/`
6. **Harness documentation** — `harness/`

Documentation contradicting executable behaviour is a **defect to report**, not a
question to resolve silently. Never change product semantics to make a document true.

## Commands

Verified against the repository. Do not invent alternatives.

| Purpose                          | Command                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| install everything               | `make setup`                                                                               |
| download model weights (~552 MB) | `make models`                                                                              |
| **the gate**                     | `make check`                                                                               |
| fast inner loop (Python only)    | `make check-fast`                                                                          |
| lint / format check              | `make lint` · fix with `make fmt`                                                          |
| type check (mypy strict)         | `make types`                                                                               |
| tests with coverage              | `make test`                                                                                |
| memory regression tier           | `make memory`                                                                              |
| golden image set                 | `make eval`                                                                                |
| web lint / types / tests         | `make web-lint` · `make web-types` · `make web-test`                                       |
| run gateway only                 | `make dev-lite`                                                                            |
| run the Celery worker            | `make worker`                                                                              |
| full local stack                 | `make dev`                                                                                 |
| start redis + postgres           | `make compose-up` · stop with `make compose-down`                                          |
| apply / create a migration       | `make migrate` · `make migration NAME="..."`                                               |
| held-out benchmarks              | `make bench-grounding` · `make bench-removal` · `make bench-ambiguity` · `make bench-tune` |

`make check` runs exactly what CI runs, in CI's order — literally: each CI step invokes
the corresponding Makefile target, so the two cannot drift.

## Operating rules

1. **Inspect before editing.** A component existing is not evidence it works.
2. **Understand existing abstractions before creating new ones.** Search first; this
   repository has already grown near-duplicate mask utilities twice.
3. **Make focused changes.** The smallest _coherent architectural unit_, not the smallest
   diff.
4. **Preserve existing behaviour** unless the task requires changing it.
5. **Avoid unnecessary dependencies.** State the tradeoff against what already exists and
   against doing nothing.
6. **Verify, then report evidence.** Never "done" without it.
7. **Measure before concluding.** Numbers rank candidates; eyes confirm them. A
   photometric score has picked the visually worse image three times here.
8. **Never weaken verification** to get green. See `harness/code_generation.md`.

## Change classification

Classify before implementing. Higher categories demand stronger verification and, where
marked, human approval.

| Class                | Meaning                                              | Required                     |
| -------------------- | ---------------------------------------------------- | ---------------------------- |
| `LOCAL`              | one module, no interface change                      | `make check-fast`            |
| `CROSS_COMPONENT`    | crosses a package boundary                           | `make check`                 |
| `ARCHITECTURAL`      | changes a boundary, dependency direction or contract | `make check` + ADR + **ask** |
| `SECURITY_SENSITIVE` | auth, secrets, permissions, input boundaries         | `make check` + **ask**       |
| `DATA_AFFECTING`     | migrations, stored artifacts, eval fixtures          | `make check` + **ask**       |
| `INFRASTRUCTURE`     | CI, Docker, deployment                               | `make check` + CI green      |
| `CROSS_REPOSITORY`   | anything outside this repo                           | **ask, always**              |

## Harness navigation

| Task                        | Read                             |
| --------------------------- | -------------------------------- |
| Understanding the system    | `harness/architecture.md`        |
| Writing or changing code    | `harness/code_generation.md`     |
| Writing or changing tests   | `harness/testing.md`             |
| Something failed; iterating | `harness/looping.md`             |
| Adding logging or metrics   | `harness/observability.md`       |
| Diagnosing containers       | `harness/docker_log_analysis.md` |
| Deciding whether to ask     | `harness/human_in_the_loop.md`   |
| Deferring a known problem   | `harness/tech_debt_tracker.md`   |
| Updating the harness itself | `harness/self_updation.md`       |
| Multi-session or risky work | `harness/exec-plans/`            |

Project documentation, not harness: `docs/EVALUATION.md` (what counts as evidence),
`docs/RUNBOOK.md` (operating it), `docs/adr/` (why it is this way).

## Completion rule

**Writing code is not completing a task.** A task is complete when verification appropriate
to its change class has run and passed, and you have reported the evidence:

```
Implemented:      what changed, behaviourally
Files changed:    paths
Verification:     command → result, for each
Tests:            added or updated, and what they pin
Known limitations: what does not work
Deferred work:    with tech-debt IDs where recorded
Harness updates:  or "none"
```

If you could not verify something, say which thing and why. An honest gap is worth more
than an unearned "done".

## Never

- weaken a test, suppress a lint rule, or alter CI to hide a failure
- commit to `main`, force-push, or use `--no-verify`
- commit `.env` or any credential
- edit `spike/` (frozen), `evals/photos/` (fixtures), or an accepted ADR
- `pip install` (uv) or `npm install` (pnpm)
- claim a performance or quality gain without a measurement
