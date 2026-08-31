# Observability

**Read when:** adding logging or metrics, or diagnosing something you cannot see.
**Solves:** making the system diagnosable without shipping noise or leaking data.
**Authority:** binding on what may be logged.

## How

`editgpt_core.logs.configure(service=...)` once at startup, then log with `extra={...}`.
One JSON object per line, on stderr; `EDITGPT_LOG_FORMAT=text` for a terminal.

**`bound(**fields)` for correlation.** Anything logged inside the block carries those
fields, including from code that knows nothing about them — the gateway binds
`request_id` per request, the worker binds `job_id` for a whole task. Threading an
identifier through forty call sites is how that stops happening by the third one.

**Credentials are removed by the formatter, not by remembering.** A field whose *name*
contains `token`, `secret`, `password`, `key`, `authorization`, `credential` or `cookie`
is replaced. Deliberately broad: a redacted field that did not need to be is a moment of
confusion, and the alternative is a leak nobody notices.

Two traps, both of which had already thrown this away once. Nothing configured logging at
all, so thirty-one `extra={...}` call sites rendered as a bare message. And Celery replaces
the root logger's handlers when a worker starts — `worker_hijack_root_logger=False` is
what keeps the worker's fields.

## Logging

A log line should let someone reconstruct what happened without a debugger. Include:

- the operation and its stage
- inputs at a **safe abstraction level** — dimensions, counts, identifiers, model name;
  never the payload
- outcome: success or failure, and the cause on failure
- duration for anything that can be slow
- correlating identifiers: job id, request id, and — once agents exist — trace id

Prefer structured logging with fields over interpolated prose. A message that must be
parsed with a regex to be useful is a message that will not be used.

Levels: `DEBUG` for development detail, `INFO` for lifecycle, `WARNING` for degraded but
handled, `ERROR` for failed work. A warning nobody acts on should be `DEBUG`.

## Never log

Credentials, tokens, API keys, raw prompts containing personal data, raw image bytes, or
file contents. If you need to correlate an image, log its content digest, not the image.

Provider error bodies may echo request content — truncate them before logging.

## Metrics

Where they earn their keep:

| Area      | Measure                                                                     |
| --------- | --------------------------------------------------------------------------- |
| requests  | latency, throughput, error rate                                             |
| jobs      | queue depth, time in queue, retries                                         |
| models    | inference latency by model, peak resident memory, load time, cache hit rate |
| editing   | passes attempted vs kept, rollback rate, escalation rate                    |
| providers | call count, failure rate, circuit state, quota consumed                     |
| quality   | score distribution over the golden set                                      |

**Pass and rollback rates are the interesting ones here.** A rising rollback rate means
the second pass is doing less than it costs; a falling one may mean the gate stopped
firing. Neither is visible from latency.

## Resource visibility

This project is memory-constrained, so memory is a first-class signal, not a footnote:
peak resident set per stage, which models are resident, evictions, and how often the
ceiling is approached. Sample the child process rather than asking it about itself where
you can — the parent interpreter otherwise counts toward the number.

## Debugging instrumentation

Temporary instrumentation during development is fine, and must be **intentional, safe,
and removed or promoted before completion**. A stray print in a hot path is a performance
bug; a stray print of user content is a data leak.

Before finishing, grep your diff for debug output.

## ML-specific

Log the model identity and version with every inference — "which model produced this" is
the first question asked about a bad output and the hardest to answer retrospectively.
Also worth recording: input dimensions, preprocessing path taken, confidence or quality
scores, and a failure category when a stage rejects its input.

Do not log the images themselves. A digest plus dimensions reproduces the case from
stored artifacts.
