# apps/worker

The Celery worker. One task, `run_job`, drives every job's lifecycle; what edits the
pixels is a separate callable looked up by name in `EDITORS`.

**The edit itself lives in `editgpt_models.execute`, not here.** `editors.py` decodes,
finds the region, calls it, and encodes. Adding editing logic here forks it away from the
golden set, which is the only thing that judges quality.

**Report what was produced, never what was asked for.** `Produced` carries the real
content type and size because `UPSCALE` changes the dimensions and an unwritable format
falls back to PNG. Recording the request's values instead is how the `images` table came
to describe an AVIF that was actually PNG.

**Add an operation as an editor, not as a task.** The lifecycle — transitions, progress,
cancellation, artifacts, the ledger — is written and tested once. A second task would be
a second copy of all of it, and the copies drift.

**Concurrency is 1, and that is not tuning.** Two edits at once hold two heavy models
resident and breach the 8 GB budget before either finishes. `ModelSlot` enforces one
model within a process; `--concurrency=1` is what enforces it between them. Throughput
comes from a bigger host, not from parallelism here.

**`max_tasks_per_child` recycles the process on purpose.** ONNX Runtime's arena does not
return memory to the OS, so a long-lived worker's RSS only ever climbs.

**Cancellation is cooperative**, checked before each transition. Killing the process
would lose the record of why the job stopped, which is the thing a user asks about.

**The job's owner travels in the message.** `run_job` takes a `user_id` and passes it to
every store call, so there is no privileged "fetch any job" path — which is the thing
everyone forgets to protect once accounts exist. Do not add one.

**`acks_late` means a message can arrive twice.** `run_job` returns early for a job that
is already terminal — re-running it would spend quota to reproduce what exists.

**Import order matters.** `app` defines `celery_app`, then `tasks` decorates against it,
and `__init__` imports them in that order. Celery's `autodiscover_tasks` imports `tasks`
from inside `app` and deadlocks on the cycle; do not reintroduce it.

Resources (`resources()`) are `lru_cache`d per process: a Postgres pool and a Redis
connection per task would dominate the cost of a fast one.
