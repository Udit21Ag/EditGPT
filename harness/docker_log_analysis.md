# Container diagnostics

**Read when:** a containerised service is unhealthy, unreachable, or behaving unexpectedly.
**Solves:** finding the cause instead of restarting until it works.
**Authority:** binding on destructive container operations.

This repository runs two services in containers: `redis` and `postgres`
(`docker-compose.yml`). The application runs on the host during development, deliberately —
on an 8 GB machine, the stack and a model benchmark must not contend.

## Procedure

Work down this path. Do not skip to the fix.

```
symptom → which service → when (timestamp) → correlated logs across services
        → configuration → dependencies → root cause → fix → verification
```

**Correlate, do not tunnel.** A service that fails to start is often reacting to another
that is unhealthy. Read the logs of the dependency at the same timestamp before
concluding.

## What to inspect

| Question                               | Command                                              |
| -------------------------------------- | ---------------------------------------------------- |
| what is running, and healthy?          | `docker compose ps`                                  |
| what stopped, and with what exit code? | `docker compose ps -a`                               |
| what did it say?                       | `docker compose logs --tail=200 <service>`           |
| what did everything say, interleaved?  | `docker compose logs --since=10m`                    |
| what is it configured with?            | `docker compose config`                              |
| is it resource-starved?                | `docker stats --no-stream`                           |
| does the healthcheck actually pass?    | `docker compose exec redis redis-cli ping`           |
|                                        | `docker compose exec postgres pg_isready -U editgpt` |

Read exit codes rather than guessing: `0` clean, `1` application error, `137` killed —
usually out of memory, which on this machine is a live hypothesis rather than a remote
one — `143` terminated.

## Common causes here

- **Port already in use.** Redis 6379 or Postgres 5432 taken by a host installation.
  `lsof -i :6379`.
- **Healthcheck passes, the app still cannot connect.** The app runs on the _host_, so it
  connects via `localhost`, not the compose service name.
- **Container killed at 137.** Memory. Check `docker stats` and what else was running.
  Docker Desktop's own overhead is roughly 1.5 GB; OrbStack or colima is the documented
  preference for that reason.
- **Postgres refuses connections briefly on first start** while it initialises. The
  healthcheck exists so dependents wait; if something ignores it, that is the bug.

## Destructive operations

**Do not restart or recreate blindly.** A restart destroys the evidence and, if the cause
is external, changes nothing.

- restarting a service: acceptable while diagnosing, _after_ capturing logs
- `docker compose down`: acceptable — it preserves named volumes
- `docker compose down -v`: **destroys the Postgres volume. Ask first.**
- `docker system prune`: **blocked**; it affects things outside this project
- deleting a volume, image or network you did not create: **ask**

## After a fix

Reproduce the original symptom's trigger and show it no longer occurs. "The container is
up" is not verification — the container was probably up during the failure too.
