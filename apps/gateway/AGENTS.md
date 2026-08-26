# apps/gateway

FastAPI. Auth, upload, job intake, progress streaming.

**No model ever loads in this process.** It is the web tier; models live in workers. If
you find yourself importing `editgpt_models` here, the work belongs in a task.

Phase 1 ships the skeleton and its health contract only. `/capabilities` advertises what
this deployment can actually do rather than what the plan hoped for — the frontend should
not offer operations with no model behind them.

Settings come from `Settings` (env prefix `EDITGPT_`), read once. No `os.environ` reads
scattered through handlers.
