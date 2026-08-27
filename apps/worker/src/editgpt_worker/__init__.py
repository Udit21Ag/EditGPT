"""Celery worker for EditGPT.

The lifecycle lives in `tasks`; what a task needs lives in `app`. Importing this package
registers the tasks but builds nothing: `resources()` is lazy, so a test can import a task
without opening a database connection.

Import order matters. `app` defines `celery_app`, then `tasks` decorates against it. The
reverse — Celery's `autodiscover_tasks` — imports `tasks` from inside `app` and deadlocks
on the cycle.
"""

from editgpt_worker.app import celery_app
from editgpt_worker.tasks import run_job

__all__ = ["celery_app", "run_job"]
