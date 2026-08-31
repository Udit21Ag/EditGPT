"""One-shot asset sweep, for a person at a terminal.

`make sweep` reports what would go; `make sweep APPLY=1` deletes it. The scheduled path
is the Celery task of the same name — this runs the identical code in the foreground so
an operator can see the answer before a cron does it unattended, which is the difference
between trusting a housekeeping job and hoping about one.
"""

from __future__ import annotations

import json
import sys

from editgpt_core.logs import configure as configure_logs
from sqlalchemy.exc import OperationalError

from editgpt_worker.tasks import sweep_assets


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    configure_logs(service="housekeeping")
    try:
        report = sweep_assets(dry_run="--apply" not in args)
    except OperationalError:
        # The common way to run this by hand is with the stack down, and a page of
        # driver traceback is a worse answer than the one sentence that fixes it.
        print("the database is not reachable; a sweep needs it. `make compose-up` first.")
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
