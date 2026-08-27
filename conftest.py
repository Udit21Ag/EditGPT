"""Make the test suite independent of whoever is running it.

**The incident this exists for.** `Settings` reads `.env`, which is correct for the
application and wrong for tests. It did no harm while no setting changed behaviour — then
Clerk landed, a developer put real keys in `.env`, and twenty tests that expect
unauthenticated mode started returning 401. Nothing was broken; the suite had simply been
reading the developer's machine all along, and only now did that machine have an opinion.

A test run must produce the same answer on a laptop with every credential configured, on
a fresh checkout with none, and in CI. So two things are cut off before collection:

1. `.env` is unhooked from every settings class, so a file on disk cannot reach a test.
2. Credential variables are cleared from the environment, so an exported one cannot
   either.

Deliberately *not* cleared: `EDITGPT_DATABASE_URL`, `EDITGPT_BENCH_DIR`,
`EDITGPT_MODELS_DIR` and `EDITGPT_THRESHOLDS`. Those point tests at local resources rather
than changing what is being tested, and the `service`-marked tests need the first one.

A test that wants a credential sets it explicitly — see `tests/test_clerk_auth.py`, which
turns authentication on by constructing settings that say so.
"""

from __future__ import annotations

import os

import pytest

CREDENTIAL_PREFIXES = ("CLERK_", "EDITGPT_CLERK_", "EDITGPT_S3_", "CLOUDFLARE_", "GEMINI_")
CREDENTIAL_NAMES = ("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY", "NEXT_PUBLIC_GATEWAY_URL")


def pytest_configure(config: pytest.Config) -> None:
    del config  # the hook's signature; nothing here depends on it
    for name in list(os.environ):
        if name.startswith(CREDENTIAL_PREFIXES) or name in CREDENTIAL_NAMES:
            del os.environ[name]

    # Unhook `.env`. Imported here rather than at module scope so collecting a package
    # that does not depend on the apps stays cheap.
    from editgpt_gateway.settings import Settings as GatewaySettings
    from editgpt_worker.settings import Settings as WorkerSettings

    for settings_class in (GatewaySettings, WorkerSettings):
        settings_class.model_config["env_file"] = None
