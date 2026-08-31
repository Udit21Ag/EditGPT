"""What the front door does under load, and where the queue stops keeping up.

Run against a stack that is already up:

    make load                      # 10 users, 60s, against localhost:8000
    make load USERS=50 TIME=120s   # harder

Two things are worth measuring here and they are not the same. **Intake** is what the
gateway can accept — uploads and job creation — and is bounded by CPU on hashing and
metadata scrubbing. **Drain** is what the worker can finish, and is bounded by one edit
at a time: concurrency is 1 by design, because two heavy models resident at once breach
the 8 GB budget. Intake will always win, so the interesting number is not requests per
second but the queue depth at which a user's wait stops being honest.

`EditTraffic` therefore does not hammer `/v1/jobs` at full speed. It uploads, creates a
job, and then polls it the way the browser does, so the reported wait is the wait a
person would actually have — and the run prints the depth it reached.

Deliberately not part of `make check`. It needs a running stack, it takes minutes, and a
throughput number measured on a laptop that is also running the worker is a number about
the laptop.
"""

from __future__ import annotations

import io
import os
import random
import time

from locust import HttpUser, between, events, task

TOKEN = os.environ.get("EDITGPT_LOAD_TOKEN", "")
"""A Clerk session token, when the target requires one. Empty against a local gateway
running without authentication, which is the default."""

MAX_WAIT_S = float(os.environ.get("EDITGPT_LOAD_MAX_WAIT_S", "180"))


def a_photograph(width: int = 640, height: int = 480) -> bytes:
    """A different image per virtual user.

    Randomised because storage is content-addressed: identical bytes are one object and
    one row, so a load test that uploads the same picture measures a cache and calls it
    throughput.
    """
    from PIL import Image

    image = Image.new("RGB", (width, height), (random.randrange(256), 90, 40))
    for _ in range(64):
        x, y = random.randrange(width - 40), random.randrange(height - 40)
        for dx in range(40):
            for dy in range(40):
                image.putpixel((x + dx, y + dy), (random.randrange(256), 200, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class EditTraffic(HttpUser):
    """One person: upload a picture, ask for an edit, wait for it like a browser does."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        if TOKEN:
            self.client.headers["Authorization"] = f"Bearer {TOKEN}"

    @task(3)
    def upload_only(self) -> None:
        """Most traffic never becomes a job. Uploading is its own load."""
        self.client.post(
            "/v1/images",
            files={"file": ("load.png", a_photograph(), "image/png")},
            name="POST /v1/images",
        )

    @task(1)
    def upload_then_edit(self) -> None:
        upload = self.client.post(
            "/v1/images",
            files={"file": ("load.png", a_photograph(), "image/png")},
            name="POST /v1/images",
        )
        if upload.status_code != 201:
            return

        created = self.client.post(
            "/v1/jobs",
            json={
                "op": "remove",
                "image_sha256": upload.json()["sha256"],
                "mask_source": "text",
                "target": "the car",
            },
            name="POST /v1/jobs",
        )
        if created.status_code not in (200, 202):
            return

        job_id = created.json()["id"]
        started = time.monotonic()
        while time.monotonic() - started < MAX_WAIT_S:
            time.sleep(2)
            view = self.client.get(f"/v1/jobs/{job_id}", name="GET /v1/jobs/{id}")
            if view.status_code != 200:
                return
            state = view.json()["state"]
            if state in ("done", "failed", "cancelled"):
                events.request.fire(
                    request_type="JOB",
                    name=f"edit -> {state}",
                    response_time=(time.monotonic() - started) * 1000,
                    response_length=0,
                    exception=None,
                    context={},
                )
                return

        events.request.fire(
            request_type="JOB",
            name="edit -> still waiting",
            response_time=MAX_WAIT_S * 1000,
            response_length=0,
            exception=TimeoutError(f"no result in {MAX_WAIT_S}s"),
            context={},
        )
