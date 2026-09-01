"""EditGPT's vision capability, as MCP tools.

**A client of the gateway, not a second copy of it.** Every tool here is an HTTP call to
the same API the web app uses, so this process holds no models, no database and no
credentials beyond a session token. That is what makes the claim "the orchestrator imports
no model code" literally true rather than aspirational — and it is why an agent driving
this server gets the same auth, the same rate limits and the same `/ready` degradations a
browser does.

**Progressive disclosure.** Three tools are listed at start-up. `enable_toolset` reveals
the rest, so an agent's prompt carries a handful of names instead of every capability with
its full schema. The plan (§2, L2) called for this on the argument that a manifest of
everything is thousands of tokens an agent pays for on every turn, whether or not it
edits a picture.

**Results carry references, never pixels.** A mask comes back as its area, its bounding
box and an id — the run-length encoding of a 15 MP selection is tens of thousands of
integers, and putting that in a model's context window is the same mistake as returning
the image itself.

Run it with `make mcp` (stdio, which is what an MCP client expects).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer

log = logging.getLogger(__name__)

TIMEOUT_S = 30.0
"""Long enough for grounding, short enough that an agent is not left hanging. Editing is
not waited on at all — `start_edit` returns a job id and `edit_status` is polled."""

TOOLSETS: dict[str, tuple[str, ...]] = {
    "grounding": ("find_region",),
    "editing": ("start_edit", "edit_status"),
}
"""What `enable_toolset` can reveal, and what each set contains.

Two sets rather than one per tool: an agent that wants to ground a phrase almost always
wants to edit the region afterwards, and disclosure that is too fine-grained is just a
round trip per tool."""


class Gateway:
    """The EditGPT API, as this server sees it."""

    def __init__(self, base_url: str, token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.token}"} if self.token else {}

    async def get(self, path: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers())
        return _read(response)

    async def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            response = await client.post(
                f"{self.base_url}{path}", json=body, headers=self._headers()
            )
        return _read(response)


def _read(response: httpx.Response) -> dict[str, Any]:
    """The gateway's answer, or its own words about why there isn't one.

    An agent can act on "nothing here matches 'the car'" and cannot act on a stack trace,
    so a 4xx comes back as a message rather than an exception.
    """
    if response.status_code >= 400:
        detail = (
            response.json().get("detail")
            if "json" in response.headers.get("content-type", "")
            else response.text[:200]
        )
        return {"error": str(detail), "status": response.status_code}
    body: dict[str, Any] = response.json()
    return body


def build(gateway: Gateway) -> MCPServer:
    """The server, with only the tier-1 tools listed."""
    server = MCPServer(
        name="editgpt-vision",
        instructions=(
            "Edit photographs by describing the change. Start with `plan_instruction` to "
            "turn a sentence into an operation, then `enable_toolset('grounding')` to find "
            "what it refers to and `enable_toolset('editing')` to run it. Masks are "
            "returned as references — area, box and id — never as pixels."
        ),
    )

    async def capabilities() -> dict[str, Any]:
        """What this deployment can do: the operations it runs and the limits it enforces."""
        return await gateway.get("/capabilities")

    async def plan_instruction(instruction: str) -> dict[str, Any]:
        """Turn one sentence into an operation, or into a question.

        Answers most instructions from rules without a model. `route` says which: `rule`,
        `model`, or `ask` when the sentence names no edit this deployment can run.
        """
        return await gateway.post("/v1/plan", {"instruction": instruction})

    async def find_region(image_sha256: str, phrase: str) -> dict[str, Any]:
        """Find what `phrase` refers to in an uploaded image.

        Returns the candidates as references — score, box, area — with `ambiguous` set
        when the best two are close enough that the right move is to ask a person which
        one they meant rather than to pick.
        """
        found = await gateway.post("/v1/masks", {"image_sha256": image_sha256, "phrase": phrase})
        if "error" in found:
            return found
        return {
            "ambiguous": found.get("ambiguous", False),
            "margin": found.get("margin"),
            "candidates": [_reference(index, c) for index, c in enumerate(found["candidates"])],
        }

    async def start_edit(
        image_sha256: str,
        op: str,
        target: str | None = None,
        content: str | None = None,
        colour: str | None = None,
    ) -> dict[str, Any]:
        """Begin an edit. Returns a job id; the work happens elsewhere.

        Not waited on: an erase takes about thirteen seconds and a generative fill longer,
        which is a poor thing to hold a tool call open for. Poll `edit_status`.
        """
        given = {"target": target, "content": content, "colour": colour}
        request: dict[str, Any] = {
            "op": op,
            "image_sha256": image_sha256,
            # A phrase is a region; without one the operation acts on the whole frame.
            "mask_source": "text" if target else "whole",
            # Omitted rather than sent as null: `EditSpec` forbids extra keys and reads a
            # missing field and an explicit `None` differently.
            **{name: value for name, value in given.items() if value},
        }
        return await gateway.post("/v1/jobs", request)

    async def edit_status(job_id: str) -> dict[str, Any]:
        """Where an edit has got to, and what checking it found.

        `review` is the critic's trail: what each attempt changed, what it cost, and
        whether the selection was widened and tried again.
        """
        return await gateway.get(f"/v1/jobs/{job_id}")

    tier_two: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
        "find_region": find_region,
        "start_edit": start_edit,
        "edit_status": edit_status,
    }

    async def enable_toolset(name: str) -> dict[str, Any]:
        """Reveal a set of tools that was not listed at start-up.

        Sets: `grounding` (find what a phrase refers to), `editing` (run one and follow it).
        """
        if name not in TOOLSETS:
            return {"error": f"no toolset {name!r}; available: {sorted(TOOLSETS)}"}
        listed = {tool.name for tool in await server.list_tools()}
        added: list[str] = []
        for tool_name in TOOLSETS[name]:
            if tool_name not in listed:
                server.add_tool(tier_two[tool_name], name=tool_name)
                added.append(tool_name)
        log.info("mcp.toolset_enabled", extra={"toolset": name, "added": added})
        return {"enabled": name, "tools": list(TOOLSETS[name]), "added": added}

    for tool in (capabilities, plan_instruction, enable_toolset):
        server.add_tool(tool)
    return server


def _reference(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
    """One candidate, without its pixels.

    The mask stays on the server. A run-length encoding of a 15 MP selection is tens of
    thousands of integers, and an agent that receives it pays for all of them on every
    subsequent turn while being no better able to act.
    """
    mask = candidate.get("mask", {})
    counts = mask.get("counts", [])
    area = sum(counts[1::2])
    width, height = mask.get("width", 0), mask.get("height", 0)
    return {
        "id": index,
        "score": candidate.get("score"),
        "box": candidate.get("box"),
        "area_px": area,
        "coverage": round(area / (width * height), 4) if width and height else 0.0,
    }


def main() -> None:
    """Serve over stdio, which is what an MCP client launches."""
    logging.basicConfig(level=logging.INFO)
    server = build(
        Gateway(
            os.environ.get("EDITGPT_GATEWAY_URL", "http://localhost:8000"),
            os.environ.get("EDITGPT_MCP_TOKEN", ""),
        )
    )
    server.run("stdio")


if __name__ == "__main__":
    main()
