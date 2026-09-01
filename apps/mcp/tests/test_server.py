"""The tool layer, in process, with the gateway replaced at the network boundary.

`harness/testing.md` asks for the transport to be replaced and never the logic: `httpx` is
the transport here, so the disclosure rules, the request shapes and the reference-only
results are all the real ones.
"""

from __future__ import annotations

from typing import Any

import httpx
import respx
from editgpt_mcp import Gateway, build

GATEWAY = "http://gateway.test"


def server() -> Any:
    return build(Gateway(GATEWAY))


def answer(result: Any) -> Any:
    """What a tool returned, as an object rather than as rendered text."""
    return result.structured_content


async def names(app: Any) -> set[str]:
    return {tool.name for tool in await app.list_tools()}


# ---------------------------------------------------------------- disclosure


async def test_only_a_handful_of_tools_are_listed_at_the_start() -> None:
    """A manifest of everything is thousands of tokens an agent pays for every turn,
    whether or not it edits a picture."""
    assert await names(server()) == {"capabilities", "plan_instruction", "enable_toolset"}


async def test_a_toolset_appears_only_after_it_is_asked_for() -> None:
    app = server()
    assert "find_region" not in await names(app)

    await app.call_tool("enable_toolset", {"name": "grounding"})

    assert "find_region" in await names(app)
    assert "start_edit" not in await names(app), "editing was not asked for"


async def test_enabling_twice_adds_nothing_the_second_time() -> None:
    app = server()
    await app.call_tool("enable_toolset", {"name": "editing"})
    again = answer(await app.call_tool("enable_toolset", {"name": "editing"}))

    assert again["added"] == []
    assert sorted(again["tools"]) == ["edit_status", "start_edit"]


async def test_an_unknown_toolset_is_a_message_an_agent_can_act_on() -> None:
    """Not an exception. An agent can read "available: [...]" and try again."""
    said = answer(await server().call_tool("enable_toolset", {"name": "matting"}))

    assert "no toolset" in said["error"]
    assert "grounding" in said["error"]


async def test_every_listed_tool_says_what_it_is_for() -> None:
    """A tool with no description is a tool an agent has to guess at."""
    app = server()
    await app.call_tool("enable_toolset", {"name": "grounding"})
    await app.call_tool("enable_toolset", {"name": "editing"})

    for tool in await app.list_tools():
        assert tool.description, tool.name


# ---------------------------------------------------------------- what the tools return


@respx.mock
async def test_a_region_comes_back_as_a_reference_and_never_as_pixels() -> None:
    """The run-length encoding of a 15 MP selection is tens of thousands of integers, and
    an agent that receives it pays for all of them on every subsequent turn."""
    respx.post(f"{GATEWAY}/v1/masks").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "box": [0.1, 0.2, 0.4, 0.6],
                        "score": 0.91,
                        "label": "car",
                        "mask": {"width": 10, "height": 10, "counts": [20, 30, 50]},
                    }
                ],
                "ambiguous": False,
                "margin": 0.42,
            },
        )
    )
    app = server()
    await app.call_tool("enable_toolset", {"name": "grounding"})

    found = answer(await app.call_tool("find_region", {"image_sha256": "a" * 64, "phrase": "car"}))

    assert found["candidates"] == [
        {"id": 0, "score": 0.91, "box": [0.1, 0.2, 0.4, 0.6], "area_px": 30, "coverage": 0.3}
    ]
    assert "counts" not in str(found), "the mask itself reached the agent"


@respx.mock
async def test_an_ambiguous_phrase_is_reported_rather_than_resolved() -> None:
    """ADR-0003's whole point, carried to an agent: two close detections mean asking, not
    picking. The margin comes with it so the agent can say why."""
    respx.post(f"{GATEWAY}/v1/masks").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "box": [0, 0, 1, 1],
                        "score": 0.5,
                        "mask": {"width": 4, "height": 4, "counts": [8, 8]},
                    },
                    {
                        "box": [0, 0, 1, 1],
                        "score": 0.46,
                        "mask": {"width": 4, "height": 4, "counts": [8, 8]},
                    },
                ],
                "ambiguous": True,
                "margin": 0.04,
            },
        )
    )
    app = server()
    await app.call_tool("enable_toolset", {"name": "grounding"})

    found = answer(
        await app.call_tool("find_region", {"image_sha256": "b" * 64, "phrase": "the minaret"})
    )

    assert found["ambiguous"] is True
    assert found["margin"] == 0.04
    assert len(found["candidates"]) == 2


@respx.mock
async def test_an_edit_returns_a_job_rather_than_waiting_for_one() -> None:
    """An erase takes about thirteen seconds. Holding a tool call open for it is how an
    agent loop times out on work that was going fine."""
    route = respx.post(f"{GATEWAY}/v1/jobs").mock(
        return_value=httpx.Response(202, json={"id": "job-7", "state": "queued", "progress": 0})
    )
    app = server()
    await app.call_tool("enable_toolset", {"name": "editing"})

    started = answer(
        await app.call_tool(
            "start_edit", {"image_sha256": "c" * 64, "op": "remove", "target": "the car"}
        )
    )

    assert started["id"] == "job-7"
    import json

    sent = json.loads(route.calls[0].request.content)
    assert sent == {
        "op": "remove",
        "image_sha256": "c" * 64,
        "mask_source": "text",
        "target": "the car",
    }


@respx.mock
async def test_an_operation_with_no_subject_asks_for_the_whole_image() -> None:
    route = respx.post(f"{GATEWAY}/v1/jobs").mock(
        return_value=httpx.Response(202, json={"id": "job-8"})
    )
    app = server()
    await app.call_tool("enable_toolset", {"name": "editing"})

    await app.call_tool("start_edit", {"image_sha256": "d" * 64, "op": "upscale"})

    import json

    assert json.loads(route.calls[0].request.content)["mask_source"] == "whole"


@respx.mock
async def test_the_gateway_s_own_words_reach_the_agent_rather_than_a_stack_trace() -> None:
    """ "nothing in this image matches 'the car'" is something an agent can act on."""
    respx.post(f"{GATEWAY}/v1/masks").mock(
        return_value=httpx.Response(404, json={"detail": "nothing in this image matches 'the car'"})
    )
    app = server()
    await app.call_tool("enable_toolset", {"name": "grounding"})

    said = answer(
        await app.call_tool("find_region", {"image_sha256": "e" * 64, "phrase": "the car"})
    )

    assert said["status"] == 404
    assert "nothing in this image matches" in said["error"]


@respx.mock
async def test_a_session_token_travels_with_every_call() -> None:
    """This server is a client like any other: the gateway's auth applies to it too."""
    route = respx.get(f"{GATEWAY}/capabilities").mock(
        return_value=httpx.Response(200, json={"operations": ["remove"]})
    )

    await build(Gateway(GATEWAY, token="session-abc")).call_tool("capabilities", {})

    assert route.calls[0].request.headers["authorization"] == "Bearer session-abc"


@respx.mock
async def test_planning_is_reachable_before_anything_is_enabled() -> None:
    """It is the first thing an agent should do, so it is listed from the start."""
    respx.post(f"{GATEWAY}/v1/plan").mock(
        return_value=httpx.Response(
            200, json={"route": "rule", "op": "remove", "target": "car", "tokens": 0}
        )
    )

    said = answer(await server().call_tool("plan_instruction", {"instruction": "remove the car"}))

    assert said["route"] == "rule"
    assert said["op"] == "remove"
