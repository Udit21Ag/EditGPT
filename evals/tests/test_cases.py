"""The golden set validates itself.

A broken cases.json is otherwise only discovered when someone runs the models, which
costs minutes and, for the add cases, real quota.
"""

from __future__ import annotations

import pytest
from editgpt_core import EditOp

from evals.cases import Case, load

ALL = load(include_deferred=True)
RUNNABLE = load()


def test_the_set_is_not_empty_and_deferred_cases_are_excluded_by_default() -> None:
    assert len(RUNNABLE) >= 10
    assert len(ALL) > len(RUNNABLE), "at least one case should be marked deferred"
    assert all(not c.deferred for c in RUNNABLE)


def test_case_ids_are_unique() -> None:
    ids = [c.id for c in ALL]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", ALL, ids=lambda c: c.id)
def test_every_photo_exists(case: Case) -> None:
    assert case.path.exists()


@pytest.mark.parametrize("case", ALL, ids=lambda c: c.id)
def test_boxes_are_fractions_in_order(case: Case) -> None:
    if case.box is None:
        return  # upscale and whole-image operations have no region
    x0, y0, x1, y1 = case.box
    assert all(0.0 <= v <= 1.0 for v in case.box), f"{case.id}: box must be fractions"
    assert x0 < x1, f"{case.id}: x0 must be left of x1"
    assert y0 < y1, f"{case.id}: y0 must be above y1"


@pytest.mark.parametrize("case", ALL, ids=lambda c: c.id)
def test_every_case_carries_a_rationale(case: Case) -> None:
    """A case without a note is a case nobody can interpret when it regresses."""
    assert len(case.note) > 20, f"{case.id}: note is too thin to be useful"


@pytest.mark.parametrize("case", [c for c in ALL if c.op is EditOp.REMOVE], ids=lambda c: c.id)
def test_removals_have_a_target_and_a_reference_box(case: Case) -> None:
    assert case.target, f"{case.id}: removal needs a target phrase"
    assert case.box is not None, f"{case.id}: removal needs a reference box to score against"


@pytest.mark.parametrize(
    "case", [c for c in ALL if c.op in {EditOp.ADD, EditOp.REPLACE}], ids=lambda c: c.id
)
def test_additions_have_content_and_a_fill_prompt(case: Case) -> None:
    assert case.content, f"{case.id}: addition needs content"
    assert case.fill, f"{case.id}: addition needs a fill prompt for a mask+prompt model"


@pytest.mark.parametrize("case", [c for c in ALL if c.op is EditOp.BACKGROUND], ids=lambda c: c.id)
def test_background_cases_name_the_subject_to_keep(case: Case) -> None:
    assert case.target, f"{case.id}: a background change needs to know what to preserve"
    assert case.content, f"{case.id}: a background change needs the replacement described"


def test_prompts_read_as_instructions() -> None:
    by_id = {c.id: c for c in ALL}
    assert by_id["i1"].prompt == "remove the car"
    assert by_id["i3"].prompt == "add a realistic moustache"
    assert by_id["i6c"].prompt == "change the background to a solid green background"
    assert by_id["i8u"].prompt == "upscale the image"
    assert by_id["i4r"].prompt == "replace the horse with a white sheep grazing"


def test_the_operations_actually_covered_are_the_ones_v1_supports() -> None:
    """The set must cover exactly what the gateway advertises, no more and no less.

    A case for an unsupported operation would fail forever; a supported operation with no
    case would regress unnoticed. RESTYLE and RETOUCH are deliberately absent — see the
    tech debt register.
    """
    assert {c.op for c in RUNNABLE} == {
        EditOp.REMOVE,
        EditOp.ADD,
        EditOp.REPLACE,
        EditOp.BACKGROUND,
        EditOp.UPSCALE,
    }
