from __future__ import annotations

import random
from typing import Any

import pytest

from typed_evals.metrics.context_units import (
    CONTEXT_SEGMENTATION_VERSION,
    segment_contexts,
)


def _assert_source_partition(contexts: tuple[str, ...], *, granularity: Any = "sentence") -> None:
    units = segment_contexts(contexts, granularity=granularity)
    assert [(unit.passage_index, unit.unit_index) for unit in units] == sorted(
        (unit.passage_index, unit.unit_index) for unit in units
    )
    for passage_index, passage in enumerate(contexts):
        passage_units = [unit for unit in units if unit.passage_index == passage_index]
        if not passage.strip():
            assert passage_units == []
            continue
        assert passage_units[0].start == 0
        assert passage_units[-1].end == len(passage)
        assert "".join(unit.text for unit in passage_units) == passage
        for unit_index, unit in enumerate(passage_units):
            assert unit.unit_index == unit_index
            assert unit.text == passage[unit.start : unit.end]
            assert unit.text.strip()
            assert unit.start < unit.end
            if unit_index:
                assert passage_units[unit_index - 1].end == unit.start


def test_sentences_preserve_exact_source_spans_and_passage_ids() -> None:
    contexts = ("  First sentence.  Second sentence!\nThird?  ", "", "Fourth.")
    units = segment_contexts(contexts)
    assert [(unit.passage_index, unit.unit_index, unit.text) for unit in units] == [
        (0, 0, "  First sentence.  "),
        (0, 1, "Second sentence!"),
        (0, 2, "\nThird?  "),
        (2, 0, "Fourth."),
    ]
    _assert_source_partition(contexts)


def test_passage_mode_has_one_exact_span_per_nonblank_passage() -> None:
    contexts = ("First. Second.\nThird.", " \t\n", "  Fourth?  ")
    units = segment_contexts(contexts, granularity="passage")
    assert [(unit.passage_index, unit.text) for unit in units] == [
        (0, contexts[0]),
        (2, contexts[2]),
    ]
    assert all(unit.unit_index == 0 for unit in units)
    _assert_source_partition(contexts, granularity="passage")


@pytest.mark.parametrize("contexts", [(), ("",), ("  ", "\t\n\r", "\u2028")])
def test_no_units_for_absent_or_blank_source(contexts: tuple[str, ...]) -> None:
    assert segment_contexts(contexts) == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dr. A. Smith paid 3.14 dollars. Next.", ["Dr. A. Smith paid 3.14 dollars. ", "Next."]),
        ("Use e.g. a U.S. source. Then verify.", ["Use e.g. a U.S. source. ", "Then verify."]),
        ("At 9 a.m. we saw Prof. Jones. Done.", ["At 9 a.m. we saw Prof. Jones. ", "Done."]),
        (
            "Go to https://example.com/v1.2?q=yes. Then stop.",
            ["Go to https://example.com/v1.2?q=yes. ", "Then stop."],
        ),
        ("Wait... maybe later. Done.", ["Wait... maybe later. ", "Done."]),
        ("Wait… Perhaps now? Yes!", ["Wait… ", "Perhaps now? ", "Yes!"]),
        ("He said “Stop!” Then left.", ["He said “Stop!” ", "Then left."]),
        ("One?! Two.", ["One?! ", "Two."]),
        ("1. First item.\n  2. Second item.", ["1. First item.", "\n  2. Second item."]),
        ("第一句。第二句！第三句？", ["第一句。", "第二句！", "第三句？"]),
        ("هل هذا صحيح؟نعم。", ["هل هذا صحيح؟", "نعم。"]),
        ("पहला वाक्य।दूसरा वाक्य॥", ["पहला वाक्य।", "दूसरा वाक्य॥"]),
    ],
)
def test_common_sentence_boundaries(text: str, expected: list[str]) -> None:
    assert [unit.text for unit in segment_contexts((text,))] == expected
    _assert_source_partition((text,))


def test_line_breaks_bullets_and_blank_lines_preserve_every_character() -> None:
    text = "\n\n- first item\r\n- second item\n\n\t• third item\u2028fourth\u2029\n "
    assert [unit.text for unit in segment_contexts((text,))] == [
        "\n\n- first item\r\n",
        "- second item\n",
        "\n\t• third item\u2028",
        "fourth\u2029\n ",
    ]
    _assert_source_partition((text,))


def test_long_source_is_not_capped_or_truncated() -> None:
    text = "A useful fact. " * 1200
    units = segment_contexts((text,))
    assert len(units) == 1200
    _assert_source_partition((text,))


def test_long_blank_prefix_and_suffix_are_retained() -> None:
    text = "\n" * 10_000 + "One sentence." + "\n" * 10_000
    units = segment_contexts((text,))
    assert len(units) == 1
    assert units[0].text == text
    _assert_source_partition((text,))


def test_source_partition_for_deterministic_varied_punctuation() -> None:
    rng = random.Random(0)
    alphabet = "abAB12 .!?…\n\r\t()[]\"'。！？؟।-•🙂\u2028\u2029"
    contexts = tuple("".join(rng.choices(alphabet, k=size)) for size in range(500))
    _assert_source_partition(contexts)
    _assert_source_partition(contexts, granularity="passage")
    assert segment_contexts(contexts) == segment_contexts(contexts)


def test_unit_is_immutable() -> None:
    unit = segment_contexts(("A sentence.",))[0]
    with pytest.raises(AttributeError):
        unit.start = 1  # type: ignore[misc]


def test_invalid_granularity_is_rejected_even_without_contexts() -> None:
    with pytest.raises(ValueError, match="granularity"):
        segment_contexts((), granularity="word")  # type: ignore[arg-type]


def test_segmentation_version_is_explicit() -> None:
    assert CONTEXT_SEGMENTATION_VERSION == "context-units-v1"
