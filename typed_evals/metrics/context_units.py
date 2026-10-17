"""Deterministic, source-preserving units for retrieval relevance judgments.

Sentence segmentation is a conservative punctuation/newline heuristic, not a
linguistic parser or a claim extractor. It preserves conjunctions, can keep more
than one sentence together after ambiguous abbreviations, and can split quoted
speech or formatted text at a newline. Callers can choose passage granularity
when those boundaries are unsuitable. Neither mode infers, paraphrases, drops,
caps, or truncates non-whitespace source content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Include this version wherever segmentation affects a metric/calibration fingerprint.
CONTEXT_SEGMENTATION_VERSION = "context-units-v1"

_TERMINATORS = frozenset(".!?。！？｡؟।॥።…")
_CLOSERS = frozenset("\"'”’»›)]}〉》」』】〕〗〙〛")
_NEWLINES = frozenset("\r\n\u2028\u2029")
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "mx",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "mt",
        "vs",
        "etc",
        "approx",
        "fig",
        "figs",
        "eq",
        "eqs",
        "no",
        "nos",
        "vol",
        "vols",
        "pp",
        "inc",
        "ltd",
        "corp",
        "dept",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "ph.d",
    }
)
_PRECEDING_WORD = re.compile(r"([^\W\d_]+(?:\.[^\W\d_]+)*)\.$", re.UNICODE)


@dataclass(frozen=True, slots=True)
class ContextUnit:
    """An exact source span; indices and offsets are zero-based.

    ``unit_index`` restarts within each original passage. ``start`` and ``end``
    use Python string character offsets, so ``text == contexts[passage_index][start:end]``.
    Whitespace is retained so the units reconstruct every nonblank passage exactly.
    """

    passage_index: int
    unit_index: int
    start: int
    end: int
    text: str


def segment_contexts(
    contexts: tuple[str, ...],
    *,
    granularity: Literal["sentence", "passage"] = "sentence",
) -> tuple[ContextUnit, ...]:
    """Return all evaluable source units in passage order, without a size cap.

    Whitespace-only passages yield no units: they contain no relevance target.
    Every other passage is partitioned into nonempty, contiguous source spans;
    each unit contains non-whitespace content. Sentence mode recognizes line
    boundaries and common sentence terminators, while avoiding decimal points,
    common abbreviations, initials, and periods inside tokens such as URLs.
    These rules are language-independent heuristics, not atomic-claim extraction.
    """
    if granularity not in {"sentence", "passage"}:
        raise ValueError("granularity must be 'sentence' or 'passage'")
    units: list[ContextUnit] = []
    for passage_index, passage in enumerate(contexts):
        if not passage.strip():
            continue
        spans = [(0, len(passage))] if granularity == "passage" else _sentence_spans(passage)
        units.extend(
            ContextUnit(
                passage_index=passage_index,
                unit_index=unit_index,
                start=start,
                end=end,
                text=passage[start:end],
            )
            for unit_index, (start, end) in enumerate(spans)
        )
    return tuple(units)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    has_content = False
    while index < len(text):
        char = text[index]
        has_content = has_content or not char.isspace()
        end = index + 1
        if char in _NEWLINES:
            if char == "\r" and end < len(text) and text[end] == "\n":
                end += 1
        elif char in _TERMINATORS:
            if char == "." and _internal_period(text, index):
                index += 1
                continue
            while end < len(text) and text[end] in _TERMINATORS:
                end += 1
            terminator_end = end
            while end < len(text) and text[end] in _CLOSERS:
                end += 1
            # ASCII punctuation within a token is commonly a URL, identifier,
            # or programming syntax. CJK and other terminators need no spaces.
            if (
                char in ".!?…"
                and end < len(text)
                and not text[end].isspace()
                and text[end] not in _NEWLINES
            ):
                index = end
                continue
            while end < len(text) and text[end].isspace() and text[end] not in _NEWLINES:
                end += 1
            if (
                (char == "…" or text[index:terminator_end] == "...")
                and end < len(text)
                and text[end].islower()
            ):
                index = end
                continue
        else:
            index += 1
            continue
        if has_content:
            spans.append((start, end))
            start = end
            has_content = False
        index = end
    if has_content:
        spans.append((start, len(text)))
    elif spans:
        # Attach trailing blank lines to the final meaningful span.
        spans[-1] = (spans[-1][0], len(text))
    return spans


def _internal_period(text: str, index: int) -> bool:
    if index + 1 == len(text):
        return False
    before = text[index - 1] if index else ""
    after = text[index + 1]
    if before.isdigit() and after.isdigit():
        return True
    if before.isalnum() and after.isalnum():
        return True
    if before.isdigit() and after.isspace():
        # A numbered line item ("1. First item") is one unit, not a bare
        # number followed by a sentence. Leave its source marker attached.
        marker_start = index - 1
        while marker_start and text[marker_start - 1].isdigit():
            marker_start -= 1
        prefix = marker_start - 1
        while prefix >= 0 and text[prefix].isspace() and text[prefix] not in _NEWLINES:
            prefix -= 1
        if prefix < 0 or text[prefix] in _NEWLINES:
            return True
    token_start = index
    while token_start and (text[token_start - 1].isalpha() or text[token_start - 1] == "."):
        token_start -= 1
    match = _PRECEDING_WORD.search(text, token_start, index + 1)
    if match is None:
        return False
    token = match.group(1)
    if token.lower() in _ABBREVIATIONS:
        return True
    # Preserve personal initials and dotted initialisms (A. Smith, U.S. policy).
    return (len(token) == 1 and token.isupper()) or (
        "." in token and all(len(part) == 1 and part.isalpha() for part in token.split("."))
    )
