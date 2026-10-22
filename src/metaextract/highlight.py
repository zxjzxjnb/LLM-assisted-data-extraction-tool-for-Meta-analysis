"""Resolve an evidence string to a highlight bbox (Architecture v3.1 §3, C1).

This is the engine behind the cockpit's make-or-break feature. Given a block the
locator pointed at and the value/span a human or Tier-3 model is confirming, it
returns the exact word geometry to light up in the PDF. The match is tolerant of
the trailing marks journals love (``0.302***a``, ``12.3 ​b``) and of values that
span several words.

It is deterministic and offline: no LLM, no API. A returned ``Match`` is also the
input to grounding (v3.1 §5) — if a claimed value cannot be located in its cited
block, that *is* the grounding failure.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from .sourcedoc import BBox, SourceBlock, Word

# Strip everything except digits, sign, decimal point, and exponent marker, so a
# table value "−0.087*" compares equal to the number "-0.087". Used only for the
# *exactness* tie-break in locate_value, on tokens parse_number already accepted.
_NUM_STRIP = re.compile(r"[^0-9eE.+\-]")

# Leading decoration we tolerate before a number (±, ~, ≈, >, <, parentheses, …),
# but NOT letters — a token that starts with a letter is a word/formula, not a
# number, so "N2O" / "CO2" / "ha-1" must stay rejected (they used to parse to a
# stray digit under the old strip-everything approach).
_LEAD_DECORATION = re.compile(r"^[^0-9A-Za-z+.\-]+")
# A number anchored at the (de-decorated) start: optional sign, digits with
# optional thousands separators and decimals, or a bare ".5", plus exponent.
_NUM_AT_START = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


class Match(BaseModel):
    """Where an evidence string was found, ready to highlight."""

    bbox: BBox  # union box, for scroll-into-view + outline
    words: list[Word]  # the individual matched tokens, for tight highlighting
    exact: bool  # True = literal token match; False = numeric-equality match


def parse_number(token: str) -> Optional[float]:
    """Parse a printed value to a float, tolerant of the marks journals add.

    Accepts a number wrapped in decoration ("−0.087*", "±0.5", "(12.3)",
    "0.302***a", "1,234") but **rejects words and formulas** ("N2O", "CO2",
    "ha-1") — the number must appear at the start once leading decoration is
    stripped. This is the single source of truth for "what number is this token",
    shared by highlighting (:func:`locate_value`, :func:`numeric_in_text`) and by
    Tier-3 value coercion, so a grounded value can never disagree with the value
    actually stored.
    """
    token = _LEAD_DECORATION.sub("", token.replace("−", "-").strip())
    m = _NUM_AT_START.match(token)
    if m is None:
        return None
    try:
        return float(m.group().replace(",", ""))  # drop thousands separators
    except ValueError:
        return None


_num = parse_number  # internal alias for the call sites below


def locate_value(block: SourceBlock, value, rel_tol: float = 1e-9) -> Optional[Match]:
    """Find a numeric ``value`` among the block's words (symbol-tolerant)."""
    matches = locate_values(block, value, rel_tol=rel_tol)
    if not matches:
        return None
    return next((m for m in matches if m.exact), matches[0])


def locate_values(block: SourceBlock, value, rel_tol: float = 1e-9) -> list[Match]:
    """Find every occurrence of a numeric ``value`` in one block."""
    target = _num(str(value))
    if target is None:
        match = locate_span(block, str(value))
        return [match] if match is not None else []

    out: list[Match] = []
    for w in block.words:
        wv = _num(w.text)
        if wv is None:
            continue
        if wv == target or (target != 0 and abs(wv - target) / abs(target) <= rel_tol):
            exact = _NUM_STRIP.sub("", w.text.replace("−", "-")) == str(value).replace(
                "−", "-"
            )
            out.append(Match(bbox=w.bbox, words=[w], exact=exact))
    return out


def locate_value_in_span(
    block: SourceBlock, value, evidence_text: str, rel_tol: float = 1e-9
) -> Optional[Match]:
    """Find ``value`` within the model's quoted evidence span.

    A block can contain the same number several times (e.g. control and treatment
    rows). For grounded AI suggestions, the evidence span is the disambiguator:
    highlight the value inside that span first, then let the cockpit fall back to
    whole-page value search if the span cannot be resolved.
    """
    span = locate_span(block, evidence_text)
    if span is None:
        return None
    span_block = SourceBlock(
        block_id=block.block_id,
        kind=block.kind,
        page=block.page,
        bbox=span.bbox,
        text=" ".join(w.text for w in span.words),
        words=span.words,
    )
    return locate_value(span_block, value, rel_tol=rel_tol) or span


def locate_span(block: SourceBlock, text: str) -> Optional[Match]:
    """Find a consecutive run of words matching the phrase ``text``."""
    query = [t for t in re.split(r"\s+", text.strip()) if t]
    if not query:
        return None
    words = block.words
    qn = [_norm(t) for t in query]
    for i in range(len(words) - len(query) + 1):
        window = words[i : i + len(query)]
        if all(_norm(w.text).startswith(q) or q.startswith(_norm(w.text)) for w, q in zip(window, qn)):
            box = BBox.union([w.bbox for w in window])
            if box is not None:
                return Match(bbox=box, words=window, exact=True)
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower().replace("−", "-")


def numeric_in_text(value, text: str, rel_tol: float = 1e-9) -> bool:
    """True when ``value`` appears as a number among ``text``'s whitespace tokens.

    Symbol-tolerant exactly like :func:`locate_value` (so a span ``"−0.087**a"``
    grounds the value ``-0.087``); falls back to a normalized substring test when
    ``value`` is not numeric. This is the *value ⊆ evidence-span* half of the
    Tier-3 grounding gate (v3.1 §5.1), kept here so the symbol tolerance lives in
    one place.
    """
    target = _num(str(value))
    if target is None:
        return _norm(str(value)) in _norm(text)
    for tok in re.split(r"\s+", text):
        tv = _num(tok)
        if tv is None:
            continue
        if tv == target or (target != 0 and abs(tv - target) / abs(target) <= rel_tol):
            return True
    return False


def text_contains(haystack: str, needle: str) -> bool:
    """Normalized substring test (whitespace / case / unicode-minus tolerant).

    The *evidence-span ⊆ cited-block* half of the grounding gate: the model's
    claimed span must really appear in the block it cited.
    """
    return bool(needle) and _norm(needle) in _norm(haystack)
