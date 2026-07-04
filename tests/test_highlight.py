"""Deterministic tests for evidence → bbox resolution (v3.1 §3 M0a).

These cover the make-or-break cockpit primitive without a PDF or API: build a
block with known word geometry and assert that values resolve to the right box,
including the symbol-tolerant matching real journal tables require.
"""

from metaextract.highlight import locate_value, locate_value_in_span, locate_values
from metaextract.sourcedoc import BBox, SourceBlock, Word


def _block(*pairs) -> SourceBlock:
    """Build a one-line block; each pair is (text, x0). y is fixed, width = 20."""
    words = [
        Word(text=t, bbox=BBox(x0=x, y0=100, x1=x + 20, y1=110)) for t, x in pairs
    ]
    bbox = BBox(x0=0, y0=100, x1=400, y1=110)
    return SourceBlock(
        block_id="p1.b1", kind="paragraph", page=1, bbox=bbox,
        text=" ".join(t for t, _ in pairs), words=words,
    )


def test_locates_plain_value():
    blk = _block(("854", 40), ("0.302", 120), ("154", 200))
    m = locate_value(blk, "0.302")
    assert m is not None and m.exact
    assert m.bbox.x0 == 120 and m.words[0].text == "0.302"


def test_tolerates_significance_marks():
    # journals print "0.302***a"; confirming the value 0.302 must still match
    blk = _block(("0.302***a", 120), ("0.398***", 200))
    m = locate_value(blk, "0.302")
    assert m is not None
    assert m.bbox.x0 == 120


def test_tolerates_unicode_minus():
    blk = _block(("−0.087*", 120))  # unicode minus sign
    m = locate_value(blk, -0.087)
    assert m is not None
    assert m.bbox.x0 == 120


def test_absent_value_returns_none():
    blk = _block(("854", 40), ("0.302", 120))
    assert locate_value(blk, "9.999") is None


def test_picks_exact_token_over_numeric_equal():
    # both read as 0.5 numerically; the literal "0.5" should win the box
    blk = _block(("0.50", 40), ("0.5", 120))
    m = locate_value(blk, "0.5")
    assert m is not None and m.exact and m.bbox.x0 == 120


def test_locates_value_inside_ai_evidence_span_not_first_duplicate():
    # Both rows contain 39.38; the AI evidence span disambiguates the treatment row.
    blk = SourceBlock(
        block_id="p1.b1",
        kind="paragraph",
        page=1,
        bbox=BBox(x0=0, y0=0, x1=100, y1=40),
        text="control 39.38 treatment 39.38",
        words=[
            Word(text="control", bbox=BBox(x0=0, y0=0, x1=20, y1=10)),
            Word(text="39.38", bbox=BBox(x0=30, y0=0, x1=50, y1=10)),
            Word(text="treatment", bbox=BBox(x0=0, y0=20, x1=20, y1=30)),
            Word(text="39.38", bbox=BBox(x0=30, y0=20, x1=50, y1=30)),
        ],
    )
    m = locate_value_in_span(blk, "39.38", "treatment 39.38")
    assert m is not None
    assert m.bbox.y0 == 20


def test_locates_all_duplicate_values_in_one_block():
    blk = _block(("39.38", 40), ("label", 80), ("39.38", 120))
    matches = locate_values(blk, "39.38")
    assert [m.bbox.x0 for m in matches] == [40, 120]
