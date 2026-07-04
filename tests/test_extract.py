"""Deterministic tests for the Tier-3 grounding gate (v3.1 §5.1).

No API: these cover the pure parts — the value⊆span / span⊆block checks and how
``apply_grounding`` turns a raw model response into grounded slots. This is the
Tier-3 analogue of locate's C1 guard: an AI number that cannot be found in its own
cited evidence is flagged, never silently trusted.
"""

from metaextract.extract import (
    RawExtractedValue,
    Tier3Response,
    _response_schema,
    apply_grounding,
    ground_verify,
)
from metaextract.families import CONTINUOUS_TWO_ARM
from metaextract.records import Citation, LocatedRegion
from metaextract.sourcedoc import BBox, SourceBlock


def _block(text: str) -> SourceBlock:
    return SourceBlock(
        block_id="p4.b14", kind="paragraph", page=4,
        bbox=BBox(x0=10, y0=20, x1=560, y1=80), text=text,
    )


def _region() -> LocatedRegion:
    return LocatedRegion(
        region_id="P1:p4.b14:soc", paper_id="P1", variable_name="soil organic carbon",
        citation=Citation(block_id="p4.b14", page=4, bbox=BBox(x0=10, y0=20, x1=560, y1=80),
                          kind="paragraph"),
    )


# --- ground_verify: the deterministic gate ---

def test_grounded_value_passes():
    block = _block("NF 39.38 ± 2.1 AF 31.69 ± 1.8")
    gr = ground_verify("39.38", "NF 39.38 ± 2.1", block)
    assert gr.verified and gr.flags == []


def test_value_not_in_its_own_span_is_flagged():
    """A hallucinated number: the model quoted a span that doesn't contain it."""
    block = _block("NF 39.38 ± 2.1 AF 31.69 ± 1.8")
    gr = ground_verify("99.9", "NF 39.38 ± 2.1", block)
    assert not gr.verified and "value_not_in_span" in gr.flags


def test_span_not_in_block_is_flagged():
    """A spliced value: the span doesn't appear in the cited block at all."""
    block = _block("NF 39.38 ± 2.1 AF 31.69 ± 1.8")
    gr = ground_verify("12.3", "SL 12.3 ± 0.9", block)  # SL row isn't in this block
    assert not gr.verified and "span_not_in_block" in gr.flags


def test_symbol_and_unicode_minus_tolerant():
    """The value -0.087 grounds against a span printed as −0.087**a."""
    block = _block("Treatment effect −0.087**a relative to control")
    gr = ground_verify("-0.087", "−0.087**a", block)
    assert gr.verified


def test_empty_evidence_span_is_flagged():
    block = _block("NF 39.38 ± 2.1")
    gr = ground_verify("39.38", "", block)
    assert not gr.verified and gr.flags == ["no_evidence_span"]


# --- apply_grounding: raw response -> grounded slots ---

def test_apply_grounding_marks_source_and_geometry():
    block = _block("NF 39.38 ± 2.1 AF 31.69 ± 1.8")
    raw = Tier3Response(values=[
        RawExtractedValue(role="mean_c", value="39.38", evidence_text="NF 39.38 ± 2.1"),
        RawExtractedValue(role="mean_t", value="31.69", evidence_text="AF 31.69 ± 1.8"),
    ])
    slots = apply_grounding(raw, block, _region())
    assert {s.role for s in slots} == {"mean_c", "mean_t"}
    for s in slots:
        assert s.source == "ai_extracted"
        assert s.grounding_verified and "evidence_unverified" not in s.flags
        assert s.citation.block_id == "p4.b14" and s.citation.page == 4  # geometry from ingest
        assert s.citation.evidence_text  # span carried for highlighting


def test_apply_grounding_redflags_ungrounded_value():
    block = _block("NF 39.38 ± 2.1")
    raw = Tier3Response(values=[
        RawExtractedValue(role="mean_c", value="88.8", evidence_text="NF 39.38 ± 2.1"),
    ])
    (slot,) = apply_grounding(raw, block, _region())
    assert not slot.grounding_verified
    assert "evidence_unverified" in slot.flags and "value_not_in_span" in slot.flags
    assert slot.value == 88.8  # kept (so the human can see + correct it), just flagged


def test_value_coercion_handles_marks_and_ints():
    block = _block("events 12 of 40; effect 0.302***")
    raw = Tier3Response(values=[
        RawExtractedValue(role="mean_c", value="0.302***", evidence_text="effect 0.302***"),
        RawExtractedValue(role="n_c", value="12", evidence_text="events 12 of 40"),
    ])
    by_role = {s.role: s for s in apply_grounding(raw, block, _region())}
    assert by_role["mean_c"].value == 0.302
    assert by_role["n_c"].value == 12 and isinstance(by_role["n_c"].value, int)


def test_thousands_separator_stored_value_matches_grounding():
    """Regression: the value written to the slot must equal the grounded value.

    "1,234" grounds against a span containing "1,234"; the stored number must be
    1234, not 1 (the old _coerce_number took the first \\d run and disagreed with
    the grounding parser)."""
    block = _block("yield 1,234 kg per plot")
    raw = Tier3Response(values=[
        RawExtractedValue(role="mean_c", value="1,234", evidence_text="yield 1,234 kg"),
    ])
    (slot,) = apply_grounding(raw, block, _region())
    assert slot.grounding_verified
    assert slot.value == 1234  # not 1


def test_chemical_formula_is_not_a_grounded_number():
    """Regression: 'N2O' / 'CO2' must not parse to a stray digit and ground a value.

    ground_verify("2", "N2O emissions", ...) used to pass because _num('N2O')==2."""
    block = _block("N2O emissions were elevated under CO2")
    gr = ground_verify("2", "N2O emissions", block)
    assert not gr.verified and "value_not_in_span" in gr.flags


def test_non_numeric_value_is_not_grounded():
    """Regression: a non-numeric AI value (e.g. 'not reported') must not show green.

    Its span may be in the block, but it carries no number, so it is flagged and
    left unverified (the cockpit must not paint it grounded over a None value)."""
    block = _block("SOC was not reported for the control")
    raw = Tier3Response(values=[
        RawExtractedValue(role="mean_c", value="not reported", evidence_text="not reported"),
    ])
    (slot,) = apply_grounding(raw, block, _region())
    assert not slot.grounding_verified
    assert "non_numeric_value" in slot.flags and "evidence_unverified" in slot.flags
    assert slot.value is None


def test_unknown_role_is_not_grounded_when_family_roles_are_enforced():
    """A model typo/extra role must not pre-fill unrelated Streamlit widget keys."""
    block = _block("control group 7")
    raw = Tier3Response(values=[
        RawExtractedValue(role="ctrl", value="7", evidence_text="control group 7"),
    ])
    (slot,) = apply_grounding(
        raw, block, _region(), allowed_roles=CONTINUOUS_TWO_ARM.roles
    )
    assert not slot.grounding_verified
    assert slot.value == 7  # visible for debugging, but not trusted
    assert "unknown_role" in slot.flags and "evidence_unverified" in slot.flags


def test_uses_family_field_set():
    """Slots are produced for whatever roles the model returned; the family drives
    which roles the prompt asks for (smoke check the field set is wired)."""
    assert {f.role for f in CONTINUOUS_TWO_ARM.fields} >= {"mean_c", "sd_c", "mean_t"}


def test_response_schema_has_no_refs():
    import json

    blob = json.dumps(_response_schema())
    assert "$ref" not in blob and "$defs" not in blob
