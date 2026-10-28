"""Deterministic tests for screen/locate scaffolding (v3.1 §2, C1).

No API: these cover the pure parts — addressable payload, and the C1 guard that
geometry is attached only for real block_ids while invented ones are dropped and
reported.
"""

import pytest

from metaextract.locate import (
    LocatorResponse,
    RawModerator,
    RawRegion,
    _response_schema,
    build_payload,
    resolve_moderators,
    resolve_regions,
)
from metaextract.sourcedoc import BBox, SourceBlock, SourceDoc


def _doc() -> SourceDoc:
    blocks = [
        SourceBlock(
            block_id="p1.b1", kind="paragraph", page=1,
            bbox=BBox(x0=10, y0=20, x1=110, y1=40), text="N2O emission was measured ...",
        ),
        SourceBlock(
            block_id="p3.b7", kind="paragraph", page=3,
            bbox=BBox(x0=5, y0=200, x1=560, y1=260), text="Tsoil 854 0.302 154 0.398 ...",
        ),
    ]
    return SourceDoc(doc_id="paper1", n_pages=3, blocks=blocks)


def test_payload_is_addressable():
    payload = build_payload(_doc())
    assert "[p1.b1]" in payload and "[p3.b7]" in payload
    assert "block_id values from this list only" in payload


def test_resolve_attaches_real_geometry():
    doc = _doc()
    raw = LocatorResponse(
        include=True,
        regions=[RawRegion(block_id="p3.b7", variable_name="N2O emission")],
    )
    regions, problems = resolve_regions(doc, raw, doc.doc_id)
    assert not problems
    assert len(regions) == 1
    cit = regions[0].citation
    assert cit.page == 3 and cit.bbox.x1 == 560  # geometry from the SourceDoc, not the model


def test_invented_block_id_is_dropped_and_reported():
    """C1 guard: a hallucinated location gets no fabricated coordinates."""
    doc = _doc()
    raw = LocatorResponse(
        include=True,
        regions=[
            RawRegion(block_id="p9.b99", variable_name="N2O emission"),  # does not exist
            RawRegion(block_id="p1.b1", variable_name="N2O emission"),
        ],
    )
    regions, problems = resolve_regions(doc, raw, doc.doc_id)
    assert len(regions) == 1 and regions[0].citation.block_id == "p1.b1"
    assert len(problems) == 1 and "p9.b99" in problems[0]


def test_duplicate_regions_are_deduped():
    doc = _doc()
    raw = LocatorResponse(
        include=True,
        regions=[
            RawRegion(block_id="p1.b1", variable_name="N2O emission"),
            RawRegion(block_id="p1.b1", variable_name="N2O emission"),
        ],
    )
    regions, _ = resolve_regions(doc, raw, doc.doc_id)
    assert len(regions) == 1


def test_unknown_target_variable_is_dropped_when_task_vocabulary_is_supplied():
    doc = _doc()
    raw = LocatorResponse(
        include=True,
        regions=[RawRegion(block_id="p1.b1", variable_name="invented metric")],
    )
    regions, problems = resolve_regions(
        doc, raw, doc.doc_id, allowed_variables={"n2o emission": "N2O emission"}
    )
    assert not regions
    assert problems and "invented metric" in problems[0]


def test_moderators_keep_value_and_attach_citation_when_block_real():
    doc = _doc()
    raw = LocatorResponse(
        include=True,
        moderators=[
            RawModerator(field="Experimental site", value="Songpan County", block_id="p1.b1"),
            RawModerator(field="Climate", value="subalpine", block_id="p9.b9"),  # invalid block
            RawModerator(field="Depth cm", value="0-20"),  # no block_id
        ],
    )
    mods = resolve_moderators(doc, raw)
    by_field = {m.field: m for m in mods}
    assert by_field["Experimental site"].citation.page == 1  # real block → jumpable
    assert by_field["Climate"].citation is None  # invalid block → value kept, no citation
    assert by_field["Depth cm"].value == "0-20" and by_field["Depth cm"].citation is None


def test_moderators_dedupe_by_field():
    doc = _doc()
    raw = LocatorResponse(
        include=True,
        moderators=[
            RawModerator(field="Climate", value="subalpine", block_id="p1.b1"),
            RawModerator(field="Climate", value="alpine"),
        ],
    )
    assert len(resolve_moderators(doc, raw)) == 1


def test_moderators_are_limited_to_task_pack_fields():
    doc = _doc()
    raw = LocatorResponse(
        include=True,
        moderators=[
            RawModerator(field="Climate", value="subalpine", block_id="p1.b1"),
            RawModerator(field="invented moderator", value="x", block_id="p1.b1"),
        ],
    )
    mods = resolve_moderators(doc, raw, allowed_fields={"climate": "Climate"})
    assert [m.field for m in mods] == ["Climate"]


def test_response_schema_has_no_refs():
    import json

    blob = json.dumps(_response_schema())
    assert "$ref" not in blob and "$defs" not in blob


def test_resolve_against_real_ingest(tmp_path):
    fitz = pytest.importorskip("fitz")

    from metaextract.ingest import ingest_pdf

    pdf_path = tmp_path / "paper.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Tsoil 854 0.302 154 0.398")
    pdf.save(pdf_path)
    pdf.close()

    doc = ingest_pdf(pdf_path)
    target = next(b for b in doc.blocks if "Tsoil" in b.text)
    raw = LocatorResponse(
        include=True,
        regions=[RawRegion(block_id=target.block_id, variable_name="N2O emission")],
    )
    regions, problems = resolve_regions(doc, raw, doc.doc_id)
    assert not problems
    assert regions[0].citation.page == target.page
    assert regions[0].citation.bbox == target.bbox
