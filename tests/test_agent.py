"""Tests for the bounded extraction agent (offline, no API).

These cover the agent's whole contract without the model: the grounding gate
accepts real printed values and rejects fabricated ones, a record is only
``verified`` when every table value grounds, an ungrounded value is routed to
human review instead of being silently committed, and the deterministic
ReplayPlanner drives the full loop end to end.
"""

import json

import pytest

from metaextract.agent import ReplayPlanner, build_toolset, run_agent
from metaextract.agent.gate import record_status
from metaextract.families import get_family
from metaextract.locate import LocateOutput
from metaextract.records import Citation, ExtractionSlot, LocatedRegion, ScreeningResult
from metaextract.sourcedoc import BBox, SourceBlock, SourceDoc, Word


def _w(text: str, x0: float, x1: float, y0: float = 100, y1: float = 110) -> Word:
    return Word(text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))


def _doc(with_figure: bool = False) -> SourceDoc:
    # A borderless two-column table row: SOC, control col (12.0±1.0), treatment col (20.0±2.0).
    words = [
        _w("SOC", 8, 28),
        _w("12.0", 90, 110),
        _w("1.0", 92, 108),
        _w("20.0", 190, 210),
        _w("2.0", 192, 208),
    ]
    blocks = [
        SourceBlock(
            block_id="p1.b1",
            kind="paragraph",
            page=1,
            bbox=BBox(x0=0, y0=90, x1=260, y1=120),
            text="SOC 12.0 1.0 20.0 2.0",
            words=words,
        )
    ]
    if with_figure:
        blocks.append(
            SourceBlock(
                block_id="p2.fig1", kind="figure", page=2,
                bbox=BBox(x0=0, y0=0, x1=300, y1=300), text="(figure)",
            )
        )
    return SourceDoc(doc_id="PX", n_pages=2, blocks=blocks)


REGION_ID = "PX:p1.b1:soil_organic_carbon"


def _out(with_figure: bool = False) -> LocateOutput:
    cit = Citation(block_id="p1.b1", page=1, bbox=BBox(x0=0, y0=90, x1=260, y1=120), kind="paragraph")
    regions = [
        LocatedRegion(
            region_id=REGION_ID, paper_id="PX", variable_name="soil organic carbon", citation=cit
        )
    ]
    if with_figure:
        fcit = Citation(block_id="p2.fig1", page=2, bbox=BBox(x0=0, y0=0, x1=300, y1=300), kind="figure")
        regions.append(
            LocatedRegion(region_id="PX:p2.fig1:mbc", paper_id="PX", variable_name="MBC", citation=fcit)
        )
    screening = ScreeningResult(paper_id="PX", include=True, target_variables_present=["soil organic carbon"])
    return LocateOutput(screening=screening, regions=regions)


# --- the gate ------------------------------------------------------------------
def test_ground_check_accepts_real_and_rejects_fabricated():
    ts = build_toolset(_doc(), _out())
    real = ts.ground_check({"block_id": "p1.b1", "value": "12.0"})
    fake = ts.ground_check({"block_id": "p1.b1", "value": "99999.0"})
    assert real["grounded"] is True
    assert fake["grounded"] is False and "span_not_in_block" in fake["flags"]


def test_propose_then_commit_is_verified():
    ts = build_toolset(_doc(), _out())
    ts.propose_values(
        {
            "region_id": REGION_ID,
            "values": [
                {"role": "mean_c", "value": "12.0"},
                {"role": "sd_c", "value": "1.0"},
                {"role": "mean_t", "value": "20.0"},
                {"role": "sd_t", "value": "2.0"},
            ],
        }
    )
    res = ts.commit_record({"region_id": REGION_ID, "treatment_group": "AF", "control_group": "NF"})
    assert res["status"] == "verified"
    rec = ts.records[0]
    assert rec.slots["mean_c"].value == 12.0 and rec.slots["mean_t"].value == 20.0
    assert all(s.grounding_verified for s in rec.slots.values())
    assert not ts.flagged


def test_ungrounded_value_routes_to_needs_review():
    ts = build_toolset(_doc(), _out())
    ts.propose_values(
        {
            "region_id": REGION_ID,
            "values": [
                {"role": "mean_c", "value": "99999.0"},  # not in the source block
                {"role": "mean_t", "value": "20.0"},
            ],
        }
    )
    res = ts.commit_record({"region_id": REGION_ID, "treatment_group": "AF", "control_group": "NF"})
    assert res["status"] == "needs_review"
    assert ts.flagged and ts.flagged[0].reason == "ungrounded_record"


def test_record_status_needs_all_highlightable_grounded():
    family = get_family(None)
    grounded = ExtractionSlot(slot_id="s1", region_id="r", role="mean_c", grounding_verified=True)
    ungrounded = ExtractionSlot(slot_id="s2", region_id="r", role="mean_t", grounding_verified=False)
    assert record_status([grounded], family) == "verified"
    assert record_status([grounded, ungrounded], family) == "needs_review"
    assert record_status([], family) == "needs_review"


def test_commit_without_proposing_is_flagged_not_committed():
    ts = build_toolset(_doc(), _out())
    res = ts.commit_record({"region_id": REGION_ID, "treatment_group": "AF", "control_group": "NF"})
    assert res["ok"] is False
    assert not ts.records and ts.flagged[0].reason == "no_grounded_values"


# --- the full loop -------------------------------------------------------------
def test_replay_loop_produces_verified_record_and_shows_the_gate():
    result = run_agent(build_toolset(_doc(), _out()), ReplayPlanner())

    assert result.verified_records, "the SOC table region should verify"
    rec = result.verified_records[0]
    assert rec.variable_name == "soil organic carbon"
    assert rec.slots["mean_t"].value == 20.0

    # The run visibly exercises the gate: one probe grounds, one is rejected.
    probes = [s.summary for s in result.trace if s.tool == "ground_check"]
    assert any("GROUNDED" in p for p in probes)
    assert any("REJECTED" in p for p in probes)
    # And it ends cleanly.
    assert result.trace[-1].tool == "finish"


def test_replay_flags_figure_regions_for_digitizing():
    result = run_agent(build_toolset(_doc(with_figure=True), _out(with_figure=True)), ReplayPlanner())
    reasons = {f.reason for f in result.flagged}
    assert "figure_region_needs_digitizing" in reasons
    # the table region still verifies despite the figure region being flagged
    assert any(r.variable_name == "soil organic carbon" for r in result.verified_records)


# --- tool schemas --------------------------------------------------------------
def test_tool_specs_are_function_call_ready():
    specs = build_toolset(_doc(), _out()).specs()
    names = {s.name for s in specs}
    assert {"list_regions", "propose_values", "commit_record", "ground_check", "finish"} <= names
    blob = json.dumps([s.parameters for s in specs])
    assert "$ref" not in blob and "$defs" not in blob  # Gemini rejects these
    for s in specs:
        assert s.name and s.description and s.parameters.get("type") == "object"
