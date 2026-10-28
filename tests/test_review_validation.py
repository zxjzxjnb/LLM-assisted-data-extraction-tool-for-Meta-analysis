import math

from metaextract.families import CONTINUOUS_TWO_ARM
from metaextract.review_validation import (
    normalize_unit,
    prepare_export_record,
    target_variable_coverage,
    validate_record,
    validate_records_for_export,
    validate_unit_comparability,
)
from metaextract.sourcedoc import BBox, SourceBlock, SourceDoc


def _record(**overrides):
    record = {
        "control_group": "C",
        "treatment_group": "T",
        "unit": "g/kg",
        "dispersion": "sd",
        "mean_c": "10",
        "sd_c": "2",
        "n_c": "3",
        "mean_t": "12",
        "sd_t": "2.5",
        "n_t": "4",
        "evidence": {
            "mean_c": {"evidence_text": "10"},
            "sd_c": {"evidence_text": "2"},
            "mean_t": {"evidence_text": "12"},
            "sd_t": {"evidence_text": "2.5"},
        },
    }
    record.update(overrides)
    return record


def test_incomplete_and_invalid_records_are_flagged():
    flags = validate_record(_record(mean_c="", n_t="2.5", treatment_group=""), CONTINUOUS_TWO_ARM)
    assert "missing_mean_c" in flags
    assert "invalid_n_t" in flags
    assert "missing_treatment_group" in flags


def test_se_is_converted_to_sd_and_original_is_retained():
    out = prepare_export_record(_record(dispersion="se", sd_c="2", sd_t="3"), CONTINUOUS_TWO_ARM)
    assert math.isclose(out["sd_c"], 2 * math.sqrt(3))
    assert math.isclose(out["sd_t"], 3 * math.sqrt(4))
    assert out["sd_c_original"] == "2"
    assert out["dispersion_input"] == "se"
    assert out["dispersion"] == "sd"
    assert out["unit_normalized"] == "g/kg"


def test_common_unit_spellings_are_normalized_without_silent_scale_conversion():
    assert normalize_unit("g⋅kg−1") == "g/kg"
    assert normalize_unit("mg kg-1") == "mg/kg"


def _source_doc():
    block = SourceBlock(
        block_id="p1.b1",
        kind="paragraph",
        page=1,
        bbox=BBox(x0=0, y0=0, x1=200, y1=20),
        text="SOC 10 2 12 2",
    )
    methods = SourceBlock(
        block_id="p2.b1",
        kind="paragraph",
        page=2,
        bbox=BBox(x0=0, y0=0, x1=200, y1=20),
        text="Samples were measured in triplicate.",
    )
    return SourceDoc(doc_id="P1", n_pages=2, blocks=[block, methods])


def test_export_rechecks_evidence_against_the_source_doc():
    record = _record(
        sd_t="2",
        n_t="3",
        evidence={
            "mean_c": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "999"},
            "sd_c": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "2"},
            "mean_t": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "12"},
            "sd_t": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "2"},
            "n_c": {"block_id": "p2.b1", "page": 2, "evidence_text": "in triplicate."},
            "n_t": {"block_id": "p2.b1", "page": 2, "evidence_text": "in triplicate."},
        }
    )
    _, errors = validate_records_for_export([record], CONTINUOUS_TWO_ARM, source_doc=_source_doc())
    assert any("evidence_mean_c_value_not_in_span" in error for error in errors)


def test_n_requires_its_own_methods_citation_when_source_is_checked():
    record = _record(
        sd_t="2",
        n_t="3",
        evidence={
            "mean_c": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "10"},
            "sd_c": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "2"},
            "mean_t": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "12"},
            "sd_t": {"block_id": "p1.b1", "page": 1, "bbox": [0, 0, 200, 20], "evidence_text": "2"},
            "n_c": {"block_id": "p2.b1", "page": 2, "evidence_text": "in triplicate."},
            "n_t": {"block_id": "p2.b1", "page": 2, "evidence_text": "in triplicate."},
        }
    )
    assert not validate_record(record, CONTINUOUS_TWO_ARM, source_doc=_source_doc())


def test_compatible_units_are_converted_and_incompatible_units_block_export():
    mg = _record(unit="mg/kg")
    out = prepare_export_record(mg, CONTINUOUS_TWO_ARM)
    assert out["unit_base"] == "g/kg"
    assert math.isclose(out["mean_c"], 0.01)
    assert validate_unit_comparability([_record(unit="g/kg"), _record(unit="%")]) == [
        "variable <missing>: incompatible_units"
    ]


def test_identity_and_moderator_binding_are_required_for_formal_export():
    record = _record()
    flags = validate_record(
        record,
        CONTINUOUS_TWO_ARM,
        identity_fields=("site", "experiment", "time"),
        require_identity=True,
        moderator_fields=("soil type",),
    )
    assert "missing_record_identity" in flags
    assert "moderators_not_bound_to_record" in flags


def test_prepared_export_flattens_confirmed_identity_and_moderators():
    record = _record(
        record_identity={"site": "S1", "experiment": "E1", "time": "2020"},
        identity_confirmed=True,
        moderator_binding_status="bound",
        moderators={"soil type": "Cambisol"},
    )
    out = prepare_export_record(
        record,
        CONTINUOUS_TWO_ARM,
        identity_fields=("site", "experiment", "time"),
        require_identity=True,
        moderator_fields=("soil type",),
    )
    assert out["site"] == "S1" and out["soil type"] == "Cambisol"


def test_target_coverage_keeps_missing_and_screened_variables_visible():
    rows = target_variable_coverage(
        ["SOC", "TN", "pH"], ["SOC", "SOC"], screened_present=["TN"]
    )
    assert [(r["variable_name"], r["status"], r["located_count"]) for r in rows] == [
        ("SOC", "duplicate", 2),
        ("TN", "screened_present_not_located", 0),
        ("pH", "not_found", 0),
    ]
