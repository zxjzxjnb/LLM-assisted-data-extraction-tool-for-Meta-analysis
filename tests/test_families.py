"""Analysis-family registry + its wiring into TaskSpec / LocateOutput."""

import pytest

from metaextract import FAMILIES, get_family
from metaextract.families import BINARY_TWO_ARM, CONTINUOUS_TWO_ARM, DEFAULT_FAMILY
from metaextract.locate import LocateOutput, TaskSpec, _system_instruction
from metaextract.pipeline import run_folder
from metaextract.records import ScreeningResult


def test_registry_and_default():
    assert DEFAULT_FAMILY == "continuous_two_arm"
    assert set(FAMILIES) == {"continuous_two_arm", "binary_two_arm"}
    assert get_family(None) is FAMILIES[DEFAULT_FAMILY]
    assert get_family("binary_two_arm") is BINARY_TWO_ARM


def test_unknown_family_fails_loudly():
    with pytest.raises(KeyError) as exc:
        get_family("survival_hr")
    # the message lists the known families so a typo is self-correcting
    assert "binary_two_arm" in str(exc.value)


def test_continuous_shape():
    fam = CONTINUOUS_TWO_ARM
    assert fam.roles == ["mean_c", "sd_c", "n_c", "mean_t", "sd_t", "n_t"]
    assert fam.pairing is True
    assert fam.has_sample_size is True  # n comes from the replicate finder
    # n fields are pre-filled from methods, not searched for in the table
    n_fields = [f for f in fam.fields if f.is_n]
    assert {f.role for f in n_fields} == {"n_c", "n_t"}
    assert all(not f.highlight for f in n_fields)


def test_binary_shape():
    fam = BINARY_TWO_ARM
    assert fam.roles == ["events_c", "total_c", "events_t", "total_t"]
    assert fam.pairing is False  # make_pairings assumes mean/SD, so no auto-pairing
    assert fam.has_sample_size is False  # totals are read in-table, not from a finder
    assert all(f.highlight for f in fam.fields)


def test_taskspec_accepts_and_rejects_family():
    spec = TaskSpec(domain="d", target_variables=["x"], analysis_family="binary_two_arm")
    assert spec.analysis_family == "binary_two_arm"

    with pytest.raises(Exception):  # pydantic wraps the KeyError as ValidationError
        TaskSpec(domain="d", target_variables=["x"], analysis_family="nope")


def test_default_family_omitted_keeps_cache_stamp_stable():
    """A pack written before analysis_family existed (field absent) must hash the
    same as one that defaults to continuous, so existing caches/drafts survive."""
    explicit = TaskSpec(domain="d", target_variables=["x"], analysis_family=DEFAULT_FAMILY)
    implicit = TaskSpec(domain="d", target_variables=["x"])
    assert explicit.digest() == implicit.digest()
    # a non-default family MUST change the stamp so binary/continuous never mix
    binary = TaskSpec(domain="d", target_variables=["x"], analysis_family="binary_two_arm")
    assert binary.digest() != implicit.digest()


def test_locate_output_carries_family():
    out = LocateOutput(
        screening=ScreeningResult(paper_id="P1", include=True),
        regions=[],
        analysis_family="binary_two_arm",
    )
    assert out.analysis_family == "binary_two_arm"
    # default stays continuous so pre-family caches render unchanged
    legacy = LocateOutput(screening=ScreeningResult(paper_id="P1", include=True), regions=[])
    assert legacy.analysis_family == "continuous_two_arm"


def test_binary_family_reaches_locator_prompt():
    spec = TaskSpec(
        domain="clinical",
        target_variables=["mortality"],
        analysis_family="binary_two_arm",
    )
    instr = _system_instruction(spec)
    assert "binary_two_arm" in instr
    assert "events_t" in instr and "total_t" in instr
    assert "leave sample_size null" in instr
    assert "replicate count n behind the reported means" not in instr


def test_run_folder_rejects_non_default_family_before_api_client(tmp_path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")
    spec = TaskSpec(
        domain="clinical",
        target_variables=["mortality"],
        analysis_family="binary_two_arm",
    )
    with pytest.raises(ValueError, match="fixed continuous mean/SD/n schema"):
        run_folder(pdf_dir, tmp_path / "out.csv", task_spec=spec)
