"""Cockpit cache isolation across task packs / analysis families."""

import json
import hashlib

import pytest

from metaextract import cockpit_cache
from metaextract.locate import LocateOutput
from metaextract.records import ScreeningResult
from metaextract.sourcedoc import SourceDoc


def _doc(study: str) -> SourceDoc:
    return SourceDoc(doc_id=study, n_pages=1)


def _out(study: str, stamp: str) -> LocateOutput:
    return LocateOutput(
        screening=ScreeningResult(paper_id=study, include=True),
        regions=[],
        task_stamp=stamp,
        analysis_family="binary_two_arm" if stamp.endswith("binary") else "continuous_two_arm",
    )


def test_papers_list_in_natural_not_string_order(tmp_path):
    # The cockpit dropdown shows these in order; string sort would read
    # P1, P10, P11, P2 — confusing in a demo. Natural sort reads P1, P2, … P10.
    for s in ["P1", "P10", "P11", "P2", "P9"]:
        cockpit_cache.save(tmp_path, s, _doc(s), _out(s, "soil.aaa"))
    order = [e.study for e in cockpit_cache.list_cache_entries(tmp_path)]
    assert order == ["P1", "P2", "P9", "P10", "P11"]


def test_stamped_locate_caches_can_coexist_for_one_study(tmp_path):
    root = tmp_path
    cockpit_cache.save(root, "P1", _doc("P1"), _out("P1", "soil.aaa"))
    cockpit_cache.save(root, "P1", _doc("P1"), _out("P1", "clinical.binary"))

    entries = cockpit_cache.list_cache_entries(root)
    assert [(e.study, e.stamp) for e in entries] == [
        ("P1", "clinical.binary"),
        ("P1", "soil.aaa"),
    ]

    _doc1, soil = cockpit_cache.load(root, "P1", stamp="soil.aaa")
    _doc2, clinical = cockpit_cache.load(root, "P1", stamp="clinical.binary")
    assert soil.task_stamp == "soil.aaa"
    assert clinical.task_stamp == "clinical.binary"
    assert clinical.analysis_family == "binary_two_arm"


def test_legacy_locate_cache_still_loads(tmp_path):
    root = tmp_path
    d = cockpit_cache.cache_dir(root)
    (d / "P1.sourcedoc.json").write_text(_doc("P1").model_dump_json())
    legacy = LocateOutput(screening=ScreeningResult(paper_id="P1", include=True), regions=[])
    (d / "P1.locate.json").write_text(legacy.model_dump_json())

    entries = cockpit_cache.list_cache_entries(root)
    assert [(e.study, e.stamp) for e in entries] == [("P1", None)]
    loaded_doc, out = cockpit_cache.load(root, "P1")
    assert loaded_doc.doc_id == "P1"
    assert out.analysis_family == "continuous_two_arm"


def test_stampless_load_falls_back_to_the_one_stamped_cache(tmp_path):
    """The eval/benchmark scripts call load(root, study) with no stamp, but
    prepare_paper now writes a stamped file. With no legacy file present, the
    stampless load must resolve to the single stamped cache instead of raising
    FileNotFoundError or silently reading a stale legacy file."""
    root = tmp_path
    cockpit_cache.save(root, "P1", _doc("P1"), _out("P1", "soil.aaa"))
    # no P1.locate.json on disk — only the stamped one
    assert not (cockpit_cache.cache_dir(root) / "P1.locate.json").exists()

    _doc1, out = cockpit_cache.load(root, "P1")  # the scripts' call shape
    assert out.task_stamp == "soil.aaa"


def test_stampless_load_refuses_when_several_task_packs_cached(tmp_path):
    """Picking arbitrarily between task packs would corrupt an eval, so a
    stampless load with multiple stamped caches and no legacy must fail loudly."""
    root = tmp_path
    cockpit_cache.save(root, "P1", _doc("P1"), _out("P1", "soil.aaa"))
    cockpit_cache.save(root, "P1", _doc("P1"), _out("P1", "clinical.binary"))

    with pytest.raises(ValueError, match="pass stamp="):
        cockpit_cache.load(root, "P1")


def test_saved_locate_cache_carries_the_schema_version(tmp_path):
    cockpit_cache.save(tmp_path, "P1", _doc("P1"), _out("P1", "soil.aaa"))
    path = cockpit_cache.cache_dir(tmp_path) / "P1.soil.aaa.locate.json"
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == cockpit_cache.CACHE_SCHEMA_VERSION
    # the stamped version is invisible to the model (extra key, ignored on load)
    _doc1, out = cockpit_cache.load(tmp_path, "P1", stamp="soil.aaa")
    assert out.task_stamp == "soil.aaa"


def test_load_rejects_a_cache_from_a_newer_version(tmp_path):
    cockpit_cache.save(tmp_path, "P1", _doc("P1"), _out("P1", "soil.aaa"))
    path = cockpit_cache.cache_dir(tmp_path) / "P1.soil.aaa.locate.json"
    raw = json.loads(path.read_text())
    raw["schema_version"] = cockpit_cache.CACHE_SCHEMA_VERSION + 99
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(cockpit_cache.CacheError, match="newer MetaExtract"):
        cockpit_cache.load(tmp_path, "P1", stamp="soil.aaa")


def test_load_translates_a_corrupt_locate_cache(tmp_path):
    cockpit_cache.save(tmp_path, "P1", _doc("P1"), _out("P1", "soil.aaa"))
    path = cockpit_cache.cache_dir(tmp_path) / "P1.soil.aaa.locate.json"
    path.write_text("{truncated", encoding="utf-8")

    with pytest.raises(cockpit_cache.CacheError, match="prepare_paper"):
        cockpit_cache.load(tmp_path, "P1", stamp="soil.aaa")


def test_load_translates_a_corrupt_sourcedoc(tmp_path):
    cockpit_cache.save(tmp_path, "P1", _doc("P1"), _out("P1", "soil.aaa"))
    (cockpit_cache.cache_dir(tmp_path) / "P1.sourcedoc.json").write_text(
        "not json", encoding="utf-8"
    )
    with pytest.raises(cockpit_cache.CacheError, match="prepare_paper"):
        cockpit_cache.load(tmp_path, "P1", stamp="soil.aaa")


def test_load_reports_a_missing_cache_clearly(tmp_path):
    # nothing prepared for P1 yet
    with pytest.raises(cockpit_cache.CacheError, match="missing"):
        cockpit_cache.load(tmp_path, "P1", stamp="soil.aaa")


def test_stamped_save_supersedes_a_stale_legacy_cache(tmp_path):
    """An old unstamped P1.locate.json on disk must not survive a fresh stamped
    prepare — otherwise a stampless load keeps reading the stale legacy."""
    root = tmp_path
    d = cockpit_cache.cache_dir(root)
    (d / "P1.sourcedoc.json").write_text(_doc("P1").model_dump_json())
    stale = LocateOutput(screening=ScreeningResult(paper_id="P1", include=True), regions=[])
    (d / "P1.locate.json").write_text(stale.model_dump_json())

    cockpit_cache.save(root, "P1", _doc("P1"), _out("P1", "soil.aaa"))

    assert not (d / "P1.locate.json").exists()  # legacy removed
    entries = cockpit_cache.list_cache_entries(root)
    assert [(e.study, e.stamp) for e in entries] == [("P1", "soil.aaa")]  # no duplicate
    _doc1, out = cockpit_cache.load(root, "P1")  # stampless eval-script call
    assert out.task_stamp == "soil.aaa"  # fresh, not stale


def test_load_rejects_a_changed_source_pdf(tmp_path):
    pdf_dir = tmp_path / "data" / "sample_papers"
    pdf_dir.mkdir(parents=True)
    pdf = pdf_dir / "P1.pdf"
    pdf.write_bytes(b"source-a")
    digest = hashlib.sha256(b"source-a").hexdigest()
    doc = _doc("P1").model_copy(update={"source_sha256": digest})
    out = _out("P1", "soil.aaa").model_copy(update={"source_sha256": digest})
    cockpit_cache.save(tmp_path, "P1", doc, out)

    pdf.write_bytes(b"source-b")
    with pytest.raises(cockpit_cache.CacheError, match="changed after ingest"):
        cockpit_cache.load(tmp_path, "P1", stamp="soil.aaa")
