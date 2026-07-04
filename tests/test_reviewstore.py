"""Durable review progress: draft round-trips and the audit log appends."""

import csv

from metaextract import reviewstore


def test_load_draft_empty_when_missing(tmp_path):
    draft = reviewstore.load_draft(tmp_path, "P1")
    assert draft == {"records": [], "moderators": {}}


def test_save_then_load_round_trips(tmp_path):
    records = [{"variable_name": "SOC", "mean_t": "12.3", "control_group": "NF"}]
    mods = {"soil type": "Cambisol"}
    saved_at = reviewstore.save_draft(tmp_path, "P1", records, mods)

    draft = reviewstore.load_draft(tmp_path, "P1")
    assert draft["records"] == records
    assert draft["moderators"] == mods
    assert saved_at and draft["saved_at"] == saved_at


def test_save_overwrites_not_appends(tmp_path):
    reviewstore.save_draft(tmp_path, "P1", [{"a": 1}], {})
    reviewstore.save_draft(tmp_path, "P1", [{"a": 1}, {"b": 2}], {})
    # whole-file rewrite: latest state only, no stale duplicates
    assert len(reviewstore.load_draft(tmp_path, "P1")["records"]) == 2


def test_drafts_are_isolated_per_study(tmp_path):
    reviewstore.save_draft(tmp_path, "P1", [{"a": 1}], {})
    reviewstore.save_draft(tmp_path, "P2", [{"b": 2}, {"c": 3}], {})
    assert len(reviewstore.load_draft(tmp_path, "P1")["records"]) == 1
    assert len(reviewstore.load_draft(tmp_path, "P2")["records"]) == 2


def test_log_appends_with_header(tmp_path):
    reviewstore.append_log(tmp_path, "P1", "add_record", variable_name="SOC", records_after=1)
    reviewstore.append_log(tmp_path, "P1", "delete_record", variable_name="SOC", records_after=0)

    with reviewstore.log_path(tmp_path, "P1").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["action"] for r in rows] == ["add_record", "delete_record"]
    assert rows[0]["variable_name"] == "SOC" and rows[0]["timestamp"]


def test_corrupt_draft_is_quarantined_and_reported(tmp_path):
    # An unreadable draft must not silently vanish (that would mask lost work):
    # load reports an error, starts empty, and preserves the bad bytes aside.
    p = reviewstore.draft_path(tmp_path, "P1")
    p.write_text("{not json", encoding="utf-8")

    draft = reviewstore.load_draft(tmp_path, "P1")
    assert draft["records"] == [] and draft["moderators"] == {}
    assert "could not be read" in draft["error"]

    # original is moved aside (preserved for recovery), not left in place or deleted
    assert not p.exists()
    backups = list(reviewstore.reviews_dir(tmp_path).glob("P1.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{not json"


def test_non_object_draft_is_treated_as_corrupt(tmp_path):
    # Valid JSON that isn't an object (e.g. a bare list) used to crash on
    # .setdefault; now it's quarantined like any other unreadable draft.
    reviewstore.draft_path(tmp_path, "P1").write_text("[1, 2, 3]", encoding="utf-8")
    draft = reviewstore.load_draft(tmp_path, "P1")
    assert draft["records"] == [] and "error" in draft


def test_quarantine_lets_a_fresh_draft_save_cleanly(tmp_path):
    # After a corrupt draft is moved aside, the next save round-trips normally.
    reviewstore.draft_path(tmp_path, "P1").write_text("{not json", encoding="utf-8")
    assert "error" in reviewstore.load_draft(tmp_path, "P1")

    reviewstore.save_draft(tmp_path, "P1", [{"a": 1}], {})
    reloaded = reviewstore.load_draft(tmp_path, "P1")
    assert reloaded["records"] == [{"a": 1}] and "error" not in reloaded


def test_stamp_isolates_drafts_for_same_study(tmp_path):
    """Same paper, two task packs (two stamps) -> two independent drafts."""
    reviewstore.save_draft(tmp_path, "P1", [{"a": 1}], {}, stamp="soil.aaaaaaaaaa")
    reviewstore.save_draft(tmp_path, "P1", [{"b": 2}, {"c": 3}], {}, stamp="water.bbbbbbbbbb")

    assert len(reviewstore.load_draft(tmp_path, "P1", stamp="soil.aaaaaaaaaa")["records"]) == 1
    assert len(reviewstore.load_draft(tmp_path, "P1", stamp="water.bbbbbbbbbb")["records"]) == 2
    # an unstamped load (old caches) does not see the stamped drafts
    assert reviewstore.load_draft(tmp_path, "P1")["records"] == []
    # filenames carry the stamp
    assert reviewstore.draft_path(tmp_path, "P1", "soil.aaaaaaaaaa").name == "P1.soil.aaaaaaaaaa.json"
    assert reviewstore.log_path(tmp_path, "P1", "soil.aaaaaaaaaa").name == "P1.soil.aaaaaaaaaa.log.csv"
