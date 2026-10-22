"""Per-record covariates: task-pack-declared fields the cockpit captures per
record and exports as their own columns (e.g. soil depth measured at several
depths for one variable). No API — pure schema/validation/digest behavior.
"""

from pathlib import Path

import pytest

from metaextract.locate import LocateOutput, TaskSpec
from metaextract.records import ScreeningResult


def _spec(**kw):
    kw.setdefault("domain", "d")
    kw.setdefault("target_variables", ["x"])
    return TaskSpec(**kw)


def test_covariates_default_empty_and_strip():
    assert _spec().record_covariates == []
    assert _spec(record_covariates=["  Depth cm  ", "fraction"]).record_covariates == [
        "Depth cm",
        "fraction",
    ]


def test_reject_blank_and_duplicate_covariates():
    with pytest.raises(ValueError, match="must not be blank"):
        _spec(record_covariates=["depth", "  "])
    with pytest.raises(ValueError, match="duplicate record covariate"):
        _spec(record_covariates=["depth", "Depth"])  # case-insensitive


@pytest.mark.parametrize("bad", ["unit", "Control_group", "mean_c", "site"])
def test_reject_collision_with_reserved_role_or_moderator(bad):
    # 'unit'/'control_group' are structural columns, 'mean_c' a continuous value
    # field, 'site' a declared moderator — a covariate must not shadow any of them.
    with pytest.raises(ValueError, match="collides"):
        _spec(record_covariates=[bad], moderators=["site"])


def test_empty_covariates_keep_cache_stamp_stable():
    """A pack with no covariates must hash like one written before the field
    existed, so existing soil caches/drafts survive untouched."""
    assert _spec().digest() == _spec(record_covariates=[]).digest()
    # declaring a covariate changes the vocabulary -> new stamp (records gain a
    # column; never mix them with covariate-free records under one stamp).
    assert _spec(record_covariates=["Depth cm"]).digest() != _spec().digest()


def test_locate_output_carries_covariates_and_legacy_default():
    out = LocateOutput(
        screening=ScreeningResult(paper_id="P1", include=True),
        regions=[],
        record_covariates=["Depth cm"],
    )
    assert out.record_covariates == ["Depth cm"]
    # a cache JSON written before covariates existed loads with an empty list,
    # so the cockpit form looks exactly as before.
    legacy = LocateOutput.model_validate_json(
        '{"screening": {"paper_id": "P1", "include": true}, "regions": []}'
    )
    assert legacy.record_covariates == []
    # round-trips through the cache serialization
    assert LocateOutput.model_validate_json(out.model_dump_json()).record_covariates == [
        "Depth cm"
    ]


def test_yaml_round_trip_preserves_covariates(tmp_path):
    spec = _spec(record_covariates=["Depth cm"], moderators=["site"])
    path = tmp_path / "pack.yaml"
    path.write_text(spec.to_yaml(), encoding="utf-8")
    assert TaskSpec.from_yaml(path).record_covariates == ["Depth cm"]


def test_seed_pack_models_depth_as_per_record_covariate():
    """Depth varies row to row, so the seed pack must carry it as a covariate, not
    a paper-level moderator (which would bind one depth to every record)."""
    seed = Path(__file__).resolve().parent.parent / "data" / "taskpacks" / "soil_C_fractions.yaml"
    if not seed.exists():
        return
    spec = TaskSpec.from_yaml(seed)
    assert "Depth cm" in spec.record_covariates
    assert "Depth cm" not in spec.moderators
