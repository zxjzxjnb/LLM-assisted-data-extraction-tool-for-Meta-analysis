"""Task pack: the editable YAML that lets researchers swap target metrics.

No API — covers the pure loader, backward-compatible string coercion, and that
synonyms reach the locator prompt so the canonical name is what gets reported.
"""

from pathlib import Path

import pytest

from metaextract.extractor import taskpack_instruction
from metaextract.locate import SOIL_AGRI_SPEC, TargetVar, TaskSpec, _system_instruction
from metaextract.pipeline import _task_cache_file


def test_bare_strings_still_work():
    """GT-derived specs pass list[str]; they must coerce to TargetVar unchanged."""
    spec = TaskSpec(domain="d", target_variables=["SOC", "pH"], moderators=["site"])
    assert all(isinstance(tv, TargetVar) for tv in spec.target_variables)
    assert spec.variable_names == ["SOC", "pH"]
    assert spec.target_variables[0].aliases == []


def test_rich_entries_keep_aliases_and_units():
    spec = TaskSpec(
        domain="d",
        target_variables=[
            {"name": "soil organic carbon", "aliases": ["SOC", "TOC"], "unit_hint": "g/kg"},
        ],
    )
    tv = spec.target_variables[0]
    assert tv.name == "soil organic carbon"
    assert tv.aliases == ["SOC", "TOC"] and tv.unit_hint == "g/kg"
    assert spec.moderators == []  # optional now


def test_yaml_round_trip(tmp_path):
    spec = TaskSpec(
        domain="my_domain",
        target_variables=[TargetVar(name="MBC", aliases=["microbial biomass C"])],
        moderators=["mean annual temperature"],
    )
    path = tmp_path / "pack.yaml"
    path.write_text(spec.to_yaml(), encoding="utf-8")
    loaded = TaskSpec.from_yaml(path)
    assert loaded.domain == "my_domain"
    assert loaded.variable_names == ["MBC"]
    assert loaded.target_variables[0].aliases == ["microbial biomass C"]


def test_taskpack_rejects_blank_and_duplicate_targets():
    with pytest.raises(ValueError, match="must not be blank"):
        TaskSpec(domain="d", target_variables=[" "])

    with pytest.raises(ValueError, match="duplicate target variable"):
        TaskSpec(domain="d", target_variables=["SOC", {"name": "soc"}])


def test_aliases_are_cleaned():
    spec = TaskSpec(
        domain="d",
        target_variables=[{"name": "SOC", "aliases": [" TOC ", "", "toc"]}],
    )
    assert spec.target_variables[0].aliases == ["TOC"]


def test_synonyms_and_canonical_rule_reach_prompt():
    instr = _system_instruction(SOIL_AGRI_SPEC)
    # a synonym is exposed so the model recognizes alternative wordings
    assert "SOC" in instr and "also written as" in instr
    # and the model is told to normalize back to the canonical name
    assert "canonical" in instr.lower()


def test_taskpack_instruction_reaches_extractor_prompt():
    spec = TaskSpec(
        domain="soil_C_fractions_landuse",
        target_variables=[
            TargetVar(name="soil organic carbon", aliases=["SOC"], unit_hint="g/kg"),
        ],
        moderators=["soil type"],
    )
    instr = taskpack_instruction(spec)
    assert "soil_C_fractions_landuse" in instr
    assert "soil organic carbon" in instr and "SOC" in instr
    assert "Return the canonical name" in instr
    assert "soil type" in instr


def test_taskpack_instruction_omits_moderators_for_foreign_domain():
    """A non-soil domain has no slot in the fixed schema, so moderators are not
    listed (target variables still are)."""
    spec = TaskSpec(
        domain="freshwater_eutrophication",
        target_variables=[TargetVar(name="chlorophyll-a", aliases=["chl-a"])],
        moderators=["lake depth", "residence time"],
    )
    instr = taskpack_instruction(spec)
    assert "chlorophyll-a" in instr and "chl-a" in instr  # targets still apply
    assert "lake depth" not in instr and "residence time" not in instr


def test_taskpack_cache_filename_depends_on_pack(tmp_path):
    pdf = Path("P1.pdf")
    first = TaskSpec(domain="d", target_variables=["SOC"])
    second = TaskSpec(domain="d", target_variables=["MBC"])

    assert _task_cache_file(tmp_path, pdf, None) == tmp_path / "P1.json"
    assert _task_cache_file(tmp_path, pdf, first) != _task_cache_file(tmp_path, pdf, second)


def test_digest_and_cache_stamp():
    a = TaskSpec(domain="soil_C_fractions_landuse", target_variables=["SOC"])
    b = TaskSpec(domain="soil_C_fractions_landuse", target_variables=["MBC"])
    # deterministic + changes when the vocabulary changes
    assert a.digest() == a.digest()
    assert a.digest() != b.digest()
    # stamp = slugged domain + digest; both files (cache, review) share it
    assert a.cache_stamp == f"soil_c_fractions_landuse.{a.digest()}"


def test_seed_taskpack_loads_if_present():
    """The committed seed pack must parse (guards against hand-edits breaking it)."""
    seed = Path(__file__).resolve().parent.parent / "data" / "taskpacks" / "soil_C_fractions.yaml"
    if not seed.exists():
        return
    spec = TaskSpec.from_yaml(seed)
    assert spec.domain == "soil_C_fractions_landuse"
    assert len(spec.target_variables) > 50  # full vocabulary imported
