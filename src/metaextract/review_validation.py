"""Validation and normalization for cockpit review records."""

from __future__ import annotations

import math
from typing import Any

from .families import AnalysisFamily
from .extract import ground_verify
from .highlight import locate_value_in_span, parse_number
from .sourcedoc import SourceDoc


_UNIT_ALIASES = {
    "g kg-1": "g/kg",
    "g kg−1": "g/kg",
    "g/kg": "g/kg",
    "mg kg-1": "mg/kg",
    "mg kg−1": "mg/kg",
    "mg/kg": "mg/kg",
    "mg g-1": "mg/g",
    "mg g−1": "mg/g",
    "mg/g": "mg/g",
    "ug/kg": "ug/kg",
    "μg kg-1": "ug/kg",
    "µg kg-1": "ug/kg",
    "ug g-1": "mg/kg",
    "μg g-1": "mg/kg",
    "µg g-1": "mg/kg",
    "%": "%",
    "g m-2": "g/m2",
    "g m−2": "g/m2",
    "g/m2": "g/m2",
    "mg m-2": "mg/m2",
    "mg m−2": "mg/m2",
    "mg/m2": "mg/m2",
    "kg ha-1": "kg/ha",
    "kg ha−1": "kg/ha",
    "kg/ha": "kg/ha",
}

# Canonical unit -> (dimension, multiplier to the canonical base unit).  The
# multiplier is deliberately explicit: normalising spelling alone can make a
# CSV look consistent while mixing values that differ by 1,000.
_UNIT_TABLE = {
    "g/kg": ("mass_fraction", 1.0, "g/kg"),
    "mg/kg": ("mass_fraction", 1e-3, "g/kg"),
    "ug/kg": ("mass_fraction", 1e-6, "g/kg"),
    "mg/g": ("mass_fraction", 1.0, "g/kg"),
    "ug/g": ("mass_fraction", 1e-3, "g/kg"),
    "ng/g": ("mass_fraction", 1e-6, "g/kg"),
    "%": ("mass_fraction_percent", 1.0, "%"),
    "g/m2": ("mass_area", 1.0, "g/m2"),
    "mg/m2": ("mass_area", 1e-3, "g/m2"),
    "kg/ha": ("mass_area", 0.1, "g/m2"),
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float | None:
    if value is None or not _text(value):
        return None
    parsed = parse_number(_text(value))
    if parsed is None or not math.isfinite(parsed):
        return None
    return parsed


def normalize_unit(value: Any) -> str:
    """Normalize common unit spellings without changing their magnitude."""
    text = " ".join(_text(value).replace("⋅", " ").split()).casefold()
    return _UNIT_ALIASES.get(text, text)


def unit_spec(value: Any) -> tuple[str, str, float] | None:
    """Return ``(canonical, dimension, factor_to_base)`` for a controlled unit."""
    normalized = normalize_unit(value)
    # A few spellings are intentionally accepted only after punctuation cleanup.
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    item = _UNIT_TABLE.get(normalized)
    if item is None:
        return None
    dimension, factor, base = item
    return normalized, dimension, factor


def record_identity(record: dict[str, Any], identity_fields: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Return a stable identity tuple for a record, suitable for moderator keys."""
    identity = record.get("record_identity")
    if isinstance(identity, dict):
        values = tuple(_text(identity.get(field)) for field in identity_fields)
    else:
        values = tuple(_text(record.get(field)) for field in identity_fields)
    return values


def target_variable_coverage(
    expected: list[str], located: list[str], *, screened_present: list[str] | None = None
) -> list[dict[str, Any]]:
    """Build an auditable expected-vs-located matrix for a paper.

    Every expected variable receives a row, including variables the locator did
    not return.  Duplicate located regions remain visible so a reviewer can spot
    over-location rather than silently collapsing it.
    """
    expected_unique: list[str] = []
    seen: set[str] = set()
    for value in expected:
        key = _text(value).casefold()
        if key and key not in seen:
            seen.add(key)
            expected_unique.append(_text(value))
    counts: dict[str, int] = {}
    for value in located:
        key = _text(value).casefold()
        counts[key] = counts.get(key, 0) + 1
    present = {_text(value).casefold() for value in (screened_present or [])}
    rows: list[dict[str, Any]] = []
    for value in expected_unique:
        key = value.casefold()
        count = counts.get(key, 0)
        rows.append({
            "variable_name": value,
            "status": (
                "duplicate" if count > 1 else
                "located" if count == 1 else
                "screened_present_not_located" if key in present else "not_found"
            ),
            "located_count": count,
        })
    return rows


def validate_unit_comparability(records: list[dict[str, Any]]) -> list[str]:
    """Return blocking unit flags grouped by target variable.

    Compatible units are converted during export. Unknown units and incompatible
    dimensions block export because a meta-analysis cannot combine them safely.
    """
    flags: list[str] = []
    by_variable: dict[str, list[tuple[int, tuple[str, str, float] | None]]] = {}
    for i, record in enumerate(records):
        key = _text(record.get("variable_name")).casefold() or "<missing>"
        by_variable.setdefault(key, []).append((i, unit_spec(record.get("unit"))))
    for variable, entries in by_variable.items():
        specs = [(i, spec) for i, spec in entries if spec is not None]
        for i, spec in entries:
            if spec is None:
                flags.append(f"row {i}: unknown_unit")
        dimensions = {spec[1] for _, spec in specs}
        if len(dimensions) > 1:
            flags.append(f"variable {variable}: incompatible_units")
    return flags


def validate_record(
    record: dict[str, Any],
    family: AnalysisFamily,
    *,
    source_doc: SourceDoc | None = None,
    identity_fields: tuple[str, ...] = (),
    require_identity: bool = False,
    moderator_fields: tuple[str, ...] = (),
) -> list[str]:
    """Return blocking flags for a cockpit record.

    When ``source_doc`` is supplied, evidence is re-checked against the actual
    cited block through the same grounding gate used by Tier 3 and the agent.
    ``require_identity`` is enabled by the cockpit export path so paper-level
    moderator suggestions cannot silently become row-level facts.
    """
    flags: list[str] = []
    for field in family.fields:
        raw = record.get(field.role)
        if not _text(raw):
            flags.append(f"missing_{field.role}")
            continue
        value = _number(raw)
        if value is None:
            flags.append(f"non_numeric_{field.role}")
            continue
        if field.role.startswith("sd_") and value < 0:
            flags.append(f"negative_{field.role}")
        if field.is_n and (value <= 0 or value != int(value)):
            flags.append(f"invalid_{field.role}")

    control = _text(record.get("control_group"))
    treatment = _text(record.get("treatment_group"))
    if not control:
        flags.append("missing_control_group")
    if not treatment:
        flags.append("missing_treatment_group")
    if control and treatment and control.casefold() == treatment.casefold():
        flags.append("treatment_equals_control")
    if not _text(record.get("unit")):
        flags.append("missing_unit")
    elif unit_spec(record.get("unit")) is None:
        flags.append("unknown_unit")

    sd_roles = [f.role for f in family.fields if f.role.startswith("sd_")]
    if sd_roles and _text(record.get("dispersion")) not in {"sd", "se"}:
        flags.append("missing_or_invalid_dispersion")

    if record.get("source") == "figure_digitized":
        if _text(record.get("status")) != "digitized":
            flags.append("figure_not_digitized")
        if not _text(record.get("digitizer_ref")):
            flags.append("missing_digitizer_ref")
        # n still comes from a methods statement for figure records.  Keep the
        # digitizer branch while applying the same source check to n fields.
        if source_doc is not None:
            evidence = record.get("evidence")
            for field in family.fields:
                if field.is_n:
                    entry = evidence.get(field.role) if isinstance(evidence, dict) else None
                    if not isinstance(entry, dict) or not _text(entry.get("evidence_text")):
                        flags.append(f"missing_evidence_{field.role}")
                        continue
                    block = source_doc.block(_text(entry.get("block_id")))
                    if block is None:
                        flags.append(f"evidence_block_missing_{field.role}")
                        continue
                    grounding = ground_verify(_text(record.get(field.role)), _text(entry.get("evidence_text")), block)
                    flags.extend(f"evidence_{field.role}_{flag}" for flag in grounding.flags)
    else:
        evidence = record.get("evidence")
        for field in family.fields:
            entry = evidence.get(field.role) if isinstance(evidence, dict) else None
            # n is not a table value, but it still needs a citation to a methods
            # statement before a record can enter a formal export.
            if not field.highlight and source_doc is None:
                continue
            if not isinstance(entry, dict) or not _text(entry.get("evidence_text")):
                flags.append(f"missing_evidence_{field.role}")
                continue
            if source_doc is not None:
                block = source_doc.block(_text(entry.get("block_id")))
                if block is None:
                    flags.append(f"evidence_block_missing_{field.role}")
                    continue
                grounding = ground_verify(_text(record.get(field.role)), _text(entry.get("evidence_text")), block)
                flags.extend(f"evidence_{field.role}_{flag}" for flag in grounding.flags)
                if entry.get("page") is not None and int(entry["page"]) != block.page:
                    flags.append(f"evidence_page_mismatch_{field.role}")
                if field.highlight:
                    bbox = entry.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        flags.append(f"missing_evidence_bbox_{field.role}")
                    elif block.words:
                        match = locate_value_in_span(
                            block, _text(record.get(field.role)), _text(entry.get("evidence_text"))
                        )
                        if match is None:
                            flags.append(f"evidence_bbox_unresolvable_{field.role}")
                        elif any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(bbox, match.bbox.as_tuple())):
                            flags.append(f"evidence_bbox_mismatch_{field.role}")

    if require_identity:
        identity = record_identity(record, identity_fields)
        if any(not value for value in identity):
            flags.append("missing_record_identity")
        if record.get("identity_confirmed") is not True:
            flags.append("identity_not_confirmed")
        if _text(record.get("moderator_binding_status")) != "bound":
            flags.append("moderators_not_bound_to_record")
        moderators = record.get("moderators") if isinstance(record.get("moderators"), dict) else {}
        for field in moderator_fields:
            if not _text(moderators.get(field)):
                flags.append(f"missing_bound_moderator_{field}")
    return flags


def prepare_export_record(
    record: dict[str, Any],
    family: AnalysisFamily,
    *,
    source_doc: SourceDoc | None = None,
    identity_fields: tuple[str, ...] = (),
    require_identity: bool = False,
    moderator_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return a validated copy, converting SE to SD while preserving input."""
    flags = validate_record(
        record,
        family,
        source_doc=source_doc,
        identity_fields=identity_fields,
        require_identity=require_identity,
        moderator_fields=moderator_fields,
    )
    if flags:
        raise ValueError("record is not exportable: " + ", ".join(flags))

    out = dict(record)
    identity = record.get("record_identity")
    if isinstance(identity, dict):
        for field, value in identity.items():
            if _text(value):
                out[field] = value
    bound_moderators = record.get("moderators")
    if isinstance(bound_moderators, dict):
        # Keep the auditable nested object and also expose one CSV column per
        # moderator so downstream meta-analysis code can group by it directly.
        for field, value in bound_moderators.items():
            if _text(value):
                out[field] = value
    spec = unit_spec(record.get("unit"))
    assert spec is not None
    normalized, dimension, factor = spec
    out["unit_original"] = record.get("unit")
    out["unit_normalized"] = normalized
    out["unit_base"] = _UNIT_TABLE[normalized][2]
    out["unit_dimension"] = dimension
    out["unit_conversion_factor"] = factor
    if factor != 1.0:
        for field in family.fields:
            if field.is_n:
                continue
            value = _number(record.get(field.role))
            if value is not None:
                out[f"{field.role}_original"] = record[field.role]
                out[field.role] = value * factor
    dispersion = _text(record.get("dispersion"))
    out["dispersion_input"] = dispersion
    if dispersion == "se":
        for sd_role in (f.role for f in family.fields if f.role.startswith("sd_")):
            n_role = sd_role.replace("sd_", "n_", 1)
            se = _number(record.get(sd_role))
            n = _number(record.get(n_role))
            assert se is not None and n is not None
            out[f"{sd_role}_original"] = record[sd_role]
            out[sd_role] = se * math.sqrt(n)
        out["dispersion"] = "sd"
    return out


def validate_records_for_export(
    records: list[dict[str, Any]],
    family: AnalysisFamily,
    *,
    source_doc: SourceDoc | None = None,
    identity_fields: tuple[str, ...] = (),
    require_identity: bool = False,
    moderator_fields: tuple[str, ...] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate and prepare a complete export in one shared entry point."""
    errors: list[str] = []
    prepared: list[dict[str, Any]] = []
    errors.extend(validate_unit_comparability(records))
    for index, record in enumerate(records):
        flags = validate_record(
            record,
            family,
            source_doc=source_doc,
            identity_fields=identity_fields,
            require_identity=require_identity,
            moderator_fields=moderator_fields,
        )
        if flags:
            errors.append(f"row {index}: {', '.join(flags)}")
            continue
        try:
            prepared.append(
                prepare_export_record(
                    record,
                    family,
                    source_doc=source_doc,
                    identity_fields=identity_fields,
                    require_identity=require_identity,
                    moderator_fields=moderator_fields,
                )
            )
        except ValueError as exc:
            errors.append(f"row {index}: {exc}")
    return prepared, errors
