"""Flatten a nested :class:`ExtractionResult` into tidy long-format rows.

One row per response-variable data point, with the study-level moderators
repeated across rows (denormalised) so the output is a single analysis-ready
table that an effect-size script can read directly.
"""

from __future__ import annotations

from .schema import ExtractionResult
from .validator import validate_result


FLAT_COLUMNS = [
    "source_file", "first_author", "year", "site_name", "coordinates", "country",
    "mean_annual_temp_c", "mean_annual_precip_mm", "soil_type_wrb", "soil_texture",
    "initial_ph", "initial_soc_g_kg", "exp_duration_years", "treatments_overview",
    "variable_name", "unit", "treatment_group", "control_group", "mean_t", "sd_t",
    "n_t", "mean_c", "sd_c", "n_c", "source", "sampling_depth_cm",
    "measurement_year", "qa_flags",
]


def result_to_rows(result: ExtractionResult, source_file: str) -> list[dict]:
    """Convert one paper's result into a list of flat row dicts."""
    si = result.study_info
    mod = result.moderator_variables
    treatments = "; ".join(
        f"{t.name}: {t.description}".strip(": ")
        for t in mod.experimental_design.treatments_overview
    )

    base = {
        "source_file": source_file,
        "first_author": si.first_author,
        "year": si.year,
        "site_name": mod.location.site_name,
        "coordinates": mod.location.coordinates,
        "country": mod.location.country,
        "mean_annual_temp_c": mod.climate.mean_annual_temperature_c,
        "mean_annual_precip_mm": mod.climate.mean_annual_precipitation_mm,
        "soil_type_wrb": mod.soil_background.soil_type_wrb,
        "soil_texture": mod.soil_background.texture,
        "initial_ph": mod.soil_background.initial_ph,
        "initial_soc_g_kg": mod.soil_background.initial_soc_g_kg,
        "exp_duration_years": mod.experimental_design.duration_years,
        "treatments_overview": treatments,
    }

    flags = validate_result(result)
    rows: list[dict] = []
    for i, rv in enumerate(result.response_variables):
        row = dict(base)
        row.update(
            {
                "variable_name": rv.variable_name,
                "unit": rv.unit,
                "treatment_group": rv.treatment_group,
                "control_group": rv.control_group,
                "mean_t": rv.mean_t,
                "sd_t": rv.sd_t,
                "n_t": rv.n_t,
                "mean_c": rv.mean_c,
                "sd_c": rv.sd_c,
                "n_c": rv.n_c,
                "source": rv.context.source,
                "sampling_depth_cm": rv.context.sampling_depth_cm,
                "measurement_year": rv.context.measurement_year,
                "qa_flags": "; ".join(flags.get(i, []) + flags.get(-1, [])),
            }
        )
        rows.append(row)

    if not rows:
        # Preserve the study even when no paired data point was found, so it is
        # visible for manual review instead of vanishing.
        row = dict(base)
        row["qa_flags"] = "no_response_variables"
        rows.append(row)

    return rows
