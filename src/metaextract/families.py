"""Analysis families: the per-record field set is data, not hard-coded UI.

Different meta-analysis designs need different numbers per record. A *continuous
two-arm* design (treatment-vs-control means) wants mean/SD/n per arm — what feeds
a log response ratio or Hedges' g. A *binary two-arm* design (e.g. odds/risk
ratio) wants events/total per arm instead. The family is the single knob a task
pack sets (``analysis_family``); everything downstream — the cockpit input form,
the slot roles, the export columns — is generated from the family's field list,
so supporting a new design means adding a family here, never editing the cockpit
by hand.

Only ``continuous_two_arm`` is validated on a real corpus (the soil/land-use
papers P1-P12). The others are structurally supported but UNTESTED on real
papers; see the README. Keep that honesty: a family existing here is not a claim
that locate/extract was ever measured for it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Arm = Literal["control", "treatment", "shared"]


class RecordField(BaseModel):
    """One number a record carries for a given analysis family.

    ``role`` is the dict key the cockpit writes and the CSV column name, so it is
    the stable identifier (e.g. ``mean_c``, ``events_t``). ``is_n`` marks a value
    that comes from the methods/sample-size finder rather than the data table, so
    the cockpit pre-fills it from the detected n and does not try to highlight it
    on the page. ``highlight`` marks a value the human should be able to find in
    the source (a table number), so the cockpit lights it up in the PDF.
    """

    role: str
    label: str  # short header shown in the cockpit form, e.g. "Xc"
    arm: Arm
    is_n: bool = False
    highlight: bool = True


class AnalysisFamily(BaseModel):
    """A meta-analysis design = the set of per-record fields it needs."""

    name: str
    label: str
    fields: list[RecordField]
    # control x treatment auto-pairing (tabular.make_pairings) assumes mean/SD
    # columns, so it only applies to continuous designs. Other families fall back
    # to manual entry.
    pairing: bool = False

    @property
    def required_roles(self) -> list[str]:
        """Roles that must be present before a record is analysis-eligible."""
        return [f.role for f in self.fields]

    @property
    def roles(self) -> list[str]:
        return [f.role for f in self.fields]

    @property
    def has_sample_size(self) -> bool:
        """True when some field is fed by the replicate-count finder (so the
        cockpit shows the n picker). False families simply read totals in-table."""
        return any(f.is_n for f in self.fields)


CONTINUOUS_TWO_ARM = AnalysisFamily(
    name="continuous_two_arm",
    label="Continuous, treatment vs control — mean/SD/n per arm (lnRR, SMD/Hedges' g)",
    fields=[
        RecordField(role="mean_c", label="Xc", arm="control"),
        RecordField(role="sd_c", label="Sc", arm="control"),
        RecordField(role="n_c", label="Nc", arm="control", is_n=True, highlight=False),
        RecordField(role="mean_t", label="Xe", arm="treatment"),
        RecordField(role="sd_t", label="Se", arm="treatment"),
        RecordField(role="n_t", label="Ne", arm="treatment", is_n=True, highlight=False),
    ],
    pairing=True,
)

# EXPERIMENTAL — structurally supported, never measured on real binary-outcome
# papers. Totals are read from the table per arm (not from the replicate finder),
# so they highlight like any other table number and there is no separate n field.
BINARY_TWO_ARM = AnalysisFamily(
    name="binary_two_arm",
    label="Binary outcome, two arms — events/total per arm (OR, RR) [EXPERIMENTAL]",
    fields=[
        RecordField(role="events_c", label="events (ctrl)", arm="control"),
        RecordField(role="total_c", label="total (ctrl)", arm="control"),
        RecordField(role="events_t", label="events (trt)", arm="treatment"),
        RecordField(role="total_t", label="total (trt)", arm="treatment"),
    ],
    pairing=False,
)

DEFAULT_FAMILY = "continuous_two_arm"

FAMILIES: dict[str, AnalysisFamily] = {
    f.name: f for f in (CONTINUOUS_TWO_ARM, BINARY_TWO_ARM)
}


def get_family(name: str | None) -> AnalysisFamily:
    """Resolve a family name to its definition; ``None`` -> the default.

    Raises ``KeyError`` with the known names on an unknown family so a typo in a
    task pack fails loudly instead of silently extracting the wrong shape.
    """
    key = name or DEFAULT_FAMILY
    try:
        return FAMILIES[key]
    except KeyError:
        known = ", ".join(sorted(FAMILIES))
        raise KeyError(f"unknown analysis_family {key!r}; known families: {known}") from None


# Structural keys the cockpit always writes into a record dict (the values that
# become CSV columns alongside the family's value fields). A task pack's
# ``record_covariates`` must not collide with any of these — nor with a family
# role or a paper-level moderator — or the covariate would overwrite real data
# when records are merged for export (``{**bound, **record}``). The TaskSpec
# validator checks this so a bad pack fails loudly instead of silently dropping a
# column on export.
RESERVED_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "paper_id",
        "variable_name",
        "source_block_id",
        "page",
        "kind",
        "control_group",
        "treatment_group",
        "unit",
        "dispersion",
        "evidence",
        "source",
        "status",
        "digitizer_ref",
        "moderator_binding_status",
        "record_identity",
        "identity_confirmed",
        "moderators",
        "n_evidence",
        "qa_flags",
        "review_status",
    }
)
