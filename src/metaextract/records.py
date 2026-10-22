"""The three-layer extraction model (Architecture v3.1 §4).

A record only exists *after* a human pairs and verifies it, so the pipeline keeps
three distinct artifacts instead of v3.0's single pre-paired ``LocatedRecord``:

    LocatedRegion   Tier 2 output — "the target variable is in this block".
                    No numbers, no treatment/control pairing.
    ExtractionSlot  a worksheet cell waiting for one value.
    VerifiedRecord  materialized only when a human pairs slots into a
                    treatment-vs-control point (filled here for completeness; the
                    cockpit milestone populates it).

Every artifact that points into the PDF does so through :class:`Citation`, whose
geometry comes from deterministic ingest (``sourcedoc.BBox``), never the LLM.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .sourcedoc import BBox

# A slot's role is one of the active analysis family's field roles (see
# families.py). It is a free ``str`` rather than a fixed Literal so a binary or
# correlation family can carry its own roles (events_t, r, …) without editing
# this model; the family registry is the single source of truth for valid roles.
SlotRole = str


class Citation(BaseModel):
    """A jumpable, verifiable pointer into the source PDF (v3.1 §4, C1)."""

    block_id: str
    page: int
    bbox: BBox
    kind: Literal["paragraph", "figure"]
    evidence_text: Optional[str] = None  # resolved to word bbox by highlight.locate_value


class LocatedRegion(BaseModel):
    """Tier 2 output: a target variable believed present in one block.

    Deliberately carries NO numbers and NO pairing — that is exactly the matching
    work v3.1 defers to a human in the cockpit.
    """

    region_id: str
    paper_id: str
    variable_name: str
    citation: Citation
    candidate_structure: Optional[str] = None  # free-text hint, not a commitment
    ambiguous: bool = False  # model unsure the variable is really here


class ExtractionSlot(BaseModel):
    """One empty worksheet cell belonging to a region (filled by human / Tier 3)."""

    slot_id: str
    region_id: str
    role: SlotRole
    citation: Optional[Citation] = None
    value: Optional[float | int] = None  # None = located, not yet filled
    unit: Optional[str] = None
    source: Literal["empty", "human", "ai_extracted"] = "empty"
    grounding_verified: bool = False  # set by the grounding stage, never the model
    flags: list[str] = Field(default_factory=list)


class VerifiedRecord(BaseModel):
    """Materialized only after a human pairs + verifies slots (v3.1 §4)."""

    record_id: str
    paper_id: str
    source_block_ids: list[str] = Field(default_factory=list)
    variable_name: str
    treatment_group: str
    control_group: str
    slots: dict[str, ExtractionSlot] = Field(default_factory=dict)
    moderator_binding_status: Literal["unbound", "bound"] = "unbound"
    status: Literal["needs_review", "verified", "invalid"] = "needs_review"
    flags: list[str] = Field(default_factory=list)
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None  # ISO timestamp


class ScreeningResult(BaseModel):
    """Tier 1 output: a paper-level worksheet (no record-level binding)."""

    paper_id: str
    include: bool
    reason: Optional[str] = None
    target_variables_present: list[str] = Field(default_factory=list)
    moderators_present: list[str] = Field(default_factory=list)


class ExtractedModerator(BaseModel):
    """A paper/site-level moderator value the model read, kept verifiable.

    These are *suggestions*: the human confirms them in the cockpit and chooses to
    bind them (v3.1 §2 — AI may screen moderators, but binding to records is human).
    ``citation`` is set when the cited block_id exists, so the value is jumpable.
    """

    field: str
    value: str
    citation: Optional[Citation] = None


class SampleSize(BaseModel):
    """The replicate count (n) — usually one statement in methods/footnote, not the
    data table. A suggestion the human confirms; ``citation`` makes it jumpable."""

    value: Optional[str] = None  # the stated n / replicate count, e.g. "3"
    n_control: Optional[str] = None  # if control/treatment differ
    n_treatment: Optional[str] = None
    note: Optional[str] = None  # the surrounding phrase, for context
    citation: Optional[Citation] = None
