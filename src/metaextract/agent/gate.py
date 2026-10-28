"""The agent's commit gate — inherited verbatim from the Tier-3 grounding check.

The agent has exactly one rule it cannot break: a number only becomes part of a
``verified`` record if it passed :func:`metaextract.extract.ground_verify` against
its cited block. This module is a thin, well-named seam over that existing gate so
the agent's "can I commit this?" decision and the cockpit's "is this suggestion
green?" decision are provably the same code path — there is no second, looser
verifier hiding in the agent.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..extract import GroundingResult, ground_verify
from ..families import AnalysisFamily
from ..records import ExtractionSlot, LocatedRegion, VerifiedRecord
from ..sourcedoc import SourceBlock


def verify(block: SourceBlock, value: str, evidence_text: str) -> GroundingResult:
    """Deterministic grounding verdict for one claimed value (A + B of v3.1 §5.1).

    Delegates to :func:`metaextract.extract.ground_verify`; the argument order is
    flipped to read ``verify(block, value, span)`` at the agent call sites.
    """
    return ground_verify(value, evidence_text, block)


def record_status(slots: list[ExtractionSlot], family: AnalysisFamily) -> str:
    """Return ``verified`` only for a complete and grounded record."""
    by_role = {s.role: s for s in slots}
    if any(
        role not in by_role or by_role[role].value is None
        for role in family.required_roles
    ):
        return "needs_review"
    return (
        "verified"
        if all(
            not field.highlight or by_role[field.role].grounding_verified
            for field in family.fields
        )
        else "needs_review"
    )


def assemble_record(
    region: LocatedRegion,
    slots: list[ExtractionSlot],
    family: AnalysisFamily,
    treatment_group: str,
    control_group: str,
) -> VerifiedRecord:
    """Materialize a :class:`VerifiedRecord` from gated slots (v3.1 §4).

    ``status`` comes from :func:`record_status`, never from the model — so the
    agent cannot label its own output verified without the slots actually passing
    the gate. Every slot keeps its citation, so each number stays jumpable.
    """
    by_role = {s.role: s for s in slots}
    flags = {f for s in slots for f in s.flags}
    flags.update(
        f"missing_required_role:{role}"
        for role in family.required_roles
        if role not in by_role or by_role[role].value is None
    )
    status = record_status(slots, family)
    return VerifiedRecord(
        record_id=f"{region.region_id}:{treatment_group}",
        paper_id=region.paper_id,
        source_block_ids=[region.citation.block_id],
        variable_name=region.variable_name,
        treatment_group=treatment_group,
        control_group=control_group,
        slots=by_role,
        status=status,
        flags=sorted(flags),
        reviewed_by="agent",
        reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
