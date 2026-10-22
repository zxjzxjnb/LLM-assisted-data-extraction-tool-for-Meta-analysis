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
    """``"verified"`` only if every highlightable value is grounding-verified.

    A family's highlightable fields are the ones that live in the source table (a
    mean/SD, not an ``n`` read from the methods text). The agent may only call a
    record ``verified`` when all of those present passed the gate; if any failed
    grounding, or none were filled, the record is ``needs_review`` so a human looks
    at it. This mirrors what the cockpit paints red.
    """
    highlight_roles = {f.role for f in family.fields if f.highlight}
    relevant = [s for s in slots if s.role in highlight_roles]
    if not relevant:
        return "needs_review"
    return "verified" if all(s.grounding_verified for s in relevant) else "needs_review"


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
    flags = sorted({f for s in slots for f in s.flags})
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
        flags=flags,
        reviewed_by="agent",
        reviewed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
