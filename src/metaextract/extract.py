"""Tier 3 — on-demand AI extraction behind a deterministic grounding gate.

This is the only path that produces AI *numbers* (Architecture v3.1 §2, §5.1), so
it is the lowest-trust tier and the one with a hard gate in front of it. The human
clicks a located region in the cockpit; the model is asked to read the region's
values **and quote the exact span it read each from**; then every value is
ground-verified *deterministically* before it is ever shown as a suggestion:

    A. the printed value must appear (symbol-tolerant) in the model's evidence span
    B. that evidence span must appear (normalized) inside the cited block's text
    C. the cited block_id must be the region's block (the caller passes that block)

Pass → ``grounding_verified=True``. Any failure → flag ``evidence_unverified``,
``grounding_verified=False``, and the cockpit shows it **red**. The check is cheap,
offline, and catches the dominant Tier-3 failures (a hallucinated number, or a
value spliced from a different table) that the model's own confidence cannot.

Mirroring :mod:`metaextract.locate`: ``build_payload``, ``ground_verify`` and
``apply_grounding`` are pure and unit-tested without the API; :func:`extract_region`
is a thin Gemini wrapper around them. The geometry on every produced slot still
comes from deterministic ingest, never from the model.
"""

from __future__ import annotations

from typing import Iterable, Optional

from pydantic import BaseModel, Field

from .families import AnalysisFamily
from .highlight import numeric_in_text, parse_number, text_contains
from .records import Citation, ExtractionSlot, LocatedRegion
from .sourcedoc import SourceBlock


# --- what we ask Gemini to return: a value PLUS the span it was read from ---
class RawExtractedValue(BaseModel):
    role: str  # one of the family's field roles (mean_c, sd_t, events_c, …)
    value: str  # the number exactly as printed (kept as text; grounded as a number)
    evidence_text: str = ""  # the exact source span the value was read from


class Tier3Response(BaseModel):
    values: list[RawExtractedValue] = Field(default_factory=list)


class GroundingResult(BaseModel):
    """Outcome of the deterministic gate for one claimed value."""

    verified: bool
    flags: list[str] = Field(default_factory=list)


def build_payload(block: SourceBlock, family: AnalysisFamily, variable_name: str) -> str:
    """Serialize the one cited block + which roles to read, for the model."""
    roles = "\n".join(f"  - {f.role}: {f.label} ({f.arm} arm)" for f in family.fields)
    return (
        f"Variable to extract: {variable_name}\n"
        f"Source block [{block.block_id}] (page {block.page}):\n"
        f"\"\"\"\n{block.text}\n\"\"\"\n\n"
        "Read each of these fields from the block above:\n"
        f"{roles}\n"
    )


def _system_instruction(family: AnalysisFamily, variable_name: str) -> str:
    return (
        "You are extracting numeric data for a meta-analysis from ONE already-located "
        "block of a paper. You are the lowest-trust step: every number you return is "
        "checked against the source before a human sees it, so accuracy and honest "
        "evidence matter more than coverage.\n\n"
        f"Analysis family fields to read for '{variable_name}':\n"
        + "\n".join(f"  - {f.role}: {f.label} ({f.arm} arm)" for f in family.fields)
        + "\n\nHARD RULES:\n"
        "- Report each value EXACTLY as printed (no rounding, no unit conversion).\n"
        "- For every value, put in `evidence_text` the SHORT exact substring of the "
        "block you read it from, copied verbatim (including any ± / significance "
        "marks). The value must be findable inside that span.\n"
        "- Only return a role if its value is genuinely in this block. Omit a role "
        "you cannot read here; never guess to fill the form.\n"
        "- Do not return a value whose evidence span you cannot quote from the block."
    )


def ground_verify(value: str, evidence_text: str, block: SourceBlock) -> GroundingResult:
    """The deterministic Tier-3 gate (v3.1 §5.1), checks A + B. Pure, no LLM.

    Check C (block_id membership) is structural — the caller hands the region's own
    block — so it is satisfied by construction here.
    """
    if not (evidence_text and evidence_text.strip()):
        return GroundingResult(verified=False, flags=["no_evidence_span"])
    flags: list[str] = []
    if not numeric_in_text(value, evidence_text):
        flags.append("value_not_in_span")  # A failed: claimed number isn't in its own span
    if not text_contains(block.text, evidence_text):
        flags.append("span_not_in_block")  # B failed: span isn't really in the source
    return GroundingResult(verified=not flags, flags=flags)


def _coerce_number(value: str) -> Optional[float | int]:
    """Parse a printed value to the number to store, using the SAME parser as
    grounding (:func:`highlight.parse_number`) so the stored value can never differ
    from the value that was ground-verified (e.g. "1,234" → 1234, not 1)."""
    num = parse_number(value)
    if num is None:
        return None
    return int(num) if float(num).is_integer() else num


def apply_grounding(
    raw: Tier3Response,
    block: SourceBlock,
    region: LocatedRegion,
    allowed_roles: Iterable[str] | None = None,
) -> list[ExtractionSlot]:
    """Turn the model's raw values into grounded ``ExtractionSlot``s.

    Pure: every slot carries deterministic geometry (the region's block) and a
    grounding verdict the model never sets. An ungrounded value is kept (so the
    human can see and correct it) but flagged ``evidence_unverified`` and left
    ``grounding_verified=False`` for the cockpit to red-flag.
    """
    slots: list[ExtractionSlot] = []
    allowed = set(allowed_roles) if allowed_roles is not None else None
    for rv in raw.values:
        gr = ground_verify(rv.value, rv.evidence_text, block)
        flags = list(gr.flags)
        number = _coerce_number(rv.value)
        role_allowed = allowed is None or rv.role in allowed
        if not role_allowed:
            flags.append("unknown_role")
        # Tier 3 extracts numbers; a value that doesn't parse to one (e.g.
        # "not reported") must not show as a grounded suggestion even if its span
        # is present in the block — the cockpit would paint it green over a None.
        if number is None:
            flags.append("non_numeric_value")
        verified = gr.verified and number is not None and role_allowed
        if not verified:
            flags.append("evidence_unverified")
        slots.append(
            ExtractionSlot(
                slot_id=f"{region.region_id}:{rv.role}",
                region_id=region.region_id,
                role=rv.role,
                citation=Citation(
                    block_id=block.block_id,
                    page=block.page,
                    bbox=block.bbox,
                    kind=block.kind,
                    evidence_text=rv.evidence_text or None,
                ),
                value=number,
                source="ai_extracted",
                grounding_verified=verified,
                flags=flags,
            )
        )
    return slots


def _response_schema() -> dict:
    """Inline $defs/$ref, which Gemini's response_schema does not accept."""
    schema = Tier3Response.model_json_schema()
    defs = schema.pop("$defs", {})

    def _inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node.pop("$ref").split("/")[-1]
                node.update(_inline(defs[ref]))
            return {k: _inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_inline(v) for v in node]
        return node

    return _inline(schema)


def extract_region(
    block: SourceBlock,
    region: LocatedRegion,
    family: AnalysisFamily,
    variable_name: str,
    model: Optional[str] = None,
    client=None,
) -> list[ExtractionSlot]:
    """Run Tier 3 over one located block via Gemini, then ground every value.

    Thin wrapper — all logic lives in the pure functions above. The returned slots
    are already ground-verified; the cockpit shows verified ones as suggestions and
    red-flags the rest.
    """
    from google.genai import types

    from .extractor import DEFAULT_MODEL, _build_client

    client = client or _build_client()
    response = client.models.generate_content(
        model=model or DEFAULT_MODEL,
        contents=[build_payload(block, family, variable_name)],
        config=types.GenerateContentConfig(
            system_instruction=_system_instruction(family, variable_name),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_response_schema(),
        ),
    )
    parsed = getattr(response, "parsed", None)
    raw = (
        Tier3Response.model_validate(parsed)
        if parsed is not None
        else Tier3Response.model_validate_json(response.text)
    )
    return apply_grounding(raw, block, region, allowed_roles=family.roles)
