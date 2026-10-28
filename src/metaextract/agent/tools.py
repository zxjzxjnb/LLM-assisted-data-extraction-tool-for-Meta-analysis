"""The agent's toolset — thin wrappers over the existing deterministic stages.

Each tool is a small, auditable wrapper around code that already exists and is
already tested: locate output, ``tabular`` geometry parsing, the ``sampling`` n
finder, the ``extract`` Tier-3 gate. The agent never reaches past these into raw
model output — its only way to turn a number into a record is through
``propose_values``/``extract_values`` (which run the grounding gate) and
``commit_record`` (which sets status from the gate verdict, not the model).

The :class:`Toolset` also holds the run's mutable state (proposed slots, committed
records, flags) so the loop can read the final :class:`AgentResult` off it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..extract import RawExtractedValue, Tier3Response, apply_grounding
from ..families import AnalysisFamily, get_family
from ..locate import LocateOutput
from ..records import ExtractionSlot, LocatedRegion, ScreeningResult
from ..sourcedoc import SourceBlock, SourceDoc
from ..tabular import column_count, make_pairings, parse_block
from ..sampling import find_sample_size_candidates
from . import gate
from .trace import FlaggedItem

_BLOCK_PREVIEW_CHARS = 1200


@dataclass
class ToolSpec:
    """A callable tool, plus the JSON schema a function-calling model needs."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


class Toolset:
    """All tools for one paper, sharing the run's mutable state."""

    def __init__(
        self,
        doc: SourceDoc,
        out: LocateOutput,
        *,
        family: AnalysisFamily,
        client: Any = None,
        model: Optional[str] = None,
    ) -> None:
        self.doc = doc
        self.out = out
        self.family = family
        self.client = client
        self.model = model
        # run state
        self.region_slots: dict[str, dict[str, ExtractionSlot]] = {}
        self.records = []  # list[VerifiedRecord]
        self.flagged: list[FlaggedItem] = []
        self.finished = False
        self.stopped_reason = ""

    # --- lookups -----------------------------------------------------------
    def _block(self, block_id: str) -> Optional[SourceBlock]:
        return self.doc.block(block_id)

    def _region(self, region_id: str) -> Optional[LocatedRegion]:
        return next((r for r in self.out.regions if r.region_id == region_id), None)

    # --- tools -------------------------------------------------------------
    def list_regions(self, _args: dict) -> dict:
        regions = [
            {
                "region_id": r.region_id,
                "variable_name": r.variable_name,
                "block_id": r.citation.block_id,
                "page": r.citation.page,
                "kind": r.citation.kind,
                "candidate_structure": r.candidate_structure,
                "ambiguous": r.ambiguous,
            }
            for r in self.out.regions
        ]
        return {"ok": True, "regions": regions, "summary": f"{len(regions)} located region(s)"}

    def read_block(self, args: dict) -> dict:
        block = self._block(args.get("block_id", ""))
        if block is None:
            return {"ok": False, "summary": f"no block {args.get('block_id')!r}"}
        numbers = [w.text for w in block.words if _looks_numeric(w.text)][:40]
        return {
            "ok": True,
            "block_id": block.block_id,
            "page": block.page,
            "kind": block.kind,
            "text": block.text[:_BLOCK_PREVIEW_CHARS],
            "n_words": len(block.words),
            "numeric_tokens": numbers,
            "summary": f"{block.block_id} p{block.page} ({len(block.words)} words)",
        }

    def infer_columns(self, args: dict) -> dict:
        """Discover how many value columns a table block has (label-free geometry)."""
        block = self._block(args.get("block_id", ""))
        if block is None:
            return {"ok": False, "summary": f"no block {args.get('block_id')!r}"}
        n = column_count(block)
        labels = [f"col{i + 1}" for i in range(n)]
        return {
            "ok": True,
            "column_count": n,
            "suggested_labels": labels,
            "suggested_control": labels[0] if labels else None,
            "summary": f"{n} value column(s)",
        }

    def parse_table(self, args: dict) -> dict:
        """Geometry-parse a table block into rows × columns and pair control×treatment."""
        block = self._block(args.get("block_id", ""))
        if block is None:
            return {"ok": False, "summary": f"no block {args.get('block_id')!r}"}
        labels = list(args.get("column_labels") or [])
        control = args.get("control_label") or (labels[0] if labels else None)
        if not labels or control is None:
            return {"ok": False, "summary": "column_labels (and a control_label) required"}
        rows = parse_block(block, labels)
        out_rows = []
        for row in rows:
            pairings = make_pairings(row.cells, labels, control)
            for p in pairings:
                p.pop("_bboxes", None)  # bboxes are for the cockpit, not the planner
            out_rows.append({"label": row.label, "candidates": pairings})
        n_cands = sum(len(r["candidates"]) for r in out_rows)
        return {
            "ok": bool(out_rows),
            "rows": out_rows,
            "summary": (
                f"{len(out_rows)} row(s), {n_cands} candidate pairing(s)"
                if out_rows
                else f"columns did not match {len(labels)} label(s) — fall back to manual"
            ),
        }

    def find_sample_sizes(self, _args: dict) -> dict:
        cands = find_sample_size_candidates(self.doc)
        out = [
            {
                "value": c.value,
                "phrase": c.phrase.strip("…").strip(),
                "evidence_text": c.evidence_text,
                "block_id": c.block_id,
                "page": c.page,
            }
            for c in cands
        ]
        return {"ok": True, "candidates": out, "summary": f"{len(out)} n candidate(s)"}

    def ground_check(self, args: dict) -> dict:
        """Probe the grounding gate for one value WITHOUT committing it.

        Lets the planner (and the demo) confirm that a real printed value grounds
        and that a fabricated one is rejected — the gate, made visible.
        """
        block = self._block(args.get("block_id", ""))
        if block is None:
            return {"ok": False, "summary": f"no block {args.get('block_id')!r}"}
        value = str(args.get("value", ""))
        evidence = str(args.get("evidence_text") or "")
        gr = gate.verify(block, value, evidence)
        return {
            "ok": True,
            "grounded": gr.verified,
            "flags": gr.flags,
            "summary": f"{value!r} → {'GROUNDED' if gr.verified else 'REJECTED ' + ','.join(gr.flags)}",
        }

    def propose_values(self, args: dict) -> dict:
        """Submit values for one region; each is run through the grounding gate.

        Grounded slots are stored against the region for ``commit_record``. Every
        value must include an explicitly quoted evidence span.
        """
        region = self._region(args.get("region_id", ""))
        if region is None:
            return {"ok": False, "summary": f"no region {args.get('region_id')!r}"}
        block = self._block(region.citation.block_id)
        if block is None:
            return {"ok": False, "summary": f"region's block {region.citation.block_id} missing"}
        raw_values = []
        grounded_slots = []
        for v in args.get("values", []) or []:
            value = str(v.get("value", ""))
            raw = RawExtractedValue(
                role=str(v.get("role", "")),
                value=value,
                evidence_text=str(v.get("evidence_text") or ""),
            )
            cited_block = self._block(str(v.get("source_block_id") or "")) or block
            # n is often stated in Methods rather than in the located data block.
            # Let it carry its own cited block while retaining the same grounding
            # implementation and role allow-list as table values.
            if cited_block.block_id != block.block_id:
                grounded_slots.extend(
                    apply_grounding(Tier3Response(values=[raw]), cited_block, region, allowed_roles=self.family.roles)
                )
            else:
                raw_values.append(raw)
        slots = grounded_slots + apply_grounding(
            Tier3Response(values=raw_values), block, region, allowed_roles=self.family.roles
        )
        self._store_slots(region.region_id, slots)
        verdicts = [
            {"role": s.role, "grounded": s.grounding_verified, "flags": s.flags} for s in slots
        ]
        n_ok = sum(1 for s in slots if s.grounding_verified)
        return {
            "ok": True,
            "verdicts": verdicts,
            "summary": f"{n_ok}/{len(slots)} value(s) grounded for {region.variable_name}",
        }

    def extract_values(self, args: dict) -> dict:
        """LIVE Tier-3: let the model read the located block, then ground every value.

        Same gate as :meth:`propose_values`; the difference is the model supplies
        the numbers and their evidence spans. Requires an API client.
        """
        if self.client is None:
            return {"ok": False, "summary": "extract_values needs a live client (offline run)"}
        region = self._region(args.get("region_id", ""))
        if region is None:
            return {"ok": False, "summary": f"no region {args.get('region_id')!r}"}
        block = self._block(region.citation.block_id)
        if block is None:
            return {"ok": False, "summary": f"region's block {region.citation.block_id} missing"}
        from ..extract import extract_region

        slots = extract_region(
            block, region, self.family, region.variable_name, model=self.model, client=self.client
        )
        self._store_slots(region.region_id, slots)
        n_ok = sum(1 for s in slots if s.grounding_verified)
        return {
            "ok": True,
            "verdicts": [{"role": s.role, "grounded": s.grounding_verified, "flags": s.flags} for s in slots],
            "summary": f"Tier-3 read {len(slots)} value(s), {n_ok} grounded",
        }

    def commit_record(self, args: dict) -> dict:
        """Assemble a VerifiedRecord from a region's gated slots.

        ``status`` comes from the gate (:func:`gate.record_status`): all
        highlightable values grounded → ``verified``; otherwise ``needs_review``.
        The agent cannot override this.
        """
        region = self._region(args.get("region_id", ""))
        if region is None:
            return {"ok": False, "summary": f"no region {args.get('region_id')!r}"}
        slots = list(self.region_slots.get(region.region_id, {}).values())
        if not slots:
            self.flagged.append(
                FlaggedItem(
                    variable_name=region.variable_name,
                    region_id=region.region_id,
                    reason="no_grounded_values",
                    detail="commit_record called before any value was proposed/grounded",
                )
            )
            return {"ok": False, "summary": "no grounded values to commit — flagged"}
        record = gate.assemble_record(
            region,
            slots,
            self.family,
            treatment_group=str(args.get("treatment_group") or "treatment"),
            control_group=str(args.get("control_group") or "control"),
        )
        self.records.append(record)
        if record.status != "verified":
            self.flagged.append(
                FlaggedItem(
                    variable_name=region.variable_name,
                    region_id=region.region_id,
                    reason="ungrounded_record",
                    detail=f"status={record.status}, flags={record.flags}",
                )
            )
        return {
            "ok": True,
            "record_id": record.record_id,
            "status": record.status,
            "summary": f"committed {record.variable_name} [{record.status}]",
        }

    def flag_for_review(self, args: dict) -> dict:
        item = FlaggedItem(
            variable_name=args.get("variable_name"),
            region_id=args.get("region_id"),
            reason=str(args.get("reason") or "flagged_by_agent"),
            detail=str(args.get("detail") or ""),
        )
        self.flagged.append(item)
        return {"ok": True, "summary": f"flagged {item.variable_name or item.region_id}: {item.reason}"}

    def finish(self, args: dict) -> dict:
        self.finished = True
        self.stopped_reason = str(args.get("summary") or "agent finished")
        return {"ok": True, "summary": self.stopped_reason}

    # --- internals ---------------------------------------------------------
    def _store_slots(self, region_id: str, slots: list[ExtractionSlot]) -> None:
        bucket = self.region_slots.setdefault(region_id, {})
        for s in slots:
            bucket[s.role] = s  # last write per role wins (re-proposing replaces)

    def screening(self) -> ScreeningResult:
        return self.out.screening

    def specs(self) -> list[ToolSpec]:
        value_obj = _obj(
            {
                "role": {"type": "string", "description": "a family field role, e.g. mean_c, sd_t, n_c"},
                "value": {"type": "string", "description": "the number exactly as printed"},
                "evidence_text": {"type": "string", "description": "short verbatim source span the value was read from"},
            },
            ["role", "value"],
        )
        specs = [
            ToolSpec("list_regions", "List the located target-variable regions for this paper.", _obj({}), self.list_regions),
            ToolSpec("read_block", "Read one source block's text and its numeric tokens.", _obj({"block_id": {"type": "string"}}, ["block_id"]), self.read_block),
            ToolSpec("infer_columns", "Count the value columns in a table block (label-free geometry).", _obj({"block_id": {"type": "string"}}, ["block_id"]), self.infer_columns),
            ToolSpec(
                "parse_table",
                "Geometry-parse a table block into rows×columns and pair control×treatment.",
                _obj(
                    {
                        "block_id": {"type": "string"},
                        "column_labels": {"type": "array", "items": {"type": "string"}},
                        "control_label": {"type": "string"},
                    },
                    ["block_id", "column_labels"],
                ),
                self.parse_table,
            ),
            ToolSpec("find_sample_sizes", "Find deterministic replicate-count (n) candidates with their source.", _obj({}), self.find_sample_sizes),
            ToolSpec(
                "ground_check",
                "Probe the grounding gate for one value without committing it.",
                _obj({"block_id": {"type": "string"}, "value": {"type": "string"}, "evidence_text": {"type": "string"}}, ["block_id", "value"]),
                self.ground_check,
            ),
            ToolSpec(
                "propose_values",
                "Submit values for a region; each is grounded against its cited block before storage.",
                _obj({"region_id": {"type": "string"}, "values": {"type": "array", "items": value_obj}}, ["region_id", "values"]),
                self.propose_values,
            ),
            ToolSpec(
                "commit_record",
                "Assemble a VerifiedRecord from a region's grounded slots; status comes from the gate.",
                _obj({"region_id": {"type": "string"}, "treatment_group": {"type": "string"}, "control_group": {"type": "string"}}, ["region_id"]),
                self.commit_record,
            ),
            ToolSpec(
                "flag_for_review",
                "Hand something to a human instead of committing it.",
                _obj({"reason": {"type": "string"}, "variable_name": {"type": "string"}, "region_id": {"type": "string"}, "detail": {"type": "string"}}, ["reason"]),
                self.flag_for_review,
            ),
            ToolSpec("finish", "End the run when every located region has been handled.", _obj({"summary": {"type": "string"}}), self.finish),
        ]
        if self.client is not None:
            specs.insert(
                7,
                ToolSpec(
                    "extract_values",
                    "LIVE Gemini Tier-3: have the model read a located block, then ground every value.",
                    _obj({"region_id": {"type": "string"}}, ["region_id"]),
                    self.extract_values,
                ),
            )
        return specs


def _looks_numeric(token: str) -> bool:
    from ..highlight import parse_number

    return parse_number(token) is not None


def build_toolset(
    doc: SourceDoc,
    out: LocateOutput,
    *,
    client: Any = None,
    model: Optional[str] = None,
) -> Toolset:
    """Build the toolset for one paper, picking the family from the locate output."""
    family = get_family(out.analysis_family)
    return Toolset(doc, out, family=family, client=client, model=model)
