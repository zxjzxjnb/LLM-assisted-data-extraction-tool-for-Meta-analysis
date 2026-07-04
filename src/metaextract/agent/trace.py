"""Observable agent run — every tool call, every gate verdict, kept for audit.

A meta-analysis value is only trustworthy if you can see where it came from. The
same is true of an *agent* that assembles those values: its run is only trustworthy
if you can replay what it did. So the agent records one :class:`AgentStep` per tool
call and returns an :class:`AgentResult` that carries the trace alongside the
records — the records say *what* was extracted, the trace says *how* and *why a
number was accepted or flagged*.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from ..records import ScreeningResult, VerifiedRecord


class AgentStep(BaseModel):
    """One tool call in the agent's loop, with a human-readable outcome."""

    index: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    summary: str = ""


class FlaggedItem(BaseModel):
    """Something the agent refused to commit and handed to a human."""

    variable_name: Optional[str] = None
    region_id: Optional[str] = None
    reason: str
    detail: str = ""


class AgentResult(BaseModel):
    """The full output of an agent run: records, flags, and the trace that made them."""

    study: str
    screening: Optional[ScreeningResult] = None
    records: list[VerifiedRecord] = Field(default_factory=list)
    flagged: list[FlaggedItem] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    stopped_reason: str = ""

    @property
    def verified_records(self) -> list[VerifiedRecord]:
        return [r for r in self.records if r.status == "verified"]

    @property
    def needs_review_records(self) -> list[VerifiedRecord]:
        return [r for r in self.records if r.status != "verified"]

    def summary_line(self) -> str:
        v = len(self.verified_records)
        nr = len(self.needs_review_records)
        return (
            f"{self.study}: {len(self.records)} record(s) "
            f"({v} verified, {nr} needs-review), "
            f"{len(self.flagged)} flagged, {len(self.trace)} step(s)"
        )
