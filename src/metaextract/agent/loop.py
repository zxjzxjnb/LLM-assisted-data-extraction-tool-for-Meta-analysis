"""The agent loop — dispatch tool calls, record the trace, enforce the budget.

Deliberately thin: it has no judgement of its own. It asks the planner for the
next action, runs the corresponding tool, appends an :class:`AgentStep`, and feeds
the result back. The bounded-autonomy guarantee is not here — it is inside the
tools (every commit goes through the grounding gate). The loop only adds a step
budget so a misbehaving planner cannot run forever.
"""

from __future__ import annotations

from typing import Optional

from .planner import Finish, Planner, ToolResult
from .tools import Toolset
from .trace import AgentResult, AgentStep


def _default_system(toolset: Toolset) -> str:
    family = toolset.family
    roles = ", ".join(f.role for f in family.fields)
    return (
        "You are an extraction agent for an ecological meta-analysis. You assemble "
        "treatment-vs-control records from a paper that has ALREADY been screened and "
        "located — your job is to turn each located region into a verified record, or "
        "flag it for a human.\n\n"
        f"Analysis family: {family.name} (record fields: {roles}).\n\n"
        "HARD RULES (the tools enforce these; do not fight them):\n"
        "- Every number you commit must ground against its cited block. Submit values "
        "with propose_values (or extract_values) and a record is only 'verified' if all "
        "its table values pass the grounding gate.\n"
        "- Never invent a number or a block_id. If a region has no groundable values, "
        "call flag_for_review instead of committing.\n"
        "- Work region by region from list_regions; call finish when all are handled.\n"
    )


def _initial_context(toolset: Toolset) -> str:
    regions = toolset.out.regions
    lines = [
        f"Paper: {toolset.doc.doc_id}",
        f"Screening: include={toolset.out.screening.include}",
        f"{len(regions)} located region(s):",
    ]
    for r in regions:
        lines.append(
            f"  - {r.region_id} | {r.variable_name} | block {r.citation.block_id} "
            f"(p{r.citation.page}, {r.citation.kind})"
        )
    lines.append("\nStart by calling list_regions, then handle each region.")
    return "\n".join(lines)


def run_agent(
    toolset: Toolset,
    planner: Planner,
    *,
    system: Optional[str] = None,
    max_steps: int = 400,
) -> AgentResult:
    """Drive ``planner`` over ``toolset`` until it finishes or the budget runs out."""
    specs = toolset.specs()
    by_name = {s.name: s for s in specs}
    planner.reset(
        system=system or _default_system(toolset),
        tools=specs,
        context=_initial_context(toolset),
    )

    steps: list[AgentStep] = []
    last: Optional[ToolResult] = None
    stopped = ""
    for i in range(max_steps):
        action = planner.next_action(last)
        if isinstance(action, Finish):
            stopped = action.reason or "planner finished"
            break
        spec = by_name.get(action.name)
        if spec is None:
            content = {"ok": False, "summary": f"unknown tool {action.name!r}"}
        else:
            content = spec.handler(action.args or {})
        steps.append(
            AgentStep(
                index=i,
                tool=action.name,
                args=action.args or {},
                ok=bool(content.get("ok", True)),
                summary=str(content.get("summary", "")),
            )
        )
        last = ToolResult(name=action.name, content=content)
        if toolset.finished:
            stopped = toolset.stopped_reason or "finish tool called"
            break
    else:
        stopped = f"reached max_steps={max_steps}"

    return AgentResult(
        study=toolset.doc.doc_id,
        screening=toolset.screening(),
        records=list(toolset.records),
        flagged=list(toolset.flagged),
        trace=steps,
        stopped_reason=stopped or toolset.stopped_reason,
    )
