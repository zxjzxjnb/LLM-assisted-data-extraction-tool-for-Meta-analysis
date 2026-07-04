"""Planners — the policy that decides the agent's next tool call.

The loop is dumb on purpose: it dispatches whatever the planner asks for and
enforces the gate inside the tools. The *policy* lives here, behind one small
protocol, so the same tools and the same grounding gate can be driven by either:

* :class:`ReplayPlanner` — a deterministic, offline policy that reacts to tool
  results (written as a generator, so it reads like a script while still being a
  real perceive→act loop). It needs no API key, so it runs the demo and the tests.
* :class:`GeminiPlanner` — the live LLM policy using Gemini function-calling.

Both are constrained identically: neither can commit a number that fails grounding,
because that rule lives in the tools, not the planner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, Union, runtime_checkable

from .tools import ToolSpec


@dataclass
class ToolCall:
    """The planner's request to run one tool."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Finish:
    """The planner's signal to stop the loop."""

    reason: str = ""


AgentAction = Union[ToolCall, Finish]


@dataclass
class ToolResult:
    """What a tool returned, fed back to the planner on the next turn."""

    name: str
    content: dict[str, Any]


@runtime_checkable
class Planner(Protocol):
    def reset(self, *, system: str, tools: list[ToolSpec], context: str) -> None: ...

    def next_action(self, last: Optional[ToolResult]) -> AgentAction: ...


def _first_candidate(rows: list[dict]) -> Optional[dict]:
    """First control×treatment pairing with both means present."""
    for row in rows:
        for cand in row.get("candidates", []):
            if cand.get("mean_c") and cand.get("mean_t"):
                return cand
    return None


class ReplayPlanner:
    """Deterministic offline policy: locate → parse → gate → commit, per region.

    Reacts to real tool output (column count, parsed candidates) rather than a
    precomputed script, so it exercises the same perceive→act loop the LLM does —
    just with fixed decisions. For each table region it also *probes* the gate with
    one real value (accepted) and one fabricated value (rejected), so a run visibly
    demonstrates that the agent cannot commit an ungrounded number.
    """

    def __init__(self) -> None:
        self._gen: Optional[Iterator[AgentAction]] = None
        self._started = False

    def reset(self, **_: Any) -> None:
        self._gen = self._script()
        self._started = False

    def next_action(self, last: Optional[ToolResult]) -> AgentAction:
        assert self._gen is not None, "call reset() before next_action()"
        try:
            if not self._started:
                self._started = True
                return next(self._gen)
            return self._gen.send(last)
        except StopIteration:
            return Finish("replay script complete")

    def _script(self) -> Iterator[AgentAction]:
        res = yield ToolCall("list_regions", {})
        regions = res.content.get("regions", []) if res else []
        for r in regions:
            rid, bid, var = r["region_id"], r["block_id"], r["variable_name"]
            if r.get("kind") == "figure":
                yield ToolCall(
                    "flag_for_review",
                    {"region_id": rid, "variable_name": var, "reason": "figure_region_needs_digitizing"},
                )
                continue
            yield ToolCall("read_block", {"block_id": bid})
            cols = yield ToolCall("infer_columns", {"block_id": bid})
            n = cols.content.get("column_count", 0)
            if n < 2:
                yield ToolCall(
                    "flag_for_review",
                    {"region_id": rid, "variable_name": var, "reason": "not_a_pairing_table"},
                )
                continue
            labels = cols.content["suggested_labels"]
            control = cols.content["suggested_control"]
            parsed = yield ToolCall(
                "parse_table", {"block_id": bid, "column_labels": labels, "control_label": control}
            )
            rows = parsed.content.get("rows", [])
            # A multi-row block (e.g. P1's merged pH+Moisture+TN) needs row-identity
            # matching — exactly the work v3.1 leaves to a human. A naive policy would
            # silently assign the first row to every variable (real numbers, WRONG row);
            # the grounding gate can't catch that, so the agent defers instead.
            if len(rows) != 1:
                yield ToolCall(
                    "flag_for_review",
                    {
                        "region_id": rid,
                        "variable_name": var,
                        "reason": "multi_row_block_needs_row_assignment" if rows else "no_pairing_candidate",
                        "detail": f"block {bid} parsed into {len(rows)} rows",
                    },
                )
                continue
            cand = _first_candidate(rows)
            if cand is None:
                yield ToolCall(
                    "flag_for_review",
                    {"region_id": rid, "variable_name": var, "reason": "no_pairing_candidate"},
                )
                continue
            # Make the gate visible: a real printed value grounds, a fabricated one is rejected.
            yield ToolCall("ground_check", {"block_id": bid, "value": cand["mean_c"]})
            yield ToolCall("ground_check", {"block_id": bid, "value": "99999.0"})
            values = [
                {"role": role, "value": cand[role]}
                for role in ("mean_c", "sd_c", "mean_t", "sd_t")
                if cand.get(role)
            ]
            yield ToolCall("propose_values", {"region_id": rid, "values": values})
            yield ToolCall(
                "commit_record",
                {
                    "region_id": rid,
                    "treatment_group": cand["treatment_group"],
                    "control_group": cand["control_group"],
                },
            )
        yield ToolCall("finish", {"summary": "all located regions handled"})


class GeminiPlanner:
    """Live LLM policy via Gemini function-calling.

    Implemented for completeness and the live ``--online`` path; it is not exercised
    in CI (it needs an API key). The loop and tools are identical to the offline
    path, so the grounding gate constrains this planner exactly the same way.
    """

    def __init__(self, client: Any = None, model: Optional[str] = None) -> None:
        self._client = client
        self._model = model
        self._system = ""
        self._contents: list[Any] = []
        self._tool = None
        self._types = None

    def reset(self, *, system: str, tools: list[ToolSpec], context: str) -> None:
        from google.genai import types

        from ..extractor import DEFAULT_MODEL, _build_client

        self._types = types
        self._client = self._client or _build_client()
        self._model = self._model or DEFAULT_MODEL
        self._system = system
        self._tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=t.name, description=t.description, parameters=t.parameters
                )
                for t in tools
            ]
        )
        self._contents = [types.Content(role="user", parts=[types.Part(text=context)])]

    def next_action(self, last: Optional[ToolResult]) -> AgentAction:
        types = self._types
        if last is not None:
            self._contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=last.name, response=last.content
                            )
                        )
                    ],
                )
            )
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._contents,
            config=types.GenerateContentConfig(
                system_instruction=self._system,
                temperature=0.0,
                tools=[self._tool],
            ),
        )
        candidate = response.candidates[0]
        self._contents.append(candidate.content)
        for part in candidate.content.parts or []:
            call = getattr(part, "function_call", None)
            if call is not None:
                return ToolCall(name=call.name, args=dict(call.args or {}))
        return Finish(getattr(response, "text", "") or "model returned no tool call")
