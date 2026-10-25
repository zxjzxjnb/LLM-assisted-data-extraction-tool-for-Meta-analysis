"""Bounded extraction agent — an autonomous orchestrator over the tiered tools.

This subpackage is the project's "agent layer". It does **not** introduce a new,
higher-trust way to read numbers; it drives the *existing* stages — Tier-2 locate
(:mod:`metaextract.locate`), deterministic table parsing (:mod:`metaextract.tabular`),
the sample-size finder (:mod:`metaextract.sampling`), and Tier-3 on-demand
extraction behind its grounding gate (:mod:`metaextract.extract`) — in an
autonomous tool-use loop, instead of a human clicking each region in the cockpit.

The whole point is *bounded* autonomy: every number the agent commits still passes
the same deterministic grounding gate (``extract.ground_verify``) the cockpit uses,
and a value that cannot be grounded against its cited block is never silently
written — it is flagged ``needs_review``. The agent can choose *what to do next*;
it cannot choose to trust a number the source does not support.

See ``docs/AGENT.md`` for the architecture and the bounded-autonomy rationale.

Three planner implementations drive the same core tools and grounding gate:

* :class:`ReplayPlanner` — a deterministic policy that runs offline on cached
  papers (no API key). Used by the demo and the tests/CI.
* :class:`GeminiPlanner` — the live LLM policy (Gemini function-calling).
* :class:`OpenAICompatiblePlanner` — the shared live policy for DeepSeek and
  other OpenAI-compatible tool-call APIs.
"""

from __future__ import annotations

from .gate import assemble_record, record_status, verify
from .loop import run_agent
from .planner import (
    Finish,
    GeminiPlanner,
    OpenAICompatiblePlanner,
    Planner,
    ReplayPlanner,
    ToolCall,
)
from .tools import Toolset, ToolSpec, build_toolset
from .trace import AgentResult, AgentStep
from .report import AgentReviewQueue, load_review_queue

__all__ = [
    "run_agent",
    "build_toolset",
    "Toolset",
    "ToolSpec",
    "Planner",
    "ReplayPlanner",
    "GeminiPlanner",
    "OpenAICompatiblePlanner",
    "ToolCall",
    "Finish",
    "AgentResult",
    "AgentStep",
    "AgentReviewQueue",
    "load_review_queue",
    "verify",
    "record_status",
    "assemble_record",
]
