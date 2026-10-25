"""Read a saved agent run for hand-off to the verification cockpit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .trace import FlaggedItem, AgentResult


@dataclass(frozen=True)
class AgentReviewQueue:
    """The reviewable flags from one saved agent run, or a safe load warning."""

    flags: list[FlaggedItem]
    warning: str = ""


def load_review_queue(path: Path, *, study: str) -> AgentReviewQueue:
    """Load an ``--report`` JSON file without letting a bad file break the cockpit.

    Agent reports are deliberately a hand-off artifact: the agent names the region
    and explains why it stopped, while the cockpit owns the human decision.  A
    report for another paper or an invalid JSON/schema is ignored with a clear
    warning instead of mixing review work between papers.
    """
    if not path.exists():
        return AgentReviewQueue(flags=[])
    try:
        result = AgentResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        return AgentReviewQueue(
            flags=[], warning=f"Could not read agent review report {path.name}: {exc}"
        )
    if result.study != study:
        return AgentReviewQueue(
            flags=[],
            warning=(
                f"Ignoring {path.name}: it is for paper {result.study!r}, not {study!r}."
            ),
        )
    return AgentReviewQueue(flags=result.flagged)
