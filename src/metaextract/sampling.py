"""Deterministic sample-size (n) candidate finder for the cockpit.

n is the one field the LLM proved unreliable at: P1 says "84 samples (7 sites x 4
land use types x 3 replicates)" and the model variously picked 5 (sub-samples) or
7 (sites), never the statistical 3. A regex, by contrast, finds the "3 replicates"
phrase immediately. So instead of trusting an LLM number we surface *all* n-like
statements with their source, and the human picks the right one (v3.1: locate
reliably, human decides). Deterministic, offline, jumpable.
"""

from __future__ import annotations

import re
from typing import Callable

from pydantic import BaseModel

from .sourcedoc import SourceDoc

# (pattern, how to read n). Ordered roughly by how often it IS the statistical n.
_PATTERNS: list[tuple[str, Callable[[re.Match], str]]] = [
    (r"\bin\s+triplicate\b", lambda m: "3"),
    (r"\bin\s+duplicate\b", lambda m: "2"),
    (r"(\d+)\s+(?:field\s+|biological\s+|true\s+)?replicat", lambda m: m.group(1)),
    (r"replicat\w*\s*(?:[=:]|was|were|of)?\s*(\d+)", lambda m: m.group(1)),
    (r"\bn\s*=\s*(\d+)", lambda m: m.group(1)),
]


class NCandidate(BaseModel):
    value: str
    phrase: str  # the surrounding text, so the human can judge
    evidence_text: str = ""  # exact substring in the source block, without UI ellipses
    block_id: str
    page: int


def find_sample_size_candidates(doc: SourceDoc, ctx: int = 45) -> list[NCandidate]:
    """Surface every n-like statement (replicate/triplicate/n=) with its source."""
    out: list[NCandidate] = []
    seen: set[tuple[str, str]] = set()
    for b in doc.blocks:
        if b.kind != "paragraph" or not b.text:
            continue
        for pattern, read in _PATTERNS:
            for m in re.finditer(pattern, b.text, re.I):
                value = read(m)
                key = (b.block_id, value + str(m.start()))
                if key in seen:
                    continue
                seen.add(key)
                s, e = max(0, m.start() - ctx), min(len(b.text), m.end() + ctx)
                out.append(
                    NCandidate(
                        value=value,
                        phrase=("…" + b.text[s:e].strip() + "…"),
                        evidence_text=b.text[s:e].strip(),
                        block_id=b.block_id,
                        page=b.page,
                    )
                )
    return out
