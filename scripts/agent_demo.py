"""Run the bounded extraction agent over a cached paper — offline, no API key.

Usage:  python scripts/agent_demo.py [STUDY]      # default P1

Loads the cached ingest+locate for STUDY (whatever ``prepare_paper.py`` already
wrote), then drives the deterministic :class:`ReplayPlanner` through the agent's
tools. Prints the full tool-call trace, the verified records, and anything the
grounding gate forced into 'needs review' — so you can watch the agent assemble
records *and* watch it refuse an ungrounded number, all without an API call.

The live LLM policy (GeminiPlanner) uses the identical tools and gate; this script
just makes the loop runnable and inspectable in CI / on a laptop.
"""

from __future__ import annotations

import json
import sys

from _gt import ROOT  # sets sys.path to src/ and gives the repo root

from metaextract import cockpit_cache
from metaextract.agent import ReplayPlanner, build_toolset, run_agent


def main() -> None:
    study = sys.argv[1] if len(sys.argv) > 1 else "P1"

    try:
        doc, out = cockpit_cache.load(ROOT, study)
    except cockpit_cache.CacheError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    toolset = build_toolset(doc, out)  # offline: no client → extract_values disabled
    result = run_agent(toolset, ReplayPlanner())

    print(f"\n=== Agent trace: {study} ({result.stopped_reason}) ===")
    for s in result.trace:
        mark = "ok " if s.ok else "ERR"
        print(f"  [{s.index:>2}] {mark} {s.tool:<16} {s.summary}")

    print(f"\n=== Result ===\n{result.summary_line()}")

    if result.verified_records:
        print("\nVerified records (every table value passed the grounding gate):")
        for r in result.verified_records:
            vals = {role: slot.value for role, slot in r.slots.items()}
            print(f"  - {r.variable_name} | {r.control_group} vs {r.treatment_group} | {vals}")

    if result.flagged:
        print("\nFlagged for human review:")
        for f in result.flagged:
            print(f"  - {f.variable_name or f.region_id}: {f.reason} ({f.detail})")

    out_dir = ROOT / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"agent_run_{study}.json"
    out_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"\nFull run (records + trace) -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
