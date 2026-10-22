"""Bootstrap a task pack YAML from the ground-truth Excel (one-time import).

Usage:  python scripts/export_taskpack.py [out.yaml]

Reads the GT sheet's 67 variable headers + the paper-level moderator columns and
writes an editable *task pack*. After this, researchers change which target
metrics the locator looks for by editing the YAML alone (add aliases/units,
add/remove metrics) — no code change, no Excel needed. See README "Changing
target metrics".
"""

from __future__ import annotations

import sys
from pathlib import Path

from _gt import gt_spec_and_truth  # GT-coupled reader, eval-only

HEADER = """\
# Task pack — the single place a researcher edits to change target metrics.
#
# Each target_variables entry:
#   name:       canonical metric name (used for export columns + cross-paper match)
#   aliases:    synonyms papers actually use; the locator normalizes them to `name`
#   unit_hint:  optional, helps the locator disambiguate (free text)
#   label:      optional human label for the cockpit
#
# Add a metric -> add an entry, then re-run scripts/prepare_paper.py for the paper.
# Remove a metric -> delete its entry (no re-run needed; it just stops showing).
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent.parent / "data" / "taskpacks" / "soil_C_fractions.yaml"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    # Any study id works; we only want the full header vocabulary, not its rows.
    all_vars, moderators, _present, _rows = gt_spec_and_truth("P1")

    # Dedupe names while preserving order (GT has e.g. C/N_MAOC twice — a known typo).
    seen: set[str] = set()
    target_variables = []
    for name in all_vars:
        if name in seen:
            continue
        seen.add(name)
        target_variables.append({"name": name, "aliases": [], "unit_hint": None})

    from metaextract.locate import TaskSpec

    spec = TaskSpec(
        domain="soil_C_fractions_landuse",
        target_variables=target_variables,
        moderators=moderators,
    )
    out.write_text(HEADER + "\n" + spec.to_yaml(), encoding="utf-8")
    print(f"wrote {out} | {len(target_variables)} metrics, {len(moderators)} moderators")


if __name__ == "__main__":
    main()
