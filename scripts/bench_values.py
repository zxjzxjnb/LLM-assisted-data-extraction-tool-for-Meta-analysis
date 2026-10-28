"""Label-free value-accuracy benchmark over table papers (P1–P12).

We don't know each paper's column labels, so this is value-anchored: for each GT
variable, discover the table block's columns by geometry (column_count), parse the
row, and check how many of the GT numbers ({Xc} ∪ {Xe…} for means, {Sc} ∪ {Se…}
for sds) appear among the parsed cell values within 2%. This measures whether the
geometry reproduces the hand-extracted numbers, without needing label semantics.

Figure papers are skipped here (their values come from the digitizer, not parsing).
"""

from __future__ import annotations

import re

from _gt import ROOT
from eval_against_gt import _num, gt_records


def _toks(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def _close(a, b, tol=0.02):
    if a is None or b is None:
        return False
    return abs(a) < 1e-9 if b == 0 else abs(a - b) / abs(b) <= tol


def _recall(gt_values, pred_values):
    """How many gt numbers appear in the predicted set (within tol)."""
    ok = 0
    for g in gt_values:
        if g is None:
            continue
        if any(_close(p, g) for p in pred_values):
            ok += 1
    return ok


def bench_paper(study):
    from metaextract import cockpit_cache
    from metaextract.tabular import column_count, parse_block

    doc, out = cockpit_cache.load(ROOT, study)
    table_regions = [r for r in out.regions if r.citation.kind == "paragraph"]
    if not table_regions:
        return None  # figure / empty paper — not value-benchmarkable here

    gt = gt_records(study)
    by_var: dict[str, dict] = {}
    for g in gt:
        d = by_var.setdefault(g["variable"], {"means": set(), "sds": set()})
        d["means"].update([g["Xc"], g["Xe"]])
        d["sds"].update([g["Sc"], g["Se"]])

    mean_ok = mean_tot = sd_ok = sd_tot = 0
    for var, vals in by_var.items():
        # best-matching located region for this GT variable
        region = max(table_regions, key=lambda r: len(_toks(r.variable_name) & _toks(var)), default=None)
        if region is None or not (_toks(region.variable_name) & _toks(var)):
            continue
        block = doc.block(region.citation.block_id)
        n = column_count(block)
        if n < 2:
            continue
        rows = parse_block(block, [f"c{i}" for i in range(n)])
        if not rows:
            continue
        row = rows[0] if len(rows) == 1 else max(rows, key=lambda r: len(_toks(r.label) & _toks(var)))
        pred_means = [_num(c.mean.text) for c in row.cells.values() if c.mean]
        pred_sds = [_num(c.dispersion.text) for c in row.cells.values() if c.dispersion]
        gmeans = [v for v in vals["means"] if v is not None]
        gsds = [v for v in vals["sds"] if v is not None]
        mean_ok += _recall(gmeans, pred_means)
        mean_tot += len(gmeans)
        sd_ok += _recall(gsds, pred_sds)
        sd_tot += len(gsds)
    return mean_ok, mean_tot, sd_ok, sd_tot


def main():
    print(f"{'paper':6} {'mean-recall':>14} {'sd-recall':>12}")
    print("-" * 36)
    tot = [0, 0, 0, 0]
    for i in range(1, 13):
        s = f"P{i}"
        try:
            res = bench_paper(s)
        except Exception as e:
            print(f"{s:6} (no cache / {type(e).__name__})")
            continue
        if res is None:
            print(f"{s:6} {'— (figure/none)':>14}")
            continue
        mo, mt, so, st = res
        for j, v in enumerate(res):
            tot[j] += v
        mr = f"{mo}/{mt}={mo/mt:.0%}" if mt else "n/a"
        sr = f"{so}/{st}={so/st:.0%}" if st else "n/a"
        print(f"{s:6} {mr:>14} {sr:>12}")
    print("-" * 36)
    mo, mt, so, st = tot
    print(f"{'TOTAL':6} {f'{mo}/{mt}={mo/mt:.0%}' if mt else 'n/a':>14} "
          f"{f'{so}/{st}={so/st:.0%}' if st else 'n/a':>12}")


if __name__ == "__main__":
    main()
