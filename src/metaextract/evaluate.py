"""Evaluate extracted data against hand-extracted ground truth.

This is the part of the project that turns "I built a tool" into "I built a
tool and measured that it works." Given a CSV of manually extracted values
(the gold standard a meta-analyst would normally produce by hand) and the
pipeline's output, it reports:

*   **Field recovery rate** -- of the numeric cells present in the ground truth,
    what fraction did the pipeline recover within a relative tolerance.
*   **Per-field MAE / median absolute error** on the matched numbers.
*   **Row matching** -- precision/recall on identifying the right
    treatment/control data points in the first place.

Matching strategy
-----------------
Predicted and ground-truth rows are aligned on a *study key* (first author +
year, falling back to source file) plus the response-variable identity
(variable name + treatment/control groups), normalised case- and
whitespace-insensitively. This mirrors how a human would line the two tables up.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

import pandas as pd

NUMERIC_FIELDS = ["mean_t", "sd_t", "n_t", "mean_c", "sd_c", "n_c"]


def _norm(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _row_key(row) -> tuple:
    study = _norm(row.get("first_author")) or _norm(row.get("source_file"))
    return (
        study,
        _norm(row.get("year")),
        _norm(row.get("variable_name")),
        _norm(row.get("treatment_group")),
        _norm(row.get("control_group")),
        _norm(row.get("sampling_depth_cm") or row.get("depth")),
        _norm(row.get("measurement_year")),
        _norm(row.get("site_name") or row.get("site")),
        _norm(row.get("experiment_id")),
    )


def _close(pred, truth, rel_tol) -> bool:
    if pred is None or truth is None:
        return False
    try:
        pred, truth = float(pred), float(truth)
    except (TypeError, ValueError):
        return _norm(pred) == _norm(truth)
    if math.isnan(pred) or math.isnan(truth):
        return False
    if truth == 0:
        return abs(pred) < 1e-9
    return abs(pred - truth) / abs(truth) <= rel_tol


def evaluate(pred_csv, truth_csv, rel_tol: float = 0.05) -> dict:
    """Compare predicted and ground-truth CSVs; return a metrics report."""
    pred = pd.read_csv(pred_csv).to_dict("records")
    truth = pd.read_csv(truth_csv).to_dict("records")

    # Keep all rows per identity. A dict[key] = row silently overwrites duplicate
    # depths/time points and lets one prediction match multiple truth rows.
    pred_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in pred:
        pred_by_key[_row_key(r)].append(r)

    matched = 0
    field_total = {f: 0 for f in NUMERIC_FIELDS}
    field_correct = {f: 0 for f in NUMERIC_FIELDS}
    abs_errors = {f: [] for f in NUMERIC_FIELDS}

    for t in truth:
        key = _row_key(t)
        for f in NUMERIC_FIELDS:
            tv = t.get(f)
            if tv is None or (isinstance(tv, float) and math.isnan(tv)):
                continue
            field_total[f] += 1
        candidates = pred_by_key.get(key, [])
        p = candidates.pop(0) if candidates else None
        if p is None:
            continue
        matched += 1
        for f in NUMERIC_FIELDS:
            tv = t.get(f)
            if tv is None or (isinstance(tv, float) and math.isnan(tv)):
                continue
            pv = p.get(f)
            if _close(pv, tv, rel_tol):
                field_correct[f] += 1
            try:
                abs_errors[f].append(abs(float(pv) - float(tv)))
            except (TypeError, ValueError):
                pass

    n_truth, n_pred = len(truth), len(pred)
    total_cells = sum(field_total.values())
    correct_cells = sum(field_correct.values())

    return {
        "rows": {
            "ground_truth": n_truth,
            "predicted": n_pred,
            "matched": matched,
            "row_recall": matched / n_truth if n_truth else 0.0,
            "row_precision": matched / n_pred if n_pred else 0.0,
        },
        "overall_field_recovery": correct_cells / total_cells
        if total_cells
        else 0.0,
        "tolerance": rel_tol,
        "per_field": {
            f: {
                "recovery": field_correct[f] / field_total[f]
                if field_total[f]
                else None,
                "n": field_total[f],
                "mae": sum(abs_errors[f]) / len(abs_errors[f])
                if abs_errors[f]
                else None,
            }
            for f in NUMERIC_FIELDS
        },
    }


def print_report(report: dict) -> None:
    r = report["rows"]
    print("=== Row matching ===")
    print(f"  ground-truth rows : {r['ground_truth']}")
    print(f"  predicted rows    : {r['predicted']}")
    print(f"  matched rows      : {r['matched']}")
    print(f"  row recall        : {r['row_recall']:.1%}")
    print(f"  row precision     : {r['row_precision']:.1%}")
    print(f"\n=== Field recovery (within {report['tolerance']:.0%}) ===")
    print(f"  overall           : {report['overall_field_recovery']:.1%}")
    for f, m in report["per_field"].items():
        rec = "n/a" if m["recovery"] is None else f"{m['recovery']:.1%}"
        mae = "n/a" if m["mae"] is None else f"{m['mae']:.3g}"
        print(f"  {f:<8} recovery={rec:>6}  MAE={mae:>8}  (n={m['n']})")
