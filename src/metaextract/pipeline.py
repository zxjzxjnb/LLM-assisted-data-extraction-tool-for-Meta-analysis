"""Batch orchestration: a folder of PDFs in, one tidy CSV out.

This is the piece the original notebook lacked entirely -- it processed a single
hand-named file at a time. A meta-analysis screens hundreds of papers, so the
unit of work here is a directory. Failures are isolated per-paper (one bad PDF
never sinks the run), retried with backoff, and reported in a run summary.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .extractor import DEFAULT_MODEL, extract_from_pdf, _build_client
from .families import DEFAULT_FAMILY
from .flatten import result_to_rows
from .locate import TaskSpec


@dataclass
class RunSummary:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    rows: int = 0
    flagged_rows: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "rows": self.rows,
            "flagged_rows": self.flagged_rows,
            "errors": self.errors,
        }


def _extract_with_retry(pdf, client, model, task_spec=None, retries=2):
    last = None
    for attempt in range(retries + 1):
        try:
            return extract_from_pdf(pdf, model=model, task_spec=task_spec, client=client)
        except Exception as e:  # noqa: BLE001 - want to retry on any API hiccup
            last = e
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise last


def _task_cache_file(
    cache: Path | None, pdf: Path, task_spec: TaskSpec | None
) -> Path | None:
    if cache is None:
        return None
    if task_spec is None:
        return cache / f"{pdf.stem}.json"
    return cache / f"{pdf.stem}.{task_spec.cache_stamp}.json"


def run_folder(
    input_dir: str | Path,
    output_csv: str | Path,
    model: str = DEFAULT_MODEL,
    cache_dir: str | Path | None = None,
    task_spec: TaskSpec | None = None,
) -> RunSummary:
    """Extract every PDF in ``input_dir`` and write a combined CSV.

    ``cache_dir`` (if given) stores the raw JSON per paper so re-runs skip
    already-processed papers -- important when an API call costs money and a
    run may be interrupted.
    """
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    pdfs = sorted(input_dir.glob("*.pdf"))
    summary = RunSummary(total=len(pdfs))

    if task_spec is not None and task_spec.analysis_family != DEFAULT_FAMILY:
        raise ValueError(
            "metaextract run currently supports only analysis_family="
            f"{DEFAULT_FAMILY!r}. The locate/cockpit workflow can render "
            f"{task_spec.analysis_family!r}, but the automatic extractor still "
            "uses the fixed continuous mean/SD/n schema."
        )

    client = _build_client()
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    if task_spec is not None:
        print(
            f"Task pack: {task_spec.domain} "
            f"({len(task_spec.target_variables)} target variable(s))"
        )

    all_rows: list[dict] = []
    for pdf in pdfs:
        cache_file = _task_cache_file(cache, pdf, task_spec)
        try:
            if cache_file and cache_file.exists():
                from .schema import ExtractionResult

                result = ExtractionResult.model_validate_json(
                    cache_file.read_text()
                )
            else:
                result = _extract_with_retry(pdf, client, model, task_spec=task_spec)
                if cache_file:
                    cache_file.write_text(result.model_dump_json(indent=2))

            rows = result_to_rows(result, pdf.name)
            all_rows.extend(rows)
            summary.succeeded += 1
            summary.rows += len(rows)
            summary.flagged_rows += sum(1 for r in rows if r.get("qa_flags"))
            print(f"[ok]   {pdf.name}: {len(rows)} row(s)")
        except Exception as e:  # noqa: BLE001
            summary.failed += 1
            summary.errors[pdf.name] = str(e)
            print(f"[fail] {pdf.name}: {e}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    (output_csv.parent / "run_summary.json").write_text(
        json.dumps(summary.as_dict(), indent=2)
    )
    print(
        f"\nDone: {summary.succeeded}/{summary.total} papers, "
        f"{summary.rows} rows ({summary.flagged_rows} flagged) -> {output_csv}"
    )
    return summary
