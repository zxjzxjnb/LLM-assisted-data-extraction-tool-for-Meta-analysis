"""Durable per-paper review progress: resume across sessions + an audit log.

The cockpit keeps the human's assembled records in ``st.session_state``, which is
lost on browser refresh or server restart. This module persists them so a
researcher can stop and pick up later:

- **draft** ``data/reviews/{study}.json`` — the current set of records plus the
  bound paper-level moderators, rewritten on every change (autosave).
- **log** ``data/reviews/{study}.log.csv`` — append-only audit trail, one row per
  action (add / accept / delete) with a UTC timestamp.

Whole-file JSON is used for the draft (not append-only JSONL) because the record
set is small and *mutable* — records get deleted/edited — so rewriting the list
is trivial and load is just "read the list back". Append-only is reserved for the
log, which is where history belongs.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .fileio import atomic_write_text


def reviews_dir(root: Path) -> Path:
    d = Path(root) / "data" / "reviews"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stem(study: str, stamp: str | None) -> str:
    """Per-paper file stem, optionally keyed by the task pack ``stamp`` so records
    assembled under different target metrics never share a file."""
    return f"{study}.{stamp}" if stamp else study


def draft_path(root: Path, study: str, stamp: str | None = None) -> Path:
    return reviews_dir(root) / f"{_stem(study, stamp)}.json"


def log_path(root: Path, study: str, stamp: str | None = None) -> Path:
    return reviews_dir(root) / f"{_stem(study, stamp)}.log.csv"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _file_stamp() -> str:
    """Filename-safe UTC timestamp (no colons), unique to the microsecond."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")


def _quarantine(path: Path) -> Path:
    """Move an unreadable draft aside so it is preserved, not overwritten.

    The cockpit autosaves on the next interaction, which would otherwise clobber
    the only copy of the corrupt file. Renaming it out of the way keeps the bad
    bytes around for manual recovery while letting a fresh draft start cleanly.
    """
    backup = path.with_name(f"{path.name}.corrupt-{_file_stamp()}")
    path.replace(backup)
    return backup


def load_draft(root: Path, study: str, stamp: str | None = None) -> dict:
    """Return the saved draft, or an empty one if none exists.

    A draft that exists but cannot be parsed (an interrupted pre-atomic write, a
    truncated or hand-edited file, disk corruption) is NOT silently discarded —
    that would mask lost human work. The unreadable file is moved aside to
    ``{name}.corrupt-{timestamp}`` (so the next autosave can't overwrite it) and
    the returned draft carries an ``error`` the cockpit surfaces to the human.
    """
    p = draft_path(root, study, stamp)
    if not p.exists():
        return {"records": [], "moderators": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("draft is not a JSON object")
    except (json.JSONDecodeError, OSError, ValueError) as e:
        backup = _quarantine(p)
        return {
            "records": [],
            "moderators": {},
            "error": (
                f"Saved draft {p.name} could not be read ({type(e).__name__}); it was "
                f"preserved as {backup.name} and an empty draft was started. If it held "
                f"records, recover them from that file before adding new ones."
            ),
        }
    data.setdefault("records", [])
    data.setdefault("moderators", {})
    return data


def save_draft(
    root: Path, study: str, records: list[dict], moderators: dict, stamp: str | None = None
) -> str:
    """Rewrite the whole draft. Cheap (tens of records); used as an autosave.

    Returns the saved-at timestamp so the caller can show it without re-reading.
    """
    saved_at = _now()
    payload = {
        "study": study,
        "task_stamp": stamp,
        "saved_at": saved_at,
        "records": records,
        "moderators": moderators,
    }
    # Atomic: a refresh / restart mid-write must not truncate the draft, since
    # that is the human's only copy of the assembled records.
    atomic_write_text(
        draft_path(root, study, stamp),
        json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return saved_at


_LOG_FIELDS = ["timestamp", "study", "action", "variable_name", "records_after", "detail"]


def append_log(
    root: Path,
    study: str,
    action: str,
    *,
    variable_name: str = "",
    records_after: int = 0,
    detail: str = "",
    stamp: str | None = None,
) -> None:
    """Append one audit row, writing a header the first time the file is created."""
    p = log_path(root, study, stamp)
    is_new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "timestamp": _now(),
            "study": study,
            "action": action,
            "variable_name": variable_name,
            "records_after": records_after,
            "detail": detail,
        })
