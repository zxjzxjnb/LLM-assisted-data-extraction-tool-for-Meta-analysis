"""Atomic file writes.

A plain ``Path.write_text`` truncates the target *before* writing it, so a crash,
Ctrl-C, or a Streamlit rerun that interrupts the write leaves a half-written —
i.e. corrupt — file behind. The cockpit autosaves the review draft on every
interaction and the cache is written by ``prepare_paper``; both must survive an
interrupted write, because a truncated draft is lost human work and a truncated
cache crashes the cockpit on open.

Writing to a temp file in the *same directory* and then ``os.replace``-ing it into
place makes the swap atomic on POSIX and Windows: a reader ever sees either the
old complete file or the new complete file, never a truncated one. (This guards
against process interruption — the dominant failure here — not power loss, which
would additionally need an ``fsync``; that is intentionally skipped to keep the
per-rerun autosave fast.)
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (temp file + ``os.replace``)."""
    path = Path(path)
    # Temp lives in the same directory so os.replace stays on one filesystem (a
    # cross-device rename is not atomic). PID keeps concurrent writers apart.
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    finally:
        # On success the temp was renamed away; only a leftover from a failed
        # write remains to clean up, so we don't litter the directory.
        if tmp.exists():
            tmp.unlink(missing_ok=True)
