"""On-disk cache so the cockpit runs offline (no API call per launch).

``prepare`` runs the expensive stages once (ingest + configured locator) and writes
two JSON files per paper; the Streamlit app only ever reads them.
"""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .fileio import atomic_write_text
from .locate import LocateOutput
from .sourcedoc import SourceDoc

# On-disk cache schema version, stamped into each locate cache by ``save`` and
# checked by ``load``. ``MIN_SUPPORTED`` is the oldest version the current models
# can still read: unversioned caches (no ``schema_version`` key, i.e. everything
# written before versioning) count as 0, and today's models load them via field
# defaults, so 0 stays supported. The day a schema change stops old caches from
# loading, bump BOTH constants together — every stale cache then gets one clear
# "re-run prepare_paper" message instead of a raw pydantic error.
CACHE_SCHEMA_VERSION = 1
MIN_SUPPORTED_CACHE_VERSION = 0


class CacheError(ValueError):
    """A cache that exists but the current code cannot use.

    Raised for a missing, corrupted, or version-incompatible cache so the cockpit
    shows one clear "re-run prepare_paper.py {study}" message instead of a raw
    pydantic/JSON traceback.
    """


@dataclass(frozen=True)
class CacheEntry:
    """One selectable cockpit cache: a study plus the task pack that located it."""

    study: str
    stamp: str | None = None

    @property
    def label(self) -> str:
        return self.study if self.stamp is None else f"{self.study} - {self.stamp}"


def cache_dir(root: Path) -> Path:
    d = root / "data" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(root: Path, study: str, doc: SourceDoc, out: LocateOutput) -> None:
    d = cache_dir(root)
    # Atomic: a Ctrl-C during prepare_paper must not leave a truncated cache that
    # crashes the cockpit on open (the cockpit's only check is "does it parse").
    atomic_write_text(d / f"{study}.sourcedoc.json", doc.model_dump_json())
    locate_name = (
        f"{study}.{out.task_stamp}.locate.json"
        if out.task_stamp
        else f"{study}.locate.json"
    )
    # Stamp the schema version alongside the data. It rides as an extra key that
    # LocateOutput ignores on load (pydantic extra='ignore'), so the model stays a
    # pure domain model and old code reading a new cache simply skips it.
    payload = out.model_dump(mode="json")
    payload["schema_version"] = CACHE_SCHEMA_VERSION
    atomic_write_text(d / locate_name, json.dumps(payload, ensure_ascii=False))
    # A stamped write supersedes any unstamped legacy cache for this study (written
    # by the pre-stamp scheme). Remove it so stampless loads (eval scripts) and the
    # cockpit picker fall through to the fresh stamped file instead of stale data.
    if out.task_stamp:
        (d / f"{study}.locate.json").unlink(missing_ok=True)


def _resolve_locate_path(d: Path, study: str, stamp: str | None) -> Path:
    """Pick the locate-cache file for a study.

    A caller that knows the stamp (the cockpit) asks for it directly. A caller
    that does not pass one (the eval/benchmark scripts) used to assume the legacy
    ``{study}.locate.json`` name, but ``save`` now stamps the file whenever the
    task pack has a stamp. So when no stamp is given and the legacy file is gone,
    fall back to the single stamped cache; refuse (rather than silently pick) when
    several exist, since choosing the wrong task pack would corrupt the eval.
    """
    if stamp:
        return d / f"{study}.{stamp}.locate.json"
    legacy = d / f"{study}.locate.json"
    if legacy.exists():
        return legacy
    stamped = sorted(d.glob(f"{study}.*.locate.json"))
    if len(stamped) == 1:
        return stamped[0]
    if len(stamped) > 1:
        names = ", ".join(p.name for p in stamped)
        raise CacheError(
            f"{study!r} has several task-pack locate caches ({names}); "
            "pass stamp= to choose one."
        )
    return legacy  # none found: let the caller surface a clear FileNotFoundError


def _rerun_hint(study: str) -> str:
    return f"re-run `python scripts/prepare_paper.py {study}` to rebuild it"


def load(root: Path, study: str, stamp: str | None = None) -> tuple[SourceDoc, LocateOutput]:
    """Load a study's cache, translating every failure into a clear ``CacheError``.

    A missing, corrupted, or version-incompatible cache must not reach the cockpit
    as a raw pydantic/JSON traceback — the remedy is always the same (re-run
    prepare_paper), so we say exactly that.
    """
    d = cache_dir(root)
    doc = _load_sourcedoc(d, study)
    out = _load_locate(d, study, stamp)
    if doc.source_sha256 and out.source_sha256 and doc.source_sha256 != out.source_sha256:
        raise CacheError(
            f"Cache files for {study!r} refer to different source versions — "
            f"{_rerun_hint(study)}."
        )
    pdf = root / "data" / "sample_papers" / f"{study}.pdf"
    if doc.source_sha256 and pdf.exists():
        actual = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if actual != doc.source_sha256:
            raise CacheError(
                f"Source PDF {pdf.name} changed after ingest — {_rerun_hint(study)}."
            )
    return doc, out


def _load_sourcedoc(d: Path, study: str) -> SourceDoc:
    path = d / f"{study}.sourcedoc.json"
    try:
        return SourceDoc.model_validate_json(path.read_text())
    except FileNotFoundError:
        raise CacheError(f"Cache file {path.name} is missing — {_rerun_hint(study)}.")
    except (ValidationError, ValueError) as e:
        raise CacheError(
            f"Cache file {path.name} could not be read ({type(e).__name__}); it is "
            f"likely corrupted or from an older MetaExtract — {_rerun_hint(study)}."
        )


def _load_locate(d: Path, study: str, stamp: str | None) -> LocateOutput:
    path = _resolve_locate_path(d, study, stamp)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise CacheError(f"Cache file {path.name} is missing — {_rerun_hint(study)}.")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        raise CacheError(
            f"Cache file {path.name} is not valid JSON; it is likely corrupted — "
            f"{_rerun_hint(study)}."
        )

    version = raw.get("schema_version", 0) if isinstance(raw, dict) else 0
    if version > CACHE_SCHEMA_VERSION:
        raise CacheError(
            f"Cache file {path.name} was written by a newer MetaExtract (cache schema "
            f"v{version} > v{CACHE_SCHEMA_VERSION}); upgrade MetaExtract, or "
            f"{_rerun_hint(study)}."
        )
    if version < MIN_SUPPORTED_CACHE_VERSION:
        raise CacheError(
            f"Cache file {path.name} is too old to read (cache schema v{version} < "
            f"v{MIN_SUPPORTED_CACHE_VERSION}) — {_rerun_hint(study)}."
        )

    try:
        return LocateOutput.model_validate(raw)
    except ValidationError:
        raise CacheError(
            f"Cache file {path.name} does not match the current cache schema; it is "
            f"likely from an older MetaExtract — {_rerun_hint(study)}."
        )


def _natural_key(s: str) -> list:
    """Sort key so studies read P1, P2, … P10 (not the string order P1, P10, P2)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_cached(root: Path) -> list[str]:
    d = cache_dir(root)
    studies = (p.name[: -len(".sourcedoc.json")] for p in d.glob("*.sourcedoc.json"))
    return sorted(studies, key=_natural_key)


def list_cache_entries(root: Path) -> list[CacheEntry]:
    """List cache entries the cockpit can open without mixing task packs."""
    d = cache_dir(root)
    entries: list[CacheEntry] = []
    for study in list_cached(root):
        legacy = d / f"{study}.locate.json"
        if legacy.exists():
            entries.append(CacheEntry(study=study, stamp=None))

        prefix = f"{study}."
        suffix = ".locate.json"
        for path in sorted(d.glob(f"{study}.*.locate.json")):
            stamp = path.name[len(prefix): -len(suffix)]
            if stamp:
                entries.append(CacheEntry(study=study, stamp=stamp))
    return sorted(entries, key=lambda e: (_natural_key(e.study), e.stamp or ""))
