"""Ingest one paper, run the configured locator, and write its cache.

Examples::

    python scripts/prepare_paper.py P1
    python scripts/prepare_paper.py P1 --profile qwen
    python scripts/prepare_paper.py P1 --profiles data/model_profiles.yaml

The default profile is Gemini and has no fallback. A fallback is attempted only
when it is listed by the selected profile (or explicitly with ``--fallback``).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:  # direct script execution and package-style test imports
    from _gt import ROOT, build_spec, load_env
except ModuleNotFoundError:  # pragma: no cover - exercised by import-based callers
    from scripts._gt import ROOT, build_spec, load_env

DEFAULT_TASKPACK = ROOT / "data" / "taskpacks" / "soil_C_fractions.yaml"


def _load_spec(study: str):
    """Prefer the editable task pack YAML; fall back to the GT-derived spec."""
    from metaextract.locate import TaskSpec

    env = os.environ.get("METAEXTRACT_TASKPACK")
    path = Path(env) if env else DEFAULT_TASKPACK
    if path.exists():
        spec = TaskSpec.from_yaml(path)
        print(f"  task pack: {path.name} ({len(spec.target_variables)} metrics)")
        return spec
    print(f"  task pack {path.name} not found -> GT-derived spec")
    spec, _present, _rows = build_spec(study)
    return spec


def _split_names(values: list[str] | None) -> list[str]:
    names: list[str] = []
    for value in values or []:
        names.extend(part.strip() for part in value.split(",") if part.strip())
    return names


def _apply_overrides(profile, args):
    updates = {}
    if args.model:
        updates["model"] = args.model
    if args.transport:
        updates["transport"] = args.transport
    if args.endpoint:
        updates["endpoint"] = args.endpoint
        updates["base_url"] = None
    if args.api_key_env:
        updates["api_key_env"] = args.api_key_env
    if args.response_mode:
        updates["response_mode"] = args.response_mode
    if not updates:
        return profile
    return profile.model_copy(update=updates)


def _locate_with_profiles(locate, doc, spec, config, *, profile_name=None, fallbacks=None, args=None):
    """Run the selected profile chain and annotate a successful fallback."""
    selected = profile_name or config.active_profile
    chain = config.chain(selected, extra_fallbacks=_split_names(fallbacks))
    failures: list[str] = []
    for index, (name, profile) in enumerate(chain):
        if index == 0 and args is not None:
            profile = _apply_overrides(profile, args)
        try:
            print(f"  profile {name}: {profile.transport}/{profile.model or '(default model)'}")
            out = locate(doc, spec=spec, profile=profile, profile_name=name)
            if failures:
                out.fallback_used = True
                out.fallback_reasons = failures
                out.fallback_reason = "; ".join(failures)
                out.problems.extend(f"fallback: {reason}" for reason in failures)
            return out
        except Exception as exc:
            reason = f"{name} failed: {type(exc).__name__}: {exc}"
            failures.append(reason)
            if index + 1 >= len(chain):
                raise
            next_name = chain[index + 1][0]
            print(f"  {reason}; trying configured fallback {next_name}", file=sys.stderr)
    raise RuntimeError("no model profile was available")


def _locate_with_retry(locate, doc, spec, models, attempts=4):
    """Legacy helper retained for callers that pass a list of Gemini models.

    It is no longer used by ``main`` and does not select a different model unless
    the caller explicitly supplies the list.
    """
    last = None
    for model in models:
        for k in range(attempts):
            try:
                print(f"  calling {model} (try {k + 1})...")
                return locate(doc, spec=spec, model=model)
            except Exception as exc:
                last = exc
                if "503" in str(exc) or "UNAVAILABLE" in str(exc):
                    time.sleep(6 * (k + 1))
                else:
                    break
    raise last


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="prepare one paper for the MetaExtract cockpit")
    parser.add_argument("study", nargs="?", default="P1", help="paper id, e.g. P1")
    parser.add_argument("model", nargs="?", help="legacy model override for the active profile")
    parser.add_argument("--profile", help="named profile from the profile config")
    parser.add_argument(
        "--profiles",
        "--config",
        dest="profiles",
        help="YAML profile config (defaults to data/model_profiles.yaml when present)",
    )
    parser.add_argument("--fallback", action="append", help="explicit fallback profile name(s), comma-separated")
    parser.add_argument("--transport", choices=["gemini", "openai_compatible"])
    parser.add_argument("--endpoint", "--base-url", dest="endpoint")
    parser.add_argument("--api-key-env")
    parser.add_argument("--response-mode", choices=["auto", "json_schema", "json_object"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_env()

    from metaextract import cockpit_cache
    from metaextract.ingest import IngestError, ingest_pdf
    from metaextract.locate import locate
    from metaextract.model_profiles import load_model_profiles

    try:
        config = load_model_profiles(args.profiles)
        selected = (
            args.profile
            or os.environ.get("METAEXTRACT_PROFILE")
            or os.environ.get("METAEXTRACT_MODEL_PROFILE")
            or config.active_profile
        )
        # Validate the name before reading a potentially large PDF.
        config.profile(selected)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    pdf = ROOT / "data" / "sample_papers" / f"{args.study}.pdf"
    spec = _load_spec(args.study)
    try:
        doc = ingest_pdf(pdf)
    except (FileNotFoundError, IngestError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"{args.study}: ingested {doc.n_pages} pages, {len(doc.blocks)} blocks.")

    try:
        out = _locate_with_profiles(
            locate,
            doc,
            spec,
            config,
            profile_name=selected,
            fallbacks=args.fallback,
            args=args,
        )
    except Exception as exc:
        print(f"ERROR: locator failed: {exc}", file=sys.stderr)
        return 1

    cockpit_cache.save(ROOT, args.study, doc, out)
    fallback_note = " (fallback used)" if out.fallback_used else ""
    print(
        f"cached -> data/cache/{args.study}.*.json | "
        f"{len(out.regions)} located regions, {len(out.problems)} problems{fallback_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
