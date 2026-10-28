"""Command-line interface for metaextract.

Examples
--------
Extract every PDF in a folder to a tidy CSV::

    metaextract run --input data/sample_papers --out data/outputs/extracted.csv

Evaluate a run against hand-extracted ground truth::

    metaextract eval --pred data/outputs/extracted.csv \\
        --truth eval/ground_truth/truth.csv
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metaextract")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Extract data from a folder of PDFs.")
    p_run.add_argument("--input", required=True, help="Folder containing PDFs.")
    p_run.add_argument("--out", required=True, help="Output CSV path.")
    p_run.add_argument("--model", default=None, help="Gemini model id.")
    p_run.add_argument(
        "--cache", default=None, help="Optional dir to cache per-paper JSON."
    )
    p_run.add_argument(
        "--taskpack",
        default=None,
        help=(
            "Optional YAML task pack that constrains target variables. "
            "Defaults to METAEXTRACT_TASKPACK when set."
        ),
    )

    p_eval = sub.add_parser("eval", help="Score predictions vs. ground truth.")
    p_eval.add_argument("--pred", required=True, help="Predicted CSV.")
    p_eval.add_argument("--truth", required=True, help="Ground-truth CSV.")
    p_eval.add_argument("--tol", type=float, default=0.05, help="Rel. tolerance.")
    p_eval.add_argument("--report", default=None, help="Optional JSON report path.")

    p_agent = sub.add_parser(
        "agent",
        help="Run the bounded extraction agent over a cached, located paper.",
    )
    p_agent.add_argument("study", help="Cached study id, e.g. P1 (run prepare_paper.py first).")
    p_agent.add_argument(
        "--root", default=".", help="Repo root holding data/cache (default: cwd)."
    )
    p_agent.add_argument(
        "--online",
        action="store_true",
        help="Use the live Gemini planner; retained as a shortcut for --provider gemini.",
    )
    p_agent.add_argument(
        "--provider",
        choices=("gemini", "deepseek", "openai-compatible"),
        default=None,
        help="Live planner provider. Omit for the offline replay policy.",
    )
    p_agent.add_argument("--model", default=None, help="Live planner model id.")
    p_agent.add_argument(
        "--base-url",
        default=None,
        help="API base URL for an OpenAI-compatible provider.",
    )
    p_agent.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable holding the provider API key.",
    )
    p_agent.add_argument("--report", default=None, help="Optional JSON path for the full run.")

    args = parser.parse_args(argv)

    if args.command == "run":
        import os

        from .extractor import DEFAULT_MODEL
        from .locate import TaskSpec
        from .pipeline import run_folder

        taskpack = args.taskpack or os.environ.get("METAEXTRACT_TASKPACK")
        task_spec = TaskSpec.from_yaml(taskpack) if taskpack else None
        run_folder(
            args.input,
            args.out,
            model=args.model or DEFAULT_MODEL,
            cache_dir=args.cache,
            task_spec=task_spec,
        )
        return 0

    if args.command == "eval":
        from .evaluate import evaluate, print_report

        report = evaluate(args.pred, args.truth, rel_tol=args.tol)
        print_report(report)
        if args.report:
            import json
            from pathlib import Path

            from .fileio import atomic_write_text

            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(args.report, json.dumps(report, indent=2))
        return 0

    if args.command == "agent":
        from pathlib import Path

        from . import cockpit_cache
        import os

        from .agent import (
            GeminiPlanner,
            OpenAICompatiblePlanner,
            ReplayPlanner,
            build_toolset,
            run_agent,
        )

        root = Path(args.root)
        try:
            doc, out = cockpit_cache.load(root, args.study)
        except cockpit_cache.CacheError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        provider = args.provider or ("gemini" if args.online else None)
        if provider == "gemini":
            from .extractor import DEFAULT_MODEL, _build_client

            try:
                client = _build_client()
            except (KeyError, RuntimeError, ValueError) as e:
                print(f"ERROR: could not configure Gemini: {e}", file=sys.stderr)
                return 2
            model = args.model or os.environ.get("METAEXTRACT_AGENT_MODEL") or DEFAULT_MODEL
            # Gemini can also power the nested Tier-3 extract_values tool, so the
            # planner and toolset intentionally share one client.
            toolset = build_toolset(doc, out, client=client, model=model)
            planner = GeminiPlanner(client=client, model=model)
        elif provider in {"deepseek", "openai-compatible"}:
            if provider == "deepseek":
                model = (
                    args.model
                    or os.environ.get("METAEXTRACT_AGENT_MODEL")
                    or "deepseek-v4-flash"
                )
                base_url = (
                    args.base_url
                    or os.environ.get("METAEXTRACT_AGENT_BASE_URL")
                    or "https://api.deepseek.com"
                )
                api_key_env = args.api_key_env or "DEEPSEEK_API_KEY"
            else:
                model = args.model or os.environ.get("METAEXTRACT_AGENT_MODEL")
                base_url = args.base_url or os.environ.get("METAEXTRACT_AGENT_BASE_URL")
                api_key_env = args.api_key_env or "OPENAI_API_KEY"
            toolset = build_toolset(doc, out)
            planner = OpenAICompatiblePlanner(
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
            )
        else:
            toolset = build_toolset(doc, out)  # offline: extract_values disabled
            planner = ReplayPlanner()

        try:
            result = run_agent(toolset, planner)
        except (KeyError, RuntimeError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        print(result.summary_line())
        for f in result.flagged:
            print(f"  flagged: {f.variable_name or f.region_id} — {f.reason}")
        if args.report:
            import json

            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            from .fileio import atomic_write_text

            atomic_write_text(
                report_path,
                json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
            )
            print(f"full run -> {args.report}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
