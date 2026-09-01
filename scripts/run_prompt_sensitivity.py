#!/usr/bin/env python3
"""CLI for the scoring-prompt sensitivity analysis.

Run from the project root:

    .venv312/bin/python scripts/run_prompt_sensitivity.py scope
    .venv312/bin/python scripts/run_prompt_sensitivity.py show-prompt --prompt prompt_3
    .venv312/bin/python scripts/run_prompt_sensitivity.py collect --iterations 100
    .venv312/bin/python scripts/run_prompt_sensitivity.py report

``collect`` sends each of the four prompts to each of the four study LLMs for
every qualitative company-sub-criterion unit, at temperature 0, ``--iterations``
times over. It appends to JSONL and resumes from whatever is already on disk, so
an interrupted run is restarted with the same command.

``report`` reduces each model's iterations to their median, takes the median
across the four models per prompt, runs FlowSort once per prompt with every other
input held at its study value, and writes the comparison tables to CSV.

Writes only under ``data/prompt_sensitivity/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_support_mcp.rating_evaluation import prompt_sensitivity as ps  # noqa: E402
from src.decision_support_mcp.rating_evaluation.collect import ModelSpec  # noqa: E402
from src.decision_support_mcp.rating_evaluation.dataset import (  # noqa: E402
    describe_units,
    load_units,
)
from src.decision_support_mcp.rating_evaluation.prompt_sensitivity_prompts import (  # noqa: E402
    PROMPT_FEATURES,
    PROMPT_IDS,
    PROMPT_LABELS,
    PROMPT_SHA256_BY_ID,
    build_prompt,
)


def _models(args: argparse.Namespace) -> list[ModelSpec]:
    if not args.model:
        return list(ps.DEFAULT_MODELS)
    return [ModelSpec.parse(item) for item in args.model]


def _prompts(args: argparse.Namespace) -> list[str]:
    if not args.prompt:
        return list(PROMPT_IDS)
    return list(dict.fromkeys(str(item).strip().lower() for item in args.prompt))


def cmd_scope(args: argparse.Namespace) -> int:
    scope = describe_units(args.sector)
    units = load_units(args.sector)
    models = _models(args)
    prompts = _prompts(args)
    total = len(units) * len(models) * len(prompts) * args.iterations
    print(json.dumps(scope, indent=2, ensure_ascii=False))
    print()
    for unit in units:
        print(f"  {unit.code:<16} {unit.company:<52} disclosure {len(unit.disclosure):>5} chars")
    print()
    print(f"prompts     {len(prompts)}  ({', '.join(prompts)})")
    print(f"models      {len(models)}  ({', '.join(spec.tag for spec in models)})")
    print(f"units       {len(units)}")
    print(f"iterations  {args.iterations}")
    print(f"total calls {total}")
    for seconds in (12, 20, 26):
        hours = total * seconds / 3600
        print(f"  at {seconds:>2}s/call   {hours:7.1f} h  ({hours/24:5.1f} d)")
    return 0


def cmd_show_prompt(args: argparse.Namespace) -> int:
    units = load_units(args.sector)
    unit = units[0]
    for prompt_id in _prompts(args):
        features = PROMPT_FEATURES[prompt_id]
        print("=" * 78)
        print(f"{prompt_id}  |  {PROMPT_LABELS[prompt_id]}")
        print(f"sha256 {PROMPT_SHA256_BY_ID[prompt_id]}")
        print(f"guidance={features['sasb_guidance']}  rubric={features['scoring_rubric']}")
        print("=" * 78)
        if args.rendered:
            print(build_prompt(prompt_id, unit.measure, unit.guidance, unit.disclosure))
        else:
            print(build_prompt(prompt_id, "{measure}", "{guidance}", "{value}"))
        print()
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    units = load_units(args.sector)
    summary = ps.collect(
        units=units,
        models=_models(args),
        prompt_ids=_prompts(args),
        iterations=args.iterations,
        out_path=Path(args.out),
        temperature=args.temperature,
        base_url=args.base_url,
        timeout_sec=args.timeout,
        max_attempts=args.max_attempts,
        progress_every=args.progress_every,
        unload_between_models=not args.keep_loaded,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


def cmd_export_baseline(args: argparse.Namespace) -> int:
    try:
        summary = ps.export_baseline_ratings(
            ratings_path=Path(args.ratings),
            out_path=Path(args.out),
            prompt_id=args.prompt_id,
            backup=not args.no_backup,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    print()
    print("Next, regenerate the agreement report and the FlowSort consensus scores:")
    print("  .venv312/bin/python scripts/run_rating_evaluation.py report")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    units = load_units(args.sector)
    try:
        result = ps.analyse(
            units=units,
            ratings_path=Path(args.ratings),
            prompt_ids=_prompts(args),
            baseline=args.baseline,
            consensus_dir=Path(args.consensus_dir),
            csv_dir=Path(args.csv_dir),
            result_path=Path(args.out),
            write_csv=not args.no_csv,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(ps.format_report(result))
    print()
    for name, path in sorted(result.get("csv_files", {}).items()):
        print(f"CSV       {name:<26} {path}")
    print(f"JSON      {'full result':<26} {result.get('result_path')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sector", default="semiconductors")
    parser.add_argument("--prompt", action="append", default=[], help="repeatable; defaults to all four")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scope = subparsers.add_parser("scope", help="show the units, the prompts and the call budget")
    scope.add_argument("--model", action="append", default=[])
    scope.add_argument("--iterations", type=int, default=ps.DEFAULT_ITERATIONS)
    scope.set_defaults(func=cmd_scope)

    show = subparsers.add_parser("show-prompt", help="print the prompt texts")
    show.add_argument("--rendered", action="store_true", help="fill in a real unit instead of placeholders")
    show.set_defaults(func=cmd_show_prompt)

    collect = subparsers.add_parser("collect", help="run the scoring calls for every prompt")
    collect.add_argument("--model", action="append", default=[])
    collect.add_argument(
        "--iterations",
        type=int,
        default=ps.DEFAULT_ITERATIONS,
        help="repeats per model, prompt and unit (default %(default)s)",
    )
    collect.add_argument("--temperature", type=float, default=ps.TEMPERATURE)
    collect.add_argument("--out", default=str(ps.RATINGS_PATH))
    collect.add_argument("--base-url", default=ps.DEFAULT_BASE_URL)
    collect.add_argument("--timeout", type=float, default=ps.DEFAULT_TIMEOUT_SEC)
    collect.add_argument("--max-attempts", type=int, default=ps.DEFAULT_MAX_ATTEMPTS)
    collect.add_argument("--progress-every", type=int, default=6)
    collect.add_argument(
        "--keep-loaded",
        action="store_true",
        help="leave each model resident after its ratings are collected",
    )
    collect.set_defaults(func=cmd_collect)

    export = subparsers.add_parser(
        "export-baseline",
        help="adopt prompt_1's ratings as the rater-agreement evaluation's input",
    )
    export.add_argument("--ratings", default=str(ps.RATINGS_PATH))
    export.add_argument("--out", default=str(ps.LLM_RATINGS_PATH))
    export.add_argument("--prompt-id", default="prompt_1", dest="prompt_id")
    export.add_argument("--no-backup", action="store_true")
    export.set_defaults(func=cmd_export_baseline)

    report = subparsers.add_parser("report", help="aggregate, classify and compare")
    report.add_argument("--ratings", default=str(ps.RATINGS_PATH))
    report.add_argument("--baseline", default="prompt_1")
    report.add_argument("--consensus-dir", default=str(ps.CONSENSUS_DIR))
    report.add_argument("--csv-dir", default=str(ps.CSV_DIR))
    report.add_argument("--out", default=str(ps.RESULT_PATH))
    report.add_argument("--no-csv", action="store_true")
    report.set_defaults(func=cmd_report)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
