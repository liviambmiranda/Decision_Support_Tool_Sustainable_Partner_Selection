#!/usr/bin/env python3
"""CLI for the criterion-weight Monte Carlo sensitivity analysis.

Every criterion weight is drawn from a uniform density over its baseline value
plus or minus ``--pct``, the whole weight vector is resampled ``--samples``
times, and FlowSort is re-run on every draw. The limiting profiles, the company
evaluations and the preference function are held at their baseline values.

    scripts/run_weight_monte_carlo.py run [--series median] [options]
    scripts/run_weight_monte_carlo.py by-series [--pcts ...] [--seeds ...]

``run`` is one simulation on one scoring series. ``by-series`` runs the identical
protocol once per LLM and once on their median, and writes the comparison as two
CSVs. Passing several values to ``--pcts`` or ``--seeds`` repeats the whole
comparison at each one: the range sweep locates where each series breaks, and the
seed sweep shows whether the ordering of the series survives a different sample
path.

Why ``by-series`` exists
------------------------

The study reports the weight robustness on one qualitative input: the median
across the four LLMs. That leaves an alternative reading open. A classification
that survives a plus/minus 20% perturbation of the weights may be robust because
the weight vector is not the binding constraint, or it may be robust because
taking the median across four models first removed the disagreement that the
perturbation would otherwise have exposed. The two are told apart by holding the
weight protocol fixed and swapping the qualitative input for one model's ratings
at a time, which is what this command does.

The per-model runs are diagnostics for the aggregated result, not five competing
headline numbers. A single model's ratings are the input the study explicitly
does not use.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decision_support_mcp.rating_evaluation import (  # noqa: E402
    prompt_sensitivity as ps,
)
from src.decision_support_mcp.rating_evaluation.dataset import EVAL_DIR  # noqa: E402
from src.decision_support_mcp.sensitivity_analysis import (  # noqa: E402
    MONTE_CARLO_CSV_DIR,
    MONTE_CARLO_DEFAULT_SAMPLES,
    run_weight_monte_carlo_sensitivity,
)

SCORES_DIR = ps.SCORES_BY_MODEL_DIR

#: Where ``run_flowsort_by_human.py`` wrote the expert score files.
HUMAN_SCORES_DIR = EVAL_DIR / "scores_by_human"

#: Human series are namespaced. Both sides of the study call their aggregate
#: "median", so a bare ``median`` has to keep meaning the LLM one -- it is the
#: study baseline that every existing run and CSV refers to.
HUMAN_PREFIX = "human:"

#: Label of the aggregated series, and the study baseline. Not a model tag, so it
#: cannot collide with one.
MEDIAN_KEY = "median"

#: Prompt the study baseline was collected under.
BASELINE_PROMPT = "prompt_1"

#: The perturbation range the paper reports for the weights. Wider than the
#: module default, which is why it is repeated here rather than inherited.
DEFAULT_PERTURBATION_PCT = 0.20

CATEGORY_CODES = ("C1", "C2", "C3")


def _scores_path(prompt_id: str, series: str) -> Path:
    """Where the FlowSort scripts wrote one series' qualitative scores.

    Read rather than recomputed, so this analysis cannot drift from the
    classification the rest of the paper reports for the same series.
    """

    if series.startswith(HUMAN_PREFIX):
        rater = series[len(HUMAN_PREFIX) :]
        return HUMAN_SCORES_DIR / f"qualitative_scores_human_{rater}.json"
    safe = series.replace(":", "_").replace("/", "_")
    return SCORES_DIR / f"qualitative_scores_{prompt_id}_{safe}.json"


def human_series() -> list[str]:
    """Expert series present on disk, in rater order, then their median.

    Read off the files rather than listed here, for the same reason the paper
    tables derive their expert rows from the data: how many experts there are is a
    property of the ratings CSV.
    """

    prefix = "qualitative_scores_human_"
    keys = sorted(
        path.stem[len(prefix) :]
        for path in HUMAN_SCORES_DIR.glob(f"{prefix}*.json")
        if path.stem[len(prefix) :] != MEDIAN_KEY
    )
    return [f"{HUMAN_PREFIX}{key}" for key in [*keys, MEDIAN_KEY]]


def llm_series() -> list[str]:
    """Model tags in alphabetical order, then their median.

    Alphabetical rather than in ``DEFAULT_MODELS`` order, to match the column
    order of the rater-comparison table; the median goes last, as the aggregation
    the individual series are being read against.
    """

    return [*sorted(spec.tag for spec in ps.DEFAULT_MODELS), MEDIAN_KEY]


def available_series(prompt_id: str) -> list[str]:
    """Every series this analysis can be pointed at, LLM side then human side."""

    return [*llm_series(), *human_series()]


def resolve_series(requested: list[str] | None, prompt_id: str) -> list[tuple[str, Path]]:
    """Pair each requested series with its score file, failing on a missing one.

    A missing file means that series was never scored under this prompt. Skipping
    it silently would leave a comparison table whose rows are the series that
    happened to exist, so it is an error instead.

    The default is the LLM side alone: it is the axis the study varies, and the
    human series are requested by name or with ``--include-humans``.
    """

    known = available_series(prompt_id)
    series = requested or llm_series()
    unknown = [name for name in series if name not in known]
    if unknown:
        raise SystemExit(
            f"Unknown series: {', '.join(unknown)}. Available under {prompt_id}: "
            f"{', '.join(known)}."
        )
    resolved = [(name, _scores_path(prompt_id, name)) for name in series]
    missing = [str(path) for _, path in resolved if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing qualitative score file(s):\n  "
            + "\n  ".join(missing)
            + "\nRun scripts/run_flowsort_by_model.py first."
        )
    return resolved


def _series_type(series: str) -> str:
    """What kind of rater the series is, for the ``series_type`` column."""

    if series.startswith(HUMAN_PREFIX):
        rater = series[len(HUMAN_PREFIX) :]
        return "median_of_experts" if rater == MEDIAN_KEY else "single_expert"
    return "median_of_models" if series == MEDIAN_KEY else "single_model"


def _sweep_label(prefix: str, values: list[float]) -> str:
    """``pct20`` for a single value, ``pcts5-10-15-20`` for a swept one."""

    parts = [f"{value:g}".replace(".", "_") for value in values]
    if len(parts) == 1:
        return f"{prefix}{parts[0]}"
    return f"{prefix}s{'-'.join(parts)}"


def _run(
    series: str,
    path: Path,
    args: argparse.Namespace,
    write_csv: bool,
    seed: int | None = None,
    pct: float | None = None,
) -> dict[str, Any]:
    return run_weight_monte_carlo_sensitivity(
        companies=args.companies,
        rater_study_scores_path=path,
        series_label=series,
        rule=args.rule,
        preference_function=args.preference_function,
        samples=args.samples,
        perturbation_pct=args.pct if pct is None else pct,
        seed=args.seed if seed is None else seed,
        write_csv=write_csv,
        csv_output_dir=str(args.output_dir),
        include_draws=getattr(args, "include_draws", False),
    )


def _summary_row(series: str, path: Path, result: dict[str, Any]) -> dict[str, Any]:
    stability = result["stability"]
    top = result["criterion_contributions"][0]
    distribution = result["baseline"]["class_distribution"]
    by_company = {item["company"]: item for item in result["companies"]}
    return {
        "qualitative_series": series,
        "series_type": _series_type(series),
        "is_study_baseline": series == MEDIAN_KEY,
        "samples": result["monte_carlo"]["samples"],
        "perturbation_pct": result["monte_carlo"]["perturbation_pct"],
        "seed": result["monte_carlo"]["seed"],
        "identical_classification_pct": stability["identical_classification_pct"],
        "mean_exact_class_match_pct": stability["mean_exact_class_match_pct"],
        "min_exact_class_match_pct": stability["min_exact_class_match_pct"],
        "ranking_preserved_pct": stability["ranking_preserved_pct"],
        "mean_ranking_spearman_vs_baseline": stability["mean_ranking_spearman_vs_baseline"],
        "min_ranking_spearman_vs_baseline": stability["min_ranking_spearman_vs_baseline"],
        "mean_reclassified_companies": stability["mean_reclassified_companies"],
        "max_reclassified_companies": stability["max_reclassified_companies"],
        "mean_absolute_class_shift": stability["mean_absolute_class_shift"],
        "max_absolute_class_shift": stability["max_absolute_class_shift"],
        "top_criterion": top["code"],
        "top_criterion_contribution_pct": top["contribution_pct"],
        **{f"baseline_{code}_count": distribution.get(code, 0) for code in CATEGORY_CODES},
        # The least stable company is what a reader of the headline number wants
        # next: an aggregate of 100% and an aggregate carried by one fragile
        # company are different results.
        "least_stable_company": min(
            by_company.values(), key=lambda item: item["stability_pct"]
        )["company"],
        "least_stable_company_pct": min(
            item["stability_pct"] for item in by_company.values()
        ),
        "qualitative_scores_path": str(path),
    }


def _company_rows(series: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per company. Carries the range and the seed, not just the series.

    A sweep writes every (range, seed, series) combination into one file, so
    without those two columns the rows of different runs are indistinguishable.
    """

    rows: list[dict[str, Any]] = []
    for item in result["companies"]:
        flow = item["net_flow"]
        rows.append(
            {
                "qualitative_series": series,
                "series_type": _series_type(series),
                "perturbation_pct": result["monte_carlo"]["perturbation_pct"],
                "seed": result["monte_carlo"]["seed"],
                "samples": result["monte_carlo"]["samples"],
                "company": item["company"],
                "baseline_category": item["baseline_category"],
                "most_frequent_category": item["most_frequent_category"],
                **{
                    f"probability_{code}_pct": item["class_probability_pct"].get(code, 0.0)
                    for code in CATEGORY_CODES
                },
                "stability_pct": item["stability_pct"],
                "mean_absolute_class_shift": item["mean_absolute_class_shift"],
                "baseline_net_flow": item["baseline_net_flow"],
                "net_flow_mean": flow["mean"],
                "net_flow_std": flow["std"],
                "net_flow_min": flow["min"],
                "net_flow_p2_5": flow["p2_5"],
                "net_flow_p50": flow["p50"],
                "net_flow_p97_5": flow["p97_5"],
                "net_flow_max": flow["max"],
                "top_driver": item["top_drivers"][0]["code"] if item["top_drivers"] else "",
                "top_driver_srcc_net_flow": (
                    item["top_drivers"][0]["srcc_net_flow"] if item["top_drivers"] else ""
                ),
            }
        )
    return rows


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def command_run(args: argparse.Namespace) -> int:
    (series, path), = resolve_series([args.series], args.prompt)
    result = _run(series, path, args, write_csv=not args.no_csv)
    stability = result["stability"]
    print(
        f"Weight Monte Carlo  series={series}  pct={args.pct:g}  "
        f"samples={args.samples}  seed={args.seed}"
    )
    print(f"  identical classification   {stability['identical_classification_pct']:.2f}%")
    print(f"  ranking preserved          {stability['ranking_preserved_pct']:.2f}%")
    print(f"  mean exact class match     {stability['mean_exact_class_match_pct']:.2f}%")
    print("\n  per company")
    for item in sorted(result["companies"], key=lambda entry: entry["stability_pct"]):
        print(
            f"    {item['company']:<56} {item['baseline_category']}  "
            f"stable {item['stability_pct']:>6.2f}%"
        )
    for name in result.get("csv_output", {}).get("files", []):
        print(f"\n  {name}")
    return 0


def command_by_series(args: argparse.Namespace) -> int:
    """Run the identical protocol on every scoring series and compare.

    Per-series CSVs are written only with ``--per-series-csv``. The comparison is
    the output being asked for here, and one full set of tables per series is a
    lot of files to leave behind for a diagnostic.
    """

    requested = args.series
    if getattr(args, "include_humans", False):
        # Appended to whatever was asked for, so --include-humans widens an
        # explicit --series list instead of silently replacing it.
        requested = [*(requested or llm_series()), *human_series()]
    resolved = resolve_series(requested, args.prompt)
    summary: list[dict[str, Any]] = []
    companies: list[dict[str, Any]] = []

    # Sweeping the range says where each series breaks rather than how it stands
    # at one range; sweeping the seed says whether the ordering of the series is a
    # property of the ratings or of the sample path. Both are the objections a
    # per-model reading invites, so both are one run.
    nested = len(args.seeds) > 1 or len(args.pcts) > 1
    for pct in args.pcts:
        for seed in args.seeds:
            if nested:
                print(f"  pct={pct:g}  seed={seed}")
            for series, path in resolved:
                result = _run(
                    series, path, args, write_csv=args.per_series_csv, seed=seed, pct=pct
                )
                summary.append(_summary_row(series, path, result))
                companies.extend(_company_rows(series, result))
                row = summary[-1]
                indent = "    " if nested else "  "
                print(
                    f"{indent}{series:<18} identical={row['identical_classification_pct']:>7.2f}%  "
                    f"ranking={row['ranking_preserved_pct']:>7.2f}%  "
                    f"weakest={row['least_stable_company_pct']:>6.2f}% "
                    f"({row['least_stable_company']})",
                    flush=True,
                )

    stem = f"weight_mcs_by_series_{_sweep_label('pct', [p * 100 for p in args.pcts])}_{_sweep_label('seed', args.seeds)}"
    summary_path = _write_dict_csv(args.output_dir / f"{stem}_summary.csv", summary)
    companies_path = _write_dict_csv(args.output_dir / f"{stem}_companies.csv", companies)
    print(f"\n  {summary_path}\n  {companies_path}")
    return 0


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--samples", type=int, default=MONTE_CARLO_DEFAULT_SAMPLES)
    parser.add_argument(
        "--pct",
        type=float,
        default=DEFAULT_PERTURBATION_PCT,
        help="perturbation range as a fraction, e.g. 0.20 for plus or minus 20%%",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule", default="net", choices=["net", "positive", "negative"])
    parser.add_argument(
        "--preference-function",
        default="t1",
        help="PROMETHEE preference function held fixed across the draws (default: t1)",
    )
    parser.add_argument("--companies", nargs="*", default=None)
    parser.add_argument(
        "--prompt",
        default=BASELINE_PROMPT,
        help=f"prompt the scores were collected under (default: {BASELINE_PROMPT})",
    )
    parser.add_argument("--output-dir", type=Path, default=MONTE_CARLO_CSV_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="one simulation on one scoring series"
    )
    _add_shared(run_parser)
    run_parser.add_argument(
        "--series",
        default=MEDIAN_KEY,
        help=(
            f"model tag, {MEDIAN_KEY!r}, or a human series as {HUMAN_PREFIX}<rater> "
            f"e.g. {HUMAN_PREFIX}{MEDIAN_KEY} for the human reference "
            f"(default: {MEDIAN_KEY}, the study baseline)"
        ),
    )
    run_parser.add_argument(
        "--include-draws",
        action="store_true",
        help="also write the draw-by-draw sample (several megabytes at 10,000 draws)",
    )
    run_parser.add_argument("--no-csv", action="store_true", help="print only, write nothing")

    series_parser = subparsers.add_parser(
        "by-series",
        help="run once per LLM and once on their median, into one comparison CSV",
    )
    _add_shared(series_parser)
    series_parser.add_argument(
        "--series",
        nargs="*",
        default=None,
        help="series to run (default: every model, then their median)",
    )
    series_parser.add_argument(
        "--include-humans",
        action="store_true",
        help=(
            "append the human expert series and their median to the default set, "
            "so the experts and the models are compared under one protocol"
        ),
    )
    series_parser.add_argument(
        "--pcts",
        type=float,
        nargs="*",
        default=None,
        help=(
            "repeat the whole comparison at each perturbation range, e.g. "
            "0.05 0.10 0.15 0.20 (default: just --pct)"
        ),
    )
    series_parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="repeat the whole comparison on each seed (default: just --seed)",
    )
    series_parser.add_argument(
        "--per-series-csv",
        action="store_true",
        help="also write the full per-run CSV tables for each series",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return command_run(args)
    args.seeds = args.seeds or [args.seed]
    args.pcts = args.pcts or [args.pct]
    return command_by_series(args)


if __name__ == "__main__":
    raise SystemExit(main())
