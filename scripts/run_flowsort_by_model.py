#!/usr/bin/env python3
"""FlowSort per model as well as per prompt, instead of only on the median.

    .venv312/bin/python scripts/run_flowsort_by_model.py
    .venv312/bin/python scripts/run_flowsort_by_model.py --prompt prompt_1

The prompt-sensitivity study reduces the four models to their median before
FlowSort runs, so ``classification_by_prompt.csv`` has a prompt axis and no model
axis: every class in it is already a consensus class. This script opens that
second axis. It re-runs the classification for every (prompt, model) pair with
that single model's ratings as the qualitative input, plus once per prompt on the
median, and holds every other input at its study value -- GH-FBWM weights,
limiting profiles, quantitative scores, net-flow rule, ``t1``.

That supports two readings the median alone cannot give: how far a single rater
would have moved the classification away from the consensus under a given prompt,
and whether a single rater is more prompt-sensitive than the median is.

The median series is rebuilt from the same JSONL rather than read back from
``consensus/qualitative_consensus_<prompt>.json``, and the run aborts if the two
disagree, so a stale consensus file cannot pass as the baseline.

Writes only under ``data/prompt_sensitivity/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_support_mcp.flowsort import (  # noqa: E402
    qualify_partners_with_flowsort,
)
from src.decision_support_mcp.rating_evaluation import (  # noqa: E402
    prompt_sensitivity as ps,
)
from src.decision_support_mcp.rating_evaluation.dataset import Unit, load_units  # noqa: E402
from src.decision_support_mcp.rating_evaluation.prompt_sensitivity_prompts import (  # noqa: E402
    PROMPT_FEATURES,
    PROMPT_IDS,
    PROMPT_LABELS,
    PROMPT_SHA256_BY_ID,
    RATING_LABELS,
    RUBRIC_SHA256,
)

SCORES_DIR = ps.SCORES_BY_MODEL_DIR

#: Label of the aggregated series. Not a model tag, so it cannot collide.
MEDIAN_KEY = "median"

#: Which prompt every "vs baseline" column is measured against.
BASELINE_PROMPT = "prompt_1"


def _class_index(code: str) -> int:
    """C1 -> 1. Ordered best to worst, so a positive shift is a downgrade."""

    return int(str(code).lstrip("Cc"))


def _series_type(key: str) -> str:
    return "median_of_models" if key == MEDIAN_KEY else "single_model"


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2


def _modal(codes: Sequence[str]) -> tuple[str, bool]:
    """Most frequent class, and whether that mode was a tie.

    ``Counter.most_common`` breaks ties on insertion order, which is what
    ``compare_classifications`` does: over the prompt axis the first prompt is
    the baseline, so a 2-2 split silently reports the baseline class. Mirrored
    here so the columns stay comparable with ``classification_by_prompt.csv``,
    with the tie flagged in its own column rather than left implicit.
    """

    counts = Counter(codes)
    top = max(counts.values())
    return counts.most_common(1)[0][0], sum(1 for n in counts.values() if n == top) > 1


# ---------------------------------------------------------------------------
# Qualitative scores, one series at a time
# ---------------------------------------------------------------------------


def single_model_payload(
    units: Sequence[Unit],
    records: Sequence[dict[str, Any]],
    prompt_id: str,
    tag: str,
) -> dict[str, Any]:
    """One model's ratings in the rater-study handoff format FlowSort reads."""

    ratings: dict[tuple[str, str], int] = {}
    family = tag
    failures = 0
    for record in records:
        if str(record.get("prompt_id")) != prompt_id:
            continue
        if str(record.get("model_tag")) != tag:
            continue
        family = str(record.get("family") or tag)
        if record.get("rating") is None:
            failures += 1
            continue
        ratings[(str(record["company"]), str(record["code"]))] = int(record["rating"])

    scores: list[dict[str, Any]] = []
    lookup: dict[str, int] = {}
    for unit in units:
        rating = ratings.get(unit.key)
        if rating is None:
            scores.append(
                {
                    "company": unit.company,
                    "code": unit.code,
                    "score": None,
                    "status": "no_ratings",
                }
            )
            continue
        scores.append(
            {
                "company": unit.company,
                "code": unit.code,
                "score": rating,
                "n_iterations": 1,
                "status": "ok",
            }
        )
        lookup[f"{unit.company}||{unit.code}"] = rating

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "qualitative sub-criterion scores for the FlowSort evaluation, "
            f"scored under {prompt_id} by {tag} alone"
        ),
        "scale": {"min": int(RATING_LABELS[0]), "max": int(RATING_LABELS[-1])},
        "aggregation": f"none; single model {tag}",
        "provenance": {
            "prompt_id": prompt_id,
            "prompt_label": PROMPT_LABELS[prompt_id],
            "prompt_features": PROMPT_FEATURES[prompt_id],
            "prompt_sha256": PROMPT_SHA256_BY_ID[prompt_id],
            "rubric_sha256": RUBRIC_SHA256,
            "rating_labels": list(RATING_LABELS),
            "temperature": ps.TEMPERATURE,
            "families": {tag: family},
        },
        "coverage": {
            "n_units": len(units),
            "n_model_tags": 1,
            "n_iterations": 1,
            "expected_llm_ratings": len(units),
            "observed_llm_ratings": len(ratings),
            "failed_llm_calls": failures,
            "complete": len(ratings) == len(units),
        },
        "scores": scores,
        "scores_by_company_code": lookup,
    }


def check_median_matches_consensus_file(payload: dict[str, Any], prompt_id: str) -> None:
    """The rebuilt median must equal the consensus file FlowSort published from."""

    path = ps.CONSENSUS_DIR / f"qualitative_consensus_{prompt_id}.json"
    if not path.exists():
        print(f"  note: no consensus file at {path.name}, median cross-check skipped")
        return
    with path.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    on_disk = {
        (str(entry["company"]), str(entry["code"])): entry.get("score")
        for entry in stored.get("scores", [])
    }
    rebuilt = {
        (str(entry["company"]), str(entry["code"])): entry.get("score")
        for entry in payload.get("scores", [])
    }
    if on_disk != rebuilt:
        differing = sorted(
            key
            for key in set(on_disk) | set(rebuilt)
            if on_disk.get(key) != rebuilt.get(key)
        )
        raise SystemExit(
            f"The median rebuilt from {ps.RATINGS_PATH.name} disagrees with {path.name} "
            f"on {len(differing)} unit(s): {differing[:5]}. Regenerate the consensus "
            "file with `run_prompt_sensitivity.py report` before comparing."
        )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def run(prompt_ids: Sequence[str], sector: str) -> dict[str, Any]:
    units = load_units(sector)
    records = ps.load_ratings(ps.RATINGS_PATH)
    available = {str(rec.get("prompt_id")) for rec in records}
    missing = [pid for pid in prompt_ids if pid not in available]
    if missing:
        raise SystemExit(f"No ratings for {', '.join(missing)} in {ps.RATINGS_PATH}.")

    tags = sorted(
        {
            str(rec["model_tag"])
            for rec in records
            if str(rec.get("prompt_id")) in set(prompt_ids)
        }
    )
    series = [*tags, MEDIAN_KEY]
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict[str, Any]]] = {}
    score_paths: dict[str, dict[str, str]] = {}
    for prompt_id in prompt_ids:
        payloads: dict[str, dict[str, Any]] = {
            tag: single_model_payload(units, records, prompt_id, tag) for tag in tags
        }
        payloads[MEDIAN_KEY] = ps.build_consensus_by_prompt(units, records, [prompt_id])[
            prompt_id
        ]
        check_median_matches_consensus_file(payloads[MEDIAN_KEY], prompt_id)

        # FlowSort reads the scores from a file, so each series is written out
        # first. These double as the provenance record for the per-model runs.
        results[prompt_id] = {}
        score_paths[prompt_id] = {}
        for key in series:
            safe = key.replace(":", "_").replace("/", "_")
            path = SCORES_DIR / f"qualitative_scores_{prompt_id}_{safe}.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payloads[key], handle, ensure_ascii=False, indent=2)
            score_paths[prompt_id][key] = str(path)

            print(f"  FlowSort  {prompt_id}  {key}", flush=True)
            results[prompt_id][key] = qualify_partners_with_flowsort(
                companies=None,
                qualitative_source="rater_study",
                rater_study_scores_path=path,
            )

    return {
        "prompt_ids": list(prompt_ids),
        "tags": tags,
        "series": series,
        "results": results,
        "score_paths": score_paths,
    }


def _by_company(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = result["companies"]
    ordered = sorted(entries, key=lambda item: -float(item["net_flow"]))
    ranks = {str(item["company"]): pos for pos, item in enumerate(ordered, start=1)}
    return {
        str(item["company"]): {**item, "rank": ranks[str(item["company"])]}
        for item in entries
    }


def check_median_matches_published_classification(
    table: dict[tuple[str, str], dict[str, dict[str, Any]]],
    prompt_ids: Sequence[str],
) -> None:
    """The median series should reproduce the published prompt-sensitivity run."""

    if not ps.RESULT_PATH.exists():
        return
    with ps.RESULT_PATH.open("r", encoding="utf-8") as handle:
        stored = json.load(handle)
    published = stored.get("flowsort_by_prompt") or {}
    mismatches: list[str] = []
    checked = 0
    for prompt_id in prompt_ids:
        entries = (published.get(prompt_id) or {}).get("companies")
        if not entries:
            continue
        rebuilt = table[(prompt_id, MEDIAN_KEY)]
        for entry in entries:
            company = str(entry["company"])
            checked += 1
            own = rebuilt[company]
            same_class = str(own["assigned_category"]) == str(entry["assigned_category"])
            same_flow = round(float(own["net_flow"]), 6) == round(
                float(entry["net_flow"]), 6
            )
            if not (same_class and same_flow):
                mismatches.append(f"{prompt_id}/{company}")
    if mismatches:
        print(
            f"  WARNING: the median series differs from {ps.RESULT_PATH.name} on "
            f"{len(mismatches)} company-prompt cell(s): {', '.join(mismatches[:5])}. "
            "That file is likely stale; rerun `run_prompt_sensitivity.py report`."
        )
    else:
        print(
            f"  median series reproduces {ps.RESULT_PATH.name} on all {checked} "
            "company-prompt cells"
        )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_csvs(run_result: dict[str, Any], csv_dir: Path) -> dict[str, Path]:
    prompt_ids: list[str] = run_result["prompt_ids"]
    tags: list[str] = run_result["tags"]
    series: list[str] = run_result["series"]

    # (prompt, series) -> company -> entry
    table = {
        (prompt_id, key): _by_company(result)
        for prompt_id, by_series in run_result["results"].items()
        for key, result in by_series.items()
    }
    check_median_matches_published_classification(table, prompt_ids)

    reference = BASELINE_PROMPT if BASELINE_PROMPT in prompt_ids else prompt_ids[0]
    companies = sorted(
        table[(reference, MEDIAN_KEY)],
        key=lambda name: -float(table[(reference, MEDIAN_KEY)][name]["net_flow"]),
    )
    written: dict[str, Path] = {}

    # 1. Tidy master. Every other table here is a view of this one.
    written["flowsort_by_model_long"] = _write_csv(
        csv_dir / "flowsort_by_model_long.csv",
        [
            "prompt_id", "prompt_label", "series", "series_type", "rank", "company",
            "assigned_class", "assigned_label", "positive_flow", "negative_flow",
            "net_flow",
        ],
        [
            [
                prompt_id,
                PROMPT_LABELS[prompt_id],
                key,
                _series_type(key),
                entry["rank"],
                entry["company"],
                entry["assigned_category"],
                entry["assigned_label"],
                round(float(entry["positive_flow"]), 6),
                round(float(entry["negative_flow"]), 6),
                round(float(entry["net_flow"]), 6),
            ]
            for prompt_id in prompt_ids
            for key in series
            for entry in sorted(
                table[(prompt_id, key)].values(), key=lambda item: item["rank"]
            )
        ],
    )

    # 2. Models side by side, one block per prompt: what each rater alone decided
    #    and how far that is from the median of the same prompt.
    rows: list[list[Any]] = []
    for prompt_id in prompt_ids:
        for company in companies:
            model_classes = [
                str(table[(prompt_id, tag)][company]["assigned_category"]) for tag in tags
            ]
            model_flows = [
                float(table[(prompt_id, tag)][company]["net_flow"]) for tag in tags
            ]
            median_entry = table[(prompt_id, MEDIAN_KEY)][company]
            median_class = str(median_entry["assigned_category"])
            indices = [_class_index(code) for code in model_classes]
            modal_class, modal_tied = _modal(model_classes)
            rows.append(
                [prompt_id, company]
                + model_classes
                + [median_class]
                + [round(value, 6) for value in model_flows]
                + [round(float(median_entry["net_flow"]), 6)]
                + [
                    modal_class,
                    modal_tied,
                    len(set(model_classes)),
                    max(indices) - min(indices),
                    sum(1 for code in model_classes if code == median_class),
                    max(abs(idx - _class_index(median_class)) for idx in indices),
                    round(min(model_flows), 6),
                    round(max(model_flows), 6),
                    round(max(model_flows) - min(model_flows), 6),
                    round(_median(model_flows), 6),
                ]
            )
    written["classification_by_model_and_prompt"] = _write_csv(
        csv_dir / "classification_by_model_and_prompt.csv",
        ["prompt_id", "company"]
        + [f"class_{tag}" for tag in tags]
        + ["class_median"]
        + [f"net_flow_{tag}" for tag in tags]
        + [
            "net_flow_median",
            "modal_class_models",
            "modal_class_models_tied",
            "n_distinct_classes_models",
            "class_span_models",
            "n_models_agreeing_with_median",
            "max_shift_vs_median",
            "net_flow_min_models",
            "net_flow_max_models",
            "net_flow_range_models",
            "net_flow_median_of_models",
        ],
        rows,
    )

    # 3. Prompts side by side, one block per series. Same trailing columns as
    #    classification_by_prompt.csv, so the median rows of this file are that
    #    file and the model rows are the same question asked of one rater.
    rows = []
    for key in series:
        for company in companies:
            classes = [
                str(table[(prompt_id, key)][company]["assigned_category"])
                for prompt_id in prompt_ids
            ]
            flows = [
                float(table[(prompt_id, key)][company]["net_flow"])
                for prompt_id in prompt_ids
            ]
            baseline_class = str(table[(reference, key)][company]["assigned_category"])
            indices = [_class_index(code) for code in classes]
            modal_class, modal_tied = _modal(classes)
            rows.append(
                [key, _series_type(key), company]
                + classes
                + [round(value, 6) for value in flows]
                + [
                    baseline_class,
                    modal_class,
                    modal_tied,
                    len(set(classes)),
                    max(indices) - min(indices),
                    max(abs(idx - _class_index(baseline_class)) for idx in indices),
                    len(set(classes)) > 1,
                    round(max(flows) - min(flows), 6),
                ]
            )
    written["classification_by_prompt_and_model"] = _write_csv(
        csv_dir / "classification_by_prompt_and_model.csv",
        ["series", "series_type", "company"]
        + [f"class_{pid}" for pid in prompt_ids]
        + [f"net_flow_{pid}" for pid in prompt_ids]
        + [
            "baseline_class", "modal_class", "modal_class_tied", "n_distinct_classes",
            "class_span", "max_shift_vs_baseline", "changed", "net_flow_range",
        ],
        rows,
    )

    # 4. Per (prompt, series): distance from the median of that same prompt.
    rows = []
    for prompt_id in prompt_ids:
        median_order = [
            entry["company"]
            for entry in sorted(
                table[(prompt_id, MEDIAN_KEY)].values(), key=lambda item: item["rank"]
            )
        ]
        for key in series:
            shifts: list[int] = []
            gaps: list[float] = []
            same = 0
            changes: list[str] = []
            for company in companies:
                own = str(table[(prompt_id, key)][company]["assigned_category"])
                med = str(table[(prompt_id, MEDIAN_KEY)][company]["assigned_category"])
                shifts.append(abs(_class_index(own) - _class_index(med)))
                gaps.append(
                    abs(
                        float(table[(prompt_id, key)][company]["net_flow"])
                        - float(table[(prompt_id, MEDIAN_KEY)][company]["net_flow"])
                    )
                )
                if own == med:
                    same += 1
                else:
                    changes.append(f"{company}: {med}->{own}")
            own_order = [
                entry["company"]
                for entry in sorted(
                    table[(prompt_id, key)].values(), key=lambda item: item["rank"]
                )
            ]
            distribution = Counter(
                str(table[(prompt_id, key)][company]["assigned_category"])
                for company in companies
            )
            rows.append(
                [
                    prompt_id,
                    key,
                    _series_type(key),
                    len(companies),
                    same,
                    round(100.0 * same / len(companies), 2),
                    round(sum(shifts) / len(shifts), 4),
                    max(shifts),
                    round(sum(gaps) / len(gaps), 6),
                    own_order == median_order,
                    "; ".join(f"{code}={distribution[code]}" for code in sorted(distribution)),
                    "; ".join(changes),
                ]
            )
    written["model_vs_median_by_prompt"] = _write_csv(
        csv_dir / "flowsort_model_vs_median_by_prompt.csv",
        [
            "prompt_id", "series", "series_type", "n_companies", "n_same_class_as_median",
            "agreement_with_median_pct", "mean_abs_class_shift_vs_median",
            "max_abs_class_shift_vs_median", "mean_abs_net_flow_gap_vs_median",
            "ranking_matches_median", "class_distribution", "changes_vs_median",
        ],
        rows,
    )

    # 5. Per series: its own prompt sensitivity, so the median's stability can be
    #    read against a single rater's rather than asserted.
    rows = []
    for key in series:
        spans: list[int] = []
        ranges: list[float] = []
        changed = 0
        for company in companies:
            classes = [
                str(table[(prompt_id, key)][company]["assigned_category"])
                for prompt_id in prompt_ids
            ]
            indices = [_class_index(code) for code in classes]
            spans.append(max(indices) - min(indices))
            ranges.append(
                max(
                    float(table[(pid, key)][company]["net_flow"]) for pid in prompt_ids
                )
                - min(float(table[(pid, key)][company]["net_flow"]) for pid in prompt_ids)
            )
            if len(set(classes)) > 1:
                changed += 1

        agreements: list[float] = []
        for left, right in combinations(prompt_ids, 2):
            same = sum(
                1
                for company in companies
                if str(table[(left, key)][company]["assigned_category"])
                == str(table[(right, key)][company]["assigned_category"])
            )
            agreements.append(100.0 * same / len(companies))

        baseline_order = [
            entry["company"]
            for entry in sorted(
                table[(reference, key)].values(), key=lambda item: item["rank"]
            )
        ]
        n_matching = sum(
            1
            for prompt_id in prompt_ids
            if [
                entry["company"]
                for entry in sorted(
                    table[(prompt_id, key)].values(), key=lambda item: item["rank"]
                )
            ]
            == baseline_order
        )
        rows.append(
            [
                key,
                _series_type(key),
                len(prompt_ids),
                len(companies),
                changed,
                round(100.0 * changed / len(companies), 2),
                max(spans),
                round(sum(agreements) / len(agreements), 2) if agreements else "",
                round(min(agreements), 2) if agreements else "",
                round(sum(ranges) / len(ranges), 6),
                round(max(ranges), 6),
                n_matching,
            ]
        )
    written["prompt_stability_by_model"] = _write_csv(
        csv_dir / "flowsort_prompt_stability_by_model.csv",
        [
            "series", "series_type", "n_prompts", "n_companies",
            "n_companies_changing_class", "pct_companies_changing_class",
            "max_class_span", "mean_pairwise_agreement_pct",
            "min_pairwise_agreement_pct", "mean_net_flow_range", "max_net_flow_range",
            "n_rankings_matching_baseline",
        ],
        rows,
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        action="append",
        help="prompt id, repeatable (default: all four)",
    )
    parser.add_argument("--sector", default="semiconductors")
    parser.add_argument("--csv-dir", default=str(ps.CSV_DIR))
    args = parser.parse_args(argv)

    prompt_ids = (
        list(PROMPT_IDS)
        if not args.prompt
        else list(dict.fromkeys(str(item).strip().lower() for item in args.prompt))
    )
    unknown = [pid for pid in prompt_ids if pid not in set(PROMPT_IDS)]
    if unknown:
        raise SystemExit(f"Unknown prompt id(s): {', '.join(unknown)}")

    print(f"FlowSort per model and per prompt  ({', '.join(prompt_ids)})")
    run_result = run(prompt_ids, args.sector)
    print()
    written = write_csvs(run_result, Path(args.csv_dir))

    print()
    for name, path in written.items():
        print(f"  {name:<36} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
