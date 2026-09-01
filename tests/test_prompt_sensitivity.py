#!/usr/bin/env python3
"""Tests for the scoring-prompt sensitivity analysis.

No LLM is called: the collection layer is exercised through synthetic rating
records, so the aggregation, the FlowSort handoff and the comparison tables are
verified without a live Ollama.

Run:  .venv312/bin/python tests/test_prompt_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_support_mcp.flowsort import qualify_partners_with_flowsort
from src.decision_support_mcp.rating_evaluation import prompt_sensitivity as ps
from src.decision_support_mcp.rating_evaluation.dataset import load_units
from src.decision_support_mcp.rating_evaluation.prompt_sensitivity_prompts import (
    PROMPT_FEATURES,
    PROMPT_IDS,
    PROMPT_SHA256_BY_ID,
    PROMPT_TEMPLATES,
    SCORING_RUBRIC,
    build_prompt,
)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def section(title: str) -> None:
    print(f"\n{title}")


MODEL_TAGS = ["llama3.1:8b", "qwen3.5:9b", "deepseek-v2:16b", "gemma4:12b"]


def synthetic_records(units, ratings_by_prompt):
    """One record per (prompt, model, unit), with a caller-chosen rating."""

    records = []
    for prompt_id in PROMPT_IDS:
        for tag in MODEL_TAGS:
            for unit in units:
                records.append(
                    {
                        "family": tag.split(":")[0],
                        "model_tag": tag,
                        "prompt_id": prompt_id,
                        "prompt_label": prompt_id,
                        "iteration": 1,
                        "seed": 1,
                        "temperature": 0.0,
                        "company": unit.company,
                        "code": unit.code,
                        "rating": ratings_by_prompt(prompt_id, tag, unit),
                        "justification": "synthetic",
                        "attempts": 1,
                        "failure": None,
                        "latency_sec": 0.0,
                        "prompt_sha256": PROMPT_SHA256_BY_ID[prompt_id],
                        "rubric_sha256": ps.RUBRIC_SHA256,
                        "recorded_at": "2026-01-01T00:00:00+00:00",
                    }
                )
    return records


def test_prompt_bank() -> None:
    section("Prompt bank")
    check(len(PROMPT_TEMPLATES) == 4, "four prompts are defined")
    check(
        len(set(PROMPT_SHA256_BY_ID.values())) == 4,
        "the four prompts are textually distinct",
    )
    check(
        list(PROMPT_IDS) == ["prompt_1", "prompt_2", "prompt_3", "prompt_4"],
        "prompt order is fixed",
    )

    rendered = {
        prompt_id: build_prompt(prompt_id, "MEASURE-X", "GUIDANCE-Y", "DISCLOSURE-Z")
        for prompt_id in PROMPT_IDS
    }
    for prompt_id, text in rendered.items():
        features = PROMPT_FEATURES[prompt_id]
        check(
            ("GUIDANCE-Y" in text) == features["sasb_guidance"],
            f"{prompt_id} includes the SASB guidance iff it declares it",
        )
        check(
            (SCORING_RUBRIC in text) == features["scoring_rubric"],
            f"{prompt_id} includes the rubric iff it declares it",
        )
        check("DISCLOSURE-Z" in text, f"{prompt_id} carries the disclosure")
        check("MEASURE-X" in text, f"{prompt_id} carries the measure")
        check(
            "integer from 1 to 5" in text,
            f"{prompt_id} states the 1-5 scale",
        )
        check("{" in text and '"rating": integer' in text, f"{prompt_id} keeps its JSON schema literal")

    # The disclosure is passed through verbatim: the templates delimit it with
    # triple quotes, so escaping would corrupt the text the model reads.
    quoted = build_prompt("prompt_4", "M", "G", 'a "quoted" value')
    check('a "quoted" value' in quoted, "the disclosure reaches the prompt unescaped")

    try:
        build_prompt("prompt_9", "M", "G", "D")
        check(False, "an unknown prompt id is rejected")
    except ValueError:
        check(True, "an unknown prompt id is rejected")


def test_aggregation_is_median_across_models() -> None:
    section("Aggregation")
    units = load_units()[:2]

    # Ratings 1, 2, 4, 5 for the four models: the lower middle value is 2.
    ladder = {tag: value for tag, value in zip(MODEL_TAGS, [1, 2, 4, 5])}
    records = synthetic_records(units, lambda prompt, tag, unit: ladder[tag])
    consensus = ps.build_consensus_by_prompt(units, records)

    for prompt_id in PROMPT_IDS:
        payload = consensus[prompt_id]
        check(
            payload["coverage"]["complete"],
            f"{prompt_id} coverage is complete with all four models present",
        )
        check(
            payload["coverage"]["expected_llm_ratings"] == len(units) * 4,
            f"{prompt_id} expects one rating per model per unit",
        )
        scores = set(payload["scores_by_company_code"].values())
        check(scores == {2}, f"{prompt_id} resolves the even-count tie downward to 2")

    # A prompt that shifts every model by one shifts the median by one.
    shifted = synthetic_records(
        units,
        lambda prompt, tag, unit: min(5, ladder[tag] + (1 if prompt == "prompt_4" else 0)),
    )
    consensus = ps.build_consensus_by_prompt(units, shifted)
    check(
        set(consensus["prompt_1"]["scores_by_company_code"].values()) == {2},
        "prompt_1 median is unchanged",
    )
    check(
        set(consensus["prompt_4"]["scores_by_company_code"].values()) == {3},
        "prompt_4 median follows the shifted ratings",
    )


def test_incomplete_coverage_is_refused() -> None:
    section("Coverage guard")
    units = load_units()[:2]
    records = synthetic_records(units, lambda prompt, tag, unit: 3)
    # Drop every prompt_3 rating from one model.
    records = [
        record
        for record in records
        if not (record["prompt_id"] == "prompt_3" and record["model_tag"] == MODEL_TAGS[0])
    ]
    consensus = ps.build_consensus_by_prompt(units, records)
    check(
        consensus["prompt_1"]["coverage"]["complete"],
        "a fully covered prompt is reported complete",
    )
    # prompt_3 now has three models for every unit, which is internally
    # consistent but is a different sample than the other prompts.
    check(
        consensus["prompt_3"]["coverage"]["n_model_tags"] == 3,
        "a prompt missing a model is visible in its coverage",
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ratings.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        # Force an actually-missing cell so `complete` is False.
        trimmed = [
            record
            for record in records
            if not (
                record["prompt_id"] == "prompt_2"
                and record["company"] == units[0].company
                and record["model_tag"] == MODEL_TAGS[1]
            )
        ]
        path.write_text(
            "\n".join(json.dumps(record) for record in trimmed) + "\n", encoding="utf-8"
        )
        try:
            ps.analyse(units=units, ratings_path=path, result_path=None, write_csv=False)
            check(False, "analyse refuses prompts with missing ratings")
        except ValueError as exc:
            check("missing ratings" in str(exc), "analyse refuses prompts with missing ratings")


def test_provenance_guard() -> None:
    section("Provenance guard")
    units = load_units()[:1]
    records = synthetic_records(units, lambda prompt, tag, unit: 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ratings.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        check(ps.check_provenance(path)["ok"], "matching prompt hashes pass the guard")

        records[0]["prompt_sha256"] = "0" * 64
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        guard = ps.check_provenance(path)
        check(not guard["ok"], "a rating from an unknown prompt fails the guard")
        check(
            guard["stale_prompt_hashes"] == ["0" * 64],
            "the guard names the offending hash",
        )


def test_resume_skips_completed_work() -> None:
    section("Resume")
    units = load_units()[:2]
    records = synthetic_records(units, lambda prompt, tag, unit: 3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ratings.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        completed = ps.load_completed(path)
        check(
            len(completed) == len(PROMPT_IDS) * len(MODEL_TAGS) * len(units),
            "every stored cell counts as completed",
        )
        outstanding = list(
            ps.plan(units, ps.DEFAULT_MODELS, PROMPT_IDS, completed)
        )
        check(not outstanding, "a fully collected run has no outstanding work")

        outstanding = list(ps.plan(units, ps.DEFAULT_MODELS, PROMPT_IDS, set()))
        check(
            len(outstanding) == len(PROMPT_IDS) * len(ps.DEFAULT_MODELS) * len(units),
            "an empty file schedules every call",
        )
        # Model-major ordering, so each model is loaded exactly once.
        tags_in_order = [spec.tag for spec, _, _, _ in outstanding]
        first_seen = list(dict.fromkeys(tags_in_order))
        blocks = [tag for index, tag in enumerate(tags_in_order)
                  if index == 0 or tags_in_order[index - 1] != tag]
        check(blocks == first_seen, "the work plan visits each model in one contiguous block")

        # The stored single iteration counts against iteration 1 only, so asking
        # for more repeats schedules the remaining ones and nothing else.
        outstanding = list(ps.plan(units, ps.DEFAULT_MODELS, PROMPT_IDS, completed, 3))
        expected = len(PROMPT_IDS) * len(ps.DEFAULT_MODELS) * len(units) * 2
        check(len(outstanding) == expected, "raising the iteration count schedules only the new repeats")
        check(
            sorted({iteration for _, _, _, iteration in outstanding}) == [2, 3],
            "iteration 1 is not re-run",
        )
        # Within a model the iteration is the outer loop, so an interrupted run
        # leaves whole iterations across all prompts rather than one prompt only.
        first_model = [entry for entry in outstanding if entry[0].tag == outstanding[0][0].tag]
        check(
            [iteration for _, _, _, iteration in first_model]
            == sorted(iteration for _, _, _, iteration in first_model),
            "iterations run in order within a model",
        )
        check(
            {prompt for _, prompt, _, iteration in first_model if iteration == 2} == set(PROMPT_IDS),
            "one iteration covers every prompt before the next begins",
        )


def test_flowsort_score_injection() -> None:
    section("FlowSort score injection")
    baseline = qualify_partners_with_flowsort()
    baseline_classes = {
        entry["company"]: entry["assigned_category"] for entry in baseline["companies"]
    }
    check(bool(baseline_classes), "the baseline classification runs")

    units = load_units()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # Every qualitative cell at the top of the scale, then at the bottom.
        for score, name in ((5, "top"), (1, "bottom")):
            payload = {
                "scale": {"min": 1, "max": 5},
                "scores": [
                    {"company": unit.company, "code": unit.code, "score": score}
                    for unit in units
                ],
            }
            path = directory / f"{name}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

        top = qualify_partners_with_flowsort(rater_study_scores_path=directory / "top.json")
        bottom = qualify_partners_with_flowsort(
            rater_study_scores_path=directory / "bottom.json"
        )

    top_flows = {entry["company"]: entry["net_flow"] for entry in top["companies"]}
    bottom_flows = {entry["company"]: entry["net_flow"] for entry in bottom["companies"]}
    check(
        all(top_flows[company] >= bottom_flows[company] for company in top_flows),
        "scoring every qualitative cell at 5 never yields a worse net flow than at 1",
    )
    check(
        top_flows != bottom_flows,
        "the injected qualitative scores actually reach the classification",
    )

    # The default path must be untouched by the new parameter.
    again = qualify_partners_with_flowsort()
    check(
        {entry["company"]: entry["assigned_category"] for entry in again["companies"]}
        == baseline_classes,
        "omitting the parameter reproduces the default rater-study result",
    )


def test_comparison_tables() -> None:
    section("Comparison tables")
    units = load_units()

    # prompt_1 and prompt_2 score every cell 3; prompt_3 and prompt_4 score 5.
    def rating(prompt_id, tag, unit):
        return 3 if prompt_id in {"prompt_1", "prompt_2"} else 5

    records = synthetic_records(units, rating)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        path = directory / "ratings.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        result = ps.analyse(
            units=units,
            ratings_path=path,
            consensus_dir=directory / "consensus",
            csv_dir=directory / "csv",
            result_path=directory / "result.json",
        )

        scores = result["score_comparison"]
        check(
            scores["summary"]["n_units_identical_across_prompts"] == 0,
            "every unit is reported as moving when the prompts disagree by 2",
        )
        check(scores["summary"]["max_range"] == 2, "the score range is the 3-to-5 gap")

        classes = result["class_comparison"]
        check(
            classes["summary"]["n_companies"] == 6,
            "all six equipment suppliers are classified",
        )
        pairs = {(row["prompt_a"], row["prompt_b"]): row for row in classes["pairwise"]}
        check(
            pairs[("prompt_1", "prompt_2")]["agreement_pct"] == 100.0,
            "two prompts producing identical medians agree fully",
        )
        check(
            pairs[("prompt_3", "prompt_4")]["agreement_pct"] == 100.0,
            "the other identical pair agrees fully",
        )
        check(
            "prompt_2" in classes["summary"]["prompts_identical_to_baseline"],
            "a prompt matching the baseline classification is named",
        )

        for name in (
            "ratings_long", "ratings_by_model", "median_scores_by_prompt",
            "classification_by_prompt", "prompt_pair_agreement", "ranking_by_prompt",
            "prompt_summary", "run_summary",
        ):
            written = Path(result["csv_files"][name])
            check(
                written.exists() and written.stat().st_size > 0,
                f"{name}.csv is written and non-empty",
            )

        check(Path(result["result_path"]).exists(), "the full result JSON is written")
        check(
            ps.format_report(result).startswith("Prompt sensitivity"),
            "the terminal report renders",
        )


def main() -> int:
    for test in (
        test_prompt_bank,
        test_aggregation_is_median_across_models,
        test_incomplete_coverage_is_refused,
        test_provenance_guard,
        test_resume_skips_completed_work,
        test_flowsort_score_injection,
        test_comparison_tables,
    ):
        test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("All prompt-sensitivity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
