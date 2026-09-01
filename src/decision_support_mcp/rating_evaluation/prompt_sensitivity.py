"""Sensitivity of the FlowSort classification to the qualitative scoring prompt.

The question this answers is how much of a company's assigned class is decided by
the wording of the prompt that scores its qualitative disclosures, rather than by
the disclosures themselves.

Protocol, mirroring the baseline scoring pipeline at every step except the one
under test:

1. Each of the four prompts in :mod:`.prompt_sensitivity_prompts` is sent to each
   of the four study LLMs for every company-sub-criterion unit, at
   ``temperature=0``, repeated :data:`DEFAULT_ITERATIONS` times.
2. Within a prompt and iteration, the four model ratings are reduced to their
   median, then the medians across iterations, ties resolved downward at both
   stages, exactly as :func:`..report.build_consensus_scores` does.
3. Each prompt's medians are written as a consensus file and fed to FlowSort.

Everything else is held constant: the default GH-FBWM criterion weights, the
limiting profiles computed from ``corrected_all.json``, the ``t1`` preference
function, and every quantitative score. Any difference between the four
classifications is therefore attributable to the prompt.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.decision_support_mcp.flowsort import (
    GH_FBWM_STUDY_WEIGHTS,
    qualify_partners_with_flowsort,
)
from src.decision_support_mcp.rating_evaluation.agreement import median_lower
from src.decision_support_mcp.rating_evaluation.collect import (
    DEFAULT_BASE_URL,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_TIMEOUT_SEC,
    DEFAULT_TOP_P,
    DeadlineExceeded,
    ModelSpec,
    _extract_json_object,
    _generate,
    _parse_rating,
    unload_model,
    verify_models,
)
from src.decision_support_mcp.rating_evaluation.dataset import (
    DATA_DIR,
    LLM_RATINGS_PATH,
    Unit,
)
from src.decision_support_mcp.rating_evaluation.prompt_sensitivity_prompts import (
    PROMPT_FEATURES,
    PROMPT_IDS,
    PROMPT_LABELS,
    PROMPT_SHA256_BY_ID,
    PROMPT_TEMPLATES,
    RATING_LABELS,
    RUBRIC_SHA256,
    build_prompt,
)

OUTPUT_DIR = DATA_DIR / "prompt_sensitivity"
RATINGS_PATH = OUTPUT_DIR / "llm_ratings_by_prompt.jsonl"

#: JSON outputs live under ``json/``, mirroring ``csv/``. Every consumer imports
#: these names rather than rebuilding the path, so a layout change is one edit
#: here instead of one per script -- which is what let an earlier move break
#: three scripts at once.
JSON_DIR = OUTPUT_DIR / "json"
CONSENSUS_DIR = JSON_DIR / "consensus"
SCORES_BY_MODEL_DIR = JSON_DIR / "scores_by_model"
CSV_DIR = OUTPUT_DIR / "csv"
RESULT_PATH = JSON_DIR / "prompt_sensitivity_result.json"

#: The four study models. Same tags, and the same family names, as the
#: rater-agreement collection, so the two studies are directly comparable.
DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("Llama", "llama3.1:8b"),
    ModelSpec("Qwen", "qwen3.5:9b"),
    ModelSpec("DeepSeek", "deepseek-v2:16b"),
    ModelSpec("Gemma", "gemma4:12b"),
)

#: Temperature is the analysis's fixed point, not a parameter: the whole study
#: varies the prompt and nothing else, and greedy decoding removes sampling
#: noise from the comparison.
TEMPERATURE = 0.0

#: Repeats per (model, prompt, unit). The rater-agreement evaluation runs the
#: same number, so the two studies rest on samples of the same size. The seed is
#: the iteration number, matching that collection.
DEFAULT_ITERATIONS = 100


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def rating_key(record: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(record["model_tag"]),
        str(record["prompt_id"]),
        str(record["company"]),
        str(record["code"]),
        int(record.get("iteration", 1)),
    )


def load_ratings(path: Path = RATINGS_PATH) -> list[dict[str, Any]]:
    """Every stored rating record, successful or not."""

    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A partial trailing line from an interrupted run; the unit is
                # simply re-scored.
                continue
    return records


def load_completed(path: Path = RATINGS_PATH) -> set[tuple[str, str, str, str, int]]:
    return {
        rating_key(record)
        for record in load_ratings(path)
        if record.get("rating") is not None
    }


def check_provenance(path: Path = RATINGS_PATH) -> dict[str, Any]:
    """Reject a file whose prompts or rubric differ from the ones in code.

    Collection appends and resumes, so an edited prompt would interleave two
    incompatible scoring regimes in one file under the same ``prompt_id``.
    """

    known_prompts = set(PROMPT_SHA256_BY_ID.values())
    found_prompt: set[str] = set()
    found_rubric: set[str] = set()
    for record in load_ratings(path):
        if record.get("prompt_sha256"):
            found_prompt.add(str(record["prompt_sha256"]))
        if record.get("rubric_sha256"):
            found_rubric.add(str(record["rubric_sha256"]))
    stale_prompt = found_prompt - known_prompts
    stale_rubric = found_rubric - {RUBRIC_SHA256}
    return {
        "ok": not stale_prompt and not stale_rubric,
        "existing_records": bool(found_prompt or found_rubric),
        "stale_prompt_hashes": sorted(stale_prompt),
        "stale_rubric_hashes": sorted(stale_rubric),
        "current_prompt_sha256": dict(PROMPT_SHA256_BY_ID),
        "current_rubric_sha256": RUBRIC_SHA256,
    }


def plan(
    units: Sequence[Unit],
    models: Sequence[ModelSpec],
    prompt_ids: Sequence[str],
    completed: set[tuple[str, str, str, str, int]],
    iterations: int = 1,
) -> Iterator[tuple[ModelSpec, str, Unit, int]]:
    """Outstanding work, model-major so each model is loaded exactly once.

    A model swap costs a 280-966 s reload against 12-26 s for a warm call, so
    every other loop sits inside the model loop rather than the other way round.

    Within a model the iteration is the outer loop and the prompt the inner one,
    so an interrupted run leaves whole iterations covering all four prompts
    rather than all iterations of one prompt. Only the first shape is analysable:
    :func:`analyse` refuses a file where the prompts rest on different samples.
    """

    for spec in models:
        for iteration in range(1, iterations + 1):
            for prompt_id in prompt_ids:
                for unit in units:
                    key = (spec.tag, prompt_id, unit.company, unit.code, iteration)
                    if key in completed:
                        continue
                    yield spec, prompt_id, unit, iteration


def collect(
    units: Sequence[Unit],
    models: Sequence[ModelSpec] = DEFAULT_MODELS,
    prompt_ids: Sequence[str] = PROMPT_IDS,
    iterations: int = DEFAULT_ITERATIONS,
    out_path: Path = RATINGS_PATH,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    progress_every: int = 10,
    verify: bool = True,
    unload_between_models: bool = True,
    disable_thinking: bool = True,
) -> dict[str, Any]:
    """Run the outstanding scoring calls and append them to ``out_path``."""

    unknown = [pid for pid in prompt_ids if pid not in PROMPT_TEMPLATES]
    if unknown:
        raise ValueError(f"Unknown prompt ids: {unknown}. Expected from {list(PROMPT_IDS)}.")
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    if iterations > 1 and temperature <= 0.0:
        # Deliberate configuration for a determinism check. Greedy decoding
        # ignores the seed, so every iteration reproduces iteration 1 and the
        # per-iteration spread is zero by construction, up to whatever
        # nondeterminism the runtime's batching contributes.
        print(
            f"NOTE temperature={temperature}: Ollama decodes greedily and ignores the "
            f"seed, so all {iterations} iterations are expected to repeat iteration 1. "
            "Stability figures measure runtime nondeterminism, not sampling spread.",
            file=sys.stderr,
            flush=True,
        )

    digests: dict[str, str] = {}
    if verify:
        check = verify_models(models, base_url)
        if not check["ok"]:
            raise RuntimeError(
                "These model tags are not installed: "
                + ", ".join(check["missing"])
                + ". Pull them with `ollama pull <tag>` before collecting."
            )
        digests = check["resolved"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = check_provenance(out_path)
    if not provenance["ok"]:
        raise RuntimeError(
            f"{out_path} holds ratings collected under a prompt or rubric that is no "
            f"longer in code (prompt {provenance['stale_prompt_hashes']}, rubric "
            f"{provenance['stale_rubric_hashes']}). Ratings from different prompt "
            "texts are not comparable and resuming would interleave them. Move the "
            "file aside and start a fresh collection."
        )

    completed = load_completed(out_path)
    outstanding = list(plan(units, models, prompt_ids, completed, iterations))
    total = len(outstanding)

    counters = {
        "written": 0, "failed": 0, "retried": 0, "truncated": 0,
        "deadline_aborts": 0, "prompt_tokens": 0, "completion_tokens": 0,
    }
    started_at = time.time()
    active_tag: str | None = None
    unloaded: list[str] = []

    with out_path.open("a", encoding="utf-8") as handle:
        for index, (spec, prompt_id, unit, iteration) in enumerate(outstanding, start=1):
            if unload_between_models and active_tag is not None and spec.tag != active_tag:
                if unload_model(active_tag, base_url):
                    unloaded.append(active_tag)
                print(f"unloaded {active_tag}", file=sys.stderr, flush=True)
            active_tag = spec.tag

            prompt = build_prompt(prompt_id, unit.measure, unit.guidance, unit.disclosure)
            rating: int | None = None
            justification: str | None = None
            raw = ""
            latency = 0.0
            metrics: dict[str, Any] = {}
            failure: str | None = None
            attempts = 0

            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                # Retries perturb the seed: repeating an identical deterministic
                # request would reproduce the same unparseable output.
                seed = iteration + 100_000 * (attempt - 1)
                try:
                    raw, latency, metrics = _generate(
                        base_url, spec.tag, prompt, temperature, top_p,
                        seed, keep_alive, timeout_sec, disable_thinking,
                    )
                    rating, justification = _parse_rating(_extract_json_object(raw))
                    failure = None
                    break
                except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    failure = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, DeadlineExceeded):
                        counters["deadline_aborts"] += 1
                    if attempt < max_attempts:
                        counters["retried"] += 1
                        time.sleep(min(2.0 * attempt, 10.0))

            record = {
                "family": spec.family,
                "model_tag": spec.tag,
                "model_digest": digests.get(spec.tag),
                "prompt_id": prompt_id,
                "prompt_label": PROMPT_LABELS[prompt_id],
                "iteration": iteration,
                "seed": iteration,
                "temperature": temperature,
                "top_p": top_p,
                "format": "json",
                "think": (False if disable_thinking else None),
                "company": unit.company,
                "code": unit.code,
                "rating": rating,
                "justification": justification,
                "attempts": attempts,
                "failure": failure,
                "latency_sec": round(latency, 3),
                # Token counts, per-phase timings and done_reason, straight from
                # the last attempt's Ollama response.
                **metrics,
                "justification_chars": len(justification or ""),
                "raw_response": raw,
                "prompt_sha256": PROMPT_SHA256_BY_ID[prompt_id],
                "rubric_sha256": RUBRIC_SHA256,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            if rating is None:
                counters["failed"] += 1
            else:
                counters["written"] += 1
            counters["prompt_tokens"] += int(metrics.get("prompt_tokens") or 0)
            counters["completion_tokens"] += int(metrics.get("completion_tokens") or 0)
            if metrics.get("done_reason") == "length":
                counters["truncated"] += 1

            if progress_every and (index % progress_every == 0 or index == total):
                elapsed = time.time() - started_at
                remaining = (total - index) * elapsed / index
                print(
                    f"[{index}/{total}] {spec.tag} {prompt_id} it{iteration} "
                    f"{unit.company[:24]}/{unit.code} -> {rating} "
                    f"| {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m left",
                    file=sys.stderr,
                    flush=True,
                )

    if unload_between_models and active_tag is not None:
        if unload_model(active_tag, base_url):
            unloaded.append(active_tag)
        print(f"unloaded {active_tag}", file=sys.stderr, flush=True)

    return {
        "iterations": iterations,
        "requested_calls": total,
        "already_on_disk": len(completed),
        "written": counters["written"],
        "failed": counters["failed"],
        "retries": counters["retried"],
        # Attempts cut off at the deadline. High counts mean the stall mode is
        # active, not that the models are slow.
        "deadline_aborts": counters["deadline_aborts"],
        # A truncated answer parsed into a rating, but the model was cut off
        # mid-sentence, so the justification behind it is incomplete.
        "truncated_answers": counters["truncated"],
        "prompt_tokens": counters["prompt_tokens"],
        "completion_tokens": counters["completion_tokens"],
        "total_tokens": counters["prompt_tokens"] + counters["completion_tokens"],
        "unloaded_models": unloaded,
        "elapsed_sec": round(time.time() - started_at, 1),
        "output": str(out_path),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_consensus_by_prompt(
    units: Sequence[Unit],
    records: Sequence[dict[str, Any]],
    prompt_ids: Sequence[str] = PROMPT_IDS,
) -> dict[str, dict[str, Any]]:
    """One consensus-score payload per prompt, in the FlowSort handoff format.

    Aggregation matches the baseline pipeline: within an iteration take the
    median across models, then the median across iterations, ties resolved
    downward at both stages. A single-iteration file makes the second stage the
    identity, so files collected before and after the repeats aggregate alike.
    """

    # (prompt, tag, iteration) -> {(company, code): rating}
    table: dict[tuple[str, str, int], dict[tuple[str, str], int]] = defaultdict(dict)
    families: dict[str, str] = {}
    failures: Counter[str] = Counter()
    for record in records:
        prompt_id = str(record.get("prompt_id"))
        if prompt_id not in set(prompt_ids):
            continue
        if record.get("rating") is None:
            failures[prompt_id] += 1
            continue
        tag = str(record["model_tag"])
        families[tag] = str(record.get("family") or tag)
        key = (prompt_id, tag, int(record.get("iteration", 1)))
        table[key][(str(record["company"]), str(record["code"]))] = int(record["rating"])

    payloads: dict[str, dict[str, Any]] = {}
    for prompt_id in prompt_ids:
        tags = sorted({tag for pid, tag, _ in table if pid == prompt_id})
        iterations = sorted({it for pid, _, it in table if pid == prompt_id})

        scores: list[dict[str, Any]] = []
        lookup: dict[str, int] = {}
        for unit in units:
            per_iteration: list[int] = []
            for iteration in iterations:
                present = [
                    table.get((prompt_id, tag, iteration), {}).get(unit.key)
                    for tag in tags
                ]
                present = [value for value in present if value is not None]
                if present:
                    per_iteration.append(int(median_lower(present)))

            if not per_iteration:
                scores.append(
                    {
                        "company": unit.company,
                        "code": unit.code,
                        "score": None,
                        "n_iterations": 0,
                        "status": "no_ratings",
                    }
                )
                continue

            score = int(median_lower(per_iteration))
            counts = Counter(per_iteration)
            scores.append(
                {
                    "company": unit.company,
                    "code": unit.code,
                    "score": score,
                    "n_iterations": len(per_iteration),
                    "iteration_median_distribution": {
                        str(value): counts[value] for value in sorted(counts)
                    },
                    "stability": round(counts[score] / len(per_iteration), 4),
                    "status": "ok",
                }
            )
            lookup[f"{unit.company}||{unit.code}"] = score

        observed = sum(
            len(cells) for (pid, _, _), cells in table.items() if pid == prompt_id
        )
        expected = len(units) * len(tags) * len(iterations)
        payloads[prompt_id] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "purpose": (
                "qualitative sub-criterion scores for the FlowSort evaluation, "
                f"scored under {prompt_id}"
            ),
            "scale": {"min": int(RATING_LABELS[0]), "max": int(RATING_LABELS[-1])},
            "aggregation": (
                "per iteration: median across LLMs; then median across iterations; "
                "even-count ties resolved downward"
            ),
            "provenance": {
                "prompt_id": prompt_id,
                "prompt_label": PROMPT_LABELS[prompt_id],
                "prompt_features": PROMPT_FEATURES[prompt_id],
                "prompt_sha256": PROMPT_SHA256_BY_ID[prompt_id],
                "rubric_sha256": RUBRIC_SHA256,
                "rating_labels": list(RATING_LABELS),
                "temperature": TEMPERATURE,
                "families": {tag: families[tag] for tag in tags},
            },
            "coverage": {
                "n_units": len(units),
                "n_model_tags": len(tags),
                "n_iterations": len(iterations),
                "expected_llm_ratings": expected,
                "observed_llm_ratings": observed,
                "failed_llm_calls": failures[prompt_id],
                "complete": observed == expected and expected > 0,
            },
            "scores": scores,
            "scores_by_company_code": lookup,
        }
    return payloads


def write_consensus_files(
    payloads: dict[str, dict[str, Any]],
    directory: Path = CONSENSUS_DIR,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for prompt_id, payload in payloads.items():
        path = directory / f"qualitative_consensus_{prompt_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        paths[prompt_id] = path
    return paths


def export_baseline_ratings(
    ratings_path: Path = RATINGS_PATH,
    out_path: Path = LLM_RATINGS_PATH,
    prompt_id: str = "prompt_1",
    backup: bool = True,
) -> dict[str, Any]:
    """Adopt one prompt's ratings as the rater-agreement evaluation's input.

    ``prompt_1`` is :data:`..prompts.SCORING_PROMPT`, the prompt the
    rater-agreement evaluation scores with, so its records from this analysis are
    the same calls that evaluation would have made. Copying them across makes the
    two studies share one baseline instead of holding two collections of the same
    prompt, and costs no re-scoring.

    Any existing ``out_path`` collected under a different prompt is moved aside
    rather than overwritten: it is the evidence behind whatever was published
    from it, and the two are not comparable.
    """

    records = [
        record
        for record in load_ratings(ratings_path)
        if str(record.get("prompt_id")) == prompt_id
    ]
    if not records:
        raise ValueError(f"No {prompt_id} records in {ratings_path}.")

    failed = [record for record in records if record.get("rating") is None]
    if failed:
        raise ValueError(
            f"{len(failed)} {prompt_id} call(s) in {ratings_path} have no rating. "
            "Re-run the collection before adopting them as the baseline."
        )

    expected_sha = PROMPT_SHA256_BY_ID[prompt_id]
    wrong_sha = sorted(
        {
            str(record.get("prompt_sha256"))
            for record in records
            if record.get("prompt_sha256") != expected_sha
        }
    )
    if wrong_sha:
        raise ValueError(
            f"{prompt_id} records carry prompt hashes {wrong_sha}, but the prompt in "
            f"code hashes to {expected_sha}. The prompt was edited after collection; "
            "re-collect rather than adopt ratings from text no longer in use."
        )

    backup_path: Path | None = None
    if backup and out_path.exists():
        existing = {
            str(record.get("prompt_sha256"))
            for record in load_ratings(out_path)
            if record.get("prompt_sha256")
        }
        if existing - {expected_sha}:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            backup_path = out_path.with_suffix(f".{stamp}.jsonl.bak")
            out_path.rename(backup_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            # ``prompt_id`` is kept: it is how a reader of the baseline file
            # learns which analysis produced it. Every consumer selects the
            # fields it needs, so the extra key is inert.
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    tags = sorted({str(record["model_tag"]) for record in records})
    units = {(str(record["company"]), str(record["code"])) for record in records}
    return {
        "prompt_id": prompt_id,
        "prompt_sha256": expected_sha,
        "source": str(ratings_path),
        "output": str(out_path),
        "backed_up_to": str(backup_path) if backup_path else None,
        "n_records": len(records),
        "n_model_tags": len(tags),
        "model_tags": tags,
        "n_units": len(units),
    }


# ---------------------------------------------------------------------------
# FlowSort under each prompt
# ---------------------------------------------------------------------------


def classify_under_each_prompt(
    consensus_paths: dict[str, Path],
    companies: list[str] | None = None,
    prompt_ids: Sequence[str] = PROMPT_IDS,
    **flowsort_kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Run FlowSort once per prompt, changing only the qualitative scores.

    ``flowsort_kwargs`` is passed straight through, but the defaults are the
    study's: the GH-FBWM weights, limiting profiles, the net-flow rule and the
    ``t1`` usual criterion. Overriding them is possible but takes the run outside
    the analysis's stated design.
    """

    results: dict[str, dict[str, Any]] = {}
    for prompt_id in prompt_ids:
        path = consensus_paths.get(prompt_id)
        if path is None:
            raise ValueError(f"No consensus score file for {prompt_id}.")
        results[prompt_id] = qualify_partners_with_flowsort(
            companies=companies,
            qualitative_source="rater_study",
            rater_study_scores_path=path,
            **flowsort_kwargs,
        )
    return results


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _class_index(code: str) -> int:
    """C1 -> 1. Ordered best to worst, so a positive shift is a downgrade."""

    return int(str(code).lstrip("Cc"))


def compare_classifications(
    flowsort_by_prompt: dict[str, dict[str, Any]],
    prompt_ids: Sequence[str] = PROMPT_IDS,
    baseline: str = "prompt_1",
) -> dict[str, Any]:
    """Class movement across the prompts, per company and pairwise."""

    present = [pid for pid in prompt_ids if pid in flowsort_by_prompt]
    if not present:
        raise ValueError("No FlowSort results to compare.")
    if baseline not in present:
        baseline = present[0]

    companies = [entry["company"] for entry in flowsort_by_prompt[present[0]]["companies"]]
    by_prompt_company: dict[str, dict[str, dict[str, Any]]] = {
        prompt_id: {entry["company"]: entry for entry in result["companies"]}
        for prompt_id, result in flowsort_by_prompt.items()
    }

    company_rows: list[dict[str, Any]] = []
    for company in companies:
        classes = {
            prompt_id: by_prompt_company[prompt_id][company]["assigned_category"]
            for prompt_id in present
        }
        indices = [_class_index(code) for code in classes.values()]
        net_flows = {
            prompt_id: float(by_prompt_company[prompt_id][company]["net_flow"])
            for prompt_id in present
        }
        baseline_index = _class_index(classes[baseline])
        counts = Counter(classes.values())
        company_rows.append(
            {
                "company": company,
                "classes": classes,
                "net_flows": net_flows,
                "baseline_class": classes[baseline],
                "distinct_classes": len(counts),
                "modal_class": counts.most_common(1)[0][0],
                "class_span": max(indices) - min(indices),
                "max_shift_vs_baseline": max(
                    abs(_class_index(code) - baseline_index) for code in classes.values()
                ),
                "changed": len(counts) > 1,
                "net_flow_min": min(net_flows.values()),
                "net_flow_max": max(net_flows.values()),
                "net_flow_range": max(net_flows.values()) - min(net_flows.values()),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    for left, right in combinations(present, 2):
        agreements = 0
        shifts: list[int] = []
        movers: list[str] = []
        for company in companies:
            left_code = by_prompt_company[left][company]["assigned_category"]
            right_code = by_prompt_company[right][company]["assigned_category"]
            shift = _class_index(right_code) - _class_index(left_code)
            shifts.append(shift)
            if shift == 0:
                agreements += 1
            else:
                movers.append(f"{company}: {left_code}->{right_code}")
        pair_rows.append(
            {
                "prompt_a": left,
                "prompt_b": right,
                "n_companies": len(companies),
                "n_same_class": agreements,
                "agreement_pct": round(100.0 * agreements / len(companies), 2),
                "mean_abs_class_shift": round(
                    sum(abs(value) for value in shifts) / len(shifts), 4
                ),
                "max_abs_class_shift": max(abs(value) for value in shifts),
                "changes": "; ".join(movers),
            }
        )

    # Ranking stability: the net-flow order of the companies under each prompt.
    rankings = {
        prompt_id: [
            entry["company"]
            for entry in sorted(
                flowsort_by_prompt[prompt_id]["companies"],
                key=lambda item: -float(item["net_flow"]),
            )
        ]
        for prompt_id in present
    }
    baseline_ranking = rankings[baseline]
    ranking_rows = [
        {
            "prompt_id": prompt_id,
            "ranking": ranking,
            "same_as_baseline": ranking == baseline_ranking,
        }
        for prompt_id, ranking in rankings.items()
    ]

    n_changed = sum(1 for row in company_rows if row["changed"])
    identical_to_baseline = [
        prompt_id
        for prompt_id in present
        if prompt_id != baseline
        and all(
            by_prompt_company[prompt_id][company]["assigned_category"]
            == by_prompt_company[baseline][company]["assigned_category"]
            for company in companies
        )
    ]

    return {
        "baseline": baseline,
        "prompt_ids": present,
        "companies": company_rows,
        "pairwise": pair_rows,
        "rankings": ranking_rows,
        "summary": {
            "n_companies": len(companies),
            "n_companies_changing_class": n_changed,
            "pct_companies_changing_class": round(100.0 * n_changed / len(companies), 2),
            "pct_companies_stable": round(
                100.0 * (len(companies) - n_changed) / len(companies), 2
            ),
            "max_class_span": max(row["class_span"] for row in company_rows),
            # A single prompt has no pair to agree with. The comparison is
            # undefined rather than perfect, so it reports None: reporting 100%
            # would claim agreement that was never measured.
            "mean_pairwise_agreement_pct": (
                round(sum(row["agreement_pct"] for row in pair_rows) / len(pair_rows), 2)
                if pair_rows else None
            ),
            "min_pairwise_agreement_pct": (
                min(row["agreement_pct"] for row in pair_rows) if pair_rows else None
            ),
            "prompts_identical_to_baseline": identical_to_baseline,
            "n_rankings_matching_baseline": sum(
                1 for row in ranking_rows if row["same_as_baseline"]
            ),
        },
    }


def compare_scores(
    units: Sequence[Unit],
    consensus_by_prompt: dict[str, dict[str, Any]],
    prompt_ids: Sequence[str] = PROMPT_IDS,
) -> dict[str, Any]:
    """How far the median qualitative scores themselves move across prompts."""

    present = [pid for pid in prompt_ids if pid in consensus_by_prompt]
    rows: list[dict[str, Any]] = []
    for unit in units:
        key = f"{unit.company}||{unit.code}"
        scores = {
            prompt_id: consensus_by_prompt[prompt_id]["scores_by_company_code"].get(key)
            for prompt_id in present
        }
        values = [value for value in scores.values() if value is not None]
        rows.append(
            {
                "company": unit.company,
                "code": unit.code,
                "scores": scores,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "range": (max(values) - min(values)) if values else None,
                "distinct": len(set(values)) if values else 0,
            }
        )

    with_range = [row for row in rows if row["range"] is not None]
    identical = sum(1 for row in with_range if row["range"] == 0)
    return {
        "prompt_ids": present,
        "units": rows,
        "summary": {
            "n_units": len(with_range),
            "n_units_identical_across_prompts": identical,
            "pct_units_identical": round(100.0 * identical / len(with_range), 2)
            if with_range
            else None,
            "mean_range": round(
                sum(row["range"] for row in with_range) / len(with_range), 4
            )
            if with_range
            else None,
            "max_range": max((row["range"] for row in with_range), default=None),
        },
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_csv_files(
    records: Sequence[dict[str, Any]],
    score_comparison: dict[str, Any],
    class_comparison: dict[str, Any],
    consensus_by_prompt: dict[str, dict[str, Any]],
    flowsort_by_prompt: dict[str, dict[str, Any]],
    csv_dir: Path = CSV_DIR,
) -> dict[str, str]:
    """Write every table of the analysis. One file per table."""

    prompt_ids = list(class_comparison["prompt_ids"])
    written: dict[str, str] = {}

    # One row per call, so every one of the collected iterations survives the
    # medians that the rest of the tables take.
    written["ratings_long"] = str(
        _write_csv(
            csv_dir / "ratings_long.csv",
            [
                "prompt_id", "prompt_label", "family", "model_tag", "iteration", "seed",
                "company", "code", "rating", "temperature", "attempts", "failure",
                "latency_sec", "prompt_tokens", "completion_tokens", "total_tokens",
                "tokens_per_sec", "load_duration_sec", "prompt_eval_duration_sec",
                "eval_duration_sec", "total_duration_sec", "done_reason",
                "justification_chars", "justification", "prompt_sha256",
                "rubric_sha256", "recorded_at",
            ],
            [
                [
                    record.get("prompt_id"),
                    record.get("prompt_label"),
                    record.get("family"),
                    record.get("model_tag"),
                    record.get("iteration", 1),
                    record.get("seed"),
                    record.get("company"),
                    record.get("code"),
                    record.get("rating"),
                    record.get("temperature"),
                    record.get("attempts"),
                    record.get("failure"),
                    record.get("latency_sec"),
                    record.get("prompt_tokens"),
                    record.get("completion_tokens"),
                    record.get("total_tokens"),
                    record.get("tokens_per_sec"),
                    record.get("load_duration_sec"),
                    record.get("prompt_eval_duration_sec"),
                    record.get("eval_duration_sec"),
                    record.get("total_duration_sec"),
                    record.get("done_reason"),
                    record.get("justification_chars"),
                    # Newlines inside a justification would break the row for
                    # readers that do not honour quoted fields.
                    " ".join(str(record.get("justification") or "").split()),
                    record.get("prompt_sha256"),
                    record.get("rubric_sha256"),
                    record.get("recorded_at"),
                ]
                for record in records
            ],
        )
    )

    # Per-model ratings side by side with the median that FlowSort consumed. A
    # model that was run for several iterations is reduced to its own median
    # first, ties downward, so the cell is one number however many repeats
    # produced it.
    samples: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    tags = sorted({str(record["model_tag"]) for record in records})
    for record in records:
        if record.get("rating") is None:
            continue
        samples[
            (
                str(record["prompt_id"]),
                str(record["company"]),
                str(record["code"]),
                str(record["model_tag"]),
            )
        ].append(int(record["rating"]))

    by_cell: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for (prompt_id, company, code, tag), values in samples.items():
        by_cell[(prompt_id, company, code)][tag] = int(median_lower(values))
    rating_matrix_rows: list[list[Any]] = []
    for prompt_id in prompt_ids:
        lookup = consensus_by_prompt[prompt_id]["scores_by_company_code"]
        for (pid, company, code), per_model in sorted(by_cell.items()):
            if pid != prompt_id:
                continue
            values = [per_model[tag] for tag in tags if tag in per_model]
            rating_matrix_rows.append(
                [prompt_id, company, code]
                + [per_model.get(tag) for tag in tags]
                + [
                    lookup.get(f"{company}||{code}"),
                    (max(values) - min(values)) if values else None,
                    len(set(values)),
                ]
            )
    written["ratings_by_model"] = str(
        _write_csv(
            csv_dir / "ratings_by_model.csv",
            ["prompt_id", "company", "code"] + tags + ["median", "range", "n_distinct"],
            rating_matrix_rows,
        )
    )

    # How much a single model moves across its repeats of one cell. This is what
    # the iterations buy: everything downstream of the median hides it.
    stability_rows: list[list[Any]] = []
    for (prompt_id, company, code, tag), values in sorted(samples.items()):
        counts = Counter(values)
        modal, modal_count = counts.most_common(1)[0]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stability_rows.append(
            [
                prompt_id, tag, company, code, len(values),
                int(median_lower(values)), modal, round(modal_count / len(values), 4),
                len(counts), min(values), max(values),
                round(mean, 4), round(variance ** 0.5, 4),
                # The whole distribution, so a bimodal cell is not mistaken for
                # a noisy unimodal one.
                ";".join(f"{value}:{counts[value]}" for value in sorted(counts)),
            ]
        )
    written["iteration_stability_by_model"] = str(
        _write_csv(
            csv_dir / "iteration_stability_by_model.csv",
            [
                "prompt_id", "model_tag", "company", "code", "n_iterations",
                "median", "modal_rating", "modal_share", "n_distinct",
                "min", "max", "mean", "sd", "distribution",
            ],
            stability_rows,
        )
    )

    # What each model-prompt pair cost to run. A longer prompt is a more
    # expensive prompt, so this is the other half of the sensitivity question:
    # whether the guidance and rubric earn the tokens they add.
    def _summarise(values: list[float]) -> tuple[Any, Any, Any]:
        if not values:
            return None, None, None
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return round(sum(values) / len(values), 2), round(median, 2), round(max(values), 2)

    cost_rows: list[list[Any]] = []
    for prompt_id in prompt_ids:
        for tag in tags:
            subset = [
                record
                for record in records
                if record.get("prompt_id") == prompt_id and record.get("model_tag") == tag
            ]
            if not subset:
                continue

            def column(name: str) -> list[float]:
                return [
                    float(record[name])
                    for record in subset
                    if record.get(name) is not None
                ]

            prompt_tokens = column("prompt_tokens")
            completion_tokens = column("completion_tokens")
            latencies = column("latency_sec")
            mean_latency, median_latency, max_latency = _summarise(latencies)
            cost_rows.append(
                [
                    prompt_id, tag, len(subset),
                    sum(record.get("rating") is None for record in subset),
                    sum(record.get("done_reason") == "length" for record in subset),
                    round(sum(prompt_tokens) / len(prompt_tokens), 1) if prompt_tokens else None,
                    round(sum(completion_tokens) / len(completion_tokens), 1)
                    if completion_tokens else None,
                    int(sum(prompt_tokens) + sum(completion_tokens))
                    if (prompt_tokens or completion_tokens) else None,
                    mean_latency, median_latency, max_latency,
                    _summarise(column("tokens_per_sec"))[0],
                    round(sum(latencies) / 3600, 3) if latencies else None,
                ]
            )
    written["cost_by_model_prompt"] = str(
        _write_csv(
            csv_dir / "cost_by_model_prompt.csv",
            [
                "prompt_id", "model_tag", "n_calls", "n_failed", "n_truncated",
                "mean_prompt_tokens", "mean_completion_tokens", "total_tokens",
                "mean_latency_sec", "median_latency_sec", "max_latency_sec",
                "mean_tokens_per_sec", "wall_clock_hours",
            ],
            cost_rows,
        )
    )

    written["median_scores_by_prompt"] = str(
        _write_csv(
            csv_dir / "median_scores_by_prompt.csv",
            ["company", "code"] + prompt_ids + ["min", "max", "range", "n_distinct"],
            [
                [row["company"], row["code"]]
                + [row["scores"].get(pid) for pid in prompt_ids]
                + [row["min"], row["max"], row["range"], row["distinct"]]
                for row in score_comparison["units"]
            ],
        )
    )

    written["classification_by_prompt"] = str(
        _write_csv(
            csv_dir / "classification_by_prompt.csv",
            ["company"]
            + [f"class_{pid}" for pid in prompt_ids]
            + [f"net_flow_{pid}" for pid in prompt_ids]
            + [
                "baseline_class", "modal_class", "n_distinct_classes", "class_span",
                "max_shift_vs_baseline", "changed", "net_flow_range",
            ],
            [
                [row["company"]]
                + [row["classes"].get(pid) for pid in prompt_ids]
                + [round(row["net_flows"].get(pid, float("nan")), 6) for pid in prompt_ids]
                + [
                    row["baseline_class"],
                    row["modal_class"],
                    row["distinct_classes"],
                    row["class_span"],
                    row["max_shift_vs_baseline"],
                    row["changed"],
                    round(row["net_flow_range"], 6),
                ]
                for row in class_comparison["companies"]
            ],
        )
    )

    written["prompt_pair_agreement"] = str(
        _write_csv(
            csv_dir / "prompt_pair_agreement.csv",
            [
                "prompt_a", "prompt_b", "n_companies", "n_same_class", "agreement_pct",
                "mean_abs_class_shift", "max_abs_class_shift", "changes",
            ],
            [
                [
                    row["prompt_a"], row["prompt_b"], row["n_companies"],
                    row["n_same_class"], row["agreement_pct"],
                    row["mean_abs_class_shift"], row["max_abs_class_shift"],
                    row["changes"],
                ]
                for row in class_comparison["pairwise"]
            ],
        )
    )

    written["ranking_by_prompt"] = str(
        _write_csv(
            csv_dir / "ranking_by_prompt.csv",
            ["prompt_id", "rank", "company", "net_flow", "assigned_class"],
            [
                [
                    row["prompt_id"],
                    rank,
                    company,
                    round(
                        float(
                            next(
                                entry["net_flow"]
                                for entry in flowsort_by_prompt[row["prompt_id"]]["companies"]
                                if entry["company"] == company
                            )
                        ),
                        6,
                    ),
                    next(
                        entry["assigned_category"]
                        for entry in flowsort_by_prompt[row["prompt_id"]]["companies"]
                        if entry["company"] == company
                    ),
                ]
                for row in class_comparison["rankings"]
                for rank, company in enumerate(row["ranking"], start=1)
            ],
        )
    )

    prompt_summary_rows = []
    for prompt_id in prompt_ids:
        coverage = consensus_by_prompt[prompt_id]["coverage"]
        features = PROMPT_FEATURES[prompt_id]
        classes = [
            row["classes"][prompt_id] for row in class_comparison["companies"]
        ]
        distribution = Counter(classes)
        scores = [
            row["scores"][prompt_id]
            for row in score_comparison["units"]
            if row["scores"].get(prompt_id) is not None
        ]
        prompt_summary_rows.append(
            [
                prompt_id,
                PROMPT_LABELS[prompt_id],
                features["sasb_guidance"],
                features["scoring_rubric"],
                coverage["observed_llm_ratings"],
                coverage["failed_llm_calls"],
                round(sum(scores) / len(scores), 4) if scores else None,
                distribution.get("C1", 0),
                distribution.get("C2", 0),
                distribution.get("C3", 0),
                PROMPT_SHA256_BY_ID[prompt_id],
            ]
        )
    written["prompt_summary"] = str(
        _write_csv(
            csv_dir / "prompt_summary.csv",
            [
                "prompt_id", "prompt_label", "has_sasb_guidance", "has_scoring_rubric",
                "n_llm_ratings", "n_failed_calls", "mean_median_score",
                "n_C1", "n_C2", "n_C3", "prompt_sha256",
            ],
            prompt_summary_rows,
        )
    )

    class_summary = class_comparison["summary"]
    score_summary = score_comparison["summary"]
    written["run_summary"] = str(
        _write_csv(
            csv_dir / "run_summary.csv",
            ["metric", "value"],
            [
                ["generated_at", datetime.now(timezone.utc).isoformat()],
                ["prompts", ", ".join(prompt_ids)],
                ["baseline_prompt", class_comparison["baseline"]],
                ["models", ", ".join(tags)],
                ["temperature", TEMPERATURE],
                [
                    "iterations_per_model_prompt_unit",
                    max(
                        (
                            consensus_by_prompt[pid]["coverage"]["n_iterations"]
                            for pid in prompt_ids
                        ),
                        default=0,
                    ),
                ],
                ["n_companies", class_summary["n_companies"]],
                ["n_qualitative_units", score_summary["n_units"]],
                ["n_companies_changing_class", class_summary["n_companies_changing_class"]],
                ["pct_companies_changing_class", class_summary["pct_companies_changing_class"]],
                ["pct_companies_stable", class_summary["pct_companies_stable"]],
                ["max_class_span", class_summary["max_class_span"]],
                ["mean_pairwise_agreement_pct", class_summary["mean_pairwise_agreement_pct"]],
                ["min_pairwise_agreement_pct", class_summary["min_pairwise_agreement_pct"]],
                ["n_rankings_matching_baseline", class_summary["n_rankings_matching_baseline"]],
                ["pct_units_identical_across_prompts", score_summary["pct_units_identical"]],
                ["mean_median_score_range", score_summary["mean_range"]],
                ["max_median_score_range", score_summary["max_range"]],
            ],
        )
    )
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyse(
    units: Sequence[Unit],
    ratings_path: Path = RATINGS_PATH,
    prompt_ids: Sequence[str] = PROMPT_IDS,
    baseline: str = "prompt_1",
    companies: list[str] | None = None,
    consensus_dir: Path = CONSENSUS_DIR,
    csv_dir: Path = CSV_DIR,
    result_path: Path | None = RESULT_PATH,
    write_csv: bool = True,
) -> dict[str, Any]:
    """Aggregate the collected ratings, run FlowSort per prompt, and compare."""

    records = load_ratings(ratings_path)
    if not records:
        raise ValueError(f"No ratings at {ratings_path}. Run the collection first.")

    consensus = build_consensus_by_prompt(units, records, prompt_ids)
    incomplete = [
        prompt_id
        for prompt_id, payload in consensus.items()
        if not payload["coverage"]["complete"]
    ]
    if incomplete:
        raise ValueError(
            "These prompts have missing ratings and would be compared against a "
            f"different sample than the others: {incomplete}. Re-run the collection "
            "so every prompt covers every model and unit."
        )

    consensus_paths = write_consensus_files(consensus, consensus_dir)
    flowsort_by_prompt = classify_under_each_prompt(
        consensus_paths, companies=companies, prompt_ids=prompt_ids
    )
    class_comparison = compare_classifications(flowsort_by_prompt, prompt_ids, baseline)
    score_comparison = compare_scores(units, consensus, prompt_ids)

    csv_files: dict[str, str] = {}
    if write_csv:
        csv_files = write_csv_files(
            records=records,
            score_comparison=score_comparison,
            class_comparison=class_comparison,
            consensus_by_prompt=consensus,
            flowsort_by_prompt=flowsort_by_prompt,
            csv_dir=csv_dir,
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "varied": "qualitative scoring prompt",
            "held_constant": [
                "criterion weights (GH-FBWM defaults)",
                "limiting profiles",
                "quantitative scores",
                "temperature 0.0",
                "preference function t1, limiting mode, net rule",
                "the four LLMs and the median aggregation",
            ],
            "prompts": {
                prompt_id: {
                    "label": PROMPT_LABELS[prompt_id],
                    "features": PROMPT_FEATURES[prompt_id],
                    "sha256": PROMPT_SHA256_BY_ID[prompt_id],
                }
                for prompt_id in prompt_ids
            },
            "criterion_weights": dict(GH_FBWM_STUDY_WEIGHTS),
            "temperature": TEMPERATURE,
        },
        "coverage": {
            prompt_id: consensus[prompt_id]["coverage"] for prompt_id in prompt_ids
        },
        "score_comparison": score_comparison,
        "class_comparison": class_comparison,
        "consensus_files": {pid: str(path) for pid, path in consensus_paths.items()},
        "csv_files": csv_files,
        "flowsort_by_prompt": {
            prompt_id: {
                "companies": [
                    {
                        "company": entry["company"],
                        "assigned_category": entry["assigned_category"],
                        "assigned_label": entry["assigned_label"],
                        "positive_flow": entry["positive_flow"],
                        "negative_flow": entry["negative_flow"],
                        "net_flow": entry["net_flow"],
                    }
                    for entry in result_payload["companies"]
                ]
            }
            for prompt_id, result_payload in flowsort_by_prompt.items()
        },
    }

    if result_path is not None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        result["result_path"] = str(result_path)
    return result


def format_report(result: dict[str, Any]) -> str:
    """Terminal rendering of the analysis."""

    prompt_ids = list(result["class_comparison"]["prompt_ids"])
    lines: list[str] = []

    lines.append("Prompt sensitivity of the FlowSort classification")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Prompts compared")
    lines.append(f"  {'prompt':<10} {'guidance':<10} {'rubric':<8} label")
    for prompt_id in prompt_ids:
        features = PROMPT_FEATURES[prompt_id]
        lines.append(
            f"  {prompt_id:<10} {str(features['sasb_guidance']):<10} "
            f"{str(features['scoring_rubric']):<8} {PROMPT_LABELS[prompt_id]}"
        )
    lines.append("")

    lines.append("Assigned class by prompt")
    header = f"  {'company':<48}" + "".join(f"{pid.replace('prompt_', 'P'):>6}" for pid in prompt_ids)
    lines.append(header + "   span")
    for row in result["class_comparison"]["companies"]:
        cells = "".join(f"{row['classes'][pid]:>6}" for pid in prompt_ids)
        flag = "  <-- moves" if row["changed"] else ""
        lines.append(f"  {row['company'][:48]:<48}{cells}   {row['class_span']}{flag}")
    lines.append("")

    lines.append("Net flow by prompt")
    lines.append(f"  {'company':<48}" + "".join(f"{pid.replace('prompt_', 'P'):>10}" for pid in prompt_ids) + f"{'range':>10}")
    for row in result["class_comparison"]["companies"]:
        cells = "".join(f"{row['net_flows'][pid]:>10.4f}" for pid in prompt_ids)
        lines.append(f"  {row['company'][:48]:<48}{cells}{row['net_flow_range']:>10.4f}")
    lines.append("")

    lines.append("Pairwise agreement")
    lines.append(f"  {'pair':<24}{'same':>6}{'agree %':>10}{'mean |shift|':>14}  changes")
    for row in result["class_comparison"]["pairwise"]:
        pair = f"{row['prompt_a']} vs {row['prompt_b']}"
        lines.append(
            f"  {pair:<24}{row['n_same_class']:>6}{row['agreement_pct']:>10.2f}"
            f"{row['mean_abs_class_shift']:>14.4f}  {row['changes'] or '-'}"
        )
    lines.append("")

    score_summary = result["score_comparison"]["summary"]
    class_summary = result["class_comparison"]["summary"]
    lines.append("Summary")
    lines.append(
        f"  qualitative units with an identical median under all prompts   "
        f"{score_summary['n_units_identical_across_prompts']}/{score_summary['n_units']} "
        f"({score_summary['pct_units_identical']}%)"
    )
    lines.append(f"  mean range of the median score across prompts                 {score_summary['mean_range']}")
    lines.append(f"  max range of the median score across prompts                  {score_summary['max_range']}")
    lines.append(
        f"  companies keeping the same class under all prompts            "
        f"{class_summary['n_companies'] - class_summary['n_companies_changing_class']}"
        f"/{class_summary['n_companies']} ({class_summary['pct_companies_stable']}%)"
    )
    lines.append(f"  mean pairwise class agreement                                 {class_summary['mean_pairwise_agreement_pct']}%")
    lines.append(f"  worst pairwise class agreement                                {class_summary['min_pairwise_agreement_pct']}%")
    baseline = result["class_comparison"]["baseline"]
    lines.append(
        f"  net-flow rankings identical to {baseline:<15}"
        f"               {class_summary['n_rankings_matching_baseline']}/{len(prompt_ids)}"
    )
    return "\n".join(lines)
