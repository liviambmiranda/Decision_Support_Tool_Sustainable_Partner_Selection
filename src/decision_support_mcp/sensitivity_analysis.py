"""Sensitivity analysis for qualitative scoring and FlowSort outcomes."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as t_distribution

from src.decision_support_mcp.flowsort import (
    CATEGORY_LABELS,
    _flowsort_details,
    _limiting_category_assignments,
    prepare_flowsort_inputs,
    qualify_partners_with_flowsort,
    single_criterion_preference_matrices,
)
from src.decision_support_mcp.qualitative_scores import (
    DEFAULT_SENSITIVITY_MODEL_FAMILIES,
    PROMPT_VARIANT_LABELS,
    QUALITATIVE_PROMPT_VARIANTS,
)
from src.decision_support_mcp.quantitative_scores import (
    CORRECTED_ALL_PATH,
    DATA_DIR,
    MEASURES_HITL_PATH,
    _load_measures_hitl,
    _metadata_by_code,
    _round_float,
    round_significant,
)

def _normalize_prompt_variants(prompt_variants: list[str] | None) -> list[str]:
    variants = prompt_variants[:] if prompt_variants else list(QUALITATIVE_PROMPT_VARIANTS)
    normalized = []
    for variant in variants:
        key = str(variant).strip().lower()
        if key not in QUALITATIVE_PROMPT_VARIANTS:
            valid = ", ".join(sorted(QUALITATIVE_PROMPT_VARIANTS))
            raise ValueError(f"Unknown prompt variant {variant!r}. Expected one of: {valid}.")
        if key not in normalized:
            normalized.append(key)
    return normalized


def _normalize_model_families(model_families: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    families = model_families[:] if model_families else DEFAULT_SENSITIVITY_MODEL_FAMILIES
    normalized = []
    for index, family in enumerate(families, start=1):
        if not family:
            continue
        enabled = bool(family.get("enabled", True))
        if not enabled:
            continue
        provider = str(family.get("provider", "ollama")).strip().lower()
        model = str(family.get("model") or "").strip()
        name = str(family.get("family") or f"Model Family {index}").strip()
        if not model:
            raise ValueError(f"Enabled model family {name!r} is missing a model.")
        normalized.append({"family": name, "provider": provider, "model": model, "enabled": True})
    if not normalized:
        raise ValueError("Enable at least one model family for sensitivity analysis.")
    return normalized


def _normalize_float_list(values: list[float] | None, default: list[float], label: str) -> list[float]:
    raw_values = values[:] if values else default[:]
    normalized = []
    for value in raw_values:
        normalized.append(float(value))
    if not normalized:
        raise ValueError(f"{label} cannot be empty.")
    return normalized


def _class_distribution(company_results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"C1": 0, "C2": 0, "C3": 0}
    for item in company_results:
        code = str(item["assigned_category"])
        counts[code] = counts.get(code, 0) + 1
    return counts


def _company_metric_map(company_results: list[dict[str, Any]], metric: str) -> dict[str, float | str]:
    return {str(item["company"]): item[metric] for item in company_results}


def _safe_spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b):
        raise ValueError("Spearman inputs must have the same length.")
    if len(values_a) < 2:
        return 1.0
    coef = spearmanr(values_a, values_b).statistic
    if coef is None or math.isnan(coef):
        return 1.0 if values_a == values_b else 0.0
    return float(coef)


def _class_change_pct(
    baseline_categories: dict[str, str],
    current_categories: dict[str, str],
    ordered_companies: list[str],
) -> float:
    changed = sum(1 for company in ordered_companies if baseline_categories[company] != current_categories[company])
    return (changed / len(ordered_companies)) * 100.0 if ordered_companies else 0.0


def _category_index(category_code: str) -> int:
    text = str(category_code).strip().upper()
    if not text.startswith("C"):
        raise ValueError(f"Unexpected category code: {category_code!r}")
    return int(text[1:])


def _exact_class_match_pct(
    baseline_categories: dict[str, str],
    current_categories: dict[str, str],
    ordered_companies: list[str],
) -> float:
    if not ordered_companies:
        return 100.0
    matches = sum(1 for company in ordered_companies if baseline_categories[company] == current_categories[company])
    return (matches / len(ordered_companies)) * 100.0


def _mean_absolute_class_shift(
    baseline_categories: dict[str, str],
    current_categories: dict[str, str],
    ordered_companies: list[str],
) -> float:
    if not ordered_companies:
        return 0.0
    shifts = [
        abs(_category_index(baseline_categories[company]) - _category_index(current_categories[company]))
        for company in ordered_companies
    ]
    return float(np.mean(shifts))


def _build_promptfoo_provider_id(provider: str, model: str) -> str:
    provider_name = provider.strip().lower()
    if provider_name == "ollama":
        return f"ollama:chat:{model}"
    if provider_name == "openai":
        return f"openai:{model}"
    if provider_name == "gemini":
        return f"google:{model}"
    raise ValueError(f"Unsupported promptfoo provider {provider!r}.")


def build_promptfoo_sensitivity_config(
    sector_id: str = "semiconductors",
    model_families: list[dict[str, Any]] | None = None,
    prompt_variants: list[str] | None = None,
    temperatures: list[float] | None = None,
    top_ps: list[float] | None = None,
    seed: int = 42,
    repetitions: int = 2,
) -> dict[str, Any]:
    rows = _load_measures_hitl()
    metadata_by_code = _metadata_by_code(sector_id)
    selected_families = _normalize_model_families(model_families)
    selected_prompts = _normalize_prompt_variants(prompt_variants)
    selected_temperatures = _normalize_float_list(temperatures, [0.0, 0.5, 1.0], "temperatures")
    selected_top_ps = _normalize_float_list(top_ps, [1.0, 0.9], "top_ps")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be a positive integer.")

    qualitative_rows = [
        row
        for row in rows
        if str(metadata_by_code.get(str(row["code"]), {}).get("category", "")).lower() == "qualitative"
    ]

    promptfoo_providers = []
    for family in selected_families:
        for temperature, top_p in product(selected_temperatures, selected_top_ps):
            provider_id = _build_promptfoo_provider_id(family["provider"], family["model"])
            promptfoo_providers.append(
                {
                    "id": provider_id,
                    "label": f"{family['family']} | temp={temperature} | top_p={top_p}",
                    "config": {
                        "temperature": temperature,
                        "top_p": top_p,
                        "seed": int(seed),
                    },
                }
            )

    promptfoo_tests = []
    for repetition in range(1, repetitions + 1):
        for row in qualitative_rows:
            meta = metadata_by_code[str(row["code"])]
            promptfoo_tests.append(
                {
                    "vars": {
                        "company": row["company"],
                        "code": row["code"],
                        "measure": meta["measure"],
                        "guidance": meta.get("explanation", ""),
                        "disclosure": row["value"],
                        "repetition": repetition,
                    },
                    "assert": [{"type": "contains-json"}],
                }
            )

    promptfoo_prompts = []
    for prompt_id in selected_prompts:
        template = QUALITATIVE_PROMPT_VARIANTS[prompt_id]
        promptfoo_prompts.append(
            {
                "label": PROMPT_VARIANT_LABELS[prompt_id],
                "raw": template.replace("{company}", "{{ company }}")
                .replace("{measure}", "{{ measure }}")
                .replace("{guidance}", "{{ guidance }}")
                .replace("{value}", "{{ disclosure }}"),
            }
        )

    config = {
        "description": "Sensitivity analysis for qualitative ESG ratings on measures_HITL.json",
        "prompts": [item["raw"] for item in promptfoo_prompts],
        "providers": promptfoo_providers,
        "tests": promptfoo_tests,
        "defaultTest": {
            "metadata": {
                "sector_id": sector_id,
                "input_dataset": str(MEASURES_HITL_PATH.name),
                "profile_dataset": str(CORRECTED_ALL_PATH.name),
                "seed": int(seed),
            }
        },
    }
    return {
        "sector_id": sector_id,
        "prompt_variants": promptfoo_prompts,
        "provider_count": len(promptfoo_providers),
        "test_count": len(promptfoo_tests),
        "config": config,
        "config_json": json.dumps(config, ensure_ascii=False, indent=2),
        "notes": [
            "Promptfoo will evaluate only the qualitative measures from measures_HITL.json.",
            "FlowSort itself remains outside promptfoo; use the promptfoo outputs as the scoring layer and this app as the decision-analysis layer.",
            "This workspace currently prepares a promptfoo-ready config in-memory without creating files.",
        ],
    }


def run_partner_sensitivity_analysis(
    sector_id: str = "semiconductors",
    criteria_formulation_result: dict[str, Any] | None = None,
    measure_weights: dict[str, float] | None = None,
    companies: list[str] | None = None,
    model_families: list[dict[str, Any]] | None = None,
    prompt_variants: list[str] | None = None,
    temperatures: list[float] | None = None,
    top_ps: list[float] | None = None,
    seed: int = 42,
    repetitions: int = 2,
    mode: str = "limiting",
    rule: str = "net",
    preference_function: str = "t1",
    indifference_threshold: float | int | dict[str, Any] | None = 0.0,
    preference_threshold: float | int | dict[str, Any] | None = 0.0,
) -> dict[str, Any]:
    selected_families = _normalize_model_families(model_families)
    selected_prompts = _normalize_prompt_variants(prompt_variants)
    selected_temperatures = _normalize_float_list(temperatures, [0.0, 0.5, 1.0], "temperatures")
    selected_top_ps = _normalize_float_list(top_ps, [1.0, 0.9], "top_ps")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be a positive integer.")

    runs = []
    baseline_signature = (selected_prompts[0], selected_temperatures[0], selected_top_ps[0], 1)

    for family in selected_families:
        family_runs = []
        baseline_result = None
        for prompt_variant, temperature, top_p in product(
            selected_prompts,
            selected_temperatures,
            selected_top_ps,
        ):
            for repetition in range(1, repetitions + 1):
                llm_config = {
                    "provider": family["provider"],
                    "model": family["model"],
                    "prompt_variant": prompt_variant,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": int(seed),
                }
                qualification = qualify_partners_with_flowsort(
                    sector_id=sector_id,
                    criteria_formulation_result=criteria_formulation_result,
                    measure_weights=measure_weights,
                    companies=companies,
                    llm_config=llm_config,
                    mode=mode,
                    rule=rule,
                    preference_function=preference_function,
                    indifference_threshold=indifference_threshold,
                    preference_threshold=preference_threshold,
                )
                signature = (prompt_variant, temperature, top_p, repetition)
                run_record = {
                    "family": family["family"],
                    "provider": family["provider"],
                    "model": family["model"],
                    "prompt_variant": prompt_variant,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": int(seed),
                    "repetition": repetition,
                    "companies": qualification["companies"],
                    "class_distribution": _class_distribution(qualification["companies"]),
                }
                family_runs.append(run_record)
                if signature == baseline_signature:
                    baseline_result = run_record

        if baseline_result is None:
            raise RuntimeError(f"Baseline run was not produced for {family['family']}.")

        ordered_companies = [item["company"] for item in baseline_result["companies"]]
        baseline_flows = _company_metric_map(baseline_result["companies"], "net_flow")
        baseline_categories = _company_metric_map(baseline_result["companies"], "assigned_category")

        for run_record in family_runs:
            current_flows = _company_metric_map(run_record["companies"], "net_flow")
            current_categories = _company_metric_map(run_record["companies"], "assigned_category")
            flow_a = [float(baseline_flows[company]) for company in ordered_companies]
            flow_b = [float(current_flows[company]) for company in ordered_companies]
            exact_match = _exact_class_match_pct(baseline_categories, current_categories, ordered_companies)
            mean_abs_shift = _mean_absolute_class_shift(baseline_categories, current_categories, ordered_companies)
            run_record["spearman_vs_baseline"] = _round_float(_safe_spearman(flow_a, flow_b))
            run_record["exact_class_match_pct_vs_baseline"] = _round_float(exact_match)
            run_record["mean_absolute_class_shift_vs_baseline"] = _round_float(mean_abs_shift)
            # Backward-compatible alias used by existing UI/automation output.
            run_record["class_change_pct_vs_baseline"] = _round_float(100.0 - exact_match)
            run_record["baseline_signature"] = {
                "prompt_variant": baseline_signature[0],
                "temperature": baseline_signature[1],
                "top_p": baseline_signature[2],
                "repetition": baseline_signature[3],
            }

        combination_summary = []
        grouped_by_combo: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
        for run_record in family_runs:
            grouped_by_combo[
                (run_record["prompt_variant"], run_record["temperature"], run_record["top_p"])
            ].append(run_record)

        for (prompt_variant, temperature, top_p), combo_runs in grouped_by_combo.items():
            combo_runs = sorted(combo_runs, key=lambda item: item["repetition"])
            reference_categories = _company_metric_map(combo_runs[0]["companies"], "assigned_category")
            reproducibility_exact_match_values = []
            reproducibility_mean_abs_shift_values = []
            reproducibility_spearman = []
            reference_flows = _company_metric_map(combo_runs[0]["companies"], "net_flow")
            for candidate in combo_runs[1:]:
                candidate_categories = _company_metric_map(candidate["companies"], "assigned_category")
                candidate_flows = _company_metric_map(candidate["companies"], "net_flow")
                reproducibility_exact_match_values.append(
                    _exact_class_match_pct(reference_categories, candidate_categories, ordered_companies)
                )
                reproducibility_mean_abs_shift_values.append(
                    _mean_absolute_class_shift(reference_categories, candidate_categories, ordered_companies)
                )
                reproducibility_spearman.append(
                    _safe_spearman(
                        [float(reference_flows[company]) for company in ordered_companies],
                        [float(candidate_flows[company]) for company in ordered_companies],
                    )
                )
            mean_spearman = float(np.mean([run["spearman_vs_baseline"] for run in combo_runs]))
            mean_exact_match = float(np.mean([run["exact_class_match_pct_vs_baseline"] for run in combo_runs]))
            mean_abs_shift = float(np.mean([run["mean_absolute_class_shift_vs_baseline"] for run in combo_runs]))
            summary_record = {
                "prompt_variant": prompt_variant,
                "temperature": temperature,
                "top_p": top_p,
                "runs": len(combo_runs),
                "mean_spearman_vs_baseline": _round_float(mean_spearman),
                "std_spearman_vs_baseline": _round_float(float(np.std([run["spearman_vs_baseline"] for run in combo_runs]))),
                "exact_class_match_pct_vs_baseline": _round_float(mean_exact_match),
                "mean_absolute_class_shift_vs_baseline": _round_float(mean_abs_shift),
                # Backward-compatible alias used by existing UI/automation output.
                "mean_class_change_pct_vs_baseline": _round_float(100.0 - mean_exact_match),
                "reproducibility_exact_class_pct": _round_float(
                    float(np.mean(reproducibility_exact_match_values))
                    if reproducibility_exact_match_values
                    else 100.0
                ),
                "reproducibility_mean_absolute_class_shift": _round_float(
                    float(np.mean(reproducibility_mean_abs_shift_values))
                    if reproducibility_mean_abs_shift_values
                    else 0.0
                ),
                "reproducibility_spearman": _round_float(
                    float(np.mean(reproducibility_spearman)) if reproducibility_spearman else 1.0
                ),
                "class_distribution_first_run": combo_runs[0]["class_distribution"],
            }
            combination_summary.append(summary_record)

        combination_summary.sort(
            key=lambda item: (
                -(item["mean_spearman_vs_baseline"] or -1.0),
                -(item["exact_class_match_pct_vs_baseline"] or -1.0),
                item["mean_absolute_class_shift_vs_baseline"] or 999.0,
                -(item["reproducibility_exact_class_pct"] or -1.0),
            )
        )

        runs.append(
            {
                "family": family["family"],
                "provider": family["provider"],
                "model": family["model"],
                "baseline": {
                    "prompt_variant": baseline_signature[0],
                    "temperature": baseline_signature[1],
                    "top_p": baseline_signature[2],
                    "repetition": baseline_signature[3],
                    "class_distribution": baseline_result["class_distribution"],
                },
                "best_configuration": combination_summary[0] if combination_summary else None,
                "combination_summary": combination_summary,
                "detailed_runs": family_runs,
            }
        )

    return {
        "sector_id": sector_id,
        "input_dataset": str(MEASURES_HITL_PATH.name),
        "profile_dataset": str(CORRECTED_ALL_PATH.name),
        "prompt_variants": selected_prompts,
        "temperatures": selected_temperatures,
        "top_ps": selected_top_ps,
        "seed": int(seed),
        "repetitions": repetitions,
        "mode": mode,
        "rule": rule,
        "model_families": runs,
        "analysis_notes": [
            "Spearman is computed against the baseline run of each family: prompt_1, the first listed temperature, the first listed top_p, repetition 1.",
            "Exact class match is the percentage of companies that remained in the same final FlowSort class relative to baseline.",
            "Mean absolute class shift is the average number of classes moved per company relative to baseline.",
            "Reproducibility compares repeated runs within the same prompt/temperature/top_p combination.",
        ],
    }


# ---------------------------------------------------------------------------
# Monte Carlo simulation over the criterion weights
# ---------------------------------------------------------------------------

MONTE_CARLO_DEFAULT_SAMPLES = 10000
MONTE_CARLO_DEFAULT_PERTURBATION_PCT = 0.10
MONTE_CARLO_CONVERGENCE_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
MONTE_CARLO_DISTRIBUTIONS = ("uniform",)
MONTE_CARLO_CSV_DIR = DATA_DIR / "monte_carlo"
#: The reference FlowSort rounds its reported flows to six decimals, so the
#: vectorised path can only be held to that precision when the two are compared.
FLOW_AGREEMENT_TOLERANCE = 1e-5

_CATEGORY_CODES = [CATEGORY_LABELS[index]["code"] for index in sorted(CATEGORY_LABELS)]


def _flow_components_by_company(prepared: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Row and column sums of the single-criterion preference matrices, per company.

    FlowSort with limiting profiles compares one alternative at a time against the
    profile set, and the resulting flows are linear in the criterion weights:

        phi+(a) = sum_k w_k * rowsum_k(a) / ((n - 1) * sum_k w_k)

    with ``rowsum_k`` and ``colsum_k`` depending only on the evaluations and the
    preference functions. Precomputing them turns each Monte Carlo draw into a
    single matrix product, so 10,000 draws cost one pass over the scoring layer
    instead of 10,000.

    Returns two arrays of shape (companies, criteria, profiles + 1).
    """

    row_sums = []
    column_sums = []
    for alternative in prepared["dataset_array"]:
        relation_matrix = np.vstack([prepared["profiles_matrix"], alternative])
        matrices = single_criterion_preference_matrices(
            relation_matrix,
            prepared["q_vector"],
            prepared["s_vector"],
            prepared["p_vector"],
            prepared["preference_functions"],
        )
        row_sums.append(matrices.sum(axis=2))
        column_sums.append(matrices.sum(axis=1))
    return np.asarray(row_sums), np.asarray(column_sums)


def _sampled_flows(
    weight_samples: np.ndarray,
    row_sums: np.ndarray,
    column_sums: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positive, negative and net flows for every weight draw.

    Each array has shape (draws, companies, profiles + 1); the last position on
    the final axis is the alternative and the ones before it are the limiting
    profiles, matching the row order used by ``flowsort_limiting``.
    """

    node_count = row_sums.shape[2]
    scale = float(node_count - 1) * weight_samples.sum(axis=1)
    positive = np.einsum("sk,ckp->scp", weight_samples, row_sums) / scale[:, None, None]
    negative = np.einsum("sk,ckp->scp", weight_samples, column_sums) / scale[:, None, None]
    return positive, negative, positive - negative


def _assign_sampled_categories(
    positive: np.ndarray,
    negative: np.ndarray,
    net: np.ndarray,
    rule: str,
) -> np.ndarray:
    """Apply the FlowSort limiting-profile assignment rule to every draw."""

    draws, company_count, node_count = positive.shape
    profile_count = node_count - 1
    categories = np.zeros((draws, company_count), dtype=np.int16)
    for draw in range(draws):
        for company in range(company_count):
            fp = positive[draw, company]
            fm = negative[draw, company]
            fn = net[draw, company]
            h_pos, h_neg, h_net = _limiting_category_assignments(
                fp[:profile_count], fp[-1], fm[:profile_count], fm[-1], fn[:profile_count], fn[-1]
            )
            categories[draw, company] = (
                h_net if rule == "net" else h_pos if rule == "positive" else h_neg
            )
    return categories


def _pearson_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation between every column of ``left`` and of ``right``.

    A column with no variance - a company that never leaves its class, say -
    yields 0.0 rather than a NaN: there is no monotonic relation to detect, and
    the callers report the zero-variance case separately.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    centered_left = left - left.mean(axis=0)
    centered_right = right - right.mean(axis=0)
    norm_left = np.sqrt((centered_left * centered_left).sum(axis=0))
    norm_right = np.sqrt((centered_right * centered_right).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (centered_left.T @ centered_right) / np.outer(norm_left, norm_right)
    return np.where(np.isfinite(result), result, 0.0)


def _correlation_p_values(rho: np.ndarray, sample_count: int) -> np.ndarray:
    """Two-sided p-values for a correlation coefficient under the t approximation."""

    coefficients = np.clip(np.asarray(rho, dtype=float), -1.0, 1.0)
    degrees_of_freedom = sample_count - 2
    if degrees_of_freedom <= 0:
        return np.ones_like(coefficients)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistic = coefficients * np.sqrt(degrees_of_freedom / (1.0 - coefficients**2))
    p_values = 2.0 * t_distribution.sf(np.abs(t_statistic), degrees_of_freedom)
    return np.where(np.isfinite(p_values), p_values, 0.0)


def _partial_rank_correlations(
    ranked_inputs: np.ndarray,
    ranked_outputs: np.ndarray,
) -> np.ndarray:
    """PRCC of every input against every output, controlling for the other inputs.

    Reported alongside the plain rank correlation because the draws move all the
    weights at once: PRCC isolates the effect of one criterion with the others
    held fixed. The inputs here are sampled independently, so the two measures
    should agree closely - a divergence would be a sign that the sampling, not
    the model, is inducing the association.
    """

    input_count = ranked_inputs.shape[1]
    output_count = ranked_outputs.shape[1]
    results = np.zeros((input_count, output_count))
    for index in range(output_count):
        output_column = ranked_outputs[:, index]
        if np.ptp(output_column) == 0:
            continue
        block = np.column_stack([ranked_inputs, output_column])
        correlation = np.corrcoef(block, rowvar=False)
        correlation = np.where(np.isfinite(correlation), correlation, 0.0)
        np.fill_diagonal(correlation, 1.0)
        precision = np.linalg.pinv(correlation)
        target = precision.shape[0] - 1
        for row in range(input_count):
            denominator = math.sqrt(abs(precision[row, row] * precision[target, target]))
            if denominator == 0:
                continue
            results[row, index] = -precision[row, target] / denominator
    return np.clip(results, -1.0, 1.0)


def _class_probability_matrix(categories: np.ndarray, category_count: int) -> np.ndarray:
    """Share of draws in each class, per company, as percentages."""

    draws = categories.shape[0]
    counts = np.stack(
        [(categories == index).sum(axis=0) for index in range(1, category_count + 1)],
        axis=1,
    )
    return counts / draws * 100.0


def _slug(text: str) -> str:
    """Make a label safe for a file name.

    Model tags carry a colon (``llama3.1:8b``), which is a path separator on some
    filesystems and awkward on the rest. Dots are kept: they already appear in the
    tags the rest of the study writes to disk, so stripping them here would break
    the correspondence between a file name and the model it came from.
    """

    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(text).strip()
    )
    return safe.strip("_") or "series"


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return str(path)


def _write_monte_carlo_csv_files(
    output_dir: Path,
    stem: str,
    *,
    run_metadata: list[tuple[str, Any]],
    stability: dict[str, Any],
    company_records: list[dict[str, Any]],
    criterion_records: list[dict[str, Any]],
    convergence: list[dict[str, Any]],
    category_codes: list[str],
    draws: dict[str, Any] | None,
) -> list[str]:
    """Persist the simulation as CSV, one file per table.

    The result is several tables of different shapes, so a single CSV would have
    to either nest or pad. Each file is named after the run so repeated runs
    accumulate side by side instead of overwriting one another.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    written.append(
        _write_csv(
            output_dir / f"{stem}_run.csv",
            ["parameter", "value"],
            [[name, value] for name, value in run_metadata]
            + [[name, value] for name, value in stability.items()],
        )
    )

    written.append(
        _write_csv(
            output_dir / f"{stem}_companies.csv",
            ["company", "baseline_category", "most_frequent_category"]
            + [f"probability_{code}_pct" for code in category_codes]
            + [
                "stability_pct",
                "mean_absolute_class_shift",
                "baseline_net_flow",
                "net_flow_mean",
                "net_flow_std",
                "net_flow_min",
                "net_flow_p2_5",
                "net_flow_p50",
                "net_flow_p97_5",
                "net_flow_max",
            ],
            [
                [
                    item["company"],
                    item["baseline_category"],
                    item["most_frequent_category"],
                ]
                + [item["class_probability_pct"][code] for code in category_codes]
                + [
                    item["stability_pct"],
                    item["mean_absolute_class_shift"],
                    item["baseline_net_flow"],
                    item["net_flow"]["mean"],
                    item["net_flow"]["std"],
                    item["net_flow"]["min"],
                    item["net_flow"]["p2_5"],
                    item["net_flow"]["p50"],
                    item["net_flow"]["p97_5"],
                    item["net_flow"]["max"],
                ]
                for item in company_records
            ],
        )
    )

    written.append(
        _write_csv(
            output_dir / f"{stem}_criterion_contributions.csv",
            [
                "code",
                "measure",
                "baseline_weight",
                "lower_bound",
                "upper_bound",
                "normalized_mean",
                "normalized_std",
                "contribution_pct",
                "mean_abs_srcc_net_flow",
                "max_abs_srcc_net_flow",
                "mean_abs_srcc_raw_weight",
                "mean_abs_prcc_net_flow",
                "srcc_vs_reclassified_companies",
                "srcc_vs_reclassified_p_value",
            ],
            [
                [
                    item["code"],
                    item["measure"],
                    item["baseline_weight"],
                    item["sampled_weight"]["lower_bound"],
                    item["sampled_weight"]["upper_bound"],
                    item["sampled_weight"]["normalized_mean"],
                    item["sampled_weight"]["normalized_std"],
                    item["contribution_pct"],
                    item["mean_abs_srcc_net_flow"],
                    item["max_abs_srcc_net_flow"],
                    item["mean_abs_srcc_raw_weight"],
                    item["mean_abs_prcc_net_flow"],
                    item["srcc_vs_reclassified_companies"]["rho"],
                    item["srcc_vs_reclassified_companies"]["p_value"],
                ]
                for item in criterion_records
            ],
        )
    )

    written.append(
        _write_csv(
            output_dir / f"{stem}_criterion_by_company.csv",
            [
                "code",
                "measure",
                "company",
                "srcc_net_flow",
                "p_value",
                "prcc_net_flow",
                "srcc_category",
                "contribution_pct_within_company",
            ],
            [
                [
                    item["code"],
                    item["measure"],
                    entry["company"],
                    entry["srcc_net_flow"],
                    entry["p_value"],
                    entry["prcc_net_flow"],
                    entry["srcc_category"],
                    entry["contribution_pct_within_company"],
                ]
                for item in criterion_records
                for entry in item["by_company"]
            ],
        )
    )

    written.append(
        _write_csv(
            output_dir / f"{stem}_convergence.csv",
            [
                "samples",
                "identical_classification_pct",
                "mean_exact_class_match_pct",
                "max_class_probability_shift_pp",
            ],
            [
                [
                    item["samples"],
                    item["identical_classification_pct"],
                    item["mean_exact_class_match_pct"],
                    item["max_class_probability_shift_pp"],
                ]
                for item in convergence
            ],
        )
    )

    if draws is not None:
        codes = draws["codes"]
        companies = draws["companies"]
        weights = draws["weight_samples"]
        normalized = draws["normalized_samples"]
        net_flows = draws["net_flows"]
        categories = draws["categories"]
        written.append(
            _write_csv(
                output_dir / f"{stem}_draws.csv",
                ["draw"]
                + [f"weight_{code}" for code in codes]
                + [f"normalized_weight_{code}" for code in codes]
                + [f"net_flow_{company}" for company in companies]
                + [f"category_{company}" for company in companies],
                [
                    [index + 1]
                    + [float(value) for value in weights[index]]
                    + [float(value) for value in normalized[index]]
                    + [float(value) for value in net_flows[index]]
                    + [_CATEGORY_CODES[int(value) - 1] for value in categories[index]]
                    for index in range(weights.shape[0])
                ],
            )
        )

    return written


def run_weight_monte_carlo_sensitivity(
    sector_id: str = "semiconductors",
    criteria_formulation_result: dict[str, Any] | None = None,
    measure_weights: dict[str, float] | None = None,
    companies: list[str] | None = None,
    llm_config: dict[str, Any] | None = None,
    qualitative_source: str = "rater_study",
    rater_study_scores_path: Path | str | None = None,
    series_label: str | None = None,
    mode: str = "limiting",
    rule: str = "net",
    preference_function: str = "t1",
    indifference_threshold: float | int | dict[str, Any] | None = 0.0,
    preference_threshold: float | int | dict[str, Any] | None = 0.0,
    s_threshold: float | int | dict[str, Any] | None = 1.0,
    samples: int = MONTE_CARLO_DEFAULT_SAMPLES,
    perturbation_pct: float = MONTE_CARLO_DEFAULT_PERTURBATION_PCT,
    distribution: str = "uniform",
    seed: int = 42,
    write_csv: bool = True,
    csv_output_dir: str | None = None,
    include_draws: bool = False,
) -> dict[str, Any]:
    """Propagate uncertainty in the criterion weights through FlowSort by Monte Carlo.

    Each criterion weight is treated as a random variable with a uniform density
    over ``[w * (1 - perturbation_pct), w * (1 + perturbation_pct)]``, the whole
    weight vector is resampled ``samples`` times, and FlowSort is re-run on every
    draw. The outputs are the class assignment and the net flow of each company;
    the contribution of each criterion is then measured by rank correlation
    between the sampled weights and those outputs.

    The qualitative and quantitative scoring layer is held fixed on purpose: this
    analysis isolates the weights. ``run_partner_sensitivity_analysis`` covers the
    LLM scoring layer instead.

    Which scores it is held fixed *at* is a choice, and ``rater_study_scores_path``
    makes it an explicit one. The default is the study baseline, the median across
    the LLMs; pointing it at one model's score file instead re-runs the identical
    weight protocol on that model's ratings, which answers whether the weight
    robustness the study reports is a property of the weight vector or an artefact
    of aggregating the models first. ``series_label`` names that choice in the CSV
    file names and in the run metadata, so per-model runs are told apart by what
    they are rather than by their timestamps.

    Every run is written to CSV under ``data/monte_carlo`` - one file per table,
    named after the run so repeated runs accumulate rather than overwrite. Set
    ``include_draws`` to also write the draw-by-draw sample, which is what a
    reader needs to reproduce the correlations independently.
    """

    samples = int(samples)
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    perturbation_pct = float(perturbation_pct)
    if not 0.0 < perturbation_pct < 1.0:
        raise ValueError("perturbation_pct must be strictly between 0 and 1.")
    distribution = str(distribution).strip().lower()
    if distribution not in MONTE_CARLO_DISTRIBUTIONS:
        supported = ", ".join(MONTE_CARLO_DISTRIBUTIONS)
        raise ValueError(f"Unsupported distribution {distribution!r}. Supported: {supported}.")

    scores_path = Path(rater_study_scores_path) if rater_study_scores_path else None
    prepared = prepare_flowsort_inputs(
        sector_id=sector_id,
        criteria_formulation_result=criteria_formulation_result,
        measure_weights=measure_weights,
        companies=companies,
        llm_config=llm_config,
        qualitative_source=qualitative_source,
        rater_study_scores_path=scores_path if qualitative_source == "rater_study" else None,
        mode=mode,
        rule=rule,
        preference_function=preference_function,
        indifference_threshold=indifference_threshold,
        preference_threshold=preference_threshold,
        s_threshold=s_threshold,
    )

    rule = prepared["rule"]
    selected_codes = prepared["selected_codes"]
    target_companies = prepared["target_companies"]
    metadata_by_code = prepared["metadata_by_code"]
    baseline_weights = np.asarray(prepared["weights_vector"], dtype=float)
    criterion_count = len(selected_codes)
    company_count = len(target_companies)
    category_count = len(_CATEGORY_CODES)

    row_sums, column_sums = _flow_components_by_company(prepared)

    # The fast path must reproduce the shipped implementation exactly, otherwise
    # every number below is measuring a different method than the one the study
    # reports. Check it on the baseline weights before spending the draws.
    reference_details = _flowsort_details(
        dataset=prepared["dataset_array"],
        profiles=prepared["profiles_matrix"],
        W=baseline_weights,
        Q=prepared["q_vector"],
        S=prepared["s_vector"],
        P=prepared["p_vector"],
        F=prepared["preference_functions"],
        mode=prepared["mode"],
    )
    reference_categories = np.asarray(
        [detail[f"{rule}_category"] for detail in reference_details], dtype=np.int16
    )
    baseline_positive, baseline_negative, baseline_net = _sampled_flows(
        baseline_weights[None, :], row_sums, column_sums
    )
    baseline_categories = _assign_sampled_categories(
        baseline_positive, baseline_negative, baseline_net, rule
    )[0]
    baseline_net_flows = baseline_net[0, :, -1]
    # All three flows are compared, not just the net one. An error that shifts the
    # positive and the negative flow in the same direction cancels out of the net
    # flow, so checking the net flow alone would let it through - and the positive
    # and negative rules assign on those two directly.
    max_flow_gap = max(
        float(
            np.max(
                np.abs(
                    sampled[0, :, -1]
                    - np.asarray([detail[key] for detail in reference_details], dtype=float)
                )
            )
        )
        for sampled, key in (
            (baseline_positive, "positive_flow"),
            (baseline_negative, "negative_flow"),
            (baseline_net, "net_flow"),
        )
    )
    if not np.array_equal(baseline_categories, reference_categories):
        raise RuntimeError(
            "The vectorised FlowSort path disagrees with the reference implementation "
            "on the baseline weights; the Monte Carlo results would not describe the "
            "method actually used."
        )
    # The classes can agree while the flows do not, if a wrong flow happens to land
    # in the same band. The reference rounds its flows to six decimals, so anything
    # past that tolerance is a real disagreement rather than the rounding.
    if max_flow_gap > FLOW_AGREEMENT_TOLERANCE:
        raise RuntimeError(
            "The vectorised FlowSort path reproduces the baseline classes but not the "
            f"baseline flows (largest gap {max_flow_gap:.3e}); the correlations would be "
            "computed on flows the method does not produce."
        )

    rng = np.random.default_rng(int(seed))
    lower_bounds = baseline_weights * (1.0 - perturbation_pct)
    upper_bounds = baseline_weights * (1.0 + perturbation_pct)
    # Draw in criterion-code order rather than in whatever order the caller's
    # weight mapping happened to be built in. The classification itself is a sum
    # over criteria and does not care, but the generator does: without this, the
    # same seed produces a different sample path depending on how the weights
    # were passed in, and a run stops being reproducible from its seed alone.
    sampling_order = sorted(range(criterion_count), key=lambda index: selected_codes[index])
    sorted_samples = rng.uniform(
        lower_bounds[sampling_order],
        upper_bounds[sampling_order],
        size=(samples, criterion_count),
    )
    weight_samples = np.empty_like(sorted_samples)
    weight_samples[:, sampling_order] = sorted_samples
    normalized_samples = weight_samples / weight_samples.sum(axis=1, keepdims=True)

    positive, negative, net = _sampled_flows(weight_samples, row_sums, column_sums)
    categories = _assign_sampled_categories(positive, negative, net, rule)
    net_flows = net[:, :, -1]

    # --- outcome stability -------------------------------------------------
    matches_baseline = categories == baseline_categories[None, :]
    exact_match_pct_per_draw = matches_baseline.mean(axis=1) * 100.0
    reclassified_per_draw = (~matches_baseline).sum(axis=1)
    absolute_shift = np.abs(categories - baseline_categories[None, :])
    class_probability_pct = _class_probability_matrix(categories, category_count)

    baseline_order = np.argsort(-baseline_net_flows, kind="stable")
    sampled_order = np.argsort(-net_flows, axis=1, kind="stable")
    ranking_preserved = (sampled_order == baseline_order[None, :]).all(axis=1)
    baseline_rank_vector = rankdata(baseline_net_flows)
    sampled_rank_vectors = rankdata(net_flows, axis=1)
    ranking_spearman = _pearson_matrix(
        sampled_rank_vectors.T, baseline_rank_vector[:, None]
    ).ravel()

    # --- criterion contribution -------------------------------------------
    # The primary correlations use the normalised draws, because FlowSort divides
    # the weighted preference degree by the sum of the weights: what reaches the
    # method is the share of the total weight held by each criterion, not its raw
    # level. The raw draws are kept as a secondary reading - they are mutually
    # independent by construction - and are what the partial correlation is
    # computed on, since the normalised weights sum to one and would make the
    # correlation matrix singular.
    ranked_inputs = rankdata(normalized_samples, axis=0)
    ranked_raw_inputs = rankdata(weight_samples, axis=0)
    ranked_net_flows = rankdata(net_flows, axis=0)
    ranked_categories = rankdata(categories.astype(float), axis=0)
    ranked_reclassified = rankdata(reclassified_per_draw.astype(float))[:, None]

    srcc_net_flow = _pearson_matrix(ranked_inputs, ranked_net_flows)
    srcc_raw_net_flow = _pearson_matrix(ranked_raw_inputs, ranked_net_flows)
    srcc_category = _pearson_matrix(ranked_inputs, ranked_categories)
    srcc_reclassified = _pearson_matrix(ranked_inputs, ranked_reclassified).ravel()
    prcc_net_flow = _partial_rank_correlations(ranked_raw_inputs, ranked_net_flows)
    p_values_net_flow = _correlation_p_values(srcc_net_flow, samples)
    p_values_reclassified = _correlation_p_values(srcc_reclassified, samples)

    mean_squared_srcc = (srcc_net_flow**2).mean(axis=1)
    total_squared_srcc = float(mean_squared_srcc.sum())
    # With a single criterion the weight cancels out of FlowSort entirely - the
    # method divides the weighted preference degree by the sum of the weights, so
    # w * D / w = D - and every draw produces the identical classification. There
    # is then no output variance to attribute, every correlation is zero, and the
    # contributions cannot be rescaled to sum to 100%. Reporting zeros is the
    # honest answer, but it has to be labelled, or a reader sees a contribution
    # column that does not add up and assumes the rescaling is broken.
    contributions_are_defined = total_squared_srcc > 0
    contribution_pct = (
        mean_squared_srcc / total_squared_srcc * 100.0
        if contributions_are_defined
        else np.zeros(criterion_count)
    )
    squared_by_company = srcc_net_flow**2
    company_totals = squared_by_company.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        contribution_pct_within_company = squared_by_company / company_totals[None, :] * 100.0
    contribution_pct_within_company = np.where(
        np.isfinite(contribution_pct_within_company), contribution_pct_within_company, 0.0
    )

    constant_output_companies = [
        target_companies[index]
        for index in range(company_count)
        if np.ptp(net_flows[:, index]) == 0
    ]

    # --- convergence -------------------------------------------------------
    convergence = []
    previous_probabilities = None
    reported_checkpoints: set[int] = set()
    for fraction in MONTE_CARLO_CONVERGENCE_FRACTIONS:
        checkpoint = max(2, int(round(samples * fraction)))
        checkpoint = min(checkpoint, samples)
        # On a small run the fractions collapse onto the same prefix - at 10 draws
        # the 10% and 25% marks are both 2. Repeating the identical row five times
        # would suggest five measurements where there is one.
        if checkpoint in reported_checkpoints:
            continue
        reported_checkpoints.add(checkpoint)
        partial = categories[:checkpoint]
        probabilities = _class_probability_matrix(partial, category_count)
        shift = (
            float(np.max(np.abs(probabilities - previous_probabilities)))
            if previous_probabilities is not None
            else None
        )
        convergence.append(
            {
                "samples": checkpoint,
                "identical_classification_pct": _round_float(
                    float((partial == baseline_categories[None, :]).all(axis=1).mean() * 100.0)
                ),
                "mean_exact_class_match_pct": _round_float(
                    float((partial == baseline_categories[None, :]).mean() * 100.0)
                ),
                "max_class_probability_shift_pp": _round_float(shift) if shift is not None else None,
            }
        )
        previous_probabilities = probabilities

    # --- reporting ---------------------------------------------------------
    company_records = []
    for index, company in enumerate(target_companies):
        baseline_index = int(baseline_categories[index])
        probabilities = {
            code: _round_float(float(class_probability_pct[index, position]))
            for position, code in enumerate(_CATEGORY_CODES)
        }
        driver_order = np.argsort(-np.abs(srcc_net_flow[:, index]))
        company_records.append(
            {
                "company": company,
                "baseline_category": CATEGORY_LABELS[baseline_index]["code"],
                "baseline_label": CATEGORY_LABELS[baseline_index]["label"],
                "baseline_net_flow": _round_float(float(baseline_net_flows[index])),
                "class_probability_pct": probabilities,
                "stability_pct": _round_float(
                    float(matches_baseline[:, index].mean() * 100.0)
                ),
                "most_frequent_category": _CATEGORY_CODES[
                    int(np.argmax(class_probability_pct[index]))
                ],
                "mean_absolute_class_shift": _round_float(float(absolute_shift[:, index].mean())),
                "net_flow": {
                    "mean": _round_float(float(net_flows[:, index].mean())),
                    "std": _round_float(float(net_flows[:, index].std(ddof=1))),
                    "min": _round_float(float(net_flows[:, index].min())),
                    "p2_5": _round_float(float(np.percentile(net_flows[:, index], 2.5))),
                    "p50": _round_float(float(np.percentile(net_flows[:, index], 50))),
                    "p97_5": _round_float(float(np.percentile(net_flows[:, index], 97.5))),
                    "max": _round_float(float(net_flows[:, index].max())),
                },
                "top_drivers": [
                    {
                        "code": selected_codes[criterion],
                        "measure": metadata_by_code[selected_codes[criterion]]["measure"],
                        "srcc_net_flow": _round_float(float(srcc_net_flow[criterion, index])),
                        "contribution_pct": _round_float(
                            float(contribution_pct_within_company[criterion, index])
                        ),
                    }
                    for criterion in driver_order[:3]
                ],
            }
        )

    criterion_records = []
    for criterion, code in enumerate(selected_codes):
        criterion_records.append(
            {
                "code": code,
                "measure": metadata_by_code[code]["measure"],
                "baseline_weight": _round_float(float(baseline_weights[criterion])),
                "sampled_weight": {
                    "lower_bound": _round_float(float(lower_bounds[criterion])),
                    "upper_bound": _round_float(float(upper_bounds[criterion])),
                    "sampled_min": _round_float(float(weight_samples[:, criterion].min())),
                    "sampled_max": _round_float(float(weight_samples[:, criterion].max())),
                    "normalized_mean": _round_float(float(normalized_samples[:, criterion].mean())),
                    "normalized_std": _round_float(
                        float(normalized_samples[:, criterion].std(ddof=1))
                    ),
                },
                "contribution_pct": _round_float(float(contribution_pct[criterion])),
                "mean_abs_srcc_net_flow": _round_float(
                    float(np.abs(srcc_net_flow[criterion]).mean())
                ),
                "max_abs_srcc_net_flow": _round_float(float(np.abs(srcc_net_flow[criterion]).max())),
                "mean_abs_srcc_raw_weight": _round_float(
                    float(np.abs(srcc_raw_net_flow[criterion]).mean())
                ),
                "mean_abs_prcc_net_flow": _round_float(
                    float(np.abs(prcc_net_flow[criterion]).mean())
                ),
                "srcc_vs_reclassified_companies": {
                    "rho": _round_float(float(srcc_reclassified[criterion])),
                    "p_value": _round_float(float(p_values_reclassified[criterion])),
                },
                "by_company": [
                    {
                        "company": target_companies[index],
                        "srcc_net_flow": _round_float(float(srcc_net_flow[criterion, index])),
                        "p_value": _round_float(float(p_values_net_flow[criterion, index])),
                        "prcc_net_flow": _round_float(float(prcc_net_flow[criterion, index])),
                        "srcc_category": _round_float(float(srcc_category[criterion, index])),
                        "contribution_pct_within_company": _round_float(
                            float(contribution_pct_within_company[criterion, index])
                        ),
                    }
                    for index in range(company_count)
                ],
            }
        )
    criterion_records.sort(key=lambda item: -(item["contribution_pct"] or 0.0))

    baseline_distribution = {code: 0 for code in _CATEGORY_CODES}
    for index in baseline_categories:
        baseline_distribution[CATEGORY_LABELS[int(index)]["code"]] += 1

    stability = {
        "identical_classification_pct": _round_float(
            float((reclassified_per_draw == 0).mean() * 100.0)
        ),
        "mean_exact_class_match_pct": _round_float(float(exact_match_pct_per_draw.mean())),
        "min_exact_class_match_pct": _round_float(float(exact_match_pct_per_draw.min())),
        "mean_absolute_class_shift": _round_float(float(absolute_shift.mean())),
        "max_absolute_class_shift": int(absolute_shift.max()),
        "mean_reclassified_companies": _round_float(float(reclassified_per_draw.mean())),
        "max_reclassified_companies": int(reclassified_per_draw.max()),
        "ranking_preserved_pct": _round_float(float(ranking_preserved.mean() * 100.0)),
        "mean_ranking_spearman_vs_baseline": _round_float(float(ranking_spearman.mean())),
        "min_ranking_spearman_vs_baseline": _round_float(float(ranking_spearman.min())),
    }

    analysis_notes = [
        "Each criterion weight is drawn from a uniform density over the original "
        f"weight plus or minus {perturbation_pct * 100:.0f}%, and FlowSort is re-run on "
        f"each of the {samples} draws.",
        "Only the weights vary. The company evaluations, the limiting profiles and "
        "the preference function are held at their baseline values, so every "
        "difference reported here is attributable to the weights.",
        "FlowSort normalises the weighted preference degree by the sum of the "
        "weights, so the assignments are invariant to a rescaling of the weight "
        "vector; renormalising each draw to sum to one would not change any class.",
        "Contribution is the squared Spearman rank correlation between a sampled "
        "weight and the company net flows, averaged over companies and rescaled to "
        "sum to 100% across criteria. The correlation is taken on the normalised "
        "draws, which is the form the weights reach FlowSort in.",
        "PRCC is reported on the raw independent draws and controls for the other "
        f"criteria. FlowSort is deterministic, so once the other {max(criterion_count - 1, 0)} "
        "weights are held fixed almost all of the remaining variation is explained "
        "and every PRCC sits near plus or minus one: read its sign, not its "
        "magnitude, and rank the criteria by contribution_pct instead.",
        "Convergence lists the same statistics recomputed on growing prefixes of the "
        "draws, so the sample size can be judged from the residual movement.",
        "Every run is written to CSV, one file per table; csv_output.files lists them.",
    ]
    if not contributions_are_defined:
        analysis_notes.append(
            "contributions_are_defined is false: the outputs did not vary across the "
            "draws, so there is no variance to attribute and every contribution is "
            "zero rather than a share of 100%. With a single criterion this is the "
            "expected result, because the weight cancels out of FlowSort."
        )

    csv_output: dict[str, Any] = {"written": False, "directory": None, "files": []}
    if write_csv:
        output_dir = Path(csv_output_dir) if csv_output_dir else MONTE_CARLO_CSV_DIR
        # The perturbation range belongs in the name: it is the parameter that
        # distinguishes one run from another, and without it two runs differ only
        # by a timestamp, so identifying a file means opening its companion
        # ``_run.csv``. Formatted as whole percent where possible (``pct10``),
        # falling back to a decimal-free form (``pct12_5``) otherwise.
        pct = perturbation_pct * 100
        pct_label = f"{pct:g}".replace(".", "_")
        # The scoring series joins the range in the name for the same reason: with
        # one run per LLM, the range and the seed no longer tell two files apart.
        series_part = (
            f"_{_slug(series_label)}" if series_label else ""
        )
        stem = (
            f"weight_mcs_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            f"{series_part}_pct{pct_label}_seed{int(seed)}"
        )
        csv_files = _write_monte_carlo_csv_files(
            output_dir,
            stem,
            run_metadata=[
                ("sector_id", prepared["sector_id"]),
                ("samples", samples),
                ("distribution", distribution),
                ("perturbation_pct", perturbation_pct),
                ("seed", int(seed)),
                ("rule", rule),
                ("preference_function", prepared["preference_function"]),
                ("qualitative_source", prepared["qualitative_source"]),
                ("qualitative_series", series_label or "baseline_median"),
                ("qualitative_scores_path", str(scores_path) if scores_path else "default"),
                ("criteria_count", criterion_count),
                ("companies_count", company_count),
            ],
            stability=stability,
            company_records=company_records,
            criterion_records=criterion_records,
            convergence=convergence,
            category_codes=_CATEGORY_CODES,
            draws={
                "codes": selected_codes,
                "companies": target_companies,
                "weight_samples": weight_samples,
                "normalized_samples": normalized_samples,
                "net_flows": net_flows,
                "categories": categories,
            }
            if include_draws
            else None,
        )
        csv_output = {"written": True, "directory": str(output_dir), "files": csv_files}

    return {
        "sector_id": prepared["sector_id"],
        "input_dataset": str(MEASURES_HITL_PATH.name),
        "profile_dataset": str(CORRECTED_ALL_PATH.name),
        "mode": prepared["mode"],
        "rule": rule,
        "preference_function": prepared["preference_function"],
        "qualitative_source": prepared["qualitative_source"],
        "qualitative_series": series_label or "baseline_median",
        "qualitative_scores_path": str(scores_path) if scores_path else None,
        "qualitative_study_provenance": prepared["study_provenance"],
        "monte_carlo": {
            "samples": samples,
            "distribution": distribution,
            "perturbation_pct": _round_float(perturbation_pct),
            "seed": int(seed),
            "criteria_count": criterion_count,
            "companies_count": company_count,
            "excluded_codes": prepared["excluded_codes"],
        },
        "baseline": {
            "class_distribution": baseline_distribution,
            "companies": [
                {
                    "company": company,
                    "assigned_category": CATEGORY_LABELS[int(baseline_categories[index])]["code"],
                    "net_flow": _round_float(float(baseline_net_flows[index])),
                }
                for index, company in enumerate(target_companies)
            ],
        },
        "stability": stability,
        "companies": company_records,
        "criterion_contributions": criterion_records,
        "contributions_are_defined": contributions_are_defined,
        "contribution_pct_total": _round_float(float(sum(contribution_pct))),
        "convergence": convergence,
        "csv_output": csv_output,
        "validation": {
            "vectorised_path_matches_reference_categories": True,
            # Significant digits, not decimal places: the gap is around 1e-7 from
            # the reference's own rounding, and _round_float would report it as
            # 0.0 - hiding the one number this field exists to show.
            "max_flow_gap_vs_reference": round_significant(max_flow_gap),
            # Kept under the original key as well: the field is already referenced
            # by the web client and by previously exported results.
            "max_net_flow_gap_vs_reference": round_significant(max_flow_gap),
            "companies_with_constant_net_flow": constant_output_companies,
        },
        "analysis_notes": analysis_notes,
    }
