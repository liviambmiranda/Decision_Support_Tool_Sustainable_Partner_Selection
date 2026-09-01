"""FlowSort utilities and partner qualification workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.decision_support_mcp.qualitative_scores import (
    DEFAULT_SENSITIVITY_MODEL_FAMILIES,
    PROMPT_VARIANT_LABELS,
    QUALITATIVE_PROMPT_VARIANTS,
    load_rater_study_scores,
    score_qualitative_company_measure,
)
from src.decision_support_mcp.quantitative_scores import (
    CORRECTED_ALL_PATH,
    MEASURES_HITL_PATH,
    TYPE_A_PERCENTAGE,
    TYPE_B_ABSOLUTE,
    TYPE_C_QUALITATIVE,
    EQUIPMENT_COMPANY_TYPE,
    _build_profiles,
    _canonical_company_name,
    _company_record_lookup,
    _is_not_reported,
    _load_measures_hitl,
    _metadata_by_code,
    _normalize_company_type,
    _revenue_musd_by_company_casefold,
    _round_float,
    _validate_equipment_revenue_coverage,
    calculate_flowsort_profiles,
    display_profiles,
    round_significant,
    score_quantitative_company_measure,
)


#: Global sub-criterion weights from the group hierarchical fuzzy BWM of the
#: study, keyed by SASB code. Sub-criterion labels in the write-up map here as:
#: s11-s13 -> Greenhouse Gas Emissions, s21-s23 -> Energy Management,
#: s31-s33 -> Water Management, s41-s42 -> Waste Management,
#: s51 -> Materials Sourcing, s61 -> Workforce Health & Safety,
#: s71 -> Recruiting & Managing a Global & Skilled Workforce.
#:
#: Four SASB codes are deliberately absent because the study excludes them:
#: TC-SC-410a.1 and TC-SC-410a.2 (Product Lifecycle Management), TC-SC-320a.2,
#: and TC-SC-520a.1. Weights are used as published and renormalised on load, so
#: the rounding in the table cannot skew the result.
GH_FBWM_STUDY_WEIGHTS: dict[str, float] = {
    # J1 Greenhouse Gas Emissions (0.229332)
    "TC-SC-110a.1.1": 0.074049,   # s11 Gross global Scope 1 emissions
    "TC-SC-110a.1.2": 0.103362,   # s12 Scope 1 from perfluorinated compounds
    "TC-SC-110a.2": 0.051921,     # s13 Emissions management strategy and targets
    # J2 Energy Management in Manufacturing (0.243837)
    "TC-SC-130a.1.1": 0.078733,   # s21 Total energy consumed
    "TC-SC-130a.1.2": 0.055205,   # s22 Percentage from grid electricity
    "TC-SC-130a.1.3": 0.109900,   # s23 Percentage from renewable energy
    # J3 Water Management (0.110765)
    "TC-SC-140a.1.1": 0.025077,   # s31 Total water withdrawn
    "TC-SC-140a.1.2": 0.049923,   # s32 Total water consumed
    "TC-SC-140a.1.3": 0.035765,   # s33 Percentage in high water-stress regions
    # J4 Waste Management (0.076700)
    "TC-SC-150a.1.1": 0.025567,   # s41 Hazardous waste from manufacturing
    "TC-SC-150a.1.2": 0.051134,   # s42 Percentage of hazardous waste recycled
    # J5 Materials Sourcing (0.207282)
    "TC-SC-440a.1": 0.207282,     # s51 Risks from critical materials
    # J6 Workforce Health & Safety (0.076700)
    "TC-SC-320a.1": 0.076700,     # s61 Workforce exposure to health hazards
    # J7 Recruiting & Managing a Global & Skilled Workforce (0.055383)
    "TC-SC-330a.1": 0.055383,     # s71 Percentage of employees requiring a visa
}


CATEGORY_LABELS = {
    1: {"code": "C1", "label": "ESG Leader"},
    2: {"code": "C2", "label": "ESG Average"},
    3: {"code": "C3", "label": "ESG Laggard"},
}


def distance_matrix(dataset, criteria: int = 0):
    dataset = np.asarray(dataset, dtype=float)
    n = dataset.shape[0]
    distance_array = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            distance_array[i, j] = dataset[i, criteria] - dataset[j, criteria]
    return distance_array


def _validate_promethee_vectors(
    dataset: np.ndarray,
    W,
    Q,
    S,
    P,
    F,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    dataset = np.asarray(dataset, dtype=float)
    if dataset.ndim != 2:
        raise ValueError("dataset must be a 2D matrix.")

    criteria_count = dataset.shape[1]
    w_vector = np.asarray(W, dtype=float)
    q_vector = np.asarray(Q, dtype=float)
    s_vector = np.asarray(S, dtype=float)
    p_vector = np.asarray(P, dtype=float)
    f_vector = [str(item).strip().lower() for item in F]

    for label, vector in (
        ("W", w_vector),
        ("Q", q_vector),
        ("S", s_vector),
        ("P", p_vector),
    ):
        if vector.ndim != 1:
            raise ValueError(f"{label} must be a 1D vector.")
        if len(vector) != criteria_count:
            raise ValueError(
                f"{label} length ({len(vector)}) must match the number of criteria ({criteria_count})."
            )

    if len(f_vector) != criteria_count:
        raise ValueError(
            f"F length ({len(f_vector)}) must match the number of criteria ({criteria_count})."
        )

    return dataset, w_vector, q_vector, s_vector, p_vector, f_vector


def _validate_limiting_profiles_matrix(dataset: np.ndarray, profiles) -> np.ndarray:
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2:
        raise ValueError("profiles must be a 2D matrix.")
    if dataset.shape[1] != profiles.shape[1]:
        raise ValueError(
            f"dataset criteria count ({dataset.shape[1]}) must match profiles criteria count ({profiles.shape[1]})."
        )
    if profiles.shape[0] != 4:
        raise ValueError(
            f"Limiting FlowSort requires exactly 4 profile rows (r1, r2, r3, r4). Received {profiles.shape[0]}."
        )
    return profiles


def _validate_limiting_mode(mode: str) -> str:
    normalized_mode = str(mode).lower().strip()
    if normalized_mode != "limiting":
        raise ValueError('Only mode="limiting" is supported by this FlowSort implementation.')
    return normalized_mode


def _ensure_valid_category_index(category: int, context: str) -> int:
    category_int = int(category)
    if category_int not in {1, 2, 3}:
        raise RuntimeError(
            f"FlowSort produced invalid category {category_int} in {context}. "
            "Expected one of: 1, 2, 3."
        )
    return category_int


def single_criterion_preference_matrices(dataset, Q, S, P, F):
    """The P_k(a,b) matrices, one per criterion, before the weighted aggregation.

    The criterion weights enter PROMETHEE only in the final weighted sum, so
    these matrices depend on the evaluations and the preference functions alone.
    Exposing them lets a caller that varies only the weights - the Monte Carlo
    weight sensitivity analysis - reuse one pass over the data instead of
    recomputing the same matrices on every draw.
    """

    dataset = np.asarray(dataset, dtype=float)
    if dataset.ndim != 2:
        raise ValueError("dataset must be a 2D matrix.")
    dataset, _, Q, S, P, F = _validate_promethee_vectors(
        dataset, np.ones(dataset.shape[1]), Q, S, P, F
    )
    n, m = dataset.shape
    matrices = np.zeros((m, n, n))

    for k in range(0, m):
        if F[k] not in {"t1", "t2", "t3", "t4", "t5", "t6", "t7"}:
            raise ValueError(
                f"Invalid preference function at criterion index {k}: {F[k]!r}. "
                "Expected one of: t1, t2, t3, t4, t5, t6, t7."
            )
        distance_array = distance_matrix(dataset, criteria=k)
        for i in range(0, n):
            for j in range(0, n):
                if i == j:
                    distance_array[i, j] = 0.0
                    continue
                d = distance_array[i, j]
                if F[k] == "t1":
                    distance_array[i, j] = 0.0 if d <= 0 else 1.0
                elif F[k] == "t2":
                    if d <= Q[k]:
                        distance_array[i, j] = 0.0
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == "t3":
                    if d <= 0:
                        distance_array[i, j] = 0.0
                    elif 0 < d <= P[k]:
                        distance_array[i, j] = d / P[k]
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == "t4":
                    if d <= Q[k]:
                        distance_array[i, j] = 0.0
                    elif Q[k] < d <= P[k]:
                        distance_array[i, j] = 0.5
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == "t5":
                    if d <= Q[k]:
                        distance_array[i, j] = 0.0
                    elif Q[k] < d <= P[k]:
                        distance_array[i, j] = (d - Q[k]) / (P[k] - Q[k])
                    else:
                        distance_array[i, j] = 1.0
                elif F[k] == "t6":
                    if d <= 0:
                        distance_array[i, j] = 0.0
                    else:
                        distance_array[i, j] = 1 - np.exp(-(d**2) / (2 * S[k] ** 2))
                elif F[k] == "t7":
                    if d == 0:
                        distance_array[i, j] = 0.0
                    elif 0 < d <= S[k]:
                        distance_array[i, j] = (d / S[k]) ** 0.5
                    else:
                        distance_array[i, j] = 1.0
        matrices[k] = distance_array
    return matrices


def preference_degree(dataset, W, Q, S, P, F):
    dataset, W, Q, S, P, F = _validate_promethee_vectors(dataset, W, Q, S, P, F)
    matrices = single_criterion_preference_matrices(dataset, Q, S, P, F)
    n = dataset.shape[0]
    pd_array = np.zeros((n, n))

    for k in range(0, dataset.shape[1]):
        pd_array = pd_array + W[k] * matrices[k]
    pd_array = pd_array / sum(W)
    return pd_array


def promethee_flows_raw(dataset, W, Q, S, P, F):
    dataset, W, Q, S, P, F = _validate_promethee_vectors(dataset, W, Q, S, P, F)
    pd_matrix = preference_degree(dataset, W, Q, S, P, F)
    n = pd_matrix.shape[0]
    flow_plus = np.sum(pd_matrix, axis=1) / (n - 1)
    flow_minus = np.sum(pd_matrix, axis=0) / (n - 1)
    flow_net = flow_plus - flow_minus
    return pd_matrix, flow_plus, flow_minus, flow_net


def _limiting_category_assignments(fp_r, fp_a, fm_r, fm_a, fn_r, fn_a):
    K1 = len(fp_r)
    K = K1 - 1

    h_pos = None
    for h in range(0, K):
        if fp_r[h] >= fp_a > fp_r[h + 1]:
            h_pos = h + 1
            break
    if h_pos is None:
        if fp_a > fp_r[0]:
            h_pos = 1
        elif fp_a <= fp_r[-1]:
            h_pos = K
        else:
            for h in range(0, K):
                if fp_r[h] >= fp_a >= fp_r[h + 1]:
                    h_pos = h + 1
                    break
            if h_pos is None:
                h_pos = 1

    h_neg = None
    for h in range(0, K):
        if fm_r[h] < fm_a <= fm_r[h + 1]:
            h_neg = h + 1
            break
    if h_neg is None:
        if fm_a <= fm_r[0]:
            h_neg = 1
        elif fm_a > fm_r[-1]:
            h_neg = K
        else:
            for h in range(0, K):
                if fm_r[h] <= fm_a <= fm_r[h + 1]:
                    h_neg = h + 1
                    break
            if h_neg is None:
                h_neg = 1

    h_net = None
    for h in range(0, K):
        if fn_r[h] >= fn_a > fn_r[h + 1]:
            h_net = h + 1
            break
    if h_net is None:
        if fn_a > fn_r[0]:
            h_net = 1
        elif fn_a <= fn_r[-1]:
            h_net = K
        else:
            for h in range(0, K):
                if fn_r[h] >= fn_a >= fn_r[h + 1]:
                    h_net = h + 1
                    break
            if h_net is None:
                h_net = 1

    return (
        _ensure_valid_category_index(h_pos, "_limiting_category_assignments/positive"),
        _ensure_valid_category_index(h_neg, "_limiting_category_assignments/negative"),
        _ensure_valid_category_index(h_net, "_limiting_category_assignments/net"),
    )


def flowsort_limiting(dataset, profiles, W, Q, S, P, F):
    dataset, W, Q, S, P, F = _validate_promethee_vectors(dataset, W, Q, S, P, F)
    profiles = _validate_limiting_profiles_matrix(dataset, profiles)
    n_alt, _ = dataset.shape
    K1 = profiles.shape[0]
    cat_pos = np.zeros(n_alt, dtype=int)
    cat_neg = np.zeros(n_alt, dtype=int)
    cat_net = np.zeros(n_alt, dtype=int)

    for i, alternative in enumerate(dataset):
        relation_matrix = np.vstack([profiles, alternative])
        _, fp, fm, fn = promethee_flows_raw(relation_matrix, W, Q, S, P, F)
        fp_r, fp_a = fp[:K1], fp[-1]
        fm_r, fm_a = fm[:K1], fm[-1]
        fn_r, fn_a = fn[:K1], fn[-1]
        h_pos, h_neg, h_net = _limiting_category_assignments(fp_r, fp_a, fm_r, fm_a, fn_r, fn_a)
        cat_pos[i] = _ensure_valid_category_index(h_pos, "flowsort_limiting/positive")
        cat_neg[i] = _ensure_valid_category_index(h_neg, "flowsort_limiting/negative")
        cat_net[i] = _ensure_valid_category_index(h_net, "flowsort_limiting/net")
    return cat_pos, cat_neg, cat_net


def flowsort_central(dataset, profiles, W, Q, S, P, F):
    raise ValueError('FlowSort central profiles are not supported. Use mode="limiting".')


def flowsort_method(dataset, profiles, W, Q, S, P, F, mode="limiting", rule="net", verbose=True):
    mode = _validate_limiting_mode(mode)
    rule = rule.lower()
    if rule not in {"net", "positive", "negative"}:
        raise ValueError("rule must be one of: net, positive, negative.")

    cat_pos, cat_neg, cat_net = flowsort_limiting(dataset, profiles, W, Q, S, P, F)

    if rule == "net":
        cat_idx = cat_net
    elif rule == "positive":
        cat_idx = cat_pos
    else:
        cat_idx = cat_neg
    if verbose:
        for i in range(0, len(cat_idx)):
            category = _ensure_valid_category_index(cat_idx[i], "flowsort_method/output")
            print("a" + str(i + 1) + ": " + "C" + str(category))
    return cat_idx



def _normalize_measure_weights(
    criteria_formulation_result: dict[str, Any] | None,
    measure_weights: dict[str, float] | None,
) -> tuple[dict[str, float], list[str]]:
    raw_weights: dict[str, float] = {}
    if measure_weights:
        raw_weights = {str(code): float(weight) for code, weight in measure_weights.items() if float(weight) > 0}
    elif criteria_formulation_result:
        for item in criteria_formulation_result.get("global_weights", []):
            code = str(item.get("code", "")).strip()
            weight = float(item.get("global_crisp_weight", 0.0))
            if code and weight > 0:
                raw_weights[code] = weight
    else:
        # No weights supplied: fall back to the study's GH-FBWM weights so a
        # FlowSort run is reproducible without re-deriving them each time.
        raw_weights = dict(GH_FBWM_STUDY_WEIGHTS)

    if not raw_weights:
        raise ValueError("No positive criterion weights were provided.")

    total = sum(raw_weights.values())
    normalized = {code: weight / total for code, weight in raw_weights.items()}
    return normalized, sorted(raw_weights)


def _normalize_threshold_argument(
    value: float | int | dict[str, Any] | None,
    selected_codes: list[str],
    *,
    label: str,
    default: float,
) -> dict[str, float]:
    if value is None:
        return {code: default for code in selected_codes}
    if isinstance(value, (int, float)):
        scalar = float(value)
        return {code: scalar for code in selected_codes}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be either a number or a mapping of criterion code to number.")
    normalized: dict[str, float] = {}
    for code in selected_codes:
        raw = value.get(code, default)
        normalized[code] = float(raw)
    return normalized


def _validate_preference_parameters(
    criteria_codes: list[str],
    preference_functions: list[str],
    q_vector: np.ndarray,
    s_vector: np.ndarray,
    p_vector: np.ndarray,
) -> None:
    for code, pref_fn, q_value, s_value, p_value in zip(
        criteria_codes, preference_functions, q_vector, s_vector, p_vector
    ):
        if pref_fn == "t1":
            continue
        if pref_fn == "t2":
            if q_value < 0:
                raise ValueError(
                    f"Invalid preference parameter for criterion {code}: function={pref_fn}, "
                    f"parameter=q ({q_value}). Expected q >= 0."
                )
            continue
        if pref_fn == "t3":
            if p_value <= 0:
                raise ValueError(
                    f"Invalid preference parameter for criterion {code}: function={pref_fn}, "
                    f"parameter=p ({p_value}). Expected p > 0."
                )
            continue
        if pref_fn in {"t4", "t5"}:
            if q_value < 0:
                raise ValueError(
                    f"Invalid preference parameter for criterion {code}: function={pref_fn}, "
                    f"parameter=q ({q_value}). Expected q >= 0."
                )
            if p_value <= q_value:
                raise ValueError(
                    f"Invalid preference parameter for criterion {code}: function={pref_fn}, "
                    f"parameter=p ({p_value}) with q={q_value}. Expected p > q."
                )
            continue
        if pref_fn in {"t6", "t7"}:
            if s_value <= 0:
                raise ValueError(
                    f"Invalid preference parameter for criterion {code}: function={pref_fn}, "
                    f"parameter=s ({s_value}). Expected s > 0."
                )
            continue

        raise ValueError(
            f"Invalid preference function for criterion {code}: function={pref_fn}. "
            "Expected one of: t1, t2, t3, t4, t5, t6, t7."
        )



def _orientation_sign(direction: str) -> float:
    return 1.0 if direction == "maximize" else -1.0


def _profile_vector(profile_values: dict[str, float | None]) -> list[float]:
    keys = ["r1", "r2", "r3", "r4"]
    values = [profile_values.get(key) for key in keys]
    if any(value is None for value in values):
        raise ValueError("FlowSort profile is incomplete for one of the selected criteria.")
    return [float(value) for value in values]


def _closest_profile_label(oriented_value: float, oriented_profiles: list[float]) -> str:
    """Nearest limiting profile by absolute distance.

    This answers "which boundary is this value nearest to", which is not the
    same question as "which category band does it fall in". On the integer 1-5
    qualitative scale a rating of 4 is nearest to r2 (0.5 away) even though it
    sits above r2 and therefore inside the C1 band. Use ``_profile_band_label``
    for the band. Exact ties are reported as ``rX~rY`` rather than silently
    resolved by argument order, because a rating of 3 is equidistant from r2 and
    r3 by construction.
    """

    labels = ["r1", "r2", "r3", "r4"]
    if not oriented_profiles:
        return "n/a"
    distances = [abs(float(oriented_value) - float(profile_value)) for profile_value in oriented_profiles]
    smallest = min(distances)
    winners = [label for label, distance in zip(labels, distances) if distance == smallest]
    return "~".join(winners)


def _profile_band_label(oriented_value: float, oriented_profiles: list[float]) -> str:
    """Which category region this criterion value falls in, on its own.

    The bands are delimited by the oriented profiles: above r2 is the C1 region,
    between r3 and r2 the C2 region, and at or below r3 the C3 region. This is a
    per-criterion diagnostic only; the company's actual FlowSort class comes from
    the weighted PROMETHEE flows across every criterion, so a company can sit in
    the C1 band on one criterion and the C3 band on another.
    """

    if len(oriented_profiles) < 4:
        return "n/a"
    value = float(oriented_value)
    _, r2, r3, _ = (float(item) for item in oriented_profiles)
    if value > r2:
        return "C1"
    if value > r3:
        return "C2"
    return "C3"


def _score_company_measure(
    company: str,
    company_type: str,
    code: str,
    metadata: dict[str, Any],
    profile_entry: dict[str, Any],
    record_lookup: dict[tuple[str, str], dict[str, Any]],
    revenue_musd_by_company: dict[str, float],
    llm_config: dict[str, Any] | None,
    qualitative_cache: dict[tuple[str, str, str], dict[str, Any]],
    study_scores: dict[tuple[str, str], int] | None = None,
) -> dict[str, Any]:
    canonical_company = _canonical_company_name(company)
    record = record_lookup.get((canonical_company, code))

    if profile_entry["metric_type"] != TYPE_C_QUALITATIVE:
        return score_quantitative_company_measure(
            company=canonical_company,
            company_type=company_type,
            profile_entry=profile_entry,
            record=record,
            revenue_musd_by_company=revenue_musd_by_company,
        )

    precomputed = None
    if study_scores:
        # The study keys on the dataset company name, which may differ from the
        # canonical form used elsewhere, so try both.
        precomputed = study_scores.get((company, code))
        if precomputed is None:
            precomputed = study_scores.get((canonical_company, code))

    return score_qualitative_company_measure(
        company=canonical_company,
        code=code,
        metadata=metadata,
        profile_entry=profile_entry,
        record=record,
        llm_config=llm_config,
        qualitative_cache=qualitative_cache,
        precomputed_score=precomputed,
    )



def _flowsort_details(dataset, profiles, W, Q, S, P, F, mode: str):
    _validate_limiting_mode(mode)
    dataset, W, Q, S, P, F = _validate_promethee_vectors(dataset, W, Q, S, P, F)
    profiles = _validate_limiting_profiles_matrix(dataset, profiles)
    n_alt = dataset.shape[0]
    details = []

    for i, alternative in enumerate(dataset):
        relation_matrix = np.vstack([profiles, alternative])
        _, fp, fm, fn = promethee_flows_raw(relation_matrix, W, Q, S, P, F)
        profile_count = profiles.shape[0]
        fp_r, fp_a = fp[:profile_count], fp[-1]
        fm_r, fm_a = fm[:profile_count], fm[-1]
        fn_r, fn_a = fn[:profile_count], fn[-1]
        positive_category, negative_category, net_category = _limiting_category_assignments(
            fp_r, fp_a, fm_r, fm_a, fn_r, fn_a
        )

        details.append(
            {
                "alternative_index": i,
                "positive_flow": _round_float(fp_a),
                "negative_flow": _round_float(fm_a),
                "net_flow": _round_float(fn_a),
                "profile_positive_flows": [_round_float(value) for value in fp_r],
                "profile_negative_flows": [_round_float(value) for value in fm_r],
                "profile_net_flows": [_round_float(value) for value in fn_r],
                "positive_category": _ensure_valid_category_index(
                    positive_category, "_flowsort_details/positive"
                ),
                "negative_category": _ensure_valid_category_index(
                    negative_category, "_flowsort_details/negative"
                ),
                "net_category": _ensure_valid_category_index(net_category, "_flowsort_details/net"),
            }
        )
    return details


def _condition_2_1_report(
    selected_codes: list[str],
    oriented_profile_vectors_by_code: dict[str, list[float]],
    raw_company_details: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check Condition (2.1) of Pelissari et al. and the domination requirement.

    Section 2.1 requires r_1 > r_2 > ... > r_{k+1} on every criterion, and that
    r_1 dominates every alternative while every alternative dominates r_{k+1}.
    Proposition 3.1, which is what makes the assignment rules well defined,
    rests on it. A criterion whose limiting profiles collapse onto one value
    carries weight into the outranking degree while separating nothing, so the
    violations are reported rather than silently absorbed by the assignment
    fallbacks. They are reported and not raised because a metric that no company
    reports is a fact about the data, not a usage error.
    """

    non_decreasing = []
    for code in selected_codes:
        oriented = oriented_profile_vectors_by_code[code]
        if not all(oriented[i] > oriented[i + 1] for i in range(len(oriented) - 1)):
            non_decreasing.append(
                {"code": code, "oriented_profiles": _profile_labels(oriented, round_significant)}
            )

    not_dominated = []
    for company_detail in raw_company_details:
        for code in selected_codes:
            oriented = oriented_profile_vectors_by_code[code]
            value = company_detail["criteria_values"][code]["_raw_oriented_value"]
            if not (oriented[0] >= value >= oriented[-1]):
                not_dominated.append(
                    {
                        "company": company_detail["company"],
                        "code": code,
                        "oriented_value": round_significant(value),
                        "oriented_r1": round_significant(oriented[0]),
                        "oriented_r4": round_significant(oriented[-1]),
                    }
                )

    return {
        "condition": (
            "Pelissari et al. (2021) Eq. (2.1): r1 > r2 > ... > r_{k+1} on every "
            "criterion, with r1 dominating every alternative and every "
            "alternative dominating r_{k+1}"
        ),
        "satisfied": not non_decreasing and not not_dominated,
        "criteria_with_non_decreasing_profiles": non_decreasing,
        "alternatives_outside_profile_range": not_dominated,
        "note": (
            "A criterion listed under criteria_with_non_decreasing_profiles does "
            "not separate any alternative, yet still carries its weight into the "
            "outranking degree. Consider dropping it before the weighting step."
        ),
    }


def _profile_labels(values, formatter) -> dict[str, float | None]:
    """Label a four-value profile vector r1..r4 for presentation."""

    return {f"r{index + 1}": formatter(value) for index, value in enumerate(values)}


def _hierarchy_groups(
    criteria_formulation_result: dict[str, Any] | None,
    selected_codes: list[str],
    normalized_weights: dict[str, float],
) -> list[dict[str, Any]]:
    """Rebuild the criteria tree that produced the weights, as FlowSort-H needs it.

    FlowSort-H (Pelissari et al., Omega 103:102381, Eq. 3.2) aggregates the
    preference degrees recursively, weighting by ``w_(r,s)`` at every level, with
    the children of each node normalised to one. Only the parent of each measure
    is needed to rebuild that tree here, because the hierarchy has two levels.

    Local weights are derived from the global weights actually used rather than
    read from the formulation result, so that dropping a measure the dataset
    cannot score renormalises its siblings instead of leaving a gap.
    """

    if not criteria_formulation_result:
        return []

    label_by_parent: dict[str, str] = {}
    parent_by_code: dict[str, str] = {}
    for item in criteria_formulation_result.get("global_weights", []):
        code = str(item.get("code", "")).strip()
        parent = str(item.get("parent_code", "")).strip()
        if not code or not parent:
            continue
        parent_by_code[code] = parent
        label_by_parent.setdefault(parent, str(item.get("parent_label") or parent))

    grouped: dict[str, list[str]] = {}
    for code in selected_codes:
        parent = parent_by_code.get(code)
        if parent is None:
            return []  # a flat weight map: there is no tree to walk
        grouped.setdefault(parent, []).append(code)

    groups: list[dict[str, Any]] = []
    for parent, codes in grouped.items():
        parent_weight = sum(normalized_weights[code] for code in codes)
        if parent_weight <= 0:
            continue
        groups.append(
            {
                "parent_code": parent,
                "parent_label": label_by_parent.get(parent, parent),
                "codes": codes,
                "parent_weight": parent_weight,
                "local_weights": {
                    code: normalized_weights[code] / parent_weight for code in codes
                },
            }
        )
    groups.sort(key=lambda group: (-group["parent_weight"], group["parent_code"]))
    return groups


def _node_assignment_details(
    dataset: np.ndarray,
    profiles: np.ndarray,
    column_index: dict[str, int],
    codes: list[str],
    weights: dict[str, float],
    q_by_code: dict[str, float],
    p_by_code: dict[str, float],
    s_by_code: dict[str, float],
    preference_function: str,
    mode: str,
) -> list[dict[str, Any]]:
    """Single-criterion FlowSort-H assignment restricted to one node's subtree.

    Equations (3.14)-(3.17): the flows are built from the same alternative and
    profiles, but only over the elementary criteria descending from the node, and
    with that node's children weights. The node's own weight ``w_r`` scales the
    alternative flow and the profile flows identically, so it cancels out of the
    assignment rule and is not applied here.
    """

    columns = [column_index[code] for code in codes]
    return _flowsort_details(
        dataset=dataset[:, columns],
        profiles=profiles[:, columns],
        W=np.asarray([weights[code] for code in codes], dtype=float),
        Q=np.asarray([q_by_code[code] for code in codes], dtype=float),
        S=np.asarray([s_by_code[code] for code in codes], dtype=float),
        P=np.asarray([p_by_code[code] for code in codes], dtype=float),
        F=[preference_function] * len(codes),
        mode=mode,
    )


def prepare_flowsort_inputs(
    sector_id: str = "semiconductors",
    criteria_formulation_result: dict[str, Any] | None = None,
    measure_weights: dict[str, float] | None = None,
    companies: list[str] | None = None,
    llm_config: dict[str, Any] | None = None,
    qualitative_source: str = "rater_study",
    mode: str = "limiting",
    rule: str = "net",
    preference_function: str = "t1",
    indifference_threshold: float | int | dict[str, Any] | None = 0.0,
    preference_threshold: float | int | dict[str, Any] | None = 0.0,
    s_threshold: float | int | dict[str, Any] | None = 1.0,
    rater_study_scores_path: Path | str | None = None,
) -> dict[str, Any]:
    """Everything FlowSort needs up to, but not including, the weighted aggregation.

    Split out of :func:`qualify_partners_with_flowsort` because the scoring layer
    - loading the datasets, normalising the company values, resolving the
    qualitative ratings - does not depend on the criterion weights. The Monte
    Carlo weight sensitivity analysis therefore builds the decision matrix once
    and then varies only the weight vector, instead of re-scoring every company
    on each of the thousands of draws.

    ``rater_study_scores_path`` overrides which consensus-score file the
    qualitative ratings are read from, defaulting to the one the rater study
    wrote. The prompt-sensitivity analysis uses it to swap in the medians
    obtained under one scoring prompt while every other input stays fixed.
    """

    mode = _validate_limiting_mode(mode)
    rule = str(rule).lower().strip()
    if rule not in {"net", "positive", "negative"}:
        raise ValueError("rule must be one of: net, positive, negative.")

    preference_function = str(preference_function).strip().lower()
    if preference_function not in {"t1", "t2", "t3", "t4", "t5", "t6", "t7"}:
        raise ValueError("preference_function must be one of: t1, t2, t3, t4, t5, t6, t7.")

    qualitative_source = str(qualitative_source or "rater_study").strip().lower()
    if qualitative_source not in {"rater_study", "llm"}:
        raise ValueError("qualitative_source must be one of: rater_study, llm.")

    # Default path: reuse the medians established by the rater-agreement study.
    # They are the median across several models rather than one model's single
    # opinion, and being fixed on disk they make a FlowSort run reproducible.
    study_scores: dict[tuple[str, str], int] = {}
    study_provenance: dict[str, Any] = {"available": False}
    if qualitative_source == "rater_study":
        study_scores, study_provenance = load_rater_study_scores(
            Path(rater_study_scores_path) if rater_study_scores_path else None
        )

    rows = _load_measures_hitl()
    metadata_by_code = _metadata_by_code(sector_id)
    profiles_by_code = calculate_flowsort_profiles(sector_id)
    normalized_weights, requested_codes = _normalize_measure_weights(
        criteria_formulation_result=criteria_formulation_result,
        measure_weights=measure_weights,
    )

    selected_codes = [code for code in normalized_weights if code in profiles_by_code and code in metadata_by_code]
    excluded_codes = [code for code in requested_codes if code not in selected_codes]
    if not selected_codes:
        raise ValueError("None of the provided weighted criteria are available in the qualification dataset.")

    dataset_companies = sorted({str(row["company"]) for row in rows})
    dataset_companies_by_canonical = {
        _canonical_company_name(company): company for company in dataset_companies
    }

    requested_companies = companies[:] if companies else dataset_companies
    target_companies = [_canonical_company_name(company) for company in requested_companies]
    unknown_companies = [
        company
        for company in requested_companies
        if _canonical_company_name(company) not in dataset_companies_by_canonical
    ]
    if unknown_companies:
        raise ValueError(f"Unknown companies in qualification request: {unknown_companies}")

    company_types = {company: _normalize_company_type(company) for company in target_companies}
    non_equipment_companies = [
        company
        for company, company_type in company_types.items()
        if company_type != EQUIPMENT_COMPANY_TYPE
    ]
    if non_equipment_companies:
        raise ValueError(
            "This study is restricted to Semiconductor & Solar Equipment suppliers. "
            f"Non-equipment companies received: {non_equipment_companies}"
        )
    revenue_musd_by_company = _revenue_musd_by_company_casefold()
    _validate_equipment_revenue_coverage(target_companies, revenue_musd_by_company)
    weights_vector = np.asarray([normalized_weights[code] for code in selected_codes], dtype=float)
    q_by_code = _normalize_threshold_argument(
        indifference_threshold,
        selected_codes,
        label="indifference_threshold",
        default=0.0,
    )
    p_by_code = _normalize_threshold_argument(
        preference_threshold,
        selected_codes,
        label="preference_threshold",
        default=0.0,
    )
    s_by_code = _normalize_threshold_argument(
        s_threshold,
        selected_codes,
        label="s_threshold",
        default=1.0,
    )
    q_vector = np.asarray([q_by_code[code] for code in selected_codes], dtype=float)
    s_vector = np.asarray([s_by_code[code] for code in selected_codes], dtype=float)
    p_vector = np.asarray([p_by_code[code] for code in selected_codes], dtype=float)
    preference_functions = [preference_function] * len(selected_codes)
    _validate_preference_parameters(
        criteria_codes=selected_codes,
        preference_functions=preference_functions,
        q_vector=q_vector,
        s_vector=s_vector,
        p_vector=p_vector,
    )
    record_lookup = _company_record_lookup(rows)
    qualitative_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    criteria_summary: dict[str, Any] = {}

    for code in selected_codes:
        profile_entry = profiles_by_code[code]
        criteria_summary[code] = {
            "measure": metadata_by_code[code]["measure"],
            "weight": _round_float(normalized_weights[code]),
            "metric_type": profile_entry["metric_type"],
            "direction": profile_entry["direction"],
            "profiles": display_profiles(profile_entry["profiles"]),
            "indifference_threshold": _round_float(q_by_code[code]),
            "preference_threshold": _round_float(p_by_code[code]),
            "s_threshold": _round_float(s_by_code[code]),
            "preference_function": preference_function,
        }

    dataset_rows = []
    raw_company_details = []
    profiles_matrix = []
    profile_vectors_by_code: dict[str, list[float]] = {}
    oriented_profile_vectors_by_code: dict[str, list[float]] = {}
    orientation_sign_by_code: dict[str, float] = {}

    for code in selected_codes:
        profile_entry = profiles_by_code[code]
        profile_vector = _profile_vector(profile_entry["profiles"])
        direction = profile_entry["direction"]
        sign = _orientation_sign(direction)
        oriented_profile_vector = [sign * value for value in profile_vector]
        profile_vectors_by_code[code] = profile_vector
        oriented_profile_vectors_by_code[code] = oriented_profile_vector
        orientation_sign_by_code[code] = sign
        profiles_matrix.append(oriented_profile_vector)

    profiles_matrix = np.asarray(profiles_matrix, dtype=float).T

    # Pre-flight: report every qualitative gap at once. Failing inside the
    # scoring loop would surface only the first missing cell, so the user would
    # fix one and immediately hit the next.
    if qualitative_source == "rater_study" and llm_config is None:
        missing: list[str] = []
        for company in target_companies:
            canonical = _canonical_company_name(company)
            for code in selected_codes:
                if profiles_by_code[code]["metric_type"] != TYPE_C_QUALITATIVE:
                    continue
                record = record_lookup.get((canonical, code))
                if record is None or _is_not_reported(record.get("value")):
                    continue  # handled by the worst-profile penalty
                if (company, code) in study_scores or (canonical, code) in study_scores:
                    continue
                missing.append(f"{company} / {code}")
        if missing:
            covered = sorted({code for _, code in study_scores})
            raise ValueError(
                f"{len(missing)} qualitative cell(s) have no score from the rater study: "
                + "; ".join(missing[:6])
                + ("; ..." if len(missing) > 6 else "")
                + f". The study covers {', '.join(covered) or 'nothing yet'}. Either drop these "
                "criteria from the weighting, run the Rating Evaluation over them, or pass "
                "llm_config to score them live."
            )

    for company in target_companies:
        criterion_details = {}
        oriented_values = []
        company_type = company_types[company]

        for code in selected_codes:
            profile_entry = profiles_by_code[code]
            detail = _score_company_measure(
                company=company,
                company_type=company_type,
                code=code,
                metadata=metadata_by_code[code],
                profile_entry=profile_entry,
                record_lookup=record_lookup,
                revenue_musd_by_company=revenue_musd_by_company,
                llm_config=llm_config,
                qualitative_cache=qualitative_cache,
                study_scores=study_scores,
            )
            direction = profile_entry["direction"]
            sign = orientation_sign_by_code[code]
            oriented_value = sign * detail["numeric_value"]
            oriented_values.append(oriented_value)
            weighted_oriented_value = oriented_value * float(normalized_weights[code])
            raw_profiles = profile_vectors_by_code[code]
            oriented_profiles = oriented_profile_vectors_by_code[code]
            distance_to_profiles = _profile_labels(
                [abs(oriented_value - value) for value in oriented_profiles],
                round_significant,
            )
            not_reported = detail["value_source"] == "not_reported_as_worst_profile_r4"
            closest_profile = (
                "r4" if not_reported
                else _closest_profile_label(oriented_value, oriented_profiles)
            )
            profile_band = (
                "C3" if not_reported
                else _profile_band_label(oriented_value, oriented_profiles)
            )
            criterion_details[code] = {
                "reported_value": detail["original_value"],
                "reported_numeric_value": _round_float(detail["reported_numeric_value"]),
                "normalized_value": _round_float(detail["normalized_value"]),
                "normalization_method": detail["normalization_method"],
                "numeric_value": _round_float(detail["numeric_value"]),
                "oriented_value": round_significant(oriented_value),
                "_raw_oriented_value": oriented_value,
                "weighted_oriented_value": _round_float(weighted_oriented_value),
                "original_value": detail["original_value"],
                "value_source": detail["value_source"],
                "rationale": detail["rationale"],
                "llm_provider": detail["llm_provider"],
                "llm_model": detail["llm_model"],
                "prompt_variant": detail["prompt_variant"],
                "direction": direction,
                "metric_type": profile_entry["metric_type"],
                "weight": _round_float(normalized_weights[code]),
                "orientation_sign": _round_float(sign),
                # Significant digits, not decimal places: revenue-normalised
                # metrics sit around 1e-7 and would all display as 0.0.
                "profiles": _profile_labels(raw_profiles, round_significant),
                "limiting_profiles": _profile_labels(raw_profiles, round_significant),
                "oriented_profiles": _profile_labels(oriented_profiles, round_significant),
                "closest_profile": closest_profile,
                "profile_band": profile_band,
                "distance_to_profiles": distance_to_profiles,
            }

        dataset_rows.append(oriented_values)
        raw_company_details.append(
            {
                "company": company,
                "company_type": company_type,
                "criteria_values": criterion_details,
            }
        )

    dataset_array = np.asarray(dataset_rows, dtype=float)

    return {
        "sector_id": sector_id,
        "mode": mode,
        "rule": rule,
        "preference_function": preference_function,
        "qualitative_source": qualitative_source,
        "study_provenance": study_provenance,
        "criteria_formulation_result": criteria_formulation_result,
        "metadata_by_code": metadata_by_code,
        "selected_codes": selected_codes,
        "excluded_codes": excluded_codes,
        "normalized_weights": normalized_weights,
        "target_companies": target_companies,
        "criteria_summary": criteria_summary,
        "dataset_array": dataset_array,
        "profiles_matrix": profiles_matrix,
        "weights_vector": weights_vector,
        "q_vector": q_vector,
        "s_vector": s_vector,
        "p_vector": p_vector,
        "q_by_code": q_by_code,
        "p_by_code": p_by_code,
        "s_by_code": s_by_code,
        "preference_functions": preference_functions,
        "oriented_profile_vectors_by_code": oriented_profile_vectors_by_code,
        "raw_company_details": raw_company_details,
    }


def qualify_partners_with_flowsort(
    sector_id: str = "semiconductors",
    criteria_formulation_result: dict[str, Any] | None = None,
    measure_weights: dict[str, float] | None = None,
    companies: list[str] | None = None,
    llm_config: dict[str, Any] | None = None,
    qualitative_source: str = "rater_study",
    mode: str = "limiting",
    rule: str = "net",
    preference_function: str = "t1",
    indifference_threshold: float | int | dict[str, Any] | None = 0.0,
    preference_threshold: float | int | dict[str, Any] | None = 0.0,
    s_threshold: float | int | dict[str, Any] | None = 1.0,
    rater_study_scores_path: Path | str | None = None,
) -> dict[str, Any]:
    prepared = prepare_flowsort_inputs(
        sector_id=sector_id,
        criteria_formulation_result=criteria_formulation_result,
        measure_weights=measure_weights,
        companies=companies,
        llm_config=llm_config,
        qualitative_source=qualitative_source,
        mode=mode,
        rule=rule,
        preference_function=preference_function,
        indifference_threshold=indifference_threshold,
        preference_threshold=preference_threshold,
        s_threshold=s_threshold,
        rater_study_scores_path=rater_study_scores_path,
    )
    mode = prepared["mode"]
    rule = prepared["rule"]
    preference_function = prepared["preference_function"]
    qualitative_source = prepared["qualitative_source"]
    study_provenance = prepared["study_provenance"]
    metadata_by_code = prepared["metadata_by_code"]
    selected_codes = prepared["selected_codes"]
    excluded_codes = prepared["excluded_codes"]
    normalized_weights = prepared["normalized_weights"]
    target_companies = prepared["target_companies"]
    criteria_summary = prepared["criteria_summary"]
    dataset_array = prepared["dataset_array"]
    profiles_matrix = prepared["profiles_matrix"]
    weights_vector = prepared["weights_vector"]
    q_vector = prepared["q_vector"]
    s_vector = prepared["s_vector"]
    p_vector = prepared["p_vector"]
    q_by_code = prepared["q_by_code"]
    p_by_code = prepared["p_by_code"]
    s_by_code = prepared["s_by_code"]
    preference_functions = prepared["preference_functions"]
    oriented_profile_vectors_by_code = prepared["oriented_profile_vectors_by_code"]
    raw_company_details = prepared["raw_company_details"]
    company_results_by_name: dict[str, dict[str, Any]] = {}

    details = _flowsort_details(
        dataset=dataset_array,
        profiles=profiles_matrix,
        W=weights_vector,
        Q=q_vector,
        S=s_vector,
        P=p_vector,
        F=preference_functions,
        mode=mode,
    )

    # FlowSort-H, Eqs. (3.14)-(3.17): besides the assignment over all criteria,
    # assign each alternative on every node of the tree, so that the topic and
    # the indicator holding it back are identifiable.
    condition_report = _condition_2_1_report(
        selected_codes, oriented_profile_vectors_by_code, raw_company_details
    )
    hierarchy_groups = _hierarchy_groups(
        criteria_formulation_result, selected_codes, normalized_weights
    )
    column_index = {code: index for index, code in enumerate(selected_codes)}
    node_kwargs = {
        "dataset": dataset_array,
        "profiles": profiles_matrix,
        "column_index": column_index,
        "q_by_code": q_by_code,
        "p_by_code": p_by_code,
        "s_by_code": s_by_code,
        "preference_function": preference_function,
        "mode": mode,
    }
    topic_details_by_parent = {
        group["parent_code"]: _node_assignment_details(
            codes=group["codes"], weights=group["local_weights"], **node_kwargs
        )
        for group in hierarchy_groups
    }
    indicator_details_by_code = {
        code: _node_assignment_details(codes=[code], weights={code: 1.0}, **node_kwargs)
        for code in selected_codes
    }

    def node_entry(node_detail: dict[str, Any]) -> dict[str, Any]:
        index = _ensure_valid_category_index(
            node_detail["net_category"]
            if rule == "net"
            else node_detail["positive_category"]
            if rule == "positive"
            else node_detail["negative_category"],
            "qualify_partners_with_flowsort/node",
        )
        return {
            "assigned_category": CATEGORY_LABELS[index]["code"],
            "assigned_label": CATEGORY_LABELS[index]["label"],
            "_category_index": index,
            "net_flow": node_detail["net_flow"],
            "positive_flow": node_detail["positive_flow"],
            "negative_flow": node_detail["negative_flow"],
            "profile_net_flows": node_detail["profile_net_flows"],
        }

    for company_index, (company_detail, flow_detail) in enumerate(
        zip(raw_company_details, details)
    ):
        topic_assignments = []
        for group in hierarchy_groups:
            entry = node_entry(topic_details_by_parent[group["parent_code"]][company_index])
            topic_assignments.append(
                {
                    "parent_code": group["parent_code"],
                    "parent_label": group["parent_label"],
                    "parent_weight": _round_float(group["parent_weight"]),
                    **entry,
                    "indicators": [
                        {
                            "code": code,
                            "measure": metadata_by_code[code]["measure"],
                            "local_weight": _round_float(group["local_weights"][code]),
                            "global_weight": _round_float(normalized_weights[code]),
                            **node_entry(indicator_details_by_code[code][company_index]),
                        }
                        for code in group["codes"]
                    ],
                }
            )
        # Worst node first: the paper's use for these is to target improvement.
        for topic in topic_assignments:
            topic["indicators"].sort(
                key=lambda item: (-item["_category_index"], -item["global_weight"])
            )
            for indicator in topic["indicators"]:
                indicator.pop("_category_index")
        topic_assignments.sort(
            key=lambda item: (-item["_category_index"], -item["parent_weight"])
        )
        for topic in topic_assignments:
            topic.pop("_category_index")
        company_detail["topic_assignments"] = topic_assignments

        selected_category_index = (
            flow_detail["net_category"]
            if rule == "net"
            else flow_detail["positive_category"]
            if rule == "positive"
            else flow_detail["negative_category"]
        )
        selected_category_index = _ensure_valid_category_index(
            selected_category_index, "qualify_partners_with_flowsort/selected_rule"
        )
        positive_category = _ensure_valid_category_index(
            flow_detail["positive_category"], "qualify_partners_with_flowsort/positive"
        )
        negative_category = _ensure_valid_category_index(
            flow_detail["negative_category"], "qualify_partners_with_flowsort/negative"
        )
        net_category = _ensure_valid_category_index(
            flow_detail["net_category"], "qualify_partners_with_flowsort/net"
        )
        category_info = CATEGORY_LABELS[selected_category_index]
        company_results_by_name[company_detail["company"]] = {
            "company": company_detail["company"],
            "company_type": company_detail["company_type"],
            "assigned_category": category_info["code"],
            "assigned_label": category_info["label"],
            "selected_rule": rule,
            "positive_category": CATEGORY_LABELS[positive_category]["code"],
            "negative_category": CATEGORY_LABELS[negative_category]["code"],
            "net_category": CATEGORY_LABELS[net_category]["code"],
            "positive_flow": flow_detail["positive_flow"],
            "negative_flow": flow_detail["negative_flow"],
            "net_flow": flow_detail["net_flow"],
            "profile_positive_flows": flow_detail["profile_positive_flows"],
            "profile_negative_flows": flow_detail["profile_negative_flows"],
            "profile_net_flows": flow_detail["profile_net_flows"],
            "topic_assignments": company_detail["topic_assignments"],
            "criteria_values": {
                code: {k: v for k, v in detail.items() if not k.startswith("_")}
                for code, detail in company_detail["criteria_values"].items()
            },
            "traceability": {
                "selected_rule": rule,
                "flow_to_category": {
                    "positive": {
                        "alternative": flow_detail["positive_flow"],
                        "profiles": flow_detail["profile_positive_flows"],
                    },
                    "negative": {
                        "alternative": flow_detail["negative_flow"],
                        "profiles": flow_detail["profile_negative_flows"],
                    },
                    "net": {
                        "alternative": flow_detail["net_flow"],
                        "profiles": flow_detail["profile_net_flows"],
                    },
                },
            },
        }

    ordered_results = [company_results_by_name[company] for company in target_companies]

    return {
        "sector_id": sector_id,
        "input_dataset": str(MEASURES_HITL_PATH.name),
        "mode": mode,
        "rule": rule,
        "preference_function": preference_function,
        "qualitative_prompt_variant_default": "prompt_1",
        "qualitative_source": qualitative_source,
        "qualitative_study_provenance": study_provenance,
        "threshold_note": (
            "Threshold usage depends on the PROMETHEE preference function: "
            "t1 ignores q/p/s; t2 uses q; t3 uses p; t4 and t5 use q and p; "
            "t6 and t7 use s."
        ),
        "categories": [
            {"code": category["code"], "label": category["label"]}
            for _, category in sorted(CATEGORY_LABELS.items())
        ],
        "criteria_weights": criteria_summary,
        "excluded_codes": excluded_codes,
        "profile_conditions": condition_report,
        "hierarchy": [
            {
                "parent_code": group["parent_code"],
                "parent_label": group["parent_label"],
                "parent_weight": _round_float(group["parent_weight"]),
                "indicators": [
                    {
                        "code": code,
                        "local_weight": _round_float(group["local_weights"][code]),
                        "global_weight": _round_float(normalized_weights[code]),
                    }
                    for code in group["codes"]
                ],
            }
            for group in hierarchy_groups
        ],
        "hierarchy_note": (
            "Assignments over all criteria follow FlowSort with limiting profiles. "
            "topic_assignments adds the FlowSort-H single-criterion assignment for "
            "every node of the criteria tree, each node scored on its own subtree "
            "with its children weights normalised to one, and ordered worst first. "
            "The assignment over all criteria is unaffected by the tree: with the "
            "global weights being the products along each path, the flat and the "
            "hierarchical aggregations coincide."
        ),
        "companies": ordered_results,
    }


def get_partner_qualification_inputs(sector_id: str = "semiconductors") -> dict[str, Any]:
    rows = _load_measures_hitl()
    metadata_by_code = _metadata_by_code(sector_id)
    companies = sorted({str(row["company"]) for row in rows})
    qualitative_codes = sorted(
        {
            str(row["code"])
            for row in rows
            if str(metadata_by_code.get(str(row["code"]), {}).get("category", "")).lower() == "qualitative"
        }
    )
    return {
        "sector_id": sector_id,
        "input_dataset": str(MEASURES_HITL_PATH.name),
        "profile_dataset": str(CORRECTED_ALL_PATH.name),
        "companies": companies,
        "available_codes": sorted(metadata_by_code),
        "qualitative_codes": qualitative_codes,
        "preference_function": "t1",
        "default_indifference_threshold": 0.0,
        "default_preference_threshold": 0.0,
        "qualitative_provider": "ollama",
        "default_ollama_model": "llama3.1:8b",
        "sensitivity_analysis": {
            "prompt_variants": [
                {"id": prompt_id, "label": PROMPT_VARIANT_LABELS[prompt_id], "template": template}
                for prompt_id, template in QUALITATIVE_PROMPT_VARIANTS.items()
            ],
            "default_temperatures": [0.0, 0.5, 1.0],
            "default_top_ps": [1.0, 0.9],
            "default_seed": 42,
            "default_repetitions": 2,
            "default_model_families": DEFAULT_SENSITIVITY_MODEL_FAMILIES,
        },
    }

if __name__ == "__main__":
    print(json.dumps(calculate_flowsort_profiles("semiconductors"), ensure_ascii=False, indent=2))
