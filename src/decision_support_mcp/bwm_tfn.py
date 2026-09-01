"""Group Hierarchical Fuzzy Best-Worst Method.

The comparison constraints use the product form |w~_B - a~_Bj (x) w~_j| <= xi
rather than the ratio form |w~_B / w~_j - a~_Bj| <= xi. Both source papers do the
same: the hierarchical FBWM paper states the ratio form as Model (5.1) and then
solves Model (5.2), and the group FBWM paper converts its Problem (1) into
Problem (2), each following Rezaei's linear BWM. The reason is that the product
form is linear in the weights and the deviations together, so the model is a
linear program with a single optimal weight vector; the ratio form is bilinear
and leaves a face of equally optimal weight vectors, which makes the resulting
ranking depend on where the solver happens to land.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import linprog


class LT:
    """Linguistic term labels."""

    EI = "EI"
    WI = "WI"
    FI = "FI"
    VI = "VI"
    AI = "AI"


LINGUISTIC_SCALE: dict[str, tuple[float, float, float]] = {
    LT.EI: (1.0, 1.0, 1.0),
    LT.WI: (2 / 3, 1.0, 3 / 2),
    LT.FI: (3 / 2, 2.0, 5 / 2),
    LT.VI: (5 / 2, 3.0, 7 / 2),
    LT.AI: (7 / 2, 4.0, 9 / 2),
}

FEASIBILITY_TOLERANCE = 1e-7
WEIGHT_LOWER_BOUND = 1e-6
SPREAD_TIE_BREAK_SLACK = 1e-9

DOMAIN_INFO: dict[str, dict[str, str]] = {
    "E": {"label": "Environmental", "sasb_domain": "Environmental"},
    "S": {"label": "Social", "sasb_domain": "Social"},
    "G": {"label": "Governance", "sasb_domain": "Governance"},
}


class TFN(NamedTuple):
    l: float
    m: float
    u: float

    def graded_mean(self) -> float:
        return (self.l + 4 * self.m + self.u) / 6


def gmir(tfn: TFN) -> float:
    return tfn.graded_mean()


def linguistic_to_tfn(value: str) -> TFN:
    key = str(value).upper().strip()
    if key not in LINGUISTIC_SCALE:
        valid = ", ".join(LINGUISTIC_SCALE)
        raise ValueError(f"Linguistic term must be one of {valid}; got {value!r}.")
    return TFN(*LINGUISTIC_SCALE[key])


def _rounded_tfn(tfn: TFN) -> dict[str, float]:
    return {"l": round(tfn.l, 6), "m": round(tfn.m, 6), "u": round(tfn.u, 6)}


def _rounded(value: float) -> float:
    if math.isinf(value) or math.isnan(value):
        raise ValueError("Non-finite value cannot be rounded safely.")
    return round(value, 6)


def _safe_numeric_or_null(value: float) -> float | None:
    if math.isinf(value) or math.isnan(value):
        return None
    return round(value, 6)


def _tfn_from_solution(x: np.ndarray, start: int) -> TFN:
    return TFN(float(x[start]), float(x[start + 1]), float(x[start + 2]))


def _tfn_product(left: TFN, right: TFN) -> TFN:
    return TFN(left.l * right.l, left.m * right.m, left.u * right.u)


def _public_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


class _LinearProgram:
    """Accumulates the rows of a linear program in ``linprog`` form."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.upper_bound_rows: list[np.ndarray] = []
        self.upper_bound_values: list[float] = []
        self.equality_rows: list[np.ndarray] = []
        self.equality_values: list[float] = []

    def row(self) -> np.ndarray:
        return np.zeros(self.size, dtype=float)

    def add_upper_bound(self, row: np.ndarray, value: float = 0.0) -> None:
        self.upper_bound_rows.append(row)
        self.upper_bound_values.append(value)

    def add_equality(self, row: np.ndarray, value: float) -> None:
        self.equality_rows.append(row)
        self.equality_values.append(value)

    def matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.upper_bound_rows, dtype=float),
            np.array(self.upper_bound_values, dtype=float),
            np.array(self.equality_rows, dtype=float),
            np.array(self.equality_values, dtype=float),
        )

    def residuals(self, x: np.ndarray) -> tuple[float, float]:
        upper_rows, upper_values, equality_rows, equality_values = self.matrices()

        equality_residual = 0.0
        if equality_rows.size:
            equality_residual = float(np.max(np.abs(equality_rows @ x - equality_values)))

        inequality_violation = 0.0
        if upper_rows.size:
            inequality_violation = float(
                np.max(np.maximum(0.0, upper_rows @ x - upper_values))
            )

        if not math.isfinite(equality_residual) or not math.isfinite(inequality_violation):
            return math.inf, math.inf
        return equality_residual, inequality_violation


class _LinearProgramSolution(NamedTuple):
    result: Any
    x: np.ndarray
    max_equality_residual: float
    max_inequality_violation: float


def _append_tfn_order_constraints(program: _LinearProgram, index: int) -> None:
    """Enforce l <= m <= u for the fuzzy weight stored at ``index``."""

    row = program.row()
    row[index] = 1.0
    row[index + 1] = -1.0
    program.add_upper_bound(row)

    row = program.row()
    row[index + 1] = 1.0
    row[index + 2] = -1.0
    program.add_upper_bound(row)


def _append_product_form_constraints(
    program: _LinearProgram,
    numerator_index: int,
    denominator_index: int,
    preference_tfn: TFN,
    deviation_index: int,
) -> None:
    """Bound |w~_numerator - a~ (x) w~_denominator| by the deviation variable.

    This is the product form of the comparison constraint: Model (5.2) of the
    hierarchical FBWM paper, and Problem (2) of the group FBWM paper, both of
    which reach it from the ratio form by way of Rezaei's linear BWM. Since
    a~ (x) w~ = (l_a*l_w, m_a*m_w, u_a*u_w), bounding the deviation component by
    component keeps every row linear in the weights *and* in the deviation, so
    the complete model is a linear program with a single optimal weight vector.

    The ratio form |w~_B / w~_j - a~_Bj| <= xi is bilinear instead, and leaves a
    whole face of equally optimal weight vectors for the solver to pick from.
    """

    for offset, preference in enumerate(preference_tfn):
        row = program.row()
        row[numerator_index + offset] += 1.0
        row[denominator_index + offset] -= preference
        row[deviation_index] -= 1.0
        program.add_upper_bound(row)

        row = program.row()
        row[numerator_index + offset] -= 1.0
        row[denominator_index + offset] += preference
        row[deviation_index] -= 1.0
        program.add_upper_bound(row)


def _append_gmir_normalisation(program: _LinearProgram, indices: list[int]) -> None:
    """Enforce sum of R(w~) over ``indices`` equal to one."""

    row = program.row()
    for index in indices:
        row[index] += 1.0 / 6.0
        row[index + 1] += 4.0 / 6.0
        row[index + 2] += 1.0 / 6.0
    program.add_equality(row, 1.0)


def _append_fixed_tfn(program: _LinearProgram, index: int, tfn: TFN) -> None:
    """Pin the fuzzy weight at ``index`` to a known triangular fuzzy number."""

    for offset, value in enumerate(tfn):
        row = program.row()
        row[index + offset] = 1.0
        program.add_equality(row, value)


def _solve_linear_program(
    program: _LinearProgram,
    objective: np.ndarray,
    bounds: list[tuple[float, float | None]],
    description: str,
) -> _LinearProgramSolution:
    upper_rows, upper_values, equality_rows, equality_values = program.matrices()
    result = linprog(
        objective,
        A_ub=upper_rows,
        b_ub=upper_values,
        A_eq=equality_rows,
        b_eq=equality_values,
        bounds=bounds,
        method="highs",
    )

    if not getattr(result, "success", False) or result.x is None:
        message = getattr(result, "message", "no message")
        raise RuntimeError(f"{description} linear program did not solve: {message}")

    x = np.asarray(result.x, dtype=float)
    if not np.all(np.isfinite(x)) or not math.isfinite(float(result.fun)):
        raise RuntimeError(f"{description} linear program returned a non-finite solution.")

    max_equality_residual, max_inequality_violation = program.residuals(x)
    if (
        max_equality_residual > FEASIBILITY_TOLERANCE
        or max_inequality_violation > FEASIBILITY_TOLERANCE
    ):
        raise RuntimeError(
            f"{description} linear program result violates constraints: "
            f"maximum equality residual={max_equality_residual:.3e}, "
            f"maximum inequality violation={max_inequality_violation:.3e}."
        )

    return _LinearProgramSolution(
        result, x, max_equality_residual, max_inequality_violation
    )


def _minimise_fuzzy_spread(
    program: _LinearProgram,
    deviation_objective: np.ndarray,
    weight_indices: list[int],
    bounds: list[tuple[float, float | None]],
    solution: _LinearProgramSolution,
    description: str,
) -> _LinearProgramSolution:
    """Break ties between equally optimal weightings by least total fuzzy spread.

    Both papers assert that the product form has a unique optimum, and for
    judgements carrying any inconsistency it does. Once the comparisons are
    consistent enough for the deviation to reach zero, though, the objective stops
    discriminating between weightings and a whole face of them is optimal -- an LP
    solver then returns an arbitrary vertex of it, typically with every l pushed
    down onto its lower bound, which is a valid but degenerate answer. Re-solving
    for the smallest total spread sum(u - l) at the optimal deviation picks the
    tightest weighting on that face, and is a no-op wherever the optimum is
    already unique.
    """

    refined = _LinearProgram(program.size)
    refined.upper_bound_rows = list(program.upper_bound_rows)
    refined.upper_bound_values = list(program.upper_bound_values)
    refined.equality_rows = list(program.equality_rows)
    refined.equality_values = list(program.equality_values)
    refined.add_upper_bound(
        deviation_objective.copy(),
        float(deviation_objective @ solution.x) + SPREAD_TIE_BREAK_SLACK,
    )

    spread_objective = np.zeros(program.size, dtype=float)
    for index in weight_indices:
        spread_objective[index] -= 1.0
        spread_objective[index + 2] += 1.0

    try:
        return _solve_linear_program(refined, spread_objective, bounds, description)
    except RuntimeError:
        return solution


def _optimizer_diagnostics(solution: _LinearProgramSolution) -> dict[str, Any]:
    return {
        "model_form": "product",
        "tie_break": "minimum total fuzzy spread at the optimal deviation",
        "maximum_equality_residual": _rounded(solution.max_equality_residual),
        "maximum_inequality_violation": _rounded(solution.max_inequality_violation),
        "optimizer_method": "linprog/highs",
        "optimizer_success": bool(solution.result.success),
        "optimizer_message": str(solution.result.message),
        "optimizer_iterations": int(getattr(solution.result, "nit", 0) or 0),
    }


def _normalize_decision_makers(decision_makers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not decision_makers:
        raise ValueError("At least one decision maker is required.")

    normalized: list[dict[str, Any]] = []
    ids_seen: set[str] = set()
    total = 0.0
    for index, dm in enumerate(decision_makers, start=1):
        dm_id = str(dm.get("id", "")).strip()
        if not dm_id:
            raise ValueError(f"Decision maker #{index} is missing an id.")
        if dm_id in ids_seen:
            raise ValueError(f"Decision maker id {dm_id!r} is duplicated.")
        ids_seen.add(dm_id)

        weight = float(dm.get("weight", 0.0))
        if weight <= 0:
            raise ValueError(f"Decision maker {dm_id!r} must have a positive weight.")

        normalized.append(
            {
                "id": dm_id,
                "name": str(dm.get("name") or dm_id),
                "weight": weight,
            }
        )
        total += weight

    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"Decision maker weights must sum to 1. Received {total:.6f}.")
    return normalized


def _normalize_elements(elements: list[dict[str, Any]], level_name: str) -> list[dict[str, Any]]:
    return _normalize_elements_with_minimum(elements, level_name, minimum=2)


def _normalize_elements_with_minimum(
    elements: list[dict[str, Any]],
    level_name: str,
    minimum: int,
) -> list[dict[str, Any]]:
    if len(elements) < minimum:
        requirement = "one element" if minimum == 1 else f"at least {minimum} elements"
        raise ValueError(f"{level_name} requires {requirement}.")

    normalized: list[dict[str, Any]] = []
    codes_seen: set[str] = set()
    for index, element in enumerate(elements, start=1):
        code = str(element.get("code", "")).strip()
        if not code:
            raise ValueError(f"{level_name} element #{index} is missing a code.")
        if code in codes_seen:
            raise ValueError(f"{level_name} contains duplicated code {code!r}.")
        codes_seen.add(code)

        normalized.append(
            {
                **element,
                "code": code,
                "label": str(
                    element.get("label")
                    or element.get("name")
                    or element.get("measure")
                    or code
                ),
            }
        )
    return normalized


def _solve_singleton_level(
    level_name: str,
    element: dict[str, Any],
    decision_makers: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_dms = _normalize_decision_makers(decision_makers)
    normalized_element = _normalize_elements_with_minimum([element], level_name, minimum=1)[0]

    weighted_element = {
        **normalized_element,
        "fuzzy_weight": _rounded_tfn(TFN(1.0, 1.0, 1.0)),
        "crisp_weight": 1.0,
    }
    # A lone element is compared against itself, so its implied best-to-worst
    # term is EI and its deviation is zero by construction.
    diagnostics = [
        {
            "decision_maker_id": dm["id"],
            "decision_maker_name": dm["name"],
            "decision_maker_weight": _rounded(dm["weight"]),
            "best_code": normalized_element["code"],
            "worst_code": normalized_element["code"],
            "best_to_worst_term": LT.EI,
            "xi_star": 0.0,
        }
        for dm in normalized_dms
    ]
    return {
        "level_name": level_name,
        "weights": [weighted_element],
        "elements": [weighted_element],
        "ranking": [
            {
                "rank": 1,
                "code": normalized_element["code"],
                "label": normalized_element["label"],
                "crisp_weight": 1.0,
            }
        ],
        "decision_maker_results": diagnostics,
        "decision_makers": diagnostics,
        "crisp_weight_sum": 1.0,
        "weighted_deviation": 0.0,
    }


def _normalize_term_map(
    values: dict[str, Any],
    element_codes: list[str],
    comparison_label: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for code in element_codes:
        if code not in values:
            raise ValueError(f"Missing {comparison_label} comparison for {code}.")
        normalized[code] = str(values[code]).upper().strip()
        linguistic_to_tfn(normalized[code])
    return normalized


def _normalize_level_comparisons(
    level_name: str,
    element_codes: list[str],
    decision_makers: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(comparisons) != len(decision_makers):
        raise ValueError(
            f"{level_name} must provide exactly one comparison set for each decision maker."
        )

    by_id = {dm["id"]: dm for dm in decision_makers}
    normalized_by_id: dict[str, dict[str, Any]] = {}

    for comparison in comparisons:
        dm_id = str(comparison.get("decision_maker_id", "")).strip()
        if dm_id not in by_id:
            raise ValueError(f"{level_name} comparison references unknown decision maker {dm_id!r}.")
        if dm_id in normalized_by_id:
            raise ValueError(f"{level_name} contains duplicated comparisons for {dm_id!r}.")

        best_code = str(comparison.get("best_code", "")).strip()
        worst_code = str(comparison.get("worst_code", "")).strip()
        if best_code not in element_codes:
            raise ValueError(f"{level_name} best code {best_code!r} is not part of the active set.")
        if worst_code not in element_codes:
            raise ValueError(f"{level_name} worst code {worst_code!r} is not part of the active set.")
        if best_code == worst_code:
            raise ValueError(f"{level_name} best and worst codes must be different for {dm_id!r}.")

        best_to_others = _normalize_term_map(
            comparison.get("best_to_others", {}),
            element_codes,
            f"{level_name} Best-to-Others",
        )
        others_to_worst = _normalize_term_map(
            comparison.get("others_to_worst", {}),
            element_codes,
            f"{level_name} Others-to-Worst",
        )

        if best_to_others[best_code] != LT.EI:
            raise ValueError(
                f"{level_name} Best-to-Others comparison for the best code "
                f"{best_code!r} must be EI for {dm_id!r}."
            )
        if others_to_worst[worst_code] != LT.EI:
            raise ValueError(
                f"{level_name} Others-to-Worst comparison for the worst code "
                f"{worst_code!r} must be EI for {dm_id!r}."
            )

        bw_from_best_vector = best_to_others[worst_code]
        bw_from_worst_vector = others_to_worst[best_code]

        if bw_from_best_vector == LT.EI:
            raise ValueError(
                f"{level_name} Best-to-Worst comparison cannot be EI for {dm_id!r}."
            )
        if bw_from_best_vector != bw_from_worst_vector:
            raise ValueError(
                f"{level_name} has inconsistent Best-to-Worst comparisons "
                f"for decision maker {dm_id!r}: {bw_from_best_vector!r} "
                f"and {bw_from_worst_vector!r}."
            )

        normalized_by_id[dm_id] = {
            "decision_maker_id": dm_id,
            "best_code": best_code,
            "worst_code": worst_code,
            "best_to_others": best_to_others,
            "others_to_worst": others_to_worst,
            "bw_term": bw_from_best_vector,
            "bw_tfn": linguistic_to_tfn(bw_from_best_vector),
        }

    return [normalized_by_id[dm["id"]] for dm in decision_makers]


def solve_ghfbwm_level(
    level_name: str,
    elements: list[dict[str, Any]],
    decision_makers: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_elements = _normalize_elements(elements, level_name)
    normalized_dms = _normalize_decision_makers(decision_makers)
    element_codes = [element["code"] for element in normalized_elements]
    code_to_index = {code: index for index, code in enumerate(element_codes)}
    normalized_comparisons = _normalize_level_comparisons(
        level_name, element_codes, normalized_dms, comparisons
    )

    n = len(normalized_elements)
    k_count = len(normalized_dms)
    program = _LinearProgram(3 * n + k_count)

    for d, comparison in enumerate(normalized_comparisons):
        deviation_index = 3 * n + d
        best_index = 3 * code_to_index[comparison["best_code"]]
        worst_index = 3 * code_to_index[comparison["worst_code"]]

        for code in element_codes:
            element_index = 3 * code_to_index[code]
            _append_product_form_constraints(
                program,
                best_index,
                element_index,
                linguistic_to_tfn(comparison["best_to_others"][code]),
                deviation_index,
            )
            _append_product_form_constraints(
                program,
                element_index,
                worst_index,
                linguistic_to_tfn(comparison["others_to_worst"][code]),
                deviation_index,
            )

    _append_gmir_normalisation(program, [3 * j for j in range(n)])
    for j in range(n):
        _append_tfn_order_constraints(program, 3 * j)

    objective = np.zeros(3 * n + k_count, dtype=float)
    for d, dm in enumerate(normalized_dms):
        objective[3 * n + d] = dm["weight"]

    bounds: list[tuple[float, float | None]] = [(WEIGHT_LOWER_BOUND, None)] * (3 * n)
    bounds += [(0.0, None)] * k_count

    weight_indices = [3 * j for j in range(n)]
    solution = _solve_linear_program(program, objective, bounds, level_name)
    solution = _minimise_fuzzy_spread(
        program, objective, weight_indices, bounds, solution, level_name
    )

    x = solution.x
    fuzzy_weights = [_tfn_from_solution(x, index) for index in weight_indices]
    crisp_weights = [gmir(weight) for weight in fuzzy_weights]
    crisp_total = sum(crisp_weights)
    if crisp_total <= 0:
        raise RuntimeError(f"{level_name} produced non-positive total crisp weight.")

    weighted_elements: list[dict[str, Any]] = []
    for element, fuzzy_weight, crisp_weight in zip(normalized_elements, fuzzy_weights, crisp_weights):
        weighted_elements.append(
            {
                **element,
                "fuzzy_weight": _rounded_tfn(fuzzy_weight),
                "crisp_weight": round(crisp_weight, 6),
                "_raw_crisp_weight": crisp_weight,
            }
        )
    public_weighted_elements = [_public_fields(item) for item in weighted_elements]

    diagnostics: list[dict[str, Any]] = []
    weighted_deviation = 0.0
    for d, (dm, comparison) in enumerate(zip(normalized_dms, normalized_comparisons)):
        xi_star = float(x[3 * n + d])
        weighted_deviation += dm["weight"] * xi_star
        diagnostics.append(
            {
                "decision_maker_id": dm["id"],
                "decision_maker_name": dm["name"],
                "decision_maker_weight": _rounded(dm["weight"]),
                "best_code": comparison["best_code"],
                "worst_code": comparison["worst_code"],
                "best_to_worst_term": comparison["bw_term"],
                "xi_star": _rounded(xi_star),
            }
        )

    ranking = sorted(
        (
            {
                "rank": index + 1,
                "code": element["code"],
                "label": element["label"],
                "crisp_weight": element["crisp_weight"],
            }
            for index, element in enumerate(
                sorted(weighted_elements, key=lambda item: (-item["_raw_crisp_weight"], item["code"]))
            )
        ),
        key=lambda item: item["rank"],
    )

    crisp_sum = sum(item["_raw_crisp_weight"] for item in weighted_elements)

    return {
        "level_name": level_name,
        "weights": public_weighted_elements,
        "elements": public_weighted_elements,
        "ranking": ranking,
        "decision_maker_results": diagnostics,
        "decision_makers": diagnostics,
        "crisp_weight_sum": _rounded(crisp_sum),
        "weighted_deviation": _rounded(weighted_deviation),
        "optimization_diagnostics": _optimizer_diagnostics(solution),
    }


def solve_gh_fbwm_hierarchy(
    sector_id: str,
    top_level_criteria: list[str],
    subcriteria_by_parent: dict[str, list[dict[str, Any]]],
    decision_makers: list[dict[str, Any]],
    criteria_level_comparisons: list[dict[str, Any]],
    subcriteria_level_comparisons: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    active_top_codes: list[str] = []
    seen_top_codes: set[str] = set()
    for code in top_level_criteria:
        normalized_code = str(code).strip()
        if not normalized_code:
            raise ValueError("Top-level criteria must use non-empty topic names.")
        if normalized_code in seen_top_codes:
            raise ValueError(f"Top-level criteria contains duplicated code {normalized_code!r}.")
        seen_top_codes.add(normalized_code)
        active_top_codes.append(normalized_code)

    if len(active_top_codes) < 2:
        raise ValueError("At least two top-level criteria must remain active.")

    normalized_dms = _normalize_decision_makers(decision_makers)
    top_level_elements = _normalize_elements(
        [{"code": code, "label": code} for code in active_top_codes],
        "Criteria Level",
    )
    criteria_codes = [element["code"] for element in top_level_elements]
    criteria_comparisons = _normalize_level_comparisons(
        "Criteria Level",
        criteria_codes,
        normalized_dms,
        criteria_level_comparisons,
    )

    subcriteria_elements_by_parent: dict[str, list[dict[str, Any]]] = {}
    subcriteria_comparisons_by_parent: dict[str, list[dict[str, Any]]] = {}
    for parent_code in active_top_codes:
        if parent_code not in subcriteria_by_parent:
            raise ValueError(f"Missing indicator group for top-level topic {parent_code}.")

        elements = _normalize_elements_with_minimum(
            subcriteria_by_parent[parent_code],
            f"Subcriteria Level {parent_code}",
            minimum=1,
        )
        subcriteria_elements_by_parent[parent_code] = elements
        if len(elements) > 1:
            if parent_code not in subcriteria_level_comparisons:
                raise ValueError(f"Missing indicator comparisons for top-level topic {parent_code}.")
            element_codes = [element["code"] for element in elements]
            subcriteria_comparisons_by_parent[parent_code] = _normalize_level_comparisons(
                f"Subcriteria Level {parent_code}",
                element_codes,
                normalized_dms,
                subcriteria_level_comparisons[parent_code],
            )

    criteria_label_by_code = {item["code"]: item["label"] for item in top_level_elements}

    cursor = 0
    criteria_weight_index: dict[str, int] = {}
    for code in active_top_codes:
        criteria_weight_index[code] = cursor
        cursor += 3

    subcriteria_weight_index: dict[tuple[str, str], int] = {}
    for parent_code in active_top_codes:
        for element in subcriteria_elements_by_parent[parent_code]:
            subcriteria_weight_index[(parent_code, element["code"])] = cursor
            cursor += 3

    criteria_deviation_index: dict[str, int] = {}
    for dm in normalized_dms:
        criteria_deviation_index[dm["id"]] = cursor
        cursor += 1

    subcriteria_deviation_index: dict[tuple[str, str], int] = {}
    for parent_code in active_top_codes:
        if len(subcriteria_elements_by_parent[parent_code]) <= 1:
            continue
        for dm in normalized_dms:
            subcriteria_deviation_index[(parent_code, dm["id"])] = cursor
            cursor += 1

    program = _LinearProgram(cursor)

    for d, comparison in enumerate(criteria_comparisons):
        dm = normalized_dms[d]
        deviation_index = criteria_deviation_index[dm["id"]]
        best_index = criteria_weight_index[comparison["best_code"]]
        worst_index = criteria_weight_index[comparison["worst_code"]]
        for code in active_top_codes:
            _append_product_form_constraints(
                program,
                best_index,
                criteria_weight_index[code],
                linguistic_to_tfn(comparison["best_to_others"][code]),
                deviation_index,
            )
            _append_product_form_constraints(
                program,
                criteria_weight_index[code],
                worst_index,
                linguistic_to_tfn(comparison["others_to_worst"][code]),
                deviation_index,
            )

    for parent_code in active_top_codes:
        elements = subcriteria_elements_by_parent[parent_code]
        if len(elements) <= 1:
            continue
        element_codes = [element["code"] for element in elements]
        comparisons = subcriteria_comparisons_by_parent[parent_code]
        for d, comparison in enumerate(comparisons):
            dm = normalized_dms[d]
            deviation_index = subcriteria_deviation_index[(parent_code, dm["id"])]
            best_index = subcriteria_weight_index[(parent_code, comparison["best_code"])]
            worst_index = subcriteria_weight_index[(parent_code, comparison["worst_code"])]
            for code in element_codes:
                _append_product_form_constraints(
                    program,
                    best_index,
                    subcriteria_weight_index[(parent_code, code)],
                    linguistic_to_tfn(comparison["best_to_others"][code]),
                    deviation_index,
                )
                _append_product_form_constraints(
                    program,
                    subcriteria_weight_index[(parent_code, code)],
                    worst_index,
                    linguistic_to_tfn(comparison["others_to_worst"][code]),
                    deviation_index,
                )

    _append_gmir_normalisation(
        program, [criteria_weight_index[code] for code in active_top_codes]
    )
    for parent_code in active_top_codes:
        elements = subcriteria_elements_by_parent[parent_code]
        if len(elements) == 1:
            # Step 2: a lone sub-criterion takes the local fuzzy weight (1, 1, 1).
            _append_fixed_tfn(
                program,
                subcriteria_weight_index[(parent_code, elements[0]["code"])],
                TFN(1.0, 1.0, 1.0),
            )
            continue
        _append_gmir_normalisation(
            program,
            [
                subcriteria_weight_index[(parent_code, element["code"])]
                for element in elements
            ],
        )

    for code in active_top_codes:
        _append_tfn_order_constraints(program, criteria_weight_index[code])
    for parent_code in active_top_codes:
        for element in subcriteria_elements_by_parent[parent_code]:
            _append_tfn_order_constraints(
                program,
                subcriteria_weight_index[(parent_code, element["code"])],
            )

    objective = np.zeros(cursor, dtype=float)
    for dm in normalized_dms:
        objective[criteria_deviation_index[dm["id"]]] = dm["weight"]
        for parent_code in active_top_codes:
            index = subcriteria_deviation_index.get((parent_code, dm["id"]))
            if index is not None:
                objective[index] = dm["weight"]

    bounds: list[tuple[float, float | None]] = [(WEIGHT_LOWER_BOUND, None)] * cursor
    for index in criteria_deviation_index.values():
        bounds[index] = (0.0, None)
    for index in subcriteria_deviation_index.values():
        bounds[index] = (0.0, None)

    weight_indices = [criteria_weight_index[code] for code in active_top_codes]
    weight_indices += list(subcriteria_weight_index.values())

    solution = _solve_linear_program(program, objective, bounds, "Integrated HFBWM")
    solution = _minimise_fuzzy_spread(
        program, objective, weight_indices, bounds, solution, "Integrated HFBWM"
    )

    x = solution.x
    aggregate_deviation_star = float(objective @ x)

    def build_diagnostics(
        comparisons: list[dict[str, Any]],
        deviation_lookup: dict[str, int],
    ) -> tuple[list[dict[str, Any]], float]:
        """Step 7: the deviations are themselves the consistency indicators."""
        diagnostics: list[dict[str, Any]] = []
        weighted_deviation = 0.0
        for dm, comparison in zip(normalized_dms, comparisons):
            xi_star = float(x[deviation_lookup[dm["id"]]])
            weighted_deviation += dm["weight"] * xi_star
            diagnostics.append(
                {
                    "decision_maker_id": dm["id"],
                    "decision_maker_name": dm["name"],
                    "decision_maker_weight": _rounded(dm["weight"]),
                    "best_code": comparison["best_code"],
                    "worst_code": comparison["worst_code"],
                    "best_to_worst_term": comparison["bw_term"],
                    "xi_star": _rounded(xi_star),
                }
            )
        return diagnostics, weighted_deviation

    criteria_tfn_by_code = {
        code: _tfn_from_solution(x, criteria_weight_index[code])
        for code in active_top_codes
    }
    criteria_weight_by_code = {
        code: gmir(criteria_tfn_by_code[code])
        for code in active_top_codes
    }
    criteria_elements = []
    for element in top_level_elements:
        fuzzy_weight = criteria_tfn_by_code[element["code"]]
        criteria_elements.append(
            {
                **element,
                "fuzzy_weight": _rounded_tfn(fuzzy_weight),
                "crisp_weight": _rounded(criteria_weight_by_code[element["code"]]),
                "_raw_crisp_weight": criteria_weight_by_code[element["code"]],
            }
        )

    criteria_diagnostics, criteria_weighted_deviation = build_diagnostics(
        criteria_comparisons,
        criteria_deviation_index,
    )
    criteria_ranking = sorted(
        (
            {
                "rank": index + 1,
                "code": element["code"],
                "label": element["label"],
                "crisp_weight": element["crisp_weight"],
            }
            for index, element in enumerate(
                sorted(criteria_elements, key=lambda item: (-item["_raw_crisp_weight"], item["code"]))
            )
        ),
        key=lambda item: item["rank"],
    )
    public_criteria_elements = [_public_fields(element) for element in criteria_elements]
    criteria_level = {
        "level_name": "Criteria Level",
        "weights": public_criteria_elements,
        "elements": public_criteria_elements,
        "ranking": criteria_ranking,
        "decision_maker_results": criteria_diagnostics,
        "decision_makers": criteria_diagnostics,
        "crisp_weight_sum": _rounded(sum(criteria_weight_by_code.values())),
        "weighted_deviation": _rounded(criteria_weighted_deviation),
    }

    subcriteria_levels: dict[str, dict[str, Any]] = {}
    global_weights: list[dict[str, Any]] = []
    subcriteria_weighted_deviations: dict[str, float] = {}

    for parent_code in active_top_codes:
        elements = subcriteria_elements_by_parent[parent_code]
        local_tfn_by_code = {
            element["code"]: _tfn_from_solution(
                x,
                subcriteria_weight_index[(parent_code, element["code"])],
            )
            for element in elements
        }
        local_weight_by_code = {
            code: gmir(tfn)
            for code, tfn in local_tfn_by_code.items()
        }
        weighted_elements: list[dict[str, Any]] = []
        for element in elements:
            fuzzy_weight = local_tfn_by_code[element["code"]]
            weighted_elements.append(
                {
                    **element,
                    "fuzzy_weight": _rounded_tfn(fuzzy_weight),
                    "crisp_weight": _rounded(local_weight_by_code[element["code"]]),
                    "_raw_crisp_weight": local_weight_by_code[element["code"]],
                }
            )

        if len(elements) == 1:
            singleton_result = _solve_singleton_level(
                f"Subcriteria Level {parent_code}", elements[0], decision_makers
            )
            diagnostics = singleton_result["decision_maker_results"]
            weighted_deviation = 0.0
        else:
            diagnostics, weighted_deviation = build_diagnostics(
                subcriteria_comparisons_by_parent[parent_code],
                {
                    dm["id"]: subcriteria_deviation_index[(parent_code, dm["id"])]
                    for dm in normalized_dms
                },
            )

        ranking = sorted(
            (
                {
                    "rank": index + 1,
                    "code": element["code"],
                    "label": element["label"],
                    "crisp_weight": element["crisp_weight"],
                }
                for index, element in enumerate(
                    sorted(weighted_elements, key=lambda item: (-item["_raw_crisp_weight"], item["code"]))
                )
            ),
            key=lambda item: item["rank"],
        )
        public_weighted_elements = [_public_fields(element) for element in weighted_elements]
        level_result = {
            "level_name": f"Subcriteria Level {parent_code}",
            "weights": public_weighted_elements,
            "elements": public_weighted_elements,
            "ranking": ranking,
            "decision_maker_results": diagnostics,
            "decision_makers": diagnostics,
            "crisp_weight_sum": _rounded(sum(local_weight_by_code.values())),
            "weighted_deviation": _rounded(weighted_deviation),
        }
        subcriteria_levels[parent_code] = {
            "parent_code": parent_code,
            "parent_label": criteria_label_by_code[parent_code],
            **level_result,
        }
        subcriteria_weighted_deviations[parent_code] = weighted_deviation

        parent_weight = criteria_weight_by_code[parent_code]
        for element in elements:
            code = element["code"]
            global_tfn = _tfn_product(
                criteria_tfn_by_code[parent_code],
                local_tfn_by_code[code],
            )
            global_weight = parent_weight * local_weight_by_code[code]
            global_weights.append(
                {
                    "parent_code": parent_code,
                    "parent_label": criteria_label_by_code[parent_code],
                    "parent_crisp_weight": _rounded(parent_weight),
                    "code": code,
                    "label": element["label"],
                    "measure": element.get("measure", element["label"]),
                    "source": element.get("source", "sasb"),
                    "local_crisp_weight": _rounded(local_weight_by_code[code]),
                    "global_crisp_weight": round(global_weight, 6),
                    "_raw_crisp_weight": global_weight,
                    "fuzzy_weight": _rounded_tfn(local_tfn_by_code[code]),
                    "global_fuzzy_weight": _rounded_tfn(global_tfn),
                    "defuzzified_global_fuzzy_product": _rounded(gmir(global_tfn)),
                }
            )

    global_ranking = sorted(
        (
            {
                "rank": index + 1,
                **item,
            }
            for index, item in enumerate(
                sorted(global_weights, key=lambda entry: (-entry["_raw_crisp_weight"], entry["code"]))
            )
        ),
        key=lambda item: item["rank"],
    )
    public_global_weights = [_public_fields(item) for item in global_weights]
    public_global_ranking = [_public_fields(item) for item in global_ranking]

    raw_global_weight_sum = sum(item["_raw_crisp_weight"] for item in global_weights)

    # Step 7: in the linear model the deviations are themselves the consistency
    # indicators, so no ratio is formed. The consistency indices tabulated for
    # the ratio formulation are on a different scale and do not transfer, which
    # is why nothing here is divided by them. Groups are ordered worst first so
    # that the comparisons to revise are the ones at the top.
    comparison_group_deviations = sorted(
        (
            {
                "group": "criteria",
                "label": "Criteria Level",
                "weighted_deviation": _rounded(criteria_weighted_deviation),
            },
            *(
                {
                    "group": parent_code,
                    "label": criteria_label_by_code[parent_code],
                    "weighted_deviation": _rounded(
                        subcriteria_weighted_deviations[parent_code]
                    ),
                }
                for parent_code in active_top_codes
                if len(subcriteria_elements_by_parent[parent_code]) > 1
            ),
        ),
        key=lambda item: (-item["weighted_deviation"], item["group"]),
    )

    return {
        "sector_id": sector_id,
        "method": "integrated_group_hfbwm",
        "active_top_level_criteria": active_top_codes,
        "criteria_level": criteria_level,
        "subcriteria_levels": subcriteria_levels,
        "global_weights": public_global_weights,
        "global_ranking": public_global_ranking,
        "global_weight_sum": _rounded(raw_global_weight_sum),
        "integrated_model": {
            "objective": "min sum_d(lambda_d * xi_0d) + sum_j sum_d(lambda_d * xi_jd)",
            "aggregate_deviation_star": _rounded(aggregate_deviation_star),
            "criteria_weighted_deviation": _rounded(criteria_weighted_deviation),
            "subcriteria_weighted_deviations": {
                parent_code: _rounded(value)
                for parent_code, value in subcriteria_weighted_deviations.items()
            },
            "comparison_group_deviations": comparison_group_deviations,
            "interpretation": (
                "The optimal deviation is itself the consistency indicator. "
                "aggregate_deviation_star = 0 means fully consistent preferences; "
                "larger values indicate proportionally wider disagreement between "
                "the Best-to-Others and Others-to-Worst comparisons. Deviations "
                "carry the same units as the weights, so the consistency indices "
                "tabulated for the ratio formulation of fuzzy BWM are on a "
                "different scale and are deliberately not applied to them. "
                "comparison_group_deviations is ordered worst first: those groups, "
                "and within them the decision makers with the largest xi_star, are "
                "the comparisons to revise."
            ),
            **_optimizer_diagnostics(solution),
        },
    }
