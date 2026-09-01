"""Smoke tests for the MCP server resources and GH-FBWM tool."""

from __future__ import annotations

import asyncio
import csv
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.decision_support_mcp.flowsort import (  # noqa: E402
    GH_FBWM_STUDY_WEIGHTS,
    TYPE_A_PERCENTAGE,
    TYPE_B_ABSOLUTE,
    _build_profiles,
    _flowsort_details,
    flowsort_method,
    preference_degree,
    prepare_flowsort_inputs,
    qualify_partners_with_flowsort,
)
import src.decision_support_mcp.sensitivity_analysis as sensitivity_analysis  # noqa: E402
from src.decision_support_mcp.sensitivity_analysis import (  # noqa: E402
    _assign_sampled_categories,
    _flow_components_by_company,
    _sampled_flows,
    run_weight_monte_carlo_sensitivity,
)
from src.decision_support_mcp.quantitative_scores import (  # noqa: E402
    worst_attainable_value,
)
import src.decision_support_mcp.bwm_tfn as bwm_tfn  # noqa: E402
from src.decision_support_mcp.bwm_tfn import (  # noqa: E402
    FEASIBILITY_TOLERANCE,
    solve_gh_fbwm_hierarchy,
    solve_ghfbwm_level,
)
from src.decision_support_mcp.sasb import build_topic_hierarchy  # noqa: E402


def payload(result: Any) -> Any:
    if hasattr(result, "contents"):
        return json.loads(result.contents[0].text)
    if getattr(result, "isError", False):
        raise RuntimeError(result.content[0].text)
    return json.loads(result.content[0].text)


def tool_error_text(result: Any) -> str:
    if not getattr(result, "isError", False):
        raise AssertionError("Expected the MCP tool call to fail.")
    return result.content[0].text if result.content else ""


def group_codes(criteria: list[dict[str, Any]], topic: str, count: int) -> list[dict[str, Any]]:
    grouped = [criterion for criterion in criteria if criterion.get("topic") == topic]
    if len(grouped) < count:
        raise AssertionError(f"Expected at least {count} criteria in topic {topic}.")
    return grouped[:count]


def make_level_comparison(
    dm_id: str,
    best_code: str,
    worst_code: str,
    best_to_others: dict[str, str],
    others_to_worst: dict[str, str],
) -> dict[str, Any]:
    return {
        "decision_maker_id": dm_id,
        "best_code": best_code,
        "worst_code": worst_code,
        "best_to_others": best_to_others,
        "others_to_worst": others_to_worst,
    }


def weighted_deviation(level_result: dict[str, Any]) -> float:
    return sum(
        dm["decision_maker_weight"] * dm["xi_star"]
        for dm in level_result["decision_maker_results"]
    )


def assert_close(value: float, expected: float, *, tol: float = 2e-6) -> None:
    if not math.isclose(value, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"Expected {expected}, got {value}.")


def assert_raises_value_error(fn, expected_substring: str) -> None:
    try:
        fn()
    except ValueError as exc:
        if expected_substring not in str(exc):
            raise AssertionError(
                f"Expected ValueError containing {expected_substring!r}, got: {exc!r}"
            ) from exc
        return
    raise AssertionError("Expected ValueError but no exception was raised.")


def assert_raises_runtime_error(fn, expected_substring: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        if expected_substring not in str(exc):
            raise AssertionError(
                f"Expected RuntimeError containing {expected_substring!r}, got: {exc!r}"
            ) from exc
        return
    raise AssertionError("Expected RuntimeError but no exception was raised.")


def assert_finite_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int | float):
        if not math.isfinite(float(value)):
            raise AssertionError(f"Expected finite numeric value at {path}, got {value!r}.")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{path}[{index}]")


def assert_tfn_order(tfn: dict[str, float]) -> None:
    if not (0 < tfn["l"] <= tfn["m"] <= tfn["u"]):
        raise AssertionError(f"Invalid TFN order: {tfn!r}")


def repeated_comparisons(
    dm_ids: list[str],
    codes: list[str],
    best_code: str | None = None,
    worst_code: str | None = None,
    term: str = "FI",
) -> list[dict[str, Any]]:
    best = best_code or codes[0]
    worst = worst_code or codes[-1]
    if best == worst:
        raise AssertionError("Repeated comparison helper requires different best and worst codes.")
    return [
        make_level_comparison(
            dm_id,
            best,
            worst,
            {code: (term if code != best else "EI") for code in codes},
            {code: (term if code != worst else "EI") for code in codes},
        )
        for dm_id in dm_ids
    ]


def base_hfbwm_inputs() -> dict[str, Any]:
    decision_makers = [
        {"id": "dm-1", "name": "DM 1", "weight": 0.6},
        {"id": "dm-2", "name": "DM 2", "weight": 0.4},
    ]
    criteria_level_comparisons = [
        make_level_comparison(
            "dm-1",
            "C1",
            "C2",
            {"C1": "EI", "C2": "FI"},
            {"C1": "FI", "C2": "EI"},
        ),
        make_level_comparison(
            "dm-2",
            "C2",
            "C1",
            {"C1": "WI", "C2": "EI"},
            {"C1": "EI", "C2": "WI"},
        ),
    ]
    subcriteria_level_comparisons = {
        "C1": [
            make_level_comparison(
                "dm-1",
                "C1S1",
                "C1S2",
                {"C1S1": "EI", "C1S2": "FI"},
                {"C1S1": "FI", "C1S2": "EI"},
            ),
            make_level_comparison(
                "dm-2",
                "C1S2",
                "C1S1",
                {"C1S1": "WI", "C1S2": "EI"},
                {"C1S1": "EI", "C1S2": "WI"},
            ),
        ],
    }
    return {
        "sector_id": "unit-test",
        "top_level_criteria": ["C1", "C2"],
        "subcriteria_by_parent": {
            "C1": [
                {"code": "C1S1", "label": "C1 Sub 1"},
                {"code": "C1S2", "label": "C1 Sub 2"},
            ],
            "C2": [{"code": "C2S1", "label": "C2 Sub 1"}],
        },
        "decision_makers": decision_makers,
        "criteria_level_comparisons": criteria_level_comparisons,
        "subcriteria_level_comparisons": subcriteria_level_comparisons,
    }


def run_bwm_regression_checks() -> None:
    inputs = base_hfbwm_inputs()
    result = solve_gh_fbwm_hierarchy(**inputs)
    assert result["method"] == "integrated_group_hfbwm"
    assert_finite_tree(result)

    assert_close(result["criteria_level"]["crisp_weight_sum"], 1.0)
    for level in result["subcriteria_levels"].values():
        assert_close(level["crisp_weight_sum"], 1.0)
    assert_close(result["global_weight_sum"], 1.0)

    for item in result["criteria_level"]["weights"]:
        assert_tfn_order(item["fuzzy_weight"])
    for level in result["subcriteria_levels"].values():
        for item in level["weights"]:
            assert_tfn_order(item["fuzzy_weight"])
    for item in result["global_weights"]:
        assert_tfn_order(item["global_fuzzy_weight"])

    singleton_level = result["subcriteria_levels"]["C2"]
    singleton_weight = singleton_level["weights"][0]
    assert singleton_weight["fuzzy_weight"] == {"l": 1.0, "m": 1.0, "u": 1.0}
    assert singleton_weight["crisp_weight"] == 1.0
    assert singleton_level["weighted_deviation"] == 0.0

    integrated_model = result["integrated_model"]
    assert integrated_model["maximum_equality_residual"] <= FEASIBILITY_TOLERANCE
    assert integrated_model["maximum_inequality_violation"] <= FEASIBILITY_TOLERANCE
    decomposition = (
        integrated_model["criteria_weighted_deviation"]
        + sum(integrated_model["subcriteria_weighted_deviations"].values())
    )
    assert_close(
        integrated_model["aggregate_deviation_star"],
        decomposition,
        tol=5e-6,
    )

    # Step 7: the deviations are the consistency indicators; nothing is divided
    # by a consistency index, and the groups are ordered worst first.
    plus_codes = [
        parent_code
        for parent_code, level in result["subcriteria_levels"].items()
        if len(level["weights"]) > 1
    ]
    group_deviations = integrated_model["comparison_group_deviations"]
    assert [item["group"] for item in group_deviations][0] == max(
        group_deviations, key=lambda item: item["weighted_deviation"]
    )["group"]
    assert {item["group"] for item in group_deviations} == {"criteria", *plus_codes}
    assert [item["weighted_deviation"] for item in group_deviations] == sorted(
        (item["weighted_deviation"] for item in group_deviations), reverse=True
    )
    by_group = {item["group"]: item["weighted_deviation"] for item in group_deviations}
    assert_close(by_group["criteria"], round(weighted_deviation(result["criteria_level"]), 6))
    for parent_code in plus_codes:
        assert_close(
            by_group[parent_code],
            round(weighted_deviation(result["subcriteria_levels"][parent_code]), 6),
        )
    # No consistency index or ratio may survive anywhere in the payload.
    banned = ("consistency_index", "consistency_ratio", "group_fit_ratio",
              "consistency_status", "weighted_group_fit_ratio")
    def assert_no_banned(value, path="result"):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in banned:
                    raise AssertionError(f"{path}.{key} should have been removed.")
                assert_no_banned(item, f"{path}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                assert_no_banned(item, f"{path}[{i}]")
    assert_no_banned(result)

    all_diagnostics = [
        *result["criteria_level"]["decision_maker_results"],
        *[
            item
            for level in result["subcriteria_levels"].values()
            for item in level["decision_maker_results"]
        ],
    ]
    assert all("xi_star" in item for item in all_diagnostics)

    for collection in (
        result["criteria_level"]["weights"],
        result["global_weights"],
        result["global_ranking"],
    ):
        for item in collection:
            if any(key.startswith("_") for key in item):
                raise AssertionError(f"Private raw field leaked into result: {item!r}")

    level_result = solve_ghfbwm_level(
        "Unit Level",
        [{"code": "A", "label": "A"}, {"code": "B", "label": "B"}],
        [{"id": "dm-1", "name": "DM 1", "weight": 1.0}],
        [
            make_level_comparison(
                "dm-1",
                "A",
                "B",
                {"A": "EI", "B": "FI"},
                {"A": "FI", "B": "EI"},
            )
        ],
    )
    assert level_result["optimization_diagnostics"]["maximum_equality_residual"] <= FEASIBILITY_TOLERANCE
    assert level_result["optimization_diagnostics"]["maximum_inequality_violation"] <= FEASIBILITY_TOLERANCE

    bad_inputs = base_hfbwm_inputs()
    bad_inputs["decision_makers"] = [
        {"id": "dm-1", "name": "DM 1", "weight": 0.7},
        {"id": "dm-2", "name": "DM 2", "weight": 0.4},
    ]
    assert_raises_value_error(
        lambda: solve_gh_fbwm_hierarchy(**bad_inputs),
        "must sum to 1",
    )

    bad_inputs = base_hfbwm_inputs()
    bad_inputs["decision_makers"] = [
        {"id": "dm-1", "name": "DM 1", "weight": 0.5},
        {"id": "dm-1", "name": "DM 1 duplicate", "weight": 0.5},
    ]
    assert_raises_value_error(
        lambda: solve_gh_fbwm_hierarchy(**bad_inputs),
        "duplicated",
    )

    bad_inputs = base_hfbwm_inputs()
    bad_inputs["top_level_criteria"] = ["C1", "C1"]
    assert_raises_value_error(
        lambda: solve_gh_fbwm_hierarchy(**bad_inputs),
        "duplicated",
    )

    bad_inputs = base_hfbwm_inputs()
    bad_inputs["subcriteria_by_parent"]["C1"] = [
        {"code": "C1S1", "label": "C1 Sub 1"},
        {"code": "C1S1", "label": "C1 Sub 1 duplicate"},
    ]
    assert_raises_value_error(
        lambda: solve_gh_fbwm_hierarchy(**bad_inputs),
        "duplicated",
    )

    base_level_args = (
        "Validation Level",
        [{"code": "A", "label": "A"}, {"code": "B", "label": "B"}],
        [{"id": "dm-1", "name": "DM 1", "weight": 1.0}],
    )
    assert_raises_value_error(
        lambda: solve_ghfbwm_level(
            *base_level_args,
            [make_level_comparison("dm-1", "A", "A", {"A": "EI", "B": "FI"}, {"A": "FI", "B": "EI"})],
        ),
        "best and worst codes must be different",
    )
    assert_raises_value_error(
        lambda: solve_ghfbwm_level(
            *base_level_args,
            [make_level_comparison("dm-1", "A", "B", {"A": "WI", "B": "FI"}, {"A": "FI", "B": "EI"})],
        ),
        "Best-to-Others comparison for the best code",
    )
    assert_raises_value_error(
        lambda: solve_ghfbwm_level(
            *base_level_args,
            [make_level_comparison("dm-1", "A", "B", {"A": "EI", "B": "FI"}, {"A": "FI", "B": "WI"})],
        ),
        "Others-to-Worst comparison for the worst code",
    )
    assert_raises_value_error(
        lambda: solve_ghfbwm_level(
            *base_level_args,
            [make_level_comparison("dm-1", "A", "B", {"A": "EI", "B": "FI"}, {"A": "WI", "B": "EI"})],
        ),
        "inconsistent Best-to-Worst comparisons",
    )

    original_linprog = bwm_tfn.linprog
    level_comparison = [
        make_level_comparison("dm-1", "A", "B", {"A": "EI", "B": "FI"}, {"A": "FI", "B": "EI"})
    ]

    # The model is a linear program, so there is a single solve and no multistart
    # filtering. What still needs guarding is that a bad solver result is rejected
    # rather than reported as a weighting.
    def failing_linprog(*_args, **_kwargs):
        return SimpleNamespace(
            success=False,
            fun=0.0,
            x=None,
            message="infeasible",
            nit=1,
        )

    def non_finite_linprog(*_args, **_kwargs):
        return SimpleNamespace(
            success=True,
            fun=2.0,
            x=bwm_tfn.np.array([math.nan, 0.5, 0.5, 0.5, 0.5, 0.5, 10.0]),
            message="non-finite",
            nit=2,
        )

    def infeasible_point_linprog(*_args, **_kwargs):
        # Feasible-looking shape, but the weights do not satisfy the GMIR
        # normalisation, so the residual check must reject it.
        return SimpleNamespace(
            success=True,
            fun=2.0,
            x=bwm_tfn.np.array([0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 10.0]),
            message="off the feasible set",
            nit=3,
        )

    for patched, expected in (
        (failing_linprog, "linear program did not solve"),
        (non_finite_linprog, "returned a non-finite solution"),
        (infeasible_point_linprog, "violates constraints"),
    ):
        try:
            bwm_tfn.linprog = patched
            assert_raises_runtime_error(
                lambda: solve_ghfbwm_level(*base_level_args, level_comparison),
                expected,
            )
        finally:
            bwm_tfn.linprog = original_linprog

    try:
        bwm_tfn.linprog = failing_linprog
        assert_raises_runtime_error(
            lambda: solve_gh_fbwm_hierarchy(**base_hfbwm_inputs()),
            "Integrated HFBWM linear program did not solve",
        )
    finally:
        bwm_tfn.linprog = original_linprog

    # The product form has a unique optimum, so the same inputs must give the
    # same weights no matter how the problem is reached.
    repeat_a = solve_ghfbwm_level(*base_level_args, level_comparison)
    repeat_b = solve_ghfbwm_level(*base_level_args, level_comparison)
    assert repeat_a["weights"] == repeat_b["weights"]
    assert repeat_a["optimization_diagnostics"]["model_form"] == "product"
    assert repeat_a["optimization_diagnostics"]["maximum_equality_residual"] <= FEASIBILITY_TOLERANCE

    near_tie_x = bwm_tfn.np.array(
        [
            0.1234561,
            0.1234561,
            0.1234561,
            0.1234565,
            0.1234565,
            0.1234565,
            0.7530874,
            0.7530874,
            0.7530874,
            10.0,
        ]
    )

    def near_tie_linprog(*_args, **_kwargs):
        return SimpleNamespace(
            success=True,
            fun=10.0,
            x=near_tie_x.copy(),
            message="near tie",
            nit=1,
        )

    try:
        bwm_tfn.linprog = near_tie_linprog
        near_tie_result = solve_ghfbwm_level(
            "Near Tie Level",
            [{"code": "A", "label": "A"}, {"code": "B", "label": "B"}, {"code": "C", "label": "C"}],
            [{"id": "dm-1", "name": "DM 1", "weight": 1.0}],
            [
                make_level_comparison(
                    "dm-1",
                    "C",
                    "A",
                    {"A": "FI", "B": "FI", "C": "EI"},
                    {"A": "EI", "B": "FI", "C": "FI"},
                )
            ],
        )
    finally:
        bwm_tfn.linprog = original_linprog

    assert [item["code"] for item in near_tie_result["ranking"]] == ["C", "B", "A"]

    hierarchy_near_tie_x = bwm_tfn.np.array(
        [
            0.7530874,
            0.7530874,
            0.7530874,
            0.1234561,
            0.1234561,
            0.1234561,
            0.1234565,
            0.1234565,
            0.1234565,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            10.0,
        ]
    )

    def hierarchy_near_tie_linprog(*_args, **_kwargs):
        return SimpleNamespace(
            success=True,
            fun=10.0,
            x=hierarchy_near_tie_x.copy(),
            message="hierarchy near tie",
            nit=1,
        )

    try:
        bwm_tfn.linprog = hierarchy_near_tie_linprog
        hierarchy_near_tie = solve_gh_fbwm_hierarchy(
            sector_id="unit-test",
            top_level_criteria=["C", "A", "B"],
            subcriteria_by_parent={
                "C": [{"code": "CS", "label": "C singleton"}],
                "A": [{"code": "AS", "label": "A singleton"}],
                "B": [{"code": "BS", "label": "B singleton"}],
            },
            decision_makers=[{"id": "dm-1", "name": "DM 1", "weight": 1.0}],
            criteria_level_comparisons=[
                make_level_comparison(
                    "dm-1",
                    "C",
                    "A",
                    {"A": "FI", "B": "FI", "C": "EI"},
                    {"A": "EI", "B": "FI", "C": "FI"},
                )
            ],
            subcriteria_level_comparisons={},
        )
    finally:
        bwm_tfn.linprog = original_linprog

    assert [item["code"] for item in hierarchy_near_tie["criteria_level"]["ranking"]] == ["C", "B", "A"]
    assert [item["parent_code"] for item in hierarchy_near_tie["global_ranking"]] == ["C", "B", "A"]
    assert_close(hierarchy_near_tie["global_weight_sum"], 1.0)
    assert not math.isclose(
        sum(item["global_crisp_weight"] for item in hierarchy_near_tie["global_weights"]),
        hierarchy_near_tie["global_weight_sum"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )

    sasb_hierarchy = build_topic_hierarchy("semiconductors")
    eight_topics = [
        "Greenhouse Gas Emissions",
        "Energy Management in Manufacturing",
        "Water Management",
        "Waste Management",
        "Product Lifecycle Management",
        "Materials Sourcing",
        "Recruiting & Managing a Global & Skilled Workforce",
        "Intellectual Property Protection & Competitive Behaviour",
    ]
    sasb_dms = [
        {"id": "dm-1", "name": "DM 1", "weight": 0.5},
        {"id": "dm-2", "name": "DM 2", "weight": 0.3},
        {"id": "dm-3", "name": "DM 3", "weight": 0.2},
    ]
    sasb_dm_ids = [dm["id"] for dm in sasb_dms]
    sasb_inputs = {
        "sector_id": "semiconductors",
        "top_level_criteria": eight_topics,
        "subcriteria_by_parent": {
            topic: sasb_hierarchy[topic]
            for topic in eight_topics
        },
        "decision_makers": sasb_dms,
        "criteria_level_comparisons": repeated_comparisons(
            sasb_dm_ids,
            eight_topics,
            best_code="Greenhouse Gas Emissions",
            worst_code="Intellectual Property Protection & Competitive Behaviour",
        ),
        "subcriteria_level_comparisons": {
            topic: repeated_comparisons(
                sasb_dm_ids,
                [item["code"] for item in sasb_hierarchy[topic]],
            )
            for topic in eight_topics
            if len(sasb_hierarchy[topic]) > 1
        },
    }
    sasb_result = solve_gh_fbwm_hierarchy(**sasb_inputs)
    assert len(sasb_result["active_top_level_criteria"]) == 8
    assert sum(1 for topic in eight_topics if len(sasb_hierarchy[topic]) > 1) == 5
    assert sum(1 for topic in eight_topics if len(sasb_hierarchy[topic]) == 1) == 3
    assert len(sasb_result["criteria_level"]["decision_maker_results"]) == 3
    assert all(
        len(level["decision_maker_results"]) == 3
        for topic, level in sasb_result["subcriteria_levels"].items()
        if len(sasb_hierarchy[topic]) > 1
    )
    assert all(
        level["weights"][0]["fuzzy_weight"] == {"l": 1.0, "m": 1.0, "u": 1.0}
        for topic, level in sasb_result["subcriteria_levels"].items()
        if len(sasb_hierarchy[topic]) == 1
    )
    assert_close(sasb_result["criteria_level"]["crisp_weight_sum"], 1.0)
    for level in sasb_result["subcriteria_levels"].values():
        assert_close(level["crisp_weight_sum"], 1.0)
    assert_close(sasb_result["global_weight_sum"], 1.0)


def run_data_source_consistency_checks() -> None:
    """corrected_all.json defines the limiting profiles and measures_HITL.json
    scores the alternatives. Any record present in both has to agree, otherwise
    a company is compared against a boundary built from a different value than
    the one it is scored with. measures_HITL.json is the authoritative source.
    """

    import src.decision_support_mcp.quantitative_scores as qs

    corrected = {}
    for row in qs._load_corrected_all():
        key = (str(row.get("code")), qs._canonical_company_name(str(row.get("company"))))
        corrected[key] = row
    mismatches = []
    for row in qs._load_measures_hitl():
        key = (str(row.get("code")), qs._canonical_company_name(str(row.get("company"))))
        other = corrected.get(key)
        if other is None:
            continue
        left = qs._clean_numeric_value(row.get("value"))
        right = qs._clean_numeric_value(other.get("value"))
        if left is None and right is None:
            continue
        if left is None or right is None or not math.isclose(
            float(left), float(right), rel_tol=1e-9, abs_tol=1e-9
        ):
            mismatches.append(f"{key[0]} / {key[1]}: HITL={left!r} corrected_all={right!r}")
    if mismatches:
        raise AssertionError(
            "measures_HITL.json and corrected_all.json disagree on "
            + f"{len(mismatches)} record(s): " + "; ".join(sorted(mismatches))
        )


def run_flowsort_regression_checks() -> None:
    # (a) _build_profiles with absolute maximize
    absolute_max = _build_profiles([10, 20, 30, 40], TYPE_B_ABSOLUTE, "maximize")
    assert_close(float(absolute_max["r1"]), 40.0)
    assert_close(float(absolute_max["r2"]), 32.5)
    assert_close(float(absolute_max["r3"]), 17.5)
    # r4 sits strictly beyond the worst observation, not on it. FlowSort bounds a
    # category with a strict ``>`` at the bottom, so an alternative sitting
    # exactly on r4 matches no interval -- which is what the not-reported penalty
    # used to produce. Two steps of 1% of the range: 10 - 2*0.3.
    assert_close(float(absolute_max["r4"]), 9.4)
    assert float(absolute_max["r4"]) < 10.0

    # (b) percentage now follows the same rule as absolute, so that r1 and r4
    # are the observed extremes and Condition (2.1) holds by construction.
    percentage_min = _build_profiles([10, 20, 30, 40], TYPE_A_PERCENTAGE, "minimize")
    assert_close(float(percentage_min["r1"]), 10.0)
    assert_close(float(percentage_min["r2"]), 17.5)
    assert_close(float(percentage_min["r3"]), 32.5)
    assert_close(float(percentage_min["r4"]), 40.6)
    assert float(percentage_min["r4"]) > 40.0
    for values in ([10, 20, 30, 40], [1.0, 1.5, 99.0]):
        for direction in ("minimize", "maximize"):
            built = _build_profiles(values, TYPE_A_PERCENTAGE, direction)
            assert min(values) <= float(built["r1"]) <= max(values)
            sign = -1.0 if direction == "minimize" else 1.0
            # r1 weakly dominates every value: the assignment rule uses ``>=``
            # at the top, so a tie there is fine.
            assert all(sign * float(built["r1"]) >= sign * v for v in values)
            # Every value STRICTLY dominates r4, and so does the not-reported
            # penalty, which sits between the two.
            assert all(sign * v > sign * float(built["r4"]) for v in values)
            penalty = worst_attainable_value(values, TYPE_A_PERCENTAGE, direction)
            assert sign * min(values, key=lambda v: sign * v) > sign * penalty
            assert sign * penalty > sign * float(built["r4"])

    # (c) flowsort_method invalid mode
    assert_raises_value_error(
        lambda: flowsort_method(
            dataset=[[1.0], [2.0]],
            profiles=[[3.0], [2.0], [1.0], [0.0]],
            W=[1.0],
            Q=[0.0],
            S=[1.0],
            P=[1.0],
            F=["t1"],
            mode="invalid_mode",
            rule="net",
            verbose=False,
        ),
        'Only mode="limiting" is supported',
    )

    # (d) flowsort_method invalid rule
    assert_raises_value_error(
        lambda: flowsort_method(
            dataset=[[1.0], [2.0]],
            profiles=[[3.0], [2.0], [1.0], [0.0]],
            W=[1.0],
            Q=[0.0],
            S=[1.0],
            P=[1.0],
            F=["t1"],
            mode="limiting",
            rule="invalid_rule",
            verbose=False,
        ),
        "rule must be one of: net, positive, negative",
    )

    base_kwargs = {
        "measure_weights": {"TC-SC-110a.1.1": 1.0},
        "companies": ["Hitachi", "Applied Materials"],
    }

    # (e) preference_function=t3 with P=0
    assert_raises_value_error(
        lambda: qualify_partners_with_flowsort(
            preference_function="t3",
            preference_threshold=0,
            **base_kwargs,
        ),
        "Expected p > 0",
    )

    # (f) preference_function=t5 with P<=Q
    assert_raises_value_error(
        lambda: qualify_partners_with_flowsort(
            preference_function="t5",
            indifference_threshold=1,
            preference_threshold=1,
            **base_kwargs,
        ),
        "Expected p > q",
    )

    # (g) preference_function=t6 with S<=0
    assert_raises_value_error(
        lambda: qualify_partners_with_flowsort(
            preference_function="t6",
            s_threshold=0,
            **base_kwargs,
        ),
        "Expected s > 0",
    )

    # (h) invalid preference function should fail explicitly
    assert_raises_value_error(
        lambda: preference_degree(
            dataset=[[1.0], [2.0]],
            W=[1.0],
            Q=[0.0],
            S=[1.0],
            P=[1.0],
            F=["t9"],
        ),
        "Invalid preference function at criterion index 0",
    )

    # (i) vector length mismatch should fail explicitly
    assert_raises_value_error(
        lambda: flowsort_method(
            dataset=[[1.0, 2.0], [2.0, 3.0]],
            profiles=[[4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
            W=[1.0],
            Q=[0.0, 0.0],
            S=[1.0, 1.0],
            P=[1.0, 1.0],
            F=["t1", "t1"],
            mode="limiting",
            rule="net",
            verbose=False,
        ),
        "W length (1) must match the number of criteria (2)",
    )

    # (j) limiting mode requires exactly four profile rows
    assert_raises_value_error(
        lambda: flowsort_method(
            dataset=[[1.0], [2.0]],
            profiles=[[3.0], [2.0], [1.0]],
            W=[1.0],
            Q=[0.0],
            S=[1.0],
            P=[1.0],
            F=["t1"],
            mode="limiting",
            rule="net",
            verbose=False,
        ),
        "requires exactly 4 profile rows",
    )


def run_weight_monte_carlo_checks() -> None:
    prepared = prepare_flowsort_inputs()
    row_sums, column_sums = _flow_components_by_company(prepared)
    baseline_weights = np.asarray(prepared["weights_vector"], dtype=float)

    # (a) the vectorised path the Monte Carlo runs on must reproduce the shipped
    # FlowSort implementation on arbitrary weight vectors, not only on the
    # baseline: everything the analysis reports rests on that equivalence.
    rng = np.random.default_rng(7)
    sampled = rng.uniform(
        baseline_weights * 0.9, baseline_weights * 1.1, size=(100, len(baseline_weights))
    )
    positive, negative, net = _sampled_flows(sampled, row_sums, column_sums)
    fast_categories = _assign_sampled_categories(positive, negative, net, "net")
    largest_flow_gap = 0.0
    for index in range(sampled.shape[0]):
        reference = _flowsort_details(
            dataset=prepared["dataset_array"],
            profiles=prepared["profiles_matrix"],
            W=sampled[index],
            Q=prepared["q_vector"],
            S=prepared["s_vector"],
            P=prepared["p_vector"],
            F=prepared["preference_functions"],
            mode="limiting",
        )
        expected = np.asarray([item["net_category"] for item in reference], dtype=np.int16)
        assert np.array_equal(expected, fast_categories[index]), (
            f"Vectorised FlowSort disagreed with the reference on weight vector {index}."
        )
        largest_flow_gap = max(
            largest_flow_gap,
            float(np.max(np.abs(np.asarray([item["net_flow"] for item in reference]) - net[index, :, -1]))),
        )
    # The reference flows are rounded to six decimals on the way out.
    assert largest_flow_gap < 1e-6, largest_flow_gap

    # (b) a scaled weight vector must classify identically, which is what lets
    # the draws be correlated against either the raw or the normalised weights.
    scaled = _assign_sampled_categories(
        *_sampled_flows(sampled * 3.5, row_sums, column_sums), "net"
    )
    assert np.array_equal(scaled, fast_categories)

    result = run_weight_monte_carlo_sensitivity(
        samples=400, perturbation_pct=0.10, seed=11, write_csv=False
    )
    assert result["validation"]["vectorised_path_matches_reference_categories"] is True
    assert result["monte_carlo"]["samples"] == 400
    assert result["csv_output"]["written"] is False

    baseline = qualify_partners_with_flowsort()
    baseline_by_company = {
        item["company"]: item["assigned_category"] for item in baseline["companies"]
    }
    for company in result["companies"]:
        assert company["baseline_category"] == baseline_by_company[company["company"]]
        assert_close(sum(company["class_probability_pct"].values()), 100.0, tol=1e-6)
        assert 0.0 <= company["stability_pct"] <= 100.0
        assert company["net_flow"]["p2_5"] <= company["net_flow"]["p50"] <= company["net_flow"]["p97_5"]

    assert result["contributions_are_defined"] is True
    assert_close(result["contribution_pct_total"], 100.0, tol=1e-4)
    assert_close(
        sum(item["contribution_pct"] for item in result["criterion_contributions"]),
        100.0,
        tol=1e-4,
    )

    # (b2) with a single criterion the weight cancels out of FlowSort, so no draw
    # can change anything. The run must say so rather than emit a contribution
    # column that silently fails to add up to 100%.
    degenerate = run_weight_monte_carlo_sensitivity(
        measure_weights={"TC-SC-440a.1": 0.2}, samples=100, seed=3, write_csv=False
    )
    assert degenerate["stability"]["identical_classification_pct"] == 100.0
    assert degenerate["contributions_are_defined"] is False
    assert degenerate["contribution_pct_total"] == 0.0
    assert any("contributions_are_defined is false" in note for note in degenerate["analysis_notes"])

    # (b3) convergence checkpoints must be distinct: on a short run the fractions
    # collapse onto the same prefix and would otherwise repeat identical rows.
    for sample_count in (2, 10, 100):
        checkpoints = [
            item["samples"]
            for item in run_weight_monte_carlo_sensitivity(
                samples=sample_count, seed=3, write_csv=False
            )["convergence"]
        ]
        assert checkpoints == sorted(set(checkpoints)), (sample_count, checkpoints)
        assert checkpoints[-1] == sample_count, (sample_count, checkpoints)
    contributions = [item["contribution_pct"] for item in result["criterion_contributions"]]
    assert contributions == sorted(contributions, reverse=True)
    for item in result["criterion_contributions"]:
        lower = item["sampled_weight"]["lower_bound"]
        upper = item["sampled_weight"]["upper_bound"]
        assert_close(lower, item["baseline_weight"] * 0.9, tol=1e-6)
        assert_close(upper, item["baseline_weight"] * 1.1, tol=1e-6)
        assert lower <= item["sampled_weight"]["sampled_min"]
        assert item["sampled_weight"]["sampled_max"] <= upper

    # (c) the same seed must reproduce the run exactly, and a different seed must
    # not change the conclusions beyond sampling noise.
    repeated = run_weight_monte_carlo_sensitivity(
        samples=400, perturbation_pct=0.10, seed=11, write_csv=False
    )
    assert repeated["stability"] == result["stability"]
    other_seed = run_weight_monte_carlo_sensitivity(
        samples=400, perturbation_pct=0.10, seed=12, write_csv=False
    )
    assert (
        result["criterion_contributions"][0]["code"]
        == other_seed["criterion_contributions"][0]["code"]
    )

    # (c1) the seed alone must fix the run: the same weights passed in a different
    # dict order have to produce the same draws, or a published seed means nothing.
    study_weights = dict(GH_FBWM_STUDY_WEIGHTS)
    ascending = {code: study_weights[code] for code in sorted(study_weights)}
    by_weight = {
        code: study_weights[code]
        for code in sorted(study_weights, key=lambda item: -study_weights[item])
    }
    first_order = run_weight_monte_carlo_sensitivity(
        measure_weights=ascending, samples=300, seed=5, write_csv=False
    )
    second_order = run_weight_monte_carlo_sensitivity(
        measure_weights=by_weight, samples=300, seed=5, write_csv=False
    )
    assert first_order["stability"] == second_order["stability"]
    assert {item["code"]: item["contribution_pct"] for item in first_order["criterion_contributions"]} == {
        item["code"]: item["contribution_pct"] for item in second_order["criterion_contributions"]
    }

    # (c2) every run is persisted as CSV, and the files must agree with the JSON
    # they were built from — that is what makes them citable on their own.
    with tempfile.TemporaryDirectory() as temporary_dir:
        csv_result = run_weight_monte_carlo_sensitivity(
            samples=50,
            perturbation_pct=0.10,
            seed=13,
            csv_output_dir=temporary_dir,
            include_draws=True,
        )
        written = csv_result["csv_output"]["files"]
        assert csv_result["csv_output"]["written"] is True
        assert len(written) == 6
        for path in written:
            assert Path(path).is_file(), path

        draws_path = next(path for path in written if path.endswith("_draws.csv"))
        with open(draws_path, newline="", encoding="utf-8") as handle:
            draw_rows = list(csv.reader(handle))
        assert len(draw_rows) == 51, len(draw_rows)  # header plus one row per draw
        assert draw_rows[0][0] == "draw"
        assert f"weight_{csv_result['criterion_contributions'][0]['code']}" in draw_rows[0]

        companies_path = next(path for path in written if path.endswith("_companies.csv"))
        with open(companies_path, newline="", encoding="utf-8") as handle:
            company_rows = list(csv.DictReader(handle))
        assert len(company_rows) == len(csv_result["companies"])
        assert company_rows[0]["company"] == csv_result["companies"][0]["company"]
        assert company_rows[0]["baseline_category"] == csv_result["companies"][0]["baseline_category"]

        contributions_path = next(
            path for path in written if path.endswith("_criterion_contributions.csv")
        )
        with open(contributions_path, newline="", encoding="utf-8") as handle:
            contribution_rows = list(csv.DictReader(handle))
        assert len(contribution_rows) == len(csv_result["criterion_contributions"])
        assert_close(
            sum(float(row["contribution_pct"]) for row in contribution_rows), 100.0, tol=1e-4
        )

    # (c3) the validation guards sit on the happy path and would otherwise never
    # run. Break the vectorised flows on purpose and confirm each guard refuses to
    # produce a result instead of reporting numbers from a method nobody ran.
    original_components = sensitivity_analysis._flow_components_by_company

    def wrong_classes(prepared):
        rows, columns = original_components(prepared)
        rows = rows.copy()
        # Move the alternative's own flow relative to the profile flows. Scaling
        # everything uniformly would not change any class, because the profiles
        # move with it — which is why the class check cannot replace the flow check.
        rows[:, :, -1] *= 2.5
        return rows, columns

    def wrong_flows_only(prepared):
        rows, columns = original_components(prepared)
        # Shifts the positive and negative flows together, so it cancels out of the
        # net flow and only the all-three-flows comparison can catch it.
        return rows + 1e-3, columns + 1e-3

    for fault, expected_fragment in (
        (wrong_classes, "disagrees with the reference implementation"),
        (wrong_flows_only, "reproduces the baseline classes but not the"),
    ):
        sensitivity_analysis._flow_components_by_company = fault
        try:
            assert_raises_runtime_error(
                lambda: run_weight_monte_carlo_sensitivity(samples=20, seed=1, write_csv=False),
                expected_fragment,
            )
        finally:
            sensitivity_analysis._flow_components_by_company = original_components

    # (d) with no spread every draw is the baseline, so the analysis must reject
    # a degenerate range rather than report a meaningless 100% stability.
    assert_raises_value_error(
        lambda: run_weight_monte_carlo_sensitivity(
            samples=10, perturbation_pct=0.0, write_csv=False
        ),
        "perturbation_pct must be strictly between 0 and 1",
    )
    assert_raises_value_error(
        lambda: run_weight_monte_carlo_sensitivity(samples=1, write_csv=False),
        "samples must be at least 2",
    )
    assert_raises_value_error(
        lambda: run_weight_monte_carlo_sensitivity(
            samples=10, distribution="normal", write_csv=False
        ),
        "Unsupported distribution",
    )


async def main() -> None:
    run_bwm_regression_checks()
    run_data_source_consistency_checks()
    run_flowsort_regression_checks()
    run_weight_monte_carlo_checks()

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "server.py")],
        cwd=str(PROJECT_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            sectors = payload(await session.read_resource(AnyUrl("sasb://sectors")))
            assert sectors[0]["id"] == "semiconductors"

            criteria = payload(await session.read_resource(AnyUrl("sasb://sectors/semiconductors/criteria")))
            assert len(criteria) >= 5

            greenhouse = group_codes(criteria, "Greenhouse Gas Emissions", 2)
            energy = group_codes(criteria, "Energy Management in Manufacturing", 2)
            singleton = group_codes(criteria, "Materials Sourcing", 1)

            top_level_criteria = ["Greenhouse Gas Emissions", "Energy Management in Manufacturing"]
            subcriteria_by_parent = {
                "Greenhouse Gas Emissions": greenhouse,
                "Energy Management in Manufacturing": energy,
            }
            decision_makers = [
                {"id": "dm-1", "name": "DM 1", "weight": 0.6},
                {"id": "dm-2", "name": "DM 2", "weight": 0.4},
            ]

            g1, g2 = [item["code"] for item in greenhouse]
            e1, e2 = [item["code"] for item in energy]

            criteria_level_comparisons = [
                make_level_comparison(
                    "dm-1",
                    "Greenhouse Gas Emissions",
                    "Energy Management in Manufacturing",
                    {
                        "Greenhouse Gas Emissions": "EI",
                        "Energy Management in Manufacturing": "VI",
                    },
                    {
                        "Greenhouse Gas Emissions": "VI",
                        "Energy Management in Manufacturing": "EI",
                    },
                ),
                make_level_comparison(
                    "dm-2",
                    "Energy Management in Manufacturing",
                    "Greenhouse Gas Emissions",
                    {
                        "Greenhouse Gas Emissions": "FI",
                        "Energy Management in Manufacturing": "EI",
                    },
                    {
                        "Greenhouse Gas Emissions": "EI",
                        "Energy Management in Manufacturing": "FI",
                    },
                ),
            ]

            subcriteria_level_comparisons = {
                "Greenhouse Gas Emissions": [
                    make_level_comparison(
                        "dm-1",
                        g1,
                        g2,
                        {g1: "EI", g2: "FI"},
                        {g1: "FI", g2: "EI"},
                    ),
                    make_level_comparison(
                        "dm-2",
                        g2,
                        g1,
                        {g1: "FI", g2: "EI"},
                        {g1: "EI", g2: "FI"},
                    ),
                ],
                "Energy Management in Manufacturing": [
                    make_level_comparison(
                        "dm-1",
                        e1,
                        e2,
                        {e1: "EI", e2: "WI"},
                        {e1: "WI", e2: "EI"},
                    ),
                    make_level_comparison(
                        "dm-2",
                        e1,
                        e2,
                        {e1: "EI", e2: "FI"},
                        {e1: "FI", e2: "EI"},
                    ),
                ],
            }

            success_result = payload(
                await session.call_tool(
                    "calculate_gh_fbwm_weights",
                    {
                        "sector_id": "semiconductors",
                        "top_level_criteria": top_level_criteria,
                        "subcriteria_by_parent": subcriteria_by_parent,
                        "decision_makers": decision_makers,
                        "criteria_level_comparisons": criteria_level_comparisons,
                        "subcriteria_level_comparisons": subcriteria_level_comparisons,
                    },
                )
            )

            assert success_result["active_top_level_criteria"] == top_level_criteria
            assert success_result["global_ranking"]
            assert_close(success_result["criteria_level"]["crisp_weight_sum"], 1.0)
            assert_close(success_result["global_weight_sum"], 1.0)
            assert_close(
                success_result["criteria_level"]["weighted_deviation"],
                round(weighted_deviation(success_result["criteria_level"]), 6),
            )

            for parent_code in top_level_criteria:
                level = success_result["subcriteria_levels"][parent_code]
                assert len(level["weights"]) >= 2
                assert_close(level["crisp_weight_sum"], 1.0)
                assert_close(
                    level["weighted_deviation"],
                    round(weighted_deviation(level), 6),
                )

            rounded_global_sum = sum(item["global_crisp_weight"] for item in success_result["global_weights"])
            assert math.isclose(rounded_global_sum, 1.0, rel_tol=5e-6, abs_tol=5e-6)

            singleton_result = payload(
                await session.call_tool(
                    "calculate_gh_fbwm_weights",
                    {
                        "sector_id": "semiconductors",
                        "top_level_criteria": ["Greenhouse Gas Emissions", "Materials Sourcing"],
                        "subcriteria_by_parent": {
                            "Greenhouse Gas Emissions": greenhouse,
                            "Materials Sourcing": singleton,
                        },
                        "decision_makers": decision_makers,
                        "criteria_level_comparisons": [
                            make_level_comparison(
                                "dm-1",
                                "Greenhouse Gas Emissions",
                                "Materials Sourcing",
                                {
                                    "Greenhouse Gas Emissions": "EI",
                                    "Materials Sourcing": "FI",
                                },
                                {
                                    "Greenhouse Gas Emissions": "FI",
                                    "Materials Sourcing": "EI",
                                },
                            ),
                            make_level_comparison(
                                "dm-2",
                                "Materials Sourcing",
                                "Greenhouse Gas Emissions",
                                {
                                    "Greenhouse Gas Emissions": "WI",
                                    "Materials Sourcing": "EI",
                                },
                                {
                                    "Greenhouse Gas Emissions": "EI",
                                    "Materials Sourcing": "WI",
                                },
                            ),
                        ],
                        "subcriteria_level_comparisons": {
                            "Greenhouse Gas Emissions": subcriteria_level_comparisons[
                                "Greenhouse Gas Emissions"
                            ],
                        },
                    },
                )
            )
            assert singleton_result["subcriteria_levels"]["Materials Sourcing"]["weights"][0]["crisp_weight"] == 1.0
            assert singleton_result["subcriteria_levels"]["Materials Sourcing"]["weighted_deviation"] == 0.0

            invalid_weights = await session.call_tool(
                "calculate_gh_fbwm_weights",
                {
                    "sector_id": "semiconductors",
                    "top_level_criteria": top_level_criteria,
                    "subcriteria_by_parent": subcriteria_by_parent,
                    "decision_makers": [
                        {"id": "dm-1", "name": "DM 1", "weight": 0.7},
                        {"id": "dm-2", "name": "DM 2", "weight": 0.4},
                    ],
                    "criteria_level_comparisons": criteria_level_comparisons,
                    "subcriteria_level_comparisons": subcriteria_level_comparisons,
                },
            )
            assert "must sum to 1" in tool_error_text(invalid_weights)

            missing_comparison = await session.call_tool(
                "calculate_gh_fbwm_weights",
                {
                    "sector_id": "semiconductors",
                    "top_level_criteria": top_level_criteria,
                    "subcriteria_by_parent": subcriteria_by_parent,
                    "decision_makers": decision_makers,
                    "criteria_level_comparisons": [
                        make_level_comparison(
                            "dm-1",
                            "Greenhouse Gas Emissions",
                            "Energy Management in Manufacturing",
                            {"Greenhouse Gas Emissions": "EI"},
                            {
                                "Greenhouse Gas Emissions": "VI",
                                "Energy Management in Manufacturing": "EI",
                            },
                        ),
                        criteria_level_comparisons[1],
                    ],
                    "subcriteria_level_comparisons": subcriteria_level_comparisons,
                },
            )
            assert "Missing Criteria Level Best-to-Others comparison for Energy Management in Manufacturing" in tool_error_text(
                missing_comparison
            )

            same_best_worst = await session.call_tool(
                "calculate_gh_fbwm_weights",
                {
                    "sector_id": "semiconductors",
                    "top_level_criteria": top_level_criteria,
                    "subcriteria_by_parent": subcriteria_by_parent,
                    "decision_makers": decision_makers,
                    "criteria_level_comparisons": [
                        make_level_comparison(
                            "dm-1",
                            "Greenhouse Gas Emissions",
                            "Greenhouse Gas Emissions",
                            {
                                "Greenhouse Gas Emissions": "EI",
                                "Energy Management in Manufacturing": "VI",
                            },
                            {
                                "Greenhouse Gas Emissions": "VI",
                                "Energy Management in Manufacturing": "EI",
                            },
                        ),
                        criteria_level_comparisons[1],
                    ],
                    "subcriteria_level_comparisons": subcriteria_level_comparisons,
                },
            )
            assert "best and worst codes must be different" in tool_error_text(same_best_worst)

            equal_best_worst_importance = await session.call_tool(
                "calculate_gh_fbwm_weights",
                {
                    "sector_id": "semiconductors",
                    "top_level_criteria": top_level_criteria,
                    "subcriteria_by_parent": subcriteria_by_parent,
                    "decision_makers": decision_makers,
                    "criteria_level_comparisons": [
                        make_level_comparison(
                            "dm-1",
                            "Greenhouse Gas Emissions",
                            "Energy Management in Manufacturing",
                            {
                                "Greenhouse Gas Emissions": "EI",
                                "Energy Management in Manufacturing": "EI",
                            },
                            {
                                "Greenhouse Gas Emissions": "EI",
                                "Energy Management in Manufacturing": "EI",
                            },
                        ),
                        criteria_level_comparisons[1],
                    ],
                    "subcriteria_level_comparisons": subcriteria_level_comparisons,
                },
            )
            assert "Best-to-Worst comparison cannot be EI" in tool_error_text(equal_best_worst_importance)

            too_small_group = await session.call_tool(
                "calculate_gh_fbwm_weights",
                {
                    "sector_id": "semiconductors",
                    "top_level_criteria": ["Greenhouse Gas Emissions", "Materials Sourcing"],
                    "subcriteria_by_parent": {
                        "Greenhouse Gas Emissions": greenhouse,
                        "Materials Sourcing": [],
                    },
                    "decision_makers": decision_makers,
                    "criteria_level_comparisons": [
                        make_level_comparison(
                            "dm-1",
                            "Greenhouse Gas Emissions",
                            "Materials Sourcing",
                            {
                                "Greenhouse Gas Emissions": "EI",
                                "Materials Sourcing": "FI",
                            },
                            {
                                "Greenhouse Gas Emissions": "FI",
                                "Materials Sourcing": "EI",
                            },
                        ),
                        make_level_comparison(
                            "dm-2",
                            "Greenhouse Gas Emissions",
                            "Materials Sourcing",
                            {
                                "Greenhouse Gas Emissions": "EI",
                                "Materials Sourcing": "WI",
                            },
                            {
                                "Greenhouse Gas Emissions": "WI",
                                "Materials Sourcing": "EI",
                            },
                        ),
                    ],
                    "subcriteria_level_comparisons": {
                        "Greenhouse Gas Emissions": subcriteria_level_comparisons[
                            "Greenhouse Gas Emissions"
                        ],
                    },
                },
            )
            assert "requires one element" in tool_error_text(too_small_group)

            profile_result = payload(
                await session.call_tool(
                    "calculate_partner_qualification_profiles",
                    {"sector_id": "semiconductors"},
                )
            )
            flow_profiles = profile_result["profiles"]
            tc_sc_130a_1_3_profiles = flow_profiles["TC-SC-130a.1.3"]["profiles"]
            assert tc_sc_130a_1_3_profiles["r1"] >= tc_sc_130a_1_3_profiles["r2"]
            assert tc_sc_130a_1_3_profiles["r2"] >= tc_sc_130a_1_3_profiles["r3"]
            assert tc_sc_130a_1_3_profiles["r3"] >= tc_sc_130a_1_3_profiles["r4"]
            assert flow_profiles["TC-SC-110a.1.1"]["metric_type"] == "absolute"
            assert "r1" in flow_profiles["TC-SC-110a.1.1"]["profiles"]
            assert (
                flow_profiles["TC-SC-110a.1.1"]["profile_company_filter"]
                == "Semiconductor & Solar Equipment"
            )

            partner_inputs = payload(
                await session.call_tool(
                    "partner_qualification_inputs",
                    {"sector_id": "semiconductors"},
                )
            )
            assert partner_inputs["input_dataset"] == "measures_HITL.json"
            assert partner_inputs["profile_dataset"] == "corrected_all.json"
            assert partner_inputs["preference_function"] == "t1"
            assert "Hitachi" in partner_inputs["companies"]
            assert len(partner_inputs["sensitivity_analysis"]["prompt_variants"]) == 4

            promptfoo_config = payload(
                await session.call_tool(
                    "build_partner_sensitivity_promptfoo_config",
                    {
                        "sector_id": "semiconductors",
                        "model_families": [
                            {
                                "family": "Llama Family",
                                "provider": "ollama",
                                "model": "llama3.1:8b",
                                "enabled": True,
                            }
                        ],
                        "prompt_variants": ["prompt_1", "prompt_2"],
                        "temperatures": [0.0, 0.5],
                        "top_ps": [1.0],
                        "seed": 42,
                        "repetitions": 2,
                    },
                )
            )
            assert promptfoo_config["provider_count"] == 2
            assert promptfoo_config["test_count"] > 0
            assert "contains-json" in promptfoo_config["config_json"]

            qualification_result = payload(
                await session.call_tool(
                    "qualify_partners_flowsort",
                    {
                        "sector_id": "semiconductors",
                        "criteria_formulation_result": success_result,
                        "companies": ["Hitachi", "Applied Materials", "Lam Research"],
                        "indifference_threshold": 0,
                        "preference_threshold": 0,
                    },
                )
            )
            assert qualification_result["rule"] == "net"
            assert qualification_result["mode"] == "limiting"
            assert qualification_result["input_dataset"] == "measures_HITL.json"
            assert qualification_result["preference_function"] == "t1"
            assert len(qualification_result["companies"]) == 3
            assert qualification_result["companies"][0]["company"] == "Hitachi"
            assert qualification_result["companies"][0]["assigned_category"] in {"C1", "C2", "C3"}
            assert math.isclose(
                sum(item["weight"] for item in qualification_result["criteria_weights"].values()),
                1.0,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )

            monte_carlo_result = payload(
                await session.call_tool(
                    "run_flowsort_weight_monte_carlo",
                    {
                        "sector_id": "semiconductors",
                        "samples": 200,
                        "perturbation_pct": 0.1,
                        "write_csv": False,
                    },
                )
            )
            assert monte_carlo_result["monte_carlo"]["samples"] == 200
            assert monte_carlo_result["monte_carlo"]["criteria_count"] == 14
            assert len(monte_carlo_result["criterion_contributions"]) == 14
            assert monte_carlo_result["validation"][
                "vectorised_path_matches_reference_categories"
            ] is True

            status_result = payload(await session.call_tool("partner_qualification_status", {}))
            assert status_result["status"] == "implemented_flowsort_foundation"

            print("Smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
