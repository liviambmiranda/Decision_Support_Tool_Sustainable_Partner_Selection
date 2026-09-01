"""MCP server for sustainable partner decision support."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.decision_support_mcp.bwm_tfn import LINGUISTIC_SCALE, solve_gh_fbwm_hierarchy
from src.decision_support_mcp.flowsort import (
    calculate_flowsort_profiles,
    display_profiles,
    get_partner_qualification_inputs,
    qualify_partners_with_flowsort,
)
from src.decision_support_mcp.sensitivity_analysis import (
    MONTE_CARLO_DEFAULT_PERTURBATION_PCT,
    MONTE_CARLO_DEFAULT_SAMPLES,
    build_promptfoo_sensitivity_config,
    run_partner_sensitivity_analysis as run_partner_sensitivity_analysis_core,
    run_weight_monte_carlo_sensitivity as run_weight_monte_carlo_sensitivity_core,
)
from src.decision_support_mcp.sasb import build_topic_hierarchy, filter_criteria, get_criterion, list_sectors, load_criteria


mcp = FastMCP(
    "Sustainable Partner Decision Support",
    instructions=(
        "Supports sustainable partner selection. Initial modules: Criteria "
        "Formulation and Partner Qualification. Criteria are based on SASB "
        "Standards; criteria weighting uses the Group Hierarchical Fuzzy "
        "Best-Worst Method with triangular fuzzy numbers, and qualification "
        "uses FlowSort with limiting profiles."
    ),
)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.resource(
    "sasb://sectors",
    name="sasb_sectors",
    title="Available SASB Sectors",
    description="Lists sectors currently available for SASB-based criteria formulation.",
    mime_type="application/json",
)
def sasb_sectors() -> str:
    return _json(list_sectors())


@mcp.resource(
    "sasb://sectors/{sector_id}/criteria",
    name="sasb_sector_criteria",
    title="SASB Criteria by Sector",
    description="Returns SASB criteria metadata for the requested sector.",
    mime_type="application/json",
)
def sasb_sector_criteria(sector_id: str) -> str:
    return _json(load_criteria(sector_id))


@mcp.resource(
    "sasb://sectors/{sector_id}/criteria/{code}",
    name="sasb_criterion",
    title="SASB Criterion Detail",
    description="Returns one SASB criterion by code.",
    mime_type="application/json",
)
def sasb_criterion(sector_id: str, code: str) -> str:
    return _json(get_criterion(code, sector_id))


@mcp.tool()
def list_sasb_criteria(
    sector_id: str = "semiconductors",
    domain: str | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """List SASB criteria for a sector, optionally filtered by domain or topic."""

    criteria = filter_criteria(sector_id, domain=domain, topic=topic)
    return {
        "sector_id": sector_id,
        "count": len(criteria),
        "criteria": criteria,
    }


@mcp.tool()
def get_sasb_criterion(code: str, sector_id: str = "semiconductors") -> dict[str, Any]:
    """Get the full metadata for a SASB criterion code."""

    return get_criterion(code, sector_id)


@mcp.tool()
def fuzzy_bwm_linguistic_scale() -> dict[str, Any]:
    """Return the linguistic terms accepted by the GH-FBWM weighting tool."""

    descriptions = {
        "EI": "Equally important",
        "WI": "Weakly important",
        "FI": "Fairly important",
        "VI": "Very important",
        "AI": "Absolutely important",
    }
    return {
        "terms": [
            {
                "term": term,
                "description": descriptions[term],
                "triangular_fuzzy_number": {"l": tfn[0], "m": tfn[1], "u": tfn[2]},
            }
            for term, tfn in LINGUISTIC_SCALE.items()
        ]
    }


@mcp.tool()
def calculate_gh_fbwm_weights(
    sector_id: str,
    top_level_criteria: list[str],
    subcriteria_by_parent: dict[str, list[dict[str, Any]]],
    decision_makers: list[dict[str, Any]],
    criteria_level_comparisons: list[dict[str, Any]],
    subcriteria_level_comparisons: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Calculate hierarchical GH-FBWM weights for the active SASB topic hierarchy."""

    normalized_top = [str(code).strip() for code in top_level_criteria if str(code).strip()]
    if len(normalized_top) < 2:
        raise ValueError("At least two top-level topics must remain active.")

    result = solve_gh_fbwm_hierarchy(
        sector_id=sector_id,
        top_level_criteria=normalized_top,
        subcriteria_by_parent=subcriteria_by_parent,
        decision_makers=decision_makers,
        criteria_level_comparisons=criteria_level_comparisons,
        subcriteria_level_comparisons=subcriteria_level_comparisons,
    )

    return result


@mcp.tool()
def sasb_topic_hierarchy(sector_id: str = "semiconductors") -> dict[str, Any]:
    """Return the SASB hierarchy grouped as topics -> indicators for a sector."""

    hierarchy = build_topic_hierarchy(sector_id)
    return {
        "sector_id": sector_id,
        "topics": [
            {
                "code": topic,
                "label": topic,
                "indicators": indicators,
            }
            for topic, indicators in hierarchy.items()
        ],
    }


@mcp.tool()
def calculate_partner_qualification_profiles(
    sector_id: str = "semiconductors",
) -> dict[str, Any]:
    """Calculate FlowSort limiting profiles for every criterion in corrected_all.json."""

    return {
        "sector_id": sector_id,
        "profiles": {
            code: {**entry, "profiles": display_profiles(entry["profiles"])}
            for code, entry in calculate_flowsort_profiles(sector_id).items()
        },
    }


@mcp.tool()
def partner_qualification_inputs(
    sector_id: str = "semiconductors",
) -> dict[str, Any]:
    """Return the partner universe and default FlowSort inputs for the UI."""

    return get_partner_qualification_inputs(sector_id)


@mcp.tool()
def qualify_partners_flowsort(
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
) -> dict[str, Any]:
    """Qualify partners with FlowSort using weights from GH-FBWM or a direct weight map."""

    return qualify_partners_with_flowsort(
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
    )


@mcp.tool()
def build_partner_sensitivity_promptfoo_config(
    sector_id: str = "semiconductors",
    model_families: list[dict[str, Any]] | None = None,
    prompt_variants: list[str] | None = None,
    temperatures: list[float] | None = None,
    top_ps: list[float] | None = None,
    seed: int = 42,
    repetitions: int = 2,
) -> dict[str, Any]:
    """Build an in-memory Promptfoo config for qualitative sensitivity analysis."""

    return build_promptfoo_sensitivity_config(
        sector_id=sector_id,
        model_families=model_families,
        prompt_variants=prompt_variants,
        temperatures=temperatures,
        top_ps=top_ps,
        seed=seed,
        repetitions=repetitions,
    )


@mcp.tool()
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
    """Run local sensitivity analysis over qualitative LLM scoring and FlowSort outcomes."""

    return run_partner_sensitivity_analysis_core(
        sector_id=sector_id,
        criteria_formulation_result=criteria_formulation_result,
        measure_weights=measure_weights,
        companies=companies,
        model_families=model_families,
        prompt_variants=prompt_variants,
        temperatures=temperatures,
        top_ps=top_ps,
        seed=seed,
        repetitions=repetitions,
        mode=mode,
        rule=rule,
        preference_function=preference_function,
        indifference_threshold=indifference_threshold,
        preference_threshold=preference_threshold,
    )


@mcp.tool()
def run_flowsort_weight_monte_carlo(
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
    samples: int = MONTE_CARLO_DEFAULT_SAMPLES,
    perturbation_pct: float = MONTE_CARLO_DEFAULT_PERTURBATION_PCT,
    distribution: str = "uniform",
    seed: int = 42,
    write_csv: bool = True,
    csv_output_dir: str | None = None,
    include_draws: bool = False,
) -> dict[str, Any]:
    """Run a Monte Carlo sensitivity analysis of FlowSort against the criterion weights.

    Each weight is drawn from a uniform density spanning the original weight plus
    or minus `perturbation_pct`, FlowSort is re-run on every draw, and rank
    correlation between the sampled weights and the outcomes identifies how much
    each criterion contributes to the classification.

    Results are written to CSV under `data/monte_carlo` on every run, one file per
    table; `include_draws` adds the draw-by-draw sample.
    """

    return run_weight_monte_carlo_sensitivity_core(
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
        samples=samples,
        perturbation_pct=perturbation_pct,
        distribution=distribution,
        seed=seed,
        write_csv=write_csv,
        csv_output_dir=csv_output_dir,
        include_draws=include_draws,
    )


@mcp.tool()
def partner_qualification_status() -> dict[str, str]:
    """Return the current implementation status for Partner Qualification."""

    return {
        "module": "Partner Qualification",
        "status": "implemented_flowsort_foundation",
        "message": (
            "Partner Qualification now computes FlowSort limiting profiles from corrected_all.json "
            "and classifies the companies in measures_HITL.json using weights from Criteria Formulation."
        ),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
