#!/usr/bin/env python3
"""Does the prompt move the FlowSort classification, or only its label?

    .venv312/bin/python scripts/run_prompt_boundary_analysis.py

A category that differs between two prompts is not by itself evidence that the
prompt mattered. In limiting FlowSort each alternative is classified inside the
sub-problem formed by the limiting profiles and that alternative alone, so every
candidate carries its own category band. A candidate sitting near the edge of its
band changes category on a movement too small to mean anything; a candidate in
the middle of a wide band keeps its category through a movement that is large.
Reading the labels alone confuses the two.

This script separates them. For every (prompt, series, candidate) it records the
net flow, the two profile flows that bound the assigned category, and the margin
to each. A prompt-induced category change is then read against the margin it had
to cross, and a *stable* category is read against the margin it happened to keep.

Three levels are reported, because the question can be answered differently at
each and the disagreement between them is the finding:

1. ratings      -- did the prompt change the model's scores at all?
2. net flows    -- did the score changes survive aggregation into a flow?
3. categories   -- did the flow changes cross a boundary?

Writes ``prompt_boundary_*.csv`` under ``data/prompt_sensitivity/csv``.
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision_support_mcp.flowsort import (  # noqa: E402
    CATEGORY_LABELS,
    _flowsort_details,
    prepare_flowsort_inputs,
)
from src.decision_support_mcp.rating_evaluation import (  # noqa: E402
    prompt_sensitivity as ps,
)

SCORES_DIR = ps.SCORES_BY_MODEL_DIR
CSV_DIR = ps.CSV_DIR
PROMPTS = ("prompt_1", "prompt_2", "prompt_3", "prompt_4")
MEDIAN_KEY = "median"
SERIES = (*sorted(spec.tag for spec in ps.DEFAULT_MODELS), MEDIAN_KEY)

#: A margin at or below this share of its band is called knife-edge: the category
#: is decided by a movement smaller than the one the prompt axis itself produces,
#: so the label is not carrying information about the prompt.
KNIFE_EDGE_BAND_SHARE = 0.10


def _code(index: int) -> str:
    return str(CATEGORY_LABELS[int(index)]["code"])


def _scores_path(prompt_id: str, series: str) -> Path:
    safe = series.replace(":", "_").replace("/", "_")
    return SCORES_DIR / f"qualitative_scores_{prompt_id}_{safe}.json"


def flowsort_with_margins(prompt_id: str, series: str) -> dict[str, dict[str, Any]]:
    """Net flow, category, and distance to both category boundaries per candidate."""

    path = _scores_path(prompt_id, series)
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Run run_flowsort_by_model.py first.")
    prepared = prepare_flowsort_inputs(
        qualitative_source="rater_study", rater_study_scores_path=path
    )
    details = _flowsort_details(
        dataset=prepared["dataset_array"],
        profiles=prepared["profiles_matrix"],
        W=np.asarray(prepared["weights_vector"], dtype=float),
        Q=prepared["q_vector"],
        S=prepared["s_vector"],
        P=prepared["p_vector"],
        F=prepared["preference_functions"],
        mode=prepared["mode"],
    )

    out: dict[str, dict[str, Any]] = {}
    for company, detail in zip(prepared["target_companies"], details):
        flow = float(detail["net_flow"])
        profile_flows = [float(value) for value in detail["profile_net_flows"]]
        category = int(detail["net_category"])
        # Assignment rule: profile_flows[h-1] >= flow > profile_flows[h] puts the
        # candidate in category h. Those two are the band it sits in.
        upper = profile_flows[category - 1]
        lower = profile_flows[category]
        band = upper - lower
        margin_up = upper - flow
        margin_down = flow - lower
        margin = min(margin_up, margin_down)
        out[company] = {
            "net_flow": flow,
            "category": _code(category),
            "band_upper": upper,
            "band_lower": lower,
            "band_width": band,
            "margin_to_upgrade": margin_up,
            "margin_to_downgrade": margin_down,
            "margin_to_nearest_boundary": margin,
            "band_share": margin / band if band > 0 else 0.0,
            "nearest_boundary": "upgrade" if margin_up <= margin_down else "downgrade",
        }
    return out


def main() -> int:
    table = {
        (prompt, series): flowsort_with_margins(prompt, series)
        for prompt in PROMPTS
        for series in SERIES
    }
    companies = sorted(table[(PROMPTS[0], SERIES[0])])
    letters = {name: f"Candidate {chr(65 + i)}" for i, name in enumerate(companies)}

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Every cell, with its margin. The two tables below are views of this one.
    cells = [
        {
            "prompt_id": prompt,
            "series": series,
            "series_type": "median_of_models" if series == MEDIAN_KEY else "single_model",
            "candidate": letters[company],
            "company": company,
            "assigned_category": entry["category"],
            "net_flow": round(entry["net_flow"], 6),
            "band_upper": round(entry["band_upper"], 6),
            "band_lower": round(entry["band_lower"], 6),
            "band_width": round(entry["band_width"], 6),
            "margin_to_upgrade": round(entry["margin_to_upgrade"], 6),
            "margin_to_downgrade": round(entry["margin_to_downgrade"], 6),
            "margin_to_nearest_boundary": round(entry["margin_to_nearest_boundary"], 6),
            "band_share_of_margin": round(entry["band_share"], 4),
            "nearest_boundary": entry["nearest_boundary"],
            "knife_edge": entry["band_share"] <= KNIFE_EDGE_BAND_SHARE,
        }
        for (prompt, series), rows in table.items()
        for company, entry in rows.items()
    ]
    cells.sort(key=lambda row: (row["series"], row["candidate"], row["prompt_id"]))

    # 2. Per (series, candidate) across the prompts: how much the flow moved, and
    #    how that compares with the margin it had. This is the discriminator.
    spread_rows = []
    for series in SERIES:
        for company in companies:
            entries = [table[(prompt, series)][company] for prompt in PROMPTS]
            flows = [entry["net_flow"] for entry in entries]
            categories = [entry["category"] for entry in entries]
            margins = [entry["margin_to_nearest_boundary"] for entry in entries]
            flow_range = max(flows) - min(flows)
            min_margin = min(margins)
            spread_rows.append(
                {
                    "series": series,
                    "candidate": letters[company],
                    "company": company,
                    "n_distinct_categories": len(set(categories)),
                    "categories_by_prompt": " ".join(
                        f"{prompt.split('_')[1]}:{code}"
                        for prompt, code in zip(PROMPTS, categories)
                    ),
                    "category_changed": len(set(categories)) > 1,
                    "net_flow_min": round(min(flows), 6),
                    "net_flow_max": round(max(flows), 6),
                    "net_flow_range_across_prompts": round(flow_range, 6),
                    "min_margin_across_prompts": round(min_margin, 6),
                    # The reading: a movement larger than the margin can flip the
                    # category, and a margin larger than the movement cannot be
                    # reached by changing the prompt.
                    "movement_exceeds_margin": flow_range > min_margin,
                    "min_band_share": round(
                        min(entry["band_share"] for entry in entries), 4
                    ),
                    "knife_edge_under_any_prompt": any(
                        entry["band_share"] <= KNIFE_EDGE_BAND_SHARE for entry in entries
                    ),
                }
            )

    # 3. Prompt pair against prompt pair, per series: ratings, flows, categories.
    pair_rows = []
    for series in SERIES:
        for left, right in combinations(PROMPTS, 2):
            changes = []
            gaps = []
            for company in companies:
                a, b = table[(left, series)][company], table[(right, series)][company]
                gaps.append(abs(a["net_flow"] - b["net_flow"]))
                if a["category"] != b["category"]:
                    # Was the crossing wide, or did it merely step over a
                    # boundary that was already within reach?
                    crossing = min(
                        a["margin_to_nearest_boundary"], b["margin_to_nearest_boundary"]
                    )
                    changes.append(
                        f"{letters[company].split()[-1]}:{a['category']}->{b['category']}"
                        f"({'knife-edge' if crossing <= 0.01 else 'wide'})"
                    )
            pair_rows.append(
                {
                    "series": series,
                    "prompt_a": left,
                    "prompt_b": right,
                    "n_category_changes": len(changes),
                    "category_agreement_pct": round(
                        100.0 * (len(companies) - len(changes)) / len(companies), 1
                    ),
                    "mean_abs_net_flow_gap": round(sum(gaps) / len(gaps), 6),
                    "max_abs_net_flow_gap": round(max(gaps), 6),
                    "changes": "; ".join(changes),
                }
            )

    written = []
    for name, rows in (
        ("prompt_boundary_cells", cells),
        ("prompt_boundary_by_candidate", spread_rows),
        ("prompt_boundary_by_prompt_pair", pair_rows),
    ):
        path = CSV_DIR / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        written.append(path)

    print("Candidates whose category is decided within 10% of a band edge\n")
    for row in cells:
        if row["knife_edge"]:
            print(
                f"  {row['series']:<18} {row['prompt_id']:<9} {row['candidate']:<12} "
                f"{row['assigned_category']}  margin={row['margin_to_nearest_boundary']:+.4f} "
                f"({row['band_share_of_margin']:.1%} of band, toward {row['nearest_boundary']})"
            )

    print("\n\nPer candidate: did the prompt move it, and could it have flipped?\n")
    print(
        f"  {'series':<18}{'cand':<6}{'categories':<28}"
        f"{'flow range':>11}{'min margin':>12}  verdict"
    )
    for row in spread_rows:
        verdict = (
            "changed" if row["category_changed"]
            else "stable, but reachable" if row["movement_exceeds_margin"]
            else "stable, out of reach"
        )
        print(
            f"  {row['series']:<18}{row['candidate'].split()[-1]:<6}"
            f"{row['categories_by_prompt']:<28}"
            f"{row['net_flow_range_across_prompts']:>11.4f}"
            f"{row['min_margin_across_prompts']:>12.4f}  {verdict}"
        )

    print()
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
