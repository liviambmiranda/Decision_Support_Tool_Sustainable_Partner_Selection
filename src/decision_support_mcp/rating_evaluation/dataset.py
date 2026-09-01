"""The fixed set of company-sub-criterion units scored in the study.

Read-only. This module never writes to the datasets the decision-support tool
depends on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

from src.decision_support_mcp.quantitative_scores import (
    _load_measures_hitl,
    _metadata_by_code,
)

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
EVAL_DIR = DATA_DIR / "eval"

LLM_RATINGS_PATH = EVAL_DIR / "llm_ratings.jsonl"
HUMAN_RATINGS_PATH = EVAL_DIR / "human_ratings.csv"
CONSENSUS_PATH = EVAL_DIR / "qualitative_consensus_scores.json"
REPORT_PATH = EVAL_DIR / "agreement_report.json"


class Unit(NamedTuple):
    """One company-sub-criterion scoring unit."""

    company: str
    code: str
    measure: str
    guidance: str
    disclosure: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.company, self.code)


def load_units(sector_id: str = "semiconductors") -> list[Unit]:
    """Return every qualitative company-sub-criterion pair, in a stable order.

    Rows whose disclosure is absent are excluded: the live tool assigns those the
    worst limiting profile rather than a rated judgement, so they carry no rater
    information. The exclusion is reported by :func:`describe_units`.
    """

    rows = _load_measures_hitl()
    metadata_by_code = _metadata_by_code(sector_id)

    units: list[Unit] = []
    for row in rows:
        code = str(row["code"])
        meta = metadata_by_code.get(code)
        if meta is None:
            continue
        if str(meta.get("category", "")).strip().lower() != "qualitative":
            continue
        disclosure = row.get("value")
        if not isinstance(disclosure, str) or not disclosure.strip():
            continue
        units.append(
            Unit(
                company=str(row["company"]),
                code=code,
                measure=str(meta["measure"]),
                guidance=str(meta.get("explanation", "")),
                # Passed through verbatim. The live tool escapes double quotes
                # for its quoted-string prompt; the study prompt uses triple-quote
                # delimiters, so escaping would corrupt the disclosure text.
                disclosure=disclosure,
            )
        )

    units.sort(key=lambda unit: (unit.code, unit.company))
    return units


def describe_units(sector_id: str = "semiconductors") -> dict[str, Any]:
    """Summarise study scope, for the report header and for sanity checks."""

    rows = _load_measures_hitl()
    metadata_by_code = _metadata_by_code(sector_id)
    qualitative_codes = sorted(
        code
        for code, meta in metadata_by_code.items()
        if str(meta.get("category", "")).strip().lower() == "qualitative"
    )
    qualitative_rows = [row for row in rows if str(row["code"]) in set(qualitative_codes)]
    units = load_units(sector_id)

    return {
        "sector_id": sector_id,
        "qualitative_codes": qualitative_codes,
        "companies": sorted({unit.company for unit in units}),
        "n_units": len(units),
        "n_excluded_not_reported": len(qualitative_rows) - len(units),
    }
