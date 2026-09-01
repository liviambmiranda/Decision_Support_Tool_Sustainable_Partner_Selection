"""Quantitative company scoring and limiting profile preparation."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.decision_support_mcp.sasb import load_criteria

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORRECTED_ALL_PATH = DATA_DIR / "corrected_all.json" # usado para reference profile
MEASURES_HITL_PATH = DATA_DIR / "measures_HITL.json" # usado para score quantitativo
FINANCIAL_HITL_EQUIPMENT_PATH = DATA_DIR / "financial_HITL_equipment.json" # usado para score quantitativo
CONVERTION_PATH = DATA_DIR / "convertion.json" # usado para score quantitativo

TYPE_A_PERCENTAGE = "percentage"
TYPE_B_ABSOLUTE = "absolute"
TYPE_C_QUALITATIVE = "qualitative"

#: Score given to a qualitative criterion with no disclosure at all. The 1-5
#: rubric has no level for "nothing was disclosed", so it sits below the scale.
QUALITATIVE_NOT_REPORTED_VALUE = 0.0
#: Bottom limiting profile for qualitative criteria, strictly below the value
#: above so that every alternative dominates it.
QUALITATIVE_PROFILE_FLOOR = -0.5

NOT_REPORTED_VALUES = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "n.a.",
    "not reported",
    "not applied",
    "nr",
    "none",
    "null",
}

COMPANY_TYPE_MAPPING = {
    "TOWA CORPORATION": "Semiconductor & Solar Equipment",
    "Advantest": "Semiconductor & Solar Equipment",
    "AMD": "Fabless",
    "ASM": "Semiconductor & Solar Equipment",
    "ASML": "Semiconductor & Solar Equipment",
    "Amkor Technology": "Outsourced Semiconductor Assembly & Testing (OSAT)",
    "Axcelis": "Semiconductor & Solar Equipment",
    "Besi": "Semiconductor & Solar Equipment",
    "Chroma ATE Inc.": "Semiconductor & Solar Equipment",
    "KLA": "Semiconductor & Solar Equipment",
    "Teradyne": "Semiconductor & Solar Equipment",
    "SUSS MicroTec SE": "Semiconductor & Solar Equipment",
    "Cohu, Inc.": "Semiconductor & Solar Equipment",
    "EBARA Group": "Semiconductor & Solar Equipment",
    "ASMPT LIMITED": "Semiconductor & Solar Equipment",
    "GlobalFoundries": "Foundry",
    "Intel": "Integrated Device Manufacturers (IDM)",
    "KOKUSAI ELECTRIC": "Semiconductor & Solar Equipment",
    "Kulicke & Soffa": "Semiconductor & Solar Equipment",
    "Dai Nippon Printing Co., Ltd.": "Semiconductor & Solar Materials",
    "TOPPAN Holdings Inc.": "Semiconductor & Solar Materials",
    "SK siltron": "Semiconductor & Solar Materials",
    "Siltronic AG": "Semiconductor & Solar Materials",
    "Anji Microelectronics Technology (Shanghai) Co., Ltd.": "Semiconductor & Solar Materials",
    "SUMCO": "Semiconductor & Solar Materials",
    "Semiconductor Manufacturing International Corporation": "Foundry",
    "Shin-Etsu Chemical Co., Ltd.": "Semiconductor & Solar Materials",
    "TSMC": "Foundry",
    "applied materials": "Semiconductor & Solar Equipment",
    "Applied Materials": "Semiconductor & Solar Equipment",
    "amec": "Semiconductor & Solar Equipment",
    "Advanced Micro-Fabrication Equipment Inc. China (AMEC)": "Semiconductor & Solar Equipment",
    "hitachi": "Semiconductor & Solar Equipment",
    "Hitachi": "Semiconductor & Solar Equipment",
    "tokyo electron": "Semiconductor & Solar Equipment",
    "Tokyo Electron Limited": "Semiconductor & Solar Equipment",
    "lam research": "Semiconductor & Solar Equipment",
    "Lam Research": "Semiconductor & Solar Equipment",
    "Samsung Electronics": "Integrated Device Manufacturers (IDM)",
    "NVIDIA": "Fabless",
}

COMPANY_TYPE_BY_CASEFOLD = {name.casefold(): value for name, value in COMPANY_TYPE_MAPPING.items()}
EQUIPMENT_COMPANY_TYPE = "Semiconductor & Solar Equipment"

#: Peer group whose observations define the limiting profiles.
#:
#: Restricted to front-end wafer-fabrication equipment makers, the segment the
#: evaluated suppliers compete in. The broader "Semiconductor & Solar Equipment"
#: type also covers back-end assembly, test and packaging houses (Advantest,
#: Teradyne, Besi, ASMPT, Cohu, Kulicke & Soffa, TOWA, Chroma ATE) plus EBARA,
#: whose scale and process footprint make their emissions, energy and water
#: figures a poor yardstick for this comparison. A tighter peer group narrows
#: the middle category and lets the alternatives separate.
#:
#: Every evaluated supplier is also a peer, so no alternative is measured
#: against a standard it had no part in setting.
#:
#: Names are canonical forms as produced by ``_canonical_company_name``.
PROFILE_PEER_GROUP: frozenset[str] = frozenset({
    "Applied Materials",
    "ASML",
    "Tokyo Electron Limited",
    "Lam Research",
    "KLA",
    "ASM",
    "KOKUSAI ELECTRIC",
    "Advanced Micro-Fabrication Equipment Inc. China (AMEC)",
    "Axcelis",
    "SUSS MicroTec SE",
    "Hitachi",
})

COMPANY_ALIASES_RAW = {
    "applied materials inc": "Applied Materials",
    "applied materials": "Applied Materials",
    "asml holding nv": "ASML",
    "asml": "ASML",
    "tokyo electron limited": "Tokyo Electron Limited",
    "tokyo electron": "Tokyo Electron Limited",
    "lam research corporation": "Lam Research",
    "lam research": "Lam Research",
    "kla corporation": "KLA",
    "kla": "KLA",
    "ebara corporation": "EBARA Group",
    "ebara group": "EBARA Group",
    "advantest corporation": "Advantest",
    "advantest": "Advantest",
    "asm international": "ASM",
    "asm": "ASM",
    "asmpt limited": "ASMPT LIMITED",
    "asmpt": "ASMPT LIMITED",
    "kokusai electric corporation": "KOKUSAI ELECTRIC",
    "kokusai electric": "KOKUSAI ELECTRIC",
    "axcelis technologies": "Axcelis",
    "axcelis technologies inc": "Axcelis",
    "axcelis": "Axcelis",
    "be semiconductor industries": "Besi",
    "besi": "Besi",
    "suss microtec se": "SUSS MicroTec SE",
    "suss microtec": "SUSS MicroTec SE",
    "cohu inc": "Cohu, Inc.",
    "cohu": "Cohu, Inc.",
    "advanced micro fabrication equipment": "Advanced Micro-Fabrication Equipment Inc. China (AMEC)",
    "advanced micro fabrication equipment china amec": "Advanced Micro-Fabrication Equipment Inc. China (AMEC)",
    "advanced micro fabrication equipment inc china amec": "Advanced Micro-Fabrication Equipment Inc. China (AMEC)",
    "advanced micro-fabrication equipment inc. china (amec)": "Advanced Micro-Fabrication Equipment Inc. China (AMEC)",
    "amec": "Advanced Micro-Fabrication Equipment Inc. China (AMEC)",
    "towa corporation": "TOWA CORPORATION",
    "towa": "TOWA CORPORATION",
    "chroma ate inc": "Chroma ATE Inc.",
    "chroma ate": "Chroma ATE Inc.",
    "kulicke and soffa industries": "Kulicke & Soffa",
    "kulicke & soffa": "Kulicke & Soffa",
}

_LIMITING_PROFILES_CACHE_BY_SECTOR: dict[str, dict[str, Any]] = {}
_LIMITING_PROFILES_COMPANY_SIGNATURE_BY_SECTOR: dict[str, frozenset[str]] = {}



def _load_corrected_all() -> list[dict[str, Any]]:
    with CORRECTED_ALL_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{CORRECTED_ALL_PATH} must contain a JSON list.")
    return rows


@lru_cache(maxsize=1)
def _load_measures_hitl() -> list[dict[str, Any]]:
    with MEASURES_HITL_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{MEASURES_HITL_PATH} must contain a JSON list.")
    return rows


@lru_cache(maxsize=1)
def _load_financial_hitl_equipment() -> list[dict[str, Any]]:
    with FINANCIAL_HITL_EQUIPMENT_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{FINANCIAL_HITL_EQUIPMENT_PATH} must contain a JSON list.")
    return rows


@lru_cache(maxsize=1)
def _load_convertion_rows() -> list[dict[str, Any]]:
    with CONVERTION_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{CONVERTION_PATH} must contain a JSON list.")
    return rows


@lru_cache(maxsize=1)
def _conversion_usd_per_local_by_currency() -> dict[str, float]:
    rates: dict[str, float] = {}
    for row in _load_convertion_rows():
        currency = str(row.get("currency", "")).strip().upper()
        if not currency:
            continue
        rate = row.get("usd_per_1_local_currency")
        try:
            rate_value = float(rate)
        except (TypeError, ValueError):
            continue
        if rate_value > 0:
            rates[currency] = rate_value
    return rates


def _normalize_company_key(name: str) -> str:
    text = str(name).casefold().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(inc|incorporated|corp|corporation|co|ltd|limited|plc|nv|se|ag)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


COMPANY_ALIASES = {
    _normalize_company_key(alias): canonical
    for alias, canonical in COMPANY_ALIASES_RAW.items()
}
_CANONICAL_FROM_MAPPING_BY_NORMALIZED = {
    _normalize_company_key(name): name
    for name in COMPANY_TYPE_MAPPING
}


def _canonical_company_name(name: str) -> str:
    raw_name = str(name).strip()
    key = _normalize_company_key(raw_name)
    if key in COMPANY_ALIASES:
        return COMPANY_ALIASES[key]
    if key in _CANONICAL_FROM_MAPPING_BY_NORMALIZED:
        return _CANONICAL_FROM_MAPPING_BY_NORMALIZED[key]
    return raw_name


@lru_cache(maxsize=1)
def _revenue_musd_by_company_casefold() -> dict[str, float]:
    """Annual revenue in millions of USD, keyed by casefolded canonical name.

    Absolute metrics are divided by this figure. Millions rather than units
    because the quotients are then of the order of 1e-2 to 1e2 instead of 1e-9
    to 1e-4, which keeps the limiting profiles readable. FlowSort compares
    alternatives and profiles on the same criterion, so a common positive factor
    on a criterion leaves every assignment unchanged.
    """

    rates_by_currency = _conversion_usd_per_local_by_currency()
    revenue_by_company: dict[str, float] = {}
    for row in _load_financial_hitl_equipment():
        company = str(row.get("company", "")).strip()
        currency = str(row.get("currency", "")).strip().upper()
        if not company or not currency:
            continue
        raw_value = row.get("value_millions")
        try:
            local_value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if local_value <= 0:
            continue
        usd_per_local = rates_by_currency.get(currency)
        if usd_per_local is None:
            continue
        revenue_musd = local_value * usd_per_local
        if revenue_musd <= 0:
            continue
        canonical = _canonical_company_name(company)
        revenue_by_company[canonical.casefold()] = revenue_musd
    return revenue_by_company


@lru_cache(maxsize=8)
def _metadata_by_code(sector_id: str) -> dict[str, dict[str, Any]]:
    return {item["code"]: item for item in load_criteria(sector_id)}


def _round_float(value: float | None, digits: int = 6) -> float | None:
    """Round for presentation. Never apply to a value that feeds the sorting."""

    if value is None:
        return None
    return round(float(value), digits)


def round_significant(value: float | None, digits: int = 6) -> float | None:
    """Round for presentation when the magnitude is not known in advance.

    Absolute metrics are divided by revenue in USD, which puts them around 1e-7
    to 1e-9. At that magnitude a fixed number of decimal places rounds every
    limiting profile of a criterion to zero, which both destroys the ordering
    that FlowSort requires of the profiles and drags every alternative below r4.
    """

    if value is None:
        return None
    number = float(value)
    if number == 0.0 or not math.isfinite(number):
        return number
    exponent = math.floor(math.log10(abs(number)))
    return round(number, max(digits, digits - 1 - exponent))


def display_profiles(profiles: dict[str, float | None]) -> dict[str, float | None]:
    """Limiting profiles rounded for display; the sorting uses the exact values."""

    return {key: round_significant(value) for key, value in profiles.items()}


def _normalize_company_type(company: str) -> str:
    canonical = _canonical_company_name(company)
    company_type = COMPANY_TYPE_BY_CASEFOLD.get(canonical.casefold())
    if company_type is None:
        raise ValueError(
            f"Company {company!r} (canonical={canonical!r}) is missing from COMPANY_TYPE_MAPPING."
        )
    return company_type


def _is_equipment_company(company: str) -> bool:
    return _normalize_company_type(company) == EQUIPMENT_COMPANY_TYPE


def _is_profile_peer(company: str) -> bool:
    """Whether this company's observations feed the limiting profiles.

    Separate from :func:`_is_equipment_company`, which classifies a company for
    revenue normalisation. Being evaluated does not require being a peer, and
    vice versa: the peer group sets the standard, the alternatives are measured
    against it.
    """

    return _canonical_company_name(company) in PROFILE_PEER_GROUP


def _is_not_reported(value: Any) -> bool:
    return str(value).strip().casefold() in NOT_REPORTED_VALUES


def _clean_numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in NOT_REPORTED_VALUES:
        return None
    normalized = text.replace(",", "").replace("%", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _metric_direction(metadata: dict[str, Any]) -> str:
    category = str(metadata.get("category", "")).upper()
    return "maximize" if "LTB" in category or category == "QUALITATIVE" else "minimize"


def _metric_type(metadata: dict[str, Any], records: list[dict[str, Any]]) -> str:
    category = str(metadata.get("category", "")).strip().lower()
    if category == "qualitative":
        return TYPE_C_QUALITATIVE

    non_missing_records = [record for record in records if not _is_not_reported(record.get("value"))]
    if any(_clean_numeric_value(record.get("value")) is None for record in non_missing_records):
        return TYPE_C_QUALITATIVE

    quantity = str(metadata.get("quantity", ""))
    if "%" in quantity or any("%" in str(record.get("unit", "")) for record in records):
        return TYPE_A_PERCENTAGE
    return TYPE_B_ABSOLUTE


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _normalize_absolute_by_revenue(
    company: str,
    value: float,
    revenue_musd_by_company: dict[str, float],
) -> float | None:
    canonical = _canonical_company_name(company)
    revenue_musd = revenue_musd_by_company.get(canonical.casefold())
    if revenue_musd is None or revenue_musd <= 0:
        return None
    return float(value) / float(revenue_musd)


def _validate_equipment_revenue_coverage(
    companies: list[str],
    revenue_musd_by_company: dict[str, float],
) -> None:
    missing = []
    for company in companies:
        canonical = _canonical_company_name(company)
        if _is_equipment_company(canonical) and canonical.casefold() not in revenue_musd_by_company:
            missing.append({"company": company, "canonical": canonical})
    if missing:
        raise ValueError(f"Missing revenue for equipment companies: {missing}")


def _build_profiles(values: list[float], metric_type: str, direction: str) -> dict[str, float | None]:
    """Limiting profiles r1..r4 at full precision.

    These values are compared against the alternatives by FlowSort, so they are
    deliberately not rounded here: revenue-normalised metrics sit around 1e-7 and
    a fixed-decimal rounding would collapse the four profiles onto zero. Use
    ``display_profiles`` at the point where they are shown.
    """

    if metric_type == TYPE_C_QUALITATIVE:
        # A company that discloses nothing scores QUALITATIVE_NOT_REPORTED_VALUE,
        # below the 1-5 rubric because "nothing disclosed" is not one of its
        # levels. r4 then sits strictly below that, so even a non-disclosing
        # company dominates the bottom profile, as FlowSort requires.
        return {"r1": 5.5, "r2": 3.5, "r3": 2.5, "r4": QUALITATIVE_PROFILE_FLOOR}
    if not values:
        return {"r1": None, "r2": None, "r3": None, "r4": None}

    # r1 and r_{k+1} are the observed extremes rather than percentiles, for both
    # percentage and absolute metrics. FlowSort requires r1 to dominate every
    # alternative and every alternative to dominate r_{k+1}; a percentile leaves
    # part of the population beyond the boundary by construction, and the
    # alternatives being sorted are drawn from the same population that defines
    # the profiles.

    p75 = _percentile(values, 75)
    p25 = _percentile(values, 25)
    minimum = min(values)
    maximum = max(values)

    # FlowSort's assignment rules bound each category as
    # ``phi(r_h) >= phi(a) > phi(r_{h+1})``. The lower comparison is strict, so
    # the bottom profile must be strictly worse than any attainable value. Using
    # the observed extreme leaves a company sitting exactly on r_{k+1} -- which
    # is precisely what the not-reported penalty produces -- and no interval
    # matches. The margin below keeps the boundary outside the attainable range,
    # mirroring what the qualitative profiles already do with r4 = 0 on a 1-5
    # scale. Its magnitude is irrelevant under the usual criterion (t1), which
    # reads only the sign of a difference.
    spread = maximum - minimum
    step = max(abs(spread) * 0.01, abs(maximum) * 1e-9, abs(minimum) * 1e-9, 1e-12)

    # Two steps beyond the observed extreme. The first step is where a
    # non-reporting company lands (see ``worst_attainable_value``): strictly
    # worse than the worst real performer, because non-disclosure is its own
    # failing. The second keeps r_{k+1} strictly below even that, so every
    # alternative dominates it and FlowSort's strict lower bound always matches.
    if direction == "maximize":
        return {"r1": maximum, "r2": p75, "r3": p25, "r4": minimum - 2 * step}
    return {"r1": minimum, "r2": p25, "r3": p75, "r4": maximum + 2 * step}


def worst_attainable_value(values: list[float], metric_type: str, direction: str) -> float | None:
    """Value assigned to a company that does not report this criterion.

    Placed one step beyond the worst observed performer, so non-disclosure costs
    more than being the weakest company that did disclose. It remains strictly
    above ``r_{k+1}``, which sits a further step out, so the alternative still
    dominates the bottom profile as FlowSort requires. Previously this value and
    ``r_{k+1}`` were the same number, which made the two indistinguishable and
    left the assignment with no matching interval.
    """

    if metric_type == TYPE_C_QUALITATIVE:
        return float(QUALITATIVE_NOT_REPORTED_VALUE)
    if not values:
        return None
    minimum, maximum = min(values), max(values)
    spread = maximum - minimum
    step = max(abs(spread) * 0.01, abs(maximum) * 1e-9, abs(minimum) * 1e-9, 1e-12)
    return float(minimum - step) if direction == "maximize" else float(maximum + step)


def _rows_by_code(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["code"])].append(row)
    return grouped


def _corrected_all_company_signature(rows: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        _canonical_company_name(str(row["company"])).casefold()
        for row in rows
        if row.get("company")
    )


def calculate_flowsort_profiles(sector_id: str = "semiconductors") -> dict[str, Any]:
    metadata_by_code = _metadata_by_code(sector_id)
    rows = _load_corrected_all()
    company_signature = _corrected_all_company_signature(rows)
    cached_signature = _LIMITING_PROFILES_COMPANY_SIGNATURE_BY_SECTOR.get(sector_id)
    cached_profiles = _LIMITING_PROFILES_CACHE_BY_SECTOR.get(sector_id)
    if cached_profiles is not None and cached_signature == company_signature:
        return cached_profiles

    grouped_rows = _rows_by_code(rows)
    revenue_musd_by_company = _revenue_musd_by_company_casefold()
    equipment_companies_in_profiles = sorted(
        {
            _canonical_company_name(str(row["company"]))
            for row in rows
            if row.get("company") and _is_profile_peer(str(row["company"]))
        }
    )
    _validate_equipment_revenue_coverage(equipment_companies_in_profiles, revenue_musd_by_company)

    profiles: dict[str, Any] = {}
    for code, metadata in metadata_by_code.items():
        records = grouped_rows.get(code, [])
        equipment_records = [
            record
            for record in records
            if _is_profile_peer(str(record["company"]))
        ]
        metric_type = _metric_type(metadata, equipment_records)
        direction = _metric_direction(metadata)

        if metric_type == TYPE_C_QUALITATIVE:
            profile_values = _build_profiles([], metric_type, direction)
            profiles[code] = {
                "worst_attainable_value": worst_attainable_value([], metric_type, direction),
                "measure": metadata["measure"],
                "topic": metadata["topic"],
                "domain": metadata["domain"],
                "category": metadata["category"],
                "metric_type": metric_type,
                "direction": direction,
                "profiles": profile_values,
                "valid_count": sum(0 if _is_not_reported(row.get("value")) else 1 for row in equipment_records),
                "profile_company_filter": EQUIPMENT_COMPANY_TYPE,
            }
            continue

        valid_numeric_records = []
        for record in equipment_records:
            numeric_value = _clean_numeric_value(record.get("value"))
            if numeric_value is None:
                continue
            valid_numeric_records.append((record, numeric_value))

        if metric_type == TYPE_A_PERCENTAGE:
            values = [value for _, value in valid_numeric_records]
            profiles[code] = {
                "measure": metadata["measure"],
                "topic": metadata["topic"],
                "domain": metadata["domain"],
                "category": metadata["category"],
                "metric_type": metric_type,
                "direction": direction,
                "profiles": _build_profiles(values, metric_type, direction),
                "worst_attainable_value": worst_attainable_value(values, metric_type, direction),
                "valid_count": len(values),
                "profile_company_filter": EQUIPMENT_COMPANY_TYPE,
            }
            continue

        normalized_values: list[float] = []
        for record, numeric_value in valid_numeric_records:
            normalized_value = _normalize_absolute_by_revenue(
                company=str(record["company"]),
                value=numeric_value,
                revenue_musd_by_company=revenue_musd_by_company,
            )
            if normalized_value is None:
                raise ValueError(
                    f"Missing revenue for equipment company {str(record['company'])!r}. "
                    "Check company name aliases between financial_HITL_equipment.json, "
                    "corrected_all.json, and measures_HITL.json."
                )
            normalized_values.append(normalized_value)

        profiles[code] = {
            "measure": metadata["measure"],
            "topic": metadata["topic"],
            "domain": metadata["domain"],
            "category": metadata["category"],
            "metric_type": metric_type,
            "direction": direction,
            "profiles": _build_profiles(normalized_values, metric_type, direction),
            "worst_attainable_value": worst_attainable_value(
                normalized_values, metric_type, direction
            ),
            "valid_count": len(normalized_values),
            "value_transformation": "raw_metric_value / revenue_in_millions_usd",
            "profile_company_filter": EQUIPMENT_COMPANY_TYPE,
        }

    _LIMITING_PROFILES_COMPANY_SIGNATURE_BY_SECTOR[sector_id] = company_signature
    _LIMITING_PROFILES_CACHE_BY_SECTOR[sector_id] = profiles
    return profiles



def _company_record_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        canonical = _canonical_company_name(str(row["company"]))
        lookup[(canonical, str(row["code"]))] = row
    return lookup


def _worst_limiting_profile_value(profile_entry: dict[str, Any]) -> float:
    """Value assigned to a company that does not report this criterion.

    This is the worst value actually observed, not ``r_{k+1}``. The two used to
    be the same number, which left every non-reporting company sitting exactly on
    the bottom profile; FlowSort's rules bound a category with a strict ``>``
    there, so no interval matched and the assignment fell through to a fallback.
    Treating non-disclosure as "worst observed performer" keeps the penalty
    meaningful while leaving the boundary strictly below it.
    """

    worst = profile_entry.get("worst_attainable_value")
    if worst is not None:
        return float(worst)

    profile_values = profile_entry.get("profiles", {})
    r4_value = profile_values.get("r4")
    if r4_value is None:
        raise ValueError(
            "Criterion has no valid limiting profile r4, so not-reported values cannot be mapped to worst profile."
        )
    return float(r4_value)



def score_quantitative_company_measure(
    company: str,
    company_type: str,
    profile_entry: dict[str, Any],
    record: dict[str, Any] | None,
    revenue_musd_by_company: dict[str, float],
) -> dict[str, Any]:
    """Score one quantitative company measure, including missing-value penalties."""

    canonical_company = _canonical_company_name(company)
    original_value = None if record is None else record.get("value")
    metric_type = profile_entry["metric_type"]
    numeric_value = None if record is None else _clean_numeric_value(record.get("value"))
    reported_numeric_value = numeric_value
    normalization_method = "none"
    if numeric_value is None:
        numeric_value = _worst_limiting_profile_value(profile_entry)
        value_source = "not_reported_as_worst_profile_r4"
        normalization_method = "penalty_to_worst_limiting_profile_r4"
    elif metric_type == TYPE_B_ABSOLUTE and company_type == EQUIPMENT_COMPANY_TYPE:
        revenue_musd = revenue_musd_by_company.get(canonical_company.casefold())
        if revenue_musd is None or revenue_musd <= 0:
            raise ValueError(
                f"Missing revenue for equipment company {company!r}. "
                "Check company name aliases between financial_HITL_equipment.json, "
                "corrected_all.json, and measures_HITL.json."
            )
        numeric_value = float(numeric_value) / float(revenue_musd)
        value_source = "reported_normalized_by_revenue"
        normalization_method = "raw_metric_value / revenue_in_millions_usd"
    else:
        value_source = "reported_numeric"
    return {
        "numeric_value": float(numeric_value),
        "original_value": original_value,
        "reported_numeric_value": None if reported_numeric_value is None else float(reported_numeric_value),
        "normalized_value": float(numeric_value),
        "normalization_method": normalization_method,
        "value_source": value_source,
        "rationale": None,
        "llm_provider": None,
        "llm_model": None,
        "prompt_variant": None,
    }
