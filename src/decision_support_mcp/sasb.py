"""SASB criteria repository used by the MCP server."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

SECTORS = {
    "semiconductors": {
        "id": "semiconductors",
        "name": "Semiconductors",
        "sasb_standard": "Technology & Communications - Semiconductors",
        "resource_uri": "sasb://sectors/semiconductors/criteria",
        "metadata_file": "sasb_semiconductors_metadata.json",
    }
}

DOMAIN_CODE_BY_NAME = {
    "Environmental": "E",
    "Social": "S",
    "Governance": "G",
}


def normalize_sector_id(sector_id: str) -> str:
    normalized = sector_id.strip().lower().replace(" ", "-").replace("_", "-")
    aliases = {
        "semiconductor": "semiconductors",
        "semiconductors": "semiconductors",
        "semi-conductors": "semiconductors",
        "tc-sc": "semiconductors",
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(SECTORS))
        raise ValueError(f"Unsupported sector {sector_id!r}. Supported sectors: {supported}.")
    return aliases[normalized]


def list_sectors() -> list[dict[str, Any]]:
    return [{k: v for k, v in sector.items() if k != "metadata_file"} for sector in SECTORS.values()]


def load_criteria(sector_id: str = "semiconductors") -> list[dict[str, Any]]:
    sector = SECTORS[normalize_sector_id(sector_id)]
    path = DATA_DIR / sector["metadata_file"]
    with path.open("r", encoding="utf-8") as f:
        criteria = json.load(f)
    return criteria


def get_criterion(code: str, sector_id: str = "semiconductors") -> dict[str, Any]:
    normalized_code = code.strip().upper()
    for criterion in load_criteria(sector_id):
        if criterion["code"].upper() == normalized_code:
            return criterion
    raise ValueError(f"Criterion code {code!r} was not found for sector {sector_id!r}.")


def filter_criteria(
    sector_id: str = "semiconductors",
    domain: str | None = None,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    criteria = load_criteria(sector_id)
    if domain:
        criteria = [c for c in criteria if c.get("domain", "").lower() == domain.lower()]
    if topic:
        criteria = [c for c in criteria if c.get("topic", "").lower() == topic.lower()]
    return criteria


def build_domain_hierarchy(sector_id: str = "semiconductors") -> dict[str, list[dict[str, Any]]]:
    hierarchy = {"E": [], "S": [], "G": []}
    for criterion in load_criteria(sector_id):
        domain_name = str(criterion.get("domain", "")).strip()
        domain_code = DOMAIN_CODE_BY_NAME.get(domain_name)
        if domain_code is not None:
            hierarchy[domain_code].append(criterion)
    return hierarchy


def build_topic_hierarchy(sector_id: str = "semiconductors") -> dict[str, list[dict[str, Any]]]:
    hierarchy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for criterion in load_criteria(sector_id):
        topic = str(criterion.get("topic", "")).strip()
        if topic:
            hierarchy[topic].append(criterion)
    return dict(hierarchy)
