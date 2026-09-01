"""Interactive MCP client for sustainable partner criteria formulation."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl


PROJECT_ROOT = Path(__file__).resolve().parent
SERVER_PATH = PROJECT_ROOT / "server.py"
VALID_TERMS = {"EI", "WI", "FI", "VI", "AI"}
TERM_HELP = {
    "EI": "Equally important",
    "WI": "Weakly important",
    "FI": "Fairly important",
    "VI": "Very important",
    "AI": "Absolutely important",
}


def clear_line() -> None:
    print()


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else (default or "")


def print_header(title: str) -> None:
    clear_line()
    print("=" * 78)
    print(title)
    print("=" * 78)


def criterion_label(criterion: dict[str, Any]) -> str:
    return f"{criterion['code']} - {criterion.get('measure', criterion['code'])}"


def compact(text: str, width: int = 86) -> str:
    return textwrap.shorten(" ".join(str(text).split()), width=width, placeholder="...")


def print_criteria(criteria: list[dict[str, Any]], title: str = "Criteria") -> None:
    print_header(title)
    if not criteria:
        print("No criteria selected.")
        return
    current_topic = None
    for idx, criterion in enumerate(criteria, 1):
        topic = criterion.get("topic", "Custom")
        if topic != current_topic:
            current_topic = topic
            print(f"\n{topic}")
        domain = criterion.get("domain", "n/a")
        category = criterion.get("category", "n/a")
        print(f"{idx:>2}. {criterion['code']} | {domain} | {category}")
        print(f"    {compact(criterion.get('measure', criterion['code']))}")


def index_by_code(criteria: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(c["code"]).upper(): c for c in criteria}


def resolve_criteria_tokens(tokens: str, criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_code = index_by_code(criteria)
    for token in [t.strip() for t in tokens.replace(";", ",").split(",") if t.strip()]:
        criterion = None
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(criteria):
                criterion = criteria[idx - 1]
        else:
            criterion = by_code.get(token.upper())
        if criterion is None:
            print(f"Skipping unknown item: {token}")
            continue
        if criterion["code"] not in [c["code"] for c in selected]:
            selected.append(criterion)
    return selected


def add_custom_criterion(selected: list[dict[str, Any]]) -> None:
    print_header("Add Custom Criterion")
    existing = {str(c["code"]).upper() for c in selected}
    next_id = 1
    while f"CUSTOM-{next_id:03d}" in existing:
        next_id += 1
    code = ask("Code", f"CUSTOM-{next_id:03d}").upper()
    measure = ask("Measure")
    if not measure:
        print("A measure is required; custom criterion was not added.")
        return
    selected.append(
        {
            "code": code,
            "measure": measure,
            "topic": ask("Topic", "Context-Specific Criterion"),
            "domain": ask("Domain", "Custom"),
            "quantity": ask("Quantity", "n/a"),
            "category": ask("Category", "Qualitative"),
            "normalization_denominator_priority": None,
            "search_terms": [],
            "explanation": ask("Explanation", "Added by the decision maker during criteria formulation."),
            "source": "user",
        }
    )
    print(f"Added {code}.")


def remove_criteria(selected: list[dict[str, Any]]) -> None:
    print_criteria(selected, "Selected Criteria")
    tokens = ask("Enter numbers or codes to remove")
    if not tokens:
        return
    to_remove = {c["code"] for c in resolve_criteria_tokens(tokens, selected)}
    selected[:] = [criterion for criterion in selected if criterion["code"] not in to_remove]
    print(f"Removed {len(to_remove)} criterion/criteria.")


def choose_criterion(criteria: list[dict[str, Any]], prompt: str) -> dict[str, Any]:
    by_code = index_by_code(criteria)
    while True:
        value = ask(prompt)
        if value.isdigit():
            idx = int(value)
            if 1 <= idx <= len(criteria):
                return criteria[idx - 1]
        criterion = by_code.get(value.upper())
        if criterion:
            return criterion
        print("Please enter a valid number or criterion code.")


def ask_linguistic(prompt: str, default: str | None = None) -> str:
    while True:
        value = ask(prompt, default).upper()
        if value in VALID_TERMS:
            return value
        print("Use one of: EI, WI, FI, VI, AI.")


def payload_from_resource(result: Any) -> Any:
    if not result.contents:
        raise RuntimeError("The MCP resource returned no content.")
    return json.loads(result.contents[0].text)


def payload_from_tool(result: Any) -> Any:
    if getattr(result, "isError", False):
        message = result.content[0].text if result.content else "Unknown MCP tool error."
        raise RuntimeError(message)
    if not result.content:
        return None
    content = result.content[0]
    if getattr(content, "type", None) == "text":
        return json.loads(content.text)
    return content


async def fetch_criteria(session: ClientSession, sector_id: str) -> list[dict[str, Any]]:
    result = await session.read_resource(AnyUrl(f"sasb://sectors/{sector_id}/criteria"))
    return payload_from_resource(result)


async def run_weighting(session: ClientSession, selected: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(selected) < 2:
        print("Select at least two criteria before weighting.")
        return None

    print_criteria(selected, "Final Criteria Set")
    print("\nLinguistic scale")
    for term, description in TERM_HELP.items():
        print(f"  {term}: {description}")

    best = choose_criterion(selected, "Best criterion number/code")
    worst = choose_criterion(selected, "Worst criterion number/code")
    while worst["code"] == best["code"]:
        print("Best and worst criteria must be different.")
        worst = choose_criterion(selected, "Worst criterion number/code")

    print_header("Best-to-Others Preferences")
    print(f"Best criterion: {criterion_label(best)}")
    best_to_others: dict[str, str] = {}
    for criterion in selected:
        code = criterion["code"]
        if code == best["code"]:
            best_to_others[code] = "EI"
            continue
        prompt = f"How much more important is the best criterion than {code}?"
        best_to_others[code] = ask_linguistic(prompt, "AI" if code == worst["code"] else None)

    print_header("Others-to-Worst Preferences")
    print(f"Worst criterion: {criterion_label(worst)}")
    others_to_worst: dict[str, str] = {}
    for criterion in selected:
        code = criterion["code"]
        if code == worst["code"]:
            others_to_worst[code] = "EI"
            continue
        if code == best["code"]:
            others_to_worst[code] = best_to_others[worst["code"]]
            print(f"{code} to worst criterion: {others_to_worst[code]} (same direct Best-to-Worst judgment)")
            continue
        prompt = f"How much more important is {code} than the worst criterion?"
        others_to_worst[code] = ask_linguistic(prompt)

    result = await session.call_tool(
        "calculate_fuzzy_bwm_weights",
        {
            "criteria": selected,
            "best_criterion_code": best["code"],
            "worst_criterion_code": worst["code"],
            "best_to_others": best_to_others,
            "others_to_worst": others_to_worst,
        },
    )
    return payload_from_tool(result)


def print_weighting_result(result: dict[str, Any]) -> None:
    print_header("Fuzzy BWM Result")
    print(f"Best criterion:  {result['best_criterion_code']}")
    print(f"Worst criterion: {result['worst_criterion_code']}")
    xi_star = result.get("xi_star", result.get("aggregate_deviation_star"))
    print(f"xi* (deviation): {xi_star}   (0 = fully consistent)")
    print("\nRanking")
    for item in result["ranking"]:
        print(f"{item['rank']:>2}. {item['code']:<18} {item['crisp_weight']:.6f}  {compact(item['measure'], 68)}")

    print("\nWeights")
    for criterion in result["criteria"]:
        fw = criterion["fuzzy_weight"]
        print(
            f"{criterion['code']:<18} crisp={criterion['crisp_weight']:.6f} "
            f"TFN=({fw['l']:.6f}, {fw['m']:.6f}, {fw['u']:.6f})"
        )


async def interactive_client() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
    )
    with open(os.devnull, "w", encoding="utf-8") as errlog:
        async with stdio_client(server_params, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                sectors = payload_from_resource(await session.read_resource(AnyUrl("sasb://sectors")))
                print_header("Sustainable Partner Selection")
                print("Module: Criteria Formulation")
                print("\nAvailable sectors")
                for idx, sector in enumerate(sectors, 1):
                    print(f"{idx}. {sector['name']} ({sector['id']})")

                sector_choice = ask("Select sector", "semiconductors")
                sector_id = "semiconductors" if sector_choice in {"", "1"} else sector_choice
                available = await fetch_criteria(session, sector_id)
                print_criteria(available, "SASB Criteria Returned by MCP Server")

                raw_selection = ask(
                    "Press Enter to start with all criteria, or enter numbers/codes to include",
                    "",
                )
                selected = available[:] if not raw_selection else resolve_criteria_tokens(raw_selection, available)
                if not selected:
                    selected = available[:]

                weighting_result = None
                while True:
                    print_criteria(selected, "Current Criteria Set")
                    print("\nActions: [a] add custom  [r] remove  [b] browse SASB  [w] weight criteria  [q] quit")
                    action = ask("Choose action", "w").lower()
                    if action == "a":
                        add_custom_criterion(selected)
                    elif action == "r":
                        remove_criteria(selected)
                    elif action == "b":
                        print_criteria(available, "SASB Criteria Returned by MCP Server")
                    elif action == "w":
                        weighting_result = await run_weighting(session, selected)
                        if weighting_result:
                            print_weighting_result(weighting_result)
                    elif action == "q":
                        break
                    else:
                        print("Unknown action.")

                print_header("Partner Qualification")
                status = payload_from_tool(await session.call_tool("partner_qualification_status", {}))
                print(status["message"])
                if weighting_result:
                    print("\nCriteria Formulation output is ready for the next module.")


if __name__ == "__main__":
    try:
        asyncio.run(interactive_client())
    except KeyboardInterrupt:
        print("\nSession ended.")
