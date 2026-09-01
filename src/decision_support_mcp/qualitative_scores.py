"""Qualitative ESG disclosure scoring with LLM providers.

The prompts and the rubric are not defined here. They are imported from
``rating_evaluation``, which holds the study's frozen texts: ``prompt_1`` is
Table 3, the prompt validated against the human raters and used to produce the
consensus scores FlowSort reads by default. Scoring a company live and scoring it
through the rater study therefore put the same instruction to the model, on the
same 1-5 scale, so the two paths yield comparable numbers against one set of
limiting profiles.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import error, request

from src.decision_support_mcp.quantitative_scores import (
    _is_not_reported,
    _worst_limiting_profile_value,
)
from src.decision_support_mcp.rating_evaluation.prompt_sensitivity_prompts import (
    MAX_RATING,
    MIN_RATING,
    PROMPT_LABELS,
    PROMPT_TEMPLATES,
    SCORING_RUBRIC,
    build_prompt,
)
from src.decision_support_mcp.rating_evaluation.prompts import SYSTEM_PROMPT

#: No separate system prompt is sent, following the study: the scoring prompt
#: already states the role, and an extra system message would make the live call
#: a different request than the one the rater study validated. Imported rather
#: than redefined so that stays true if the study ever adds one.
QUALITATIVE_RATING_SYSTEM_PROMPT: str | None = SYSTEM_PROMPT

#: The study rubric, levels 1-5. Same object as
#: ``rating_evaluation.prompts.SCORING_RUBRIC``.
QUALITATIVE_SCORING_RUBRIC = SCORING_RUBRIC

#: The four scoring prompts of the prompt-sensitivity analysis, ``prompt_1``
#: being Table 3, the baseline. Aliased under the tool's own name so the MCP
#: tools and the web client keep their existing ``prompt_1``..``prompt_4`` API.
QUALITATIVE_PROMPT_VARIANTS = PROMPT_TEMPLATES

PROMPT_VARIANT_LABELS = PROMPT_LABELS


DEFAULT_SENSITIVITY_MODEL_FAMILIES = [
    {"family": "Qwen Family", "provider": "ollama", "model": "qwen3.5:9b", "enabled": True},
    {"family": "Llama Family", "provider": "ollama", "model": "llama3.1:8b", "enabled": True},
    {"family": "DeepSeek Family", "provider": "ollama", "model": "deepseek-r1:8b", "enabled": True},
    {"family": "GPT Family", "provider": "openai", "model": "gpt-5-mini", "enabled": False},
    {"family": "Gemini Family", "provider": "gemini", "model": "gemini-2.5-flash", "enabled": False},
]


RATER_STUDY_SCORES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval" / "qualitative_consensus_scores.json"
)


def load_rater_study_scores(
    path: Path | None = None,
) -> tuple[dict[tuple[str, str], int | float], dict[str, Any]]:
    """Qualitative scores established by the rater-agreement study.

    Returns a ``(company, code) -> score`` mapping plus the provenance recorded
    with it, so a FlowSort result can state which rubric and which models the
    qualitative scores came from. Missing file yields an empty mapping rather
    than an error: the caller decides whether that is fatal.
    """

    target = path or RATER_STUDY_SCORES_PATH
    if not target.exists():
        return {}, {"available": False, "path": str(target), "reason": "file_not_found"}

    with target.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    scores: dict[tuple[str, str], int | float] = {}
    for entry in payload.get("scores", []):
        if entry.get("score") is None:
            continue
        # A non-integer score is read back as it was written. ``int()`` here used
        # to floor it silently, which turned a mean aggregation of 3.75 into a 3
        # without saying so. Integral values still come back as ``int``, so every
        # file written by the median pipeline behaves exactly as before.
        value = float(entry["score"])
        scores[(str(entry["company"]), str(entry["code"]))] = (
            int(value) if value.is_integer() else value
        )

    provenance = {
        "available": bool(scores),
        "path": str(target),
        "generated_at": payload.get("generated_at"),
        "aggregation": payload.get("aggregation"),
        "scale": payload.get("scale"),
        "n_scores": len(scores),
    }
    provenance.update(payload.get("provenance") or {})
    provenance["coverage"] = payload.get("coverage")
    return scores, provenance


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        match = re.search(r"\{[\s\S]*\}", stripped)
        if not match:
            raise ValueError("LLM response does not contain a JSON object.")
        stripped = match.group(0)
    return json.loads(stripped)


def _build_qualitative_prompt(
    variant: str,
    company: str,
    metadata: dict[str, Any],
    disclosure: str,
) -> str:
    """Render one scoring prompt, byte-identical to what the study sends.

    ``company`` is accepted for call-site compatibility but deliberately unused:
    none of the four prompts names the company, because the rating must rest on
    the disclosure alone rather than on what a model recalls about the firm.

    The disclosure is passed through verbatim. The study templates delimit it
    with triple quotes, so the double-quote escaping this function used to apply
    would now corrupt the text the model reads.
    """

    variant_key = str(variant or "prompt_1").strip().lower()
    if variant_key not in QUALITATIVE_PROMPT_VARIANTS:
        valid = ", ".join(sorted(QUALITATIVE_PROMPT_VARIANTS))
        raise ValueError(f"Unknown prompt variant {variant!r}. Expected one of: {valid}.")
    return build_prompt(
        variant_key,
        measure=metadata["measure"],
        guidance=metadata.get("explanation", "No SASB guidance available."),
        disclosure=disclosure,
    )


def _extract_text_from_openai_response(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                elif isinstance(content.get("value"), str) and content["value"].strip():
                    texts.append(content["value"].strip())
    if texts:
        return "\n".join(texts)

    choices = payload.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    raise ValueError("Could not extract text from OpenAI response.")


def _extract_text_from_gemini_response(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini response did not include candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "").strip() for part in parts if isinstance(part.get("text"), str)]
    joined = "\n".join(text for text in texts if text)
    if not joined:
        raise ValueError("Could not extract text from Gemini response.")
    return joined


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_sec: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail[:400]}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc.reason}") from exc
    return json.loads(raw)


def request_qualitative_rating(
    company: str,
    metadata: dict[str, Any],
    disclosure: str,
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    provider = str(llm_config.get("provider", "ollama")).strip().lower()
    model = str(llm_config.get("model") or "").strip()
    timeout_sec = float(llm_config.get("timeout_sec", 120))
    prompt_variant = str(llm_config.get("prompt_variant") or "prompt_1").strip().lower()
    temperature = float(llm_config.get("temperature", 0.0))
    top_p = float(llm_config.get("top_p", 1.0))
    seed = int(llm_config.get("seed", 42))
    prompt = _build_qualitative_prompt(prompt_variant, company, metadata, disclosure)

    if provider == "ollama":
        model = model or "llama3.1:8b"
        base_url = str(
            llm_config.get("base_url")
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).rstrip("/")
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "top_p": top_p, "seed": seed},
        }
        if QUALITATIVE_RATING_SYSTEM_PROMPT:
            body["system"] = QUALITATIVE_RATING_SYSTEM_PROMPT
        try:
            response_payload = _http_post_json(
                f"{base_url}/api/generate",
                body,
                {"Content-Type": "application/json"},
                timeout_sec,
            )
        except RuntimeError as exc:
            detail = str(exc)
            if "Could not reach" in detail:
                raise RuntimeError(
                    "Ollama is not reachable at "
                    f"{base_url}. Start the local Ollama server and ensure the model "
                    f"{model!r} is available. Typical steps: run `ollama serve` "
                    f"and then `ollama pull {model}`."
                ) from exc
            raise
        raw_text = str(response_payload.get("response", "")).strip()
    elif provider == "openai":
        model = model or "gpt-5-mini"
        api_key = str(
            llm_config.get("api_key")
            or os.getenv(str(llm_config.get("api_key_env") or "OPENAI_API_KEY"))
            or ""
        ).strip()
        if not api_key:
            raise ValueError("OpenAI qualitative scoring requires OPENAI_API_KEY or llm_config.api_key.")
        response_payload = _http_post_json(
            "https://api.openai.com/v1/responses",
            {
                "model": model,
                "input": (
                    ([{"role": "system", "content": QUALITATIVE_RATING_SYSTEM_PROMPT}]
                     if QUALITATIVE_RATING_SYSTEM_PROMPT else [])
                    + [{"role": "user", "content": prompt}]
                ),
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            },
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout_sec,
        )
        raw_text = _extract_text_from_openai_response(response_payload)
    elif provider == "gemini":
        model = model or "gemini-2.5-flash"
        api_key = str(
            llm_config.get("api_key")
            or os.getenv(str(llm_config.get("api_key_env") or "GEMINI_API_KEY"))
            or ""
        ).strip()
        if not api_key:
            raise ValueError("Gemini qualitative scoring requires GEMINI_API_KEY or llm_config.api_key.")
        response_payload = _http_post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    f"{QUALITATIVE_RATING_SYSTEM_PROMPT}\n\n{prompt}"
                                    if QUALITATIVE_RATING_SYSTEM_PROMPT
                                    else prompt
                                )
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "seed": seed,
                },
            },
            {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout_sec,
        )
        raw_text = _extract_text_from_gemini_response(response_payload)
    else:
        raise ValueError("llm_config.provider must be one of: ollama, openai, gemini.")

    parsed = _extract_json_object(raw_text)
    rating = parsed.get("rating")
    rationale = parsed.get("rationale") or parsed.get("justification")
    try:
        rating_value = float(str(rating).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LLM rating for {company}/{metadata['code']} is not numeric: {rating!r}") from exc
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"LLM rationale for {company}/{metadata['code']} is missing.")
    return {
        "rating": max(float(MIN_RATING), min(float(MAX_RATING), rating_value)),
        "rationale": rationale.strip(),
        "provider": provider,
        "model": model,
        "prompt_variant": prompt_variant,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }



def score_qualitative_company_measure(
    company: str,
    code: str,
    metadata: dict[str, Any],
    profile_entry: dict[str, Any],
    record: dict[str, Any] | None,
    llm_config: dict[str, Any] | None,
    qualitative_cache: dict[tuple[str, str, str], dict[str, Any]],
    precomputed_score: float | None = None,
    precomputed_source: str | None = None,
) -> dict[str, Any]:
    """Score one qualitative company disclosure.

    Three paths, in priority order: a score already established by the rater
    study, a live LLM call, or the worst-profile penalty when nothing is
    reported. The study score is preferred because it is the median across
    several models rather than one model's single opinion, and because it is
    fixed on disk, so a FlowSort run is reproducible instead of depending on
    whichever model happens to be loaded.
    """

    original_value = None if record is None else record.get("value")

    if precomputed_score is not None and record is not None and not _is_not_reported(
        record.get("value")
    ):
        return {
            "numeric_value": float(precomputed_score),
            "original_value": original_value,
            "reported_numeric_value": None,
            "normalized_value": float(precomputed_score),
            "normalization_method": "rater_study_median",
            "value_source": precomputed_source or "rater_study_median",
            "rationale": (
                "Median rating across the study models, from the rater-agreement "
                "evaluation."
            ),
            "llm_provider": None,
            "llm_model": None,
            "prompt_variant": None,
        }

    if record is None or _is_not_reported(record.get("value")):
        numeric_value = _worst_limiting_profile_value(profile_entry)
        return {
            "numeric_value": numeric_value,
            "original_value": original_value,
            "reported_numeric_value": None,
            "normalized_value": float(numeric_value),
            "normalization_method": "penalty_to_worst_limiting_profile_r4",
            "value_source": "not_reported_as_worst_profile_r4",
            "rationale": "Not reported.",
            "llm_provider": None,
            "llm_model": None,
            "prompt_variant": None,
        }

    if llm_config is None:
        raise ValueError(
            f"Criterion {code} is qualitative and has no score for {company}. Either run the "
            "Rating Evaluation so the study median is available, or pass llm_config with "
            "provider/model details to score it live."
        )

    cache_key = (
        company,
        code,
        json.dumps(llm_config, sort_keys=True, ensure_ascii=False),
    )
    if cache_key not in qualitative_cache:
        qualitative_cache[cache_key] = request_qualitative_rating(
            company=company,
            metadata=metadata,
            disclosure=str(record.get("value")),
            llm_config=llm_config,
        )
    llm_result = qualitative_cache[cache_key]
    return {
        "numeric_value": float(llm_result["rating"]),
        "original_value": original_value,
        "reported_numeric_value": None,
        "normalized_value": float(llm_result["rating"]),
        "normalization_method": f"llm_qualitative_rating_{MIN_RATING}_to_{MAX_RATING}",
        "value_source": "llm_scored_text",
        "rationale": llm_result["rationale"],
        "llm_provider": llm_result["provider"],
        "llm_model": llm_result["model"],
        "prompt_variant": llm_result["prompt_variant"],
    }


_request_qualitative_rating = request_qualitative_rating
