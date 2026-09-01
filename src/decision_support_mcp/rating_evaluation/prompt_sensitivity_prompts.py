"""The four scoring prompts compared by the prompt-sensitivity analysis.

The four prompts differ only in how much scaffolding they give the model, so the
analysis isolates the effect of prompt detail on the FlowSort classification:

===========  ==============  ========  ==================================
prompt       SASB guidance   rubric    rating anchored by
===========  ==============  ========  ==================================
``prompt_1``  yes            yes       the five rubric levels
``prompt_2``  no             yes       the five rubric levels
``prompt_3``  yes            no        a bare 1-5 scale
``prompt_4``  no             no        a bare 1-5 scale
===========  ==============  ========  ==================================

The texts are transcribed verbatim from Tables 3-6 of the paper. They are held
fixed once collection starts: every stored rating is tagged with the prompt's
``sha256`` so a silently edited prompt is detectable rather than invisible.

``prompt_1`` is not redefined here. It is :data:`..prompts.SCORING_PROMPT`
imported, the same object the rater-agreement evaluation scores with, so the
baseline of this analysis and the baseline of that one are the same text by
construction and cannot drift apart.
"""

from __future__ import annotations

import hashlib

from src.decision_support_mcp.rating_evaluation.prompts import (
    MAX_RATING,
    MIN_RATING,
    RATING_LABELS,
    RUBRIC_SHA256,
    SCORING_PROMPT,
    SCORING_RUBRIC,
)

__all__ = [
    "MAX_RATING",
    "MIN_RATING",
    "PROMPT_IDS",
    "PROMPT_LABELS",
    "PROMPT_SHA256_BY_ID",
    "PROMPT_TEMPLATES",
    "PROMPT_FEATURES",
    "RATING_LABELS",
    "RUBRIC_SHA256",
    "SCORING_RUBRIC",
    "build_prompt",
]


#: Table 3. The study's baseline prompt, imported rather than copied so the
#: rater-agreement evaluation and this analysis can never diverge.
_PROMPT_1 = SCORING_PROMPT


_PROMPT_2 = '''You are an ESG expert specializing in the semiconductor equipment industry. Your task is to assess the adequacy of the company's ESG disclosure for one specific Sustainability Accounting Standards Board (SASB) measure.

**SASB Measure**:
"""{measure}"""

**Company Disclosure**:
"""{value}"""

**Scoring Rubric**:
"""{rubric}"""

**Rules**:
Use only the Company Disclosure as evidence about the company.
Use the Scoring Rubric only to determine the final rating.
Treat information that is not explicitly stated in the Company Disclosure as not disclosed.
Do not use external knowledge or infer information about the company.
**Assessment Instruction**:
Identify the information explicitly supported by the Company Disclosure.
Compare the information with every level of the Scoring Rubric.
Assign the highest rating whose core description is fully supported by the Company Disclosure.
When the evidence falls between two rating levels, assign the lower rating unless every requirement of the higher rating is explicitly supported.
Assign ratings 1 and 5 whenever their respective rubric definitions are satisfied. Do not prefer intermediate ratings just because they appear more conservative.
Write a concise justification of two or three sentences explaining why the Company Disclosure supports the selected rating.

**Output Format**:
Return only a single valid JSON object with exactly two fields:
{{
  "justification": "string",
  "rating": integer
}}

The rating must be an integer from 1 to 5.
Do not include text before or after the JSON.'''


_PROMPT_3 = '''You are an ESG expert specializing in the semiconductor equipment industry. Your task is to assess the adequacy of the company's ESG disclosure for one specific Sustainability Accounting Standards Board (SASB) measure.

**SASB Measure**:
"""{measure}"""

**SASB Measure Guidance**:
"""{guidance}"""

**Company Disclosure**:
"""{value}"""

**Rules**:
Use only the Company Disclosure as evidence about the company.
Use the SASB Measure Guidance only to determine which information is expected for the measure.
Treat information that is not explicitly stated in the Company Disclosure as not disclosed.
Do not use external knowledge or infer information about the company.
**Assessment Instruction**:
Identify the information explicitly supported by the Company Disclosure.
Compare the information with the SASB Measure Guidance.
Assign an integer rating from 1 to 5, where 1 indicates that the topic is mentioned only in generic or vague terms and 5 indicates the highest-quality disclosure. Do not prefer intermediate ratings just because they appear more conservative.
Write a concise justification of two or three sentences explaining why the Company Disclosure supports the selected rating.

**Output Format**:
Return only a single valid JSON object with exactly two fields:
{{
  "justification": "string",
  "rating": integer
}}

The rating must be an integer from 1 to 5.
Do not include text before or after the JSON.'''


_PROMPT_4 = '''You are an ESG expert specializing in the semiconductor equipment industry. Your task is to assess the adequacy of the company's ESG disclosure for one specific Sustainability Accounting Standards Board (SASB) measure.

**SASB Measure**:
"""{measure}"""

**Company Disclosure**:
"""{value}"""

**Rules**:
Use only the Company Disclosure as evidence about the company.
Treat information that is not explicitly stated in the Company Disclosure as not disclosed.
Do not use external knowledge or infer information about the company.
**Assessment Instruction**:
Identify the information explicitly supported by the Company Disclosure.
Assign an integer rating from 1 to 5, where 1 indicates that the topic is mentioned only in generic or vague terms and 5 indicates the highest-quality disclosure. Do not prefer intermediate ratings just because they appear more conservative.
Write a concise justification of two or three sentences explaining why the Company Disclosure supports the selected rating.

**Output Format**:
Return only a single valid JSON object with exactly two fields:
{{
  "justification": "string",
  "rating": integer
}}

The rating must be an integer from 1 to 5.
Do not include text before or after the JSON.'''


PROMPT_TEMPLATES: dict[str, str] = {
    "prompt_1": _PROMPT_1,
    "prompt_2": _PROMPT_2,
    "prompt_3": _PROMPT_3,
    "prompt_4": _PROMPT_4,
}

#: Fixed presentation order, most scaffolding first.
PROMPT_IDS: tuple[str, ...] = ("prompt_1", "prompt_2", "prompt_3", "prompt_4")

PROMPT_LABELS: dict[str, str] = {
    "prompt_1": "Prompt 1 | guidance + rubric",
    "prompt_2": "Prompt 2 | rubric only",
    "prompt_3": "Prompt 3 | guidance only",
    "prompt_4": "Prompt 4 | disclosure only",
}

#: Which scaffolding each prompt carries, so a result table can be read without
#: re-reading the prompt texts.
PROMPT_FEATURES: dict[str, dict[str, bool]] = {
    "prompt_1": {"sasb_guidance": True, "scoring_rubric": True},
    "prompt_2": {"sasb_guidance": False, "scoring_rubric": True},
    "prompt_3": {"sasb_guidance": True, "scoring_rubric": False},
    "prompt_4": {"sasb_guidance": False, "scoring_rubric": False},
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PROMPT_SHA256_BY_ID: dict[str, str] = {
    prompt_id: _sha256(template) for prompt_id, template in PROMPT_TEMPLATES.items()
}


def build_prompt(prompt_id: str, measure: str, guidance: str, disclosure: str) -> str:
    """Render one scoring prompt for one company-sub-criterion unit.

    ``guidance`` and ``rubric`` are formatted into every template that declares
    them; templates that omit a field simply never reference it, so the same
    call signature serves all four prompts.
    """

    key = str(prompt_id or "").strip().lower()
    if key not in PROMPT_TEMPLATES:
        valid = ", ".join(PROMPT_IDS)
        raise ValueError(f"Unknown prompt id {prompt_id!r}. Expected one of: {valid}.")
    return PROMPT_TEMPLATES[key].format(
        measure=measure,
        guidance=guidance or "No SASB guidance available.",
        value=disclosure,
        rubric=SCORING_RUBRIC,
    )
