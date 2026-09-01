"""Rubric and scoring prompt for rater-agreement.

:data:`SCORING_PROMPT` is Table 3 of the paper, transcribed verbatim. It is the
study's baseline prompt in both senses: the rater-agreement evaluation against
the human raters uses it, and the prompt-sensitivity analysis imports it as
``prompt_1`` and measures the other three prompts against it. Keeping one copy
here rather than two guarantees the two studies cannot drift apart.

Once a collection run has started they must
not be edited, because every stored rating is tagged with ``RUBRIC_SHA256`` and
``PROMPT_SHA256`` for provenance.
"""

from __future__ import annotations

import hashlib

#: Ordered rating labels. Five levels, 1-5 inclusive.
RATING_LABELS: tuple[int, ...] = (1, 2, 3, 4, 5)

MIN_RATING = RATING_LABELS[0]
MAX_RATING = RATING_LABELS[-1]

SCORING_RUBRIC = """1: The topic is mentioned only in generic or vague terms, without specific policies, commitments, or management practices.

2: General policies, commitments, or management practices are described, but their scope and relevance are limited, with little evidence of implementation or outcomes.

3: The disclosure describes management practices with a moderate implementation scope. Material information remains absent, such as responsibilities, targets, performance against targets, or risk management (where applicable).

4: The disclosure describes management practices with a broad implementation scope. It covers most applicable elements, such as responsibilities, targets, performance against targets, or risk management.

5: The disclosure describes management practices with broad implementation scope and clearly demonstrates their effectiveness through documented improvements, progress against targets, or sustained performance over time. It covers all applicable elements, such as responsibilities, targets, performance against targets, or risk management."""

SCORING_PROMPT = '''You are an ESG expert specializing in the semiconductor equipment industry. Your task is to assess the adequacy of the company's ESG disclosure for one specific Sustainability Accounting Standards Board (SASB) measure.

**SASB Measure**:
"""{measure}"""

**SASB Measure Guidance**:
"""{guidance}"""

**Company Disclosure**:
"""{value}"""

**Scoring Rubric**:
"""{rubric}"""

**Rules**:
Use only the Company Disclosure as evidence about the company.
Use the SASB Measure Guidance only to determine which information is expected for the measure.
Use the Scoring Rubric only to determine the final rating.
Treat information that is not explicitly stated in the Company Disclosure as not disclosed.
Do not use external knowledge or infer information about the company.

**Assessment Instruction**:
Identify the information explicitly supported by the Company Disclosure.
Compare the information with the SASB Measure Guidance.
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

#: No separate system prompt is sent. The scoring prompt already states the role,
#: and adding an undocumented system message would introduce an unrecorded
#: variable into the study.
SYSTEM_PROMPT: str | None = None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


RUBRIC_SHA256 = _sha256(SCORING_RUBRIC)
PROMPT_SHA256 = _sha256(SCORING_PROMPT)


def build_prompt(measure: str, guidance: str, disclosure: str) -> str:
    """Render the scoring prompt for one company-sub-criterion unit."""

    return SCORING_PROMPT.format(
        measure=measure,
        guidance=guidance or "No SASB guidance available.",
        value=disclosure,
        rubric=SCORING_RUBRIC,
    )
