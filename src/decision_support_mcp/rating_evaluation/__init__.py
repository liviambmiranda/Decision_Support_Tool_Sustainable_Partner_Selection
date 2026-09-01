"""Standalone rater-agreement evaluation for qualitative ESG sub-criteria.

The dependency runs one way: this package imports only read-only dataset helpers
from the tool, never the reverse, so an evaluation can never be perturbed by the
tool's own state. Its products are a consensus score file that FlowSort consumes,
and the frozen prompt texts in :mod:`.prompts` and
:mod:`.prompt_sensitivity_prompts`.

Those prompt modules are the one exception to the isolation, by design:
``qualitative_scores.py`` imports them so that scoring a company live uses the
same instruction and the same 1-5 rubric the study validated against the human
raters. They import nothing themselves beyond ``hashlib``, so the direction of
dependency stays acyclic.
"""
