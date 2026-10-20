"""LLMs take the released TIMSS mathematics items as a test, and are scored with the IRT models and parameters.

A condition is an LLM with its settings (the provider, model, reasoning effort, ``max_tokens``, and the rest of
``LLMSettings``) and the role it takes: ``"test_taker"``, the test written for students of a grade (evaluating the
LLM), or ``"student"``, as a student of that grade (the LLM as a participant). Every condition answers the same units
with the same prompts, apart from the role; an LLM scorer scores the constructed responses with each item's scoring
guide, and each condition's θ is estimated on each calibration's scale and put on the TIMSS reporting scale.

The two roles make two uses: as test takers, the LLMs take a benchmark of 724 tasks whose difficulty for students is
known; as students, they are synthetic participants, whose answers can be checked against real students' percent
correct and the IRT models. ``notebooks/llm_test_taking.ipynb`` shows both.

The modules, with the settings of the run of 2026-10-19 as defaults:

- ``units``: the released items with calibrated parameters (724), grouped into the units a student sees together
  (652): ``ItemUnits`` and ``Unit``.
- ``answering``: the prompts and output schema with which a condition answers a unit, by role.
- ``conditions``: ``Condition`` (an LLM and a role), ``conditions_of``, ``DEFAULT_MODELS``, ``DEFAULT_SCORER``, and
  the cost of the answers (``TOKEN_PRICES`` for providers that report none).
- ``requesting``: requests saved as JSON records, sent directly or as OpenRouter batches (``send``, ``RunReport``).
- ``scoring``: the scorer's prompts and schema, the points of the scoring-guide categories, and the scores.
- ``irt``: the 3PL, 2PL, and GPCM models (``ItemModel``), the EAP and MLE of θ (``estimate_theta``), and their checks.
- ``scales``: the calibrations and their TIMSS reporting scales (``Calibrations``), and the International Benchmarks.
- ``results``: ``ScoringResults``, the scores of the conditions on the IRT and TIMSS scales, and their workbook.
- ``administration``: ``Administration``, which puts the steps together.
- ``plots``: θ of each condition by calibration, and the item scores on the IRT scale.
- ``cli``: the ``timss-llm-test`` command.

It uses the items, LLM client, and OpenRouter batches that the package shares (``timss_math.items``, ``llm``, and
``batch``), and the scale constants of its ``data`` directory. ``notebooks/llm_test_taking.ipynb`` runs the whole
analysis, step by step.

Example::

    from timss_math.llm_test_taking import Administration, LLMSettings, conditions_of, plots

    administration = Administration()  # the ten conditions and the scorer of the run of 2026-10-19
    results = administration.results()  # from the saved answers and verdicts, without any request
    results.score_table(lambda row: f"{row.timss_eap:.0f}, {row.timss_benchmark}")
    plots.theta_figure(results)

    models = [LLMSettings(model="openai/gpt-6-luna", reasoning_effort="high", max_tokens=8192)]
    scorer = LLMSettings(model="google/gemini-3.8-flash", reasoning_effort="medium", max_tokens=8192)
    administration = Administration(conditions_of(models, roles=["student"]), scorer)
    administration.request_answers(max_requests=1)  # one unit; None answers every unit, which costs money
    administration.request_verdicts(max_requests=None)
    administration.results().to_excel("llm_answers_scored.xlsx")
"""

from ..llm import LLMSettings
from .administration import Administration
from .answering import ROLES, Role
from .conditions import (
    DEFAULT_MODELS,
    DEFAULT_SCORER,
    TOKEN_PRICES,
    Condition,
    conditions_of,
)
from .irt import ItemModel, estimate_theta
from .requesting import Request, RunReport
from .results import ScoringResults
from .scales import Calibration, Calibrations, benchmark
from .units import ItemUnits, Unit

__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_SCORER",
    "ROLES",
    "TOKEN_PRICES",
    "Administration",
    "Calibration",
    "Calibrations",
    "Condition",
    "ItemModel",
    "ItemUnits",
    "LLMSettings",
    "Request",
    "Role",
    "RunReport",
    "ScoringResults",
    "Unit",
    "benchmark",
    "conditions_of",
    "estimate_theta",
]
