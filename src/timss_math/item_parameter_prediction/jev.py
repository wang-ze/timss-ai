"""Predict the IRT parameters of a new item with Jev, with or without the most similar calibrated items.

Jev (TypeSafe's ``typesafe/jev-1.13``, through OpenRouter or TypeSafe's own API, with ``typesafe-sdk``) answers
questions about a state, a JSON object, instead of following a prompt. A score question offers an ordered rubric of
at most 10 criteria, and Jev answers it with a probability for each criterion and a score, the probability-weighted
mean of the criteria's positions (0, 1, ...). So each parameter is asked on a value scale (``ValueScale``), one
criterion per value, such as ``-2.0 (very easy item)``, ``-1.0``, ..., ``2.0 (very hard item)`` for the location, and
Jev's probabilities over the values are a discrete distribution of the parameter (``posterior``). Jev's score is not a
prediction of the parameter, because the values need not be evenly spaced. The point prediction is one of:

- ``eap``: the probability-weighted mean of the values (expected a posteriori), which minimizes the expected squared
  error. The default.
- ``median``: the value at which the cumulative probability reaches 0.5, which minimizes the expected absolute error,
  the mean absolute error that the leave-one-out evaluation reports.
- ``map``: the most likely value (maximum a posteriori), which maximizes the chance of hitting the scale value.

The estimator is applied when a saved answer is read, so changing it requests nothing. Every prediction lies within
the range of its scale, and the spacing of the values limits its precision, so the values chosen for a parameter, and
how finely they cover its likely range, shape the predictions much as a prior does.

``JevPredictor.prepare`` builds what Jev sees, the state: the item's fields (stem, options and answer, scoring guide,
figure description, and classification; never its ID, percent correct, or parameters) and the name of its
calibration, and with ``with_references=True`` also its reference items (``Predictor.references``) with their percent
correct, calibration, and calibrated parameters. Each question names the item's IRT model, points, and calibration,
defines the parameter, and, with reference items, ends with ``REFERENCE_NOTE``. With the default scales, the state and
the questions on the slope, location, and guessing are byte for byte those of the value-scale sections of
``test_jev.ipynb``, so its saved answers are reused. The GPCM steps are defined as TIMSS scales them, theta - b + d_k,
so that scores k - 1 and k are equally likely at theta = b - d_k, as in the LLM's system prompt.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

import httpx2
import numpy as np
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, Score, ScoreAnswer

from ..config import api_key, slug
from ..items import TimssItem, model_parameters, predicted_parameters
from ..running import UnusableOutputError
from .index import ITEM_PARAMETERS_PAYLOAD, ItemIndex, SearchMethod
from .predictor import Predictor

JevProvider = Literal["openrouter", "typesafe"]
Estimator = Literal["eap", "median", "map"]
ESTIMATORS = ("eap", "median", "map")
CONTEXTS = ("item", "item+references")
BASE_URLS = {"openrouter": "https://openrouter.ai/api", "typesafe": "https://api.typesafe.ai"}
API_KEY_VARIABLES = {"openrouter": "OPENROUTER_API_KEY", "typesafe": "TYPESAFE_API_KEY"}
COST_SOURCES = {"openrouter": "OpenRouter response", "typesafe": "TypeSafe response"}
# Jev rounds its probabilities to two decimals, so they sum to about 1.
PROBABILITY_SUM_TOLERANCE = 0.05
MAX_SCORE_LEVELS = 10
"""The most criteria that Jev accepts in a score question."""


@dataclass(frozen=True)
class JevSettings:
    """Which Jev model answers, and how it is called."""

    provider: JevProvider = "openrouter"
    model: str = "typesafe/jev-1.13"
    """The model's name at the provider: ``typesafe/jev-1.13`` on OpenRouter, ``jev-1.13`` on TypeSafe's API. A dated
    version keeps runs comparable; ``jev-latest`` would change under them."""
    base_url: str | None = None
    """The provider's API URL; None uses its public one."""
    api_key: str | None = field(default=None, repr=False)
    """None reads the provider's variable (OPENROUTER_API_KEY or TYPESAFE_API_KEY) from the environment or the
    project .env."""
    timeout: float = 120
    rate_limit_retries: int = 3
    """Retries of a request answered with HTTP 429 (rate limit), which OpenRouter does not bill. Nothing else is
    retried, so no answer is paid for twice."""

    def __post_init__(self):
        if self.provider not in BASE_URLS:
            raise ValueError(f"Unknown Jev provider {self.provider!r}; use one of {sorted(BASE_URLS)}")
        if self.provider == "typesafe" and self.model.startswith("typesafe/"):
            raise ValueError(f"TypeSafe's API names its models without 'typesafe/', such as {self.model[9:]!r}")

    def describe(self) -> dict:
        return {name: value for name, value in asdict(self).items() if name != "api_key"}


@dataclass(frozen=True)
class ValueScale:
    """The ordered criteria of a score question on one parameter, one per value, from the lowest: each starts with its
    value, such as ``0.0 (no guessing)``, and Jev picks the value closest to the parameter."""

    criteria: tuple[str, ...]
    values: tuple[float, ...] = field(init=False, repr=False)
    """The value of each criterion: its first word, as a number."""

    def __post_init__(self):
        criteria = tuple(self.criteria)
        if not 2 <= len(criteria) <= MAX_SCORE_LEVELS:
            raise ValueError(f"A value scale needs 2 to {MAX_SCORE_LEVELS} criteria, not {len(criteria)}: {criteria}")
        try:
            values = tuple(float(criterion.split()[0]) for criterion in criteria)
        except (ValueError, IndexError) as error:
            raise ValueError(f"Each criterion must start with its value: {criteria}") from error
        if any(later <= earlier for earlier, later in itertools.pairwise(values)):
            raise ValueError(f"The values of a scale must increase: {criteria}")
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "values", values)

    @property
    def ticks(self) -> list[str]:
        """Each criterion's value as written, for graphs."""
        return [criterion.split()[0] for criterion in self.criteria]

    def nearest(self, values) -> np.ndarray:
        """The position (from 0) of the scale value closest to each value; NaN stays NaN."""
        values = np.asarray(values, dtype=float)
        grid = np.asarray(self.values)
        positions = np.abs(np.nan_to_num(values)[..., None] - grid).argmin(axis=-1).astype(float)
        return np.where(np.isnan(values), np.nan, positions)


STEP1_SCALE = ("-3", "-2", "-1.5", "-1", "-0.5", "0", "1", "2")
# The value scales of the first Jev test, criteria as written there, and one for the steps of the GPCM items.
DEFAULT_VALUE_SCALES: Mapping[str, ValueScale] = MappingProxyType(
    {
        "slope": ValueScale(("0.4", "0.6", "0.8", "1.0", "1.2", "1.4", "1.6", "1.8", "2.0 (high discrimination)")),
        "location": ValueScale(
            ("-2.0 (very easy item)", "-1.0", "-0.5", "0.0", "0.2", "0.5", "0.8", "1.0", "1.5", "2.0 (very hard item)")
        ),
        "guessing": ValueScale(("0.0 (no guessing)", "0.1", "0.15", "0.2", "0.25", "0.4")),
        "step1": ValueScale(STEP1_SCALE),
        "step2": ValueScale(STEP1_SCALE),  # asked only of the three 3-point items
    }
)
IRT_MODEL_DESCRIPTIONS = {
    "3PL": "the three-parameter logistic (3PL) model, which TIMSS uses for multiple-choice items",
    "2PL": "the two-parameter logistic (2PL) model, which TIMSS uses for constructed-response items worth 1 point",
    "GPCM": "the generalized partial credit model (GPCM), which TIMSS uses for constructed-response items worth 2 or 3 points",
}
ORDINALS = {1: "first", 2: "second", 3: "third"}


def step_question(step: int) -> str:
    """The question on the GPCM step ``step``, as TIMSS scales it: P(score = k) is proportional to exp(sum over v of
    1.7 a (theta - b + d_v)), so scores k - 1 and k are equally likely at theta = b - d_k."""
    earlier = "no point" if step == 1 else f"only the {ORDINALS[step - 1]}" if step == 2 else f"{step - 1} points"
    return (
        f"What is the item's step {step} parameter d{step}? In the GPCM, b - d{step} is the ability at which earning the "
        f"{ORDINALS[step]} point becomes more likely than earning {earlier}, so a higher d{step} means that this point "
        "is earned at a lower ability; the steps sum to 0."
    )


PARAMETER_QUESTIONS: Mapping[str, str] = MappingProxyType(
    {
        "slope": "What is the item's slope (discrimination) parameter a? A higher a means that the item separates "
        "students of lower and higher mathematics ability more sharply.",
        "location": "What is the item's location (difficulty) parameter b? b is on a logit scale on which the students "
        "of the calibration average about 0; a higher b means a harder item.",
        "guessing": "What is the item's guessing (lower asymptote) parameter c: the probability that a student of very "
        "low ability answers it correctly, such as by guessing among the options?",
        "step1": step_question(1),
        "step2": step_question(2),
    }
)
# Added to the instructions when Jev also sees reference items, which the state's field names alone do not explain.
REFERENCE_NOTE = (
    "The state also lists reference items: the calibrated TIMSS items of the same grade most similar to this one "
    "(never the item itself or its other parts), most similar first, with their percent correct and their calibrated "
    "parameters, each on the scale of its own calibration. Use them to place this item."
)


class JevAnswerError(UnusableOutputError):
    """Jev's answers do not fit the questions: an answer is missing, of another type, or has probabilities off the
    rubric; a rerun may succeed."""


def posterior(probabilities: Sequence[float], values: Sequence[float]) -> dict[str, float]:
    """The point predictions and spread of a parameter from Jev's probabilities over the values of its scale.

    The probabilities are first scaled to sum to 1 (Jev rounds them to two decimals). ``map`` is the most likely value
    (the lowest of tied ones) with its probability ``map_probability``, ``eap`` the probability-weighted mean, ``sd``
    the standard deviation around it, and ``median`` the lowest value at which the cumulative probability reaches 0.5,
    or the midpoint of two values when it is exactly 0.5 at the first of them.
    """
    p = np.asarray(probabilities, dtype=float)
    v = np.asarray(values, dtype=float)
    if p.shape != v.shape or (p < 0).any() or p.sum() <= 0:
        raise ValueError(f"Probabilities {p.tolist()} do not fit the values {v.tolist()}")
    p = p / p.sum()
    cumulative = np.cumsum(p)
    middle = int(np.searchsorted(cumulative, 0.5 - 1e-9))
    median = v[middle]
    if abs(cumulative[middle] - 0.5) <= 1e-9 and middle + 1 < len(v):
        median = (v[middle] + v[middle + 1]) / 2
    mode = int(p.argmax())
    eap = float(p @ v)
    return {
        "map": float(v[mode]),
        "map_probability": float(p[mode]),
        "eap": eap,
        "median": float(median),
        "sd": float(np.sqrt(p @ (v - eap) ** 2)),
    }


def answer_problems(answers: Mapping, questions: Mapping[str, Score]) -> list[str]:
    """What is wrong with Jev's answers: an answer missing, of another type, or with probabilities off the rubric."""
    problems = [f"no answer to {name}" for name in questions if name not in answers]
    problems += [f"unexpected answer {name}" for name in answers if name not in questions]
    for name, question in questions.items():
        answer = answers.get(name)
        if answer is None:
            continue
        if not isinstance(answer, ScoreAnswer):
            problems.append(f"{name} is a {answer.type} answer, not a score")
        elif set(answer.probabilities) != set(range(len(question.criteria))):
            problems.append(f"{name} has probabilities for {sorted(answer.probabilities)}")
        elif abs(sum(answer.probabilities.values()) - 1) > PROBABILITY_SUM_TOLERANCE:
            problems.append(f"{name}'s probabilities sum to {sum(answer.probabilities.values()):.3f}")
    return problems


def item_fields(item: TimssItem) -> dict:
    """What Jev sees of an item: its TimssItem fields (no percent correct or parameters), without its ID."""
    return item.model_dump(mode="json", exclude_none=True, exclude={"item_id"})


class JevPredictor(Predictor):
    """Predicts the IRT parameters of TimssItems with Jev, from the item alone or with similar calibrated items of an
    ItemIndex."""

    kind = "jev"

    def __init__(
        self,
        index: ItemIndex,
        jev: JevSettings = JevSettings(),
        with_references: bool = True,
        estimator: Estimator = "eap",
        scales: Mapping[str, ValueScale | Sequence[str]] | None = None,
        questions: Mapping[str, str] | None = None,
        search_method: SearchMethod = "hybrid",
        n_similar: int = 10,
        n_same_model: int = 5,
    ):
        """``jev`` is the Jev model and how it is called. ``with_references`` adds the reference items to what Jev
        sees, found by ``search_method``, ``n_similar``, and ``n_same_model`` (``Predictor``); without them, the index
        only supplies the item bank. ``estimator`` turns Jev's probabilities into a value (``eap``, ``median``, or
        ``map``).

        ``scales`` replaces the value scales of ``DEFAULT_VALUE_SCALES`` of the parameters it names, each with a
        ``ValueScale`` or its criteria, and ``questions`` replaces their questions of ``PARAMETER_QUESTIONS``.
        """
        super().__init__(index, search_method, n_similar, n_same_model)
        if estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator {estimator!r}; use one of {ESTIMATORS}")
        self.jev = jev
        self.with_references = with_references
        self.estimator = estimator
        self.scales = dict(DEFAULT_VALUE_SCALES) | {
            name: scale if isinstance(scale, ValueScale) else ValueScale(tuple(scale))
            for name, scale in (scales or {}).items()
        }
        self.questions = dict(PARAMETER_QUESTIONS) | dict(questions or {})

    @property
    def context(self) -> str:
        """What Jev sees: "item", or "item+references"."""
        return CONTEXTS[self.with_references]

    def describe(self) -> dict:
        """The settings of every step, for run reports: the retrieval only with reference items."""
        jev = self.jev.describe() | {
            "context": self.context,
            "estimator": self.estimator,
            "value_scales": {name: list(scale.criteria) for name, scale in self.scales.items()},
            "questions": self.questions,
        }
        return (self.describe_retrieval() if self.with_references else {}) | {"jev": jev}

    def question(self, item: TimssItem, calibration: str | None, name: str) -> Score:
        """A score question on the parameter ``name`` with the values of its scale as criteria."""
        if name not in self.scales or name not in self.questions:
            raise ValueError(f"No value scale or question for {name}; give both in scales and questions")
        where = f'in the calibration "{calibration.rstrip(".")}"' if calibration else "in its TIMSS calibration"
        instructions = (
            f"TIMSS scaled this {item.points}-point item with {IRT_MODEL_DESCRIPTIONS[item.irt_model]}, {where}. "
            f"{self.questions[name]} Choose the value closest to it."
        )
        if self.with_references:
            instructions += f" {REFERENCE_NOTE}"
        return Score(instructions=instructions, criteria=list(self.scales[name].criteria))

    def reference_state(self, entry: dict) -> dict:
        """A reference item as Jev sees it: its rank, similarity, fields, percent correct, and calibrated parameters."""
        payload = entry["payload"]
        parameters = payload[ITEM_PARAMETERS_PAYLOAD]
        reference = TimssItem.model_validate(
            self.bank.items_by_key[entry["item_key"]] | {"scaled_points": parameters["scaled_points"]}
        )
        fields = {
            "rank": entry["rank"],
            "cosine_similarity": round(entry["cosine_similarity"], 3),
            **item_fields(reference),
            "pct_correct_intl_avg": payload["pct_correct_intl_avg"],
            "calibration": parameters["calibration"],
            "irt_model": parameters["irt_model"],
            "calibrated_parameters": {
                name: parameters[name] for name in model_parameters(parameters["irt_model"], parameters["scaled_points"])
            },
        }
        return {name: value for name, value in fields.items() if value is not None}

    def prepare(self, item: TimssItem, exclude_keys: Iterable[str] = ()) -> dict:
        """What Jev sees (the state) and is asked (a question per predicted parameter) about ``item``, without a
        request. ``exclude_keys`` (such as the item and its other parts) are left out of the reference items."""
        exclude_keys = list(exclude_keys)
        calibration = self.bank.calibrations.get((item.assessment, item.grade))
        state = {"item": item_fields(item) | ({"calibration": calibration} if calibration else {})}
        references = self.references(item, exclude_keys) if self.with_references else []
        if self.with_references:
            state["reference_items"] = [self.reference_state(entry) for entry in references]
        questions = {
            name: self.question(item, calibration, name) for name in predicted_parameters(item.irt_model, item.points)
        }
        return {"exclude_keys": sorted(exclude_keys), "references": references, "state": state, "questions": questions}

    def identity(self, prepared: dict) -> dict:
        """What a saved answer must match to be reused: the provider, the model, the state, and the questions."""
        return {
            "provider": self.jev.provider,
            "model": self.jev.model,
            "state": prepared["state"],
            "questions": {name: question.model_dump(mode="json") for name, question in prepared["questions"].items()},
        }

    def record(self, item: TimssItem, prepared: dict) -> dict:
        """Everything about a prediction except Jev's answers: the item, its reference items, the state, and the
        questions."""
        return self.item_record(item, prepared) | {"context": self.context} | self.identity(prepared)

    def session(self, http_client: httpx2.AsyncClient | None = None) -> AsyncTypeSafeClient:
        """A TypeSafe client for the requests of one event loop, closed at the end; ``http_client`` replaces its HTTP
        client, such as one with a mock transport in tests."""
        jev = self.jev
        retry = RetryPolicy(
            max_retries=jev.rate_limit_retries, http_statuses={429}, api_connection_error=False, api_timeout_error=False
        )
        return AsyncTypeSafeClient(
            api_key=api_key(API_KEY_VARIABLES[jev.provider], jev.api_key),
            base_url=jev.base_url or BASE_URLS[jev.provider],
            model=jev.model,
            retry=retry,
            timeout=jev.timeout,
            http_client=http_client,
        )

    async def request_async(self, item: TimssItem, prepared: dict, session: AsyncTypeSafeClient) -> dict:
        """Ask Jev the questions of ``prepare`` about its state. Returns the answers, the model that answered, the
        usage, cost, and seconds of the request, and the response.

        Raises JevAnswerError for answers that do not fit the questions, and TypeSafeError for API errors.
        """
        started = time.monotonic()
        response = await session.system_one(state=prepared["state"], questions=prepared["questions"])
        seconds = time.monotonic() - started
        problems = answer_problems(response.answers, prepared["questions"])
        if problems:
            raise JevAnswerError("; ".join(problems))
        body = response.raw_http_response.json()
        reported_cost = (body.get("usage") or {}).get("cost")
        return {
            "answers": {name: answer.model_dump(mode="json") for name, answer in response.answers.items()},
            "answered_by": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": None if reported_cost is None else float(reported_cost),
            "cost_source": None if reported_cost is None else COST_SOURCES[self.jev.provider],
            "generation_id": body.get("id"),
            "response": body,
            "seconds": round(seconds, 3),
            "requested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def cache_name(self) -> tuple[str, str]:
        return self.jev.provider, self.jev.model

    def posterior(self, record: dict, name: str) -> dict[str, float]:
        """``posterior`` of Jev's answer on the parameter ``name`` of a saved record, on the values of its question."""
        values = ValueScale(tuple(record["questions"][name]["criteria"])).values
        probabilities = record["answers"][name]["probabilities"]
        return posterior([probabilities[str(position)] for position in range(len(values))], values)

    def parameters(self, record: dict) -> dict[str, float]:
        """Each parameter's ``estimator`` value, and the last GPCM step as minus the sum of the others."""
        values = {name: self.posterior(record, name)[self.estimator] for name in record["questions"]}
        if record["irt_model"] == "GPCM":
            points = record["scaled_points"]
            values[f"step{points}"] = -sum(values[f"step{step}"] for step in range(1, points))
        return values

    def parameter_details(self, record: dict, name: str) -> dict:
        """The posterior of a parameter that Jev was asked about: every point prediction, the spread, Jev's own
        confidence, and the probabilities of the values (JSON)."""
        if name not in record["questions"]:
            return {}
        answer = record["answers"][name]
        probabilities = np.array([answer["probabilities"][str(position)] for position in range(len(answer["probabilities"]))])
        return self.posterior(record, name) | {
            "confidence": answer.get("confidence"),
            "probabilities": json.dumps(np.round(probabilities / probabilities.sum(), 4).tolist()),
        }

    def row_details(self, record: dict) -> dict:
        """Jev, what it saw, its usage and cost, and the state and questions (JSON)."""
        return {
            "jev_provider": record["provider"],
            "jev_model": record["model"],
            "context": record["context"],
            "estimator": self.estimator,
            "answered_by": record["answered_by"],
            **{column: record.get(column) for column in ("input_tokens", "output_tokens", "cost_usd", "seconds")},
            "requested_at": record["requested_at"],
            "state": json.dumps(record["state"], ensure_ascii=False),
            "questions": json.dumps(record["questions"], ensure_ascii=False),
        }

    def run_name(self) -> str:
        """``jev-<provider>-<model>-<estimator>``, with a hash of the scales and questions if they are not the
        defaults, then the retrieval settings, or ``item-alone``."""
        name = f"jev-{self.jev.provider}-{slug(self.jev.model)}-{self.estimator}"
        customized = {
            "scales": {name: list(scale.criteria) for name, scale in self.scales.items()},
            "questions": self.questions,
        }
        defaults = {
            "scales": {name: list(scale.criteria) for name, scale in DEFAULT_VALUE_SCALES.items()},
            "questions": dict(PARAMETER_QUESTIONS),
        }
        if customized != defaults:
            name += "-" + hashlib.sha256(json.dumps(customized, sort_keys=True).encode()).hexdigest()[:8]
        return f"{name}__{self.retrieval_name() if self.with_references else 'item-alone'}"
