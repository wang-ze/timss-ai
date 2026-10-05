"""Predict the IRT parameters of a new item with an LLM grounded in the most similar calibrated items.

``ParameterPredictor.predict(item)`` works in two steps, as ``predict_item_parameters.ipynb`` does:

1. ``prepare`` finds the reference items: the ``n_similar`` calibrated items of the item's grade most similar to it
   (by ``search_method``), then the ``n_same_model`` most similar ones scored with its IRT model and points that are
   not among them, so the LLM always sees calibrated examples of every parameter it predicts. It writes the system
   prompt (an expert in testing the item's grade, content domain, and cognitive domain, and an IRT measurement expert)
   and the user prompt (a calibration summary, the reference items with their parameters, and the new item).
2. ``request`` sends the prompts to the LLM of ``LLMSettings`` and returns its reasoning and parameters, parsed with a
   structured-output schema.

LLM providers:

- ``"openrouter"`` (the default, with ``google/gemini-3.8-flash``): reasoning effort goes in OpenRouter's ``reasoning``
  object, and the response reports the cost.
- ``"openai"``: the OpenAI API, or any OpenAI-compatible server through ``base_url``.
- ``"gemini"``: the Gemini API's OpenAI-compatible endpoint, with models such as ``gemini-3.8-flash``.
- ``"ollama"``: a local Ollama server through its native chat API, which takes the JSON schema as ``format``, the
  reasoning effort as ``think`` (a level such as "medium", or True or False for models without levels; None for
  models that cannot think), and the context window as ``num_ctx``.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Literal

import openai
import requests
from pydantic import BaseModel, Field, create_model

from .config import api_key
from .index import ITEM_PARAMETERS_PAYLOAD, ItemIndex, SearchMethod
from .items import PARAMETER_LABELS, ItemBank, TimssItem, model_parameters, predicted_parameters

LLMProvider = Literal["openrouter", "openai", "gemini", "ollama"]

BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434",
}
API_KEY_VARIABLES = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
STUDENTS = {
    4: "fourth-grade students",
    8: "eighth-grade students",
    12: "students in their final year of secondary school who have taken advanced mathematics courses (TIMSS Advanced)",
}


@dataclass(frozen=True)
class LLMSettings:
    """Which LLM predicts the parameters, and how it is called."""

    provider: LLMProvider = "openrouter"
    model: str = "google/gemini-3.8-flash"
    reasoning_effort: str | bool | None = "medium"
    """Reasoning effort, such as "low", "medium", or "high"; None sends none. Ollama also takes True or False."""
    max_tokens: int = 16384
    """Output cap, reasoning included."""
    base_url: str | None = None
    """The provider's API URL; None uses its public one (Ollama: http://localhost:11434)."""
    api_key: str | None = field(default=None, repr=False)
    """None reads the provider's variable (OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY) from the environment
    or the project .env. Ollama needs none."""
    timeout: float = 600
    max_retries: int = 2
    """Retries of a failed request by the OpenAI client (not used for Ollama)."""
    ollama_num_ctx: int = 32768
    """Ollama context window in tokens, which must hold the prompts (about 6,000 tokens) and the output."""

    def __post_init__(self):
        if self.provider not in BASE_URLS:
            raise ValueError(f"Unknown LLM provider {self.provider!r}; use one of {sorted(BASE_URLS)}")
        if isinstance(self.reasoning_effort, bool) and self.provider != "ollama":
            raise ValueError("Only Ollama takes True or False as the reasoning effort")

    def describe(self) -> dict:
        return {name: value for name, value in asdict(self).items() if name != "api_key"}


class NoPredictionError(RuntimeError):
    """The LLM returned no usable prediction, such as output cut off at max_tokens; a rerun may succeed."""


class ContextWindowError(RuntimeError):
    """The prompts and output do not fit the Ollama context window; raise ollama_num_ctx."""


@dataclass
class Completion:
    parsed: BaseModel
    thinking: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None


class LLMClient:
    """Sends a system and a user prompt to the LLM of an LLMSettings and parses its output with a pydantic schema."""

    def __init__(self, settings: LLMSettings = LLMSettings()):
        self.settings = settings
        self.base_url = (settings.base_url or BASE_URLS[settings.provider]).rstrip("/")

    @functools.cached_property
    def _client(self) -> openai.OpenAI:
        """Made when the first request is sent, so that preparing prompts and reading saved predictions need no API
        key."""
        settings = self.settings
        return openai.OpenAI(
            base_url=self.base_url,
            api_key=api_key(API_KEY_VARIABLES[settings.provider], settings.api_key),
            max_retries=settings.max_retries,
            timeout=settings.timeout,
        )

    def complete(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> Completion:
        """The LLM's output parsed with ``schema``.

        Output that does not fit the schema raises pydantic's ValidationError, output cut off at max_tokens raises
        openai.LengthFinishReasonError (or NoPredictionError from Ollama), and no output at all raises
        NoPredictionError. API errors propagate.
        """
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if self.settings.provider == "ollama":
            return self._ollama(messages, schema)
        return self._openai_compatible(messages, schema)

    def _openai_compatible(self, messages: list[dict], schema: type[BaseModel]) -> Completion:
        settings = self.settings
        effort = settings.reasoning_effort
        if settings.provider == "openrouter":
            extra_body = {"reasoning": {"effort": effort}} if effort is not None else {}
            options = {"max_tokens": settings.max_tokens, "extra_body": extra_body | {"usage": {"include": True}}}
        elif settings.provider == "openai":
            options = {"max_completion_tokens": settings.max_tokens}
        else:
            options = {"max_tokens": settings.max_tokens}
        if settings.provider != "openrouter" and effort is not None:
            options["reasoning_effort"] = effort
        completion = self._client.chat.completions.parse(
            model=settings.model, messages=messages, response_format=schema, **options
        )
        choice = completion.choices[0]
        if choice.message.parsed is None:
            raise NoPredictionError(f"No prediction (finish_reason={choice.finish_reason}): {choice.message.refusal}")
        usage = completion.usage
        details = usage.completion_tokens_details if usage else None
        return Completion(
            parsed=choice.message.parsed,
            # The reasoning summary that OpenRouter returns for a reasoning model, if any.
            thinking=getattr(choice.message, "reasoning", None),
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            reasoning_tokens=details.reasoning_tokens if details else None,
            cost_usd=getattr(usage, "cost", None),
        )

    def _ollama(self, messages: list[dict], schema: type[BaseModel]) -> Completion:
        settings = self.settings
        body = {
            "model": settings.model,
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"num_ctx": settings.ollama_num_ctx, "num_predict": settings.max_tokens},
        }
        if settings.reasoning_effort is not None:
            body["think"] = settings.reasoning_effort
        response = requests.post(f"{self.base_url}/api/chat", json=body, timeout=settings.timeout)
        if not response.ok:
            hint = "; set reasoning_effort=None" if "does not support thinking" in response.text else ""
            raise RuntimeError(f"Ollama {settings.model} returned HTTP {response.status_code}: {response.text}{hint}")
        result = response.json()
        prompt_tokens, completion_tokens = result.get("prompt_eval_count"), result.get("eval_count")
        # Ollama cuts a prompt that does not fit the context window instead of failing.
        if prompt_tokens is not None and prompt_tokens + (completion_tokens or 0) >= settings.ollama_num_ctx:
            raise ContextWindowError(
                f"{prompt_tokens} prompt and {completion_tokens} output tokens fill the context window of "
                f"{settings.ollama_num_ctx}; raise ollama_num_ctx"
            )
        if result.get("done_reason") == "length":
            raise NoPredictionError(f"Output cut off at max_tokens ({settings.max_tokens})")
        content = result["message"].get("content")
        if not content:
            raise NoPredictionError(f"No prediction (done_reason={result.get('done_reason')})")
        return Completion(
            parsed=schema.model_validate_json(content),
            thinking=result["message"].get("thinking"),
            finish_reason=result.get("done_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=None,
            cost_usd=None,
        )


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def system_prompt(item: TimssItem) -> str:
    """System prompt: an expert in testing the subject for the item's grade, content domain, and cognitive domain, and
    a measurement expert in IRT models."""
    subject = item.subject.lower()
    students = STUDENTS[item.grade]
    topic = f", especially {item.main_topic}," if item.main_topic else ""
    return f"""\
You are an expert in testing {subject} for {students}. You are very familiar with assessing these students in the \
{item.content_domain} content domain{topic} and in the {item.cognitive_domain} cognitive domain, as the \
{item.assessment} framework defines them. You have written, reviewed, and analyzed many {subject} items for these \
students, and you know from experience how students across the TIMSS countries perform on them.

You are also a measurement expert in item response theory (IRT). You know the three-parameter logistic (3PL), \
two-parameter logistic (2PL), and generalized partial credit (GPCM) models, how TIMSS calibrates items with them, and \
how content, cognitive demand, the number and difficulty of steps, the numbers involved, reading load, figures, \
response format, distractor plausibility, and scoring rules shape item parameters.

Your task is to predict the IRT parameters that a new item would receive in a TIMSS calibration, using calibrated \
reference items of the same grade.

TIMSS scales items with these models, where theta is student ability and 1.7 is the scaling constant:
- 3PL, for multiple-choice items: P(correct | theta) = c + (1 - c) / (1 + exp(-1.7 a (theta - b))).
- 2PL, for constructed-response items scored 0 or 1: P(correct | theta) = 1 / (1 + exp(-1.7 a (theta - b))).
- GPCM, for constructed-response items scored from 0 to m points with partial credit: P(score = k | theta) is \
proportional to exp(sum over v = 0..k of 1.7 a (theta - b + d_v)), with d_0 = 0 and d_1 + ... + d_m = 0.

The parameters:
- Slope (a): discrimination, how sharply the chance of success rises with ability. Always positive.
- Location (b): overall difficulty on the theta scale. Higher is harder.
- Guessing (c), 3PL only: the chance that a student of very low ability answers correctly. It is usually below 1 \
divided by the number of options, because plausible distractors attract low-ability students.
- Steps (d_1 to d_m), GPCM only: scores k - 1 and k are equally likely at theta = b - d_k. A positive d_1 means the \
first point is reached well below b. When these boundaries are out of order, a middle score is rarely the most \
likely one. The steps sum to 0, so give all steps but the last, which is minus the sum of the others.

The reference items come from the concurrent calibrations of the TIMSS assessments of this grade. TIMSS links each \
calibration to the one before through trend items, so their scales are close but not identical. Each reference item \
names its calibration; when the evidence is otherwise equal, give more weight to reference items from the calibration \
that the new item is predicted for.

How to work:
1. Study the reference items, chosen for their similarity to the new item. Their percent correct shows how hard each \
one was for the students of its assessment, and their calibrated parameters anchor the scale.
2. Compare the new item with the most relevant reference items on content, cognitive demand, the number and difficulty \
of steps, the numbers involved, reading load, and figures, and also on distractor plausibility for a multiple-choice \
item or on how demanding the scoring guide is for a constructed-response item. Decide whether the new item should be \
easier or harder, and more or less discriminating, than each of them.
3. Treat the calibration summary for the new item's model as a prior, and stay within its range unless the new item \
is clearly unusual.
4. Combine the evidence instead of copying the parameters of one reference item.
5. First state which reference items are most comparable and how the new item differs from them. Then, for each \
parameter, give your reasoning before its value."""


def describe_item(item: TimssItem) -> str:
    """Plain-text rendering of a TimssItem, the same for reference items and the new item."""
    topic = f"; main topic: {item.main_topic}" if item.main_topic else ""
    lines = [f"Content domain: {item.content_domain}{topic}; cognitive domain: {item.cognitive_domain}"]
    if item.item_type == "multiple_choice":
        lines.append(f"Format: multiple choice with {len(item.options)} options, 1 point, scored with the 3PL model")
    else:
        points = plural(item.max_points, "point")
        if item.points != item.max_points:
            points += f" in the scoring guide, scaled as {plural(item.points, 'point')}"
        lines.append(f"Format: constructed response, {points}, scored with the {item.irt_model} model")
    lines.append(f"Stem:\n{item.stem}")
    if item.figure_description:
        lines.append(f"Figure, as described by a model (may be inexact):\n{item.figure_description}")
    if item.options:
        lines.append("Options:\n" + "\n".join(f"{option.label}. {option.text}" for option in item.options))
        lines.append(f"Correct option: {item.correct_option}")
    if item.scoring_guide:
        lines.append(
            "Scoring guide:\n"
            + "\n".join(f"- {category.category}: {' | '.join(category.descriptors)}" for category in item.scoring_guide)
        )
    if item.scoring_notes:
        lines.append(f"Scoring notes: {item.scoring_notes}")
    return "\n".join(lines)


def calibration_summary(bank: ItemBank, item: TimssItem, exclude_keys: Iterable[str] = ()) -> str:
    """The distribution of each parameter over the calibrated items of the item's grade scored with its model and
    points (``ItemBank.same_scoring_items``), the prior in the prompt."""
    scoring = f"the {item.irt_model} model" + (f" and {plural(item.points, 'point')}" if item.irt_model == "GPCM" else "")
    calibrated = bank.same_scoring_items(item, exclude_keys)
    if calibrated.empty:
        return f"Calibration summary: no calibrated items of this grade are scored with {scoring}."
    summary = (
        calibrated[model_parameters(item.irt_model, item.points)]
        .rename(columns=PARAMETER_LABELS)
        .describe(percentiles=[0.1, 0.5, 0.9])
        .drop(index="count")
        .round(3)
    )
    return (
        f"Calibration summary: the parameters of the {len(calibrated)} calibrated items of this grade, from all its "
        f"assessments, that are scored with {scoring} (the new item is not among them):\n{summary.to_string()}"
    )


def format_parameters(parameters: dict) -> str:
    """Calibrated parameters from an item_parameters payload, such as "slope (a) = 0.927, location (b) = 0.111"."""
    names = model_parameters(parameters["irt_model"], parameters["scaled_points"])
    return ", ".join(f"{PARAMETER_LABELS[name]} = {parameters[name]:.3f}" for name in names)


def target_scale(bank: ItemBank, item: TimssItem) -> str:
    """The calibration whose scale the prediction targets, as the end of a sentence."""
    calibration = bank.calibrations.get((item.assessment, item.grade))
    if calibration is None:
        return "on the scale of the most recent calibration among the reference items"
    return f"that it would receive in the {calibration[0].lower()}{calibration[1:].rstrip('.')}"


def user_prompt(bank: ItemBank, item: TimssItem, references: list[dict], exclude_keys: Iterable[str] = ()) -> str:
    """User prompt: the calibration summary, the reference items with their parameters, and the new item."""
    blocks = [calibration_summary(bank, item, exclude_keys), "Reference items, most similar first:"]
    for reference in references:
        payload = reference["payload"]
        parameters = payload[ITEM_PARAMETERS_PAYLOAD]
        reference_item = TimssItem.model_validate(
            bank.items_by_key[reference["item_key"]] | {"scaled_points": parameters["scaled_points"]}
        )
        pct_correct = payload["pct_correct_intl_avg"]
        blocks.append(
            f"### Reference item {reference['rank']}: {payload['item_id']} ({payload['assessment']}, "
            f"grade {payload['grade']})\n"
            f"Cosine similarity with the new item: {reference['cosine_similarity']:.3f}\n"
            f"{describe_item(reference_item)}\n"
            f"Percent correct, international average: {'not reported' if pct_correct is None else pct_correct}\n"
            f"Calibration: {parameters['calibration']}\n"
            f"Calibrated parameters: {format_parameters(parameters)}"
        )
    names = predicted_parameters(item.irt_model, item.points)
    request = f"Predict the parameters of the new item {target_scale(bank, item)}. "
    request += f"Give its {', '.join(PARAMETER_LABELS[name] for name in names)}."
    if item.irt_model == "GPCM":
        request += f" Its {PARAMETER_LABELS[f'step{item.points}']} is minus the sum of the other steps."
    blocks.append(f"### New item ({item.assessment}, grade {item.grade})\n{describe_item(item)}\n\n{request}")
    return "\n\n".join(blocks)


@functools.cache
def prediction_schema(irt_model: str, points: int) -> type[BaseModel]:
    """Structured-output schema of a prediction: the comparison with the reference items, then each parameter the LLM
    predicts, with its reasoning first."""
    fields = {
        "reasoning": (
            str,
            Field(
                description="The reference items most comparable to the new item, and how the new item differs from them."
            ),
        )
    }
    for name in predicted_parameters(irt_model, points):
        fields[f"{name}_reasoning"] = (str, Field(description=f"Why the new item gets this {PARAMETER_LABELS[name]}."))
        if name == "slope":
            fields[name] = (float, Field(ge=0, description="Slope (a), positive."))
        elif name == "guessing":
            fields[name] = (float, Field(ge=0, le=1, description="Guessing (c), from 0 to 1."))
        else:
            fields[name] = (float, Field(description=f"{PARAMETER_LABELS[name].capitalize()}."))
    return create_model(f"{irt_model}Prediction{points}Points", **fields)


def reference_summary(entry: dict) -> dict:
    """A reference item as stored with a prediction: its rank, key, similarity, percent correct, and parameters."""
    payload = entry["payload"]
    parameters = payload[ITEM_PARAMETERS_PAYLOAD]
    return {
        "rank": entry["rank"],
        "found_by": entry["found_by"],
        "item_key": entry["item_key"],
        "item_id": payload["item_id"],
        "assessment": payload["assessment"],
        "cosine_similarity": round(entry["cosine_similarity"], 4),
        "pct_correct_intl_avg": payload["pct_correct_intl_avg"],
        "irt_model": parameters["irt_model"],
        "scaled_points": parameters["scaled_points"],
        **{name: parameters[name] for name in model_parameters(parameters["irt_model"], parameters["scaled_points"])},
    }


class ParameterPredictor:
    """Predicts the IRT parameters of TimssItems with an LLM, using similar calibrated items of an ItemIndex."""

    def __init__(
        self,
        index: ItemIndex,
        llm: LLMSettings = LLMSettings(),
        search_method: SearchMethod = "hybrid",
        n_similar: int = 10,
        n_same_model: int = 5,
    ):
        """``search_method`` finds the reference items; ``n_similar`` is the number of most similar items of any
        format, and ``n_same_model`` the number of most similar items scored with the new item's IRT model and points
        that are added if they are not among them."""
        self.index = index
        self.bank = index.bank
        self.llm = llm
        self.search_method = search_method
        self.n_similar = n_similar
        self.n_same_model = n_same_model
        self.client = LLMClient(llm)

    def describe(self) -> dict:
        """The settings of every step, for run reports."""
        return {
            "embedding": self.index.embedder.settings.describe(),
            "index": self.index.settings.describe(),
            "retrieval": {
                "search_method": self.search_method,
                "n_similar": self.n_similar,
                "n_same_model": self.n_same_model,
            },
            "llm": self.llm.describe(),
        }

    def prepare(self, item: TimssItem, exclude_keys: Iterable[str] = ()) -> dict:
        """Reference items and prompts for predicting ``item``, without calling the LLM.

        ``exclude_keys`` (such as the item and its other parts) are left out of the reference items and the
        calibration summary.
        """
        exclude_keys = list(exclude_keys)
        find = functools.partial(self.index.find_similar_items, item, exclude_keys=exclude_keys, method=self.search_method)
        nearest = find(self.n_similar)
        nearest_keys = {entry["item_key"] for entry in nearest}
        same_scoring = [
            entry
            for entry in find(self.n_same_model, irt_model=item.irt_model, scaled_points=item.points)
            if entry["item_key"] not in nearest_keys
        ]
        references = [
            entry | {"rank": rank, "found_by": found_by}
            for rank, (found_by, entry) in enumerate(
                [*(("most similar", entry) for entry in nearest), *(("same model", entry) for entry in same_scoring)], 1
            )
        ]
        if not references:
            raise ValueError(f"No calibrated grade {item.grade} items in '{self.index.settings.collection}'")
        return {
            "exclude_keys": sorted(exclude_keys),
            "references": references,
            "system_prompt": system_prompt(item),
            "user_prompt": user_prompt(self.bank, item, references, exclude_keys),
        }

    def record(self, item: TimssItem, prepared: dict) -> dict:
        """Everything about a prediction except the LLM's output: the item, its reference items, and the prompts."""
        return {
            **item.model_dump(
                include={
                    "item_id",
                    "assessment",
                    "grade",
                    "subject",
                    "content_domain",
                    "main_topic",
                    "cognitive_domain",
                    "item_type",
                    "max_points",
                }
            ),
            "scaled_points": item.points,
            "irt_model": item.irt_model,
            "target_calibration": self.bank.calibrations.get((item.assessment, item.grade)),
            "excluded_item_keys": prepared["exclude_keys"],
            "similar_items": [reference_summary(entry) for entry in prepared["references"]],
            "system_prompt": prepared["system_prompt"],
            "user_prompt": prepared["user_prompt"],
        }

    def request(self, item: TimssItem, prepared: dict) -> dict:
        """Send the prompts of ``prepare`` to the LLM. Returns its reasoning, parameters, and token usage.

        Raises as ``LLMClient.complete`` does.
        """
        completion = self.client.complete(
            prepared["system_prompt"], prepared["user_prompt"], prediction_schema(item.irt_model, item.points)
        )
        output = completion.parsed.model_dump()
        names = predicted_parameters(item.irt_model, item.points)
        parameters = {name: output[name] for name in names}
        if item.irt_model == "GPCM":
            parameters[f"step{item.points}"] = -sum(parameters[f"step{step}"] for step in range(1, item.points))
        return {
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "reasoning_effort": self.llm.reasoning_effort,
            "reasoning": output["reasoning"],
            **{f"{name}_reasoning": output[f"{name}_reasoning"] for name in names},
            **parameters,
            "llm_thinking": completion.thinking,
            "finish_reason": completion.finish_reason,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "reasoning_tokens": completion.reasoning_tokens,
            "cost_usd": completion.cost_usd,
            "predicted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def predict(self, item: TimssItem, exclude_keys: Iterable[str] = ()) -> dict:
        """Predict the IRT parameters of a TimssItem with the LLM, using similar calibrated items as context.

        Returns the prediction record: the item, its reference items, the prompts, and the LLM's reasoning and
        parameters.
        """
        prepared = self.prepare(item, exclude_keys)
        return self.record(item, prepared) | self.request(item, prepared)
