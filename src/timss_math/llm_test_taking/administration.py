"""An administration of the released items as a test: conditions answer every unit, an LLM scorer scores the
constructed responses, and every condition is scored with the IRT models and the calibrated parameters.

``Administration(conditions, scorer)`` holds the units (``ItemUnits``), the conditions by label, the scorer, the item
models (``irt``), and the calibrations with their TIMSS scale constants (``scales``):

1. ``request_answers`` asks every condition to answer every unit that has no saved answer, all conditions at once,
   with the prompts of ``answering``, in batches or directly (``requesting.send``). The units are requested in a fixed
   random order, so a capped run samples every assessment and grade.
2. ``answers`` reads the saved answers, one row per condition and item of an answered unit.
3. ``request_verdicts`` asks the scorer for a verdict on every constructed response to a calibrated item that has
   none, with the prompts of ``scoring``; identical responses to an item share one verdict.
4. ``scores`` scores every answer: a multiple-choice answer against the key, a constructed response by its verdict.
5. ``results`` estimates each condition's θ on each calibration's scale and puts it on the TIMSS reporting scale, and
   puts each item score on the IRT scale (``ScoringResults``).

Answers are saved in ``<answers_dir>/<provider>/<model>/<reasoning effort>/<role>/``, one JSON file per unit, and
verdicts in ``<verdicts_dir>/<provider>/<model>/<reasoning effort>/``, one per item and response, each named by the
unit or item and a hash of the model, reasoning effort, both prompts, and the output schema, so that a changed prompt
asks again and a rerun requests only what is missing. By default both are in the data directory
(``config.data_dir``): ``cache/llm_answers/`` and ``cache/llm_answer_scores/`` of ``notebooks/data`` in a clone, which
hold the answers and verdicts of the run of 2026-10-19, so that ``Administration()`` reproduces it without any
request.

Requests cost money, through OpenRouter or the provider of a model asked directly, such as the Claude API, so
``max_requests`` must be given: 0 only collects the batches that an earlier run submitted (and sends nothing at all
when none is open), 1 is a test request per condition, and None requests everything missing.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..batch import BatchManifests, OpenRouterBatches
from ..config import data_dir as resolve_data_dir
from ..llm import LLMSettings
from ..running import run_sync
from .answering import (
    answer_fields,
    answer_schema,
    answer_system_prompt,
    answer_user_prompt,
)
from .conditions import (
    DEFAULT_MODELS,
    DEFAULT_SCORER,
    TOKEN_PRICES,
    Condition,
    batch_cost,
    by_label,
    conditions_of,
    direct_cost,
    settings_folder,
)
from .irt import (
    ItemModel,
    check_item_models,
    implied_percent_correct,
    item_models,
    recover_theta,
)
from .requesting import Request, RunReport, make_request, send
from .results import (
    ScoringResults,
    item_scores_on_scale,
    item_scores_wide,
    overall_scores,
)
from .scales import Calibrations
from .scoring import (
    check_keys_and_guides,
    score_answers,
    scorer_schema,
    scorer_system_prompt,
    scorer_user_prompt,
)
from .units import ItemUnits, Unit

ANSWERS_DIR = Path("cache") / "llm_answers"
"""Where answers are saved in the data directory."""
VERDICTS_DIR = Path("cache") / "llm_answer_scores"
"""Where the scorer's verdicts are saved in the data directory."""


class Administration:
    """Conditions take the released items as a test, and their answers are scored on the IRT and TIMSS scales."""

    def __init__(
        self,
        conditions: Iterable[Condition] | Mapping[str, Condition] | None = None,
        scorer: LLMSettings = DEFAULT_SCORER,
        units: ItemUnits | None = None,
        data_dir: str | Path | None = None,
        answers_dir: str | Path | None = None,
        verdicts_dir: str | Path | None = None,
        constants_path: str | Path | None = None,
        token_prices: dict[tuple[str, str], tuple[float, float]] = TOKEN_PRICES,
    ):
        """``conditions`` are the LLMs and roles that take the test, as a list or by label (by default the ten of the
        run of 2026-10-19: ``DEFAULT_MODELS`` in both roles), and ``scorer`` the LLM that scores the constructed
        responses of every condition, so that all are scored alike. ``answers_dir`` and ``verdicts_dir`` replace the
        caches in ``data_dir`` (see ``config.data_dir``), ``constants_path`` the package's TIMSS scale constants, and
        ``token_prices`` the prices of the models whose provider reports no cost."""
        if conditions is None:
            conditions = conditions_of(DEFAULT_MODELS)
        elif isinstance(conditions, Mapping):
            conditions = conditions.values()
        self.conditions: dict[str, Condition] = by_label(conditions)
        if not self.conditions:
            raise ValueError("No conditions take the test")
        self.scorer = scorer
        self.units = units or ItemUnits.load()
        data = resolve_data_dir(data_dir)
        self.answers_dir = Path(answers_dir) if answers_dir else data / ANSWERS_DIR
        self.verdicts_dir = Path(verdicts_dir) if verdicts_dir else data / VERDICTS_DIR
        self.calibrations = Calibrations(self.units.parameters, constants_path)
        self.token_prices = token_prices
        self._scorer_schemas: dict[str, tuple[type[BaseModel], dict]] = {}

    @functools.cached_property
    def models(self) -> dict[str, ItemModel]:
        """The IRT model of every calibrated item, checked against the definitions (``irt.check_item_models``)."""
        models = item_models(self.units.parameters)
        check_item_models(models)
        return models

    @functools.cached_property
    def request_order(self) -> list[Unit]:
        """The units in a fixed random order, so that a capped run samples every assessment and grade."""
        units = self.units.units
        return [units[position] for position in np.random.default_rng(0).permutation(len(units))]

    @functools.cached_property
    def _answer_schemas(self) -> dict[Unit, tuple[type[BaseModel], dict]]:
        """The output schema of every unit, and its JSON schema, made once for all conditions."""
        schemas = {}
        for unit in self.units.units:
            schema = answer_schema(unit)
            schemas[unit] = (schema, schema.model_json_schema())
        return schemas

    def answer_request(self, condition: Condition, unit: Unit) -> Request:
        """The request of a condition's answer to a unit."""
        schema, schema_json = self._answer_schemas[unit]
        return make_request(
            condition.cache_dir(self.answers_dir),
            condition.llm,
            unit.key,
            answer_system_prompt(unit.grade, condition.role),
            answer_user_prompt(unit),
            schema,
            schema_json,
            unit_key=unit.key,
            item_keys=list(unit.item_keys),
            role=condition.role,
        )

    @functools.cached_property
    def answer_requests(self) -> dict[str, dict[Unit, Request]]:
        """The request of every condition's answer to every unit, by condition label, the units in ``request_order``."""
        return {
            label: {unit: self.answer_request(condition, unit) for unit in self.request_order}
            for label, condition in self.conditions.items()
        }

    def usage(self) -> pd.DataFrame:
        """Per condition: its settings, the units answered and in open batches, and the tokens, time, and cost of the
        answers. OpenRouter reports the cost of each direct request, and of each batch as a whole; the cost of the
        other providers' requests comes from their tokens at ``token_prices``, or is NaN for a model without a price."""
        rows = []
        for label, by_unit in self.answer_requests.items():
            condition = self.conditions[label]
            folder = condition.cache_dir(self.answers_dir)
            records = [record for request in by_unit.values() if (record := request.saved()) is not None]
            in_batches = BatchManifests(folder / "batches").open_custom_ids()
            frame = pd.DataFrame(
                records,
                columns=["prompt_tokens", "completion_tokens", "reasoning_tokens", "seconds", "cost_usd", "batch_id"],
            )
            usage = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "seconds")
            rows.append(
                {
                    "condition": label,
                    "provider": condition.llm.provider,
                    "model": condition.llm.model,
                    "reasoning_effort": condition.llm.reasoning_effort,
                    "role": condition.role,
                    "max_tokens": condition.llm.max_tokens,
                    "units": len(by_unit),
                    "answered_units": len(records),
                    "answered_in_batches": int(frame["batch_id"].notna().sum()),
                    "in_open_batches": sum(request.custom_id in in_batches for request in by_unit.values()),
                    "direct_cost_usd": direct_cost(condition.llm, frame[frame["batch_id"].isna()], self.token_prices),
                    "batch_cost_usd": batch_cost(folder),
                    **{f"mean_{column}": frame[column].mean() for column in usage},
                }
            )
        return pd.DataFrame(rows).set_index("condition")

    async def request_answers_async(
        self,
        max_requests: int | None,
        mode: str = "batch",
        wait: bool = False,
        workers: int = 8,
        batch_size: int = 200,
        poll_seconds: float = 60,
        session=None,
        batch_client: OpenRouterBatches | None = None,
    ) -> dict[str, RunReport]:
        """``request_answers`` as a coroutine, for a notebook: ``await administration.request_answers_async(...)``."""
        options = {
            "wait": wait,
            "workers": workers,
            "batch_size": batch_size,
            "poll_seconds": poll_seconds,
            "session": session,
            "batch_client": batch_client,
        }
        async with asyncio.TaskGroup() as group:
            tasks = {
                label: group.create_task(
                    send(
                        self.conditions[label].llm,
                        self.conditions[label].cache_dir(self.answers_dir),
                        list(by_unit.values()),
                        max_requests,
                        mode,
                        **options,
                    )
                )
                for label, by_unit in self.answer_requests.items()
            }
        return {label: task.result() for label, task in tasks.items()}

    def request_answers(
        self,
        max_requests: int | None,
        mode: str = "batch",
        wait: bool = False,
        workers: int = 8,
        batch_size: int = 200,
        poll_seconds: float = 60,
    ) -> dict[str, RunReport]:
        """Request the answers that are not saved yet, at most ``max_requests`` units per condition (None for all, 0
        only collects open batches), all conditions at once. ``mode`` is "batch" (OpenRouter batches for the models
        that have them, at about half the price, direct requests for the others) or "direct" (every request now,
        ``workers`` at a time per condition); with ``wait``, waits until every batch is collected, which can take up
        to 24 hours, polling every ``poll_seconds``; a batch holds at most ``batch_size`` requests. Returns a report
        per condition label. Runs ``request_answers_async`` (see ``run_sync``)."""
        return run_sync(self.request_answers_async(max_requests, mode, wait, workers, batch_size, poll_seconds))

    def answers(self) -> pd.DataFrame:
        """One row per condition and item of a unit with a saved answer: ``condition``, ``item_key``, ``unit_key``, and
        ``answer``, an option letter or a written answer."""
        rows = []
        for label, by_unit in self.answer_requests.items():
            for unit, request in by_unit.items():
                if (record := request.saved()) is None:
                    continue
                for key, field in answer_fields(unit).items():
                    answer = record["output"][field]
                    rows.append({"condition": label, "item_key": key, "unit_key": unit.key, "answer": answer})
        return pd.DataFrame(rows, columns=["condition", "item_key", "unit_key", "answer"])

    @property
    def scorer_folder(self) -> Path:
        """Where the scorer's verdicts are saved: ``<verdicts_dir>/<provider>/<model>/<reasoning effort>``."""
        return settings_folder(self.verdicts_dir, self.scorer)

    def scorer_request(self, item_key: str, response: str) -> Request:
        """The request of the scorer's verdict on a constructed response to an item."""
        if item_key not in self._scorer_schemas:
            schema = scorer_schema(self.units.items_by_key[item_key])
            self._scorer_schemas[item_key] = (schema, schema.model_json_schema())
        schema, schema_json = self._scorer_schemas[item_key]
        return make_request(
            self.scorer_folder,
            self.scorer,
            item_key,
            scorer_system_prompt(self.units.items_by_key[item_key]["grade"]),
            scorer_user_prompt(self.units.unit_of_item[item_key], item_key, response),
            schema,
            schema_json,
            item_key=item_key,
            response=response,
        )

    def scorer_requests(self, answers: pd.DataFrame | None = None) -> list[Request]:
        """The requests of a verdict on every non-empty constructed response to a calibrated item in ``answers`` (by
        default every saved answer), one per item and response, so that identical responses share one verdict."""
        answers = self.answers() if answers is None else answers
        items_by_key, calibrated = self.units.items_by_key, self.units.parameters.index
        constructed = answers["item_key"].map(lambda key: items_by_key[key]["item_type"] == "constructed_response")
        to_score = answers[answers["item_key"].isin(calibrated) & constructed]
        requests: dict[Path, Request] = {}
        for row in to_score[to_score["answer"].str.strip().astype(bool)].itertuples():
            request = self.scorer_request(row.item_key, row.answer)
            requests[request.path] = request
        return list(requests.values())

    async def request_verdicts_async(
        self,
        max_requests: int | None,
        mode: str = "direct",
        wait: bool = False,
        workers: int = 8,
        batch_size: int = 200,
        poll_seconds: float = 60,
        answers: pd.DataFrame | None = None,
        session=None,
        batch_client: OpenRouterBatches | None = None,
    ) -> RunReport:
        """``request_verdicts`` as a coroutine, for a notebook: ``await administration.request_verdicts_async(...)``."""
        return await send(
            self.scorer,
            self.scorer_folder,
            self.scorer_requests(answers),
            max_requests,
            mode,
            wait=wait,
            workers=workers,
            batch_size=batch_size,
            poll_seconds=poll_seconds,
            session=session,
            batch_client=batch_client,
        )

    def request_verdicts(
        self,
        max_requests: int | None,
        mode: str = "direct",
        wait: bool = False,
        workers: int = 8,
        batch_size: int = 200,
        poll_seconds: float = 60,
    ) -> RunReport:
        """Request the scorer's verdicts that are not saved yet on the constructed responses of the saved answers, at
        most ``max_requests`` (None for all, 0 only collects open batches), in ``mode`` "direct" (the default) or
        "batch", as ``request_answers`` does. Answers still in open batches are scored once they are collected, on a
        later run. Runs ``request_verdicts_async`` (see ``run_sync``)."""
        return run_sync(self.request_verdicts_async(max_requests, mode, wait, workers, batch_size, poll_seconds))

    def verdict(self, item_key: str, response: str) -> dict | None:
        """The saved verdict on a constructed response to an item, or None."""
        return self.scorer_request(item_key, response).saved()

    def scores(self, answers: pd.DataFrame | None = None) -> pd.DataFrame:
        """One row per condition and calibrated item of ``answers`` (by default every saved answer): the answer, the
        key or the scorer's category and reasoning, the points by the scoring guide, and the score in the scaling
        (``score``), which differs from the points only for K14 and L18 of TIMSS Advanced 1995. A constructed response
        without a verdict yet has no score."""
        check_keys_and_guides(self.units)
        return score_answers(self.answers() if answers is None else answers, self.units, self.verdict)

    def check_irt(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Check the models and estimators against the data, and raise if a check fails: the percent correct that the
        parameters imply against the international percent correct of the items (``irt.implied_percent_correct``),
        and θ recovered from simulated response patterns (``irt.recover_theta``). Returns both tables."""
        percent_correct = {key: self.units.items_by_key[key]["pct_correct_intl_avg"] for key in self.models}
        items = self.calibrations.items_of()
        return implied_percent_correct(self.models, percent_correct, items), recover_theta(self.models, items)

    def results(self, answers: pd.DataFrame | None = None) -> ScoringResults:
        """The scores of every condition on the IRT and TIMSS scales, from ``answers`` (by default every saved answer)
        and the saved verdicts."""
        answers = self.answers() if answers is None else answers
        scores = self.scores(answers)
        parameters = self.units.parameters
        overall = overall_scores(scores, self.conditions, self.calibrations, self.models)
        on_scale = item_scores_on_scale(scores, overall, parameters, self.calibrations, self.models)
        return ScoringResults(
            conditions=self.usage(),
            overall=overall,
            item_scores=on_scale,
            item_scores_wide=item_scores_wide(on_scale, self.conditions, parameters, self.calibrations),
            calibrations=self.calibrations.table(parameters),
        )
