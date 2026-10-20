"""What every predictor of IRT parameters shares: the reference items, and what a leave-one-out run asks of it.

A predictor predicts the IRT parameters of a new item from the calibrated released items most similar to it. Two
predictors share this base and differ only in how they are asked and how they answer:

- ``LLMPredictor`` (``llm``) prompts an LLM with the reference items and the new item and gets a value per parameter.
- ``JevPredictor`` (``jev``) asks Jev a rubric question per parameter, whose criteria are the values of a scale, and
  gets a probability for each value.

Both find the reference items the same way (``Predictor.references``): the ``n_similar`` calibrated items of the
item's grade most similar to it (by ``search_method``), then the ``n_same_model`` most similar ones scored with its IRT
model and points that are not among them, so the predictor sees calibrated examples of every parameter it predicts.
The embedding model and the index (``ItemIndex``) are therefore chosen the same way for both.

``LeaveOneOut`` (``evaluation``) uses only the methods of ``Predictor``: ``prepare`` finds the reference items and
builds what is sent without any request, ``identity`` is what a saved prediction must match to be reused, ``record``
and ``request_async`` make the saved record, and ``parameters`` and ``parameter_details`` read the predicted values
back from it.
"""

from __future__ import annotations

import abc
import functools
from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, ClassVar

from ..config import data_dir as resolve_data_dir
from ..config import slug
from ..items import (  # noqa: F401 (plural, here before it was shared)
    ItemBank,
    TimssItem,
    model_parameters,
    plural,
)
from ..running import UnusableOutputError, run_sync
from .index import ITEM_PARAMETERS_PAYLOAD, ItemIndex, SearchMethod

PREDICTIONS_DIR = Path("cache") / "parameter_predictions" / "timss_math"
"""Where predictions are saved in the data directory, one subdirectory per provider and model."""


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


class Predictor(abc.ABC):
    """Predicts the IRT parameters of TimssItems, using similar calibrated items of an ItemIndex."""

    kind: ClassVar[str]
    """The kind of predictor, "llm" or "jev": the ``predictor`` column of the results."""
    UNUSABLE_ERRORS: ClassVar[tuple[type[Exception], ...]] = (UnusableOutputError,)
    """Errors of output that a rerun may get right: a leave-one-out run reports the item and requests it again on the
    next run. Any other error, such as an API error from exhausted credit, stops the run."""

    def __init__(
        self, index: ItemIndex, search_method: SearchMethod = "hybrid", n_similar: int = 10, n_same_model: int = 5
    ):
        """``search_method`` finds the reference items; ``n_similar`` is the number of most similar items of any
        format, and ``n_same_model`` the number of most similar items scored with the new item's IRT model and points
        that are added if they are not among them."""
        self.index = index
        self.bank: ItemBank = index.bank
        self.search_method = search_method
        self.n_similar = n_similar
        self.n_same_model = n_same_model

    def references(self, item: TimssItem, exclude_keys: Iterable[str] = ()) -> list[dict]:
        """The reference items of ``item``, most similar first, each with its ``rank`` and how it was ``found_by``.

        ``exclude_keys`` (such as the item and its other parts) are left out.
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
        return references

    def describe_retrieval(self) -> dict:
        """The settings of the steps before the prediction: embedding, index, and retrieval."""
        return {
            "embedding": self.index.embedder.settings.describe(),
            "index": self.index.settings.describe(),
            "retrieval": {
                "search_method": self.search_method,
                "n_similar": self.n_similar,
                "n_same_model": self.n_same_model,
            },
        }

    def retrieval_name(self) -> str:
        """The embedding and retrieval settings, for the name of a run's output directory."""
        embedding = self.index.embedder.settings
        return "__".join(
            [
                f"{embedding.provider}-{slug(embedding.model)}-{embedding.dimension or 'full'}",
                f"{self.search_method}-{self.n_similar}+{self.n_same_model}",
            ]
        )

    def item_record(self, item: TimssItem, prepared: dict) -> dict:
        """The part of a saved prediction that every predictor shares: the item, the calibration whose scale it is
        predicted on, the items left out of its context, and its reference items."""
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
        }

    def predict(self, item: TimssItem, exclude_keys: Iterable[str] = ()) -> dict:
        """Predict the IRT parameters of a TimssItem, using similar calibrated items as context.

        Returns the prediction record: the item, its reference items, what was sent, and the answer, from which
        ``parameters`` reads the predicted values.
        """
        prepared = self.prepare(item, exclude_keys)
        return self.record(item, prepared) | self.request(item, prepared)

    def request(self, item: TimssItem, prepared: dict) -> dict:
        """Send what ``prepare`` built and return the answer (``request_async`` in a session of its own)."""

        async def once() -> dict:
            async with self.session() as session:
                return await self.request_async(item, prepared, session)

        return run_sync(once())

    def cache_dir(self, data_dir: str | Path | None = None) -> Path:
        """``cache/parameter_predictions/timss_math/<provider>/<model>/`` in the data directory."""
        provider, model = self.cache_name()
        return resolve_data_dir(data_dir) / PREDICTIONS_DIR / slug(provider) / slug(model)

    def notebook_cache_dir(self, data_dir: str | Path | None = None) -> Path | None:
        """Where a notebook saved predictions that a leave-one-out run may reuse, if anywhere."""
        return None

    def parameter_details(self, record: dict, name: str) -> dict:
        """More about the prediction of the parameter ``name`` of a saved record, one column each in the errors."""
        return {}

    @abc.abstractmethod
    def describe(self) -> dict:
        """The settings of every step, for run reports."""

    @abc.abstractmethod
    def prepare(self, item: TimssItem, exclude_keys: Iterable[str] = ()) -> dict:
        """The reference items and what is sent to predict ``item``, without any request: a dict with at least
        ``exclude_keys`` (sorted) and ``references``. ``exclude_keys`` (such as the item and its other parts) are left
        out of the reference items and of anything else the item is compared with."""

    @abc.abstractmethod
    def identity(self, prepared: dict) -> dict:
        """What a saved prediction must match to be reused: who answers, and everything sent."""

    @abc.abstractmethod
    def record(self, item: TimssItem, prepared: dict) -> dict:
        """Everything about a prediction except the answer: the item, its reference items, and what is sent."""

    @abc.abstractmethod
    def session(self) -> AbstractAsyncContextManager[Any]:
        """A session for the requests of one event loop, closed at the end."""

    @abc.abstractmethod
    async def request_async(self, item: TimssItem, prepared: dict, session: Any) -> dict:
        """``request`` as a coroutine, through a session of ``session()``. Raises one of ``UNUSABLE_ERRORS`` for
        output that cannot be used, and API errors as they come."""

    @abc.abstractmethod
    def cache_name(self) -> tuple[str, str]:
        """The provider and model that name the directory of the saved predictions."""

    @abc.abstractmethod
    def parameters(self, record: dict) -> dict[str, float]:
        """The predicted value of every parameter of the item's IRT model, the last GPCM step included, from a saved
        record."""

    @abc.abstractmethod
    def row_details(self, record: dict) -> dict:
        """The predictor's own columns of a saved record in the predictions table, such as the LLM's reasoning."""

    @abc.abstractmethod
    def run_name(self) -> str:
        """A name for the run's output directory, from its main settings."""
