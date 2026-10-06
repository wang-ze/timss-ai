"""Leave-one-out evaluation: predict every calibrated released item as if it were new, and measure the errors.

``LeaveOneOut(predictor)`` predicts each calibrated item of the collection (722) with ``ParameterPredictor`` and
compares the prediction with the item's calibrated parameters, as ``predict_item_parameters.ipynb`` does:

- The item and its other parts (``ItemBank.parts``: the items whose IDs differ only in a final part letter, and the
  other items of its task) are left out of its context, both as reference items and from the calibration summary.
- The item's own percent correct is not in its prompt, and each item is predicted with the IRT model and points of
  its calibration, on the scale of its own calibration.
- Each item reuses its stored vector, so the run makes no embedding requests.

Each prediction is saved as it arrives, one JSON file per item and prompt, named by the item and a hash of the LLM
provider, model, reasoning effort, and both prompts. A rerun therefore requests only predictions that are missing, and
runs that differ in any setting (embedding model, search method, LLM, ...) never overwrite each other's predictions.
For the notebook's OpenRouter model, the predictions the notebook saved are reused when their prompts match.

``request`` sends one request per item, ``workers`` at a time, as coroutines in one event loop (``request_async`` in a
notebook, where ``await loo.request_async(...)`` runs in the notebook's own loop). On an API error it sends no further
requests but saves the ones already running, which are paid for; an interrupt cancels them all at once.
``request_batch`` sends the same requests as OpenRouter batches instead (see ``batch``), at about half the price, and
saves the predictions under the same names once the batches complete, which can take up to 24 hours. Submitted batches
are kept in manifests next to the predictions, so a run that stops collects them on the next run instead of submitting
them again.

``PredictionResults`` holds one row per item (``predictions``) and one row per item and predicted parameter
(``errors``, with two baselines that need no LLM), summarizes the errors by grade or assessment, and exports them to
Excel. Run ``uv run timss-predict-loo --help`` for the command line.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import re
import sys
import threading
import time
from collections.abc import Coroutine, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from pydantic import ValidationError

from .batch import TERMINAL_STATUSES, BatchAPIError, BatchManifests, OpenRouterBatches
from .config import data_dir as resolve_data_dir
from .config import slug, write_json
from .embeddings import DEFAULT_MODELS as DEFAULT_EMBEDDING_MODELS
from .embeddings import EMBEDDING_PROVIDERS, EmbeddingSettings
from .index import SEARCH_METHODS, IndexSettings, ItemIndex
from .items import PARAMETER_NAMES, ItemBank, predicted_parameters
from .predictor import BASE_URLS as LLM_PROVIDERS
from .predictor import (
    AsyncLLMSession,
    LLMSettings,
    NoPredictionError,
    ParameterPredictor,
    plural,
)

log = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("cache") / "parameter_predictions" / "timss_math"
RUNS_DIR = "parameter_prediction_runs"
# Output that the LLM may get right on a rerun: the item is reported and left for the next run. Any other error, such
# as an API error from exhausted credit, stops the run.
UNUSABLE_OUTPUT_ERRORS = (
    ValidationError,
    openai.LengthFinishReasonError,
    openai.ContentFilterFinishReasonError,
    NoPredictionError,
)
EXCEL_CELL_LIMIT = 32_767
SAVED_FILE = re.compile(r"(.+)_[0-9a-f]{16}\.json")  # <item file stem>_<identity hash>.json


def prediction_identity(llm: LLMSettings, prepared: dict) -> dict:
    """What a saved prediction must match to be reused: the LLM, its reasoning effort, and both prompts."""
    return {
        "llm_provider": llm.provider,
        "llm_model": llm.model,
        "reasoning_effort": llm.reasoning_effort,
        "system_prompt": prepared["system_prompt"],
        "user_prompt": prepared["user_prompt"],
    }


def item_file_stem(key: str) -> str:
    """File name of an item key, as the notebook writes it: TIMSS_2011_grade_4_Mathematics_M031346A."""
    return re.sub(r"[^A-Za-z0-9]+", "_", key)


class PredictionCache:
    """Saved predictions: ``<directory>/<item>_<hash of prediction_identity>.json``.

    ``notebook_directory`` holds the notebook's predictions, one ``<item>.json`` per item, which are reused when they
    match the identity.
    """

    def __init__(self, directory: Path, notebook_directory: Path | None = None):
        self.directory = Path(directory)
        self.notebook_directory = notebook_directory

    def stem(self, key: str, identity: dict) -> str:
        """``<item>_<hash of the identity>``, the name of the prediction file and the custom_id of its batch request."""
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        return f"{item_file_stem(key)}_{digest}"

    def path(self, key: str, identity: dict) -> Path:
        return self.directory / f"{self.stem(key, identity)}.json"

    def find(self, key: str, identity: dict) -> Path | None:
        """The file of the saved prediction of this item and identity, if any."""
        path = self.path(key, identity)
        if path.exists():
            return path
        if self.notebook_directory is not None:
            path = self.notebook_directory / f"{item_file_stem(key)}.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                # The notebook predicts through OpenRouter only, so its records name no provider.
                if record.get("llm_provider", "openrouter") == identity["llm_provider"] and all(
                    record[name] == identity[name]
                    for name in ("llm_model", "reasoning_effort", "system_prompt", "user_prompt")
                ):
                    return path
        return None

    def predicted_item_stems(self) -> set[str]:
        """The item file stems (``item_file_stem``) with a saved prediction for any prompts and LLM settings."""
        stems = set()
        if self.directory.exists():
            stems |= {match[1] for path in self.directory.iterdir() if (match := SAVED_FILE.fullmatch(path.name))}
        if self.notebook_directory is not None and self.notebook_directory.exists():
            stems |= {path.stem for path in self.notebook_directory.glob("*.json")}
        return stems

    def save(self, key: str, identity: dict, record: dict) -> Path:
        return self.save_stem(self.stem(key, identity), record)

    def save_stem(self, stem: str, record: dict) -> Path:
        path = self.directory / f"{stem}.json"
        write_json(path, record)
        return path


def default_cache_dir(llm: LLMSettings, data_dir: str | Path | None = None) -> Path:
    """``cache/parameter_predictions/timss_math/<provider>/<model>/`` in the data directory."""
    return resolve_data_dir(data_dir) / PREDICTIONS_DIR / slug(llm.provider) / slug(llm.model)


def notebook_cache_dir(llm: LLMSettings, data_dir: str | Path | None = None) -> Path | None:
    """Where the notebook saves the predictions of an OpenRouter model; None for other providers."""
    if llm.provider != "openrouter":
        return None
    return resolve_data_dir(data_dir) / PREDICTIONS_DIR / llm.model.replace("/", "_")


@dataclass
class RequestReport:
    requested: int
    saved: int
    """Predictions saved, including those of batches submitted by an earlier run."""
    unusable: dict[str, str]
    """Item key -> error, for output that could not be used; a rerun requests these items again."""
    cost_usd: float
    seconds: float
    failed: dict[str, str] = field(default_factory=dict)
    """Item key -> error, for batch requests that got no response; a rerun requests these items again."""
    in_flight: int = 0
    """Items in batches that have not completed yet; a rerun collects them."""


class LeaveOneOut:
    """Predicts calibrated released items as if they were new, with their parts left out of their context."""

    def __init__(
        self,
        predictor: ParameterPredictor,
        keys: Iterable[str] | None = None,
        cache_dir: str | Path | None = None,
        data_dir: str | Path | None = None,
        reuse_notebook_predictions: bool = True,
    ):
        """``keys`` limits the run to some calibrated items, such as
        ``bank.calibrated.query("grade == 4").index``; by default every calibrated item is predicted."""
        self.predictor = predictor
        self.bank: ItemBank = predictor.bank
        calibrated = self.bank.calibrated
        keys = list(calibrated.index if keys is None else keys)
        unknown = sorted(set(keys) - set(calibrated.index))
        if unknown:
            raise ValueError(f"These items are not calibrated items of the collection: {unknown}")
        self.items = {key: self.bank.item(key) for key in keys}
        for key, item in self.items.items():
            if item.irt_model != calibrated.at[key, "irt_model"]:
                raise ValueError(f"{key} would be predicted with {item.irt_model} but is calibrated with another model")
        self.cache = PredictionCache(
            Path(cache_dir) if cache_dir else default_cache_dir(predictor.llm, data_dir),
            notebook_cache_dir(predictor.llm, data_dir) if reuse_notebook_predictions else None,
        )
        self.batches = BatchManifests(self.cache.directory / "batches")

    @functools.cached_property
    def prepared(self) -> dict[str, dict]:
        """The reference items and prompts of every item, found without API calls."""
        started = time.monotonic()
        prepared = {
            key: self.predictor.prepare(item, exclude_keys=self.bank.parts[key]) for key, item in self.items.items()
        }
        log.info("Prepared %d items in %.0f s", len(prepared), time.monotonic() - started)
        return prepared

    def identity(self, key: str) -> dict:
        return prediction_identity(self.predictor.llm, self.prepared[key])

    def saved(self) -> dict[str, Path]:
        """The file of the saved prediction of each item that has one for the current prompts and LLM settings."""
        paths = {key: self.cache.find(key, self.identity(key)) for key in self.items}
        return {key: path for key, path in paths.items() if path is not None}

    def status(self) -> pd.Series:
        """Per item: "saved", "in a submitted batch", "saved for other prompts or LLM settings", or
        "not predicted yet"."""
        saved, predicted, in_batches = self.saved(), self.cache.predicted_item_stems(), self.batches.open_custom_ids()
        return pd.Series(
            {
                key: "saved"
                if key in saved
                else "in a submitted batch"
                if self.cache.stem(key, self.identity(key)) in in_batches
                else "saved for other prompts or LLM settings"
                if item_file_stem(key) in predicted
                else "not predicted yet"
                for key in self.items
            },
            name="status",
        )

    def pending(self, max_requests: int | None) -> list[str]:
        """The items with no saved prediction for the current prompts and LLM settings that are not in a submitted
        batch either, at most ``max_requests`` of them (None for all)."""
        in_batches = self.batches.open_custom_ids()
        pending = [
            key
            for key in self.items
            if self.cache.find(key, self.identity(key)) is None
            and self.cache.stem(key, self.identity(key)) not in in_batches
        ]
        return pending if max_requests is None else pending[:max_requests]

    def request(self, max_requests: int | None, workers: int = 8) -> RequestReport:
        """Request the predictions that are not saved yet, at most ``max_requests`` of them (None for all),
        ``workers`` at a time, saving each as it arrives.

        LLM requests cost money (about $9 for all 722 items with the default LLM), so ``max_requests`` must be given.
        Items in a submitted batch are left to the batch. Runs ``request_async`` (see ``run_sync``).
        """
        return run_sync(self.request_async(max_requests, workers))

    async def request_async(
        self, max_requests: int | None, workers: int = 8, session: AsyncLLMSession | None = None
    ) -> RequestReport:
        """``request`` as a coroutine, for a notebook: ``await loo.request_async(...)``.

        On the first API error (such as exhausted credit) no further request starts; the requests already running
        are saved, as they are paid for, and then the error is raised. Cancelling the coroutine, as an interrupt
        does, cancels every request at once. ``session`` replaces the LLM session, such as a fake one in tests.
        """
        if session is None:
            async with self.predictor.client.session() as opened:
                return await self.request_async(max_requests, workers, opened)
        pending = self.pending(max_requests)
        llm = self.predictor.llm
        log.info("Requesting %d predictions from %s (%s), %d at a time", len(pending), llm.model, llm.provider, workers)
        report = RequestReport(0, 0, {}, 0.0, 0.0)
        started = time.monotonic()
        slots = asyncio.Semaphore(max(1, workers))
        stopped = asyncio.Event()  # set by the first API error

        async def predict(key: str) -> tuple[str, dict | Exception | None]:
            """The item's output, the exception it raised, or None if it was not requested; never raises, so that
            one failed request does not cancel the others."""
            async with slots:
                if stopped.is_set():
                    return key, None
                try:
                    return key, await self.predictor.request_async(self.items[key], self.prepared[key], session)
                except Exception as error:  # noqa: BLE001 - returned, and raised after the running requests are saved
                    if not isinstance(error, UNUSABLE_OUTPUT_ERRORS):
                        stopped.set()
                    return key, error

        first_error = None
        tasks = [asyncio.create_task(predict(key)) for key in pending]
        try:
            for done, finished in enumerate(asyncio.as_completed(tasks), 1):
                key, output = await finished
                if output is None:
                    continue
                report.requested += 1
                if isinstance(output, UNUSABLE_OUTPUT_ERRORS):
                    report.unusable[key] = f"{type(output).__name__}: {output}"
                    log.warning("%s: unusable output (%s), left for a rerun", key, type(output).__name__)
                elif isinstance(output, Exception):
                    first_error = first_error or output
                    log.error("%s: %s: %s; no further requests are sent", key, type(output).__name__, output)
                else:
                    record = self.predictor.record(self.items[key], self.prepared[key]) | output
                    self.cache.save(key, self.identity(key), record)
                    report.saved += 1
                    report.cost_usd += output["cost_usd"] or 0.0
                if done % 50 == 0 or done == len(pending):
                    log.info("%d/%d in %.0f s, $%.2f", done, len(pending), time.monotonic() - started, report.cost_usd)
        finally:
            # On a cancellation (an interrupt) or an error here, stop every request still running or waiting.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        report.seconds = time.monotonic() - started
        if first_error is not None:
            raise first_error
        return report

    def request_batch(
        self,
        max_requests: int | None,
        wait: bool = True,
        poll_seconds: float = 60,
        client: OpenRouterBatches | None = None,
    ) -> RequestReport:
        """Request the predictions that are not saved yet as OpenRouter batches, at about half the price of
        ``request``, and save them once the batches complete.

        First collects the batches that earlier runs submitted, then submits at most ``max_requests`` items (None for
        all; 0 only collects) as one batch per prediction schema, as Google models require. With ``wait``, polls
        every ``poll_seconds`` until every batch is collected; otherwise returns at once, and a later call collects.
        The items of a batch that fails or expires get no prediction and are requested again by a later call.
        Submitting and collecting hold the lock of the manifests (``BatchManifests.locked``), so another process that
        shares them, such as a notebook and a ``timss-predict-loo --batch`` run, waits instead of submitting the same
        items again.
        """
        started = time.monotonic()
        client = client or OpenRouterBatches(self.predictor.llm)
        report = RequestReport(0, 0, {}, 0.0, 0.0)
        _ = self.prepared  # the prompts, prepared before taking the lock, which other processes wait for
        with self.batches.locked():  # another process may share the manifests (see batch)
            self.batches.adopt_submitting(client)
            self._collect(client, report)
            self._submit(client, self.pending(max_requests), report)
        while wait and self.batches.open():
            time.sleep(poll_seconds)
            with self.batches.locked():
                self._collect(client, report)
        report.in_flight = len(self.batches.open_custom_ids())
        report.seconds = time.monotonic() - started
        return report

    def _submit(self, client: OpenRouterBatches, pending: list[str], report: RequestReport) -> None:
        """Submit ``pending`` as one batch per prediction schema, each with its manifest."""
        groups: dict[tuple[str, int], list[str]] = {}
        for key in pending:
            item = self.items[key]
            groups.setdefault((item.irt_model, item.points), []).append(key)
        for (irt_model, points), keys in groups.items():
            entries = {
                self.cache.stem(key, self.identity(key)): {
                    "item_key": key,
                    "record": self.predictor.record(self.items[key], self.prepared[key]),
                }
                for key in keys
            }
            path = self.batches.start(self.predictor.llm.model, entries)
            try:
                batch = client.submit(
                    [
                        {"custom_id": custom_id, "body": self.predictor.request_body(self.items[key], self.prepared[key])}
                        for custom_id, key in zip(entries, keys)
                    ]
                )
            except BatchAPIError:
                path.unlink()  # rejected, so nothing to collect; after any other error the next run looks for it
                raise
            self.batches.accept(path, batch)
            report.requested += len(keys)
            log.info(
                "Submitted batch %s: %s, %s with %s", batch["id"], plural(len(keys), "item"), irt_model, plural(points, "point")
            )

    def _collect(self, client: OpenRouterBatches, report: RequestReport) -> None:
        """Save the predictions of every open batch that has reached a terminal status, and close its manifest."""
        for manifest in self.batches.open():
            batch = client.get(manifest["batch_id"])
            counts = batch.get("request_counts") or {}
            if batch["status"] not in TERMINAL_STATUSES:
                log.info(
                    "Batch %s: %s, %s of %s completed, %s failed",
                    batch["id"],
                    batch["status"],
                    counts.get("completed"),
                    counts.get("total"),
                    counts.get("failed"),
                )
                continue
            results = {result["custom_id"]: result for result in batch.get("results") or []}
            outcomes = {}
            for custom_id, entry in manifest["requests"].items():
                key, result = entry["item_key"], results.get(custom_id)
                error = None
                if result is None:
                    error = report.failed[key] = f"no result: the batch {batch['status']}"
                elif result.get("error") or (result.get("response") or {}).get("status_code") != 200:
                    error = report.failed[key] = json.dumps(result.get("error") or result.get("response"), ensure_ascii=False)
                else:
                    try:
                        output = self.predictor.parse_response(self.items[key], result["response"]["body"])
                    except UNUSABLE_OUTPUT_ERRORS as exception:
                        error = report.unusable[key] = f"{type(exception).__name__}: {exception}"
                    else:
                        self.cache.save_stem(custom_id, entry["record"] | output | {"batch_id": batch["id"]})
                        report.saved += 1
                outcomes[custom_id] = "saved" if error is None else error
            cost = (batch.get("usage") or {}).get("cost") or 0.0
            report.cost_usd += cost
            if batch["status"] != "completed":
                log.warning("Batch %s %s: %s", batch["id"], batch["status"], batch.get("error"))
            log.info(
                "Collected batch %s (%s): %d saved of %d, $%.2f",
                batch["id"],
                batch["status"],
                sum(outcome == "saved" for outcome in outcomes.values()),
                len(outcomes),
                cost,
            )
            self.batches.close(manifest, batch, outcomes)

    def results(self) -> PredictionResults:
        """The predictions saved for the current prompts and LLM settings, with their errors."""
        saved = self.saved()
        if len(saved) < len(self.items):
            log.warning("%d items have no current prediction and are left out", len(self.items) - len(saved))
        records = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in saved.items()}
        predictions = pd.DataFrame(
            [prediction_row(self.bank, key, record, saved[key].name) for key, record in records.items()]
        )
        if not predictions.empty:
            predicted = predictions[[f"{name}_predicted" for name in PARAMETER_NAMES]].notna().to_numpy()
            calibrated = predictions[[f"{name}_calibrated" for name in PARAMETER_NAMES]].notna().to_numpy()
            mismatched = predictions.loc[(predicted != calibrated).any(axis=1), "item_key"].tolist()
            if mismatched:
                raise ValueError(f"Predicted and calibrated items have different parameters: {mismatched}")
        settings = flatten(self.predictor.describe()) | {
            "evaluation.items": len(self.items),
            "evaluation.predicted_items": len(records),
            "evaluation.cache_dir": str(self.cache.directory),
        }
        return PredictionResults(predictions, prediction_errors(self.bank, records), settings)


def run_sync(coroutine: Coroutine):
    """Run a coroutine to the end from synchronous code, and return its result.

    Without a running event loop (a script or the command line) it runs in a new one. Inside a running loop (a
    notebook) it runs in a new loop in a thread, and an interrupt cancels it there before it is raised here.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    loop = asyncio.new_event_loop()
    task = loop.create_task(coroutine)

    def run():
        with contextlib.suppress(BaseException):  # task.result() raises it in the calling thread
            loop.run_until_complete(task)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()

    thread = threading.Thread(target=run, name="run_sync")
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        with contextlib.suppress(RuntimeError):  # the loop has closed: the task is done
            loop.call_soon_threadsafe(task.cancel)
        thread.join()
        raise
    return task.result()


def prediction_row(bank: ItemBank, key: str, record: dict, file_name: str) -> dict:
    """One row of ``predictions``: the item, its context, each parameter with its error, and the LLM's output."""
    calibrated = bank.calibrated.loc[key]
    row = {
        "item_key": key,
        **{column: record[column] for column in ("assessment", "grade", "subject", "item_id")},
        "item_label": bank.items_by_key[key]["item_label"],
        **{
            column: record[column]
            for column in (
                "content_domain",
                "main_topic",
                "cognitive_domain",
                "item_type",
                "max_points",
                "scaled_points",
                "irt_model",
            )
        },
        "calibration": calibrated["calibration"],
        "excluded_item_ids": ", ".join(
            bank.items_by_key[part]["item_id"] for part in record["excluded_item_keys"] if part != key
        ),
        "n_similar_items": len(record["similar_items"]),
        "similar_item_ids": ", ".join(f"{entry['item_id']} ({entry['assessment']})" for entry in record["similar_items"]),
    }
    for name in PARAMETER_NAMES:
        predicted = np.nan if record.get(name) is None else float(record[name])
        error = predicted - calibrated[name]  # NaN for parameters the item's model lacks
        row |= {
            f"{name}_predicted": predicted,
            f"{name}_calibrated": calibrated[name],
            f"{name}_error": error,
            f"{name}_abs_error": abs(error),
        }
    return row | {
        "llm_provider": record.get("llm_provider", "openrouter"),
        "llm_model": record["llm_model"],
        "reasoning_effort": record["reasoning_effort"],
        "batch_id": record.get("batch_id"),
        "reasoning": record["reasoning"],
        **{f"{name}_reasoning": record.get(f"{name}_reasoning") for name in PARAMETER_NAMES},
        **{
            column: record[column]
            for column in (
                "llm_thinking",
                "finish_reason",
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "cost_usd",
                "predicted_at",
            )
        },
        "similar_items": json.dumps(record["similar_items"], ensure_ascii=False),
        "system_prompt": record["system_prompt"],
        "user_prompt": record["user_prompt"],
        "prediction_file": file_name,
    }


def prediction_errors(bank: ItemBank, records: dict[str, dict]) -> pd.DataFrame:
    """One row per item and parameter the LLM predicted (the last GPCM step is derived, so it is left out), with the
    predicted and calibrated values, the error (predicted minus calibrated), and two baselines that need no LLM:
    ``calibration_mean``, the mean in the item's calibration summary (the calibrated items of its grade with its
    model and points, its parts left out), and ``reference_mean``, the mean over its reference items with its model
    and points."""
    rows = []
    for key, record in records.items():
        item = bank.item(key)
        prior = bank.same_scoring_items(item, exclude_keys=record["excluded_item_keys"])
        same_scoring_references = [
            entry
            for entry in record["similar_items"]
            if entry["irt_model"] == item.irt_model and entry["scaled_points"] == item.points
        ]
        for name in predicted_parameters(item.irt_model, item.points):
            rows.append(
                {
                    "item_key": key,
                    "assessment": item.assessment,
                    "grade": item.grade,
                    "item_id": item.item_id,
                    "irt_model": item.irt_model,
                    "parameter": name,
                    "predicted": record[name],
                    "calibrated": bank.calibrated.at[key, name],
                    "calibration_mean": prior[name].mean(),
                    "reference_mean": np.mean([entry[name] for entry in same_scoring_references])
                    if same_scoring_references
                    else np.nan,
                }
            )
    columns = ["item_key", "assessment", "grade", "item_id", "irt_model", "parameter", "predicted", "calibrated"]
    errors = pd.DataFrame(rows, columns=[*columns, "calibration_mean", "reference_mean"])
    errors["error"] = errors["predicted"] - errors["calibrated"]
    errors["parameter"] = pd.Categorical(errors["parameter"], categories=PARAMETER_NAMES, ordered=True)
    return errors


def summarize_errors(rows: pd.DataFrame) -> pd.Series:
    """Bias, spread, mean absolute error, RMSE, and correlation of the predictions, and the baselines' MAE."""
    error = rows["predicted"] - rows["calibrated"]

    def baseline_mae(baseline):
        return (rows[baseline] - rows["calibrated"]).abs().mean()

    return pd.Series(
        {
            "items": len(rows),
            "mean_error": error.mean(),
            "sd_error": error.std(),
            "mean_absolute_error": error.abs().mean(),
            "rmse": np.sqrt((error**2).mean()),
            "correlation": rows["predicted"].corr(rows["calibrated"]) if len(rows) > 2 else np.nan,
            "baseline_mae_calibration_mean": baseline_mae("calibration_mean"),
            "baseline_mae_reference_mean": baseline_mae("reference_mean"),
        }
    )


def error_summary(errors: pd.DataFrame, by: str = "grade") -> pd.DataFrame:
    """summarize_errors for each value of ``by`` and each parameter, then for all values together ("all")."""
    columns = ["predicted", "calibrated", "calibration_mean", "reference_mean"]
    summary = pd.concat(
        [
            errors.groupby([by, "parameter"], observed=True)[columns].apply(summarize_errors),
            errors.assign(**{by: "all"}).groupby([by, "parameter"], observed=True)[columns].apply(summarize_errors),
        ]
    )
    return summary.astype({"items": int})


def flatten(settings: dict, prefix: str = "") -> dict:
    """Nested settings as one level: {"llm": {"model": ...}} -> {"llm.model": ...}."""
    flat = {}
    for name, value in settings.items():
        if isinstance(value, dict):
            flat |= flatten(value, f"{prefix}{name}.")
        else:
            flat[f"{prefix}{name}"] = value
    return flat


def excel_value(value):
    """A value Excel accepts: no control characters, and at most EXCEL_CELL_LIMIT characters."""
    if not isinstance(value, str):
        return value
    value = ILLEGAL_CHARACTERS_RE.sub("", value)
    if len(value) <= EXCEL_CELL_LIMIT:
        return value
    note = " [cut at the Excel cell limit; the saved prediction has the full text]"
    return value[: EXCEL_CELL_LIMIT - len(note)] + note


@dataclass
class PredictionResults:
    """The leave-one-out predictions of a run and their errors."""

    predictions: pd.DataFrame
    """One row per item: the item, its context, each parameter predicted and calibrated with its error, and the LLM's
    reasoning, usage, and prompts."""
    errors: pd.DataFrame
    """One row per item and predicted parameter: predicted, calibrated, error, and the two baselines."""
    settings: dict = field(default_factory=dict)
    """The settings of the run, such as {"llm.model": "google/gemini-3.8-flash"}."""

    def summary(self, by: str = "grade") -> pd.DataFrame:
        """Error summary by ``by`` (such as "grade", "assessment", or "irt_model") and parameter, and over all items."""
        return error_summary(self.errors, by)

    def to_excel(self, path: str | Path) -> Path:
        """Export the predictions, the summaries by grade and assessment, the errors, and the settings.

        An Excel cell holds at most 32,767 characters, so a longer text, such as a long prompt, is cut there; the
        saved prediction has it in full.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path) as writer:
            self.predictions.map(excel_value).to_excel(writer, sheet_name="predictions", index=False)
            self.summary("grade").reset_index().to_excel(writer, sheet_name="summary_by_grade", index=False)
            self.summary("assessment").reset_index().to_excel(writer, sheet_name="summary_by_assessment", index=False)
            self.errors.to_excel(writer, sheet_name="prediction_errors", index=False)
            values = [json.dumps(value, default=str) for value in self.settings.values()]
            pd.DataFrame({"setting": list(self.settings), "value": values}).to_excel(
                writer, sheet_name="settings", index=False
            )
        return path

    @classmethod
    def read_excel(cls, path: str | Path) -> PredictionResults:
        """Read a workbook written by ``to_excel`` or by the notebook (which has no settings sheet)."""
        with pd.ExcelFile(path) as workbook:
            predictions = workbook.parse("predictions")
            errors = workbook.parse("prediction_errors")
            settings = {}
            if "settings" in workbook.sheet_names:
                # Values are JSON, such as "null", which pandas would otherwise read as NaN.
                rows = workbook.parse("settings", dtype=str, keep_default_na=False)
                settings = {row.setting: json.loads(row.value) for row in rows.itertuples()}
        errors["parameter"] = pd.Categorical(errors["parameter"], categories=PARAMETER_NAMES, ordered=True)
        return cls(predictions, errors, settings)


def run_name(predictor: ParameterPredictor) -> str:
    """A name for the run's output directory, from its main settings."""
    llm, embedding = predictor.llm, predictor.index.embedder.settings
    return "__".join(
        [
            f"{llm.provider}-{slug(llm.model)}-{slug(str(llm.reasoning_effort))}",
            f"{embedding.provider}-{slug(embedding.model)}-{embedding.dimension or 'full'}",
            f"{predictor.search_method}-{predictor.n_similar}+{predictor.n_same_model}",
        ]
    )


def _optional_int(value: str) -> int | None:
    return None if value.lower() in ("full", "none", "all") else int(value)


def _reasoning_effort(value: str) -> str | bool | None:
    return {"omit": None, "true": True, "false": False}.get(value.lower(), value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    embedding_defaults, index_defaults, llm_defaults = EmbeddingSettings(), IndexSettings(), LLMSettings()
    default_models = ", ".join(f"{model} for {provider}" for provider, model in DEFAULT_EMBEDDING_MODELS.items())
    parser = argparse.ArgumentParser(
        prog="timss-predict-loo",
        description=(
            "Predict the IRT parameters of every calibrated TIMSS released mathematics item as if it were new, and "
            "write the predictions, their errors, and graphs of the errors. Defaults are the notebook's settings. "
            "LLM requests cost money, so none are sent unless --max-requests is given. With --batch they are sent as "
            "OpenRouter batches at about half the price, which can take up to 24 hours."
        ),
    )
    paths = parser.add_argument_group("data")
    paths.add_argument(
        "--data-dir",
        type=Path,
        help="directory of the caches and runs (default: $TIMSS_DATA_DIR, else notebooks/data of the clone, else "
        "timss_math_data in the working folder)",
    )
    paths.add_argument("--grade", type=int, nargs="+", choices=[4, 8, 12], help="predict only items of these grades")
    paths.add_argument("--assessment", nargs="+", help='predict only items of these assessments, such as "TIMSS 2011"')

    embedding = parser.add_argument_group("embedding")
    embedding.add_argument("--embedding-provider", choices=EMBEDDING_PROVIDERS, default=embedding_defaults.provider)
    embedding.add_argument("--embedding-model", help=f"default: {default_models}; other providers need a model")
    embedding.add_argument(
        "--embedding-dimension",
        type=_optional_int,
        default=argparse.SUPPRESS,
        help=f"vector size, or 'full' for the model's own (default: full for fastembed, which cannot shorten vectors, "
        f"else {embedding_defaults.dimension})",
    )
    embedding.add_argument(
        "--fastembed-cache-dir", type=Path, help="where FastEmbed keeps its models (default: FastEmbed's own)"
    )
    embedding.add_argument("--document-prefix", default="", help="prepended to item texts, such as 'search_document: '")
    embedding.add_argument("--query-prefix", default="", help="prepended to queries, such as 'search_query: '")
    embedding.add_argument("--embedding-base-url", help="API URL of the embedding provider")
    embedding.add_argument("--embedding-cache", type=Path, help="item embeddings file (default: by embedding settings)")

    retrieval = parser.add_argument_group("retrieval")
    retrieval.add_argument("--search-method", choices=SEARCH_METHODS, default="hybrid")
    retrieval.add_argument("--n-similar", type=int, default=10, help="most similar items of any format (default: 10)")
    retrieval.add_argument(
        "--n-same-model", type=int, default=5, help="most similar items with the item's IRT model and points (default: 5)"
    )
    retrieval.add_argument("--hybrid-candidates", type=int, default=index_defaults.hybrid_candidates)
    retrieval.add_argument("--qdrant-path", type=Path, help="local Qdrant storage (default: in memory)")
    retrieval.add_argument("--qdrant-url", help="Qdrant server URL (default: in memory)")
    retrieval.add_argument("--collection", default=index_defaults.collection)

    llm = parser.add_argument_group("LLM")
    llm.add_argument("--llm-provider", choices=sorted(LLM_PROVIDERS), default=llm_defaults.provider)
    llm.add_argument("--llm-model", default=llm_defaults.model)
    llm.add_argument(
        "--reasoning-effort",
        type=_reasoning_effort,
        default=llm_defaults.reasoning_effort,
        help="such as low, medium, or high; 'omit' sends none; Ollama also takes true or false (default: %(default)s)",
    )
    llm.add_argument("--max-tokens", type=int, default=llm_defaults.max_tokens)
    llm.add_argument("--llm-base-url", help="API URL of the LLM provider")
    llm.add_argument("--ollama-num-ctx", type=int, default=llm_defaults.ollama_num_ctx)

    run = parser.add_argument_group("run")
    run.add_argument(
        "--max-requests",
        type=_optional_int,
        default=0,
        help="LLM requests to send at most, or 'all' (default: 0, which only evaluates saved predictions)",
    )
    run.add_argument("--workers", type=int, default=8, help="concurrent LLM requests (default: 8)")
    run.add_argument(
        "--batch",
        action="store_true",
        help="send the requests as OpenRouter batches at about half the price, and collect the batches of earlier "
        "runs (OpenRouter only)",
    )
    run.add_argument(
        "--no-wait",
        action="store_true",
        help="with --batch, submit and stop without waiting; rerun the same command to collect",
    )
    run.add_argument(
        "--poll-seconds", type=float, default=60, help="with --batch, seconds between status checks (default: 60)"
    )
    run.add_argument("--cache-dir", type=Path, help="where predictions are saved (default: by LLM provider and model)")
    run.add_argument(
        "--output-dir",
        type=Path,
        help="where the workbook and graphs go (default: <data>/parameter_prediction_runs/<settings>/)",
    )
    run.add_argument("--no-plots", action="store_true", help="write the workbook only")
    args = parser.parse_args(argv)
    if args.batch and args.llm_provider != "openrouter":
        parser.error("--batch sends requests through OpenRouter only")
    if args.no_wait and not args.batch:
        parser.error("--no-wait needs --batch")
    if args.embedding_model is None:
        if args.embedding_provider not in DEFAULT_EMBEDDING_MODELS:
            parser.error(f"--embedding-provider {args.embedding_provider} needs --embedding-model")
        args.embedding_model = DEFAULT_EMBEDDING_MODELS[args.embedding_provider]
    if "embedding_dimension" not in args:
        args.embedding_dimension = None if args.embedding_provider == "fastembed" else embedding_defaults.dimension
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # keeps printed lines in order with the log lines on stderr
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "httpx2"):  # one line per request otherwise; openai sends through httpx2
        logging.getLogger(name).setLevel(logging.WARNING)
    embedding = EmbeddingSettings(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimension=args.embedding_dimension,
        document_prefix=args.document_prefix,
        query_prefix=args.query_prefix,
        base_url=args.embedding_base_url,
        cache_path=args.embedding_cache,
        fastembed_cache_dir=args.fastembed_cache_dir,
    )
    index_settings = IndexSettings(
        collection=args.collection, path=args.qdrant_path, url=args.qdrant_url, hybrid_candidates=args.hybrid_candidates
    )
    llm = LLMSettings(
        provider=args.llm_provider,
        model=args.llm_model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        base_url=args.llm_base_url,
        ollama_num_ctx=args.ollama_num_ctx,
    )
    with ItemIndex.build(embedding, index_settings, data_dir=args.data_dir) as index:
        predictor = ParameterPredictor(index, llm, args.search_method, args.n_similar, args.n_same_model)
        calibrated = index.bank.calibrated
        keep = pd.Series(True, index=calibrated.index)
        if args.grade:
            keep &= calibrated["grade"].isin(args.grade)
        if args.assessment:
            keep &= calibrated["assessment"].isin(args.assessment)
        selected = calibrated.index[keep]
        if selected.empty:
            raise SystemExit("No calibrated items match --grade and --assessment")
        loo = LeaveOneOut(predictor, selected, args.cache_dir, args.data_dir)
        print(loo.status().value_counts().rename("items").to_frame().to_string(), end="\n\n")
        if args.batch:
            # Also with --max-requests 0, which collects the batches of earlier runs without submitting any.
            report = loo.request_batch(args.max_requests, not args.no_wait, args.poll_seconds)
            print(
                f"Submitted {report.requested} predictions in batches and saved {report.saved} in "
                f"{report.seconds:.0f} s for ${report.cost_usd:.2f}"
            )
        elif args.max_requests != 0:
            report = loo.request(args.max_requests, args.workers)
            print(f"Requested {report.requested} predictions in {report.seconds:.0f} s for ${report.cost_usd:.2f}")
        if args.batch or args.max_requests != 0:
            for key, error in report.unusable.items():
                print(f"  unusable output, left for a rerun: {key}: {error}")
            for key, error in report.failed.items():
                print(f"  no response, left for a rerun: {key}: {error}")
            if report.in_flight:
                print(f"{report.in_flight} predictions are in batches still running; rerun with --batch to collect them")
        results = loo.results()
        name = run_name(predictor)
    if results.predictions.empty:
        raise SystemExit(
            "No item has a saved prediction for these settings; pass --max-requests to request some, or rerun with "
            "--batch to collect submitted batches"
        )

    output_dir = args.output_dir or resolve_data_dir(args.data_dir) / RUNS_DIR / name
    workbook = results.to_excel(output_dir / "predictions.xlsx")
    print(f"Wrote {len(results.predictions)} predictions to {workbook}")
    if not args.no_plots:
        from .plots import save_all

        for path in save_all(results.errors, output_dir / "plots"):
            print(f"Wrote {path}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nPrediction errors (predicted minus calibrated) by grade:")
        print(results.summary("grade").round(3).to_string())


if __name__ == "__main__":
    main()
