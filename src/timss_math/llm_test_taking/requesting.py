"""Requests to an LLM whose outputs are saved as JSON records, sent directly or as OpenRouter batches.

A ``Request`` is a system prompt, a user prompt, an output schema, and the JSON file where its output is saved, named
by a hash of everything sent (``make_request``), so that a saved output is reused only for the same request.

``send`` sends the requests of one LLM that have no saved output, at most ``max_requests`` of them (None for all, 0
for none), in one of two modes:

- ``"direct"``: the requests go now, ``workers`` at a time through the package's async client (``LLMClient.session``),
  and each output is saved as soon as it arrives.
- ``"batch"``: the requests go as OpenRouter batches through the package's ``OpenRouterBatches`` (the Batch API, at
  about half the price, which can take up to 24 hours) for the models that OpenRouter batches (a ``:batch`` variant
  in its model list), and directly for the other models and for every other provider. A model whose batch OpenRouter
  rejects although its list has a ``:batch`` variant (HTTP 400, "does not have a :batch endpoint", as for
  ``gpt-4o-mini``) is asked directly too.

Every run first collects the batches that earlier runs submitted, in either mode, then submits new ones; with
``wait``, it polls every ``poll_seconds`` until all are collected, and otherwise returns at once, so that a later run
(with ``max_requests`` 0) collects them. ``BatchManifests`` keeps a manifest of every submitted batch in the
``batches/`` folder of the records, locked while a process submits or collects, so that an interrupted run collects
its batches instead of paying for them again; never delete an open manifest (``<batch id>.json`` or
``submitting-*.json``). Google models need the same output schema for every request of a batch, so their requests
are batched by schema; a batch holds at most ``batch_size`` requests. OpenRouter reserves a batch's worst case, every
request using all its ``max_tokens``, against the key's balance until it completes, and answers HTTP 402 if it does
not fit, so submit large runs in parts.

Direct requests are awaited on the async client, and the blocking batch calls run in threads (``asyncio.to_thread``)
with ``asyncio.sleep`` between polls, so the requests of several LLMs can run at once in one event loop. Output that
cannot be used (cut off at ``max_tokens``, not fitting the schema, or missing) is reported and requested again on the
next run; any other error, such as an API error from exhausted credit, stops the run, and what was saved or submitted
until then stays saved.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import requests
from pydantic import BaseModel
from requests import RequestException

from ..batch import (
    TERMINAL_STATUSES,
    BatchAPIError,
    BatchManifests,
    OpenRouterBatches,
)
from ..config import slug, write_json
from ..llm import (
    UNUSABLE_ERRORS,
    Completion,
    LLMClient,
    LLMSettings,
)

log = logging.getLogger(__name__)

MAX_POLL_FAILURES = 10
"""Status checks in a row that may fail, such as on a dropped connection, before a waiting run stops."""
NO_BATCH_ENDPOINT = "does not have a :batch endpoint"
"""OpenRouter's HTTP 400 for a model that it cannot batch."""


@dataclass(frozen=True)
class Request:
    """A request whose output is saved as a JSON record in ``path``, together with ``fields``."""

    name: str
    system_prompt: str
    user_prompt: str
    schema: type[BaseModel]
    path: Path
    fields: dict

    @property
    def custom_id(self) -> str:
        """The hash of everything sent: the custom_id of the request in a batch."""
        return self.path.stem.rsplit("_", 1)[1]

    def saved(self) -> dict | None:
        """The saved record, or None."""
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None

    def save(self, completion: Completion, seconds: float | None, batch_id: str | None = None) -> None:
        """Save the output with ``fields``, the usage, and when it was requested."""
        write_json(
            self.path,
            self.fields
            | {
                "output": completion.parsed.model_dump(),
                "llm_thinking": completion.thinking,
                "finish_reason": completion.finish_reason,
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "reasoning_tokens": completion.reasoning_tokens,
                "cost_usd": completion.cost_usd,
                "seconds": None if seconds is None else round(seconds, 3),
                "batch_id": batch_id,
                "requested_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )


def make_request(
    folder: Path,
    settings: LLMSettings,
    name: str,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
    schema_json: dict | None = None,
    **fields,
) -> Request:
    """A request whose record is ``<folder>/<name>_<hash>.json``, the hash covering the LLM, its reasoning effort,
    both prompts, and the output schema (``schema_json``, ``schema.model_json_schema()`` if not given). ``max_tokens``
    and how the LLM is called are not in it, so changing them reuses the saved outputs."""
    identity = {
        "llm_provider": settings.provider,
        "llm_model": settings.model,
        "reasoning_effort": settings.reasoning_effort,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "output_schema": schema.model_json_schema() if schema_json is None else schema_json,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    path = Path(folder) / f"{slug(name)}_{digest}.json"
    return Request(name, system_prompt, user_prompt, schema, path, fields | identity)


@dataclass
class RunReport:
    """What a run did with the requests of one LLM."""

    mode: str
    """"direct", "batch", or "nothing sent"."""
    missing: int = 0
    """Requests without a saved output (and not in an open batch) before the run."""
    requested: int = 0
    saved: int = 0
    """Outputs saved, including those of batches that earlier runs submitted."""
    in_batches: int = 0
    """Requests in batches still open after the run; a later run collects them."""
    cost_usd: float = 0.0
    """The cost that OpenRouter reported for the requests and batches of the run."""
    uncosted: int = 0
    """Outputs whose provider reports no cost, such as the Claude API's."""
    unusable: dict[str, str] = field(default_factory=dict)
    """Request -> error, for output that could not be used; a rerun requests it again."""
    failed: dict[str, str] = field(default_factory=dict)
    """Request -> error, for batch requests that got no response; a rerun requests them again."""

    def describe(self, name: str) -> str:
        """One line on the run, then one line per unusable or failed request."""
        lines = [
            f"{name} ({self.mode}): {self.requested} of {self.missing} missing requests sent, {self.saved} outputs "
            f"saved, {self.in_batches} in batches still open, ${self.cost_usd:.4f}"
            + (f" and the unreported cost of {self.uncosted} outputs" if self.uncosted else "")
        ]
        lines += [f"  {key}: {error}" for key, error in {**self.unusable, **self.failed}.items()]
        return "\n".join(lines)


@functools.cache
def batch_models() -> frozenset[str]:
    """The OpenRouter models with a batch (":batch") variant, from its public model list."""
    response = requests.get("https://openrouter.ai/api/v1/models", timeout=60)
    response.raise_for_status()
    ids = [model["id"] for model in response.json()["data"]]
    return frozenset(model.removesuffix(":batch") for model in ids if model.endswith(":batch"))


class RequestBatches:
    """The OpenRouter batches of the requests of one LLM whose records share a folder, with their manifests in
    ``<folder>/batches``. ``client`` replaces the Batch API, such as a fake one in tests."""

    def __init__(self, settings: LLMSettings, folder: Path, client: OpenRouterBatches | None = None):
        self.settings = settings
        self.client = client or OpenRouterBatches(settings)
        self.llm = LLMClient(settings)
        self.manifests = BatchManifests(Path(folder) / "batches")

    def submit(self, pending: list[Request], batch_size: int, report: RunReport) -> None:
        """Submit ``pending`` in batches of at most ``batch_size`` requests, by output schema for a Google model."""
        groups: dict[str, list[Request]] = {}
        for request in pending:
            schema = json.dumps(request.schema.model_json_schema(), sort_keys=True)
            groups.setdefault(schema if self.settings.model.startswith("google/") else "", []).append(request)
        for group in groups.values():
            for start in range(0, len(group), batch_size):
                chunk = group[start : start + batch_size]
                path = self.manifests.start(
                    self.settings.model, {request.custom_id: {"item_key": request.name} for request in chunk}
                )
                bodies = [
                    {
                        "custom_id": request.custom_id,
                        "body": self.llm.chat_body(request.system_prompt, request.user_prompt, request.schema),
                    }
                    for request in chunk
                ]
                try:
                    batch = self.client.submit(bodies)
                except BatchAPIError:
                    path.unlink()  # rejected, so there is nothing to collect
                    raise
                self.manifests.accept(path, batch)
                report.requested += len(chunk)
                log.info("%s: submitted batch %s of %d requests", self.settings.model, batch["id"], len(chunk))

    def collect(self, requests: list[Request], report: RunReport) -> None:
        """Save the outputs of every open batch that has reached a terminal status, and close its manifest. A result
        whose request has changed since it was submitted is not saved."""
        by_id = {request.custom_id: request for request in requests}
        for manifest in self.manifests.open():
            batch = self.client.get(manifest["batch_id"])
            if batch["status"] not in TERMINAL_STATUSES:
                continue
            results = {result["custom_id"]: result for result in batch.get("results") or []}
            outcomes = {}
            for custom_id in manifest["requests"]:
                request, result = by_id.get(custom_id), results.get(custom_id)
                if request is None:
                    outcome = "not saved: the request has changed since it was submitted"
                elif result is None:
                    outcome = report.failed[custom_id] = f"no result: the batch {batch['status']}"
                elif result.get("error") or (result.get("response") or {}).get("status_code") != 200:
                    outcome = report.failed[custom_id] = json.dumps(result.get("error") or result.get("response"))
                else:
                    try:
                        completion = LLMClient.parse_chat_response(result["response"]["body"], request.schema)
                    except UNUSABLE_ERRORS as error:
                        outcome = report.unusable[custom_id] = f"{type(error).__name__}: {error}"
                    else:
                        request.save(completion, None, batch["id"])
                        outcome = "saved"
                        report.saved += 1
                outcomes[custom_id] = outcome
            report.cost_usd += (batch.get("usage") or {}).get("cost") or 0.0
            self.manifests.close(manifest, batch, outcomes)

    def submit_and_collect(
        self, requests: list[Request], max_requests: int | None, batch_size: int, report: RunReport
    ) -> None:
        with self.manifests.locked():  # another process may share the manifests
            self.manifests.adopt_submitting(self.client)
            self.collect(requests, report)
            in_batches = self.manifests.open_custom_ids()
            pending = [request for request in requests if not request.path.exists() and request.custom_id not in in_batches]
            report.missing = len(pending)
            self.submit(pending if max_requests is None else pending[:max_requests], batch_size, report)

    def collect_locked(self, requests: list[Request], report: RunReport) -> None:
        with self.manifests.locked():
            self.collect(requests, report)


async def send_direct(
    settings: LLMSettings,
    folder: Path,
    requests: list[Request],
    max_requests: int | None,
    workers: int = 8,
    session=None,
) -> RunReport:
    """Send the requests without a saved record (and not in a submitted batch) now, ``workers`` at a time.
    ``session`` replaces the LLM's async session (``LLMClient.session``), such as a fake one in tests."""
    report = RunReport("direct")
    in_batches = BatchManifests(Path(folder) / "batches").open_custom_ids()
    pending = [request for request in requests if not request.path.exists() and request.custom_id not in in_batches]
    todo = pending if max_requests is None else pending[:max_requests]
    report.missing, report.requested, report.in_batches = len(pending), len(todo), len(in_batches)
    if not todo:  # nothing to send, so no session, which would need an API key
        return report
    if session is None:
        async with LLMClient(settings).session() as opened:
            return await send_direct(settings, folder, requests, max_requests, workers, opened)
    semaphore = asyncio.Semaphore(max(1, workers))

    async def send_one(request: Request) -> None:
        async with semaphore:
            started = time.monotonic()
            try:
                completion = await session.complete(request.system_prompt, request.user_prompt, request.schema)
            except UNUSABLE_ERRORS as error:
                report.unusable[request.path.stem] = f"{type(error).__name__}: {error}"
                return
        request.save(completion, time.monotonic() - started)
        report.saved += 1
        if completion.cost_usd is None:  # only OpenRouter reports the cost of a request
            report.uncosted += 1
        else:
            report.cost_usd += completion.cost_usd

    async with asyncio.TaskGroup() as group:
        for request in todo:
            group.create_task(send_one(request))
    return report


async def send_batches(
    settings: LLMSettings,
    folder: Path,
    requests: list[Request],
    max_requests: int | None,
    wait: bool = False,
    batch_size: int = 200,
    poll_seconds: float = 60,
    client: OpenRouterBatches | None = None,
) -> RunReport:
    """Collect the batches of earlier runs and submit the pending requests as new batches; with ``wait``, poll until
    every batch is collected. ``client`` replaces the Batch API, such as a fake one in tests."""
    report = RunReport("batch")
    batches = RequestBatches(settings, folder, client)
    await asyncio.to_thread(batches.submit_and_collect, requests, max_requests, batch_size, report)
    failures = 0
    while wait and batches.manifests.open():
        await asyncio.sleep(poll_seconds)
        try:
            await asyncio.to_thread(batches.collect_locked, requests, report)
            failures = 0
        except (RequestException, BatchAPIError) as error:
            # A failed status check changes nothing: the batches stay open, and the next check tries again.
            failures += 1
            if failures > MAX_POLL_FAILURES:
                raise
            log.warning("%s: status check failed (%d in a row), trying again: %s", settings.model, failures, error)
    report.in_batches = len(batches.manifests.open_custom_ids())
    return report


def has_open_batches(folder: Path) -> bool:
    """Whether the folder has a batch submitted but not collected, or a submission not confirmed."""
    manifests = BatchManifests(Path(folder) / "batches")
    return bool(manifests.open()) or manifests.directory.exists() and any(manifests.directory.glob("submitting-*.json"))


async def send(
    settings: LLMSettings,
    folder: Path,
    requests: list[Request],
    max_requests: int | None,
    mode: str,
    wait: bool = False,
    workers: int = 8,
    batch_size: int = 200,
    poll_seconds: float = 60,
    session=None,
    batch_client: OpenRouterBatches | None = None,
) -> RunReport:
    """Send the pending requests of one LLM in batches or directly (see above), and collect the batches of earlier runs
    in either mode. With ``max_requests`` 0 and no batch to collect, nothing is sent, not even a request for
    OpenRouter's model list. ``session`` and ``batch_client`` replace the LLM's session and the Batch API, such as
    fakes in tests."""
    if mode not in ("batch", "direct"):
        raise ValueError(f"Unknown request mode {mode!r}; use 'batch' or 'direct'")
    open_batches = has_open_batches(folder)
    if max_requests == 0 and not open_batches:
        return RunReport("nothing sent", missing=sum(not request.path.exists() for request in requests))
    batched = {"wait": wait, "batch_size": batch_size, "poll_seconds": poll_seconds, "client": batch_client}
    if mode == "batch" and settings.provider == "openrouter" and settings.model in batch_models():
        try:
            return await send_batches(settings, folder, requests, max_requests, **batched)
        except BatchAPIError as error:
            # OpenRouter lists a ":batch" variant of some models whose batches it then rejects, such as gpt-4o-mini.
            if NO_BATCH_ENDPOINT not in str(error):
                raise
            log.warning("%s: OpenRouter takes no batches of it after all, so it is asked directly", settings.model)
    collected = await send_batches(settings, folder, requests, 0, **batched) if open_batches else RunReport("batch")
    report = await send_direct(settings, folder, requests, max_requests, workers, session)
    return replace(
        report,
        saved=report.saved + collected.saved,
        cost_usd=report.cost_usd + collected.cost_usd,
        in_batches=collected.in_batches,
        unusable=collected.unusable | report.unusable,
        failed=collected.failed,
    )
