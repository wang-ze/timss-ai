"""Batches of LLM requests through OpenRouter's Batch API, billed at about half the price of single requests.

``OpenRouterBatches`` submits a batch (``POST /api/v1/batches``), gets its status and, once it reaches a terminal
status, its results (``GET /api/v1/batches/{id}``), and lists recent batches. A batch can take up to 24 hours, and
OpenRouter returns results only when the whole batch has completed: a batch that failed, expired, or was cancelled
returns none. On Google models every request of a batch must ask for the same ``response_format``, so a batch holds
items of one prediction schema only.

``BatchManifests`` keeps a manifest of every submitted batch that has not been collected yet, so that an interrupted
run collects its batches on the next run instead of submitting (and paying for) them again:

- ``<batch id>.json``: an open batch, with the item and prediction record of each request.
- ``submitting-<uuid>.json``: written before a batch is submitted and renamed once OpenRouter accepts it. If a run stops
  in between, the next run finds the batch in OpenRouter's list (``adopt_submitting``).
- ``done/<batch id>.json``: a collected batch, with its status, usage, and the outcome of each request.

Two processes that share a cache, such as a notebook and a ``timss-predict-loo --batch`` run, would otherwise collect
the same batch twice, or take the other's batch in submission for one that OpenRouter never received and submit its
items again. So ``BatchManifests.locked`` holds an exclusive lock on ``.lock`` in the manifest directory while a
process submits or collects. The lock is ``filelock``'s, which locks the file through the operating system on Unix
(``fcntl.flock``) and on Windows (``LockFileEx``), so the operating system releases it when the process ends.

``LeaveOneOut.request_batch`` (``evaluation``) uses both.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import requests
from filelock import FileLock, Timeout

from .config import api_key, write_json
from .predictor import BASE_URLS, LLMSettings

log = logging.getLogger(__name__)

CHAT_COMPLETIONS = "/v1/chat/completions"
TERMINAL_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
SUBMITTING_PREFIX = "submitting-"


class BatchAPIError(RuntimeError):
    """OpenRouter answered a batch API call with an HTTP error, so a submitted batch was not received."""


class OpenRouterBatches:
    """OpenRouter's Batch API for the model of an ``LLMSettings``, whose provider must be "openrouter"."""

    def __init__(self, settings: LLMSettings):
        if settings.provider != "openrouter":
            raise ValueError(f"Batches are sent through OpenRouter only, not {settings.provider!r}")
        self.settings = settings
        self.url = f"{(settings.base_url or BASE_URLS['openrouter']).rstrip('/')}/batches"

    def _call(self, method: str, url: str, **kwargs) -> dict:
        headers = {"Authorization": f"Bearer {api_key('OPENROUTER_API_KEY', self.settings.api_key)}"}
        if "data" in kwargs:
            headers["Content-Type"] = "application/json"
        response = requests.request(method, url, headers=headers, timeout=self.settings.timeout, **kwargs)
        if not response.ok:
            raise BatchAPIError(f"OpenRouter {method} {url} returned HTTP {response.status_code}: {response.text}")
        return response.json()

    def submit(self, batch_requests: list[dict]) -> dict:
        """Submit ``[{"custom_id": ..., "body": ...}, ...]`` as one chat-completions batch of the settings' model.
        Returns the batch object, whose status is "validating". Raises BatchAPIError if OpenRouter rejects it; on a
        network error or timeout it may have received it anyway."""
        # OpenRouter stream-parses the body and rejects it if "requests" comes before "endpoint" and "model".
        body = {"endpoint": CHAT_COMPLETIONS, "model": self.settings.model, "requests": batch_requests}
        return self._call("POST", self.url, data=json.dumps(body, ensure_ascii=False).encode())

    def get(self, batch_id: str) -> dict:
        """The batch object: its status, request counts, usage, and, in a terminal status, its results."""
        return self._call("GET", f"{self.url}/{batch_id}")

    def list_created_after(self, created_after: float) -> list[dict]:
        """The batches of the workspace created after a Unix time, newest first, without their results."""
        batches, after = [], None
        while True:
            params = {"limit": 100, "created_after": int(created_after)} | ({"after": after} if after else {})
            page = self._call("GET", self.url, params=params)
            batches += page["data"]
            if not page.get("has_more"):
                return batches
            after = page["last_id"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class BatchManifests:
    """The manifests of the batches of a prediction cache, in ``directory`` (``<cache dir>/batches``)."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the exclusive lock of the manifests, waiting while another process holds it."""
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(self.directory / ".lock")
        try:
            lock.acquire(blocking=False)
        except Timeout:
            log.info("Waiting for another process that is submitting or collecting batches")
            lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def open(self) -> list[dict]:
        """The manifests of the batches submitted but not collected yet, oldest first."""
        if not self.directory.exists():
            return []
        manifests = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in self.directory.glob("*.json")
            if not path.name.startswith(SUBMITTING_PREFIX)
        ]
        return sorted(manifests, key=lambda manifest: manifest["submitted_at"])

    def open_custom_ids(self) -> set[str]:
        return {custom_id for manifest in self.open() for custom_id in manifest["requests"]}

    def start(self, model: str, entries: dict[str, dict]) -> Path:
        """Write the manifest of a batch about to be submitted: ``entries`` maps each custom_id to its item key and
        prediction record."""
        path = self.directory / f"{SUBMITTING_PREFIX}{uuid.uuid4().hex}.json"
        write_json(
            path,
            {
                "batch_id": None,
                "model": model,
                "submitted_at": now_iso(),
                "submitted_unix": time.time(),
                "requests": entries,
            },
        )
        return path

    def accept(self, path: Path, batch: dict) -> dict:
        """Name the manifest at ``path`` after the batch that OpenRouter accepted."""
        manifest = json.loads(path.read_text(encoding="utf-8")) | {"batch_id": batch["id"]}
        write_json(self.directory / f"{batch['id']}.json", manifest)
        path.unlink()
        return manifest

    def adopt_submitting(self, client: OpenRouterBatches) -> None:
        """Resolve the manifests of submissions that a stopped run left unconfirmed: accept the batch OpenRouter
        received, or drop the manifest if it received none."""
        known = {path.stem for path in self.directory.rglob("*.json")}
        for path in sorted(self.directory.glob(f"{SUBMITTING_PREFIX}*.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            candidates = [
                batch
                for batch in client.list_created_after(manifest["submitted_unix"] - 60)
                if batch["id"] not in known
                and batch["model"] == manifest["model"]
                and batch["request_counts"]["total"] == len(manifest["requests"])
            ]
            if len(candidates) > 1:
                ids = ", ".join(batch["id"] for batch in candidates)
                raise RuntimeError(
                    f"{path} is a batch whose submission was not confirmed, and OpenRouter has several batches that "
                    f"match it ({ids}). Rename the file to <batch id>.json with the right batch id, or delete it if "
                    f"none is right."
                )
            if candidates:
                log.info("Found batch %s, whose submission was not confirmed", candidates[0]["id"])
                known.add(self.accept(path, candidates[0])["batch_id"])
            else:
                log.info("No batch was received for %s; its items are pending again", path.name)
                path.unlink()

    def close(self, manifest: dict, batch: dict, outcomes: dict[str, str]) -> None:
        """Move a collected batch to ``done/``, with its status and usage and the outcome of each request, and
        without the prediction records (the saved predictions have them)."""
        write_json(
            self.directory / "done" / f"{manifest['batch_id']}.json",
            {
                **{name: manifest[name] for name in ("batch_id", "model", "submitted_at")},
                "collected_at": now_iso(),
                **{name: batch.get(name) for name in ("status", "finalized_at", "request_counts", "usage", "error")},
                "requests": {
                    custom_id: {"item_key": entry["item_key"], "outcome": outcomes[custom_id]}
                    for custom_id, entry in manifest["requests"].items()
                },
            },
        )
        (self.directory / f"{manifest['batch_id']}.json").unlink()
