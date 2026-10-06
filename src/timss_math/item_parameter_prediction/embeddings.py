"""Embedding models for item retrieval, and the cache of item embeddings.

The default settings are those of ``predict_item_parameters.ipynb``: ``gemini-embedding-001`` through the Gemini API,
768 dimensions, items embedded as ``RETRIEVAL_DOCUMENT`` and search queries as ``RETRIEVAL_QUERY``. Other providers:

- ``"openai"``: the OpenAI embeddings API, or any OpenAI-compatible server through ``base_url``.
- ``"openrouter"``: the OpenRouter embeddings API, with models such as ``openai/text-embedding-3-small``.
- ``"ollama"``: a local Ollama server, with models such as ``nomic-embed-text``.
- ``"fastembed"``: Qdrant's FastEmbed library, which runs the model on this computer with ONNX Runtime, so it needs no
  API key and has no quota, with models such as its default ``BAAI/bge-small-en-v1.5`` (384 dimensions), as in
  ``predict_item_parameters_FastEmbed.ipynb``. FastEmbed downloads the model from Hugging Face on first use. Each text
  is embedded on its own, because in a batch a text's vector depends slightly on the other texts of the batch. A text
  longer than the model's input (512 tokens for bge-small-en-v1.5) is cut there, with a warning. FastEmbed cannot
  shorten vectors, so ``dimension`` must be the model's size or None.

Only the Gemini API takes task types. Models of other providers that expect a prefix on queries and documents (such as
``search_query: `` and ``search_document: `` for ``nomic-embed-text``) get it from ``query_prefix`` and
``document_prefix``. Every vector is normalized to unit length, so cosine similarity is a dot product.

A failed request raises instead of being retried, so retries do not use up the Gemini free-tier quota.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import openai
import requests
from fastembed import TextEmbedding

from .config import api_key, data_dir, slug

log = logging.getLogger(__name__)

EmbeddingProvider = Literal["gemini", "openai", "openrouter", "ollama", "fastembed"]
EMBEDDING_PROVIDERS = ("gemini", "openai", "openrouter", "ollama", "fastembed")
TextKind = Literal["document", "query"]

BASE_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434",
}
API_KEY_VARIABLES = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
# The quota of each provider that has one, in texts a minute: the Gemini API free tier counts every text of a batch
# request as one request and allows 100 a minute.
TEXTS_PER_MINUTE = {"gemini": 100}
# The model of each provider that has a default one; the others need a model.
DEFAULT_MODELS = {"gemini": "gemini-embedding-001", "fastembed": "BAAI/bge-small-en-v1.5"}
# The item embeddings of the notebook, which the default settings read and extend.
NOTEBOOK_EMBEDDINGS_FILE = "timss_math_item_embeddings.npz"


@dataclass(frozen=True)
class EmbeddingSettings:
    """Which embedding model embeds the items and queries, and how it is called."""

    provider: EmbeddingProvider = "gemini"
    model: str = "gemini-embedding-001"
    dimension: int | None = 768
    """Requested vector size; None keeps the model's own size."""
    document_task_type: str = "RETRIEVAL_DOCUMENT"
    """Gemini task type of items, both stored and new ones."""
    query_task_type: str = "RETRIEVAL_QUERY"
    """Gemini task type of search queries."""
    document_prefix: str = ""
    """Prepended to every item text before it is embedded."""
    query_prefix: str = ""
    """Prepended to every search query before it is embedded."""
    batch_size: int = 100
    """The most texts in one request; 100 is the most a Gemini batchEmbedContents request takes."""
    texts_per_minute: int | None = None
    """The most texts to send in any minute; None uses the provider's quota (Gemini: 100, the others: no limit)."""
    quota_seconds: float = 61
    """The quota window, with a second to spare for differences between this clock and the server's."""
    timeout: float = 120
    base_url: str | None = None
    """The provider's API URL; None uses its public one (Ollama: http://localhost:11434). FastEmbed has none."""
    api_key: str | None = field(default=None, repr=False)
    """None reads the provider's variable (GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY) from the environment
    or the project .env. Ollama and FastEmbed need none."""
    cache_path: Path | None = None
    """Where the item embeddings are cached; None uses ``default_cache_path``."""
    fastembed_cache_dir: Path | None = None
    """Where FastEmbed keeps the models it downloads; None uses FastEmbed's default, ``$FASTEMBED_CACHE_PATH`` or else
    ``fastembed_cache`` in the system temp directory."""

    def __post_init__(self):
        if self.provider not in EMBEDDING_PROVIDERS:
            raise ValueError(f"Unknown embedding provider {self.provider!r}; use one of {EMBEDDING_PROVIDERS}")

    @property
    def identity(self) -> dict:
        """The settings that determine the item vectors; a cache holds vectors of one identity only."""
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "document_task_type": self.document_task_type if self.provider == "gemini" else None,
            "document_prefix": self.document_prefix,
        }

    @property
    def quota(self) -> int | None:
        return self.texts_per_minute or TEXTS_PER_MINUTE.get(self.provider)

    def default_cache_path(self, data_dir_path: str | Path | None = None) -> Path:
        """The notebook's ``timss_math_item_embeddings.npz`` for its settings, else a file of these settings in
        ``cache/item_embeddings/``."""
        directory = data_dir(data_dir_path)
        if self.identity == EmbeddingSettings().identity:
            return directory / NOTEBOOK_EMBEDDINGS_FILE
        digest = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()[:8]
        name = f"{self.provider}_{slug(self.model)}_{self.dimension or 'full'}_{digest}.npz"
        return directory / "cache" / "item_embeddings" / name

    def describe(self) -> dict:
        """The settings without the API key, for run reports."""
        return {name: value for name, value in asdict(self).items() if name != "api_key"}


class Embedder:
    """Embeds texts with the model of an EmbeddingSettings, within the provider's quota."""

    def __init__(self, settings: EmbeddingSettings = EmbeddingSettings()):
        self.settings = settings
        base_url = settings.base_url or BASE_URLS.get(settings.provider)
        self.base_url = base_url.rstrip("/") if base_url else None
        self._recent_requests: deque[tuple[float, int]] = deque()  # (time.monotonic(), number of texts)
        self._cache: dict[tuple[str, TextKind], np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._quota_lock = threading.Lock()

    @functools.cached_property
    def _api_key(self) -> str | None:
        """Read when the first request is sent, so that cached embeddings need no API key."""
        variable = API_KEY_VARIABLES.get(self.settings.provider)
        return api_key(variable, self.settings.api_key) if variable else None

    @functools.cached_property
    def _openai(self) -> openai.OpenAI:
        return openai.OpenAI(
            base_url=self.base_url, api_key=self._api_key, timeout=self.settings.timeout, max_retries=0
        )

    @functools.cached_property
    def fastembed_model(self) -> TextEmbedding:
        """The FastEmbed model, loaded when first used and downloaded the first time on this computer."""
        if self.settings.provider != "fastembed":
            raise ValueError(f"{self.settings.provider} is not FastEmbed")
        cache_dir = self.settings.fastembed_cache_dir
        return TextEmbedding(self.settings.model, cache_dir=None if cache_dir is None else str(cache_dir))

    @functools.cached_property
    def fastembed_max_tokens(self) -> int:
        """The most tokens the FastEmbed model reads of a text; its tokenizer cuts longer texts there."""
        return self.fastembed_model.model.tokenizer.truncation["max_length"]

    def reaches_input_limit(self, text: str) -> bool:
        """Whether ``text`` fills the FastEmbed model's input, so that anything beyond it is cut."""
        return self.fastembed_model.token_count(text) >= self.fastembed_max_tokens

    def embed(self, texts: Sequence[str], kind: TextKind = "document") -> np.ndarray:
        """Unit-length embeddings of ``texts``, one row per text, in requests of at most ``batch_size`` texts."""
        batches = [
            self._embed_batch(texts[start : start + self.settings.batch_size], kind)
            for start in range(0, len(texts), self.settings.batch_size)
        ]
        return np.concatenate(batches) if batches else np.empty((0, self.settings.dimension or 0), np.float32)

    def embed_one(self, text: str, kind: TextKind = "query") -> np.ndarray:
        """Embedding of one text, cached so that repeating a search does not call the API again."""
        with self._cache_lock:
            cached = self._cache.get((text, kind))
        if cached is None:
            cached = self._embed_batch([text], kind)[0]
            with self._cache_lock:
                self._cache[(text, kind)] = cached
        return cached

    def _embed_batch(self, texts: Sequence[str], kind: TextKind) -> np.ndarray:
        settings = self.settings
        prefix = settings.document_prefix if kind == "document" else settings.query_prefix
        texts = [prefix + text for text in texts]
        self._wait_for_quota(len(texts))
        if settings.provider == "gemini":
            vectors = self._gemini(texts, settings.document_task_type if kind == "document" else settings.query_task_type)
        elif settings.provider == "ollama":
            vectors = self._ollama(texts)
        elif settings.provider == "fastembed":
            vectors = self._fastembed(texts, kind)
        else:
            vectors = self._openai_compatible(texts)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise RuntimeError(f"{settings.model} returned {vectors.shape} for {len(texts)} texts")
        if settings.dimension and vectors.shape[1] != settings.dimension:
            raise RuntimeError(
                f"{settings.model} returned {vectors.shape[1]}-dimensional vectors instead of {settings.dimension}; "
                f"set dimension={vectors.shape[1]} or None"
            )
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def _wait_for_quota(self, text_count: int) -> None:
        """Sleep until sending ``text_count`` more texts keeps the texts sent in the quota window within the quota."""
        quota, window = self.settings.quota, self.settings.quota_seconds
        if quota is None:
            return
        if text_count > quota:
            raise ValueError(f"A request can embed at most {quota} texts; lower batch_size")
        with self._quota_lock:
            while True:
                now = time.monotonic()
                while self._recent_requests and now - self._recent_requests[0][0] >= window:
                    self._recent_requests.popleft()
                if sum(count for _, count in self._recent_requests) + text_count <= quota:
                    break
                time.sleep(window - (now - self._recent_requests[0][0]))
            self._recent_requests.append((time.monotonic(), text_count))

    def _gemini(self, texts: list[str], task_type: str) -> list[list[float]]:
        model = self.settings.model
        dimension = {"outputDimensionality": self.settings.dimension} if self.settings.dimension else {}
        response = requests.post(
            f"{self.base_url}/models/{model}:batchEmbedContents",
            headers={"x-goog-api-key": self._api_key},
            json={
                "requests": [
                    {"model": f"models/{model}", "content": {"parts": [{"text": text}]}, "taskType": task_type, **dimension}
                    for text in texts
                ]
            },
            timeout=self.settings.timeout,
        )
        if not response.ok:
            raise RuntimeError(f"{model} returned HTTP {response.status_code}: {response.text}")
        return [embedding["values"] for embedding in response.json()["embeddings"]]

    def _ollama(self, texts: list[str]) -> list[list[float]]:
        dimension = {"dimensions": self.settings.dimension} if self.settings.dimension else {}
        # truncate=False makes a text longer than the model's context an error instead of a silently cut embedding.
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.settings.model, "input": texts, "truncate": False, **dimension},
            timeout=self.settings.timeout,
        )
        if not response.ok:
            raise RuntimeError(f"Ollama {self.settings.model} returned HTTP {response.status_code}: {response.text}")
        return response.json()["embeddings"]

    def _fastembed(self, texts: list[str], kind: TextKind) -> list[np.ndarray]:
        model = self.fastembed_model
        embed = model.passage_embed if kind == "document" else model.query_embed
        # One text at a time: in a batch, a text's vector depends slightly on the other texts of the batch (by up to
        # 0.0002 in a dimension for bge-small-en-v1.5), so the same text could get different vectors.
        vectors = [vector for text in texts for vector in embed([text])]
        at_limit = sum(map(self.reaches_input_limit, texts))
        if at_limit:
            log.warning(
                "%d of %d texts reach the %d-token input of %s; their vectors cover only their first %d tokens",
                at_limit,
                len(texts),
                self.fastembed_max_tokens,
                self.settings.model,
                self.fastembed_max_tokens,
            )
        return vectors

    def _openai_compatible(self, texts: list[str]) -> list[list[float]]:
        dimension = {"dimensions": self.settings.dimension} if self.settings.dimension else {}
        response = self._openai.embeddings.create(
            model=self.settings.model, input=texts, encoding_format="float", **dimension
        )
        return [embedding.embedding for embedding in sorted(response.data, key=lambda embedding: embedding.index)]


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_item_embeddings(chunks: list[dict], embedder: Embedder, path: Path) -> dict[str, np.ndarray]:
    """The vector of every chunk's text by the text's SHA-256, embedding only the texts not cached in ``path``.

    The cache stores the vectors in chunk order with each item's key and the hash of its text, so a changed item is
    re-embedded and an interrupted run resumes where it stopped. It is written in the notebook's format, so the
    notebook and these modules share the cache of the default settings.
    """
    identity = embedder.settings.identity
    vectors_by_hash: dict[str, np.ndarray] = {}
    stored_rows: list[tuple[str, str]] = []  # (item key, text hash) of each stored vector
    if path.exists():
        with np.load(path) as stored:
            stored_identity = (
                json.loads(str(stored["settings"]))
                if "settings" in stored.files
                else {  # written by the notebook, which embeds with the Gemini API only
                    "provider": "gemini",
                    "model": str(stored["model"]),
                    "dimension": int(stored["dimension"]),
                    "document_task_type": str(stored["task_type"]),
                    "document_prefix": "",
                }
            )
            if stored_identity != identity:
                raise ValueError(f"{path} holds embeddings of {stored_identity}, not {identity}; use another cache_path")
            vectors_by_hash = dict(zip(stored["text_sha256"].tolist(), stored["embeddings"]))
            stored_rows = list(zip(stored["item_keys"].tolist(), stored["text_sha256"].tolist()))

    chunk_hashes = [text_sha256(chunk["text"]) for chunk in chunks]
    texts_by_hash = dict(zip(chunk_hashes, (chunk["text"] for chunk in chunks)))
    hashes_to_embed = [text_hash for text_hash in texts_by_hash if text_hash not in vectors_by_hash]
    log.info("%d texts already embedded, %d to embed", len(texts_by_hash) - len(hashes_to_embed), len(hashes_to_embed))
    batch_size = embedder.settings.batch_size
    try:
        for start in range(0, len(hashes_to_embed), batch_size):
            batch = hashes_to_embed[start : start + batch_size]
            vectors = embedder.embed([texts_by_hash[text_hash] for text_hash in batch], "document")
            vectors_by_hash.update(zip(batch, vectors))
            log.info("Embedded %d/%d texts", start + len(batch), len(hashes_to_embed))
    finally:
        # Store whatever is embedded, also when a request fails, so a rerun resumes from there. Vectors of items no
        # longer in the item bank, or of their old text, are dropped.
        rows = [
            (chunk["item_key"], text_hash)
            for chunk, text_hash in zip(chunks, chunk_hashes)
            if text_hash in vectors_by_hash
        ]
        if rows != stored_rows:
            _save_embeddings(path, rows, vectors_by_hash, embedder.settings)
            log.info("Stored %d vectors in %s", len(rows), path)
    return {text_hash: vectors_by_hash[text_hash] for text_hash in chunk_hashes}


def _save_embeddings(path: Path, rows: list[tuple[str, str]], vectors_by_hash: dict, settings: EmbeddingSettings) -> None:
    vectors = np.array([vectors_by_hash[text_hash] for _, text_hash in rows], dtype=np.float32)
    dimension = vectors.shape[1] if len(vectors) else settings.dimension or 0
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part.npz")
    np.savez_compressed(
        partial,
        item_keys=np.array([key for key, _ in rows]),
        text_sha256=np.array([text_hash for _, text_hash in rows]),
        embeddings=vectors.reshape(-1, dimension),
        # The notebook's keys, then the full identity of the vectors.
        model=settings.model,
        dimension=dimension,
        task_type=settings.document_task_type,
        settings=json.dumps(settings.identity, sort_keys=True),
    )
    partial.replace(path)
