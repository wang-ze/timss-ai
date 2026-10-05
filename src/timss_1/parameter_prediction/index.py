"""The Qdrant collection of the released items, searched by keyword, semantic, or hybrid search.

Each point is one chunk of the item bank, with the chunk and the item's calibrated parameters (``item_parameters``,
null for the six items without them) as its payload, and two named vectors:

- ``dense``, the item's embedding, compared by cosine similarity, for semantic search.
- ``bm25``, a sparse BM25 vector of the text, for keyword search. Qdrant computes each term's inverse document
  frequency over the collection at query time (the IDF modifier), so the stored vector holds only the term-frequency
  part. Terms are lowercase words folded to the singular by Harman's S stemmer, and numbers without thousands
  separators, so "1,000 marbles" matches "1000 marble"; a term's index is a 32-bit hash of it.

``ItemIndex.build`` rebuilds the collection from the cached item embeddings. By default it is kept in memory, which
takes a second and never conflicts with the lock that a notebook kernel holds on a local store; ``IndexSettings`` can
put it in a local storage directory or on a Qdrant server instead.
"""

from __future__ import annotations

import logging
import re
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from .embeddings import Embedder, EmbeddingSettings, TextKind, load_item_embeddings, text_sha256
from .items import PARAMETER_NAMES, ItemBank, TimssItem

log = logging.getLogger(__name__)

SearchMethod = Literal["keyword", "semantic", "hybrid"]
SEARCH_METHODS = ("keyword", "semantic", "hybrid")
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
ITEM_PARAMETERS_PAYLOAD = "item_parameters"
# Matches the items without parameters: Qdrant's is_empty condition matches a null payload field.
NO_PARAMETERS = models.IsEmptyCondition(is_empty=models.PayloadField(key=ITEM_PARAMETERS_PAYLOAD))
TERM = re.compile(r"(?P<number>\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?)|(?P<word>[^\W\d_]+)")


@dataclass(frozen=True)
class IndexSettings:
    """Where the collection is kept, and the search settings."""

    collection: str = "timss_math_items"
    path: Path | None = None
    """Local Qdrant storage directory; with neither ``path`` nor ``url``, the collection is kept in memory."""
    url: str | None = None
    """URL of a Qdrant server."""
    api_key: str | None = field(default=None, repr=False)
    """API key of the Qdrant server, if it needs one."""
    hybrid_candidates: int = 50
    """Items each method contributes to the reciprocal rank fusion of hybrid search."""
    bm25_k1: float = 1.2
    """BM25 term-frequency saturation."""
    bm25_b: float = 0.75
    """BM25 document-length normalization."""

    def __post_init__(self):
        if self.path is not None and self.url is not None:
            raise ValueError("Give a local storage path or a server URL, not both")

    def describe(self) -> dict:
        return {name: value for name, value in asdict(self).items() if name != "api_key"}


def s_stem(word: str) -> str:
    """Harman's S stemmer, which folds English plurals to the singular: cards -> card, pennies -> penny.

    Words of up to three letters, such as "is", "bus", and the "s" of "Maria's", are kept as they are.
    """
    if len(word) <= 3:
        return word
    if word.endswith("ies") and not word.endswith(("eies", "aies")):
        return word[:-3] + "y"
    if word.endswith("es") and not word.endswith(("aes", "ees", "oes")):
        return word[:-1]
    if word.endswith("s") and not word.endswith(("us", "ss")):
        return word[:-1]
    return word


def terms(text: str) -> list[str]:
    """Keyword terms of a text: lowercase words folded to the singular, and numbers without thousands separators."""
    return [
        match["number"].replace(",", "") if match["number"] else s_stem(match["word"])
        for match in TERM.finditer(text.lower())
    ]


def term_index(term: str) -> int:
    """Index of a term in sparse vectors: a stable 32-bit hash, so no vocabulary has to be stored."""
    return zlib.crc32(term.encode("utf-8"))


def bm25_query_vector(text: str) -> models.SparseVector:
    """Weight 1 for each distinct term of a query, so an item scores the sum of its BM25 weights of those terms."""
    indices = sorted(set(map(term_index, terms(text))))
    return models.SparseVector(indices=indices, values=[1.0] * len(indices))


def point_id(key: str) -> str:
    """Qdrant point ID of the item with this item_key, the same on every rebuild."""
    return str(uuid5(NAMESPACE_URL, key))


class ItemIndex:
    """The item bank in a Qdrant collection, with its embedder for queries and new items."""

    def __init__(
        self,
        bank: ItemBank,
        embedder: Embedder,
        vectors_by_hash: dict[str, np.ndarray],
        client: QdrantClient,
        settings: IndexSettings,
    ):
        self.bank = bank
        self.embedder = embedder
        self.vectors_by_hash = vectors_by_hash
        self.client = client
        self.settings = settings
        self.payload_fields = set(bank.chunks[0]) | {ITEM_PARAMETERS_PAYLOAD}
        self.bm25_average_length = float(np.mean([len(terms(chunk["text"])) for chunk in bank.chunks]))

    @classmethod
    def build(
        cls,
        embedding: EmbeddingSettings = EmbeddingSettings(),
        settings: IndexSettings = IndexSettings(),
        bank: ItemBank | None = None,
        data_dir: str | Path | None = None,
    ) -> ItemIndex:
        """Embed the items (only those not cached yet) and rebuild the collection with their parameters.

        ``bank`` defaults to ``ItemBank.load(data_dir)``, and the embedding cache to
        ``embedding.default_cache_path(data_dir)``.
        """
        bank = bank or ItemBank.load(data_dir)
        embedder = Embedder(embedding)
        cache_path = embedding.cache_path or embedding.default_cache_path(data_dir)
        vectors_by_hash = load_item_embeddings(bank.chunks, embedder, Path(cache_path))
        if settings.url is not None:
            client = QdrantClient(url=settings.url, api_key=settings.api_key)
        elif settings.path is not None:
            client = QdrantClient(path=str(settings.path))
        else:
            client = QdrantClient(location=":memory:")
        index = cls(bank, embedder, vectors_by_hash, client, settings)
        try:
            index._create_collection()
        except BaseException:
            client.close()
            raise
        return index

    def _create_collection(self) -> None:
        collection, chunks = self.settings.collection, self.bank.chunks
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        dimension = len(next(iter(self.vectors_by_hash.values())))
        self.client.create_collection(
            collection,
            vectors_config={DENSE_VECTOR: models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self.client.upload_points(
            collection,
            [
                models.PointStruct(
                    id=point_id(chunk["item_key"]),
                    vector={
                        DENSE_VECTOR: self.vectors_by_hash[text_sha256(chunk["text"])].tolist(),
                        SPARSE_VECTOR: self.bm25_document_vector(chunk["text"]),
                    },
                    payload=chunk | {ITEM_PARAMETERS_PAYLOAD: self.bank.parameter_payload(chunk["item_key"])},
                )
                for chunk in chunks
            ],
            wait=True,
        )
        point_count = self.client.count(collection, exact=True).count
        with_parameters = self.client.count(
            collection, count_filter=models.Filter(must_not=[NO_PARAMETERS]), exact=True
        ).count
        if (point_count, with_parameters) != (len(chunks), len(self.bank.calibrated)):
            raise RuntimeError(
                f"{collection} holds {point_count} points, {with_parameters} with parameters, instead of "
                f"{len(chunks)} and {len(self.bank.calibrated)}"
            )
        log.info("Stored %d items, %d with parameters, in collection '%s'", point_count, with_parameters, collection)

    def close(self) -> None:
        """Close the Qdrant client, which releases the lock on a local storage directory."""
        self.client.close()

    def __enter__(self) -> ItemIndex:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def bm25_document_vector(self, text: str) -> models.SparseVector:
        """BM25 term-frequency weights of a chunk's terms. Qdrant multiplies them by each term's IDF at query time."""
        k1, b = self.settings.bm25_k1, self.settings.bm25_b
        counts = Counter(map(term_index, terms(text)))
        length_norm = k1 * (1 - b + b * counts.total() / self.bm25_average_length)
        return models.SparseVector(
            indices=list(counts),
            values=[count * (k1 + 1) / (count + length_norm) for count in counts.values()],
        )

    def metadata_filter(self, **conditions) -> models.Filter:
        """Qdrant filter that keeps the items matching every condition: field=value, or field=[values] for any of them.

        None leaves a field unfiltered. For example, metadata_filter(assessment=["TIMSS 2007", "TIMSS 2011"], grade=4).
        Fields of the item parameters are written as in Qdrant, such as ``**{"item_parameters.irt_model": "GPCM"}``.
        """
        unknown_fields = {name.split(".")[0] for name in conditions} - self.payload_fields
        if unknown_fields:
            raise ValueError(
                f"Unknown fields {sorted(unknown_fields)}; the payload fields are {sorted(self.payload_fields)}"
            )
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=name,
                    match=models.MatchAny(any=value) if isinstance(value, list) else models.MatchValue(value=value),
                )
                for name, value in conditions.items()
                if value is not None
            ]
        )

    def search(
        self,
        text: str,
        method: SearchMethod = "hybrid",
        limit: int = 10,
        query_filter: models.Filter | None = None,
        embed_as: TextKind = "query",
        vector: Iterable[float] | None = None,
    ) -> list[models.ScoredPoint]:
        """Items that best match ``text``, best first, by "keyword" (BM25), "semantic" (embeddings), or "hybrid" (both,
        fused by reciprocal rank fusion; its scores are fusion scores, not similarities).

        ``text`` is embedded as a search query; to find the items most similar to an item, pass its text with
        ``embed_as="document"``, as the stored items were embedded. ``vector`` is the embedding of ``text`` if it is
        already known, so the semantic methods do not request it again.
        """
        if method not in SEARCH_METHODS:
            raise ValueError(f"method must be one of {SEARCH_METHODS}, not {method!r}")
        queries = {}
        if method in ("keyword", "hybrid"):
            queries[SPARSE_VECTOR] = bm25_query_vector(text)
        if method in ("semantic", "hybrid"):
            query_vector = self.embedder.embed_one(text, embed_as) if vector is None else vector
            queries[DENSE_VECTOR] = [float(value) for value in query_vector]
        collection = self.settings.collection
        if method == "hybrid":
            return self.client.query_points(
                collection,
                prefetch=[
                    models.Prefetch(query=query, using=using, filter=query_filter, limit=self.settings.hybrid_candidates)
                    for using, query in queries.items()
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            ).points
        [(using, query)] = queries.items()
        return self.client.query_points(collection, query=query, using=using, query_filter=query_filter, limit=limit).points

    def item_vector(self, item: TimssItem) -> np.ndarray:
        """Embedding of an item's text: the stored vector if its text is in the collection, else a new embedding."""
        stored = self.vectors_by_hash.get(text_sha256(item.text))
        return np.asarray(self.embedder.embed_one(item.text, "document") if stored is None else stored)

    def find_similar_items(
        self,
        item: TimssItem,
        limit: int = 10,
        irt_model: str | None = None,
        scaled_points: int | None = None,
        exclude_keys: Iterable[str] = (),
        method: SearchMethod = "hybrid",
    ) -> list[dict]:
        """The calibrated items of the collection most similar to ``item``, most similar first, all of its grade.

        ``irt_model`` and ``scaled_points`` keep only items scored with that model and worth that many points in the
        scaling. ``exclude_keys`` leaves items out, such as the item itself and its other parts. Returns one dict per
        item with its rank, item_key, cosine similarity with ``item``, and payload.
        """
        must = [models.FieldCondition(key="grade", match=models.MatchValue(value=item.grade))]
        if irt_model is not None:
            must.append(
                models.FieldCondition(key=f"{ITEM_PARAMETERS_PAYLOAD}.irt_model", match=models.MatchValue(value=irt_model))
            )
        if scaled_points is not None:
            must.append(
                models.FieldCondition(
                    key=f"{ITEM_PARAMETERS_PAYLOAD}.scaled_points", match=models.MatchValue(value=scaled_points)
                )
            )
        must_not = [NO_PARAMETERS]
        exclude_keys = list(exclude_keys)
        if exclude_keys:
            must_not.append(models.HasIdCondition(has_id=[point_id(key) for key in exclude_keys]))
        vector = self.item_vector(item)
        points = self.search(
            item.text, method, limit, models.Filter(must=must, must_not=must_not), embed_as="document", vector=vector
        )
        return [
            {
                "rank": rank,
                "item_key": point.payload["item_key"],
                "cosine_similarity": float(vector @ self.vectors_by_hash[text_sha256(point.payload["text"])]),
                "payload": point.payload,
            }
            for rank, point in enumerate(points, 1)
        ]


def results_table(points: list[models.ScoredPoint]) -> pd.DataFrame:
    """One row per search result: its score, item, and the start of its text."""
    return pd.DataFrame(
        [
            {
                "score": point.score,
                **{name: point.payload[name] for name in ("assessment", "grade", "item_id", "content_domain")},
                "text": " ".join(point.payload["text"].split())[:100],
            }
            for point in points
        ]
    )


def similar_items_table(similar_items: list[dict]) -> pd.DataFrame:
    """One row per item from find_similar_items: rank, similarity, classification, and calibrated parameters."""
    return pd.DataFrame(
        [
            {
                "rank": entry["rank"],
                "cosine_similarity": entry["cosine_similarity"],
                **{
                    name: entry["payload"][name]
                    for name in ("assessment", "item_id", "content_domain", "cognitive_domain", "item_type")
                },
                "pct_correct_intl_avg": entry["payload"]["pct_correct_intl_avg"],
                **{
                    name: entry["payload"][ITEM_PARAMETERS_PAYLOAD][name]
                    for name in ("irt_model", "scaled_points", *PARAMETER_NAMES)
                },
            }
            for entry in similar_items
        ]
    )
