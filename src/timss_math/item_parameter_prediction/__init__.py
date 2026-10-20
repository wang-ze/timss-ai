"""Predict the IRT parameters of TIMSS mathematics items from the most similar calibrated released items.

The modules of ``predict_item_parameters.ipynb``, with its settings as defaults and every setting a parameter, and a
second predictor, Jev. ``predict_item_parameters_FastEmbed.ipynb`` uses them with FastEmbed embeddings and an Ollama
LLM, and ``predict_item_parameters_compare_jev.ipynb`` compares Jev with two LLMs.

They build on the modules that the package shares: ``timss_math.items`` (the released items, their chunks and
calibrated parameters, ``ItemBank``, and ``TimssItem``), ``llm`` (``LLMSettings`` and the LLM client of OpenRouter,
OpenAI, the Gemini API, the Claude API, or Ollama), ``batch`` (OpenRouter batches, ``OpenRouterBatches``), and
``config``. Their old names here, such as ``timss_math.item_parameter_prediction.items``, still work.

- ``embeddings``: embedding models of the Gemini API, OpenAI, OpenRouter, Ollama, or FastEmbed (``EmbeddingSettings``),
  and the cache of item embeddings.
- ``index``: the Qdrant collection, searched by keyword, semantic, or hybrid search (``ItemIndex``, ``IndexSettings``).
- ``predictor``: what every predictor shares (``Predictor``): the reference items, found by a search method, and the
  methods a leave-one-out run uses.
- ``llm``: the prompts of the LLM predictor (``LLMPredictor``).
- ``jev``: Jev's state and value-scale questions, and the point predictions from its probabilities (``JevPredictor``,
  ``JevSettings``, ``ValueScale``).
- ``evaluation``: leave-one-out predictions of the calibrated items and their errors (``LeaveOneOut``,
  ``PredictionResults``), single or as OpenRouter batches at about half the price, and the ``timss-predict-loo``
  command.
- ``factorial``: leave-one-out runs under every combination of levels of a few settings, such as the embedding model
  and the LLM, compared on the same items with main effects and interactions (``FactorialResults``), as in
  ``predict_item_parameters_compare.ipynb``.
- ``comparison``: any leave-one-out runs compared on the same items, with paired tests and agreement on value scales
  (``RunComparison``), as in ``predict_item_parameters_compare_jev.ipynb``.
- ``plots``: graphs of the prediction errors, and the ``timss-plot-errors`` command.

Example::

    from timss_math.item_parameter_prediction import EmbeddingSettings, ItemIndex, JevPredictor, LLMPredictor, LLMSettings

    index = ItemIndex.build()  # the notebook's embeddings, in an in-memory Qdrant collection
    # or FastEmbed's default model, on this computer:
    # index = ItemIndex.build(EmbeddingSettings(provider="fastembed", model="BAAI/bge-small-en-v1.5", dimension=None))
    index.search("fraction of a shape that is shaded", method="keyword")
    predictor = LLMPredictor(index, LLMSettings(provider="ollama", model="gpt-oss:20b"), search_method="semantic")
    prediction = predictor.predict(TimssItem(...))
    jev = JevPredictor(index, with_references=False)  # Jev, from the item alone
    jev.parameters(jev.predict(TimssItem(...)))
"""

from ..batch import OpenRouterBatches
from ..items import (
    PARAMETER_NAMES,
    ItemBank,
    TimssItem,
    item_key,
    item_text,
    model_parameters,
    predicted_parameters,
)
from ..llm import LLMSettings, NoPredictionError
from ..running import UnusableOutputError
from .comparison import RunComparison
from .embeddings import Embedder, EmbeddingSettings
from .evaluation import LeaveOneOut, PredictionResults, error_summary
from .factorial import FactorialResults
from .index import IndexSettings, ItemIndex, results_table, similar_items_table
from .jev import (
    DEFAULT_VALUE_SCALES,
    JevAnswerError,
    JevPredictor,
    JevSettings,
    ValueScale,
)
from .llm import LLMPredictor
from .predictor import Predictor

__all__ = [
    "DEFAULT_VALUE_SCALES",
    "PARAMETER_NAMES",
    "Embedder",
    "EmbeddingSettings",
    "FactorialResults",
    "IndexSettings",
    "ItemBank",
    "ItemIndex",
    "JevAnswerError",
    "JevPredictor",
    "JevSettings",
    "LLMPredictor",
    "LLMSettings",
    "LeaveOneOut",
    "NoPredictionError",
    "OpenRouterBatches",
    "PredictionResults",
    "Predictor",
    "RunComparison",
    "TimssItem",
    "UnusableOutputError",
    "ValueScale",
    "error_summary",
    "item_key",
    "item_text",
    "model_parameters",
    "predicted_parameters",
    "results_table",
    "similar_items_table",
]
