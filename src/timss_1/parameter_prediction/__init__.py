"""Predict the IRT parameters of TIMSS mathematics items from the most similar calibrated released items.

The modules of ``predict_item_parameters.ipynb``, with its settings as defaults and every setting a parameter.
``predict_item_parameters_FastEmbed.ipynb`` uses them with FastEmbed embeddings and an Ollama LLM.

- ``items``: the released items, their chunks and calibrated parameters (``ItemBank``), and ``TimssItem``.
- ``embeddings``: embedding models of the Gemini API, OpenAI, OpenRouter, Ollama, or FastEmbed (``EmbeddingSettings``),
  and the cache of item embeddings.
- ``index``: the Qdrant collection, searched by keyword, semantic, or hybrid search (``ItemIndex``, ``IndexSettings``).
- ``predictor``: the prompts and the LLM of OpenRouter, OpenAI, the Gemini API, or Ollama (``ParameterPredictor``,
  ``LLMSettings``).
- ``evaluation``: leave-one-out predictions of the calibrated items and their errors (``LeaveOneOut``,
  ``PredictionResults``), and the ``timss-predict-loo`` command.
- ``plots``: graphs of the prediction errors, and the ``timss-plot-errors`` command.

Example::

    from timss_1.parameter_prediction import EmbeddingSettings, ItemIndex, LLMSettings, ParameterPredictor, TimssItem

    index = ItemIndex.build()  # the notebook's embeddings, in an in-memory Qdrant collection
    # or FastEmbed's default model, on this computer:
    # index = ItemIndex.build(EmbeddingSettings(provider="fastembed", model="BAAI/bge-small-en-v1.5", dimension=None))
    index.search("fraction of a shape that is shaded", method="keyword")
    predictor = ParameterPredictor(index, LLMSettings(provider="ollama", model="gpt-oss:20b"), search_method="semantic")
    prediction = predictor.predict(TimssItem(...))
"""

from .embeddings import Embedder, EmbeddingSettings
from .evaluation import LeaveOneOut, PredictionResults, error_summary
from .index import IndexSettings, ItemIndex, results_table, similar_items_table
from .items import PARAMETER_NAMES, ItemBank, TimssItem, item_key, item_text, model_parameters, predicted_parameters
from .predictor import LLMSettings, NoPredictionError, ParameterPredictor

__all__ = [
    "PARAMETER_NAMES",
    "Embedder",
    "EmbeddingSettings",
    "IndexSettings",
    "ItemBank",
    "ItemIndex",
    "LLMSettings",
    "LeaveOneOut",
    "NoPredictionError",
    "ParameterPredictor",
    "PredictionResults",
    "TimssItem",
    "error_summary",
    "item_key",
    "item_text",
    "model_parameters",
    "predicted_parameters",
    "results_table",
    "similar_items_table",
]
