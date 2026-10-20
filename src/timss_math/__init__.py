"""TIMSS mathematics items: the released items with their IRT parameters, and two uses of them.

- ``item_parameter_prediction``: predicts the IRT parameters of new items with an LLM or with Jev, grounded in the
  most similar calibrated items.
- ``llm_test_taking``: LLMs answer the released items, as test takers (a benchmark) or as students (synthetic
  participants), and are scored with the IRT models and calibrated parameters, on each calibration's θ scale and on the
  TIMSS reporting scale.

Both build on the modules the package shares:

- ``items``: the released items and their calibrated parameters, from the package's ``data`` directory, which also
  holds the TIMSS reporting-scale constants.
- ``llm``: ``LLMSettings`` and the LLM client of OpenRouter, OpenAI, the Gemini API, the Claude API, or Ollama.
- ``batch``: OpenRouter batches and their manifests.
- ``running``: ``run_sync`` and ``UnusableOutputError``, for running requests to a model.
- ``config``: the data directory, API keys, and files.
- ``plot_style`` and ``command_line``: what the graphs and the commands share.

The released-item datasets are built by ``notebooks/scratch/get_data.ipynb``, a local notebook that is not in the
repository, with the helpers in ``src/released_items.py``; ``notebooks/timss_math_released_items.md`` documents how.
"""
