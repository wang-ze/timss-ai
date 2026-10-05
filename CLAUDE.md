# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project on TIMSS mathematics items.
It builds item-level datasets from every TIMSS released mathematics set (1995 to 2011, grades 4 and 8, and TIMSS Advanced), attaches the official IRT item parameters, and predicts the IRT parameters of new items with an LLM grounded in the most similar calibrated items.
`README.md` summarizes each part, and `notebooks/timss_math_released_items.md` is the detailed record of sources, booklet quirks, parameter sources, and costs.
Update both when the pipeline or its outputs change.

## Commands

The project uses uv with Python 3.13 (`.venv/`).

```sh
uv sync                                                          # install dependencies
uv run timss2011-convert --zip T11_G4_SPSSData_pt3.zip --country usa  # TIMSS 2011 SPSS -> Parquet in notebooks/data/
uv run timss-predict-loo --help                                  # leave-one-out predictions, errors, and graphs
uv run timss-plot-errors <predictions.xlsx> [...]                # graphs of a run's errors, or a comparison of runs
```

There is no test suite and no linter configuration.
Correctness is enforced inside the notebooks: every parse and transcription cell checks its results against a second source and raises on any disagreement, so rerunning a section is the test.

The notebooks are normally run in VS Code with the `.venv` kernel and **`notebooks/` as the working directory**: every path is `Path.cwd() / "data"`, and paths in the docs are relative to `notebooks/`.
To run cells headlessly, nbconvert, nbclient, and papermill are not installed; either exec the cell sources in one namespace with cwd `notebooks/` and the project `.env` loaded, or start `.venv/bin/python -m ipykernel_launcher` with cwd `notebooks/` and drive it with `jupyter_client`.
Skip cell 0 of `get_data.ipynb`, which runs `uv add`.

## Architecture

### `src/timss_1/` (installed package, imported by the notebooks)

- `timss2011.py`: downloads the TIMSS 2011 international database and converts one country's `.sav` files to Parquet.
  Each column's SPSS label, value labels, and missing-value ranges are stored in the Parquet field metadata (`load`, `variable_info`); missing codes are kept as values unless `missing_to_nan=True`.
- `released_items.py` (imported as `ri`): what the released-set sections of `get_data.ipynb` share.
  PDF word and line helpers, Acrobat watermark removal, image parts for the model, `RECORD_FIELDS` (the JSONL schema, in order), `write_jsonl`, and `Transcriber`.
  `Transcriber` sends batches of items to OpenRouter with structured output, never retries, keeps a transcription only if the section's `problems` check passes, caches each kept one as `<cache_dir>/<item_id>.json`, re-checks every cached transcription on every run (so tightened checks re-request old results), and raises if any item still lacks one.
- `parameter_prediction/`: `predict_item_parameters.ipynb` as modules, with the notebook's settings as defaults (see README).
  `items` (`ItemBank`, `TimssItem`), `embeddings` (`EmbeddingSettings` for Gemini, OpenAI, OpenRouter, Ollama, or FastEmbed), `index` (`ItemIndex`, keyword, semantic, or hybrid search), `predictor` (`ParameterPredictor`, `LLMSettings` for OpenRouter, OpenAI, Gemini, or Ollama), `evaluation` (`LeaveOneOut`, `PredictionResults`, `timss-predict-loo`), and `plots` (matplotlib, `timss-plot-errors`).
  The prompts and retrieval must stay identical to the notebook's: with the default settings, `LeaveOneOut(ParameterPredictor(ItemIndex.build())).status()` must report all 722 notebook predictions as saved, which checks every prompt byte for byte.
  Change the notebook and the modules together.

### `notebooks/get_data.ipynb` (dataset building)

One section per released set, each with three steps: a deterministic parse of the PDF or official files (IDs, classifications, keys, scoring categories, statistics), a transcription of stem, options, scoring guide, and figure description from rendered images with `google/gemini-3.8-flash` through `ri.Transcriber`, and a cell that validates the transcriptions against the parse and writes `data/timss<YY>_g<G>_math_released_items.jsonl`.
All sections share one kernel namespace, so each uses its own name prefix (`T07_`, `T03G8_`, `TA08_`, ...) to avoid clobbering the 2011 grade 4 globals.
The last sections combine the sets into `data/timss_math_released_items.jsonl` (731 items) and collect IRT parameters from the official workbooks, technical-report exhibits, and derived ID crosswalks into `data/timss_math_released_item_parameters.xlsx`.
The early 2011 grade 4 sections on embeddings, the `data/qdrant/` store, and LLM prediction (including the TIMSS 2007 prediction comparison) are the first prototype, superseded by `predict_item_parameters.ipynb`.

### `notebooks/predict_item_parameters.ipynb` (retrieval and prediction)

1. One chunk per item (728: three verbatim duplicate texts are left out via `DUPLICATE_ITEMS`), embedded with `gemini-embedding-001` through the Gemini API and BM25-indexed in the local Qdrant store `data/qdrant_timss_math_items/` (named vectors `dense` and `bm25`).
   Embeddings are cached by text SHA-256 in `data/timss_math_item_embeddings.npz`, so a rebuild embeds only new or changed texts.
2. Calibrated parameters go into each point's `item_parameters` payload.
3. `find_similar_items` retrieves same-grade calibrated items (hybrid search by default) and `predict_item_parameters` asks `google/gemini-3.8-flash` on OpenRouter for 3PL, 2PL, or GPCM parameters.
4. A leave-one-out run predicts all 722 calibrated items and exports `data/timss_math_predicted_item_parameters.xlsx`.
   Predictions are cached per item under `data/cache/parameter_predictions/timss_math/` and reused only if the model, reasoning effort, and both prompts are unchanged.

### `notebooks/predict_item_parameters_FastEmbed.ipynb` (the same with FastEmbed and Ollama)

Imports `timss_1.parameter_prediction` instead of defining its own code, so it changes with the modules.
Embeds with FastEmbed's default `BAAI/bge-small-en-v1.5` on this computer (one text at a time, because batched vectors depend on the batch; the vectors are cached in `data/cache/item_embeddings/`), stores the collection in `data/qdrant_timss_math_items_fastembed/`, and predicts with `llama3.2` in local Ollama (no reasoning effort, `max_tokens` 4096).
`LOO_MAX_REQUESTS` (10) caps the predictions requested per run, because a local model needs hours for all 722.

## Invariants

- `(assessment, grade, item_id)` identifies an item, because short IDs such as K1 and L10 repeat across sets; the notebooks use `item_key` strings like `TIMSS 1995/grade 4/Mathematics/K1`.
- The embedded and keyword-indexed text is the stem and options only.
  Never add `correct_option`, `scoring_guide`, `scoring_notes`, figure descriptions, or accessibility text to it.
- Leave-one-out predictions must exclude the item's other parts (same base ID) and the other items of its task (`SHARED_STIMULUS_TASKS`).
- Model output is accepted only after checks against deterministically parsed fields.
  Never alter page pixels that may overlap text before sending images to the model (whitening a watermark once made the model invent wrong options); `ri.remove_watermarks` removes the watermark from the PDF instead.
- IRT models follow item type and points: 3PL for multiple choice, 2PL for 1-point constructed response, GPCM for 2 or 3 points, with GPCM steps summing to 0.
  Parameters from different calibrations (1995-2003, 1999-2003, 2003-2007, 2007-2011, TIMSS Advanced 1995-2008) are on different scales.

## LLM costs, quotas, and caches

- `.env` holds `OPENROUTER_API_KEY` (transcription and prediction) and `GEMINI_API_KEY` (embeddings, including query embeddings for semantic and hybrid search).
  FastEmbed and Ollama run locally and need neither; `llama3.2` sometimes repeats itself until `max_tokens`, which the run reports as unusable output and requests again on the next run.
- OpenRouter calls cost real money: a full leave-one-out run is about $9, and any prompt change invalidates every cached prediction.
  Bumping a section's `PROMPT_VERSION` moves its transcription cache to a new directory and re-transcribes the whole set.
  Ask before spending beyond a test request (`*_MAX_REQUESTS = 1`).
- The Gemini API free tier allows about 20 `gemini-3.8-flash` requests per day and 100 embedded texts per minute, and often answers large requests with 503; do not retry in a loop.
- Caches under `notebooks/data/cache/` are committed, so reruns make no API calls.
  The modules save predictions as `parameter_predictions/timss_math/<provider>/<model>/<item>_<hash>.json`, the hash covering the LLM, reasoning effort, and both prompts, and reuse the notebook's `<item>.json` files when they match.
  Their item embeddings for non-default settings go to `cache/item_embeddings/`.

## Local Qdrant

Local Qdrant storage takes a process lock, and the user's Jupyter kernel usually holds it.
Test Qdrant code against a copy of the store in a scratch directory rather than opening the live store from another process.
`ItemIndex.build` keeps its collection in memory unless `IndexSettings` names a path or URL, so it never takes the lock.

## Git

`origin` (`wang-ze/timss`) is private and `upstream` (`wang-ze/timss-ai`) is public; only finished work on `main` goes to `upstream` (see `two-repository-split.md`).
Downloads go to `notebooks/data/raw/`, which is gitignored; GitHub rejects files over 100 MB.
