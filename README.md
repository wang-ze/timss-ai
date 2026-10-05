# timss-1

## TIMSS 2011 data

`src/timss_1/timss2011.py` downloads the TIMSS 2011 international database (SPSS version) and converts one country's files to Parquet.

```sh
uv run timss2011-convert --zip T11_G4_SPSSData_pt3.zip --country usa
```

This writes `notebooks/data/timss11_g4_usa/`, with a `.parquet` file and a `_codebook.csv` for each data file:

| File | Contents | USA grade 4 |
| --- | --- | --- |
| `acgusam5` | School background | 369 schools |
| `asausam5` | Student achievement | 12,569 students |
| `asgusam5` | Student background | 12,569 students |
| `asrusam5` | Within-country scoring reliability | 3,076 students |
| `astusam5` | Student-teacher linkage | 15,061 links |
| `atgusam5` | Teacher background | 767 teachers |

Missing-value codes such as 96 (NOT REACHED) and 99 (OMITTED) are kept as values, as in SPSS.
Pass `missing_to_nan=True` to turn them into NaN.

```python
from timss_1.timss2011 import load, variable_info

path = "data/timss11_g4_usa/asausam5.parquet"  # relative to notebooks/
students = load(path)                      # codes kept
students = load(path, missing_to_nan=True) # codes -> NaN
info = variable_info(path)                 # label, format, value labels, missing values
info["M031346A"]["value_labels"]           # {10.0: 'CORRECT RESPONSE', 79.0: 'INCORRECT RESPONSE', ...}
```

## TIMSS released mathematics items

`notebooks/get_data.ipynb` builds item-level JSONL files from every TIMSS released mathematics set (1995 to 2011, grades 4, 8, and TIMSS Advanced), combines them, and finds the official IRT item parameters of every item.
`notebooks/timss_math_released_items.md` documents the sources, extraction, checks, and parameter sources.

## Item parameter prediction

`notebooks/predict_item_parameters.ipynb` predicts the IRT parameters of new mathematics items from the most similar released items, whose parameters are known.
The notebook documents each step.

### Retrieval

The notebook indexes the released mathematics items for retrieval, one chunk per item.
Three items whose stem and options repeat another item's word for word are left out (K7 of TIMSS 1995, D11 of TIMSS 1999, and L15B of TIMSS Advanced 1995), so the collection holds 728 items.
The stem and options of each item, never its answer or scoring guide, are embedded with `gemini-embedding-001` (768 dimensions) and BM25-indexed in a local Qdrant collection, `timss_math_items` in `notebooks/data/qdrant_timss_math_items/`.
Each item's payload has its metadata for filters and its calibrated IRT parameters from `notebooks/data/timss_math_released_item_parameters.xlsx` (722 items have parameters).
The vectors are also saved in `notebooks/data/timss_math_item_embeddings.npz`, so a rebuild embeds only new or changed items.
`search` finds items by keyword (BM25), semantic (embedding), or hybrid (reciprocal rank fusion of both) search, optionally filtered by metadata such as assessment and grade.

### Prediction

`find_similar_items` returns the calibrated items of the same grade that are most similar to a new item.
`predict_item_parameters` gives them, with their full text, percent correct, and parameters, to an LLM (`google/gemini-3.8-flash` through OpenRouter, so `OPENROUTER_API_KEY` must be in `.env`).
The LLM, prompted as an expert in testing the item's grade, content domain, and cognitive domain and as an IRT measurement expert, returns its reasoning and the item's parameters: 3PL for multiple-choice items, 2PL for 1-point constructed-response items, and GPCM for partial-credit constructed-response items.

To check the method, the notebook predicts each of the 722 calibrated items as if it were new, with the item's other parts (and the other items of its task) left out of its context.
It exports one row per item, with the reference items, the LLM's reasoning, the predicted and calibrated parameters, and their errors, plus error summaries by grade and assessment, to `notebooks/data/timss_math_predicted_item_parameters.xlsx`.
Each prediction is saved in `notebooks/data/cache/parameter_predictions/timss_math/`, so a rerun requests only missing or changed predictions.

### FastEmbed and Ollama

`notebooks/predict_item_parameters_FastEmbed.ipynb` does the same without any API, by importing the Python modules below.
FastEmbed, Qdrant's embedding library, embeds the items on this computer with its default model, `BAAI/bge-small-en-v1.5` (384 dimensions), and a local Qdrant collection stores them, `timss_math_items` in `notebooks/data/qdrant_timss_math_items_fastembed/`.
The LLM is `llama3.2` in a local Ollama server; `LLM_PROVIDER` and `PARAMETER_LLM_MODEL` select another Ollama model, or a model of OpenRouter, OpenAI, or the Gemini API.
Keyword search is the same as in `predict_item_parameters.ipynb`, so only semantic and hybrid search, and the LLM, differ.

FastEmbed embeds each text on its own, because in a batch a text's vector depends slightly on the other texts of the batch.
The model reads at most 512 tokens, so the three parts of the TIMSS 2007 grade 8 class trip task (M032753A to M032753C), about 900 tokens each, are cut there.
The item embeddings are cached in `notebooks/data/cache/item_embeddings/`, the predictions in `notebooks/data/cache/parameter_predictions/timss_math/ollama/llama3.2/`, and the predictions and their errors are exported to `notebooks/data/timss_math_predicted_item_parameters_fastembed.xlsx`.
`llama3.2` predicts an item in about 10 seconds here, so the notebook requests at most `LOO_MAX_REQUESTS` (10) predictions in a run; `None` requests all 722, which take hours.

### Python modules

`src/timss_1/parameter_prediction/` holds the notebook's code as modules.
Every setting is a parameter, and its default is the notebook's value.

| Module | Contents |
| --- | --- |
| `items` | `ItemBank` (the released items, their chunks, calibrated parameters, and parts) and `TimssItem`, a new item. |
| `embeddings` | `EmbeddingSettings`: provider (`gemini`, `openai`, `openrouter`, `ollama`, or `fastembed`), model, dimension, Gemini task types or query and document prefixes, and quota. |
| `index` | `ItemIndex`, the Qdrant collection, with `search(text, method)` by `keyword`, `semantic`, or `hybrid` search, and `IndexSettings` (in memory by default, or a local path or server URL; hybrid candidates; BM25 `k1` and `b`). |
| `predictor` | `ParameterPredictor` (search method, `n_similar`, `n_same_model`) and `LLMSettings` (provider `openrouter`, `openai`, `gemini`, or `ollama`; model; reasoning effort; output cap). |
| `evaluation` | `LeaveOneOut` and `PredictionResults`: predict every calibrated item as if it were new, compute the errors and baselines, summarize them, and export them to Excel. |
| `plots` | Graphs of the errors: predicted against calibrated values, error distributions, mean absolute error against the baselines, bias by assessment, and a comparison of runs. |

```python
from timss_1.parameter_prediction import (
    EmbeddingSettings, ItemIndex, LeaveOneOut, LLMSettings, ParameterPredictor, TimssItem, plots
)

index = ItemIndex.build(
    EmbeddingSettings(provider="ollama", model="nomic-embed-text", document_prefix="search_document: ",
                      query_prefix="search_query: ")
)
index.search("fraction of a shape that is shaded", method="keyword")
predictor = ParameterPredictor(index, LLMSettings(provider="ollama", model="gpt-oss:20b"), search_method="semantic")
prediction = predictor.predict(TimssItem(...))

loo = LeaveOneOut(predictor)
loo.request(max_requests=1)  # LLM requests cost money; None requests every missing prediction
results = loo.results()
results.summary("grade")
plots.mae_vs_baselines(results.errors)
```

The same from the command line, which writes `predictions.xlsx` (with a `settings` sheet) and `plots/` to `notebooks/data/parameter_prediction_runs/<settings>/`:

```sh
uv run timss-predict-loo                                   # the notebook's settings; reuses its predictions
uv run timss-predict-loo --search-method semantic --llm-provider ollama --llm-model gpt-oss:20b --max-requests all
uv run timss-predict-loo --embedding-provider fastembed --llm-provider ollama --llm-model llama3.2 \
  --reasoning-effort omit --max-tokens 4096 --workers 1 --max-requests 10  # the FastEmbed notebook's settings
uv run timss-plot-errors notebooks/data/timss_math_predicted_item_parameters.xlsx
uv run timss-plot-errors run_a/predictions.xlsx run_b/predictions.xlsx --names hybrid semantic  # compare runs
```

`timss-predict-loo` sends no LLM requests unless `--max-requests` is given (`all` for every missing prediction).
Run it with `--help` for every setting.

The index is rebuilt in memory by default, which takes a second and never conflicts with the lock that a notebook kernel holds on a local Qdrant store.
Item embeddings are cached per embedding setting: the notebook's settings use its `notebooks/data/timss_math_item_embeddings.npz`, and other settings a file in `notebooks/data/cache/item_embeddings/`.
`--embedding-provider fastembed` defaults to FastEmbed's `BAAI/bge-small-en-v1.5` at its own size; FastEmbed downloads the model (67 MB) on first use into `FASTEMBED_CACHE_PATH`, else into the system temp directory.
Predictions are saved in `notebooks/data/cache/parameter_predictions/timss_math/<provider>/<model>/`, one file per item, named by a hash of the LLM, reasoning effort, and prompts, so runs with different settings never overwrite each other's predictions.
Predictions that the notebook saved are reused when their prompts match, so the default settings reproduce the notebook's 722 predictions and error summaries without any request.

Ollama is called through its native API: the reasoning effort becomes `think` (pass `reasoning_effort=None` for models that cannot think, such as `llama3.2`), and `ollama_num_ctx` (32,768 tokens by default) must hold the prompts, about 6,000 tokens, and the output.
