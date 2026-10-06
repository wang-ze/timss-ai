# timss-math

Item-level datasets of the 731 released TIMSS mathematics items (1995 to 2011, grades 4 and 8, and TIMSS Advanced), with the official IRT item parameters of 724 of them, and a Python package that predicts the IRT parameters of new items with an LLM grounded in the most similar calibrated items.

- [Prerequisites](#prerequisites) and [installation](#installation) cover what you need before using the package.
- [The datasets](#the-datasets) can be used without the package.
- [Item parameter prediction](#item-parameter-prediction) explains the method, and [using the package](#using-the-package) shows how to run it.
- [Results](#results-embedding-models-and-llms-compared) compares two embedding models and two LLMs in a 2 × 2 experiment.

## Prerequisites

- **macOS, Linux, or Windows.**
- **Git**, to clone the repository.
- **[uv](https://docs.astral.sh/uv/)**, which installs Python 3.13 and every dependency into the project's own environment:

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS and Linux; or, with Homebrew: brew install uv
  uv --version
  ```

  On Windows, in PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
- **API keys** for the default embedding model and LLM, in a file named `.env` at the root of the repository:

  ```sh
  OPENROUTER_API_KEY=sk-or-...
  GEMINI_API_KEY=...
  ```

  | Variable | Used for | Where to get it | Cost |
  | --- | --- | --- | --- |
  | `OPENROUTER_API_KEY` | Predictions with the default LLM, `google/gemini-3.8-flash`, through [OpenRouter](https://openrouter.ai), single or as batches | A key from your OpenRouter account, which needs credit | About $0.012 per item, or $9 for all 722 calibrated items; half that through the Batch API |
  | `GEMINI_API_KEY` | Embeddings with the default embedding model, `gemini-embedding-001`, through the Gemini API | [Google AI Studio](https://aistudio.google.com/apikey) | The free tier allows about 100 embedded texts per minute |
  | `OPENAI_API_KEY` | Only for `provider="openai"` embeddings or LLMs | Your OpenAI account | Per token |

  Variables set in the environment take precedence over `.env`, and a `.env` in the working folder (or one of its parents) over the one at the root of the repository.
  `.env` is gitignored; never commit it.
  The released items' embeddings and the predictions of the experiments are saved in `notebooks/data/`, so reproducing the saved results needs no key; predicting a new item needs both default keys, because its stem and options are embedded to find similar items.
- **Optional, for local models without API keys:** [Ollama](https://ollama.com), with a model pulled (`ollama pull llama3.2` for the LLM of the 2 × 2 experiment).
  FastEmbed, the other local option, comes with the package and downloads its embedding model (67 MB) on first use.
- **Optional, for the notebook:** Jupyter, such as VS Code with its Jupyter extension and the project's `.venv` kernel, or `uv run --with jupyterlab jupyter lab`.

## Installation

```sh
git clone https://github.com/wang-ze/timss-ai.git
cd timss-ai
uv sync                                     # creates .venv with Python 3.13 and installs the package
# create .env with your API keys (see Prerequisites)
uv run python -m unittest discover tests    # checks the installation
```

Run from the clone, the package reads `.env` from the repository root and keeps its caches and runs in `notebooks/data/`, which holds the saved embeddings and predictions.

The package can also be installed into another project, such as with `uv add git+https://github.com/wang-ze/timss-ai`.
It then reads `.env` from the working folder or one of its parents, and keeps its caches and runs in a `timss_math_data/` folder: the nearest one in the working folder or its parents, or a new one in the working folder.
That folder starts empty, so the first `ItemIndex.build()` embeds all 728 items (about 8 minutes on the Gemini API's free tier), and every leave-one-out prediction is requested anew; to reuse the saved ones instead, set `TIMSS_DATA_DIR` to the `notebooks/data` of a clone.

## Layout

```
CITATION.cff                                 how to cite the package (GitHub's "Cite this repository")
LICENSE                                      MIT
pyproject.toml, uv.lock                      the project and its locked dependencies
src/
├── timss_math/                              the installed package
│   └── item_parameter_prediction/           prediction of IRT item parameters (see Modules below)
│       └── data/                            its inputs: timss_math_released_items.jsonl and timss_math_released_item_parameters.csv
└── released_items.py                        helpers that built the released-item datasets (not part of the package)
tests/                                       unit tests (stdlib unittest)
notebooks/
├── predict_item_parameters_compare.ipynb    the 2 × 2 comparison of embedding models and LLMs, with its outputs
├── timss_math_released_items.md             sources, extraction, checks, and parameter sources of the datasets
├── released_items_2007_vs_2011.md           the TIMSS 2007 and 2011 grade 4 sources compared
└── data/                                    the datasets, item embeddings, and caches of transcriptions and predictions
results/
└── timss_math_predicted_item_parameters_compare.xlsx    the workbook of the 2 × 2 comparison
```

## The datasets

`notebooks/data/timss_math_released_items.jsonl` holds the 731 released items, one JSON object per line, and `notebooks/data/timss_math_released_item_parameters.xlsx` their official IRT parameters, with the source of each (sheets `item_parameters`, `sources`, and two ID crosswalks).
The files of the single sets, such as `timss99_g8_math_released_items.jsonl`, are next to them.
The package ships copies of the items and of the `item_parameters` sheet (as `timss_math_released_item_parameters.csv`) in `src/timss_math/item_parameter_prediction/data/`, which `tests/` checks against the files in `notebooks/data/`.

| Assessment | Grade | Items | With IRT parameters | Calibration (scale) of the parameters |
| --- | --- | --- | --- | --- |
| TIMSS 1995 | 4 | 70 | 70 | Joint 1995-2003, grade 4 |
| TIMSS Advanced 1995 | 12 | 37 | 31 | TIMSS Advanced 1995-2008 |
| TIMSS 1999 | 8 | 82 | 82 | Joint 1999-2003, grade 8 |
| TIMSS 2003 | 4 | 79 | 79 | Joint 1995-2003, grade 4 |
| TIMSS 2003 | 8 | 99 | 99 | Joint 1999-2003, grade 8 |
| TIMSS 2007 | 4 | 74 | 74 | Joint 2003-2007, grade 4 |
| TIMSS 2007 | 8 | 89 | 88 | Joint 2003-2007, grade 8 |
| TIMSS Advanced 2008 | 12 | 40 | 40 | TIMSS Advanced 1995-2008 |
| TIMSS 2011 | 4 | 73 | 73 | Joint 2007-2011, grade 4 |
| TIMSS 2011 | 8 | 88 | 88 | Joint 2007-2011, grade 8 |
| **All** | | **731** | **724** | |

`(assessment, grade, item_id)` identifies an item, because short IDs such as K1 and L10 repeat across sets.
IDs, classifications, answer keys, scoring categories, and statistics are parsed from the booklets and official files; stems, options, scoring guides, and figure descriptions are transcribed from page images by `google/gemini-3.8-flash` and kept only after checks against the parsed fields.
The parameters are those of the 3PL model for multiple-choice items, the 2PL model for 1-point constructed-response items, and the generalized partial credit model (GPCM) for items worth 2 or 3 points.
Parameters from different calibrations are on different scales; compare them across calibrations only after linking them.

```python
import pandas as pd

data = "src/timss_math/item_parameter_prediction/data"
items = pd.read_json(f"{data}/timss_math_released_items.jsonl", lines=True)
parameters = pd.read_csv(f"{data}/timss_math_released_item_parameters.csv")
items = items.merge(parameters, on=["assessment", "grade", "item_id"], suffixes=("", "_parameters"))
```

The datasets were built by a notebook that is not in this repository, with the helpers in `src/released_items.py`.
`notebooks/timss_math_released_items.md` documents the sets, every field, the extraction and its checks, the booklet quirks, the parameter sources, and the costs; please cite the IEA and NCES sources it lists.

## Item parameter prediction

`timss_math.item_parameter_prediction` predicts the IRT parameters of a new mathematics item from the most similar released items, whose parameters are known.

1. **Retrieval.**
   Each released item is one chunk, except three items whose stem and options repeat another item's word for word (K7 of TIMSS 1995, D11 of TIMSS 1999, and L15B of TIMSS Advanced 1995), so the index holds 728 items, 722 of them with parameters.
   Only the stem and options of each item are embedded and BM25-indexed, never its answer or scoring guide, in an in-memory Qdrant collection.
   Items can be found by keyword (BM25), semantic (embedding), or hybrid (reciprocal rank fusion of both) search.
2. **Reference items.**
   For a new item, the 10 most similar calibrated items of its grade (hybrid search by default), plus up to 5 of the most similar items with its IRT model and points, become its reference items.
3. **Prediction.**
   An LLM, prompted as an expert in testing the item's grade, content domain, and cognitive domain and as an IRT measurement expert, gets the reference items with their full text, percent correct, and parameters, and returns its reasoning and the item's parameters on the scale of the calibration of the item's assessment.
4. **Leave-one-out evaluation.**
   Each of the 722 calibrated items is predicted as if it were new, with its other parts (and the other items of its task) left out of its context, and its predictions are compared with its calibrated parameters and with two baselines that need no LLM: the mean parameter of the calibrated items of its grade, and of its reference items, with its IRT model and points.

Every step is a setting: the embedding provider and model (Gemini, OpenAI, OpenRouter, Ollama, or FastEmbed), the search method, the numbers of reference items, and the LLM provider, model, reasoning effort, and output cap (OpenRouter, OpenAI, Gemini, or Ollama).

## Using the package

### In Python

```python
from timss_math.item_parameter_prediction import ItemIndex, LeaveOneOut, ParameterPredictor, TimssItem, plots

index = ItemIndex.build()  # the released items, with the saved embeddings of gemini-embedding-001
index.search("fraction of a shape that is shaded", method="keyword")

predictor = ParameterPredictor(index)  # hybrid search; google/gemini-3.8-flash through OpenRouter
item = TimssItem(
    assessment="TIMSS 2011",  # the calibration whose scale the parameters are on
    grade=4,
    subject="Mathematics",
    content_domain="Number",
    main_topic="Whole Numbers",
    cognitive_domain="Applying",
    item_type="multiple_choice",
    stem="Sam has 4 boxes with 12 pencils in each box. How many pencils does Sam have?",
    options=[{"label": "A", "text": "16"}, {"label": "B", "text": "36"}, {"label": "C", "text": "48"}, {"label": "D", "text": "52"}],
    correct_option="C",
)
prepared = predictor.prepare(item)  # reference items and prompts, without calling the LLM
prediction = predictor.predict(item)  # one LLM request, about $0.01
prediction["reasoning"], prediction["slope"], prediction["location"], prediction["guessing"]

loo = LeaveOneOut(predictor)
loo.status()  # with the default settings, all 722 predictions are saved
results = loo.results()
results.summary("grade")
plots.mae_vs_baselines(results.errors)
```

`loo.request(max_requests=1)` requests missing predictions (`None` for all of them, which costs money), `await loo.request_async(...)` does the same in a notebook's event loop, and `loo.request_batch(...)` sends them through OpenRouter's Batch API at about half the price.
Local models need no API key:

```python
from timss_math.item_parameter_prediction import EmbeddingSettings, LLMSettings

index = ItemIndex.build(EmbeddingSettings(provider="fastembed", model="BAAI/bge-small-en-v1.5", dimension=None))
predictor = ParameterPredictor(index, LLMSettings(provider="ollama", model="llama3.2", reasoning_effort=None, max_tokens=4096))
```

### From the command line

`timss-predict-loo` runs a leave-one-out evaluation and writes `predictions.xlsx` (with a `settings` sheet) and `plots/` to `notebooks/data/parameter_prediction_runs/<settings>/`.
It sends no LLM request unless `--max-requests` is given (`all` for every missing prediction); run it with `--help` for every setting.

```sh
uv run timss-predict-loo                                   # the default settings; reuses the saved predictions
uv run timss-predict-loo --search-method semantic --llm-provider ollama --llm-model gpt-oss:20b --max-requests all
uv run timss-predict-loo --embedding-provider fastembed --llm-provider ollama --llm-model llama3.2 \
  --reasoning-effort omit --max-tokens 4096 --workers 1 --max-requests 10  # FastEmbed and llama3.2
uv run timss-predict-loo --search-method semantic --batch --max-requests all --no-wait  # OpenRouter batches
uv run timss-predict-loo --search-method semantic --batch  # later: collect the batches, then evaluate
uv run timss-plot-errors notebooks/data/parameter_prediction_runs/<settings>/predictions.xlsx
uv run timss-plot-errors run_a/predictions.xlsx run_b/predictions.xlsx --names hybrid semantic  # compare runs
```

The requests run as coroutines, `--workers` at a time (8 by default).
On an API error, such as exhausted credit, no further request starts, and the requests already running are saved before the error is raised, since they are paid for.
An interrupt (Ctrl-C) cancels every running request at once.

### OpenRouter batches

`--batch` (or `LeaveOneOut.request_batch`) sends the requests through OpenRouter's Batch API, which bills `google/gemini-3.8-flash` at half its per-token price, so a full leave-one-out run costs about $4.50 instead of $9.
A batch can take up to 24 hours, and OpenRouter returns its results only once the whole batch has completed.
Each submitted batch has a manifest in `batches/` of the predictions directory, so a run that stops (or `--no-wait`) collects its batches on the next `--batch` run instead of submitting them again; `--batch` with `--max-requests 0` only collects.
Never delete an open manifest, or its items are submitted and paid for again.
OpenRouter turns a batch away (HTTP 402) when its worst-case cost, the full output cap of every request, exceeds the available balance, and holds that amount until the batch completes: about $0.033 per item, while an item costs about $0.006 in a batch.
With a small balance, submit a few hundred items at a time (`--max-requests 200 --no-wait`) and repeat once the open batches have completed.
Batch and single requests send the same request body and save their predictions under the same names, so either reuses the other's predictions.

### Caches and the data directory

The caches and runs are in `notebooks/data/` of the clone, or in `timss_math_data/` when the package is installed elsewhere (see [Installation](#installation)); `--data-dir`, the `data_dir` arguments, or `$TIMSS_DATA_DIR` choose another directory.
The items and parameters are always read from the package's own `data/`.
Item embeddings are cached per embedding setting: the default settings use `notebooks/data/timss_math_item_embeddings.npz`, and other settings a file in `notebooks/data/cache/item_embeddings/`.
Predictions are saved in `notebooks/data/cache/parameter_predictions/timss_math/<provider>/<model>/`, one file per item, named by a hash of the LLM, reasoning effort, and prompts, so runs with different settings never overwrite each other's predictions and any change of the prompts requests every prediction again.
The index is kept in memory by default and rebuilt in about a second.

FastEmbed embeds each text on its own, because in a batch a text's vector depends slightly on the other texts of the batch.
Its default model reads at most 512 tokens, so the three parts of the TIMSS 2007 grade 8 class trip task (M032753A to M032753C), about 900 tokens each, are cut there.
Ollama is called through its native API: the reasoning effort becomes `think` (pass `reasoning_effort=None` for models that cannot think, such as `llama3.2`), and `ollama_num_ctx` (32,768 tokens by default) must hold the prompts, about 6,000 tokens, and the output.
`llama3.2` predicts an item in about 10 seconds on a laptop, and sometimes repeats itself until `max_tokens`; such output is reported as unusable and requested again on the next run.

### Modules

| Module | Contents |
| --- | --- |
| `items` | `ItemBank` (the released items, their chunks, calibrated parameters, and parts) and `TimssItem`, a new item. |
| `embeddings` | `EmbeddingSettings`: provider (`gemini`, `openai`, `openrouter`, `ollama`, or `fastembed`), model, dimension, Gemini task types or query and document prefixes, and quota. |
| `index` | `ItemIndex`, the Qdrant collection, with `search(text, method)` by `keyword`, `semantic`, or `hybrid` search, and `IndexSettings` (in memory by default, or a local path or server URL; hybrid candidates; BM25 `k1` and `b`). |
| `predictor` | `ParameterPredictor` (search method, `n_similar`, `n_same_model`) and `LLMSettings` (provider `openrouter`, `openai`, `gemini`, or `ollama`; model; reasoning effort; output cap). |
| `evaluation` | `LeaveOneOut` and `PredictionResults`: predict every calibrated item as if it were new, compute the errors and baselines, summarize them, and export them to Excel; the `timss-predict-loo` command. |
| `batch` | `OpenRouterBatches` and the manifests of submitted batches, for `LeaveOneOut.request_batch`. |
| `factorial` | `FactorialResults`: the runs of every combination of levels of a few settings, side by side, with main effects and interactions on the absolute error. |
| `plots` | Graphs of the errors: predicted against calibrated values, error distributions, mean absolute error against the baselines, bias by assessment, a comparison of runs, and the conditions of a factorial experiment; the `timss-plot-errors` command. |

## Results: embedding models and LLMs compared

`notebooks/predict_item_parameters_compare.ipynb` is a 2 × 2 factorial experiment on the leave-one-out predictions of all 722 calibrated items, with its outputs.
It crosses the embedding model (`gemini-embedding-001`, or FastEmbed's `BAAI/bge-small-en-v1.5` on the local computer) with the LLM (`google/gemini-3.8-flash` through OpenRouter with medium reasoning effort, or `llama3.2` in Ollama without reasoning), and keeps everything else the same.
`FactorialResults` compares the four conditions on the items that every condition predicted, with the main effects and the interaction on the absolute error, each with a paired t-test over the items.
The notebook reruns from the saved predictions without any LLM request, and `results/timss_math_predicted_item_parameters_compare.xlsx` is a copy of its workbook.

Mean absolute error of the predictions:

| Parameter | Items | Gemini embedding, Gemini LLM | Gemini embedding, llama3.2 | bge-small, Gemini LLM | bge-small, llama3.2 | Baseline: mean of the calibration |
| --- | --- | --- | --- | --- | --- | --- |
| Slope | 722 | 0.21 | 0.31 | 0.22 | 0.34 | 0.25 |
| Location | 722 | 0.39 | 0.65 | 0.40 | 0.66 | 0.54 |
| Guessing | 442 | 0.056 | 0.084 | 0.056 | 0.086 | 0.058 |
| Step 1 | 56 | 0.75 | 1.20 | 0.75 | 1.20 | 0.77 |

The baseline predicts each parameter as the mean of the calibrated items of the item's grade with its IRT model and points.

The LLM matters and the embedding model does not: replacing `google/gemini-3.8-flash` with `llama3.2` raises the mean absolute error of the location by 0.26 and of the slope by 0.10 (both p < 0.001), while replacing `gemini-embedding-001` with `bge-small-en-v1.5` raises them by 0.01 and 0.02 (p > 0.2), and the interaction is not significant.
With `google/gemini-3.8-flash`, the slopes and especially the locations beat the baseline, and the guessing and step parameters are about as good as the baseline; with `llama3.2`, all four are worse than the baseline.

## Development

```sh
uv run python -m unittest discover tests    # package data, requests against fakes, factorial tables
uv run ruff check src tests                 # lint (ruff's default rules; settings in pyproject.toml)
```

The tests make no API calls.

## Citation

If you use this package or its datasets, please cite it.
`CITATION.cff` holds the same metadata, from which GitHub's "Cite this repository" offers APA and BibTeX; keep its `version` equal to `pyproject.toml`'s.

> Wang, Z. (2026). *timss-math: Item-level datasets of the released TIMSS mathematics items and LLM predictions of their IRT parameters* (Version 0.1.0) [Computer software]. https://github.com/wang-ze/timss-ai

```bibtex
@software{wang_2026_timss_math,
  author  = {Wang, Ze},
  title   = {{timss-math}: Item-Level Datasets of the Released {TIMSS} Mathematics Items and {LLM} Predictions of Their {IRT} Parameters},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/wang-ze/timss-ai},
  license = {MIT}
}
```

The released items and their IRT parameters come from IEA and NCES publications, listed under "The sets" and "Item parameters" in `notebooks/timss_math_released_items.md`; please cite those sources as well.

## License

MIT; see `LICENSE`.
