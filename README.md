# timss-math

Item-level datasets of the 731 released TIMSS mathematics items (1995 to 2011, grades 4 and 8, and TIMSS Advanced), with the official IRT item parameters of 724 of them and the constants of the TIMSS reporting scales, and a Python package with two uses of them:

- **Item parameter prediction**: predict the IRT parameters of new items with an LLM or with Jev, grounded in the most similar calibrated items.
- **LLM test taking**: LLMs take the released items as a test, as test takers (a benchmark of LLMs) or as students (synthetic participants), and are scored with the IRT models on each calibration's scale and on the TIMSS reporting scale.

The README covers:

- [Prerequisites](#prerequisites) and [installation](#installation), what you need before using the package, and [the package&#39;s structure](#package-structure).
- [The datasets](#the-datasets), which can be used without the package.
- [Item parameter prediction](#item-parameter-prediction): the method, [how to run it](#predicting-item-parameters-with-the-package), and the results of a [2 × 2 experiment](#results-embedding-models-and-llms-compared) on embedding models and LLMs and of [Jev](#results-jev-compared-with-two-llms) against two LLMs.
- [LLMs taking the items as a test](#llms-taking-the-items-as-a-test): the method, how to run it, and the results of five LLMs in both roles, [as a benchmark and as synthetic data](#two-uses-a-benchmark-and-synthetic-data).

## Prerequisites

- **macOS, Linux, or Windows.**
- **Git**, to clone the repository.
- **[uv](https://docs.astral.sh/uv/)**, which installs Python 3.13 and every dependency into the project's own environment:

  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS and Linux; or, with Homebrew: brew install uv
  uv --version
  ```

  On Windows, in PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.
- **API keys** for the default embedding model, LLM, and Jev, in a file named `.env` at the root of the repository:

  ```sh
  OPENROUTER_API_KEY=sk-or-...
  GEMINI_API_KEY=...
  ```

  | Variable               | Used for                                                                                                                                                                                                                                                   | Where to get it                                                | Cost                                                                                                                                                                                                                   |
  | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `OPENROUTER_API_KEY` | Predictions with the default LLM, `google/gemini-3.8-flash`, through [OpenRouter](https://openrouter.ai), single or as batches, and with Jev, `typesafe/jev-1.13`; LLMs answering the items through OpenRouter, and the default scorer of their answers | A key from your OpenRouter account, which needs credit         | LLM: about $0.012 per item, or $9 for all 722 calibrated items, and half that through the Batch API; Jev: about $0.00005 per item from the item alone and $0.0003 with reference items, or $0.04 and $0.20 for all 722 |
  | `GEMINI_API_KEY`     | Embeddings with the default embedding model, `gemini-embedding-001`, through the Gemini API                                                                                                                                                              | [Google AI Studio](https://aistudio.google.com/apikey)          | The free tier allows about 100 embedded texts per minute                                                                                                                                                               |
  | `OPENAI_API_KEY`     | Only for `provider="openai"` embeddings or LLMs                                                                                                                                                                                                          | Your OpenAI account                                            | Per token                                                                                                                                                                                                              |
  | `ANTHROPIC_API_KEY`  | Only for `provider="anthropic"` LLMs, Claude through the Claude API's OpenAI-compatible endpoint                                                                                                                                                         | The [Claude Console](https://platform.claude.com/settings/keys) | Per token                                                                                                                                                                                                              |
  | `TYPESAFE_API_KEY`   | Only for Jev through TypeSafe's own API (`JevSettings(provider="typesafe", model="jev-1.13")`)                                                                                                                                                           | Your [TypeSafe](https://typesafe.ai) account                    | Per token                                                                                                                                                                                                              |

  Variables set in the environment take precedence over `.env`, and a `.env` in the working folder (or one of its parents) over the one at the root of the repository.
  `.env` is gitignored; never commit it.
  The released items' embeddings, the predictions of the experiments, and the LLMs' answers and their scores are saved in `notebooks/data/`, so reproducing the saved results needs no key; predicting a new item needs both default keys, because its stem and options are embedded to find similar items.
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
├── timss_math/                              the installed package (see Package structure below)
│   ├── items.py, llm.py, batch.py, ...      the modules both uses share
│   ├── data/                                the released items, their parameters, and the TIMSS scale constants
│   ├── item_parameter_prediction/           prediction of IRT item parameters
│   └── llm_test_taking/                     LLMs taking the released items as a test, scored with IRT
└── released_items.py                        helpers that built the released-item datasets (not part of the package)
tests/                                       unit tests (stdlib unittest)
notebooks/
├── predict_item_parameters_compare.ipynb    the 2 × 2 comparison of embedding models and LLMs, with its outputs
├── predict_item_parameters_compare_jev.ipynb  Jev, alone and with reference items, against two LLMs, with its outputs
├── llm_test_taking.ipynb                    LLMs answering the released items, scored with IRT, with its outputs
├── timss_math_released_items.md             sources, extraction, checks, and parameter sources of the datasets
├── released_items_2007_vs_2011.md           the TIMSS 2007 and 2011 grade 4 sources compared
├── timss_math_scale_transformation_constants.md  the constants that put each calibration's θ on the TIMSS reporting scales, and their sources
└── data/                                    the datasets, item embeddings, caches of transcriptions, predictions, and LLM answers and scores, and the workbook of the LLMs' scores
results/
├── timss_math_predicted_item_parameters_compare.xlsx    the workbook of the 2 × 2 comparison
├── predict_item_parameters_compare_jev.xlsx             the workbook of the Jev comparison
└── timss_math_llm_answers_scored.xlsx                   the workbook of the LLMs taking the items as a test
```

## Package structure

`timss_math` has two subpackages, one per use, on top of the modules that both share:

| Module                           | Contents                                                                                                                                                                                                                                                               |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `items`                        | The released items and their calibrated parameters (`load_released_items`, `load_calibrated_parameters`, `ItemBank`), the parts of each item, `TimssItem`, and `PACKAGE_DATA_DIR`, the package's `data/`.                                                  |
| `llm`                          | `LLMSettings` (provider `openrouter`, `openai`, `gemini`, `anthropic`, or `ollama`; model; reasoning effort; output cap) and `LLMClient`, which sends prompts with a structured-output schema, single, as coroutines, or as the body of a batch request. |
| `batch`                        | `OpenRouterBatches` and `BatchManifests`, OpenRouter's Batch API and the manifests of submitted batches.                                                                                                                                                           |
| `running`                      | `run_sync`, which runs a run of requests from a script or a notebook, and `UnusableOutputError`, output that a rerun may get right.                                                                                                                                |
| `config`                       | The data directory of the caches and runs, the API keys, and file writing.                                                                                                                                                                                             |
| `plot_style`, `command_line` | The look of the graphs, and what the commands share.                                                                                                                                                                                                                   |
| `item_parameter_prediction`    | Prediction of the IRT parameters of new items (see [its modules](#modules)).                                                                                                                                                                                            |
| `llm_test_taking`              | LLMs taking the released items as a test (see [its modules](#modules-of-llm_test_taking)).                                                                                                                                                                              |

The shared modules were in `item_parameter_prediction` before version 0.3.0, and their old names still work: `timss_math.item_parameter_prediction.items`, `.config`, and `.batch` are the same modules as `timss_math.items`, `.config`, and `.batch`, and the client's names (`LLMClient`, `LLMSettings`, ...) can still be imported from `timss_math.item_parameter_prediction.llm`.

## The datasets

`notebooks/data/timss_math_released_items.jsonl` holds the 731 released items, one JSON object per line, and `notebooks/data/timss_math_released_item_parameters.xlsx` their official IRT parameters, with the source of each (sheets `item_parameters`, `sources`, and two ID crosswalks).
The files of the single sets, such as `timss99_g8_math_released_items.jsonl`, are next to them.
The package ships copies of the items and of the `item_parameters` sheet (as `timss_math_released_item_parameters.csv`) in `src/timss_math/data/`, which `tests/` checks against the files in `notebooks/data/`.

| Assessment          | Grade | Items         | With IRT parameters | Calibration (scale) of the parameters |
| ------------------- | ----- | ------------- | ------------------- | ------------------------------------- |
| TIMSS 1995          | 4     | 70            | 70                  | Joint 1995-2003, grade 4              |
| TIMSS Advanced 1995 | 12    | 37            | 31                  | TIMSS Advanced 1995-2008              |
| TIMSS 1999          | 8     | 82            | 82                  | Joint 1999-2003, grade 8              |
| TIMSS 2003          | 4     | 79            | 79                  | Joint 1995-2003, grade 4              |
| TIMSS 2003          | 8     | 99            | 99                  | Joint 1999-2003, grade 8              |
| TIMSS 2007          | 4     | 74            | 74                  | Joint 2003-2007, grade 4              |
| TIMSS 2007          | 8     | 89            | 88                  | Joint 2003-2007, grade 8              |
| TIMSS Advanced 2008 | 12    | 40            | 40                  | TIMSS Advanced 1995-2008              |
| TIMSS 2011          | 4     | 73            | 73                  | Joint 2007-2011, grade 4              |
| TIMSS 2011          | 8     | 88            | 88                  | Joint 2007-2011, grade 8              |
| **All**       |       | **731** | **724**       |                                       |

`(assessment, grade, item_id)` identifies an item, because short IDs such as K1 and L10 repeat across sets.
IDs, classifications, answer keys, scoring categories, and statistics are parsed from the booklets and official files; stems, options, scoring guides, and figure descriptions are transcribed from page images by `google/gemini-3.8-flash` and kept only after checks against the parsed fields.
The parameters are those of the 3PL model for multiple-choice items, the 2PL model for 1-point constructed-response items, and the generalized partial credit model (GPCM) for items worth 2 or 3 points.
Parameters from different calibrations are on different scales; compare them across calibrations only after linking them.

`notebooks/data/timss_math_scale_transformation_constants.csv` holds, for each of the seven calibrations, the linear transformation constants that put its θ on the TIMSS reporting scale (scale score = A + B θ, one A and B per plausible value).
They are published in the TIMSS 2007 and TIMSS Advanced 2008 Technical Reports and in Methods and Procedures in TIMSS and PIRLS 2011 for five calibrations, and derived by common-item linking for the joint 1995-2003 and 1999-2003 calibrations, whose constants TIMSS never published; the same derivation reproduces the published 2007 constants within 2 scale points.
`notebooks/timss_math_scale_transformation_constants.md` documents the sources, the derivation, and its check.
The package ships a copy in `src/timss_math/data/`, which `tests/` checks against the file in `notebooks/data/`.

```python
import pandas as pd

data = "src/timss_math/data"
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
3. **Prediction**, by one of two predictors that share the steps before it:
   - **An LLM**, prompted as an expert in testing the item's grade, content domain, and cognitive domain and as an IRT measurement expert, gets the reference items with their full text, percent correct, and parameters, and returns its reasoning and the item's parameters on the scale of the calibration of the item's assessment.
   - **Jev** (TypeSafe's `typesafe/jev-1.13`) answers questions about a JSON state, such as a score question with an ordered rubric of at most 10 criteria, with a probability for each criterion and a score, the probability-weighted mean of the criteria's positions.
     So each parameter is asked on a value scale, one criterion per value (such as `-2.0 (very easy item)`, `-1.0`, ..., `2.0 (very hard item)` for the location), about the item alone or the item and its reference items.
     Its probabilities over the values are a distribution of the parameter, and its point prediction is their probability-weighted mean (EAP), their median, or the most likely value (MAP).
4. **Leave-one-out evaluation.**
   Each of the 722 calibrated items is predicted as if it were new, with its other parts (and the other items of its task) left out of its context, and its predictions are compared with its calibrated parameters and with two baselines that need neither an LLM nor Jev: the mean parameter of the calibrated items of its grade, and of its reference items, with its IRT model and points.

Every step is a setting: the embedding provider and model (Gemini, OpenAI, OpenRouter, Ollama, or FastEmbed), the search method, the numbers of reference items, and then either the LLM provider, model, reasoning effort, and output cap (OpenRouter, OpenAI, Gemini, Anthropic, or Ollama), or Jev's provider and model (OpenRouter or TypeSafe), whether it sees the reference items, its value scales and questions, and its estimator.

## Predicting item parameters with the package

### In Python

```python
from timss_math.item_parameter_prediction import ItemIndex, LeaveOneOut, LLMPredictor, TimssItem, plots

index = ItemIndex.build()  # the released items, with the saved embeddings of gemini-embedding-001
index.search("fraction of a shape that is shaded", method="keyword")

predictor = LLMPredictor(index)  # hybrid search; google/gemini-3.8-flash through OpenRouter
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
predictor = LLMPredictor(index, LLMSettings(provider="ollama", model="llama3.2", reasoning_effort=None, max_tokens=4096))
```

Jev takes the same index and retrieval settings in place of the LLM:

```python
from timss_math.item_parameter_prediction import JevPredictor, RunComparison

jev = JevPredictor(index)  # Jev 1.13 through OpenRouter, with the item's reference items (hybrid search)
record = jev.predict(item)  # one request, about $0.0003
jev.parameters(record)  # {"slope": ..., "location": ..., "guessing": ...}, the probability-weighted means
jev.posterior(record, "location")  # the most likely value, the mean, the median, and the spread

alone = JevPredictor(index, with_references=False, estimator="median")  # the item alone; the median of each answer
custom = JevPredictor(index, scales={"slope": ["0.3", "0.6", "0.9", "1.2", "1.5", "1.8"]})  # another value scale

runs = {"jev": LeaveOneOut(jev).results(), "gemini": LeaveOneOut(predictor).results()}
comparison = RunComparison(runs)  # the runs on the items both predicted, with the baselines as runs
comparison.mae(), comparison.paired(), comparison.scale_agreement()
plots.comparison_mae_by_grade(comparison)
```

The estimator is applied when a saved answer is read, so changing it requests nothing; changing the scales or the questions asks Jev again.

### From the command line

`timss-predict-loo` runs a leave-one-out evaluation and writes `predictions.xlsx` (with a `settings` sheet) and `plots/` to `notebooks/data/parameter_prediction_runs/<settings>/`.
It sends no request to an LLM or Jev unless `--max-requests` is given (`all` for every missing prediction); run it with `--help` for every setting.

```sh
uv run timss-predict-loo                                   # the default settings; reuses the saved predictions
uv run timss-predict-loo --search-method semantic --llm-provider ollama --llm-model gpt-oss:20b --max-requests all
uv run timss-predict-loo --embedding-provider fastembed --llm-provider ollama --llm-model llama3.2 \
  --reasoning-effort omit --max-tokens 4096 --workers 1 --max-requests 10  # FastEmbed and llama3.2
uv run timss-predict-loo --search-method semantic --batch --max-requests all --no-wait  # OpenRouter batches
uv run timss-predict-loo --search-method semantic --batch  # later: collect the batches, then evaluate
uv run timss-predict-loo --predictor jev                   # Jev with reference items; reuses the saved answers
uv run timss-predict-loo --predictor jev --jev-context item --estimator median --max-requests all  # Jev, item alone
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
Predictions are saved in `notebooks/data/cache/parameter_predictions/timss_math/<provider>/<model>/`, one file per item, named by a hash of the LLM, reasoning effort, and prompts, or of Jev's model, state, and questions, so runs with different settings never overwrite each other's predictions and any change of the prompts or questions requests every prediction again.
Jev's answers are in `openrouter/typesafe_jev-1.13/`; a request takes about 0.2 seconds, and 8 at a time answered 55 items in about 2 seconds.
The index is kept in memory by default and rebuilt in about a second.

FastEmbed embeds each text on its own, because in a batch a text's vector depends slightly on the other texts of the batch.
Its default model reads at most 512 tokens, so the three parts of the TIMSS 2007 grade 8 class trip task (M032753A to M032753C), about 900 tokens each, are cut there.
Ollama is called through its native API: the reasoning effort becomes `think` (pass `reasoning_effort=None` for models that cannot think, such as `llama3.2`), and `ollama_num_ctx` (32,768 tokens by default) must hold the prompts, about 6,000 tokens, and the output.
`llama3.2` predicts an item in about 10 seconds on a laptop, and sometimes repeats itself until `max_tokens`; such output is reported as unusable and requested again on the next run.
Claude (`provider="anthropic"`, with model IDs such as `claude-haiku-5-5`) is called through the Claude API's OpenAI-compatible endpoint with the same OpenAI SDK.
That endpoint follows the structured-output schema, but it ignores `reasoning_effort` and rejects adaptive thinking, so the reasoning effort is `None`, which leaves thinking to the model, or `"none"`, which turns it off (Claude Opus 5.5 and Sonnet 5.5 refuse that); for effort levels, use a Claude model through OpenRouter, such as `anthropic/claude-haiku-5.5`.
Its responses report neither the cost nor the reasoning tokens, and it takes no batches.

### Modules

The modules of `timss_math.item_parameter_prediction`, which use the shared modules (see [Package structure](#package-structure)):

| Module         | Contents                                                                                                                                                                                                                                                                                 |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `embeddings` | `EmbeddingSettings`: provider (`gemini`, `openai`, `openrouter`, `ollama`, or `fastembed`), model, dimension, Gemini task types or query and document prefixes, and quota.                                                                                                   |
| `index`      | `ItemIndex`, the Qdrant collection, with `search(text, method)` by `keyword`, `semantic`, or `hybrid` search, and `IndexSettings` (in memory by default, or a local path or server URL; hybrid candidates; BM25 `k1` and `b`).                                           |
| `predictor`  | `Predictor`, what every predictor shares: the reference items (search method, `n_similar`, `n_same_model`) and the methods a leave-one-out run uses.                                                                                                                               |
| `llm`        | `LLMPredictor` and its prompts, with an LLM of `timss_math.llm`.                                                                                                                                                                                                                     |
| `jev`        | `JevPredictor` (with or without reference items; value scales and questions; estimator `eap`, `median`, or `map`), `JevSettings` (provider `openrouter` or `typesafe`; model), and `ValueScale`.                                                                         |
| `evaluation` | `LeaveOneOut` and `PredictionResults`: predict every calibrated item as if it were new with either predictor, single or in OpenRouter batches, compute the errors and baselines, summarize them, and export them to Excel; the `timss-predict-loo` command.                        |
| `factorial`  | `FactorialResults`: the runs of every combination of levels of a few settings, side by side, with main effects and interactions on the absolute error.                                                                                                                                 |
| `comparison` | `RunComparison`: any runs on the same items, with the baselines as runs, paired t-tests (Holm-adjusted), agreement on the value scales (exact, within one value, quadratic weighted kappa), Jev's answers, and cost.                                                                   |
| `plots`      | Graphs of the errors: predicted against calibrated values, error distributions, mean absolute error against the baselines, bias by assessment, a comparison of runs, the conditions of a factorial experiment, and the tables of a `RunComparison`; the `timss-plot-errors` command. |

## Results: embedding models and LLMs compared

`notebooks/predict_item_parameters_compare.ipynb` is a 2 × 2 factorial experiment on the leave-one-out predictions of all 722 calibrated items, with its outputs.
It crosses the embedding model (`gemini-embedding-001`, or FastEmbed's `BAAI/bge-small-en-v1.5` on the local computer) with the LLM (`google/gemini-3.8-flash` through OpenRouter with medium reasoning effort, or `llama3.2` in Ollama without reasoning), and keeps everything else the same.
`FactorialResults` compares the four conditions on the items that every condition predicted, with the main effects and the interaction on the absolute error, each with a paired t-test over the items.
The notebook reruns from the saved predictions without any LLM request, and `results/timss_math_predicted_item_parameters_compare.xlsx` is a copy of its workbook.

Mean absolute error of the predictions:

| Parameter | Items | Gemini embedding, Gemini LLM | Gemini embedding, llama3.2 | bge-small, Gemini LLM | bge-small, llama3.2 | Baseline: mean of the calibration |
| --------- | ----- | ---------------------------- | -------------------------- | --------------------- | ------------------- | --------------------------------- |
| Slope     | 722   | 0.21                         | 0.31                       | 0.22                  | 0.34                | 0.25                              |
| Location  | 722   | 0.39                         | 0.65                       | 0.40                  | 0.66                | 0.54                              |
| Guessing  | 442   | 0.056                        | 0.084                      | 0.056                 | 0.086               | 0.058                             |
| Step 1    | 56    | 0.75                         | 1.20                       | 0.75                  | 1.20                | 0.77                              |

The baseline predicts each parameter as the mean of the calibrated items of the item's grade with its IRT model and points.

The LLM matters and the embedding model does not: replacing `google/gemini-3.8-flash` with `llama3.2` raises the mean absolute error of the location by 0.26 and of the slope by 0.10 (both p < 0.001), while replacing `gemini-embedding-001` with `bge-small-en-v1.5` raises them by 0.01 and 0.02 (p > 0.2), and the interaction is not significant.
With `google/gemini-3.8-flash`, the slopes and especially the locations beat the baseline, and the guessing and step parameters are about as good as the baseline; with `llama3.2`, all four are worse than the baseline.

## Results: Jev compared with two LLMs

`notebooks/predict_item_parameters_compare_jev.ipynb` compares four leave-one-out runs on all 722 calibrated items, with its outputs: Jev from the item alone, Jev with the item's reference items, and the two LLMs of the 2 × 2 experiment (with `gemini-embedding-001`), all with the same retrieval, so Jev with reference items sees exactly the reference items the LLMs see.
Jev answers each parameter on a value scale, and its point prediction is the probability-weighted mean of the values.
`RunComparison` compares the runs and both baselines on the same items, with paired t-tests of the absolute errors (Holm-adjusted) and agreement on the value scales.
The notebook reruns from the saved predictions and answers without any request, and `results/predict_item_parameters_compare_jev.xlsx` is a copy of its workbook.

Mean absolute error of the predictions:

| Parameter | Items | Jev, item alone | Jev, with reference items | Gemini LLM | llama3.2 | Baseline: mean of the calibration | Baseline: mean of the reference items |
| --------- | ----- | --------------- | ------------------------- | ---------- | -------- | --------------------------------- | ------------------------------------- |
| Slope     | 722   | 0.32            | 0.22                      | 0.21       | 0.31     | 0.25                              | 0.23                                  |
| Location  | 722   | 0.68            | 0.50                      | 0.39       | 0.65     | 0.54                              | 0.53                                  |
| Guessing  | 442   | 0.071           | 0.059                     | 0.056      | 0.084    | 0.058                             | 0.060                                 |
| Step 1    | 56    | 0.83            | 0.79                      | 0.75       | 1.20     | 0.77                              | 0.86                                  |

`google/gemini-3.8-flash` predicts every parameter best.
Jev needs the reference items: alone it gives almost every item a slope of 1.0 and predicts items too easy, and with them its location error falls by 0.18, so that it beats the mean of the calibration (p = 0.02) and llama3.2 (p < 0.001) but not the mean of the reference items, and stays behind the Gemini LLM (by 0.11, p < 0.001).
Its value scales bound its predictions and set their precision, so the choice of the values matters; with reference items, the most likely value or the median of its answers predicts the slope, location, and guessing worse than their probability-weighted mean.
Jev is cheap and fast: all 722 items cost $0.04 from the item alone and $0.20 with reference items, against about $9 with the Gemini LLM ($4.50 as batches), and a request takes about 0.2 seconds.

## LLMs taking the items as a test

`timss_math.llm_test_taking` has LLMs answer the released items and scores their answers with the IRT models and the calibrated parameters.
Its two roles make two uses: as test takers, the LLMs take a benchmark of 724 tasks whose difficulty for students is known; as students, they are synthetic participants, whose answers can be checked against real students' (see [Two uses](#two-uses-a-benchmark-and-synthetic-data)).

1. **Units.**
   The 724 items with parameters are grouped into 652 units, the items a student sees together: an item alone, or all parts of a multi-part item and all items of a shared-stimulus task, in booklet order, since later parts often depend on earlier ones.
   Each unit is asked in one request.
2. **Conditions.**
   A condition is an LLM, with its provider, model, reasoning effort, output cap, and the rest of `LLMSettings`, and a role: `"test_taker"` takes the test written for students of the grade (evaluating the LLM), and `"student"` takes it as a student of that grade (the LLM as a participant), without ever being asked to make mistakes.
   The system prompt names the test and its grade, and the user prompt holds each item's stem, its figure described in words, and its options, never its ID, answer key, scoring guide, or statistics.
   The answer comes back as structured output: an option letter or a written answer per item.
3. **Scoring.**
   A multiple-choice answer is scored against the official answer key, and a constructed response by an LLM scorer, which sees the unit, the item's scoring guide and notes, and the response, and chooses a category of the guide; the points come from the category.
4. **IRT.**
   Each condition's θ is estimated on each calibration's scale, the EAP with a standard normal prior and the maximum-likelihood estimate, never pooled over calibrations, and put on the TIMSS reporting scale with the calibration's transformation constants, with the highest International Benchmark it reaches.
   Each item score is put on the IRT scale too, at the item's location, with the probability and expected score that the condition's θ gives it.

### In a notebook

`notebooks/llm_test_taking.ipynb` runs the whole analysis and shows every step, with its outputs, and then uses the answers both ways, as a benchmark and as synthetic data.
Run its cells as they are to reproduce the run of 2026-10-19 from the saved answers, without API keys or requests, or change its Settings cell (the LLMs, roles, scorer, and how many requests to send) to test other LLMs; a rerun requests only what is missing.
`results/timss_math_llm_answers_scored.xlsx` is a copy of its workbook.

### In Python

```python
from timss_math.llm_test_taking import Administration, LLMSettings, ScoringResults, conditions_of, plots

administration = Administration()  # the ten conditions and the scorer of the run of 2026-10-19
results = administration.results()  # from the saved answers and verdicts, without any request
results.score_table(lambda row: f"{row.timss_eap:.0f}, {row.timss_benchmark}")  # TIMSS scale score by calibration
plots.theta_figure(results)  # θ of each condition on each calibration's scale
plots.item_scores_figure(results)  # every item score at the item's location
results.to_excel("llm_answers_scored.xlsx")

models = [
    LLMSettings(model="openai/gpt-6-luna", reasoning_effort="high"),  # through OpenRouter
    LLMSettings(provider="anthropic", model="claude-haiku-5-5", reasoning_effort="none", max_tokens=4096),
    LLMSettings(provider="ollama", model="llama3.2", reasoning_effort=None, max_tokens=4096),  # local, no key
]
scorer = LLMSettings(model="google/gemini-3.8-flash", reasoning_effort="medium", max_tokens=8192)
administration = Administration(conditions_of(models, roles=["test_taker", "student"]), scorer)
administration.request_answers(max_requests=1)  # one unit per condition; None for every unit, which costs money
administration.request_verdicts(max_requests=None)  # the scorer's verdicts on the constructed responses
results = administration.results()
```

Requests cost money, so `max_requests` must be given: 0 only collects the batches that an earlier run submitted, 1 is a test request, and `None` requests everything missing.
`request_answers` sends OpenRouter batches by default (`mode="batch"`, at about half the price, which can take up to 24 hours) for the models that have them, and direct requests for the others and for every other provider, all conditions at once; `wait=True` waits for the batches, and otherwise a later call collects them.
`request_verdicts` sends its requests directly by default.
In a notebook, `await administration.request_answers_async(...)` and `await administration.request_verdicts_async(...)` run in its event loop.
Any provider of `LLMSettings` can answer or score: OpenRouter, OpenAI, Gemini, Anthropic (with `reasoning_effort` `None` or `"none"` only, see above), or a local Ollama model.
`ScoringResults.read_excel` reads a workbook back, such as to draw its graphs.

### From the command line

`timss-llm-test` runs the same steps and writes `scores.xlsx` and `plots/` to `notebooks/data/llm_test_runs/<settings>/` (`--output-dir` for another folder), and prints each condition's share of the points and its TIMSS scale scores.
It sends no request unless `--max-requests` is given (`all` for every missing unit); run it with `--help` for every setting.

```sh
uv run timss-llm-test                                  # the run of 2026-10-19, from the saved answers and verdicts
uv run timss-llm-test --llm-model openai/gpt-6-luna --reasoning-effort none high --role student --max-requests 1
uv run timss-llm-test --llm-provider anthropic --llm-model claude-haiku-5-5 --reasoning-effort omit none --max-requests all
uv run timss-llm-test --llm-provider ollama --llm-model llama3.2 --reasoning-effort omit --max-tokens 4096 --max-requests 10
uv run timss-llm-test --conditions my_llms.json --max-requests all         # submits OpenRouter batches and returns
uv run timss-llm-test --conditions my_llms.json --scorer-max-requests all  # later: collects them, then scores
```

`--llm-model` gives one LLM, one condition per `--reasoning-effort` and `--role` (both roles by default), and `--conditions` several, as a JSON list of `LLMSettings` fields, each with its own `roles` if it names them, such as `[{"model": "openai/gpt-6-luna", "reasoning_effort": "high"}, {"provider": "anthropic", "model": "claude-haiku-5-5", "reasoning_effort": null, "roles": ["student"]}]`.
`--scorer-model` and its siblings change the scorer, and `--token-price` prices a model whose provider reports no cost.
Requested answers are scored in the same run; answers collected later from batches need `--scorer-max-requests all`.

### Caches and the run of 2026-10-19

Answers are saved in `notebooks/data/cache/llm_answers/<provider>/<model>/<reasoning effort>/<role>/`, one file per unit, and the scorer's verdicts in `notebooks/data/cache/llm_answer_scores/<provider>/<model>/<reasoning effort>/`, one file per item and response, so identical responses share one verdict.
Each file is named by a hash of the model, its reasoning effort, both prompts, and the output schema, but not `max_tokens`, so a rerun requests only what is missing, and a changed prompt asks again.
Submitted batches have manifests in the `batches/` folder of their cache; never delete an open one, or its requests are submitted and paid for again.

The caches hold the answers and verdicts of the run of 2026-10-19: `openai/gpt-6-luna` without and with high reasoning and `google/gemma-3-12b-it` through OpenRouter, and `claude-haiku-5-5` through the Claude API without thinking and with its default thinking, each in both roles, scored by `google/gemini-3.8-flash` with medium reasoning.
Every unit is answered except one of `gemma-3-12b-it` as a student (L16 of TIMSS Advanced 1995, cut off at `max_tokens` twice), and all 2,230 distinct constructed responses are scored.
The run cost $1.75 as test takers and $1.98 as students, most of it the scorer's verdicts.
`Administration().results()` reproduces its workbook, `notebooks/data/timss_math_llm_answers_scored.xlsx` (copied to `results/`), without any request.

| LLM                                    | Points earned, test taker | As a student | TIMSS scale (EAP) over the seven calibrations, test taker | As a student |
| -------------------------------------- | ------------------------- | ------------ | --------------------------------------------------------- | ------------ |
| `gpt-6-luna`, high reasoning         | 97.6%                     | 97.3%        | 715 to 843                                                | 716 to 843   |
| `claude-haiku-5-5`, default thinking | 95.9%                     | 95.0%        | 691 to 843                                                | 676 to 806   |
| `gpt-6-luna`, no reasoning           | 85.2%                     | 84.3%        | 597 to 672                                                | 600 to 674   |
| `claude-haiku-5-5`, no reasoning     | 83.7%                     | 84.0%        | 583 to 657                                                | 589 to 674   |
| `gemma-3-12b-it`                     | 59.9%                     | 63.8%        | 522 to 581                                                | 528 to 585   |

With reasoning, `gpt-6-luna` and `claude-haiku-5-5` are above the Advanced International Benchmark (625) in every calibration and in both roles.
A perfect or near-perfect pattern only says that θ is above most of the items, so the EAP of the strongest conditions is set partly by the prior and understates them, and the MLE of a perfect pattern is infinite.
The scorer is an LLM, so read a sample of its verdicts (`scorer_reasoning` in the workbook) before trusting small differences between conditions.

### Two uses: a benchmark and synthetic data

The notebook analyzes the item scores of the run of 2026-10-19 both ways.

**As a benchmark** (the LLMs as test takers), with 95% confidence intervals from resampling the items:

- Reasoning adds about 12 points to both `gpt-6-luna` and `claude-haiku-5-5`, and nearly saturates the benchmark: both reasoning conditions earn every point of TIMSS Advanced, and `gpt-6-luna` leads `claude-haiku-5-5` by only 1.7 points (0.6 to 2.8).
- Without reasoning, the LLMs earn more of the points of constructed-response items than of multiple-choice items (`claude-haiku-5-5` 93.0% against 76.6%), the opposite of students; a constructed-response answer includes the work, which gives an LLM room to think, and is scored by an LLM, which may be lenient.
- Items with a figure, which an LLM sees only as a description, cost the LLMs without reasoning 4 to 7 points.
- What is hard for LLMs is not what is hard for students: the rank correlation of the number of LLMs with full credit and students' percent correct is 0.15, and nine items that 60% or more of students answered correctly were missed by at least three of the five LLMs.

**As synthetic data** (the LLMs as students), checked against real students' percent correct and the IRT models:

- Every synthetic student answers more items correctly than the average real student of every grade: real students answered 52%, 45%, and 43% of the grade 4, grade 8, and TIMSS Advanced items correctly, and the synthetic students 53% to 100%.
- The student role changes almost nothing: compared item by item with the same LLM as a test taker, four of the five LLMs differ by at most 2.4 points at every grade, and only `gemma-3-12b-it` does better as a student (by 5.4 points at grade 8); two items given to both fourth and eighth graders, on which real eighth graders did 14 and 12 points better, were answered alike at both grades.
- Synthetic students do not find the same items hard (rank correlations with students' percent correct of 0.14 at most), `gemma-3-12b-it`'s response patterns misfit the IRT models (l<sub>z</sub> below -2 on six of the seven calibrations), and every synthetic student ranks above the median real student, between the 62nd and the 99.9th percentile.

So the released items make a useful benchmark, and its students' statistics show where LLMs and students differ; synthetic students, however, need more than a role to resemble real ones, such as personas, which the notebook outlines with the checks that would tell whether they work.

### Modules of `llm_test_taking`

| Module             | Contents                                                                                                                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `units`          | `ItemUnits`, the released items with calibrated parameters and their units, and `Unit`.                                                                                                                             |
| `answering`      | The prompts and output schema with which a condition answers a unit, by role (`"test_taker"` or `"student"`).                                                                                                       |
| `conditions`     | `Condition` (an `LLMSettings` and a role), `conditions_of`, `DEFAULT_MODELS` and `DEFAULT_SCORER` (the run of 2026-10-19), and the cost of the answers, with `TOKEN_PRICES` for providers that report none. |
| `requesting`     | Requests saved as JSON records, sent directly or as OpenRouter batches (`send`, `RunReport`).                                                                                                                       |
| `scoring`        | The scorer's prompts and schema, the points of the scoring-guide categories, and the scores of the answers.                                                                                                             |
| `irt`            | The 3PL, 2PL, and GPCM models (`ItemModel`), the EAP and MLE of θ (`estimate_theta`), and checks of both.                                                                                                          |
| `scales`         | The calibrations and their TIMSS reporting scales (`Calibrations`), and the International Benchmarks.                                                                                                                 |
| `results`        | `ScoringResults`: the conditions, θ and TIMSS scale scores by calibration, and item scores on the IRT scale, to and from Excel.                                                                                      |
| `administration` | `Administration`: the conditions answer, the scorer scores, and the results are estimated.                                                                                                                            |
| `plots`          | θ of each condition by calibration, and the item scores on the IRT scale.                                                                                                                                              |
| `cli`            | The `timss-llm-test` command.                                                                                                                                                                                         |

They use the shared modules (see [Package structure](#package-structure)) and the scale constants in the package's `data/`.

## Development

```sh
uv run python -m unittest discover tests    # package data, requests against fakes, factorial tables
uv run ruff check src tests                 # lint (ruff's default rules; settings in pyproject.toml)
```

The tests make no API calls.

## Citation

If you use this package or its datasets, please cite it.
`CITATION.cff` holds the same metadata, from which GitHub's "Cite this repository" offers APA and BibTeX; keep its `version` equal to `pyproject.toml`'s.

> Wang, Z. (2026). *timss-math: Released TIMSS mathematics items with IRT parameters, for predicting item parameters and testing LLMs* (Version 0.3.0) [Computer software]. https://github.com/wang-ze/timss-ai

```bibtex
@software{wang_2026_timss_math,
  author  = {Wang, Ze},
  title   = {{timss-math}: Released {TIMSS} Mathematics Items with {IRT} Parameters, for Predicting Item Parameters and Testing {LLMs}},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/wang-ze/timss-ai},
  license = {MIT}
}
```

The released items and their IRT parameters come from IEA and NCES publications, listed under "The sets" and "Item parameters" in `notebooks/timss_math_released_items.md`; please cite those sources as well.

## License

MIT; see `LICENSE`.
