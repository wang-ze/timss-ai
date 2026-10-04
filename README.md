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
