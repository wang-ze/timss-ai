"""Compare leave-one-out runs on the same items: any predictors, settings, or estimators, with paired tests.

``RunComparison(runs)`` takes the ``PredictionResults`` of a few runs, such as Jev from the item alone, Jev with
reference items, and two LLMs in ``predict_item_parameters_compare_jev.ipynb``, and compares them on the items that
every run predicted. Where ``FactorialResults`` crosses factors, a comparison takes any runs, without a design:

- ``errors``: one row per run, item, and parameter, with the two baselines that need no LLM or Jev as runs whose
  prediction is the baseline's value: the calibration-summary mean (the mean parameter of the calibrated items of the
  item's grade with its model and points, its parts left out), and the reference-item mean (the mean over its
  reference items with its model and points, from ``reference_run``).
- ``mae``, ``summary``: the mean absolute error of every run and baseline, by parameter and by grade, and the bias,
  spread, RMSE, and Pearson and Spearman correlations.
- ``paired``: every pair of runs, and every run against each baseline, as the mean per-item difference of their
  absolute errors with a paired t-test, Holm-adjusted per parameter.
- ``scale_choices``, ``scale_agreement``, ``value_spread``: the value scales (``scales``, Jev's by default) as
  categories. A Jev run's value is its most likely one (``map``), any other run's the scale value closest to its
  prediction, and both are compared with the scale value closest to the calibrated parameter: exact, within one value,
  and quadratic weighted kappa, against the best constant value.
- ``posteriors``, ``cost``, ``coverage``, ``wide``, ``predictions``, ``settings``, and ``to_excel`` for all of them.

The paired tests treat the items as independent, although the parts of a multi-part item, and the items of one task,
are not, so their p-values are somewhat too small.
"""

from __future__ import annotations

import functools
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import excel_value
from ..items import PARAMETER_NAMES
from .evaluation import PredictionResults, correlation
from .factorial import SAVED_FILE_COLUMNS, paired_t_test
from .jev import DEFAULT_VALUE_SCALES, ValueScale

BASELINE_LABELS = {"calibration_mean": "Calibration-summary mean", "reference_mean": "Reference-item mean"}
ITEM_COLUMNS = ["item_key", "assessment", "grade", "item_id", "irt_model"]
POSTERIOR_COLUMNS = ["map", "map_probability", "eap", "median", "sd", "confidence", "probabilities"]
PAIRED_COLUMNS = ["parameter", "comparison", "items", "estimate", "se", "t", "df", "p_value", "ci_low", "ci_high", "p_holm"]


def holm(p_values: pd.Series) -> pd.Series:
    """Holm's step-down adjustment of a family of p-values; NaN stays NaN and is not counted."""
    ordered = p_values.dropna().sort_values()
    adjusted = (ordered * np.arange(len(ordered), 0, -1)).cummax().clip(upper=1)
    return adjusted.reindex(p_values.index)


def quadratic_weighted_kappa(first: np.ndarray, second: np.ndarray, n_categories: int) -> float:
    """Cohen's kappa with quadratic weights of two ratings on categories 1 to ``n_categories``; NaN if both use one
    category only."""
    first, second = np.asarray(first, dtype=int) - 1, np.asarray(second, dtype=int) - 1
    if len(set(first) | set(second)) < 2:
        return np.nan
    observed = np.zeros((n_categories, n_categories))
    np.add.at(observed, (first, second), 1)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    positions = np.arange(n_categories)
    weights = (positions[:, None] - positions[None, :]) ** 2
    return float(1 - (weights * observed).sum() / (weights * expected).sum())


def category_agreement(calibrated: np.ndarray, predicted: np.ndarray, n_categories: int) -> dict:
    """Exact and within-one-category agreement, and quadratic weighted kappa, of the predicted categories."""
    keep = ~np.isnan(predicted)
    calibrated, predicted = calibrated[keep].astype(int), predicted[keep].astype(int)
    if not len(calibrated):
        return {"exact": np.nan, "within_one": np.nan, "weighted_kappa": np.nan}
    distance = np.abs(calibrated - predicted)
    return {
        "exact": (distance == 0).mean(),
        "within_one": (distance <= 1).mean(),
        "weighted_kappa": quadratic_weighted_kappa(calibrated, predicted, n_categories),
    }


def _summarize(rows: pd.DataFrame) -> pd.Series:
    """Bias, spread, mean absolute error, RMSE, and correlations of one run's predictions of one parameter."""
    rows = rows.dropna(subset=["predicted"])
    error = rows["error"]
    return pd.Series(
        {
            "items": len(rows),
            "mean_error": error.mean(),
            "sd_error": error.std(),
            "mae": error.abs().mean(),
            "rmse": np.sqrt((error**2).mean()),
            "pearson": correlation(rows["predicted"], rows["calibrated"]),
            "spearman": correlation(rows["predicted"], rows["calibrated"], "spearman"),
        }
    )


@dataclass
class RunComparison:
    """Leave-one-out runs compared on the items that every run predicted, with the baselines as runs."""

    runs: dict[str, PredictionResults]
    """The results of each run by its name, such as ``{"jev_item": ..., "gemini": ...}``, in the order of the tables
    and graphs."""
    labels: dict[str, str] | None = None
    """A longer name of each run for graphs, such as "gemini-3.8-flash, with reference items"; by default its name."""
    parameters: Sequence[str] | None = None
    """The parameters compared; by default every parameter that every run predicted."""
    scales: Mapping[str, ValueScale] = DEFAULT_VALUE_SCALES
    """The value scales of the categorical comparison (``scale_agreement``), which must be those of the Jev runs."""
    reference_run: str | None = None
    """The run whose reference items give the reference-item mean; by default the first run that has reference items.
    Without such a run, that baseline is left out."""

    def __post_init__(self):
        if not self.runs:
            raise ValueError("Give at least one run")
        clashes = sorted(set(self.runs) & set(BASELINE_LABELS))
        if clashes:
            raise ValueError(f"Run names {clashes} are the names of baselines")
        present = [set(results.errors["parameter"].astype(str)) for results in self.runs.values()]
        common = set.intersection(*present)
        chosen = [name for name in PARAMETER_NAMES if name in common] if self.parameters is None else list(self.parameters)
        missing = sorted(set(chosen) - common)
        if missing or not chosen:
            raise ValueError(f"Not every run predicted {missing or 'any parameter'}")
        self.parameters = chosen
        self.labels = {run: (self.labels or {}).get(run, run) for run in self.runs}
        with_references = [run for run, results in self.runs.items() if results.errors["reference_mean"].notna().any()]
        if self.reference_run is None:
            self.reference_run = with_references[0] if with_references else None
        elif self.reference_run not in with_references:
            raise ValueError(f"{self.reference_run!r} is not a run with reference items: {with_references}")

    @property
    def run_names(self) -> list[str]:
        return list(self.runs)

    @property
    def baselines(self) -> list[str]:
        """The baselines compared: the calibration-summary mean, and the reference-item mean if a run has it."""
        return [name for name in BASELINE_LABELS if name == "calibration_mean" or self.reference_run is not None]

    def label(self, run: str) -> str:
        """The longer name of a run or baseline, for graphs."""
        return BASELINE_LABELS.get(run) or self.labels[run]

    @functools.cached_property
    def common_items(self) -> list[str]:
        """The items that every run predicted, in the order of the first run."""
        common = set.intersection(*(set(results.errors["item_key"]) for results in self.runs.values()))
        first = next(iter(self.runs.values())).errors["item_key"]
        return [key for key in dict.fromkeys(first) if key in common]

    @functools.cached_property
    def errors(self) -> pd.DataFrame:
        """One row per run (baselines included), item, and compared parameter, on the common items: the run's errors
        with ``error`` and ``abs_error``, the baselines, and for a Jev run the posterior of its answer.

        Raises if the runs disagree on a calibrated value or on the calibration-summary mean, which involve no
        retrieval or prediction."""
        pair = ["item_key", "parameter"]
        frames = []
        for run, results in self.runs.items():
            errors = results.errors.astype({"parameter": str})
            keep = errors["parameter"].isin(self.parameters) & errors["item_key"].isin(self.common_items)
            frames.append(errors[keep].assign(run=run))
        errors = pd.concat(frames, ignore_index=True)
        values = errors.groupby(pair)[["calibrated", "calibration_mean"]]
        spread = values.max() - values.min()
        if (spread.to_numpy() > 1e-9).any():
            disagreeing = spread[(spread > 1e-9).any(axis=1)].index[:5].tolist()
            raise ValueError(f"The runs disagree on calibrated values or the calibration-summary mean: {disagreeing}")
        if self.reference_run is not None:
            reference = frames[self.run_names.index(self.reference_run)].set_index(pair)["reference_mean"]
            errors["reference_mean"] = reference.reindex(pd.MultiIndex.from_frame(errors[pair])).to_numpy()
        else:
            errors["reference_mean"] = np.nan
        # The baselines as runs whose prediction is the baseline's value
        first = errors[errors["run"] == self.run_names[0]]
        baselines = [
            first.assign(run=baseline, predicted=first[baseline], **{name: np.nan for name in POSTERIOR_COLUMNS if name in first})
            for baseline in self.baselines
        ]
        errors = pd.concat([errors, *baselines], ignore_index=True)
        errors["run"] = pd.Categorical(errors["run"], categories=self.run_names + self.baselines, ordered=True)
        errors["parameter"] = pd.Categorical(errors["parameter"], categories=self.parameters, ordered=True)
        errors["error"] = errors["predicted"] - errors["calibrated"]
        errors["abs_error"] = errors["error"].abs()
        columns = [*ITEM_COLUMNS, "parameter", "run", "predicted", "calibrated", "error", "abs_error", *BASELINE_LABELS]
        others = [column for column in errors.columns if column not in columns]
        return errors[columns + others]

    def _runs_only(self) -> pd.DataFrame:
        return self.errors[self.errors["run"].isin(self.run_names)]

    def coverage(self) -> pd.DataFrame:
        """Per run: the items it was to predict, those it predicted, and those that every run predicted."""
        rows = {}
        for run, results in self.runs.items():
            predicted = results.errors["item_key"].nunique()
            rows[run] = {
                "items": results.settings.get("evaluation.items", predicted),
                "predicted": predicted,
                "in_every_run": len(self.common_items),
            }
        return pd.DataFrame(rows).T.rename_axis("run")

    def wide(self) -> pd.DataFrame:
        """One row per common item and compared parameter: the calibrated value, the baselines, and each run's
        prediction (``predicted (<run>)``) and error, predicted minus calibrated (``error (<run>)``)."""
        pair = ["item_key", "parameter"]
        runs = self._runs_only()
        first = runs[runs["run"] == self.run_names[0]]
        table = first[[*ITEM_COLUMNS, "parameter", "calibrated", *self.baselines]].reset_index(drop=True)
        predicted = runs.pivot(index=pair, columns="run", values="predicted")
        indexed = pd.MultiIndex.from_frame(table[pair])
        for run in self.run_names:
            table[f"predicted ({run})"] = predicted[run].reindex(indexed).to_numpy()
        for run in self.run_names:
            table[f"error ({run})"] = table[f"predicted ({run})"] - table["calibrated"]
        return table

    def mae(self, by: str | None = None) -> pd.DataFrame:
        """Mean absolute error of every run and baseline (one column each) per parameter, with the number of items;
        with ``by`` (such as "grade"), per value of ``by`` and parameter, and then for all items ("all")."""
        if by is None:
            return self._mae_table(self.errors, ["parameter"])
        groups = [by, "parameter"]
        return pd.concat(
            [self._mae_table(self.errors, groups), self._mae_table(self.errors.assign(**{by: "all"}), groups)]
        )

    def _mae_table(self, rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        table = rows.groupby([*by, "run"], observed=True)["abs_error"].mean().unstack("run")
        items = rows[rows["run"] == self.run_names[0]].groupby(by, observed=True).size().rename("items")
        return pd.concat([items, table[self.run_names + self.baselines]], axis=1)

    def summary(self) -> pd.DataFrame:
        """Per parameter and run (baselines included): the items, the bias (``mean_error``, predicted minus
        calibrated), the spread of the errors, the mean absolute error, the RMSE, and the Pearson and Spearman
        correlations of the predicted with the calibrated values.

        A baseline that is (almost) constant within a grade and model has a correlation near 0, and the
        calibration-summary mean, which leaves the item's parts out, is a little lower for an item with a higher value,
        which makes its Spearman correlation meaningless."""
        columns = ["predicted", "calibrated", "error"]
        summary = self.errors.groupby(["parameter", "run"], observed=True)[columns].apply(_summarize)
        return summary.astype({"items": int})

    def paired(self) -> pd.DataFrame:
        """Every pair of runs, and every run against each baseline, per parameter: the mean per-item difference of
        their absolute errors (first minus second, so a negative estimate means that the first predicts better), with
        its standard error, t statistic, 95% confidence interval, and p-value (``factorial.paired_t_test``), and the
        p-value Holm-adjusted over the comparisons of the parameter (``p_holm``)."""
        comparisons = [*itertools.combinations(self.run_names, 2), *itertools.product(self.run_names, self.baselines)]
        rows = []
        for parameter, group in self.errors.groupby("parameter", observed=True):
            absolute = group.pivot(index="item_key", columns="run", values="abs_error")
            for first, second in comparisons:
                contrasts = (absolute[first] - absolute[second]).dropna().to_numpy()
                rows.append({"parameter": parameter, "comparison": f"{first} − {second}"} | paired_t_test(contrasts))
        paired = pd.DataFrame(rows, columns=PAIRED_COLUMNS[:-1])
        paired["p_holm"] = paired.groupby("parameter")["p_value"].transform(holm)
        return paired.set_index(["parameter", "comparison"])

    @functools.cached_property
    def scale_choices(self) -> pd.DataFrame:
        """One row per run, common item, and parameter with a scale: the run's value on the scale and the value
        closest to the calibrated parameter, each with its category (its position on the scale, from 1).

        A Jev run's value is the most likely one of its answer, which must be a value of the scale; any other run's is
        the value closest to its prediction."""
        rows = self._runs_only()
        rows = rows[rows["parameter"].isin(list(self.scales))]
        choices = rows[[*ITEM_COLUMNS, "parameter", "run", "predicted", "calibrated"]].copy()
        mode = rows["map"] if "map" in rows else pd.Series(np.nan, index=rows.index)
        choices["category"] = np.nan
        choices["calibrated_category"] = np.nan
        for name, group in choices.groupby("parameter", observed=True):
            scale = self.scales[name]
            grid = np.asarray(scale.values)
            from_mode = mode.loc[group.index].notna()
            positions = scale.nearest(np.where(from_mode, mode.loc[group.index], group["predicted"]))
            off_scale = from_mode & ~np.isclose(grid[np.nan_to_num(positions).astype(int)], mode.loc[group.index])
            if off_scale.any():
                runs = sorted(set(group.loc[off_scale, "run"].astype(str)))
                raise ValueError(f"The {name} values of {runs} are not on the comparison's scale; pass their scales")
            choices.loc[group.index, "category"] = positions + 1
            choices.loc[group.index, "calibrated_category"] = scale.nearest(group["calibrated"]) + 1
        if choices["category"].isna().any():
            raise ValueError("Some predictions have no value on the scale")
        values = {name: np.asarray(scale.values) for name, scale in self.scales.items()}
        choices["value"] = [values[name][int(position) - 1] for name, position in zip(choices["parameter"], choices["category"])]
        choices["calibrated_value"] = [
            values[name][int(position) - 1] for name, position in zip(choices["parameter"], choices["calibrated_category"])
        ]
        return choices.reset_index(drop=True)

    def scale_agreement(self, by: str | None = None) -> pd.DataFrame:
        """Per parameter with a scale (and per value of ``by``, such as "grade"): for every run, how often its value
        is the calibrated parameter's closest value (``exact``), within one value of it (``within_one``), and the
        quadratic weighted kappa; and the same shares for the best constant value (``baseline_exact``,
        ``baseline_within_one``), the one that is exact, or within one, for the most items.

        The scales are uneven (the location's steps run from 0.2 to 1.0), so a step is not the same distance
        everywhere."""
        choices = self.scale_choices
        groups = ["parameter"] if by is None else ["parameter", by]
        table = {
            key if by is not None else key[0]: self._agreement(rows)
            for key, rows in choices.groupby(groups, observed=True)
        }
        names = ["parameter"] if by is None else ["parameter", by]
        return pd.DataFrame(table).T.rename_axis(names).astype({"items": int})

    def _agreement(self, rows: pd.DataFrame) -> pd.Series:
        n_categories = len(self.scales[str(rows["parameter"].iat[0])].values)
        calibrated = rows.loc[rows["run"] == self.run_names[0], "calibrated_category"].to_numpy(dtype=float)
        result = {"items": len(calibrated)}
        for run in self.run_names:
            run_rows = rows[rows["run"] == run]
            measures = category_agreement(
                run_rows["calibrated_category"].to_numpy(dtype=float), run_rows["category"].to_numpy(dtype=float), n_categories
            )
            result |= {f"{run}_{measure}": value for measure, value in measures.items()}
        constant = range(1, n_categories + 1)
        result["baseline_exact"] = max(np.mean(calibrated == category) for category in constant)
        result["baseline_within_one"] = max(np.mean(np.abs(calibrated - category) <= 1) for category in constant)
        return pd.Series(result)

    def value_spread(self) -> pd.DataFrame:
        """How often each value of a scale is picked, by the calibrated parameters (their closest value) and by every
        run: a predictor that ignores the item picks one value for almost every item."""
        choices = self.scale_choices
        first = choices[choices["run"] == self.run_names[0]]
        counts = {"calibrated": first.groupby("parameter", observed=True)["calibrated_value"].value_counts()}
        for run in self.run_names:
            counts[run] = choices[choices["run"] == run].groupby("parameter", observed=True)["value"].value_counts()
        spread = pd.concat(counts, axis=1).fillna(0).astype(int)
        spread.index.names = ["parameter", "value"]
        index = [(name, value) for name in self.parameters if name in self.scales for value in self.scales[name].values]
        return spread.reindex(pd.MultiIndex.from_tuples(index, names=["parameter", "value"]), fill_value=0)

    def posteriors(self) -> pd.DataFrame:
        """The answers of the Jev runs, one row per run, common item, and parameter: the point prediction, every
        estimator's value, the spread, Jev's confidence, and the probabilities of the scale's values."""
        rows = self._runs_only()
        columns = [*ITEM_COLUMNS, "parameter", "run", "predicted", "calibrated"]
        if "map" not in rows:
            return pd.DataFrame(columns=[*columns, *POSTERIOR_COLUMNS])
        rows = rows[rows["map"].notna()]
        return rows[[*columns, *[column for column in POSTERIOR_COLUMNS if column in rows]]].reset_index(drop=True)

    def cost(self) -> pd.DataFrame:
        """Per run, over all the items it predicted: the input and output tokens, the cost as the saved predictions
        recorded it (an LLM batch's at its batch price; NaN where none is recorded, as for a local model), and the
        median seconds of a request where recorded."""
        rows = {}
        for run, results in self.runs.items():
            predictions = results.predictions

            def total(*columns, frame=predictions):
                present = [column for column in columns if column in frame]
                return frame[present].sum(axis=1, min_count=1).sum() if present else np.nan

            seconds = predictions["seconds"].dropna() if "seconds" in predictions else pd.Series(dtype=float)
            cost = predictions["cost_usd"].sum(min_count=1) if "cost_usd" in predictions else np.nan
            rows[run] = {
                "items": len(predictions),
                "input_tokens": total("prompt_tokens", "input_tokens"),
                "output_tokens": total("completion_tokens", "output_tokens"),
                "cost_usd": cost,
                "cost_per_item_usd": cost / len(predictions) if len(predictions) else np.nan,
                "requests_timed": len(seconds),
                "median_seconds": seconds.median() if len(seconds) else np.nan,
            }
        return pd.DataFrame(rows).T.rename_axis("run")

    def predictions(self) -> pd.DataFrame:
        """One row per run and item: the runs' ``predictions`` without the prompts, Jev's states and questions, and
        the JSON of the reference items, which the saved prediction files hold (``prediction_file``)."""
        frames = [
            results.predictions.drop(columns=SAVED_FILE_COLUMNS, errors="ignore").assign(run=run)
            for run, results in self.runs.items()
        ]
        predictions = pd.concat(frames, ignore_index=True)
        return predictions[["run", *[column for column in predictions.columns if column != "run"]]]

    def settings(self) -> pd.DataFrame:
        """Every setting of each run, as JSON, one column per run."""
        names = list(dict.fromkeys(name for results in self.runs.values() for name in results.settings))
        return pd.DataFrame(
            {run: [json.dumps(results.settings.get(name), default=str) for name in names] for run, results in self.runs.items()},
            index=pd.Index(names, name="setting"),
        )

    def to_excel(self, path: str | Path) -> Path:
        """Export every table, one sheet each: ``predictions_wide``, ``prediction_errors``, ``jev_posteriors``,
        ``predictions``, ``coverage``, ``mae``, ``mae_by_grade``, ``summary``, ``paired_comparisons``,
        ``value_agreement``, ``value_agreement_by_grade``, ``value_spread``, ``value_choices``, ``cost``, and
        ``settings``.

        An Excel cell holds at most 32,767 characters, so a longer text, such as a long reasoning, is cut there; the
        saved prediction has it in full.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        has_scales = any(name in self.scales for name in self.parameters)
        with pd.ExcelWriter(path) as writer:
            self.wide().to_excel(writer, sheet_name="predictions_wide", index=False)
            self.errors.to_excel(writer, sheet_name="prediction_errors", index=False)
            self.posteriors().to_excel(writer, sheet_name="jev_posteriors", index=False)
            self.predictions().map(excel_value).to_excel(writer, sheet_name="predictions", index=False)
            self.coverage().to_excel(writer, sheet_name="coverage")
            self.mae().to_excel(writer, sheet_name="mae")
            self.mae("grade").to_excel(writer, sheet_name="mae_by_grade")
            self.summary().to_excel(writer, sheet_name="summary")
            self.paired().to_excel(writer, sheet_name="paired_comparisons")
            if has_scales:
                self.scale_agreement().to_excel(writer, sheet_name="value_agreement")
                self.scale_agreement("grade").to_excel(writer, sheet_name="value_agreement_by_grade")
                self.value_spread().to_excel(writer, sheet_name="value_spread")
                self.scale_choices.to_excel(writer, sheet_name="value_choices", index=False)
            self.cost().to_excel(writer, sheet_name="cost")
            self.settings().to_excel(writer, sheet_name="settings")
        return path
