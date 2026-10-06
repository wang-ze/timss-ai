"""Factorial experiments: leave-one-out runs under every combination of the levels of a few settings.

A factorial experiment crosses factors, such as the embedding model and the LLM, each with a few levels, and makes one
leave-one-out run (``LeaveOneOut``) per combination of levels, a condition, on the same items.
``predict_item_parameters_compare.ipynb`` crosses two embedding models with two LLMs. ``FactorialResults`` combines
the ``PredictionResults`` of the conditions:

- ``errors``: one row per condition, item, and predicted parameter, with a column per factor, the absolute error, and
  whether every condition predicted the item (``in_all_conditions``).
- ``predictions``: one row per condition and item, the runs' ``predictions`` without the prompts and the JSON of the
  reference items, which the saved prediction files hold (``prediction_file``).
- ``wide()``: one row per item and predicted parameter, with the calibrated value and the prediction and error of each
  condition side by side.
- ``coverage()``: the items each condition predicted, and those that every condition predicted.
- ``mae()``, ``summary()``: the mean absolute error, and the full error summary, of each condition and parameter.
- ``effects()``: for factors of two levels, the main effects and interactions on the absolute error, each with a
  paired t-test over the items.
- ``to_excel``: all of these in one workbook.

``mae``, ``summary``, and ``effects`` use only the items that every condition predicted, so that the conditions are
compared on the same items.
"""

from __future__ import annotations

import functools
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .evaluation import PredictionResults, excel_value, summarize_errors
from .items import PARAMETER_NAMES

ITEM_COLUMNS = ["item_key", "assessment", "grade", "item_id", "irt_model"]
# Long texts that the saved prediction files hold in full; the workbook names the file of each prediction instead.
SAVED_FILE_COLUMNS = ["system_prompt", "user_prompt", "similar_items"]
BASELINE_COLUMN = "calibration-summary mean"
EFFECT_COLUMNS = [
    "parameter", "effect", "comparison", "items", "estimate", "se", "t", "df", "p_value", "ci_low", "ci_high"
]


@dataclass
class FactorialResults:
    """The leave-one-out results of every condition of a factorial experiment."""

    factors: dict[str, list[str]]
    """Each factor's levels, such as ``{"embedding_model": ["gemini-embedding-001", "bge-small-en-v1.5"], "llm":
    ["gemini-3.8-flash", "llama3.2"]}``. For ``effects``, the first level of each factor is the reference."""
    runs: dict[tuple[str, ...], PredictionResults]
    """The results of each condition, keyed by its levels in the order of ``factors``."""

    def __post_init__(self):
        for factor, levels in self.factors.items():
            if len(levels) < 2 or len(set(levels)) != len(levels):
                raise ValueError(f"Factor {factor!r} needs at least two distinct levels, not {levels}")
        conditions = set(self.conditions)
        if set(self.runs) != conditions:
            raise ValueError(
                f"Give one run per condition; missing: {sorted(conditions - set(self.runs))}, "
                f"not a condition: {sorted(set(self.runs) - conditions)}"
            )
        # The conditions predict the same calibrated items, so the calibrated values, and the calibration-summary
        # baseline, which involves no retrieval or LLM, must agree.
        values = self.errors.groupby(["item_key", "parameter"], observed=True)[["calibrated", "calibration_mean"]]
        disagreeing = values.nunique(dropna=False).max(axis=1) > 1
        if disagreeing.any():
            raise ValueError(f"The runs disagree on calibrated values: {disagreeing[disagreeing].index.tolist()[:5]}")

    @property
    def conditions(self) -> list[tuple[str, ...]]:
        """Every combination of levels, the first factor's levels varying slowest."""
        return list(itertools.product(*self.factors.values()))

    @staticmethod
    def label(condition: tuple[str, ...]) -> str:
        """The name of a condition: its levels, such as "gemini-embedding-001, llama3.2"."""
        return ", ".join(condition)

    @functools.cached_property
    def item_order(self) -> list[str]:
        """Every item that any condition predicted, in the order of the runs (the item bank's order)."""
        return list(dict.fromkeys(key for run in self.runs.values() for key in run.errors["item_key"]))

    @functools.cached_property
    def common_items(self) -> list[str]:
        """The items that every condition predicted."""
        common = set.intersection(*(set(run.errors["item_key"]) for run in self.runs.values()))
        return [key for key in self.item_order if key in common]

    def _with_factors(self, frame: pd.DataFrame, condition: tuple[str, ...]) -> pd.DataFrame:
        """``frame`` with a first column per factor and the condition's label."""
        levels = dict(zip(self.factors, condition)) | {"condition": self.label(condition)}
        return pd.concat([pd.DataFrame(levels, index=frame.index), frame], axis=1)

    def _ordered(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The factor and condition columns as categories in the order of their levels."""
        labels = [self.label(condition) for condition in self.conditions]
        categories = self.factors | {"condition": labels}
        return frame.astype({name: pd.CategoricalDtype(levels, ordered=True) for name, levels in categories.items()})

    @functools.cached_property
    def errors(self) -> pd.DataFrame:
        """One row per condition, item, and predicted parameter: the runs' ``errors`` with a column per factor, the
        condition, ``abs_error``, and ``in_all_conditions``."""
        frames = [self._with_factors(self.runs[condition].errors, condition) for condition in self.conditions]
        errors = self._ordered(pd.concat(frames, ignore_index=True))
        errors["parameter"] = pd.Categorical(errors["parameter"], categories=PARAMETER_NAMES, ordered=True)
        errors["abs_error"] = errors["error"].abs()
        errors["in_all_conditions"] = errors["item_key"].isin(self.common_items)
        return errors

    @functools.cached_property
    def predictions(self) -> pd.DataFrame:
        """One row per condition and item: the runs' ``predictions`` (each parameter predicted and calibrated with its
        error, and the LLM's reasoning and usage) without the prompts and the JSON of the reference items."""
        frames = []
        for condition in self.conditions:
            predictions = self.runs[condition].predictions.drop(columns=SAVED_FILE_COLUMNS, errors="ignore")
            frames.append(self._with_factors(predictions, condition))
        return self._ordered(pd.concat(frames, ignore_index=True))

    def wide(self) -> pd.DataFrame:
        """One row per item and predicted parameter, for every item that any condition predicted: the item, the
        calibrated value and the calibration-summary baseline, whether every condition predicted the item, and then
        each condition's prediction (``predicted (<condition>)``) and error, predicted minus calibrated
        (``error (<condition>)``), blank where a condition has no prediction."""
        errors = self.errors.assign(parameter=self.errors["parameter"].astype(str))
        index = ["item_key", "parameter"]
        columns = [*ITEM_COLUMNS, "parameter", "calibrated", "calibration_mean", "in_all_conditions"]
        table = errors.drop_duplicates(index)[columns].set_index(index)
        condition_columns = []
        for value in ("predicted", "error"):
            for condition in self.conditions:
                label = self.label(condition)
                condition_columns.append(f"{value} ({label})")
                table[condition_columns[-1]] = errors[errors["condition"] == label].set_index(index)[value]
        positions = {key: position for position, key in enumerate(self.item_order)}
        table = table.reset_index()[[*columns, *condition_columns]].sort_values(
            index, key=lambda column: column.map(positions if column.name == "item_key" else PARAMETER_NAMES.index)
        )
        table["parameter"] = pd.Categorical(table["parameter"], categories=PARAMETER_NAMES, ordered=True)
        return table.reset_index(drop=True)

    def coverage(self) -> pd.DataFrame:
        """Per condition: the items it was to predict, those it predicted, those it has no prediction of yet, and
        those that every condition predicted."""
        rows = []
        for condition in self.conditions:
            run = self.runs[condition]
            predicted = run.errors["item_key"].nunique()
            items = run.settings.get("evaluation.items", predicted)
            counts = {"items": items, "predicted": predicted, "missing": items - predicted}
            rows.append(dict(zip(self.factors, condition)) | counts | {"in_all_conditions": len(self.common_items)})
        return pd.DataFrame(rows)

    @property
    def _compared(self) -> pd.DataFrame:
        """The errors of the items that every condition predicted."""
        return self.errors[self.errors["in_all_conditions"]]

    def mae(self) -> pd.DataFrame:
        """Mean absolute error of each condition (one column each) and of the calibration-summary baseline, per
        parameter, on the items that every condition predicted."""
        rows = self._compared
        labels = [self.label(condition) for condition in self.conditions]
        first = rows[rows["condition"] == labels[0]]
        table = pd.DataFrame({"items": first.groupby("parameter", observed=True).size()})
        for label in labels:
            condition_rows = rows[rows["condition"] == label]
            table[label] = condition_rows.groupby("parameter", observed=True)["abs_error"].mean()
        baseline_errors = (first["calibration_mean"] - first["calibrated"]).abs()
        table[BASELINE_COLUMN] = baseline_errors.groupby(first["parameter"], observed=True).mean()
        return table

    def summary(self) -> pd.DataFrame:
        """The error summary of ``PredictionResults.summary`` (bias, spread, mean absolute error, RMSE, correlation, and
        the baselines' mean absolute errors) per condition and parameter, on the items that every condition
        predicted."""
        columns = ["predicted", "calibrated", "calibration_mean", "reference_mean"]
        rows = self._compared
        if rows.empty:
            return pd.DataFrame(columns=summarize_errors(rows[columns]).index)
        summary = rows.groupby([*self.factors, "parameter"], observed=True)[columns].apply(summarize_errors)
        return summary.astype({"items": int})

    def effects(self) -> pd.DataFrame:
        """Main effects and interactions of factors of two levels on the absolute error, per parameter, on the items
        that every condition predicted (none while no item has a prediction in every condition).

        The main effect of a factor is the mean absolute error at its second level minus that at its first,
        averaged over the levels of the other factors, so a negative effect means that the second level predicts
        better. The interaction of two factors is the main effect of one at the second level of the other minus its
        main effect at the first level (a difference of differences), averaged over any other factors; likewise for
        more factors.

        Every item's parameter has an absolute error in every condition, so each effect is computed per item, as a
        contrast of its absolute errors, and the mean of the contrasts is the effect on the mean absolute error. The
        standard error, t statistic, 95% confidence interval, and two-sided p-value come from a paired (one-sample)
        t-test of the contrasts, which for two factors is the F-test of a repeated-measures ANOVA (F = t²). The test
        treats the items as independent, although the parts of a multi-part item, and the items of one task, are not.
        """
        if any(len(levels) != 2 for levels in self.factors.values()):
            raise ValueError("Effects need factors of two levels")
        if not self.common_items:
            return pd.DataFrame(columns=EFFECT_COLUMNS)
        rows = self._compared.assign(
            parameter=self._compared["parameter"].astype(str), condition=self._compared["condition"].astype(str)
        )
        labels = [self.label(condition) for condition in self.conditions]
        table = rows.set_index(["parameter", "item_key", "condition"])["abs_error"].unstack("condition")[labels]
        names, level_pairs = list(self.factors), list(self.factors.values())
        signs = np.array(
            [[1 if level == levels[1] else -1 for level, levels in zip(condition, level_pairs)] for condition in self.conditions]
        )
        results = []
        for size in range(1, len(names) + 1):
            for term in itertools.combinations(range(len(names)), size):
                weights = signs[:, list(term)].prod(axis=1) / 2 ** (len(names) - size)
                contrasts = pd.Series(table.to_numpy() @ weights, index=table.index)
                for parameter, values in contrasts.groupby(level="parameter"):
                    results.append(
                        {
                            "parameter": parameter,
                            "effect": " × ".join(names[factor] for factor in term),
                            "comparison": _comparison(names, level_pairs, term),
                            **paired_t_test(values.to_numpy()),
                        }
                    )
        effects = pd.DataFrame(results, columns=EFFECT_COLUMNS)
        effects["parameter"] = pd.Categorical(effects["parameter"], categories=PARAMETER_NAMES, ordered=True)
        return effects.sort_values("parameter", kind="stable").reset_index(drop=True)

    def settings(self) -> pd.DataFrame:
        """Every setting of each condition's run (embedding, index, retrieval, LLM, and evaluation), as JSON, one
        column per condition."""
        names = list(dict.fromkeys(name for run in self.runs.values() for name in run.settings))
        return pd.DataFrame(
            {
                self.label(condition): [
                    json.dumps(self.runs[condition].settings.get(name), default=str) for name in names
                ]
                for condition in self.conditions
            },
            index=pd.Index(names, name="setting"),
        )

    def to_excel(self, path: str | Path) -> Path:
        """Export every table: ``predictions_wide`` (``wide``), ``prediction_errors`` (``errors``), ``predictions``,
        ``coverage``, ``mae``, ``effects`` (if every factor has two levels), ``summary``, and ``settings``.

        An Excel cell holds at most 32,767 characters, so a longer text, such as a long reasoning, is cut there; the
        saved prediction has it in full.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path) as writer:
            self.wide().to_excel(writer, sheet_name="predictions_wide", index=False)
            self.errors.to_excel(writer, sheet_name="prediction_errors", index=False)
            self.predictions.map(excel_value).to_excel(writer, sheet_name="predictions", index=False)
            self.coverage().to_excel(writer, sheet_name="coverage", index=False)
            self.mae().reset_index().to_excel(writer, sheet_name="mae", index=False)
            if all(len(levels) == 2 for levels in self.factors.values()):
                self.effects().to_excel(writer, sheet_name="effects", index=False)
            self.summary().reset_index().to_excel(writer, sheet_name="summary", index=False)
            self.settings().reset_index().to_excel(writer, sheet_name="settings", index=False)
        return path


def _comparison(names: list[str], level_pairs: list[list[str]], term: tuple[int, ...]) -> str:
    """What an effect compares: "B − A" for a main effect, a difference of differences for an interaction."""
    if len(term) == 1:
        first, second = level_pairs[term[0]]
        return f"{second} − {first}"
    if len(term) == 2:
        moderator, factor = term
        first, second = level_pairs[moderator]
        return f"{names[factor]} effect with {second} − {names[factor]} effect with {first}"
    return "interaction contrast of " + ", ".join(names[factor] for factor in term)


def paired_t_test(contrasts: np.ndarray) -> dict:
    """Mean of per-item contrasts with its standard error, t statistic, degrees of freedom, two-sided p-value, and
    95% confidence interval."""
    items = len(contrasts)
    estimate = float(np.mean(contrasts)) if items else np.nan
    if items < 2:
        return {"items": items, "estimate": estimate} | dict.fromkeys(["se", "t", "df", "p_value", "ci_low", "ci_high"], np.nan)
    se = float(np.std(contrasts, ddof=1)) / math.sqrt(items)
    df = items - 1
    t = estimate / se if se > 0 else np.nan
    half_width = float(stats.t.ppf(0.975, df)) * se
    return {
        "items": items,
        "estimate": estimate,
        "se": se,
        "t": t,
        "df": df,
        "p_value": float(2 * stats.t.sf(abs(t), df)) if se > 0 else np.nan,
        "ci_low": estimate - half_width,
        "ci_high": estimate + half_width,
    }
