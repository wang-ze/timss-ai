"""FactorialResults: the tables of a factorial experiment and the effects on the absolute error, on made-up runs.

Run with ``uv run python -m unittest discover tests``.
"""

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from timss_math.item_parameter_prediction import (
    FactorialResults,
    PredictionResults,
    plots,
)

FACTORS = {"embedding_model": ["gemini-embedding-001", "bge-small-en-v1.5"], "llm": ["gemini-3.8-flash", "llama3.2"]}
CONDITIONS = list(itertools.product(*FACTORS.values()))
# Calibrated location and slope of three items, and each condition's error of each (predicted minus calibrated).
CALIBRATED = {"A": {"slope": 1.0, "location": 0.5}, "B": {"slope": 0.8, "location": -0.2}, "C": {"slope": 1.2, "location": 1.1}}
LOCATION_ERRORS = {  # condition -> item -> error
    CONDITIONS[0]: {"A": 0.1, "B": -0.3, "C": 0.2},
    CONDITIONS[1]: {"A": 0.5, "B": 0.4, "C": -0.9},
    CONDITIONS[2]: {"A": -0.2, "B": 0.1, "C": 0.3},
    CONDITIONS[3]: {"A": 0.6, "B": -0.7, "C": 0.4},
}


def run(condition, items=("A", "B", "C"), calibrated_shift=0.0) -> PredictionResults:
    rows, predictions = [], []
    for item in items:
        for parameter in ("slope", "location"):
            error = LOCATION_ERRORS[condition][item] if parameter == "location" else 0.1 * (CONDITIONS.index(condition) + 1)
            calibrated = CALIBRATED[item][parameter] + calibrated_shift
            rows.append(
                {
                    "item_key": f"TIMSS 2011/grade 4/Mathematics/{item}",
                    "assessment": "TIMSS 2011",
                    "grade": 4,
                    "item_id": item,
                    "irt_model": "2PL",
                    "parameter": parameter,
                    "predicted": calibrated + error,
                    "calibrated": calibrated,
                    "calibration_mean": 0.6 if parameter == "location" else 1.0,
                    "reference_mean": 0.4,
                    "error": error,
                }
            )
        predictions.append(
            {"item_key": f"TIMSS 2011/grade 4/Mathematics/{item}", "reasoning": "why", "system_prompt": "long", "user_prompt": "long"}
        )
    errors = pd.DataFrame(rows)
    errors["error"] = errors["predicted"] - errors["calibrated"]
    errors["parameter"] = pd.Categorical(errors["parameter"], categories=["slope", "location", "guessing"], ordered=True)
    settings = {"llm.model": condition[1], "embedding.model": condition[0], "evaluation.items": 3}
    return PredictionResults(pd.DataFrame(predictions), errors, settings)


def absolute_location_errors(condition) -> np.ndarray:
    return np.abs([LOCATION_ERRORS[condition][item] for item in "ABC"])


class FactorialResultsTest(unittest.TestCase):
    def setUp(self):
        self.results = FactorialResults(FACTORS, {condition: run(condition) for condition in CONDITIONS})

    def test_effects_are_contrasts_of_the_absolute_errors(self):
        y11, y12, y21, y22 = map(absolute_location_errors, CONDITIONS)
        effects = self.results.effects().query("parameter == 'location'").set_index("effect")
        expected = {
            "embedding_model": (y21 + y22 - y11 - y12) / 2,
            "llm": (y12 + y22 - y11 - y21) / 2,
            "embedding_model × llm": (y22 - y21) - (y12 - y11),
        }
        self.assertEqual(list(effects.index), list(expected))
        for name, contrasts in expected.items():
            test = stats.ttest_1samp(contrasts, 0)
            self.assertAlmostEqual(effects.at[name, "estimate"], contrasts.mean())
            self.assertAlmostEqual(effects.at[name, "t"], test.statistic)
            self.assertAlmostEqual(effects.at[name, "p_value"], test.pvalue)
            self.assertEqual(effects.at[name, "items"], 3)
            self.assertLess(effects.at[name, "ci_low"], contrasts.mean())
        self.assertEqual(effects.at["llm", "comparison"], "llama3.2 − gemini-3.8-flash")
        # The main effect of the embedding model is the difference of its levels' mean absolute errors.
        mae = self.results.mae().loc["location"]
        marginal = {
            level: np.mean([mae[FactorialResults.label(condition)] for condition in CONDITIONS if condition[0] == level])
            for level in FACTORS["embedding_model"]
        }
        self.assertAlmostEqual(effects.at["embedding_model", "estimate"], marginal["bge-small-en-v1.5"] - marginal["gemini-embedding-001"])

    def test_comparisons_use_only_items_that_every_condition_predicted(self):
        runs = {condition: run(condition) for condition in CONDITIONS}
        runs[CONDITIONS[3]] = run(CONDITIONS[3], items=("A", "B"))
        results = FactorialResults(FACTORS, runs)
        self.assertEqual([key[-1] for key in results.common_items], ["A", "B"])
        self.assertEqual(results.mae()["items"].tolist(), [2, 2])
        self.assertEqual(set(results.effects()["items"]), {2})
        coverage = results.coverage().set_index("llm")
        self.assertEqual(coverage["missing"].tolist(), [0, 0, 0, 1])
        wide = results.wide()
        self.assertEqual(len(wide), 6)  # every item that any condition predicted, with two parameters each
        missing = wide[wide["item_id"] == "C"]
        self.assertFalse(missing["in_all_conditions"].any())
        self.assertTrue(missing[f"predicted ({FactorialResults.label(CONDITIONS[3])})"].isna().all())
        self.assertAlmostEqual(
            missing.set_index("parameter").at["location", f"error ({FactorialResults.label(CONDITIONS[1])})"], -0.9
        )

    def test_condition_without_predictions(self):
        runs = {condition: run(condition) for condition in CONDITIONS}
        empty = run(CONDITIONS[3])  # as LeaveOneOut.results() before any prediction is saved
        runs[CONDITIONS[3]] = PredictionResults(pd.DataFrame([]), empty.errors.iloc[:0], empty.settings)
        results = FactorialResults(FACTORS, runs)
        self.assertEqual(results.common_items, [])
        self.assertEqual(results.coverage()["predicted"].tolist(), [3, 3, 3, 0])
        self.assertTrue(results.wide()[f"predicted ({FactorialResults.label(CONDITIONS[3])})"].isna().all())
        self.assertTrue(results.mae().empty and results.summary().empty and results.effects().empty)
        with (
            tempfile.TemporaryDirectory() as directory,
            pd.ExcelFile(results.to_excel(Path(directory) / "compare.xlsx")) as workbook,
        ):
            self.assertEqual(len(workbook.parse("predictions")), 9)
            self.assertTrue(workbook.parse("effects").empty)

    def test_rejects_missing_conditions_and_disagreeing_calibrations(self):
        with self.assertRaisesRegex(ValueError, "one run per condition"):
            FactorialResults(FACTORS, {condition: run(condition) for condition in CONDITIONS[:3]})
        runs = {condition: run(condition) for condition in CONDITIONS}
        runs[CONDITIONS[2]] = run(CONDITIONS[2], calibrated_shift=0.01)
        with self.assertRaisesRegex(ValueError, "disagree"):
            FactorialResults(FACTORS, runs)

    def test_excel_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.results.to_excel(Path(directory) / "compare.xlsx")
            with pd.ExcelFile(path) as workbook:
                self.assertEqual(
                    workbook.sheet_names,
                    ["predictions_wide", "prediction_errors", "predictions", "coverage", "mae", "effects", "summary", "settings"],
                )
                predictions = workbook.parse("predictions")
                wide = workbook.parse("predictions_wide")
        self.assertNotIn("system_prompt", predictions.columns)
        self.assertEqual(predictions.columns[:3].tolist(), ["embedding_model", "llm", "condition"])
        self.assertEqual(len(predictions), 12)
        self.assertEqual(len(wide), 6)
        np.testing.assert_allclose(
            wide[f"error ({FactorialResults.label(CONDITIONS[0])})"], self.results.wide()[f"error ({FactorialResults.label(CONDITIONS[0])})"]
        )

    def test_plot(self):
        figure = plots.factorial_mae(self.results.errors, x="llm", lines="embedding_model")
        self.assertIsInstance(figure, Figure)


if __name__ == "__main__":
    unittest.main()
