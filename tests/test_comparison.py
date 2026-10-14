"""RunComparison: runs compared on the same items, with the baselines as runs, paired tests, and agreement on value
scales, on made-up runs.

Run with ``uv run python -m unittest discover tests``.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from timss_math.item_parameter_prediction import (
    PredictionResults,
    RunComparison,
    ValueScale,
    plots,
)
from timss_math.item_parameter_prediction.comparison import (
    holm,
    quadratic_weighted_kappa,
)

SCALES = {"slope": ValueScale(("0.5", "1.0", "1.5")), "location": ValueScale(("-1", "0", "1"))}
CALIBRATED = {
    "A": {"slope": 1.0, "location": 0.4},
    "B": {"slope": 0.6, "location": -0.8},
    "C": {"slope": 1.4, "location": 0.9},
    "D": {"slope": 1.1, "location": 0.1},
}
GRADES = {"A": 4, "B": 4, "C": 8, "D": 8}
# Each run's prediction of each item and parameter; Jev's are values of the scales (its most likely ones).
PREDICTED = {
    "llm": {"A": (1.1, 0.3), "B": (0.7, -0.4), "C": (1.2, 0.6), "D": (1.0, 0.2)},
    "jev": {"A": (1.0, 0.0), "B": (1.0, -1.0), "C": (1.5, 1.0), "D": (0.5, 0.0)},
}


def run(name, items="ABCD", references=True, shift=0.0) -> PredictionResults:
    rows, predictions = [], []
    for item in items:
        for position, parameter in enumerate(("slope", "location")):
            predicted = PREDICTED[name][item][position]
            row = {
                "item_key": f"TIMSS 2011/grade {GRADES[item]}/Mathematics/{item}",
                "assessment": "TIMSS 2011",
                "grade": GRADES[item],
                "item_id": item,
                "irt_model": "2PL",
                "parameter": parameter,
                "predicted": predicted,
                "calibrated": CALIBRATED[item][parameter] + shift,
                "calibration_mean": 0.9 if parameter == "slope" else 0.0,
                "reference_mean": (1.0 if parameter == "slope" else 0.2) if references else np.nan,
            }
            if name == "jev":
                row |= {"map": predicted, "map_probability": 0.6, "eap": predicted, "median": predicted, "sd": 0.3}
            rows.append(row)
        predictions.append(
            {
                "item_key": f"TIMSS 2011/grade {GRADES[item]}/Mathematics/{item}",
                "cost_usd": 0.01 if name == "llm" else 0.0001,
                "seconds": 2.0 if name == "llm" else 0.2,
                "user_prompt": "long",
                "state": "long",
            }
        )
    errors = pd.DataFrame(rows)
    errors.insert(10, "error", errors["predicted"] - errors["calibrated"])
    return PredictionResults(pd.DataFrame(predictions), errors, {"evaluation.items": 4, "evaluation.predictor": name})


class HelpersTest(unittest.TestCase):
    def test_holm(self):
        adjusted = holm(pd.Series([0.01, 0.04, 0.03, np.nan]))
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, np.nan])

    def test_quadratic_weighted_kappa(self):
        # scikit-learn's cohen_kappa_score(..., labels=[1, 2, 3, 4], weights="quadratic") gives 4 / 7.
        self.assertAlmostEqual(quadratic_weighted_kappa([1, 2, 3, 3, 2, 1], [1, 3, 3, 2, 2, 2], 4), 4 / 7)
        self.assertEqual(quadratic_weighted_kappa([1, 2, 3], [1, 2, 3], 3), 1.0)
        self.assertTrue(np.isnan(quadratic_weighted_kappa([2, 2], [2, 2], 3)))


class RunComparisonTest(unittest.TestCase):
    def setUp(self):
        self.comparison = RunComparison(
            {"jev": run("jev", references=False), "llm": run("llm", items="ABC")},
            labels={"llm": "An LLM, with reference items"},
            scales=SCALES,
        )

    def test_common_items_and_baselines(self):
        errors = self.comparison.errors
        self.assertEqual(len(self.comparison.common_items), 3)  # the LLM run has no prediction of D
        self.assertEqual(list(errors["run"].cat.categories), ["jev", "llm", "calibration_mean", "reference_mean"])
        self.assertEqual(self.comparison.reference_run, "llm")
        # Jev saw no reference items, so its reference-item mean comes from the LLM run.
        self.assertTrue((errors.loc[errors["run"] == "jev", "reference_mean"].notna()).all())
        baseline = errors[errors["run"] == "reference_mean"]
        np.testing.assert_allclose(baseline["predicted"], baseline["reference_mean"])
        self.assertTrue(baseline["map"].isna().all())
        self.assertEqual(self.comparison.label("llm"), "An LLM, with reference items")
        self.assertEqual(self.comparison.label("jev"), "jev")

    def test_disagreeing_calibrated_values_raise(self):
        comparison = RunComparison({"jev": run("jev"), "llm": run("llm", shift=0.1)}, scales=SCALES)
        with self.assertRaises(ValueError):
            comparison.errors  # noqa: B018

    def test_mae_and_paired(self):
        mae = self.comparison.mae()
        llm = self.comparison.errors.query("run == 'llm' and parameter == 'location'")
        self.assertAlmostEqual(mae.at["location", "llm"], llm["abs_error"].mean())
        self.assertEqual(mae.at["location", "items"], 3)
        self.assertIn("all", self.comparison.mae("grade").index.get_level_values("grade"))
        paired = self.comparison.paired()
        self.assertEqual(len(paired), 2 * (1 + 2 * 2))  # per parameter: one pair of runs, two runs × two baselines
        absolute = self.comparison.errors.query("parameter == 'slope'").pivot(index="item_key", columns="run", values="abs_error")
        self.assertAlmostEqual(paired.loc[("slope", "jev − llm"), "estimate"], (absolute["jev"] - absolute["llm"]).mean())
        self.assertTrue((paired["p_holm"] >= paired["p_value"]).all())

    def test_scale_choices(self):
        choices = self.comparison.scale_choices.set_index(["run", "item_id", "parameter"])
        # Jev's value is its most likely one; the LLM's is the value closest to its prediction.
        self.assertEqual(choices.loc[("jev", "B", "slope"), "value"], 1.0)
        self.assertEqual(choices.loc[("llm", "B", "location"), "value"], 0.0)
        self.assertEqual(choices.loc[("llm", "B", "location"), "calibrated_value"], -1.0)
        agreement = self.comparison.scale_agreement()
        exact = (choices.xs("jev")["category"] == choices.xs("jev")["calibrated_category"]).groupby("parameter").mean()
        self.assertAlmostEqual(agreement.at["slope", "jev_exact"], exact["slope"])
        self.assertIn("baseline_within_one", agreement)
        spread = self.comparison.value_spread()
        self.assertEqual(spread.loc[("slope", 1.0), "jev"], 2)

    def test_values_off_the_scale_raise(self):
        scales = {"slope": ValueScale(("0.4", "0.8", "1.2")), "location": SCALES["location"]}
        with self.assertRaisesRegex(ValueError, "not on the comparison's scale"):
            RunComparison({"jev": run("jev"), "llm": run("llm")}, scales=scales).scale_choices  # noqa: B018

    def test_cost_posteriors_and_export(self):
        cost = self.comparison.cost()
        self.assertAlmostEqual(cost.at["llm", "cost_usd"], 0.03)
        self.assertAlmostEqual(cost.at["jev", "median_seconds"], 0.2)
        self.assertEqual(set(self.comparison.posteriors()["run"]), {"jev"})
        self.assertNotIn("state", self.comparison.predictions())
        path = self.comparison.to_excel(Path(tempfile.mkdtemp()) / "comparison.xlsx")
        with pd.ExcelFile(path) as workbook:
            self.assertEqual(
                workbook.sheet_names,
                [
                    "predictions_wide",
                    "prediction_errors",
                    "jev_posteriors",
                    "predictions",
                    "coverage",
                    "mae",
                    "mae_by_grade",
                    "summary",
                    "paired_comparisons",
                    "value_agreement",
                    "value_agreement_by_grade",
                    "value_spread",
                    "value_choices",
                    "cost",
                    "settings",
                ],
            )

    def test_graphs(self):
        for draw in (
            plots.comparison_mae_by_grade,
            plots.comparison_predicted_vs_calibrated,
            plots.comparison_scale_agreement,
            plots.comparison_scale_confusion,
        ):
            with self.subTest(graph=draw.__name__):
                self.assertIsInstance(draw(self.comparison), Figure)

    def test_bad_runs(self):
        with self.assertRaises(ValueError):
            RunComparison({"calibration_mean": run("llm")})
        with self.assertRaises(ValueError):
            RunComparison({"jev": run("jev")}, parameters=["guessing"])
        with self.assertRaises(ValueError):
            RunComparison({"jev": run("jev", references=False), "llm": run("llm")}, reference_run="jev")


if __name__ == "__main__":
    unittest.main()
