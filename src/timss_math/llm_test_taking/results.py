"""The scores of the conditions on the IRT and TIMSS scales, and their workbook.

``ScoringResults`` holds:

- ``conditions``: one row per condition, with its settings, role, usage, and cost.
- ``overall``: one row per condition and calibration with scored items: the items scored, their points and the share
  of them earned, θ (EAP and MLE, with standard errors), and the same on the TIMSS reporting scale, with the highest
  benchmark that the EAP scale score reaches; ``mle_note`` says why an MLE is infinite.
- ``item_scores``: one row per condition and calibrated item with a score: the answer, the key or the scorer's category
  and reasoning, the points, the score in the scaling, the item's calibrated parameters (``PARAMETER_COLUMNS``) with
  their standard errors and source (``PARAMETER_DETAILS``), its place on the IRT scale (its location b) and on the
  TIMSS scale (A + B b), and the score that the condition's θ (EAP) predicts: the probability of the observed score,
  the expected score, and their difference (``residual``).
- ``item_scores_wide``: one row per item that any condition answered, in order of location within each calibration,
  with the same parameters and every condition's score.
- ``calibrations``: the calibrations with their TIMSS scale constants.

``summary`` gives each condition's share of the points over every calibration and the range of its TIMSS scale
scores, and ``score_table`` any value of ``overall`` by condition and calibration. ``to_excel`` writes the tables to a
workbook, which ``read_excel`` reads back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import excel_value
from ..items import STEP_NAMES
from .conditions import Condition
from .irt import ItemModel, estimate_theta
from .scales import Calibrations, benchmark

PARAMETER_COLUMNS = ["slope", "location", "guessing", *STEP_NAMES]
"""The calibrated parameters of each item in the item tables, empty where its model has none."""
PARAMETER_DETAILS = [f"{name}_se" for name in PARAMETER_COLUMNS] + ["source_item_id", "source_file", "source_location"]
"""The standard errors of the parameters, and where they come from."""
INFINITE_MLE = {"perfect score": np.inf, "zero score": -np.inf}
"""The infinite MLE of each ``mle_note`` that fixes its sign, which Excel cannot hold."""
OVERALL_COLUMNS = [
    "condition",
    "role",
    "calibration",
    "items",
    "calibration_items",
    "points",
    "max_points",
    "theta_eap",
    "theta_eap_se",
    "theta_mle",
    "theta_mle_se",
    "mle_note",
    "percent_of_points",
    "timss_eap",
    "timss_eap_se",
    "timss_mle",
    "timss_mle_se",
    "timss_benchmark",
]


def overall_scores(
    scores: pd.DataFrame,
    conditions: Mapping[str, Condition],
    calibrations: Calibrations,
    models: Mapping[str, ItemModel],
) -> pd.DataFrame:
    """One row per condition and calibration with scored items: the points earned, θ, and the TIMSS scale score."""
    rows = []
    scored = scores.dropna(subset=["score"]).assign(calibration=lambda frame: frame["item_key"].map(calibrations.of_item))
    for name, condition in conditions.items():
        for label, calibration in calibrations.items():
            group = scored[(scored["condition"] == name) & (scored["calibration"] == label)]
            if group.empty:
                continue
            row = {
                "condition": name,
                "role": condition.role,
                "calibration": label,
                "items": len(group),
                "calibration_items": len(calibration.item_keys),
                "points": int(group["score"].sum()),
                "max_points": int(group["scaled_points"].sum()),
                **estimate_theta([models[key] for key in group["item_key"]], group["score"].astype(int).tolist()),
            }
            row["percent_of_points"] = 100 * row["points"] / row["max_points"]
            row |= {
                "timss_eap": calibration.timss_score(row["theta_eap"]),
                "timss_eap_se": calibration.b * row["theta_eap_se"],
                "timss_mle": calibration.timss_score(row["theta_mle"]),
                "timss_mle_se": calibration.b * row["theta_mle_se"],
            }
            row["timss_benchmark"] = benchmark(row["timss_eap"], calibration.grade)
            rows.append(row)
    return pd.DataFrame(rows, columns=OVERALL_COLUMNS)


def item_scores_on_scale(
    scores: pd.DataFrame,
    overall: pd.DataFrame,
    parameters: pd.DataFrame,
    calibrations: Calibrations,
    models: Mapping[str, ItemModel],
) -> pd.DataFrame:
    """The item scores with each item's calibrated parameters, its place on the IRT scale (its location b, and A + B b
    on the TIMSS scale), the score that the condition's θ (EAP) predicts (the probability of the observed score, the
    expected score, and the residual), and the standard errors and source of its parameters."""
    frame = scores.dropna(subset=["score"]).assign(calibration=lambda frame: frame["item_key"].map(calibrations.of_item))
    frame = frame.merge(overall[["condition", "calibration", "theta_eap", "timss_eap"]], on=["condition", "calibration"])
    frame = frame.join(parameters[PARAMETER_COLUMNS], on="item_key")
    frame["location_timss"] = [
        calibrations[label].timss_score(b) for label, b in zip(frame["calibration"], frame["location"])
    ]
    frame["probability_of_score"] = [
        models[key].probabilities(theta)[0, int(score)]
        for key, theta, score in zip(frame["item_key"], frame["theta_eap"], frame["score"])
    ]
    frame["expected_score"] = [
        models[key].expected_score(theta)[0] for key, theta in zip(frame["item_key"], frame["theta_eap"])
    ]
    frame["residual"] = frame["score"].astype(float) - frame["expected_score"]
    return frame.join(parameters[PARAMETER_DETAILS], on="item_key")


def item_scores_wide(
    item_scores: pd.DataFrame, conditions: Mapping[str, Condition], parameters: pd.DataFrame, calibrations: Calibrations
) -> pd.DataFrame:
    """One row per item that any condition answered, with its place on the IRT and TIMSS scales and every condition's
    score, the items in order of their location within each calibration."""
    answered = [condition for condition in conditions if condition in set(item_scores["condition"])]
    wide = (
        parameters[["item_id", "irt_model", "scaled_points", *PARAMETER_COLUMNS, *PARAMETER_DETAILS]]
        .assign(calibration=pd.Categorical(parameters.index.map(calibrations.of_item), categories=list(calibrations)))
        .join(item_scores.pivot(index="item_key", columns="condition", values="score")[answered], how="inner")
        .sort_values(["calibration", "location"])
    )
    location_timss = [calibrations[label].timss_score(b) for label, b in zip(wide["calibration"], wide["location"])]
    wide.insert(wide.columns.get_loc("location") + 1, "location_timss", location_timss)
    return wide[["calibration", *wide.columns.drop("calibration")]]


@dataclass
class ScoringResults:
    """The scores of the conditions: their usage, abilities by calibration, item scores, and the calibrations."""

    conditions: pd.DataFrame
    """One row per condition (the index), in the order of the analysis."""
    overall: pd.DataFrame
    """One row per condition and calibration with scored items."""
    item_scores: pd.DataFrame
    """One row per condition and scored item."""
    item_scores_wide: pd.DataFrame
    """One row per item (the index), with every condition's score."""
    calibrations: pd.DataFrame
    """One row per calibration (the index, its label), with its TIMSS scale constants."""

    def summary(self) -> pd.DataFrame:
        """One row per condition with scored items: its role, the items scored and the points earned over every
        calibration, their share, and the lowest and highest TIMSS scale score (EAP) of its calibrations."""
        summary = self.overall.groupby("condition", sort=False).agg(
            role=("role", "first"),
            items=("items", "sum"),
            points=("points", "sum"),
            max_points=("max_points", "sum"),
            lowest_timss_eap=("timss_eap", "min"),
            highest_timss_eap=("timss_eap", "max"),
        )
        summary.insert(4, "percent_of_points", 100 * summary["points"] / summary["max_points"])
        return summary.reindex([condition for condition in self.conditions.index if condition in summary.index])

    def score_table(self, value: Callable[[tuple], object]) -> pd.DataFrame:
        """One cell per condition and calibration, from a function of an ``overall`` row (a named tuple), such as
        ``lambda row: f"{row.timss_eap:.0f}"``."""
        table = self.overall.assign(cell=[value(row) for row in self.overall.itertuples()]).pivot(
            index="condition", columns="calibration", values="cell"
        )
        return table.reindex(
            index=[condition for condition in self.conditions.index if condition in set(self.overall["condition"])],
            columns=[label for label in self.calibrations.index if label in set(self.overall["calibration"])],
        )

    def to_excel(self, path: str | Path) -> Path:
        """Write the conditions, the overall scores (and ``theta_eap`` and ``timss_eap`` as tables), the item scores,
        long and wide, and the calibrations. Excel has no infinity, so an infinite MLE is left empty and ``mle_note``
        says why; a text longer than an Excel cell holds is cut there."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        overall = self.overall
        with pd.ExcelWriter(path) as writer:
            self.conditions.reset_index().to_excel(writer, sheet_name="conditions", index=False)
            overall.replace([np.inf, -np.inf], np.nan).to_excel(writer, sheet_name="overall_scores", index=False)
            for value in ("theta_eap", "timss_eap"):
                overall.pivot(index="condition", columns="calibration", values=value).to_excel(writer, sheet_name=value)
            self.item_scores.map(excel_value).to_excel(writer, sheet_name="item_scores", index=False)
            self.item_scores_wide.to_excel(writer, sheet_name="item_scores_wide")
            self.calibrations.to_excel(writer, sheet_name="calibrations")
        return path

    @classmethod
    def read_excel(cls, path: str | Path) -> ScoringResults:
        """Read a workbook written by ``to_excel``, such as ``results/timss_math_llm_answers_scored.xlsx``. The infinite
        MLEs of a perfect or a zero score come back as ±inf; an MLE beyond ±6 stays empty, as in the workbook."""
        with pd.ExcelFile(path) as workbook:
            conditions = workbook.parse("conditions", index_col="condition")
            overall = workbook.parse("overall_scores")
            item_scores = workbook.parse("item_scores")
            wide = workbook.parse("item_scores_wide", index_col="item_key")
            calibrations = workbook.parse("calibrations", index_col="label")
        infinite = overall["mle_note"].map(INFINITE_MLE)
        for column in ("theta_mle", "timss_mle"):
            overall[column] = overall[column].where(infinite.isna(), infinite)
        # An empty answer, which earns no points, is an empty cell.
        item_scores = item_scores.astype({"points": "Int64", "score": "Int64"}).fillna({"answer": ""})
        wide["calibration"] = pd.Categorical(wide["calibration"], categories=list(calibrations.index))
        score_columns = [column for column in wide.columns if column in conditions.index]
        wide[score_columns] = wide[score_columns].astype("Int64")
        return cls(conditions, overall, item_scores, wide, calibrations)
