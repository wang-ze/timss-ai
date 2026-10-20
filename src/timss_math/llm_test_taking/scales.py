"""The calibrations of the item parameters, and the TIMSS reporting scale of each.

Every θ goes on the TIMSS reporting scale, scale score = A + B θ, with the linear transformation constants of its
calibration, in ``timss_math_scale_transformation_constants.csv`` of the package's ``data`` directory (a copy of
``notebooks/data/``'s): one A and B per plausible value, averaged over the five here.
``notebooks/timss_math_scale_transformation_constants.md`` documents each source:

- Grade 4 and grade 8, 2007-2011: Exhibits 13 and 14 of "Scaling the TIMSS and PIRLS 2011 Achievement Data" (Methods
  and Procedures in TIMSS and PIRLS 2011), equal to ``T11_G<G>_Transform.xlsx``.
- Grade 4 and grade 8, 2003-2007: Exhibit 11.15 of the TIMSS 2007 Technical Report (the 2007 proficiency scores).
- Advanced, 1995-2008: Exhibit 8.22 of the TIMSS Advanced 2008 Technical Report.
- Grade 4, 1995-2003, and grade 8, 1999-2003: not published (the TIMSS 2003 Technical Report describes the
  transformation but gives no constants), so derived by Stocking-Lord linking of the 2003 items to the 2003-2007
  calibration and the bridge constants of Exhibit 11.14 of the TIMSS 2007 Technical Report; the same procedure
  reproduces the published 2007 constants within 1.7 scale points from θ = -2 to 2. Scale scores of these two
  calibrations are therefore estimates, good to a few scale points.

The TIMSS International Benchmarks (Advanced 625, High 550, Intermediate 475, and Low 400, the last not for TIMSS
Advanced) place each scale score (``benchmark``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..items import GRADE_SHORT_NAMES, PACKAGE_DATA_DIR

SCALE_CONSTANTS_FILE = "timss_math_scale_transformation_constants.csv"
GRADE_ROWS = [("Grade 4", 4), ("Grade 8", 8), ("Advanced", 12)]
"""The short name and grade of each grade's calibrations, in the order of the tables and graphs."""
BENCHMARKS = [(625, "Advanced"), (550, "High"), (475, "Intermediate"), (400, "Low")]


def calibration_label(calibration: str, grade: int) -> str:
    """A short name of a calibration, such as "Grade 4, 2007-2011", from its first and last years."""
    years = re.findall(r"\b(?:19|20)\d\d\b", calibration)
    return f"{GRADE_SHORT_NAMES[grade]}, {years[0]}-{years[-1]}"


def benchmark(scale_score: float, grade: int) -> str:
    """The highest TIMSS International Benchmark that a scale score reaches (TIMSS Advanced has no Low benchmark)."""
    for cut, name in BENCHMARKS:
        if scale_score >= cut and not (grade == 12 and name == "Low"):
            return f"{name} ({cut})"
    return "below the lowest benchmark"


@dataclass(frozen=True)
class Calibration:
    """A calibration of the item parameters, its items, and its TIMSS reporting scale."""

    label: str
    """A short name, such as "Grade 4, 2007-2011"."""
    name: str
    """The calibration as the item parameters name it."""
    grade: int
    item_keys: tuple[str, ...]
    a: float
    """A of scale score = A + B θ, averaged over the five plausible values."""
    b: float
    constants: str
    """How the constants were found: "published" or "derived"."""

    def timss_score(self, theta):
        """θ on the TIMSS reporting scale."""
        return self.a + self.b * theta


class Calibrations(Mapping[str, Calibration]):
    """The calibrations of a table of calibrated parameters by label, each grade's in turn (``GRADE_ROWS``), with the
    TIMSS scale constants of ``constants_path`` (the package's copy by default)."""

    def __init__(self, parameters: pd.DataFrame, constants_path: str | Path | None = None):
        of_item = {key: calibration_label(row["calibration"], row["grade"]) for key, row in parameters.iterrows()}
        if parameters["calibration"].nunique() != len(set(of_item.values())):
            raise ValueError("Two calibrations have the same short name")
        prefixes = [prefix for prefix, _ in GRADE_ROWS]
        labels = sorted(set(of_item.values()), key=lambda label: (prefixes.index(label.split(",")[0]), label))
        path = Path(constants_path or PACKAGE_DATA_DIR / SCALE_CONSTANTS_FILE)
        constants = pd.read_csv(path)
        by_calibration = constants[constants["use_for_scoring"]].groupby("calibration")
        if set(by_calibration.groups) != set(parameters["calibration"]) or by_calibration.size().ne(5).any():
            raise ValueError(f"{path.name} needs five plausible values of constants for every calibration")
        self.of_item: dict[str, str] = of_item
        """The label of each item's calibration."""
        self.by_label: dict[str, Calibration] = {}
        for label in labels:
            keys = tuple(key for key in parameters.index if of_item[key] == label)
            name = parameters.at[keys[0], "calibration"]
            rows = by_calibration.get_group(name)
            self.by_label[label] = Calibration(
                label,
                name,
                int(parameters.at[keys[0], "grade"]),
                keys,
                rows["A"].mean(),
                rows["B"].mean(),
                rows["method"].iloc[0].split(":")[0],
            )

    def __getitem__(self, label: str) -> Calibration:
        return self.by_label[label]

    def __iter__(self) -> Iterator[str]:
        return iter(self.by_label)

    def __len__(self) -> int:
        return len(self.by_label)

    def items_of(self) -> dict[str, list[str]]:
        """The item keys of each calibration."""
        return {label: list(calibration.item_keys) for label, calibration in self.items()}

    def table(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """One row per calibration: its name, assessments, items, points in the scaling, and TIMSS scale constants."""
        return pd.DataFrame(
            {
                "calibration": [calibration.name for calibration in self.values()],
                "assessments": [
                    ", ".join(sorted(set(parameters.loc[list(calibration.item_keys), "assessment"])))
                    for calibration in self.values()
                ],
                "items": [len(calibration.item_keys) for calibration in self.values()],
                "points": [
                    int(parameters.loc[list(calibration.item_keys), "scaled_points"].sum())
                    for calibration in self.values()
                ],
                "timss_scale_a": [calibration.a for calibration in self.values()],
                "timss_scale_b": [calibration.b for calibration in self.values()],
                "constants": [calibration.constants for calibration in self.values()],
            },
            index=pd.Index(list(self), name="label"),
        )
