"""The released TIMSS mathematics items and their calibrated IRT parameters, as the prediction modules use them.

``ItemBank.load`` reads ``timss_math_released_items.jsonl`` and ``timss_math_released_item_parameters.csv`` (the
``item_parameters`` sheet of ``timss_math_released_item_parameters.xlsx``) from the ``data`` directory of this package,
where ``get_data.ipynb`` writes them, and checks them as ``predict_item_parameters.ipynb`` does:

- One chunk per item, except the three items in ``DUPLICATE_ITEMS`` whose stem and options repeat another item's word
  for word, so the 731 items give 728 chunks. A chunk's ``text`` (stem and options) is the only text that is embedded
  and keyword-indexed; answers, scoring guides, and figure descriptions never are.
- The calibrated parameters of the chunks that have them (722), with each item's IRT model checked against its format
  and points in the scaling, and the steps of every GPCM item summing to 0.
- The parts of each item: the items of its assessment and grade whose IDs differ only in a final part letter, and the
  other items of its task (``SHARED_STIMULUS_TASKS``). A leave-one-out prediction leaves all of them out.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"
RELEASED_ITEMS_FILE = "timss_math_released_items.jsonl"
ITEM_PARAMETERS_FILE = "timss_math_released_item_parameters.csv"

# Copied into each chunk for filters and display. Answers and scoring (correct_option, scoring_guide, scoring_notes)
# stay out of the index altogether.
METADATA_FIELDS = [
    "item_id",
    "item_label",
    "assessment",
    "grade",
    "subject",
    "content_domain",
    "main_topic",
    "cognitive_domain",
    "item_type",
    "max_points",
    "has_figure",
    "pct_correct_intl_avg",
    "pct_correct_usa",
]
# Items whose stem and options repeat another item's word for word, each with the item it repeats. They are left out
# of the index.
DUPLICATE_ITEMS = {
    "TIMSS 1995/grade 4/Mathematics/K7": "TIMSS 2003/grade 8/Mathematics/M012030",
    "TIMSS 1999/grade 8/Mathematics/D11": "TIMSS 2003/grade 4/Mathematics/M012023",
    "TIMSS Advanced 1995/grade 12/Advanced Mathematics/L15B": "TIMSS Advanced 1995/grade 12/Advanced Mathematics/L15A",
}
# Tasks of several items around one situation whose items have IDs of their own: assessment, grade, and the base IDs
# of the task's items.
SHARED_STIMULUS_TASKS = [
    ("TIMSS 2003", 4, ["M011009", "M011012"]),  # bar graph of bottles
    ("TIMSS 2003", 4, ["M031344", "M031345"]),  # number tiles
    ("TIMSS 2003", 8, ["M032743", "M032744", "M032745"]),  # geometry tiling
    ("TIMSS 2003", 8, ["M032762", "M032763", "M032764"]),  # phone plans
    ("TIMSS 2007", 8, ["M032753", "M032754", "M032755", "M032756"]),  # class trip
    ("TIMSS 2011", 4, ["M031346", "M031379", "M031380"]),  # trading cards
    ("TIMSS 2011", 8, ["M032757", "M032760", "M032761"]),  # red and black tiles
]
PARAMETER_NAMES = ["slope", "location", "guessing", "step1", "step2", "step3"]
STEP_NAMES = ["step1", "step2", "step3"]
PARAMETER_LABELS = {
    "slope": "slope (a)",
    "location": "location (b)",
    "guessing": "guessing (c)",
    "step1": "step 1 (d_1)",
    "step2": "step 2 (d_2)",
    "step3": "step 3 (d_3)",
}


def item_key(item) -> str:
    """Unique key of an item, such as "TIMSS 1995/grade 4/Mathematics/K1"; item IDs alone repeat across assessments."""
    return f"{item['assessment']}/grade {item['grade']}/{item['subject']}/{item['item_id']}"


def item_text(item: dict) -> str:
    """Text to embed and keyword-index: the stem, then one line per option. Never the answer or scoring."""
    options = "\n".join(f"{option['label']}. {option['text']}" for option in item["options"] or [])
    return "\n\n".join(part for part in (item["stem"], options) if part)


def base_item_id(item_id: str) -> str:
    """The item ID without its part letter: M031346 for M031346A, V4 for V4A, T02 for T02B."""
    return re.sub(r"(?<=\d)[A-Z]$", "", item_id)


def model_parameters(irt_model: str, points: int) -> list[str]:
    """The parameters of an item scored with ``irt_model`` and worth ``points`` in the scaling."""
    if irt_model == "3PL":
        return ["slope", "location", "guessing"]
    if irt_model == "2PL":
        return ["slope", "location"]
    return ["slope", "location", *[f"step{step}" for step in range(1, points + 1)]]


def predicted_parameters(irt_model: str, points: int) -> list[str]:
    """The parameters an LLM predicts: all but the last step of a GPCM item, which is minus the sum of the others."""
    names = model_parameters(irt_model, points)
    return names[:-1] if irt_model == "GPCM" else names


class ItemOption(BaseModel):
    label: str
    text: str


class ScoringCategory(BaseModel):
    category: str
    descriptors: list[str]


class TimssItem(BaseModel):
    """An item whose parameters are predicted, with the fields of the released-items JSONL that describe it.

    ``assessment`` names the calibration the prediction targets (``ItemBank.calibrations``). ``scaled_points`` is the
    item's points in the IRT scaling when they differ from ``max_points``, as for K14 and L18 of TIMSS Advanced 1995.
    """

    item_id: str | None = None
    assessment: str
    grade: Literal[4, 8, 12]
    subject: str
    content_domain: str
    main_topic: str | None = None
    cognitive_domain: str
    item_type: Literal["multiple_choice", "constructed_response"]
    max_points: int = Field(default=1, ge=1)
    scaled_points: int | None = Field(default=None, ge=1)
    stem: str
    options: list[ItemOption] | None = None
    correct_option: str | None = None
    scoring_guide: list[ScoringCategory] | None = None
    scoring_notes: str | None = None
    figure_description: str | None = None

    @model_validator(mode="after")
    def check_format(self):
        if self.item_type == "multiple_choice":
            labels = [option.label for option in self.options or []]
            if len(labels) < 2 or self.correct_option not in labels or self.points != 1:
                raise ValueError("A multiple-choice item needs options, a correct option among them, and 1 point")
        elif self.options or self.correct_option:
            raise ValueError("A constructed-response item has no options or correct option")
        return self

    @property
    def points(self) -> int:
        """The item's points in the IRT scaling."""
        return self.scaled_points or self.max_points

    @property
    def irt_model(self) -> str:
        """3PL for multiple choice, 2PL for a 1-point and GPCM for a polytomous constructed-response item."""
        if self.item_type == "multiple_choice":
            return "3PL"
        return "2PL" if self.points == 1 else "GPCM"

    @property
    def text(self) -> str:
        """The text that is embedded and keyword-indexed: stem and options."""
        return item_text(self.model_dump())


def load_released_items(path: Path) -> dict[str, dict]:
    """The released items of a JSONL file by item_key."""
    released_items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    items_by_key = {item_key(item): item for item in released_items}
    if len(items_by_key) != len(released_items):
        raise ValueError(f"Item keys must be unique in {path.name}")
    return items_by_key


def build_chunks(items_by_key: dict[str, dict]) -> list[dict]:
    """One chunk per item except DUPLICATE_ITEMS: its item_key, METADATA_FIELDS, and text, checked as in the notebook."""
    for duplicate, original in DUPLICATE_ITEMS.items():
        if item_text(items_by_key[duplicate]) != item_text(items_by_key[original]):
            raise ValueError(f"{duplicate} no longer repeats the text of {original}; update DUPLICATE_ITEMS")
    chunks = [
        {"item_key": key, **{field: item[field] for field in METADATA_FIELDS}, "text": item_text(item)}
        for key, item in items_by_key.items()
        if key not in DUPLICATE_ITEMS
    ]
    texts = pd.Series([chunk["text"] for chunk in chunks])
    if not texts.str.strip().astype(bool).all():
        raise ValueError("Every chunk needs text")
    repeated = [chunk["item_key"] for chunk, repeats in zip(chunks, texts.duplicated(keep=False)) if repeats]
    if repeated:
        raise ValueError(f"Texts repeat in {repeated}; add the repeats to DUPLICATE_ITEMS")
    return chunks


def load_calibrated_parameters(path: Path, chunk_keys: list[str]) -> pd.DataFrame:
    """The calibrated parameters of the chunks that have them, indexed by item_key, with ``scaled_points`` added.

    Checks that every chunk has a row, that each item's IRT model fits its format and points in the scaling, that
    each model has exactly its parameters, that GPCM steps sum to 0, and that each assessment and grade has one
    calibration.
    """
    parameter_table = pd.read_csv(path)
    parameter_table.index = parameter_table.apply(item_key, axis=1).rename("item_key")
    if not parameter_table.index.is_unique:
        raise ValueError(f"Item keys repeat in {path.name}")
    missing_rows = sorted(set(chunk_keys) - set(parameter_table.index))
    if missing_rows:
        raise ValueError(f"No row in {path.name} for {missing_rows}")

    calibrated = parameter_table.loc[chunk_keys]
    calibrated = calibrated[calibrated["irt_model"].notna()].copy()
    calibrated["scaled_points"] = np.where(calibrated["irt_model"] == "GPCM", calibrated[STEP_NAMES].notna().sum(axis=1), 1)

    # The model must fit the format and points in the scaling, as in the TIMSS scaling.
    expected_model = np.where(
        calibrated["item_type"] == "multiple_choice",
        "3PL",
        np.where(calibrated["scaled_points"] == 1, "2PL", "GPCM"),
    )
    misfits = calibrated.index[calibrated["irt_model"] != expected_model].tolist()
    if misfits:
        raise ValueError(f"The IRT model does not fit the item format of {misfits}")
    for model_name, parameters in {"3PL": ["slope", "location", "guessing"], "2PL": ["slope", "location"]}.items():
        rows = calibrated[calibrated["irt_model"] == model_name]
        others = [name for name in PARAMETER_NAMES if name not in parameters]
        if rows[parameters].isna().any().any() or rows[others].notna().any().any():
            raise ValueError(f"{model_name} items must have exactly the parameters {parameters}")
    gpcm_rows = calibrated[calibrated["irt_model"] == "GPCM"]
    if gpcm_rows[["slope", "location", "step1", "step2"]].isna().any().any() or gpcm_rows["guessing"].notna().any():
        raise ValueError("GPCM items need a slope, a location, and at least two steps, and no guessing parameter")
    unbalanced_steps = gpcm_rows.index[gpcm_rows[STEP_NAMES].sum(axis=1).abs() > 0.002].tolist()
    if unbalanced_steps:
        raise ValueError(f"The steps of {unbalanced_steps} do not sum to 0")
    if calibrated.groupby(["assessment", "grade"])["calibration"].nunique().gt(1).any():
        raise ValueError("Every assessment and grade should have one calibration")
    return calibrated


def find_item_parts(chunks: list[dict]) -> dict[str, list[str]]:
    """All parts of each chunk's item, the item included: the chunks of its assessment and grade whose IDs differ only
    in the part letter, and the other items of its task (SHARED_STIMULUS_TASKS)."""
    task_ids = {
        (assessment, grade, base_id): base_ids[0]
        for assessment, grade, base_ids in SHARED_STIMULUS_TASKS
        for base_id in base_ids
    }
    chunk_bases = [(chunk["assessment"], chunk["grade"], base_item_id(chunk["item_id"])) for chunk in chunks]
    unknown_task_items = sorted(set(task_ids) - set(chunk_bases))
    if unknown_task_items:
        raise ValueError(f"SHARED_STIMULUS_TASKS lists items that are not in the collection: {unknown_task_items}")
    parts_by_group: dict[tuple, list[str]] = {}
    for chunk, (assessment, grade, base_id) in zip(chunks, chunk_bases):
        group = (assessment, grade, task_ids.get((assessment, grade, base_id), base_id))
        parts_by_group.setdefault(group, []).append(chunk["item_key"])
    return {key: sorted(parts) for parts in parts_by_group.values() for key in parts}


@dataclass(eq=False)
class ItemBank:
    """The released items, their chunks, their calibrated parameters, and their parts."""

    items_by_key: dict[str, dict]
    """Every released item (731) by item_key, with all its JSONL fields."""
    chunks: list[dict]
    """One chunk per indexed item (728): item_key, METADATA_FIELDS, and text."""
    calibrated: pd.DataFrame
    """The parameters of the indexed items that have them (722), indexed by item_key."""
    parts: dict[str, list[str]]
    """The keys of all parts of each indexed item, the item included."""

    @classmethod
    def load(
        cls, released_items_path: str | Path | None = None, parameters_path: str | Path | None = None
    ) -> ItemBank:
        """Load and check the items and parameters, by default from the ``data`` directory of this package."""
        items_by_key = load_released_items(Path(released_items_path or PACKAGE_DATA_DIR / RELEASED_ITEMS_FILE))
        chunks = build_chunks(items_by_key)
        calibrated = load_calibrated_parameters(
            Path(parameters_path or PACKAGE_DATA_DIR / ITEM_PARAMETERS_FILE), [chunk["item_key"] for chunk in chunks]
        )
        return cls(items_by_key, chunks, calibrated, find_item_parts(chunks))

    @functools.cached_property
    def calibrations(self) -> dict[tuple[str, int], str]:
        """The calibration of each assessment and grade: the scale that a prediction for an item of it targets."""
        return self.calibrated.groupby(["assessment", "grade"])["calibration"].first().to_dict()

    def item(self, key: str) -> TimssItem:
        """The item with this key as a TimssItem, with its points in the scaling if it is calibrated."""
        fields = self.items_by_key[key]
        if key in self.calibrated.index:
            fields = fields | {"scaled_points": int(self.calibrated.at[key, "scaled_points"])}
        return TimssItem.model_validate(fields)

    def same_scoring_items(self, item: TimssItem, exclude_keys: Iterable[str] = ()) -> pd.DataFrame:
        """Calibrated items of the item's grade scored with its IRT model and points, except ``exclude_keys``."""
        calibrated = self.calibrated
        return calibrated[
            (calibrated["grade"] == item.grade)
            & (calibrated["irt_model"] == item.irt_model)
            & (calibrated["scaled_points"] == item.points)
            & ~calibrated.index.isin(list(exclude_keys))
        ]

    def parameter_payload(self, key: str) -> dict | None:
        """The item_parameters payload of the item with this key, or None if it has no parameters."""
        if key not in self.calibrated.index:
            return None
        row = self.calibrated.loc[key]

        def number(value):
            return None if pd.isna(value) else float(value)

        return {
            "irt_model": row["irt_model"],
            "scaled_points": int(row["scaled_points"]),
            **{name: number(row[name]) for name in PARAMETER_NAMES},
            **{f"{name}_se": number(row[f"{name}_se"]) for name in PARAMETER_NAMES},
            "calibration": row["calibration"],
        }
