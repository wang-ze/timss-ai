"""Scoring the answers: multiple-choice answers against the official keys, and constructed responses by an LLM scorer
with the item's scoring guide.

A multiple-choice answer earns 1 point if it is the item's official answer key (``correct_option``), which
``get_data.ipynb`` parsed from the official released-item documents and checked against a second source; the LLM
scorer never sees multiple-choice answers.

A constructed response is scored by an LLM scorer with the item's scoring guide, as TIMSS scorers score responses: it
sees the unit as the test taker saw it, the item's scoring guide and scoring notes, and the response, explains briefly
how the response compares with the guide, and chooses a category of the guide (``scorer_system_prompt``,
``scorer_user_prompt``, ``scorer_schema``). The scorer only chooses the category; the points come from the category
(``category_points``): Correct (or Fully Correct) earns the item's points, Partial one point less, Minimal 1 (in
3-point items), and Incorrect and Nonresponse 0. M042300Z of TIMSS 2011 grade 8 scores its parts A and B separately,
one category each and 1 point for a correct part, and its 2 points are their sum. An empty response earns 0 points
without the scorer.

K14 and L18 of TIMSS Advanced 1995 have 2 points in the coding guide but 1 point in the 1995-2008 scaling, and how
their codes were collapsed is not documented; here only full credit counts as their 1 point (``score_answers``).

The scorer is an LLM, so it can be wrong; read a sample of its verdicts (``scorer_reasoning``) before trusting small
differences between conditions.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, create_model

from ..items import plural
from .answering import TESTS, unit_text
from .units import ItemUnits, Unit

SCORE_COLUMNS = [
    "condition",
    "item_key",
    "assessment",
    "grade",
    "item_id",
    "content_domain",
    "cognitive_domain",
    "item_type",
    "irt_model",
    "max_points",
    "scaled_points",
    "answer",
    "correct_option",
    "scorer_category",
    "scorer_reasoning",
    "points",
    "score",
]


def category_points(category: str, max_points: int) -> int:
    """The points of a scoring-guide category of an item, or of a part, worth ``max_points``."""
    name = category.lower()
    if name.startswith(("incorrect", "nonresponse", "non response")):
        return 0
    if name.startswith("minimal") and max_points == 3:
        return 1
    if name.startswith("partial") and max_points >= 2:
        return max_points - 1
    if name.startswith(("correct", "fully correct")):
        return max_points
    raise ValueError(f"No points for the category {category!r} of an item worth {plural(max_points, 'point')}")


def scoring_categories(item: dict) -> dict[str | None, dict[str, int]]:
    """The points of each category of a constructed-response item's scoring guide, by part: one group (None) for most
    items, and one group per part, each part worth 1 point, for a guide whose categories name their part (M042300Z)."""
    by_part: dict[str | None, list[str]] = {}
    for category in item["scoring_guide"] or []:
        match = re.fullmatch(r"(M\d{6}[A-Z]) (.+)", category["category"])
        by_part.setdefault(match[1] if match else None, []).append(category["category"])
    if list(by_part) == [None]:
        groups = {None: {name: category_points(name, item["max_points"]) for name in by_part[None]}}
    elif None not in by_part and len(by_part) == item["max_points"]:
        groups = {
            part: {name: category_points(name.split(" ", 1)[1], 1) for name in names} for part, names in by_part.items()
        }
    else:
        raise ValueError(f"Cannot score the guide of {item['item_id']}, whose categories are by part {list(by_part)}")
    for part, points in groups.items():
        top = item["max_points"] if part is None else 1
        if set(points.values()) != set(range(top + 1)):
            raise ValueError(f"The categories of {item['item_id']} {part or ''} give {sorted(points.values())} points")
    return groups


def check_keys_and_guides(units: ItemUnits) -> list[str]:
    """Check that every calibrated multiple-choice item has an official key among its options, that the scoring guide
    of every calibrated constructed-response item gives every score from 0 to its points, and that no item has more
    points in the scaling than in its guide. Returns the items scaled with fewer points than their guide gives."""
    parameters, items_by_key = units.parameters, units.items_by_key
    for key in parameters.index:
        item = items_by_key[key]
        if item["item_type"] == "multiple_choice":
            if item["correct_option"] not in [option["label"] for option in item["options"]]:
                raise ValueError(f"{key} has no answer key among its options")
        else:
            scoring_categories(item)
    collapsed = [key for key in parameters.index if parameters.at[key, "scaled_points"] != items_by_key[key]["max_points"]]
    if any(parameters.at[key, "scaled_points"] > items_by_key[key]["max_points"] for key in collapsed):
        raise ValueError("An item has more points in the scaling than in its scoring guide")
    return collapsed


def scorer_system_prompt(grade: int) -> str:
    return f"""\
You are an experienced scorer of {TESTS[grade]}. You score the responses to its constructed-response questions with \
each question's scoring guide, as TIMSS trains its scorers to do.

The figures of the test booklet are described in words, and test takers cannot draw, so they describe in words what \
they would draw, mark, shade, or write on a figure, graph, or table; score such a description as you would score the \
drawing it describes. Score only what the response says: when the scoring guide asks for work or an explanation, give \
no credit for work or reasoning that the response does not show. First explain briefly how the response compares with \
the scoring guide, then choose the category of the guide that the response belongs to."""


def scorer_user_prompt(unit: Unit, item_key: str, response: str) -> str:
    """The unit as the test taker saw it, the item's scoring guide with the points of each category, and the response."""
    item = unit.items[unit.item_keys.index(item_key)]
    if len(unit.item_keys) == 1:
        question, target = f"# Question\n\n{unit_text(unit)}", "the question"
    else:
        question, target = f"# Questions\n\n{unit_text(unit)}", f"question {unit.item_keys.index(item_key) + 1}"
    points = {name: value for group in scoring_categories(item).values() for name, value in group.items()}
    lines = [
        f"- {category['category']} ({plural(points[category['category']], 'point')}):"
        + "".join(f"\n  - {descriptor}" for descriptor in category["descriptors"])
        for category in item["scoring_guide"]
    ]
    guide = f"# Scoring guide of {target} ({plural(item['max_points'], 'point')})\n\n" + "\n".join(lines)
    if len(scoring_categories(item)) > 1:
        guide += "\n\nEach part is scored on its own, and the points of the parts add up."
    if item["scoring_notes"]:
        guide += f"\n\nScoring notes: {item['scoring_notes']}"
    return "\n\n".join([question, guide, f"# Response to {target}\n\n{response}", f"Score the response to {target}."])


def category_field(part: str | None) -> str:
    """The output field of the category of a part: "category", or "category_<part>" for a guide by part."""
    return "category" if part is None else f"category_{part}"


def scorer_schema(item: dict) -> type[BaseModel]:
    """Structured output: the scorer's reasoning, then a category of the guide (one per part for M042300Z)."""
    fields = {
        "reasoning": (str, Field(description="How the response compares with the scoring guide, in a few sentences."))
    }
    for part, points in scoring_categories(item).items():
        whose = "the response" if part is None else f"the response to part {part}"
        description = f"The category of the scoring guide that {whose} belongs to."
        fields[category_field(part)] = (Literal[tuple(points)], Field(description=description))
    return create_model("Score", **fields)


def score_answers(
    answers: pd.DataFrame, units: ItemUnits, verdict: Callable[[str, str], dict | None]
) -> pd.DataFrame:
    """The points and scaled score of every answer to a calibrated item, None while a verdict is missing.

    ``answers`` has one row per condition and item (``condition``, ``item_key``, ``answer``), and ``verdict`` returns
    the saved scorer record of a constructed response to an item, or None.
    """
    parameters, items_by_key = units.parameters, units.items_by_key
    rows = []
    for row in answers[answers["item_key"].isin(parameters.index)].itertuples(index=False):
        item = items_by_key[row.item_key]
        category = reasoning = None
        if item["item_type"] == "multiple_choice":
            points = int(row.answer == item["correct_option"])  # the official key
        elif not row.answer.strip():
            points, category = 0, "(no response)"
        elif (record := verdict(row.item_key, row.answer)) is None:
            points = None
        else:
            groups = scoring_categories(item)
            chosen = {part: record["output"][category_field(part)] for part in groups}
            points = sum(groups[part][name] for part, name in chosen.items())
            category, reasoning = " + ".join(chosen.values()), record["output"]["reasoning"]
        scaled_points = int(parameters.at[row.item_key, "scaled_points"])
        if points is None or scaled_points == item["max_points"]:
            score = points
        else:  # K14 and L18: only full credit earns their 1 point in the scaling
            score = int(points == item["max_points"])
        rows.append(
            {
                "condition": row.condition,
                "item_key": row.item_key,
                "assessment": item["assessment"],
                "grade": item["grade"],
                "item_id": item["item_id"],
                "content_domain": item["content_domain"],
                "cognitive_domain": item["cognitive_domain"],
                "item_type": item["item_type"],
                "irt_model": parameters.at[row.item_key, "irt_model"],
                "max_points": item["max_points"],
                "scaled_points": scaled_points,
                "answer": row.answer,
                "correct_option": item["correct_option"],
                "scorer_category": category,
                "scorer_reasoning": reasoning,
                "points": points,
                "score": score,
            }
        )
    return pd.DataFrame(rows, columns=SCORE_COLUMNS).astype({"points": "Int64", "score": "Int64"})
