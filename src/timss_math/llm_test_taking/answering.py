"""The prompts and output schema with which an LLM answers a unit.

The system prompt names the test and its grade, in one of two roles:

- ``"test_taker"`` (evaluating the LLM): "You are taking the TIMSS mathematics test for students in the fourth grade.
  Answer each question as well as you can." It asks for the LLM's best answers to a test written for that grade,
  without asking it to act as a student, which would invite made-up mistakes.
- ``"student"`` (the LLM as a participant): "You are a fourth-grade student taking the TIMSS mathematics test for
  students in your grade. Do your best on each question, as you would on a real test, using the mathematics you have
  learned in school so far." It asks the LLM to answer as a student of that grade, with what such a student has
  learned, but never to make mistakes on purpose.

The grade is the fourth or the eighth, or the twelfth for TIMSS Advanced. The rest of the system prompt, the same in
both roles, says how the paper test is given as text (``TEXT_FORMAT``).

The user prompt is the unit: for each item its stem, its figure described in words (as ``get_data.ipynb`` transcribed
it with ``gemini-3.8-flash``), and its options, numbered in booklet order in a unit of several items. It never holds
the item's ID, classification, answer key, scoring guide, or statistics.

The answer comes back as structured output (``answer_schema``): for each item, the letter of an option for a
multiple-choice item, or the written answer, with any work, explanation, or description of a drawing that the item
asks for. No field asks for reasoning, so how much an LLM thinks before it answers is set by its reasoning effort
alone.

The saved answers are reused only while the prompts and the schema stay the same (``requesting.make_request``), so
any change here asks every unit again.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, create_model

from .units import Unit

Role = Literal["test_taker", "student"]
ROLES: tuple[Role, ...] = ("test_taker", "student")
TESTS = {  # the test_taker role's description of the test
    4: "the TIMSS mathematics test for students in the fourth grade",
    8: "the TIMSS mathematics test for students in the eighth grade",
    12: (
        "the TIMSS Advanced mathematics test for students in the twelfth grade, the final year of secondary school, "
        "who have taken advanced mathematics courses"
    ),
}
STUDENTS = {  # the student role's description of the student and the test
    4: "a fourth-grade student taking the TIMSS mathematics test for students in your grade",
    8: "an eighth-grade student taking the TIMSS mathematics test for students in your grade",
    12: (
        "a twelfth-grade student in the final year of secondary school who has taken advanced mathematics courses, "
        "taking the TIMSS Advanced mathematics test for students like you"
    ),
}
TEXT_FORMAT = """\
The pictures, figures, and graphs of the test booklet are described in words, and its tables are written out. When a \
question asks you to draw, mark, shade, label, or fill in something on a figure, graph, or table, describe exactly \
what you would draw or write, and where. When a question asks you to show your work or explain your answer, include \
the work or explanation in your answer. For a multiple-choice question, choose one option."""


def answer_system_prompt(grade: int, role: Role) -> str:
    """The test, its grade, and the role, then how the paper test is given as text."""
    if role == "test_taker":
        opening = f"You are taking {TESTS[grade]}. Answer each question as well as you can."
    elif role == "student":
        opening = (
            f"You are {STUDENTS[grade]}. Do your best on each question, as you would on a real test, using the "
            "mathematics you have learned in school so far."
        )
    else:
        raise ValueError(f"Unknown role {role!r}")
    return f"{opening}\n\n{TEXT_FORMAT}"


def question_text(item: dict) -> str:
    """An item as the test shows it: the stem, the figure described in words, and the options."""
    blocks = [item["stem"]]
    if item["figure_description"]:
        blocks.append(f"Figure, described in words:\n{item['figure_description']}")
    if item["options"]:
        blocks.append("\n".join(f"{option['label']}. {option['text']}" for option in item["options"]))
    return "\n\n".join(blocks)


def unit_text(unit: Unit) -> str:
    """The question of an item alone, or the numbered questions of a unit of several items."""
    if len(unit.item_keys) == 1:
        return question_text(unit.items[0])
    questions = [f"## Question {number}\n\n{question_text(item)}" for number, item in enumerate(unit.items, 1)]
    intro = f"These {len(questions)} questions belong together and follow one another in the test booklet."
    return "\n\n".join([intro, *questions])


def answer_user_prompt(unit: Unit) -> str:
    if len(unit.item_keys) == 1:
        return unit_text(unit)
    return f"{unit_text(unit)}\n\nAnswer each of the {len(unit.item_keys)} questions."


def answer_fields(unit: Unit) -> dict[str, str]:
    """The output field of each item of a unit: "answer", or "answer_1", "answer_2", ... in booklet order."""
    if len(unit.item_keys) == 1:
        return {unit.item_keys[0]: "answer"}
    return {key: f"answer_{number}" for number, key in enumerate(unit.item_keys, 1)}


def answer_schema(unit: Unit) -> type[BaseModel]:
    """Structured output: an option letter for each multiple-choice item, and a written answer for each other item."""
    fields = {}
    for name, item in zip(answer_fields(unit).values(), unit.items):
        question = "the question" if name == "answer" else f"question {name.removeprefix('answer_')}"
        if item["item_type"] == "multiple_choice":
            labels = tuple(option["label"] for option in item["options"])
            fields[name] = (Literal[labels], Field(description=f"The letter of the option you choose for {question}."))
        else:
            description = (
                f"Your answer to {question}, as you would write it in the test booklet, with any work, explanation, "
                "or description of a drawing that the question asks for."
            )
            fields[name] = (str, Field(description=description))
    return create_model("Answers" if len(fields) > 1 else "Answer", **fields)
