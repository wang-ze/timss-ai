"""LLMs taking the released items as a test: the units, prompts, scoring, IRT models and estimates, TIMSS scales,
requests against fakes (direct and in OpenRouter batches), and the saved run of 2026-10-19. No request leaves this
computer.

Run with ``uv run python -m unittest discover tests``.
"""

import asyncio
import contextlib
import io
import json
import math
import tempfile
import typing
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from timss_math.config import CLONE_ROOT
from timss_math.llm import Completion, NoPredictionError
from timss_math.llm_test_taking import (
    DEFAULT_SCORER,
    Administration,
    Condition,
    ItemModel,
    ItemUnits,
    LLMSettings,
    ScoringResults,
    benchmark,
    conditions_of,
    estimate_theta,
    plots,
)
from timss_math.llm_test_taking.administration import ANSWERS_DIR, VERDICTS_DIR
from timss_math.llm_test_taking.answering import (
    answer_fields,
    answer_schema,
    answer_system_prompt,
    answer_user_prompt,
)
from timss_math.llm_test_taking.cli import main, parse_args, run_name
from timss_math.llm_test_taking.conditions import direct_cost, settings_folder
from timss_math.llm_test_taking.irt import THETA, check_item_models, item_models
from timss_math.llm_test_taking.requesting import make_request, send, send_direct
from timss_math.llm_test_taking.scales import (
    Calibrations,
)
from timss_math.llm_test_taking.scoring import (
    category_points,
    check_keys_and_guides,
    score_answers,
    scorer_schema,
    scoring_categories,
)

TRADING_CARDS = "TIMSS 2011/grade 4/Mathematics/M031346A"
M042300Z = "TIMSS 2011/grade 8/Mathematics/M042300Z"
K14 = "TIMSS Advanced 1995/grade 12/Advanced Mathematics/K14"
L16 = "TIMSS Advanced 1995/grade 12/Advanced Mathematics/L16"
GPT = LLMSettings(model="openai/gpt-6-luna", reasoning_effort="high", api_key="test")
GEMINI = LLMSettings(model="google/gemini-3.8-flash", reasoning_effort="minimal", api_key="test")
NOTEBOOK_DATA_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "data"


def setUpModule():
    global UNITS
    UNITS = ItemUnits.load()


def valid_output(schema) -> dict:
    """An output that fits a schema: the first allowed value of every choice, and a text for every other field."""
    return {
        name: typing.get_args(field.annotation)[0] if typing.get_origin(field.annotation) is typing.Literal else "text"
        for name, field in schema.model_fields.items()
    }


class FakeSession:
    """An AsyncLLMSession whose requests answer valid output after ``delay`` seconds, or raise the error that
    ``errors`` maps their user prompt to."""

    def __init__(self, delay=0.01, errors=None, cost=0.001):
        self.delay, self.errors, self.cost = delay, errors or {}, cost
        self.prompts: list[str] = []
        self.in_flight = self.max_in_flight = 0

    async def complete(self, system_prompt, user_prompt, schema):
        self.prompts.append(user_prompt)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        if user_prompt in self.errors:
            raise self.errors[user_prompt]
        return Completion(schema.model_validate(valid_output(schema)), None, "stop", 100, 50, 0, self.cost)


class FakeBatches:
    """OpenRouter's Batch API in memory: each batch is in progress on its first get and completed after."""

    def __init__(self, reject: str | None = None):
        self.reject = reject
        self.submitted: dict[str, list[dict]] = {}
        self.gets: dict[str, int] = {}

    def submit(self, batch_requests):
        if self.reject:
            from timss_math.item_parameter_prediction.batch import BatchAPIError

            raise BatchAPIError(self.reject)
        batch_id = f"batch_{len(self.submitted) + 1}"
        self.submitted[batch_id], self.gets[batch_id] = batch_requests, 0
        return {"id": batch_id, "status": "validating"}

    def get(self, batch_id):
        self.gets[batch_id] += 1
        if self.gets[batch_id] == 1:
            return {"id": batch_id, "status": "in_progress"}
        return {
            "id": batch_id,
            "status": "completed",
            "usage": {"cost": 0.01},
            "results": [self.result(request) for request in self.submitted[batch_id]],
        }

    @staticmethod
    def result(request):
        properties = request["body"]["response_format"]["json_schema"]["schema"]["properties"]
        output = {name: spec["enum"][0] if "enum" in spec else "text" for name, spec in properties.items()}
        body = {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 1,
            "model": request["body"]["model"],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": json.dumps(output)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
        return {"custom_id": request["custom_id"], "response": {"status_code": 200, "body": body}, "error": None}

    def list_created_after(self, created_after):
        return []


class UnitsTest(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(UNITS.items_by_key), 731)
        self.assertEqual(len(UNITS.parameters), 724)
        self.assertEqual(len(UNITS.units), 652)
        self.assertEqual(sum(len(unit.item_keys) > 1 for unit in UNITS.units), 39)
        self.assertEqual(sorted(key for unit in UNITS.all_units for key in unit.item_keys), sorted(UNITS.items_by_key))

    def test_a_task_is_one_unit_in_booklet_order(self):
        unit = UNITS.unit_of_item[TRADING_CARDS]
        self.assertEqual(unit.key, "TIMSS 2011/grade 4/Mathematics/M031346A+M031346B+M031346C+M031379+M031380")
        self.assertEqual(unit.grade, 4)


class PromptsTest(unittest.TestCase):
    def test_roles(self):
        self.assertTrue(
            answer_system_prompt(4, "test_taker").startswith(
                "You are taking the TIMSS mathematics test for students in the fourth grade. Answer each question as "
                "well as you can.\n\n"
            )
        )
        self.assertTrue(answer_system_prompt(8, "student").startswith("You are an eighth-grade student taking"))
        with self.assertRaises(ValueError):
            answer_system_prompt(4, "teacher")
        with self.assertRaises(ValueError):
            Condition(GPT, "teacher")

    def test_unit_of_several_items(self):
        unit = UNITS.unit_of_item[TRADING_CARDS]
        prompt = answer_user_prompt(unit)
        self.assertTrue(prompt.startswith("These 5 questions belong together"))
        self.assertTrue(prompt.endswith("Answer each of the 5 questions."))
        for item in unit.items:  # never the ID, key, or scoring guide
            self.assertNotIn(item["item_id"], prompt)
        self.assertEqual(list(answer_fields(unit).values()), [f"answer_{number}" for number in range(1, 6)])
        self.assertEqual(list(answer_schema(unit).model_fields), list(answer_fields(unit).values()))

    def test_multiple_choice_schema(self):
        unit = UNITS.unit_of_item["TIMSS 1999/grade 8/Mathematics/F07"]
        field = answer_schema(unit).model_fields["answer"]
        labels = [option["label"] for option in unit.items[0]["options"]]
        self.assertEqual(list(typing.get_args(field.annotation)), labels)

    def test_condition_labels_and_folders(self):
        claude = LLMSettings(provider="anthropic", model="claude-haiku-5-5", reasoning_effort=None)
        self.assertEqual(Condition(claude, "student").label, "claude-haiku-5-5 (anthropic), default thinking, as a student")
        self.assertEqual(Condition(GPT).label, "gpt-6-luna, high reasoning")
        gemma = LLMSettings(model="google/gemma-3-12b-it", reasoning_effort=None)
        self.assertEqual(Condition(gemma).label, "gemma-3-12b-it")
        self.assertEqual(settings_folder(Path("cache"), gemma), Path("cache/openrouter/google_gemma-3-12b-it/default"))
        roles = [condition.role for condition in conditions_of([GPT, gemma])]
        self.assertEqual(roles, ["test_taker"] * 2 + ["student"] * 2)
        with self.assertRaisesRegex(ValueError, "same label"):
            Administration([Condition(GPT), Condition(GPT)], units=UNITS)

    def test_request_names(self):
        """A request is named by its unit or item and a hash of everything sent, and only the same request reuses it."""
        schema = answer_schema(UNITS.units[0])
        name = "TIMSS 2011/grade 4/Mathematics/M1"
        request = make_request(Path("folder"), GPT, name, "system", "user", schema, unit_key=name)
        self.assertEqual(request.path, make_request(Path("folder"), GPT, name, "system", "user", schema).path)
        self.assertTrue(request.path.name.startswith("TIMSS_2011_grade_4_Mathematics_M1_"))
        self.assertEqual(request.fields["unit_key"], name)
        for other in (
            make_request(Path("folder"), GPT, name, "system", "user 2", schema),
            make_request(Path("folder"), GEMINI, name, "system", "user", schema),
            make_request(Path("folder"), GPT, name, "system", "user", answer_schema(UNITS.unit_of_item[TRADING_CARDS])),
        ):
            self.assertNotEqual(request.custom_id, other.custom_id)
        # max_tokens is not part of it, so changing it reuses the saved outputs.
        capped = LLMSettings(model=GPT.model, reasoning_effort=GPT.reasoning_effort, max_tokens=1000)
        self.assertEqual(request.path, make_request(Path("folder"), capped, name, "system", "user", schema).path)


class ScoringTest(unittest.TestCase):
    def test_category_points(self):
        self.assertEqual(category_points("Correct Response", 2), 2)
        self.assertEqual(category_points("Partial Response", 3), 2)
        self.assertEqual(category_points("Minimal Response", 3), 1)
        self.assertEqual(category_points("Nonresponse", 1), 0)
        with self.assertRaises(ValueError):
            category_points("Partial Response", 1)

    def test_guide_by_part(self):
        groups = scoring_categories(UNITS.items_by_key[M042300Z])
        self.assertEqual(list(groups), ["M042300A", "M042300B"])
        self.assertEqual(groups["M042300A"], {"M042300A Correct Response": 1, "M042300A Incorrect Response": 0})
        fields = list(scorer_schema(UNITS.items_by_key[M042300Z]).model_fields)
        self.assertEqual(fields, ["reasoning", "category_M042300A", "category_M042300B"])

    def test_keys_and_guides(self):
        self.assertEqual(check_keys_and_guides(UNITS), [K14, "TIMSS Advanced 1995/grade 12/Advanced Mathematics/L18"])

    def test_score_answers(self):
        mc = "TIMSS 1999/grade 8/Mathematics/F07"
        key = UNITS.items_by_key[mc]["correct_option"]
        wrong = next(option["label"] for option in UNITS.items_by_key[mc]["options"] if option["label"] != key)
        k14_categories = {points: name for name, points in scoring_categories(UNITS.items_by_key[K14])[None].items()}
        parts = {"category_M042300A": "M042300A Correct Response", "category_M042300B": "M042300B Incorrect Response"}
        verdicts = {
            (K14, "full"): {"output": {"reasoning": "why", "category": k14_categories[2]}},
            (K14, "partly"): {"output": {"reasoning": "why", "category": k14_categories[1]}},
            (M042300Z, "half"): {"output": {"reasoning": "why", **parts}},
        }
        answers = pd.DataFrame(
            [
                ("a", mc, key),
                ("b", mc, wrong),
                ("a", K14, "full"),
                ("b", K14, "partly"),
                ("c", K14, " "),
                ("a", M042300Z, "half"),
                ("b", M042300Z, "no verdict yet"),
            ],
            columns=["condition", "item_key", "answer"],
        )
        scores = score_answers(answers, UNITS, lambda key, answer: verdicts.get((key, answer)))
        self.assertEqual(scores["points"].tolist(), [1, 0, 2, 1, 0, 1, pd.NA])
        # K14 has 1 point in the scaling, earned only with full credit.
        self.assertEqual(scores["score"].tolist(), [1, 0, 1, 0, 0, 1, pd.NA])
        self.assertEqual(scores.loc[4, "scorer_category"], "(no response)")
        self.assertEqual(scores.loc[5, "scorer_category"], "M042300A Correct Response + M042300B Incorrect Response")


class IRTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = item_models(UNITS.parameters)

    def test_models_follow_their_definitions(self):
        check_item_models(self.models)

    def test_3pl(self):
        model = ItemModel("3PL", slope=1.0, location=0.5, guessing=0.2, steps=())
        expected = 0.2 + 0.8 / (1 + math.exp(-1.7 * (1.5 - 0.5)))
        self.assertAlmostEqual(model.probabilities(1.5)[0, 1], expected)
        self.assertAlmostEqual(model.expected_score(0.5)[0], 0.6)

    def test_gpcm_steps_cross_at_b_minus_d(self):
        model = ItemModel("GPCM", slope=0.8, location=0.2, guessing=0.0, steps=(0.6, -0.6))
        at = model.probabilities(0.2 - 0.6)[0]
        self.assertAlmostEqual(at[0], at[1])
        self.assertEqual(model.points, 2)

    def test_estimates(self):
        models = [ItemModel("2PL", 1.0, b, 0.0, ()) for b in (-1.0, 0.0, 1.0)]
        perfect = estimate_theta(models, [1, 1, 1])
        self.assertEqual((perfect["theta_mle"], perfect["mle_note"]), (math.inf, "perfect score"))
        self.assertGreater(perfect["theta_eap"], 0)
        zero = estimate_theta(models, [0, 0, 0])
        self.assertEqual((zero["theta_mle"], zero["mle_note"]), (-math.inf, "zero score"))
        # One right and two wrong of symmetric items: the likelihood peaks where the expected score is 1.
        mixed = estimate_theta(models, [1, 0, 0])
        expected = sum(model.expected_score(mixed["theta_mle"])[0] for model in models)
        self.assertAlmostEqual(expected, 1, places=4)
        self.assertLess(abs(mixed["theta_eap"]), abs(mixed["theta_mle"]))  # pulled toward 0 by the prior
        self.assertIsNone(mixed["mle_note"])

    def test_log_probabilities_cover_the_grid(self):
        model = next(iter(self.models.values()))
        self.assertEqual(model.log_probabilities.shape, (len(THETA), model.points + 1))


class ScalesTest(unittest.TestCase):
    def test_calibrations(self):
        calibrations = Calibrations(UNITS.parameters)
        self.assertEqual(
            list(calibrations),
            [
                "Grade 4, 1995-2003",
                "Grade 4, 2003-2007",
                "Grade 4, 2007-2011",
                "Grade 8, 1999-2003",
                "Grade 8, 2003-2007",
                "Grade 8, 2007-2011",
                "Advanced, 1995-2008",
            ],
        )
        self.assertEqual(sum(len(calibration.item_keys) for calibration in calibrations.values()), 724)
        self.assertEqual(calibrations["Grade 4, 1995-2003"].constants, "derived")
        self.assertEqual(calibrations["Grade 4, 2007-2011"].constants, "published")
        self.assertEqual(calibrations.table(UNITS.parameters)["points"].sum(), UNITS.parameters["scaled_points"].sum())

    def test_benchmarks(self):
        self.assertEqual(benchmark(625, 8), "Advanced (625)")
        self.assertEqual(benchmark(420, 4), "Low (400)")
        self.assertEqual(benchmark(420, 12), "below the lowest benchmark")  # TIMSS Advanced has no Low benchmark


class RequestTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        units = UNITS.units[:5]
        self.requests = [
            make_request(self.folder, GPT, unit.key, "system", answer_user_prompt(unit), answer_schema(unit))
            for unit in units
        ]

    def test_direct_saves_every_output_within_the_worker_bound(self):
        session = FakeSession()
        report = asyncio.run(send_direct(GPT, self.folder, self.requests, None, workers=2, session=session))
        self.assertEqual((report.requested, report.saved, report.missing), (5, 5, 5))
        self.assertAlmostEqual(report.cost_usd, 0.005)
        self.assertEqual(session.max_in_flight, 2)
        self.assertTrue(all(request.saved()["finish_reason"] == "stop" for request in self.requests))

    def test_unusable_output_is_left_for_a_rerun(self):
        session = FakeSession(errors={self.requests[1].user_prompt: NoPredictionError("cut off")})
        report = asyncio.run(send_direct(GPT, self.folder, self.requests, 3, session=session))
        self.assertEqual((report.requested, report.saved, list(report.unusable)), (3, 2, [self.requests[1].path.stem]))
        report = asyncio.run(send_direct(GPT, self.folder, self.requests, None, session=FakeSession()))
        self.assertEqual((report.missing, report.saved), (3, 3))

    def test_nothing_is_sent_with_no_requests_and_no_open_batch(self):
        with mock.patch("timss_math.llm_test_taking.requesting.batch_models", side_effect=AssertionError("network")):
            report = asyncio.run(send(GPT, self.folder, self.requests, 0, "batch"))
        self.assertEqual((report.mode, report.missing), ("nothing sent", 5))

    def test_batches_are_submitted_then_collected(self):
        fake = FakeBatches()
        with mock.patch("timss_math.llm_test_taking.requesting.batch_models", return_value=frozenset({GPT.model})):
            report = asyncio.run(send(GPT, self.folder, self.requests, None, "batch", batch_size=2, batch_client=fake))
            self.assertEqual((report.requested, report.in_batches, len(fake.submitted)), (5, 5, 3))
            # A later run with no new requests collects them.
            collect = send(GPT, self.folder, self.requests, 0, "batch", wait=True, poll_seconds=0, batch_client=fake)
            report = asyncio.run(collect)
        self.assertEqual((report.saved, report.in_batches, report.requested), (5, 0, 0))
        self.assertAlmostEqual(report.cost_usd, 0.03)
        self.assertTrue(all(request.saved()["batch_id"].startswith("batch_") for request in self.requests))
        self.assertEqual(len(list((self.folder / "batches" / "done").glob("*.json"))), 3)

    def test_google_batches_hold_one_schema(self):
        requests = [
            make_request(self.folder, GEMINI, unit.key, "system", answer_user_prompt(unit), answer_schema(unit))
            for unit in UNITS.units[:20]
        ]
        fake = FakeBatches()
        with mock.patch("timss_math.llm_test_taking.requesting.batch_models", return_value=frozenset({GEMINI.model})):
            asyncio.run(send(GEMINI, self.folder, requests, None, "batch", batch_client=fake))
        schemas = {json.dumps(request.schema.model_json_schema(), sort_keys=True) for request in requests}
        self.assertEqual(len(fake.submitted), len(schemas))
        for batch_requests in fake.submitted.values():
            self.assertEqual(len({json.dumps(request["body"]["response_format"]) for request in batch_requests}), 1)

    def test_a_model_without_batches_is_asked_directly(self):
        fake = FakeBatches(reject="HTTP 400: openai/gpt-6-luna does not have a :batch endpoint")
        with (
            mock.patch("timss_math.llm_test_taking.requesting.batch_models", return_value=frozenset({GPT.model})),
            self.assertLogs("timss_math.llm_test_taking.requesting", "WARNING"),
        ):
            sending = send(GPT, self.folder, self.requests, 2, "batch", session=FakeSession(), batch_client=fake)
            report = asyncio.run(sending)
        self.assertEqual((report.mode, report.saved), ("direct", 2))
        self.assertFalse(list((self.folder / "batches").glob("*.json")))  # the rejected batch left no manifest

    def test_cost_of_providers_without_one(self):
        records = pd.DataFrame({"prompt_tokens": [1_000_000], "completion_tokens": [2_000_000], "cost_usd": [None]})
        claude = LLMSettings(provider="anthropic", model="claude-haiku-5-5", reasoning_effort="none")
        self.assertAlmostEqual(direct_cost(claude, records), 1.10)
        self.assertTrue(math.isnan(direct_cost(LLMSettings(provider="openai", model="gpt-5"), records)))


class AdministrationTest(unittest.TestCase):
    """A small administration end to end, against fakes, in a new data directory."""

    def test_answer_score_and_export(self):
        data = Path(tempfile.mkdtemp())
        conditions = conditions_of([GPT])
        administration = Administration(conditions, DEFAULT_SCORER, UNITS, data_dir=data)
        self.assertEqual(administration.answers_dir, data / ANSWERS_DIR)
        self.assertEqual(administration.verdicts_dir, data / VERDICTS_DIR)
        same = Administration(administration.conditions, units=UNITS, data_dir=data)  # conditions by label
        self.assertEqual(same.conditions, administration.conditions)
        session = FakeSession(delay=0)
        reports = asyncio.run(administration.request_answers_async(30, mode="direct", session=session))
        saved = {label: report.saved for label, report in reports.items()}
        self.assertEqual(saved, {condition.label: 30 for condition in conditions})
        answers = administration.answers()
        self.assertEqual(set(answers["condition"]), {condition.label for condition in conditions})
        verdicts = administration.scorer_requests(answers)
        report = asyncio.run(administration.request_verdicts_async(None, answers=answers, session=session))
        self.assertEqual(report.saved, len(verdicts))
        self.assertTrue(all(request.path.is_relative_to(data / VERDICTS_DIR) for request in verdicts))

        self.assertEqual(administration.scores()["score"].isna().sum(), 0)
        results = administration.results()
        self.assertEqual(list(results.conditions.index), [condition.label for condition in conditions])
        self.assertEqual(results.conditions["answered_units"].tolist(), [30, 30])
        path = results.to_excel(data / "results.xlsx")
        back = ScoringResults.read_excel(path)
        pd.testing.assert_frame_equal(back.calibrations, results.calibrations)
        self.assertEqual(back.overall["points"].tolist(), results.overall["points"].tolist())
        np.testing.assert_allclose(back.overall["theta_mle"], results.overall["theta_mle"])
        plots.theta_figure(results)
        plots.item_scores_figure(results)

    def test_no_answers(self):
        administration = Administration(conditions_of([GPT]), units=UNITS, data_dir=tempfile.mkdtemp())
        self.assertEqual(administration.request_answers(0)["gpt-6-luna, high reasoning"].mode, "nothing sent")
        results = administration.results()
        self.assertTrue(results.overall.empty)
        self.assertEqual(results.conditions["answered_units"].tolist(), [0, 0])


class CommandLineTest(unittest.TestCase):
    def parse_fails(self, argv):
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            parse_args(argv)
        return stderr.getvalue()

    def conditions_file(self, entries) -> Path:
        path = Path(tempfile.mkdtemp()) / "conditions.json"
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_defaults_are_the_run_of_2026_10_19(self):
        args = parse_args([])
        self.assertEqual(len(args.conditions), 10)
        self.assertEqual(args.scorer, DEFAULT_SCORER)
        run = (args.max_requests, args.scorer_max_requests, args.mode, args.scorer_mode)
        self.assertEqual(run, (0, 0, "batch", "direct"))

    def test_one_llm_in_several_efforts(self):
        args = parse_args(["--llm-model", "openai/gpt-6-luna", "--reasoning-effort", "none", "high", "--role", "student"])
        self.assertEqual(
            [condition.label for condition in args.conditions],
            ["gpt-6-luna, no reasoning, as a student", "gpt-6-luna, high reasoning, as a student"],
        )
        self.assertEqual(
            run_name(args.conditions, args.scorer),
            "openrouter-openai_gpt-6-luna-none+high__student__scorer-openrouter-google_gemini-3.8-flash-medium",
        )
        args = parse_args(["--llm-model", "claude-haiku-5-5", "--llm-provider", "anthropic", "--reasoning-effort", "omit"])
        self.assertEqual(args.conditions[0].label, "claude-haiku-5-5 (anthropic), default thinking")
        # Requested answers get their verdicts unless the scorer's requests are capped.
        self.assertIsNone(parse_args(["--max-requests", "1"]).scorer_max_requests)
        self.assertEqual(parse_args(["--max-requests", "1", "--scorer-max-requests", "0"]).scorer_max_requests, 0)

    def test_conditions_file(self):
        path = self.conditions_file(
            [
                {"model": "openai/gpt-6-luna", "reasoning_effort": "high", "max_tokens": 8192},
                {"provider": "anthropic", "model": "claude-haiku-5-5", "reasoning_effort": None, "roles": ["student"]},
            ]
        )
        args = parse_args(["--conditions", str(path), "--role", "test_taker"])
        self.assertEqual(
            [condition.label for condition in args.conditions],
            ["gpt-6-luna, high reasoning", "claude-haiku-5-5 (anthropic), default thinking, as a student"],
        )
        self.assertEqual(args.conditions[0].llm.max_tokens, 8192)
        unknown = self.conditions_file([{"model": "m", "temperature": 0}])
        self.assertIn("Unknown settings ['temperature']", self.parse_fails(["--conditions", str(unknown)]))
        no_model = self.conditions_file([{"reasoning_effort": "high"}])
        self.assertIn("needs a model", self.parse_fails(["--conditions", str(no_model)]))

    def test_errors(self):
        self.assertIn("--reasoning-effort needs --llm-model", self.parse_fails(["--reasoning-effort", "high"]))
        both = ["--conditions", "x.json", "--llm-model", "m"]
        self.assertIn("--conditions takes the place of --llm-model", self.parse_fails(both))
        claude = ["--llm-provider", "anthropic", "--llm-model", "claude-haiku-5-5"]  # with the default effort, medium
        self.assertIn("cannot set the reasoning effort 'medium'", self.parse_fails(claude))
        self.assertIn("invalid choice: 'teacher'", self.parse_fails(["--role", "teacher"]))

    def test_scorer_and_token_prices(self):
        argv = ["--scorer-model", "openai/gpt-6-luna", "--scorer-reasoning-effort", "high"]
        args = parse_args([*argv, "--token-price", "openai", "gpt-5", "1", "8"])
        scorer = args.scorer
        self.assertEqual((scorer.model, scorer.reasoning_effort, scorer.max_tokens), ("openai/gpt-6-luna", "high", 8192))
        self.assertEqual(args.token_prices[("openai", "gpt-5")], (1.0, 8.0))
        self.assertIn(("anthropic", "claude-haiku-5-5"), args.token_prices)

    def run_main(self, argv) -> str:
        """The output of the command, without the logging setup that would outlive the test."""
        with contextlib.redirect_stdout(io.StringIO()) as stdout, mock.patch("logging.basicConfig"):
            main(argv)
        return stdout.getvalue()

    def test_nothing_to_score_stops(self):
        argv = ["--llm-model", "openai/gpt-6-luna", "--data-dir", tempfile.mkdtemp(), "--no-plots"]
        with self.assertRaisesRegex(SystemExit, "No condition has a scored answer"):
            self.run_main(argv)

    @unittest.skipUnless(CLONE_ROOT and (NOTEBOOK_DATA_DIR / ANSWERS_DIR).is_dir(), "the saved answers are not here")
    def test_the_saved_run(self):
        output = Path(tempfile.mkdtemp())
        printed = self.run_main(["--output-dir", str(output), "--no-plots"])
        results = ScoringResults.read_excel(output / "scores.xlsx")
        self.assertEqual(round(results.summary().loc["gemma-3-12b-it", "percent_of_points"], 1), 59.9)
        self.assertIn("TIMSS scale score (EAP) by calibration", printed)
        self.assertNotIn("missing requests sent", printed)  # nothing was requested


@unittest.skipUnless(CLONE_ROOT and (NOTEBOOK_DATA_DIR / ANSWERS_DIR).is_dir(), "the saved answers are not here")
class SavedRunTest(unittest.TestCase):
    """The answers and verdicts of the run of 2026-10-19 match the default prompts and settings."""

    @classmethod
    def setUpClass(cls):
        cls.administration = Administration(units=UNITS)

    def test_every_answer_is_saved(self):
        missing = [
            (label, unit.key)
            for label, by_unit in self.administration.answer_requests.items()
            for unit, request in by_unit.items()
            if not request.path.exists()
        ]
        self.assertEqual(missing, [("gemma-3-12b-it, as a student", L16)])  # cut off at max_tokens twice

    def test_every_verdict_is_saved_and_the_scores_hold(self):
        requests = self.administration.scorer_requests()
        self.assertEqual(len(requests), 2230)
        self.assertTrue(all(request.path.exists() for request in requests))
        results = self.administration.results()
        self.assertEqual(len(results.overall), 70)
        totals = results.overall.groupby("condition")[["points", "max_points"]].sum()
        share = (100 * totals["points"] / totals["max_points"]).round(1)
        self.assertEqual(share["gpt-6-luna, high reasoning"], 97.6)
        self.assertEqual(share["gemma-3-12b-it, as a student"], 63.8)


if __name__ == "__main__":
    unittest.main()
