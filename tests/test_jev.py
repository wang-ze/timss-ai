"""Jev as a predictor: value scales, the point predictions from its probabilities, the checks of its answers, the
reuse of the committed answers, and leave-one-out requests through the TypeSafe client against a mock transport. No
request leaves this computer.

Run with ``uv run python -m unittest discover tests``.
"""

import asyncio
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx2
import numpy as np
from typesafe_sdk import Score, ScoreAnswer, TypeSafeError

from timss_math.item_parameter_prediction import (
    DEFAULT_VALUE_SCALES,
    ItemIndex,
    JevPredictor,
    JevSettings,
    LeaveOneOut,
    ValueScale,
)
from timss_math.item_parameter_prediction.batch import BatchManifests
from timss_math.item_parameter_prediction.evaluation import parse_args
from timss_math.item_parameter_prediction.jev import (
    MAX_SCORE_LEVELS,
    REFERENCE_NOTE,
    answer_problems,
    posterior,
)

TEST_SETTINGS = JevSettings(api_key="test")


class ValueScaleTest(unittest.TestCase):
    def test_values_and_ticks(self):
        location = DEFAULT_VALUE_SCALES["location"]
        self.assertEqual(location.values[0], -2.0)
        self.assertEqual(location.ticks[:2], ["-2.0", "-1.0"])
        self.assertEqual(location.criteria[0], "-2.0 (very easy item)")

    def test_nearest(self):
        positions = DEFAULT_VALUE_SCALES["location"].nearest([0.15, np.nan, 5.0, -3.0])
        np.testing.assert_array_equal(positions, [4, np.nan, 9, 0])

    def test_rejects_bad_scales(self):
        too_many = tuple(str(value) for value in range(MAX_SCORE_LEVELS + 1))
        for criteria in (("1",), ("1", "1"), ("2", "1"), ("low", "high"), ("1", ""), too_many):
            with self.subTest(criteria=criteria), self.assertRaises(ValueError):
                ValueScale(criteria)


class PosteriorTest(unittest.TestCase):
    def test_estimators(self):
        result = posterior([0.1, 0.2, 0.3, 0.4], [0, 1, 2, 3])
        self.assertEqual((result["map"], result["median"]), (3.0, 2.0))
        self.assertAlmostEqual(result["map_probability"], 0.4)
        self.assertAlmostEqual(result["eap"], 2.0)
        self.assertAlmostEqual(result["sd"], 1.0)

    def test_ties_and_a_median_between_two_values(self):
        result = posterior([0.25, 0.25, 0.25, 0.25], [0, 1, 2, 4])
        self.assertEqual(result["map"], 0.0)  # the lowest of tied values
        self.assertEqual(result["median"], 1.5)  # cumulative probability exactly 0.5 at 1
        self.assertAlmostEqual(result["eap"], 1.75)

    def test_rounded_probabilities_are_scaled_to_one(self):
        result = posterior([0.2, 0.3, 0.51], [0, 1, 2])
        self.assertAlmostEqual(result["eap"], (0.3 + 2 * 0.51) / 1.01)
        self.assertAlmostEqual(result["map_probability"], 0.51 / 1.01)

    def test_rejects_probabilities_that_do_not_fit(self):
        for probabilities in ([0.5, 0.5], [0, 0, 0], [-0.1, 0.6, 0.5]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                posterior(probabilities, [0, 1, 2])


def score_answer(probabilities: dict[int, float]) -> ScoreAnswer:
    return ScoreAnswer(
        score=0.0, confidence=0.5, legend={k: str(k) for k in probabilities}, probabilities=probabilities
    )


QUESTIONS = {"slope": Score(criteria=["0.5", "1.0"])}


class AnswerProblemsTest(unittest.TestCase):

    def test_good_answers(self):
        self.assertEqual(answer_problems({"slope": score_answer({0: 0.49, 1: 0.5})}, QUESTIONS), [])

    def test_problems(self):
        cases = {
            "no answer to slope": {},
            "unexpected answer location": {"slope": score_answer({0: 0.5, 1: 0.5}), "location": score_answer({0: 1.0})},
            "has probabilities for [0]": {"slope": score_answer({0: 1.0})},
            "sum to 0.500": {"slope": score_answer({0: 0.25, 1: 0.25})},
        }
        for expected, answers in cases.items():
            with self.subTest(expected=expected):
                self.assertIn(expected, "; ".join(answer_problems(answers, QUESTIONS)))


class JevSettingsTest(unittest.TestCase):
    def test_checks(self):
        with self.assertRaises(ValueError):
            JevSettings(provider="typesafe")  # TypeSafe's API names the model jev-1.13
        with self.assertRaises(ValueError):
            JevSettings(provider="anthropic")
        JevSettings(provider="typesafe", model="jev-1.13")
        self.assertNotIn("api_key", TEST_SETTINGS.describe())


def keys_by_scoring(bank) -> dict[str, str]:
    """One calibrated item of each prediction schema: 3PL, 2PL, and GPCM worth 2 and 3 points."""
    calibrated = bank.calibrated
    kinds = {"3PL": ("3PL", 1), "2PL": ("2PL", 1), "GPCM2": ("GPCM", 2), "GPCM3": ("GPCM", 3)}
    return {
        kind: calibrated.index[(calibrated["irt_model"] == model) & (calibrated["scaled_points"] == points)][0]
        for kind, (model, points) in kinds.items()
    }


class JevPredictorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ItemIndex.build()
        cls.keys = keys_by_scoring(cls.index.bank)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()

    def test_committed_answers_are_reused(self):
        """The default state and questions must stay byte for byte those of the committed answers."""
        for with_references in (False, True):
            loo = LeaveOneOut(JevPredictor(self.index, with_references=with_references), self.keys.values())
            with self.subTest(with_references=with_references):
                self.assertEqual(set(loo.status()), {"saved"})

    def test_state_and_questions(self):
        key = self.keys["GPCM3"]
        item = self.index.bank.item(key)
        alone = JevPredictor(self.index, with_references=False).prepare(item, self.index.bank.parts[key])
        with_references = JevPredictor(self.index).prepare(item, self.index.bank.parts[key])
        self.assertEqual(list(alone["state"]), ["item"])
        self.assertNotIn("item_id", alone["state"]["item"])
        self.assertEqual(alone["state"]["item"]["calibration"], self.index.bank.calibrations[(item.assessment, item.grade)])
        self.assertEqual(alone["references"], [])
        references = with_references["state"]["reference_items"]
        self.assertTrue(references)
        self.assertFalse({entry["item_key"] for entry in with_references["references"]} & set(self.index.bank.parts[key]))
        self.assertTrue(all("calibrated_parameters" in reference for reference in references))
        self.assertEqual(list(alone["questions"]), ["slope", "location", "step1", "step2"])
        step1 = alone["questions"]["step1"]
        self.assertIn("b - d1 is the ability", step1.instructions)
        self.assertEqual(list(step1.criteria), list(DEFAULT_VALUE_SCALES["step1"].criteria))
        self.assertNotIn(REFERENCE_NOTE, step1.instructions)
        self.assertTrue(with_references["questions"]["step1"].instructions.endswith(REFERENCE_NOTE))

    def test_custom_scales_and_missing_scales(self):
        predictor = JevPredictor(self.index, scales={"slope": ["0.5", "1.5 (high)"]})
        item = self.index.bank.item(self.keys["2PL"])
        prepared = predictor.prepare(item)
        self.assertEqual(list(prepared["questions"]["slope"].criteria), ["0.5", "1.5 (high)"])
        self.assertEqual(prepared["questions"]["location"].criteria, list(DEFAULT_VALUE_SCALES["location"].criteria))
        self.assertRegex(predictor.run_name(), r"^jev-openrouter-typesafe_jev-1.13-eap-[0-9a-f]{8}__gemini-")
        self.assertEqual(JevPredictor(self.index, with_references=False).run_name(), "jev-openrouter-typesafe_jev-1.13-eap__item-alone")
        del predictor.scales["step2"]
        with self.assertRaisesRegex(ValueError, "step2"):
            predictor.prepare(self.index.bank.item(self.keys["GPCM3"]))
        with self.assertRaises(ValueError):
            JevPredictor(self.index, estimator="mean")

    def test_results_under_each_estimator(self):
        for estimator in ("eap", "median", "map"):
            loo = LeaveOneOut(JevPredictor(self.index, estimator=estimator), self.keys.values())
            results = loo.results()
            errors = results.errors
            with self.subTest(estimator=estimator):
                np.testing.assert_allclose(errors["predicted"], errors[estimator])
                self.assertEqual(set(errors["parameter"].astype(str)), {"slope", "location", "guessing", "step1", "step2"})
                gpcm2 = results.predictions.set_index("item_key").loc[self.keys["GPCM2"]]
                self.assertAlmostEqual(gpcm2["step2_predicted"], -gpcm2["step1_predicted"])
                gpcm3 = results.predictions.set_index("item_key").loc[self.keys["GPCM3"]]
                steps = [gpcm3[f"step{step}_predicted"] for step in (1, 2, 3)]
                self.assertAlmostEqual(sum(steps), 0.0)
                self.assertEqual(set(results.predictions["predictor"]), {"jev"})
                self.assertEqual(results.settings["jev.estimator"], estimator)
                # Relative to the data directory, so that a published workbook names no local path
                self.assertEqual(
                    results.settings["evaluation.cache_dir"], "cache/parameter_predictions/timss_math/openrouter/typesafe_jev-1.13"
                )

    def test_nothing_to_request_opens_no_session(self):
        loo = LeaveOneOut(JevPredictor(self.index), self.keys.values())  # every answer is committed
        with mock.patch.object(JevPredictor, "session", side_effect=AssertionError("a session was opened")):
            report = asyncio.run(loo.request_async(None))
        self.assertEqual((report.requested, report.saved), (0, 0))

    def test_batches_are_for_llms(self):
        loo = LeaveOneOut(JevPredictor(self.index), self.keys.values(), tempfile.mkdtemp())
        with self.assertRaises(TypeError):
            loo.request_batch(1)


def jev_response(request: httpx2.Request, total: float = 1.0) -> httpx2.Response:
    """A System One response to a request: every question answered with all its probability (``total``) on its middle
    criterion."""
    body = json.loads(request.content)
    answers = {}
    for name, question in body["questions"].items():
        criteria = question["criteria"]
        probabilities = {str(position): total if position == len(criteria) // 2 else 0.0 for position in range(len(criteria))}
        answers[name] = {
            "type": "score",
            "score": float(len(criteria) // 2),
            "confidence": 0.9,
            "legend": {str(position): criterion for position, criterion in enumerate(criteria)},
            "probabilities": probabilities,
        }
    usage = {"input_tokens": 1000, "output_tokens": 40, "cost": 0.0001}
    return httpx2.Response(200, json={"model": "typesafe/jev-1.13-20260917", "usage": usage, "answers": answers, "id": "gen-1"})


class JevRequestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ItemIndex.build()
        cls.keys = keys_by_scoring(cls.index.bank)
        cls.predictor = JevPredictor(cls.index, TEST_SETTINGS)
        cls.loo = LeaveOneOut(cls.predictor, cls.keys.values(), tempfile.mkdtemp())
        cls.loo.prepared  # noqa: B018 (prepares the states once for every test)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()

    def setUp(self):
        self.loo.cache.directory = Path(tempfile.mkdtemp())
        self.loo.batches = BatchManifests(self.loo.cache.directory / "batches")
        self.sent = []

    def run_requests(self, respond, workers=4):
        def handler(request: httpx2.Request) -> httpx2.Response:
            self.sent.append(json.loads(request.content))
            return respond(request)

        async def run():
            async with self.predictor.session(httpx2.AsyncClient(transport=httpx2.MockTransport(handler))) as session:
                return await self.loo.request_async(None, workers=workers, session=session)

        return asyncio.run(run())

    def stem_of(self, key):
        return self.loo.prepared[key]["state"]["item"]["stem"]

    def test_saves_answers_and_sends_the_state_and_questions(self):
        report = self.run_requests(jev_response)
        self.assertEqual((report.requested, report.saved, report.unusable), (4, 4, {}))
        self.assertAlmostEqual(report.cost_usd, 0.0004)
        self.assertEqual(set(self.loo.status()), {"saved"})
        identities = {json.dumps(self.loo.identity(key), sort_keys=True) for key in self.keys.values()}
        sent = {json.dumps({"provider": "openrouter", "model": body["model"], "state": body["state"], "questions": body["questions"]}, sort_keys=True) for body in self.sent}
        self.assertEqual(sent, identities)
        record = json.loads(self.loo.saved()[self.keys["3PL"]].read_text(encoding="utf-8"))
        self.assertEqual((record["cost_usd"], record["cost_source"], record["context"]), (0.0001, "OpenRouter response", "item+references"))
        self.assertIsNotNone(record["seconds"])
        results = self.loo.results()
        middle = {name: scale.values[len(scale.values) // 2] for name, scale in DEFAULT_VALUE_SCALES.items()}
        errors = results.errors.astype({"parameter": str})
        np.testing.assert_allclose(errors["predicted"], errors["parameter"].map(middle))

    def test_unusable_answers_are_left_for_a_rerun(self):
        unusable = self.keys["2PL"]

        def respond(request):
            body = json.loads(request.content)
            return jev_response(request, total=0.5 if body["state"]["item"]["stem"] == self.stem_of(unusable) else 1.0)

        report = self.run_requests(respond)
        self.assertEqual(list(report.unusable), [unusable])
        self.assertIn("sum to 0.500", report.unusable[unusable])
        self.assertEqual(self.loo.pending(None), [unusable])

    def test_api_error_stops_the_run(self):
        with self.assertRaises(TypeSafeError):
            self.run_requests(lambda request: httpx2.Response(402, json={"error": {"message": "Insufficient credits"}}), workers=1)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.loo.saved(), {})

    def test_rate_limit_is_retried(self):
        def respond(request):
            if len(self.sent) == 1:
                return httpx2.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "Rate limited"}})
            return jev_response(request)

        report = self.run_requests(respond, workers=1)
        self.assertEqual(report.saved, 4)
        self.assertEqual(len(self.sent), 5)


class CommandLineTest(unittest.TestCase):
    def parse_fails(self, argv):
        with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
            parse_args(argv)
        return stderr.getvalue()

    def test_jev_options(self):
        args = parse_args(["--predictor", "jev", "--jev-context", "item", "--estimator", "median"])
        self.assertEqual((args.jev_context, args.estimator, args.jev_model), ("item", "median", "typesafe/jev-1.13"))
        self.assertEqual(args.reasoning_effort, "medium")  # the LLM defaults stay in place, unused

    def test_options_of_the_other_predictor_are_errors(self):
        self.assertIn("--reasoning-effort apply to --predictor llm", self.parse_fails(["--predictor", "jev", "--reasoning-effort", "low"]))
        self.assertIn("--estimator apply to --predictor jev", self.parse_fails(["--estimator", "map"]))
        self.assertIn("--batch", self.parse_fails(["--predictor", "jev", "--batch"]))

    def test_llm_options_still_work(self):
        args = parse_args(["--reasoning-effort", "omit", "--llm-provider", "ollama"])
        self.assertIsNone(args.reasoning_effort)
        self.assertEqual(args.llm_provider, "ollama")
        self.assertTrue(math.isfinite(args.max_tokens))


if __name__ == "__main__":
    unittest.main()
