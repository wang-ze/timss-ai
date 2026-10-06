"""Batch requests: the same request body and parsing as single requests, and LeaveOneOut.request_batch against a fake
OpenRouter Batch API. No request leaves this computer.

Run with ``uv run python -m unittest discover tests``.
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import httpx2
import openai

from timss_math.item_parameter_prediction import (
    ItemIndex,
    LeaveOneOut,
    LLMSettings,
    NoPredictionError,
    ParameterPredictor,
)
from timss_math.item_parameter_prediction.batch import BatchManifests
from timss_math.item_parameter_prediction.predictor import LLMClient, prediction_schema

SCHEMAS = [("3PL", 1), ("2PL", 1), ("GPCM", 2), ("GPCM", 3)]


def valid_output(schema) -> dict:
    return {name: "why" if field.annotation is str else 0.5 for name, field in schema.model_fields.items()}


def response_body(content: str | None, finish_reason: str = "stop") -> dict:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": "google/gemini-3.8-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "reasoning": "thinking"},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.001,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    }


class RequestBodyTest(unittest.TestCase):
    def test_batch_body_is_the_single_request_body(self):
        """chat_body must be exactly what complete sends, so batch and single predictions are interchangeable."""
        for settings in (
            LLMSettings(api_key="test"),
            LLMSettings(reasoning_effort=None, api_key="test"),
            LLMSettings(provider="openai", model="gpt-5", api_key="test"),
        ):
            for irt_model, points in SCHEMAS:
                self.check_batch_body(settings, prediction_schema(irt_model, points))

    def check_batch_body(self, settings: LLMSettings, schema):
        sent = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            sent.append(json.loads(request.content))
            return httpx2.Response(200, json=response_body(json.dumps(valid_output(schema))))

        client = LLMClient(settings)
        client.__dict__["_client"] = openai.OpenAI(
            base_url=client.base_url, api_key="test", http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
        )
        completion = client.complete("system", "user", schema)
        self.assertEqual(completion.cost_usd, 0.001)
        expected = {name: value for name, value in sent[0].items() if name != "stream"}
        self.assertEqual(client.chat_body("system", "user", schema), expected)


class ParseResponseTest(unittest.TestCase):
    schema = prediction_schema("3PL", 1)

    def parse(self, body):
        return LLMClient.parse_chat_response(body, self.schema)

    def test_valid(self):
        completion = self.parse(response_body(json.dumps(valid_output(self.schema))))
        self.assertEqual(completion.parsed.slope, 0.5)
        self.assertEqual(
            (completion.thinking, completion.prompt_tokens, completion.reasoning_tokens, completion.cost_usd),
            ("thinking", 100, 20, 0.001),
        )

    def test_cut_off(self):
        with self.assertRaises(openai.LengthFinishReasonError):
            self.parse(response_body('{"reasoning": "', finish_reason="length"))

    def test_no_output(self):
        with self.assertRaises(NoPredictionError):
            self.parse(response_body(None))

    def test_invalid_output(self):
        with self.assertRaises(Exception) as caught:
            self.parse(response_body(json.dumps(valid_output(self.schema) | {"guessing": 2.0})))
        self.assertIn("ValidationError", type(caught.exception).__name__)


class FakeBatches:
    """OpenRouter's Batch API in memory: each batch is in progress on its first get and terminal after."""

    def __init__(self, final_status="completed", answers=None):
        self.final_status = final_status
        self.answers = answers or {}  # custom_id -> result to return instead of a valid response
        self.submitted: dict[str, list[dict]] = {}
        self.gets: dict[str, int] = {}

    def submit(self, batch_requests):
        batch_id = f"batch_{len(self.submitted) + 1}"
        self.submitted[batch_id] = batch_requests
        self.gets[batch_id] = 0
        return self.batch(batch_id, "validating")

    def batch(self, batch_id, status):
        return {
            "id": batch_id,
            "model": "google/gemini-3.8-flash",
            "status": status,
            "request_counts": {"total": len(self.submitted[batch_id]), "completed": 0, "failed": 0},
            "usage": {"cost": 0.01} if status == "completed" else None,
            "results": None,
            "error": None,
        }

    def get(self, batch_id):
        self.gets[batch_id] += 1
        if self.gets[batch_id] == 1:
            return self.batch(batch_id, "in_progress")
        batch = self.batch(batch_id, self.final_status)
        if self.final_status == "completed":
            batch["results"] = [self.result(request) for request in self.submitted[batch_id]]
        return batch

    def result(self, request):
        custom_id = request["custom_id"]
        if custom_id in self.answers:
            return {"custom_id": custom_id} | self.answers[custom_id]
        schema = request["body"]["response_format"]["json_schema"]["schema"]
        output = {name: "why" if spec["type"] == "string" else 0.5 for name, spec in schema["properties"].items()}
        body = response_body(json.dumps(output))
        return {"custom_id": custom_id, "response": {"status_code": 200, "body": body}, "error": None}

    def list_created_after(self, created_after):
        return [self.batch(batch_id, "in_progress") for batch_id in self.submitted]


class RequestBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ItemIndex.build()
        cls.predictor = ParameterPredictor(cls.index, LLMSettings(api_key="test"))
        calibrated = cls.index.bank.calibrated
        # Two items of each prediction schema.
        cls.keys = [
            key
            for irt_model, points in SCHEMAS
            for key in calibrated.index[(calibrated["irt_model"] == irt_model) & (calibrated["scaled_points"] == points)][:2]
        ]
        cls.loo = LeaveOneOut(cls.predictor, cls.keys, tempfile.mkdtemp(), reuse_notebook_predictions=False)
        cls.loo.prepared  # noqa: B018 (prepares the prompts once for every test)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()

    def setUp(self):
        self.loo.cache.directory = Path(tempfile.mkdtemp())
        self.loo.batches = BatchManifests(self.loo.cache.directory / "batches")

    def test_submit_then_collect(self):
        fake = FakeBatches()
        report = self.loo.request_batch(None, wait=False, client=fake)
        self.assertEqual(report.requested, len(self.keys))
        self.assertEqual(report.in_flight, len(self.keys))
        # One batch per schema, each with one response_format.
        self.assertEqual(len(fake.submitted), len(SCHEMAS))
        for batch_requests in fake.submitted.values():
            self.assertEqual(len({json.dumps(request["body"]["response_format"]) for request in batch_requests}), 1)
        self.assertEqual(set(self.loo.status()), {"in a submitted batch"})
        self.assertEqual(self.loo.pending(None), [])

        # A later run collects without submitting again.
        report = self.loo.request_batch(5, wait=True, poll_seconds=0, client=fake)
        self.assertEqual(len(fake.submitted), len(SCHEMAS))
        self.assertEqual((report.requested, report.saved, report.in_flight), (0, len(self.keys), 0))
        self.assertAlmostEqual(report.cost_usd, 0.01 * len(SCHEMAS))
        self.assertEqual(set(self.loo.status()), {"saved"})
        for key in self.keys:
            record = json.loads(self.loo.cache.path(key, self.loo.identity(key)).read_text())
            self.assertTrue(record["batch_id"].startswith("batch_"))
            self.assertEqual(record["user_prompt"], self.loo.prepared[key]["user_prompt"])
        self.assertEqual(len(self.loo.batches.open()), 0)
        self.assertEqual(len(list((self.loo.batches.directory / "done").glob("*.json"))), len(SCHEMAS))
        self.assertEqual(len(self.loo.results().predictions), len(self.keys))

    def test_failed_and_unusable_requests_are_pending_again(self):
        error_key, unusable_key = self.keys[0], self.keys[1]
        answers = {
            self.loo.cache.stem(error_key, self.loo.identity(error_key)): {
                "response": None,
                "error": {"type": "provider_error", "message": "unavailable"},
            },
            self.loo.cache.stem(unusable_key, self.loo.identity(unusable_key)): {
                "response": {"status_code": 200, "body": response_body('{"reasoning": "', finish_reason="length")},
                "error": None,
            },
        }
        report = self.loo.request_batch(None, wait=True, poll_seconds=0, client=FakeBatches(answers=answers))
        self.assertEqual(list(report.failed), [error_key])
        self.assertEqual(list(report.unusable), [unusable_key])
        self.assertEqual(report.saved, len(self.keys) - 2)
        self.assertEqual(self.loo.pending(None), [error_key, unusable_key])

    def test_expired_batch(self):
        report = self.loo.request_batch(1, wait=True, poll_seconds=0, client=FakeBatches(final_status="expired"))
        self.assertEqual((report.requested, report.saved, list(report.failed)), (1, 0, self.keys[:1]))
        self.assertEqual(self.loo.pending(None), self.keys)

    def test_unconfirmed_submission_is_adopted(self):
        fake = FakeBatches()
        self.loo.request_batch(1, wait=False, client=fake)
        # As if the run had stopped after OpenRouter accepted the batch and before the manifest was renamed.
        manifest_path = next(self.loo.batches.directory.glob("batch_1.json"))
        manifest = json.loads(manifest_path.read_text())
        manifest_path.unlink()
        self.loo.batches.start(manifest["model"], manifest["requests"])
        report = self.loo.request_batch(0, wait=True, poll_seconds=0, client=fake)
        self.assertEqual((len(fake.submitted), report.saved), (1, 1))

    def test_unreceived_submission_is_dropped(self):
        key = self.keys[0]
        entry = {"item_key": key, "record": {}}
        self.loo.batches.start("google/gemini-3.8-flash", {self.loo.cache.stem(key, self.loo.identity(key)): entry})
        fake = FakeBatches()
        self.loo.request_batch(0, wait=False, client=fake)
        self.assertEqual(list(self.loo.batches.directory.glob("*.json")), [])
        self.assertIn(key, self.loo.pending(None))

    def test_waits_while_another_process_submits_or_collects(self):
        held, release = threading.Event(), threading.Event()

        def hold_lock():  # a second open file of the lock conflicts like another process's
            with BatchManifests(self.loo.batches.directory).locked():
                held.set()
                release.wait(5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        self.assertTrue(held.wait(5))
        started = time.monotonic()
        threading.Timer(0.3, release.set).start()
        with self.assertLogs("timss_math.item_parameter_prediction.batch", "INFO") as logs:
            report = self.loo.request_batch(1, wait=False, client=FakeBatches())
        holder.join()
        self.assertGreaterEqual(time.monotonic() - started, 0.25)
        self.assertIn("Waiting for another process", "\n".join(logs.output))
        self.assertEqual(report.requested, 1)


class ManifestLockTest(unittest.TestCase):
    def test_another_process_holds_the_lock_until_it_releases_it(self):
        holder_code = (
            "import sys\n"
            "from timss_math.item_parameter_prediction.batch import BatchManifests\n"
            "with BatchManifests(sys.argv[1]).locked():\n"
            "    print('held', flush=True)\n"
            "    sys.stdin.read()\n"  # until the test closes stdin
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            # Leaving the block closes stdin, which ends the holder even if the test fails, and waits for it.
            subprocess.Popen(
                [sys.executable, "-c", holder_code, directory], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            ) as holder,
        ):
            self.assertEqual(holder.stdout.readline().strip(), "held")
            threading.Timer(0.3, holder.stdin.close).start()
            started = time.monotonic()
            with (
                self.assertLogs("timss_math.item_parameter_prediction.batch", "INFO") as logs,
                BatchManifests(Path(directory)).locked(),
            ):
                waited = time.monotonic() - started
            self.assertGreaterEqual(waited, 0.25)
            self.assertIn("Waiting for another process", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
