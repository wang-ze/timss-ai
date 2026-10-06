"""Leave-one-out requests as coroutines: the async LLM session sends what the sync client sends, and
LeaveOneOut.request_async bounds, saves, stops, and cancels its requests. No request leaves this computer.

Run with ``uv run python -m unittest discover tests``.
"""

import asyncio
import json
import tempfile
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
from timss_math.item_parameter_prediction.evaluation import run_sync
from timss_math.item_parameter_prediction.predictor import (
    Completion,
    ContextWindowError,
    LLMClient,
    messages_of,
    prediction_schema,
)

SCHEMAS = [("3PL", 1), ("2PL", 1), ("GPCM", 2), ("GPCM", 3)]
OLLAMA = LLMSettings(provider="ollama", model="llama3.2", reasoning_effort=None, max_tokens=4096)


def valid_output(schema) -> dict:
    return {name: "why" if field.annotation is str else 0.5 for name, field in schema.model_fields.items()}


def chat_response(schema) -> dict:
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": "google/gemini-3.8-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(valid_output(schema))},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.001},
    }


def ollama_response(content: str | None, done_reason: str = "stop", prompt_tokens: int = 6000) -> dict:
    return {
        "message": {"role": "assistant", "content": content},
        "done_reason": done_reason,
        "prompt_eval_count": prompt_tokens,
        "eval_count": 300,
    }


class AsyncSessionTest(unittest.TestCase):
    def test_same_body_as_the_sync_client(self):
        """The async session must send exactly what LLMClient.complete sends."""
        for settings in (
            LLMSettings(api_key="test"),
            LLMSettings(reasoning_effort=None, api_key="test"),
            LLMSettings(provider="gemini", model="gemini-3.8-flash", api_key="test"),
        ):
            for irt_model, points in SCHEMAS:
                self.check_same_body(settings, prediction_schema(irt_model, points))

    def check_same_body(self, settings: LLMSettings, schema):
        sent = {"sync": [], "async": []}

        def handler(mode):
            def respond(request: httpx2.Request) -> httpx2.Response:
                sent[mode].append((request.url.path, json.loads(request.content)))
                return httpx2.Response(200, json=chat_response(schema))

            return respond

        client = LLMClient(settings)
        client.__dict__["_client"] = openai.OpenAI(
            base_url=client.base_url,
            api_key="test",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handler("sync"))),
        )
        sync_completion = client.complete("system", "user", schema)

        async def complete_async():
            http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler("async")))
            async with client.session(http_client) as session:
                return await session.complete("system", "user", schema)

        async_completion = asyncio.run(complete_async())
        self.assertEqual(sent["async"], sent["sync"])
        self.assertEqual(async_completion, sync_completion)

    def complete_ollama(self, status_code: int, body: dict | str):
        schema = prediction_schema("2PL", 1)
        sent = []

        def respond(request: httpx2.Request) -> httpx2.Response:
            sent.append(json.loads(request.content))
            content = body if isinstance(body, str) else json.dumps(body)
            return httpx2.Response(status_code, text=content)

        async def complete_async():
            async with LLMClient(OLLAMA).session(httpx2.AsyncClient(transport=httpx2.MockTransport(respond))) as session:
                return await session.complete("system", "user", schema)

        try:
            return asyncio.run(complete_async())
        finally:
            self.assertEqual(sent, [LLMClient(OLLAMA)._ollama_body(messages_of("system", "user"), schema)])

    def test_ollama(self):
        completion = self.complete_ollama(200, ollama_response(json.dumps(valid_output(prediction_schema("2PL", 1)))))
        self.assertEqual((completion.parsed.slope, completion.prompt_tokens, completion.cost_usd), (0.5, 6000, None))

    def test_ollama_cut_off(self):
        with self.assertRaises(NoPredictionError):
            self.complete_ollama(200, ollama_response('{"reasoning": "', done_reason="length"))

    def test_ollama_full_context(self):
        with self.assertRaises(ContextWindowError):
            self.complete_ollama(200, ollama_response("{}", prompt_tokens=32768))

    def test_ollama_http_error(self):
        with self.assertRaisesRegex(RuntimeError, "set reasoning_effort=None"):
            self.complete_ollama(400, '{"error": "llama3.2 does not support thinking"}')


class FakeSession:
    """An AsyncLLMSession whose requests take ``delay`` seconds and answer valid output. ``answers`` maps a user
    prompt to an exception to raise instead, or to (exception or None, delay)."""

    def __init__(self, delay=0.01, answers=None):
        self.delay = delay
        self.answers = answers or {}
        self.started: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, system_prompt, user_prompt, schema):
        self.started.append(user_prompt)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            answer = self.answers.get(user_prompt)
            await asyncio.sleep(answer[1] if isinstance(answer, tuple) else self.delay)
        except asyncio.CancelledError:
            self.cancelled.append(user_prompt)
            raise
        finally:
            self.in_flight -= 1
        self.finished.append(user_prompt)
        error = answer[0] if isinstance(answer, tuple) else answer
        if error is not None:
            raise error
        return Completion(schema.model_validate(valid_output(schema)), None, "stop", 100, 50, 20, 0.001)


class RequestAsyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = ItemIndex.build()
        cls.predictor = ParameterPredictor(cls.index, LLMSettings(api_key="test"))
        calibrated = cls.index.bank.calibrated
        cls.keys = [
            key
            for irt_model, points in SCHEMAS
            for key in calibrated.index[(calibrated["irt_model"] == irt_model) & (calibrated["scaled_points"] == points)][:3]
        ]
        cls.loo = LeaveOneOut(cls.predictor, cls.keys, tempfile.mkdtemp(), reuse_notebook_predictions=False)
        cls.loo.prepared  # noqa: B018 (prepares the prompts once for every test)

    @classmethod
    def tearDownClass(cls):
        cls.index.close()

    def setUp(self):
        self.loo.cache.directory = Path(tempfile.mkdtemp())
        self.loo.batches = BatchManifests(self.loo.cache.directory / "batches")

    def prompt(self, key):
        return self.loo.prepared[key]["user_prompt"]

    def test_saves_every_prediction_within_the_worker_bound(self):
        session = FakeSession()
        report = asyncio.run(self.loo.request_async(None, workers=3, session=session))
        self.assertEqual((report.requested, report.saved, report.unusable), (len(self.keys), len(self.keys), {}))
        self.assertAlmostEqual(report.cost_usd, 0.001 * len(self.keys))
        self.assertEqual(session.max_in_flight, 3)
        self.assertEqual(set(self.loo.status()), {"saved"})
        self.assertEqual(self.loo.pending(None), [])

    def test_unusable_output_is_left_for_a_rerun(self):
        unusable = self.keys[1]
        session = FakeSession(answers={self.prompt(unusable): NoPredictionError("cut off")})
        report = asyncio.run(self.loo.request_async(None, workers=4, session=session))
        self.assertEqual(list(report.unusable), [unusable])
        self.assertEqual(report.saved, len(self.keys) - 1)
        self.assertEqual(self.loo.pending(None), [unusable])

    def test_api_error_saves_running_requests_and_sends_no_more(self):
        # The first request fails at once; the other two running requests finish later and must be saved.
        failing, running = self.keys[0], self.keys[1:3]
        response = httpx2.Response(402, request=httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
        error = openai.APIStatusError("Insufficient credits", response=response, body=None)
        session = FakeSession(delay=0.05, answers={self.prompt(failing): (error, 0.0)})
        with self.assertRaises(openai.APIStatusError):
            asyncio.run(self.loo.request_async(None, workers=3, session=session))
        self.assertEqual(len(session.started), 3)
        self.assertEqual(sorted(self.loo.saved()), sorted(running))
        self.assertEqual(session.cancelled, [])

    def test_cancelling_the_run_cancels_running_requests(self):
        session = FakeSession(delay=10)

        async def cancel_soon():
            run = asyncio.create_task(self.loo.request_async(None, workers=2, session=session))
            await asyncio.sleep(0.05)
            run.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await run

        asyncio.run(cancel_soon())
        self.assertEqual(len(session.started), 2)
        self.assertEqual(len(session.cancelled), 2)
        self.assertEqual(self.loo.saved(), {})

    def test_request_runs_inside_a_running_loop(self):
        """As in a notebook: the sync request() inside a running event loop."""

        async def notebook_cell():
            return run_sync(self.loo.request_async(2, workers=2, session=FakeSession()))

        report = asyncio.run(notebook_cell())
        self.assertEqual(report.saved, 2)


if __name__ == "__main__":
    unittest.main()
