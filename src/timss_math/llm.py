"""LLMs of several providers, called through one client, with structured output.

``LLMSettings`` names an LLM and how it is called, and ``LLMClient.complete`` sends a system and a user prompt and
parses the output with a pydantic schema; ``LLMClient.session`` opens an ``AsyncLLMSession`` that does the same as a
coroutine, so that many requests can run at once in one event loop, and ``chat_body`` and ``parse_chat_response``
split a request for OpenRouter batches (``batch``). All of them send the same request body and check the output the
same way. The parameter predictions (``item_parameter_prediction``) and the test taking (``llm_test_taking``) both use
them.

LLM providers:

- ``"openrouter"`` (the default, with ``google/gemini-3.8-flash``): reasoning effort goes in OpenRouter's ``reasoning``
  object, and the response reports the cost.
- ``"openai"``: the OpenAI API, or any OpenAI-compatible server through ``base_url``.
- ``"gemini"``: the Gemini API's OpenAI-compatible endpoint, with models such as ``gemini-3.8-flash``.
- ``"anthropic"``: the Claude API's OpenAI-compatible endpoint, through the same OpenAI SDK, with models such as
  ``claude-haiku-5-5``. It follows the structured-output schema (``response_format``), but it ignores
  ``reasoning_effort`` (and Anthropic's ``output_config``) and rejects adaptive thinking, so the reasoning effort is
  None, which leaves thinking to the model (Claude 5 models think adaptively at their default effort), or "none",
  which turns thinking off (``thinking`` disabled; Claude Opus 5.5 and Sonnet 5.5 refuse it). Its responses report
  neither the cost nor the reasoning tokens (``completion_tokens`` includes them), and never the thinking.
- ``"ollama"``: a local Ollama server through its native chat API, which takes the JSON schema as ``format``, the
  reasoning effort as ``think`` (a level such as "medium", or True or False for models without levels; None for
  models that cannot think), and the context window as ``num_ctx``.
"""

from __future__ import annotations

import contextlib
import functools
import json
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from typing import Literal

import httpx2
import openai
import requests
from openai.lib._parsing._completions import (
    parse_chat_completion,
    type_to_response_format_param,
)
from openai.types.chat import ChatCompletion, ParsedChatCompletion
from pydantic import BaseModel, ValidationError

from .config import api_key
from .running import UnusableOutputError

LLMProvider = Literal["openrouter", "openai", "gemini", "anthropic", "ollama"]

BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1/",
    "ollama": "http://localhost:11434",
}
API_KEY_VARIABLES = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
ANTHROPIC_REASONING_EFFORTS = (None, "none")
"""What Anthropic's OpenAI-compatible endpoint can set: the model's own thinking (None), or no thinking ("none")."""


@dataclass(frozen=True)
class LLMSettings:
    """Which LLM answers, and how it is called."""

    provider: LLMProvider = "openrouter"
    model: str = "google/gemini-3.8-flash"
    reasoning_effort: str | bool | None = "medium"
    """Reasoning effort, such as "low", "medium", or "high"; None sends none. Ollama also takes True or False;
    Anthropic takes only None (the model's own thinking) or "none" (no thinking)."""
    max_tokens: int = 16384
    """Output cap, reasoning included."""
    base_url: str | None = None
    """The provider's API URL; None uses its public one (Ollama: http://localhost:11434)."""
    api_key: str | None = field(default=None, repr=False)
    """None reads the provider's variable (OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY)
    from the environment or the project .env. Ollama needs none."""
    timeout: float = 600
    max_retries: int = 2
    """Retries of a failed request by the OpenAI client (not used for Ollama)."""
    ollama_num_ctx: int = 32768
    """Ollama context window in tokens, which must hold the prompts (about 6,000 tokens for a parameter prediction)
    and the output."""

    def __post_init__(self):
        if self.provider not in BASE_URLS:
            raise ValueError(f"Unknown LLM provider {self.provider!r}; use one of {sorted(BASE_URLS)}")
        if isinstance(self.reasoning_effort, bool) and self.provider != "ollama":
            raise ValueError("Only Ollama takes True or False as the reasoning effort")
        if self.provider == "anthropic" and self.reasoning_effort not in ANTHROPIC_REASONING_EFFORTS:
            raise ValueError(
                f"Anthropic's OpenAI-compatible API cannot set the reasoning effort {self.reasoning_effort!r}: it "
                "ignores reasoning_effort and rejects adaptive thinking. Use None, which leaves thinking to the model, "
                "or 'none', which turns it off"
            )

    def describe(self) -> dict:
        return {name: value for name, value in asdict(self).items() if name != "api_key"}


class NoPredictionError(UnusableOutputError):
    """The LLM returned no usable output, such as output cut off at max_tokens; a rerun may succeed. (Its name is from
    the parameter predictions, the first use of the client.)"""


class ContextWindowError(RuntimeError):
    """The prompts and output do not fit the Ollama context window; raise ollama_num_ctx."""


@dataclass
class Completion:
    parsed: BaseModel
    thinking: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None


class LLMClient:
    """Sends a system and a user prompt to the LLM of an LLMSettings and parses its output with a pydantic schema."""

    def __init__(self, settings: LLMSettings = LLMSettings()):
        self.settings = settings
        self.base_url = (settings.base_url or BASE_URLS[settings.provider]).rstrip("/")

    @functools.cached_property
    def _client(self) -> openai.OpenAI:
        """Made when the first request is sent, so that preparing prompts and reading saved predictions need no API
        key."""
        return openai.OpenAI(**self._openai_options())

    def _openai_options(self) -> dict:
        settings = self.settings
        return {
            "base_url": self.base_url,
            "api_key": api_key(API_KEY_VARIABLES[settings.provider], settings.api_key),
            "max_retries": settings.max_retries,
            "timeout": settings.timeout,
        }

    @contextlib.asynccontextmanager
    async def session(self, http_client: httpx2.AsyncClient | None = None) -> AsyncIterator[AsyncLLMSession]:
        """An ``AsyncLLMSession`` for the requests of one event loop, closed at the end.

        A session is opened for each run instead of once per client, because an async connection pool belongs to the
        event loop that opened it. ``http_client`` replaces the HTTP client, such as one with a mock transport in tests.
        """
        if self.settings.provider == "ollama":
            async with http_client or httpx2.AsyncClient(timeout=self.settings.timeout) as http:
                yield AsyncLLMSession(self, http=http)
        else:
            async with openai.AsyncOpenAI(**self._openai_options(), http_client=http_client) as client:
                yield AsyncLLMSession(self, openai_client=client)

    def complete(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> Completion:
        """The LLM's output parsed with ``schema``.

        Output that does not fit the schema raises pydantic's ValidationError, output cut off at max_tokens raises
        openai.LengthFinishReasonError (or NoPredictionError from Ollama), and no output at all raises
        NoPredictionError. API errors propagate.
        """
        messages = messages_of(system_prompt, user_prompt)
        if self.settings.provider == "ollama":
            return self._ollama(messages, schema)
        options, extra_body = self._options()
        completion = self._client.chat.completions.parse(
            model=self.settings.model, messages=messages, response_format=schema, **options, extra_body=extra_body
        )
        return completion_of(completion)

    def _options(self) -> tuple[dict, dict]:
        """The output cap and reasoning effort of an OpenAI-compatible request: the SDK's own parameters, and the
        provider's extra body fields."""
        settings = self.settings
        if settings.provider == "ollama":
            raise ValueError("Ollama is not called through the OpenAI-compatible chat completions API")
        effort = settings.reasoning_effort
        if settings.provider == "openrouter":
            reasoning = {"reasoning": {"effort": effort}} if effort is not None else {}
            return {"max_tokens": settings.max_tokens}, reasoning | {"usage": {"include": True}}
        if settings.provider == "anthropic":
            # Anthropic's thinking setting, which its OpenAI-compatible API takes in place of reasoning_effort.
            thinking = {"thinking": {"type": "disabled"}} if effort == "none" else {}
            return {"max_tokens": settings.max_tokens}, thinking
        options = {"max_completion_tokens" if settings.provider == "openai" else "max_tokens": settings.max_tokens}
        if effort is not None:
            options["reasoning_effort"] = effort
        return options, {}

    def chat_body(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> dict:
        """The JSON body that ``complete`` sends to an OpenAI-compatible provider, for a batch request."""
        options, extra_body = self._options()
        return {
            "messages": messages_of(system_prompt, user_prompt),
            "model": self.settings.model,
            **options,
            "response_format": type_to_response_format_param(schema),
            **extra_body,
        }

    @staticmethod
    def parse_chat_response(body: dict, schema: type[BaseModel]) -> Completion:
        """A chat-completions response body, such as a batch result, parsed and checked as ``complete`` does."""
        completion = parse_chat_completion(
            response_format=schema, input_tools=openai.omit, chat_completion=ChatCompletion.construct(**body)
        )
        return completion_of(completion)

    def _ollama(self, messages: list[dict], schema: type[BaseModel]) -> Completion:
        response = requests.post(
            f"{self.base_url}/api/chat", json=self._ollama_body(messages, schema), timeout=self.settings.timeout
        )
        return self._ollama_completion(response.status_code, response.text, schema)

    def _ollama_body(self, messages: list[dict], schema: type[BaseModel]) -> dict:
        settings = self.settings
        body = {
            "model": settings.model,
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "options": {"num_ctx": settings.ollama_num_ctx, "num_predict": settings.max_tokens},
        }
        if settings.reasoning_effort is not None:
            body["think"] = settings.reasoning_effort
        return body

    def _ollama_completion(self, status_code: int, text: str, schema: type[BaseModel]) -> Completion:
        """The output of an Ollama chat response, checked for errors, a full context window, and missing output."""
        settings = self.settings
        if not 200 <= status_code < 300:
            hint = "; set reasoning_effort=None" if "does not support thinking" in text else ""
            raise RuntimeError(f"Ollama {settings.model} returned HTTP {status_code}: {text}{hint}")
        result = json.loads(text)
        prompt_tokens, completion_tokens = result.get("prompt_eval_count"), result.get("eval_count")
        # Ollama cuts a prompt that does not fit the context window instead of failing.
        if prompt_tokens is not None and prompt_tokens + (completion_tokens or 0) >= settings.ollama_num_ctx:
            raise ContextWindowError(
                f"{prompt_tokens} prompt and {completion_tokens} output tokens fill the context window of "
                f"{settings.ollama_num_ctx}; raise ollama_num_ctx"
            )
        if result.get("done_reason") == "length":
            raise NoPredictionError(f"Output cut off at max_tokens ({settings.max_tokens})")
        content = result["message"].get("content")
        if not content:
            raise NoPredictionError(f"No prediction (done_reason={result.get('done_reason')})")
        return Completion(
            parsed=schema.model_validate_json(content),
            thinking=result["message"].get("thinking"),
            finish_reason=result.get("done_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=None,
            cost_usd=None,
        )


class AsyncLLMSession:
    """``LLMClient.complete`` as a coroutine, on the HTTP client of one event loop (``LLMClient.session``)."""

    def __init__(
        self, llm: LLMClient, openai_client: openai.AsyncOpenAI | None = None, http: httpx2.AsyncClient | None = None
    ):
        self.llm = llm
        self.openai_client = openai_client
        self.http = http

    async def complete(self, system_prompt: str, user_prompt: str, schema: type[BaseModel]) -> Completion:
        """The LLM's output parsed with ``schema``; raises as ``LLMClient.complete`` does."""
        llm = self.llm
        messages = messages_of(system_prompt, user_prompt)
        if llm.settings.provider == "ollama":
            response = await self.http.post(f"{llm.base_url}/api/chat", json=llm._ollama_body(messages, schema))
            return llm._ollama_completion(response.status_code, response.text, schema)
        options, extra_body = llm._options()
        completion = await self.openai_client.chat.completions.parse(
            model=llm.settings.model, messages=messages, response_format=schema, **options, extra_body=extra_body
        )
        return completion_of(completion)


def messages_of(system_prompt: str, user_prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def completion_of(completion: ParsedChatCompletion) -> Completion:
    """The parsed output, reasoning, and usage of an OpenAI-compatible response; NoPredictionError if none parsed."""
    choice = completion.choices[0]
    if choice.message.parsed is None:
        raise NoPredictionError(f"No prediction (finish_reason={choice.finish_reason}): {choice.message.refusal}")
    usage = completion.usage
    details = usage.completion_tokens_details if usage else None
    return Completion(
        parsed=choice.message.parsed,
        # The reasoning summary that OpenRouter returns for a reasoning model, if any.
        thinking=getattr(choice.message, "reasoning", None),
        finish_reason=choice.finish_reason,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        reasoning_tokens=details.reasoning_tokens if details else None,
        cost_usd=getattr(usage, "cost", None),
    )


UNUSABLE_ERRORS = (
    ValidationError,
    openai.LengthFinishReasonError,
    openai.ContentFilterFinishReasonError,
    UnusableOutputError,
)
"""Output that the LLM may get right on a rerun: output that does not fit the schema, is cut off, or is missing."""


def parse_reasoning_effort(value: str) -> str | bool | None:
    """A reasoning effort given as text, as on the command line: "omit" is None, "true" and "false" are booleans."""
    return {"omit": None, "true": True, "false": False}.get(value.lower(), value)
