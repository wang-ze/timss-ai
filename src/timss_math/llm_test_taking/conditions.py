"""Conditions: an LLM with its settings, and the role in which it takes the test.

The LLM is the package's ``LLMSettings``: its provider (``"openrouter"``, ``"openai"``, ``"gemini"``, ``"anthropic"``,
or ``"ollama"``), model, reasoning effort, ``max_tokens`` (16,384 by default, reasoning included), and how it is called.
The role is ``"test_taker"`` (the test written for students of a grade, evaluating the LLM) or ``"student"`` (as a
student of that grade, the LLM as a participant); a condition's label ends in ", as a student" in the student role.

``DEFAULT_MODELS`` are the five LLMs of the run of 2026-10-19, each in both roles, so ten conditions, and
``DEFAULT_SCORER`` the LLM that scored their constructed responses; their saved answers and verdicts reproduce that
run without any request.

Providers other than OpenRouter report the tokens of a request but not its cost, so ``direct_cost`` computes it from
the tokens at the prices of ``TOKEN_PRICES`` (thinking tokens are billed as output), or leaves it empty for a model
without a price there (a local Ollama model costs nothing).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import slug
from ..llm import LLMSettings
from .answering import ROLES, Role

DEFAULT_MODELS = (
    LLMSettings(model="openai/gpt-6-luna", reasoning_effort="none"),
    LLMSettings(model="openai/gpt-6-luna", reasoning_effort="high"),
    LLMSettings(model="google/gemma-3-12b-it", reasoning_effort=None),
    # Claude Haiku 5.5 through the Claude API (ANTHROPIC_API_KEY), with thinking off ("none") or left to the model
    # (None, adaptive at its default effort).
    LLMSettings(provider="anthropic", model="claude-haiku-5-5", reasoning_effort="none"),
    LLMSettings(provider="anthropic", model="claude-haiku-5-5", reasoning_effort=None),
)
"""The LLMs of the run of 2026-10-19, which took the test in both roles."""
DEFAULT_SCORER = LLMSettings(model="google/gemini-3.8-flash", reasoning_effort="medium", max_tokens=8192)
"""The scorer of the run of 2026-10-19. Its max_tokens is lower than the default, because OpenRouter reserves a batch's
worst case, every request using all of it."""
TOKEN_PRICES = {("anthropic", "claude-haiku-5-5"): (0.10, 0.50)}
"""Prices per million tokens (input, output) of the models whose provider reports the tokens of a request but not its
cost; the Claude API's prices of Claude Haiku 5.5 hold for prompts up to 100,000 tokens."""


@dataclass(frozen=True)
class Condition:
    """An LLM with its settings, and the role in which it takes the test."""

    llm: LLMSettings
    role: Role = "test_taker"

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"Unknown role {self.role!r}; use one of {ROLES}")

    @property
    def label(self) -> str:
        """The model without its vendor, its provider unless OpenRouter, its reasoning effort, and the student role,
        such as "gpt-6-luna, high reasoning" or "claude-haiku-5-5 (anthropic), no reasoning, as a student"."""
        effort = self.llm.reasoning_effort
        parts = [self.llm.model.split("/")[-1]]
        if self.llm.provider != "openrouter":
            parts[0] += f" ({self.llm.provider})"
        if effort is not None:
            parts.append("no reasoning" if effort in ("none", False) else f"{effort} reasoning")
        elif self.llm.provider == "anthropic":  # Claude thinks unless it is told not to
            parts.append("default thinking")
        if self.role == "student":
            parts.append("as a student")
        return ", ".join(parts)

    def cache_dir(self, answers_dir: Path) -> Path:
        """Where the condition's answers are saved: ``<answers_dir>/<provider>/<model>/<reasoning effort>/<role>``."""
        return settings_folder(answers_dir, self.llm) / self.role


def settings_folder(directory: Path, settings: LLMSettings) -> Path:
    """``<directory>/<provider>/<model>/<reasoning effort>``, where no reasoning setting (None) is "default": a file
    system that ignores case, such as macOS's, would take a folder "None" for the folder "none" of the same model."""
    effort = "default" if settings.reasoning_effort is None else str(settings.reasoning_effort)
    return Path(directory) / slug(settings.provider) / slug(settings.model) / slug(effort)


def conditions_of(models: Iterable[LLMSettings], roles: Iterable[Role] = ROLES) -> list[Condition]:
    """Every model in every role, the roles in turn: the models as test takers, then as students."""
    return [Condition(llm, role) for role in roles for llm in models]


def by_label(conditions: Iterable[Condition]) -> dict[str, Condition]:
    """The conditions by label; ValueError if two have the same label."""
    conditions = list(conditions)
    labelled = {condition.label: condition for condition in conditions}
    if len(labelled) != len(conditions):
        raise ValueError("Two conditions have the same label")
    return labelled


def direct_cost(settings: LLMSettings, records: pd.DataFrame, token_prices: dict = TOKEN_PRICES) -> float:
    """The cost of the direct requests of a condition: as OpenRouter reports it, 0 for a local Ollama model, from the
    tokens at ``token_prices`` for the other providers, or NaN for a model without a price."""
    if settings.provider in ("openrouter", "ollama"):
        return records["cost_usd"].fillna(0).sum()
    if (settings.provider, settings.model) not in token_prices:
        return np.nan
    input_price, output_price = token_prices[settings.provider, settings.model]
    return (records["prompt_tokens"].sum() * input_price + records["completion_tokens"].sum() * output_price) / 1e6


def batch_cost(folder: Path) -> float:
    """The cost of the collected batches of a folder: OpenRouter reports it per batch, not per request."""
    done = Path(folder) / "batches" / "done"
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in done.glob("*.json")] if done.exists() else []
    return sum((manifest.get("usage") or {}).get("cost") or 0.0 for manifest in manifests)
