"""The ``timss-llm-test`` command: LLMs take the released items as a test, and their answers are scored with IRT.

It runs an ``Administration``: it requests the missing answers of every condition (at most ``--max-requests`` units
per condition) and the scorer's missing verdicts, estimates every condition's θ on each calibration's scale and on the
TIMSS reporting scale, and writes ``scores.xlsx`` (``ScoringResults.to_excel``) and two graphs to
``<data>/llm_test_runs/<settings>/``. Requests cost money, so none are sent unless ``--max-requests`` is given; with
no setting at all, it reproduces the run of 2026-10-19 from the saved answers and verdicts.

The conditions come from one of:

- ``--llm-model`` (with ``--llm-provider``, ``--reasoning-effort``, ``--max-tokens``, ``--llm-base-url``, and
  ``--ollama-num-ctx``): one LLM, one condition per reasoning effort and role (``--role``).
- ``--conditions FILE``: a JSON list of LLM settings, the fields of ``LLMSettings`` but its API key, each in the roles
  of ``--role`` unless it names its own ``roles``, such as::

      [
        {"model": "openai/gpt-6-luna", "reasoning_effort": "high"},
        {"provider": "anthropic", "model": "claude-haiku-5-5", "reasoning_effort": null, "roles": ["student"]},
        {"provider": "ollama", "model": "llama3.2", "reasoning_effort": null, "max_tokens": 4096}
      ]

- neither: the ten conditions of the run of 2026-10-19 (``DEFAULT_MODELS`` in both roles).

Run ``uv run timss-llm-test --help`` for every setting.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import pandas as pd

from ..command_line import configure_output, optional_int
from ..config import data_dir as resolve_data_dir
from ..config import slug
from ..llm import BASE_URLS as LLM_PROVIDERS
from ..llm import LLMSettings, parse_reasoning_effort
from .administration import Administration
from .answering import ROLES
from .conditions import (
    DEFAULT_MODELS,
    DEFAULT_SCORER,
    TOKEN_PRICES,
    Condition,
    conditions_of,
)

RUNS_DIR = "llm_test_runs"
"""Where the runs' workbooks and graphs go in the data directory, one directory per run's settings."""
LLM_FIELDS = {field.name for field in dataclasses.fields(LLMSettings)} - {"api_key"}
"""The settings of an LLM that a conditions file may give."""


def settings_name(settings: LLMSettings) -> str:
    return f"{settings.provider}-{slug(settings.model)}-{slug(str(settings.reasoning_effort))}"


def run_name(conditions: list[Condition], scorer: LLMSettings) -> str:
    """The name of a run's output directory: its model and reasoning efforts (or, for several models, their number and
    a hash of their settings), its roles, and its scorer."""
    efforts: dict[tuple[str, str], list[str]] = {}
    for condition in conditions:
        llm = condition.llm
        efforts.setdefault((llm.provider, llm.model), []).append(slug(str(llm.reasoning_effort)))
    if len(efforts) == 1:
        [((provider, model), levels)] = efforts.items()
        llm_part = f"{provider}-{slug(model)}-{'+'.join(dict.fromkeys(levels))}"
    else:
        settings = sorted(dict.fromkeys(settings_name(condition.llm) for condition in conditions))
        llm_part = f"{len(efforts)}-models-{hashlib.sha256('|'.join(settings).encode()).hexdigest()[:8]}"
    roles = "+".join(dict.fromkeys(condition.role for condition in conditions))
    return f"{llm_part}__{roles}__scorer-{settings_name(scorer)}"


def read_conditions(path: Path, roles: list[str]) -> list[Condition]:
    """The conditions of a JSON list of LLM settings, each in ``roles`` unless it names its own; ValueError if the
    file names an unknown setting or role, or settings that its provider cannot take."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{path} must hold a JSON list of objects, one per LLM")
    conditions = []
    for entry in entries:
        unknown = sorted(set(entry) - LLM_FIELDS - {"roles"})
        if unknown:
            raise ValueError(f"Unknown settings {unknown} in {path}; use {sorted(LLM_FIELDS)} and roles")
        if "model" not in entry:
            raise ValueError(f"Every LLM in {path} needs a model")
        settings = LLMSettings(**{name: value for name, value in entry.items() if name != "roles"})
        conditions += conditions_of([settings], entry.get("roles", roles))
    return conditions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    llm_defaults = LLMSettings()
    parser = argparse.ArgumentParser(
        prog="timss-llm-test",
        description=(
            "LLMs answer the released TIMSS mathematics items as a test, as test takers or as students; an LLM scorer "
            "scores their constructed responses, and each condition is scored with the IRT models and calibrated "
            "parameters on each calibration's scale and on the TIMSS reporting scale. Writes a workbook and graphs. "
            "Requests cost money, so none are sent unless --max-requests is given; without settings, it reproduces "
            "the run of 2026-10-19 from the saved answers."
        ),
    )
    paths = parser.add_argument_group("data")
    paths.add_argument(
        "--data-dir",
        type=Path,
        help="directory of the caches and runs (default: $TIMSS_DATA_DIR, else notebooks/data of the clone, else "
        "timss_math_data in the working folder)",
    )
    paths.add_argument("--answers-dir", type=Path, help="where answers are saved (default: <data>/cache/llm_answers)")
    paths.add_argument(
        "--verdicts-dir",
        type=Path,
        help="where the scorer's verdicts are saved (default: <data>/cache/llm_answer_scores)",
    )

    # The options of one LLM are left out of the namespace unless given, so that they can be told from the defaults.
    llm_options = {
        "llm_provider": llm_defaults.provider,
        "reasoning_effort": [llm_defaults.reasoning_effort],
        "max_tokens": llm_defaults.max_tokens,
        "llm_base_url": None,
        "ollama_num_ctx": llm_defaults.ollama_num_ctx,
    }
    llm = parser.add_argument_group(
        "conditions", "one LLM with --llm-model, several with --conditions, or neither for the run of 2026-10-19"
    )
    llm.add_argument("--llm-model", help="the LLM that takes the test, such as openai/gpt-6-luna")
    llm.add_argument(
        "--llm-provider", choices=sorted(LLM_PROVIDERS), default=argparse.SUPPRESS, help=f"default: {llm_defaults.provider}"
    )
    llm.add_argument(
        "--reasoning-effort",
        type=parse_reasoning_effort,
        nargs="+",
        default=argparse.SUPPRESS,
        help="one or more, a condition each, such as none, low, medium, or high; 'omit' sends none (the model's own "
        "default); Ollama also takes true or false, and Anthropic only omit or none (default: "
        f"{llm_defaults.reasoning_effort})",
    )
    llm.add_argument("--max-tokens", type=int, default=argparse.SUPPRESS, help=f"default: {llm_defaults.max_tokens}")
    llm.add_argument("--llm-base-url", default=argparse.SUPPRESS, help="API URL of the LLM provider")
    llm.add_argument(
        "--ollama-num-ctx", type=int, default=argparse.SUPPRESS, help=f"default: {llm_defaults.ollama_num_ctx}"
    )
    llm.add_argument(
        "--conditions",
        type=Path,
        help='JSON file of a list of LLM settings, such as [{"model": "openai/gpt-6-luna", "reasoning_effort": "high"}, '
        '{"provider": "anthropic", "model": "claude-haiku-5-5", "reasoning_effort": null, "roles": ["student"]}], '
        "each in the roles of --role unless it names its own",
    )
    llm.add_argument(
        "--role",
        choices=ROLES,
        nargs="+",
        default=list(ROLES),
        help="test_taker takes the test written for the grade (evaluating the LLM), student takes it as a student of "
        "the grade (the LLM as a participant) (default: both)",
    )
    llm.add_argument(
        "--token-price",
        nargs=4,
        action="append",
        default=[],
        metavar=("PROVIDER", "MODEL", "INPUT", "OUTPUT"),
        help="USD per million input and output tokens of a model whose provider reports no cost, such as anthropic "
        "claude-haiku-5-5 0.10 0.50 (repeatable; that one is known)",
    )

    scorer = parser.add_argument_group(
        "scorer", f"the LLM that scores the constructed responses (default: {DEFAULT_SCORER.model})"
    )
    scorer.add_argument("--scorer-provider", choices=sorted(LLM_PROVIDERS), help=f"default: {DEFAULT_SCORER.provider}")
    scorer.add_argument("--scorer-model", help=f"default: {DEFAULT_SCORER.model}")
    scorer.add_argument(
        "--scorer-reasoning-effort",
        type=parse_reasoning_effort,
        default=argparse.SUPPRESS,
        help=f"'omit' sends none (default: {DEFAULT_SCORER.reasoning_effort})",
    )
    scorer.add_argument("--scorer-max-tokens", type=int, help=f"default: {DEFAULT_SCORER.max_tokens}")
    scorer.add_argument("--scorer-base-url", help="API URL of the scorer's provider")

    run = parser.add_argument_group("run")
    run.add_argument(
        "--max-requests",
        type=optional_int,
        default=0,
        help="units each condition requests at most, or 'all' (default: 0, which only collects open batches)",
    )
    run.add_argument(
        "--scorer-max-requests",
        type=optional_int,
        default=argparse.SUPPRESS,
        help="verdicts to request at most, or 'all' (default: all when --max-requests is not 0, else 0; pass all to "
        "score answers collected from batches)",
    )
    run.add_argument(
        "--mode",
        choices=["batch", "direct"],
        default="batch",
        help="batch: OpenRouter batches for the models that have them, at about half the price and up to 24 hours, "
        "and direct requests for the others; direct: every request now (default: batch)",
    )
    run.add_argument("--scorer-mode", choices=["batch", "direct"], default="direct", help="default: direct")
    run.add_argument("--wait", action="store_true", help="wait until every batch is collected (up to 24 hours)")
    run.add_argument("--workers", type=int, default=8, help="concurrent direct requests per condition (default: 8)")
    run.add_argument("--batch-size", type=int, default=200, help="requests per OpenRouter batch at most (default: 200)")
    run.add_argument("--poll-seconds", type=float, default=60, help="with --wait, seconds between checks (default: 60)")
    run.add_argument(
        "--output-dir", type=Path, help="where the workbook and graphs go (default: <data>/llm_test_runs/<settings>/)"
    )
    run.add_argument("--no-plots", action="store_true", help="write the workbook only")
    args = parser.parse_args(argv)

    given = [f"--{name.replace('_', '-')}" for name in llm_options if name in args]
    if args.conditions and (given or args.llm_model):
        parser.error(f"--conditions takes the place of {', '.join((['--llm-model'] if args.llm_model else []) + given)}")
    if given and not args.llm_model:
        parser.error(f"{', '.join(given)} {'needs' if len(given) == 1 else 'need'} --llm-model")
    for name, default in llm_options.items():
        if name not in args:
            setattr(args, name, default)
    try:
        if args.conditions:
            args.conditions = read_conditions(args.conditions, args.role)
        elif args.llm_model:
            models = [
                LLMSettings(
                    provider=args.llm_provider,
                    model=args.llm_model,
                    reasoning_effort=effort,
                    max_tokens=args.max_tokens,
                    base_url=args.llm_base_url,
                    ollama_num_ctx=args.ollama_num_ctx,
                )
                for effort in dict.fromkeys(args.reasoning_effort)
            ]
            args.conditions = conditions_of(models, args.role)
        else:
            args.conditions = conditions_of(DEFAULT_MODELS, args.role)
        args.scorer = dataclasses.replace(
            DEFAULT_SCORER,
            **{
                name: value
                for name, value in {
                    "provider": args.scorer_provider,
                    "model": args.scorer_model,
                    "max_tokens": args.scorer_max_tokens,
                    "base_url": args.scorer_base_url,
                }.items()
                if value is not None
            },
            **({"reasoning_effort": args.scorer_reasoning_effort} if "scorer_reasoning_effort" in args else {}),
        )
        args.token_prices = TOKEN_PRICES | {
            (provider, model): (float(input_price), float(output_price))
            for provider, model, input_price, output_price in args.token_price
        }
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if "scorer_max_requests" not in args:
        args.scorer_max_requests = 0 if args.max_requests == 0 else None
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_output()
    administration = Administration(
        args.conditions,
        args.scorer,
        data_dir=args.data_dir,
        answers_dir=args.answers_dir,
        verdicts_dir=args.verdicts_dir,
        token_prices=args.token_prices,
    )
    print(administration.usage()[["units", "answered_units", "in_open_batches"]].to_string(), end="\n\n")
    run = {"workers": args.workers, "batch_size": args.batch_size, "poll_seconds": args.poll_seconds}
    reports = administration.request_answers(args.max_requests, args.mode, args.wait, **run)
    for label, report in reports.items():
        if report.mode != "nothing sent":
            print(report.describe(label))
    report = administration.request_verdicts(args.scorer_max_requests, args.scorer_mode, args.wait, **run)
    if report.mode != "nothing sent":
        print(report.describe(f"Scorer {args.scorer.model}"))

    answers = administration.answers()
    scores = administration.scores(answers)
    in_batches = sum(report.in_batches for report in reports.values())
    if in_batches:
        print(f"{in_batches} units are in batches still running; rerun to collect them")
    if awaiting := int(scores["score"].isna().sum()):
        print(f"{awaiting} constructed responses await the scorer; rerun with --scorer-max-requests all to score them")
    results = administration.results(answers)
    if results.overall.empty:
        raise SystemExit("No condition has a scored answer yet; pass --max-requests to request some")

    output_dir = args.output_dir or resolve_data_dir(args.data_dir) / RUNS_DIR / run_name(args.conditions, args.scorer)
    print(f"\nWrote {results.to_excel(output_dir / 'scores.xlsx')}")
    if not args.no_plots:
        from .plots import save_all

        for path in save_all(results, output_dir / "plots"):
            print(f"Wrote {path}")
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print("\nPoints earned over every calibration, and the range of the TIMSS scale scores (EAP):")
        print(results.summary().round(1).to_string())
        print("\nTIMSS scale score (EAP) by calibration:")
        print(results.score_table(lambda row: f"{row.timss_eap:.0f}").to_string())


if __name__ == "__main__":
    main()
