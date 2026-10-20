"""What the package's commands share: argument types, and how they print and log.

The commands are ``timss-predict-loo`` (``item_parameter_prediction.evaluation``), ``timss-plot-errors``
(``item_parameter_prediction.plots``), and ``timss-llm-test`` (``llm_test_taking.cli``).
"""

from __future__ import annotations

import logging
import sys


def optional_int(value: str) -> int | None:
    """An integer, or None for "full", "none", or "all"."""
    return None if value.lower() in ("full", "none", "all") else int(value)


def configure_output() -> None:
    """Print lines as they come, in order with the log lines on stderr, and log the package's progress, but not one
    line per HTTP request."""
    if hasattr(sys.stdout, "reconfigure"):  # not a captured stdout
        sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "httpx2"):  # openai sends through httpx2
        logging.getLogger(name).setLevel(logging.WARNING)
