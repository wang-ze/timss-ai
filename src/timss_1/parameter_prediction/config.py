"""Paths and API keys shared by the parameter-prediction modules."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"


def data_dir(path: str | Path | None = None) -> Path:
    """``path`` if given, else ``$TIMSS_DATA_DIR``, else ``notebooks/data`` of the project (the notebooks' ``data``)."""
    if path is not None:
        return Path(path)
    return Path(os.environ.get("TIMSS_DATA_DIR") or PROJECT_ROOT / "notebooks" / "data")


def api_key(env_name: str, explicit: str | None = None) -> str:
    """``explicit`` if given, else the variable from the environment, else from the project ``.env``."""
    value = explicit or os.environ.get(env_name) or dotenv_values(ENV_PATH).get(env_name)
    if not value:
        raise RuntimeError(f"Set {env_name} in the environment or in {ENV_PATH}, or pass the API key")
    return value


def slug(text: str) -> str:
    """``text`` with every run of characters other than letters, digits, dots, and hyphens replaced by ``_``."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", text).strip("_")


def write_text_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file, so an interrupted write never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def write_json(path: Path, value: object) -> None:
    write_text_atomically(path, json.dumps(value, ensure_ascii=False, indent=2))
