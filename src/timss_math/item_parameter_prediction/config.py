"""Paths and API keys shared by the parameter-prediction modules.

The package runs either from a clone of the repository (the editable install that ``uv sync`` makes), whose
``notebooks/data`` holds the saved embeddings and predictions and whose root holds ``.env``, or installed elsewhere,
where it looks for both in the working folder instead.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import dotenv_values, find_dotenv

_SOURCE_ROOT = Path(__file__).resolve().parents[3]
CLONE_ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else None
"""The root of the clone that the package runs from, or None when it is installed elsewhere."""
WORKING_DATA_DIR = "timss_math_data"
"""The data directory in the working folder, when the package does not run from a clone."""


def data_dir(path: str | Path | None = None) -> Path:
    """The directory of the embedding and prediction caches and of the runs: ``path`` if given, else
    ``$TIMSS_DATA_DIR``, else ``notebooks/data`` of the clone that the package runs from, else the nearest
    ``timss_math_data`` directory in the working folder or its parents, or a new one in the working folder.

    The items and parameters are read from the package's own ``data`` directory instead (``items.PACKAGE_DATA_DIR``).
    """
    if path is not None:
        return Path(path)
    if os.environ.get("TIMSS_DATA_DIR"):
        return Path(os.environ["TIMSS_DATA_DIR"])
    if CLONE_ROOT is not None:
        return CLONE_ROOT / "notebooks" / "data"
    working = Path.cwd()
    for directory in (working, *working.parents):
        if (directory / WORKING_DATA_DIR).is_dir():
            return directory / WORKING_DATA_DIR
    return working / WORKING_DATA_DIR


def env_files() -> list[Path]:
    """The ``.env`` files that API keys are read from, in order: the nearest in the working folder or its parents,
    then the one at the root of the clone that the package runs from."""
    files = []
    if found := find_dotenv(usecwd=True):
        files.append(Path(found).resolve())
    if CLONE_ROOT is not None and (CLONE_ROOT / ".env").is_file() and (CLONE_ROOT / ".env").resolve() not in files:
        files.append((CLONE_ROOT / ".env").resolve())
    return files


def api_key(env_name: str, explicit: str | None = None) -> str:
    """``explicit`` if given, else the variable from the environment, else from the first of ``env_files()`` that
    sets it."""
    value = explicit or os.environ.get(env_name)
    for path in env_files():
        if value:
            break
        value = dotenv_values(path).get(env_name)
    if not value:
        places = "the working folder, one of its parents" + (", the repository root" if CLONE_ROOT else "")
        searched = ", ".join(map(str, env_files())) or "none found"
        raise RuntimeError(
            f"Set {env_name} in the environment or in a .env file in {places} (.env files read: {searched}), "
            "or pass the API key"
        )
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
