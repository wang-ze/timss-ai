"""Where the package finds its data directory and API keys, from a clone of the repository or installed elsewhere.

Run with ``uv run python -m unittest discover tests``.
"""

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from timss_math.item_parameter_prediction import config

KEY = "TIMSS_TEST_API_KEY"


@contextlib.contextmanager
def working_folder(clone_root: Path | None, environ: dict[str, str] | None = None):
    """A new working folder, with the package running from ``clone_root`` (None: installed elsewhere) and the
    environment variables of this test only."""
    environ = environ or {}
    with (
        tempfile.TemporaryDirectory() as directory,
        contextlib.chdir(directory),
        mock.patch.object(config, "CLONE_ROOT", clone_root),
        mock.patch.dict(os.environ, environ),
    ):
        for name in ("TIMSS_DATA_DIR", KEY):
            if name not in environ:
                os.environ.pop(name, None)
        yield Path(directory).resolve()


class DataDirTest(unittest.TestCase):
    def test_this_clone(self):
        self.assertIsNotNone(config.CLONE_ROOT)
        with working_folder(config.CLONE_ROOT):  # wherever the work happens
            self.assertEqual(config.data_dir(), config.CLONE_ROOT / "notebooks" / "data")

    def test_explicit_path_and_environment_come_first(self):
        with working_folder(config.CLONE_ROOT, {"TIMSS_DATA_DIR": "/somewhere"}):
            self.assertEqual(config.data_dir("/explicit"), Path("/explicit"))
            self.assertEqual(config.data_dir(), Path("/somewhere"))

    def test_installed_elsewhere_uses_the_working_folder(self):
        with working_folder(None) as folder:
            self.assertEqual(config.data_dir(), folder / config.WORKING_DATA_DIR)
            (folder / config.WORKING_DATA_DIR).mkdir()
            (folder / "analysis").mkdir()
            os.chdir(folder / "analysis")
            self.assertEqual(config.data_dir(), folder / config.WORKING_DATA_DIR)  # the nearest in a parent


class ApiKeyTest(unittest.TestCase):
    def test_order(self):
        with tempfile.TemporaryDirectory() as clone:
            clone_root = Path(clone).resolve()
            (clone_root / ".env").write_text(f"{KEY}=from-clone\n", encoding="utf-8")
            with working_folder(clone_root) as folder:
                self.assertEqual(config.api_key(KEY), "from-clone")
                (folder / ".env").write_text("OTHER_KEY=other\n", encoding="utf-8")
                self.assertEqual(config.env_files(), [folder / ".env", clone_root / ".env"])
                self.assertEqual(config.api_key(KEY), "from-clone")  # the working folder's .env does not set it
                (folder / ".env").write_text(f"{KEY}=from-working-folder\n", encoding="utf-8")
                self.assertEqual(config.api_key(KEY), "from-working-folder")
                os.environ[KEY] = "from-environment"
                self.assertEqual(config.api_key(KEY), "from-environment")
                self.assertEqual(config.api_key(KEY, "explicit"), "explicit")

    def test_installed_elsewhere_reads_the_working_folder(self):
        with working_folder(None) as folder:
            with self.assertRaisesRegex(RuntimeError, KEY):
                config.api_key(KEY)
            (folder / ".env").write_text(f"{KEY}=from-working-folder\n", encoding="utf-8")
            nested = folder / "notebooks"
            nested.mkdir()
            os.chdir(nested)
            self.assertEqual(config.api_key(KEY), "from-working-folder")  # the nearest .env in a parent


if __name__ == "__main__":
    unittest.main()
