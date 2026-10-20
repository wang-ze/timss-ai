"""The package's data, its agreement with the files that get_data.ipynb and timss_scale_constants.py write, and the
import paths that the shared modules had before they moved out of item_parameter_prediction.

Run with ``uv run python -m unittest discover tests``.
"""

import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

import timss_math
from timss_math import batch, config, items, llm, running
from timss_math.items import (
    ITEM_PARAMETERS_FILE,
    PACKAGE_DATA_DIR,
    RELEASED_ITEMS_FILE,
    ItemBank,
)
from timss_math.llm_test_taking.scales import SCALE_CONSTANTS_FILE

NOTEBOOK_DATA_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "data"
NOTEBOOK_PARAMETERS_PATH = NOTEBOOK_DATA_DIR / "timss_math_released_item_parameters.xlsx"


class ItemBankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = ItemBank.load()

    def test_counts(self):
        self.assertEqual(len(self.bank.items_by_key), 731)
        self.assertEqual(len(self.bank.chunks), 728)
        self.assertEqual(len(self.bank.calibrated), 722)


@unittest.skipUnless(NOTEBOOK_PARAMETERS_PATH.exists(), "the notebooks' data directory is not here")
class PackageDataTest(unittest.TestCase):
    """The package's copies must match the notebooks' files, which get_data.ipynb writes in the same cell."""

    def test_released_items_match(self):
        package_items = (PACKAGE_DATA_DIR / RELEASED_ITEMS_FILE).read_bytes()
        self.assertEqual(package_items, (NOTEBOOK_DATA_DIR / RELEASED_ITEMS_FILE).read_bytes())

    def test_scale_constants_match(self):
        """timss_scale_constants.py writes both copies."""
        package_copy = (PACKAGE_DATA_DIR / SCALE_CONSTANTS_FILE).read_bytes()
        self.assertEqual(package_copy, (NOTEBOOK_DATA_DIR / SCALE_CONSTANTS_FILE).read_bytes())

    def test_item_parameters_match(self):
        package_parameters = pd.read_csv(PACKAGE_DATA_DIR / ITEM_PARAMETERS_FILE)
        assert_frame_equal(package_parameters, pd.read_excel(NOTEBOOK_PARAMETERS_PATH, sheet_name="item_parameters"))


class OldImportPathsTest(unittest.TestCase):
    """The shared modules moved from item_parameter_prediction to timss_math; the old names still work."""

    def test_moved_modules_are_the_same_modules(self):
        from timss_math.item_parameter_prediction import batch as old_batch
        from timss_math.item_parameter_prediction import config as old_config
        from timss_math.item_parameter_prediction import items as old_items

        self.assertIs(old_batch, batch)
        self.assertIs(old_config, config)
        self.assertIs(old_items, items)
        self.assertEqual(PACKAGE_DATA_DIR, Path(timss_math.__file__).parent / "data")

    def test_moved_names_are_importable_where_they_were(self):
        from timss_math.item_parameter_prediction import evaluation, plots, predictor
        from timss_math.item_parameter_prediction import llm as prediction_llm

        self.assertIs(prediction_llm.LLMClient, llm.LLMClient)
        self.assertIs(prediction_llm.Completion, llm.Completion)
        self.assertIs(predictor.UnusableOutputError, running.UnusableOutputError)
        self.assertIs(predictor.run_sync, running.run_sync)
        self.assertIs(predictor.plural, items.plural)
        self.assertIs(evaluation.excel_value, config.excel_value)
        self.assertEqual(plots.GRADE_SHORT_NAMES, items.GRADE_SHORT_NAMES)


if __name__ == "__main__":
    unittest.main()
