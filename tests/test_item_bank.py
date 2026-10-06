"""The data of the item_parameter_prediction package, and its agreement with the files that get_data.ipynb builds.

Run with ``uv run python -m unittest discover tests``.
"""

import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from timss_math.item_parameter_prediction import ItemBank
from timss_math.item_parameter_prediction.items import (
    ITEM_PARAMETERS_FILE,
    PACKAGE_DATA_DIR,
    RELEASED_ITEMS_FILE,
)

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

    def test_item_parameters_match(self):
        package_parameters = pd.read_csv(PACKAGE_DATA_DIR / ITEM_PARAMETERS_FILE)
        assert_frame_equal(package_parameters, pd.read_excel(NOTEBOOK_PARAMETERS_PATH, sheet_name="item_parameters"))


if __name__ == "__main__":
    unittest.main()
