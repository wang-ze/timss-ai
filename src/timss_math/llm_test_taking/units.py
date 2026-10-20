"""The released items as a test: the items that have calibrated parameters, grouped into the units a student sees
together.

A unit is an item alone, or all parts of a multi-part item and all items of a shared-stimulus task
(``SHARED_STIMULUS_TASKS``), in booklet order. Later parts often depend on earlier ones ("Trading Cards (Continued)",
or "What fraction of the figure is shaded in part B above?"), and many repeat only part of the shared stimulus, so a
unit is asked in one request, with one answer per item.

``ItemUnits.load`` reads every released item (731) and the calibrated parameters of every item that has them (724),
including the three items whose stem and options repeat another item's, which the prediction index leaves out: here
each is an item of its own test. A unit without any calibrated item is not asked; an uncalibrated item of a unit that
is asked is answered with it but not scored. The 724 calibrated items are in 652 units.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..items import (
    ITEM_PARAMETERS_FILE,
    PACKAGE_DATA_DIR,
    RELEASED_ITEMS_FILE,
    find_item_parts,
    load_calibrated_parameters,
    load_released_items,
)


@dataclass(frozen=True)
class Unit:
    """Items that a student sees together, in booklet order: an item alone, or the parts of a multi-part item and the
    items of a shared-stimulus task. Units are equal when their item keys are."""

    item_keys: tuple[str, ...]
    items: tuple[dict, ...] = field(compare=False, repr=False)
    """The released items of ``item_keys``, with all their JSONL fields."""

    @property
    def key(self) -> str:
        """The item_key of an item alone, or the item IDs of a unit of several items joined by "+"."""
        first = self.items[0]
        ids = "+".join(item["item_id"] for item in self.items)
        return f"{first['assessment']}/grade {first['grade']}/{first['subject']}/{ids}"

    @property
    def grade(self) -> int:
        return self.items[0]["grade"]


@dataclass(eq=False)
class ItemUnits:
    """The released items, the calibrated parameters of those that have them, and the units that are asked."""

    items_by_key: dict[str, dict]
    """Every released item (731) by item_key, each set in booklet order."""
    parameters: pd.DataFrame
    """The calibrated parameters of every released item that has them (724), indexed by item_key, with
    ``scaled_points``."""
    units: list[Unit]
    """The units with at least one calibrated item (652), in booklet order."""
    all_units: list[Unit]
    """Every unit of the released items, including those without any calibrated item."""

    @classmethod
    def load(cls, released_items_path: str | Path | None = None, parameters_path: str | Path | None = None) -> ItemUnits:
        """Load the items and parameters, by default from the package's data, group them into units, and check that
        every item is in exactly one unit and every calibrated item in a unit that is asked."""
        items_by_key = load_released_items(Path(released_items_path or PACKAGE_DATA_DIR / RELEASED_ITEMS_FILE))
        parameters = load_calibrated_parameters(
            Path(parameters_path or PACKAGE_DATA_DIR / ITEM_PARAMETERS_FILE), list(items_by_key)
        )
        booklet_position = {key: position for position, key in enumerate(items_by_key)}
        item_parts = find_item_parts([{"item_key": key, **item} for key, item in items_by_key.items()])

        def unit(keys: Iterable[str]) -> Unit:
            ordered = tuple(sorted(keys, key=booklet_position.get))
            return Unit(ordered, tuple(items_by_key[key] for key in ordered))

        all_units = sorted(
            {unit(keys) for keys in item_parts.values()}, key=lambda unit: booklet_position[unit.item_keys[0]]
        )
        units = [unit for unit in all_units if any(key in parameters.index for key in unit.item_keys)]
        if sorted(key for unit in all_units for key in unit.item_keys) != sorted(items_by_key):
            raise ValueError("Every released item must be in exactly one unit")
        asked = {key for unit in units for key in unit.item_keys}
        if not parameters.index.isin(list(asked)).all():
            raise ValueError("Every calibrated item must be in a unit that is asked")
        return cls(items_by_key, parameters, units, all_units)

    @functools.cached_property
    def unit_of_item(self) -> dict[str, Unit]:
        """The unit of each item of a unit that is asked."""
        return {key: unit for unit in self.units for key in unit.item_keys}

    def summary(self) -> pd.DataFrame:
        """The units, units of several items, items, and calibrated items of each grade and assessment."""
        table = pd.DataFrame(
            [
                {
                    "grade": unit.grade,
                    "assessment": unit.items[0]["assessment"],
                    "units": 1,
                    "units_of_several_items": int(len(unit.item_keys) > 1),
                    "items": len(unit.item_keys),
                    "calibrated_items": sum(key in self.parameters.index for key in unit.item_keys),
                }
                for unit in self.units
            ]
        )
        return table.groupby(["grade", "assessment"]).sum()
