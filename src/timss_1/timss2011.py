"""Download the TIMSS 2011 SPSS data and convert one country's files to Parquet.

TIMSS 2011 file names follow ``<grade><type><country>m5.sav``: grade ``a``
is fourth grade and ``b`` is eighth grade, ``<type>`` is one of
:data:`FILE_TYPES`, and ``<country>`` is a three-letter code such as
``usa``.

Each ``.sav`` file becomes two files in the output directory:

* ``<stem>.parquet`` holds the data. As in SPSS, missing-value codes such as
  96 (NOT REACHED) and 99 (OMITTED) are kept as values, because TIMSS
  analyses often treat them differently; :func:`load` can turn them into
  NaN. Each column's label, format, value labels, and missing values are
  stored in the Parquet schema and returned by :func:`variable_info`.
* ``<stem>_codebook.csv`` lists the same metadata, one row per variable.

Run from the project root::

    uv run timss2011-convert --zip T11_G4_SPSSData_pt3.zip --country usa
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat
import requests

BASE_URL = "https://timssandpirls.bc.edu/timss2011/downloads/"

FILE_TYPES = {
    "cg": "School background",
    "sa": "Student achievement",
    "sg": "Student background",
    "sh": "Home background",
    "sr": "Within-country scoring reliability",
    "st": "Student-teacher linkage",
    "tg": "Teacher background",
}

_GRADE_PREFIX = {"4": "a", "8": "b"}


def download(filename: str, directory: Path) -> Path:
    """Download ``filename`` from the TIMSS 2011 site unless already present."""
    target = directory / filename
    if target.exists():
        return target
    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with requests.get(BASE_URL + filename, stream=True, timeout=60) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    partial.rename(target)
    return target


def _grade(zip_name: str) -> str:
    match = re.search(r"_G(\d)_", zip_name)
    if match is None or match.group(1) not in _GRADE_PREFIX:
        raise ValueError(f"Cannot tell the grade from {zip_name}")
    return match.group(1)


def extract_country(zip_path: Path, country: str, directory: Path) -> list[Path]:
    """Extract every SPSS file for ``country`` from a TIMSS 2011 SPSS zip."""
    prefix = _GRADE_PREFIX[_grade(zip_path.name)]
    pattern = re.compile(rf"{prefix}[a-z]{{2}}{re.escape(country)}m5\.sav", re.IGNORECASE)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if pattern.fullmatch(name)]
        if not members:
            raise ValueError(f"No files for country {country!r} in {zip_path.name}")
        directory.mkdir(parents=True, exist_ok=True)
        for member in members:
            archive.extract(member, directory)
    return sorted(directory / member for member in members)


def convert(sav_path: Path, directory: Path) -> pd.DataFrame:
    """Convert one ``.sav`` file to Parquet plus a codebook CSV.

    Returns the data as read, with missing-value codes kept as values.
    """
    data, meta = pyreadstat.read_sav(sav_path, user_missing=True)
    info = {
        name: {
            "label": meta.column_names_to_labels.get(name) or "",
            "format": meta.original_variable_types[name],
            "measure": meta.variable_measure.get(name, "unknown"),
            "value_labels": sorted(meta.variable_value_labels.get(name, {}).items()),
            "missing_ranges": sorted((r["lo"], r["hi"]) for r in meta.missing_ranges.get(name, [])),
        }
        for name in data.columns
    }

    table = pa.Table.from_pandas(data, preserve_index=False)
    schema = pa.schema(
        [field.with_metadata({"timss": json.dumps(info[field.name])}) for field in table.schema],
        metadata=table.schema.metadata,
    )
    directory.mkdir(parents=True, exist_ok=True)
    stem = sav_path.stem.lower()
    pq.write_table(pa.Table.from_arrays(table.columns, schema=schema), directory / f"{stem}.parquet")
    _codebook(info).to_csv(directory / f"{stem}_codebook.csv", index=False)
    return data


def _codebook(info: dict[str, dict]) -> pd.DataFrame:
    def missing(ranges: list[tuple[float, float]]) -> str:
        return "; ".join(f"{lo:g}" if lo == hi else f"{lo:g}..{hi:g}" for lo, hi in ranges)

    def labels(pairs: list[tuple[float, str]]) -> str:
        return "; ".join(f"{value:g}={label}" for value, label in pairs)

    return pd.DataFrame(
        {
            "name": list(info),
            "label": [v["label"] for v in info.values()],
            "format": [v["format"] for v in info.values()],
            "measure": [v["measure"] for v in info.values()],
            "missing_values": [missing(v["missing_ranges"]) for v in info.values()],
            "value_labels": [labels(v["value_labels"]) for v in info.values()],
        }
    )


def variable_info(parquet_path: str | Path) -> dict[str, dict]:
    """Return each variable's label, format, measure, value labels, and missing values.

    Value labels are a ``{value: label}`` dict and missing values a list of
    inclusive ``(low, high)`` ranges.
    """
    info = {}
    for field in pq.read_schema(parquet_path):
        entry = json.loads(field.metadata[b"timss"])
        entry["value_labels"] = {value: label for value, label in entry["value_labels"]}
        entry["missing_ranges"] = [tuple(pair) for pair in entry["missing_ranges"]]
        info[field.name] = entry
    return info


def load(parquet_path: str | Path, *, missing_to_nan: bool = False) -> pd.DataFrame:
    """Load a converted file, optionally replacing missing-value codes with NaN."""
    data = pd.read_parquet(parquet_path)
    if missing_to_nan:
        for name, entry in variable_info(parquet_path).items():
            is_missing = pd.Series(False, index=data.index)
            for low, high in entry["missing_ranges"]:
                is_missing |= data[name].between(low, high)
            data[name] = data[name].mask(is_missing)
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", default="T11_G4_SPSSData_pt3.zip", help="SPSS data zip to download")
    parser.add_argument("--country", default="usa", help="three-letter country code")
    parser.add_argument("--data-dir", type=Path, default=Path("notebooks/data"))
    args = parser.parse_args(argv)

    country = args.country.lower()
    raw_dir = args.data_dir / "raw"
    zip_path = download(args.zip, raw_dir)
    sav_paths = extract_country(zip_path, country, raw_dir / zip_path.stem)
    output_dir = args.data_dir / f"timss11_g{_grade(zip_path.name)}_{country}"
    for sav_path in sav_paths:
        rows, columns = convert(sav_path, output_dir).shape
        description = FILE_TYPES.get(sav_path.name[1:3].lower(), "Unknown file type")
        print(f"{sav_path.stem:<10} {description:<36} {rows:>7,} rows x {columns:>4} columns")
    print(f"Wrote Parquet files and codebooks to {output_dir}")


if __name__ == "__main__":
    main()
