"""Shared building blocks for turning TIMSS released-item booklets into item JSONL files.

Each booklet has its own layout, so the notebook sections that read them keep their own page parsing, prompts, and
checks. This module holds what they share: reading words off PDF pages, rendering page areas and embedded images for
the model, the OpenRouter transcription loop with its per-item cache, and the record schema of the released-items
JSONL files (the fields of ``timss11_g4_math_released_items.jsonl``, in order).

``notebooks/scratch/get_data.ipynb`` imports it as ``ri``. It sits in ``src/`` outside the ``timss_math`` package, so
the editable install, which puts ``src/`` on the path, can import it, but the package's wheel does not include it.
"""

from __future__ import annotations

import base64
import json
import re
import zipfile
from collections.abc import Callable, Iterable, Sequence
from itertools import groupby, pairwise
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import pydantic
import pymupdf
import requests

RECORD_FIELDS = [
    "item_id",
    "item_label",
    "assessment",
    "grade",
    "subject",
    "content_domain",
    "main_topic",
    "cognitive_domain",
    "item_type",
    "max_points",
    "stem",
    "options",
    "correct_option",
    "scoring_guide",
    "scoring_notes",
    "has_figure",
    "figure_description",
    "accessibility_text",
    "pct_correct_intl_avg",
    "pct_correct_usa",
    "pct_correct",
    "booklet_page",
    "pdf_page",
    "source_url",
    "transcription_model",
]
VISIBLE_WORDS = pymupdf.TEXTFLAGS_WORDS | pymupdf.TEXT_IGNORE_ACTUALTEXT
OPENROUTER_URL = "https://openrouter.ai/api/v1"


def download(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` unless it is already there, checking that a PDF or zip is what came back."""
    if path.exists():
        return path
    response = requests.get(url, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    magic = {".pdf": b"%PDF", ".zip": b"PK", ".xlsx": b"PK", ".xls": b"\xd0\xcf\x11\xe0"}.get(path.suffix.lower())
    if magic and not response.content.startswith(magic):
        raise ValueError(f"{url} did not return a {path.suffix} file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def center(word: Sequence) -> pymupdf.Point:
    """The center of a word (or span) tuple whose first four values are its bounding box."""
    return pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)


def lines(words: Iterable[Sequence], tolerance: float = 4) -> list[list[Sequence]]:
    """Group words into lines by vertical center, each line left to right."""
    grouped: list[tuple[float, list[Sequence]]] = []
    for word in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        middle = (word[1] + word[3]) / 2
        if grouped and middle - grouped[-1][0] < tolerance:
            grouped[-1][1].append(word)
        else:
            grouped.append((middle, [word]))
    return [sorted(line, key=lambda w: w[0]) for _, line in grouped]


def text(words: Iterable[Sequence]) -> str:
    return " ".join(w[4] for w in words)


def visible_words(page: pymupdf.Page, flags: int = VISIBLE_WORDS) -> list[tuple]:
    """The page's visible words, each once.

    Booklet templates leave placeholder text under opaque boxes drawn later, and some pages draw a table twice, on
    top of itself. ``get_text`` returns all of that text, so a word inside a text span that an opaque fill drawn
    later covers is kept only if visible characters start and end it, and copies at the same place count once.
    """
    fills = [(d["seqno"], d["rect"]) for d in page.get_drawings() if d["fill"] is not None and d["fill_opacity"] == 1]
    covered, visible_chars = [], []
    for span in page.get_texttrace():
        if any(seqno > span["seqno"] and rect.contains(span["bbox"]) for seqno, rect in fills):
            covered.append(pymupdf.Rect(span["bbox"]))
        else:
            visible_chars += [(chr(c[0]), pymupdf.Rect(c[3])) for c in span["chars"]]

    def visible_char_at(char: str, x: float, word: Sequence, edge: str) -> bool:
        return any(
            c == char and abs(getattr(rect, edge) - x) < 0.5 and rect.y0 < word[3] and rect.y1 > word[1]
            for c, rect in visible_chars
        )

    def is_visible(word: Sequence) -> bool:
        if not any(center(word) in rect for rect in covered):
            return True
        return visible_char_at(word[4][0], word[0], word, "x0") and visible_char_at(word[4][-1], word[2], word, "x1")

    unique = {}
    for word in page.get_text("words", flags=flags):
        if is_visible(word):
            unique.setdefault((round(word[0]), round(word[1]), word[4]), word)
    return list(unique.values())


def remove_watermarks(doc: pymupdf.Document) -> int:
    """Empty the form XObjects that Acrobat marks as watermarks (``/PieceInfo /ADBE_CompoundType /Private
    /Watermark``), in memory, so the pages render and extract without them, including any text a watermark covers.
    Returns the number of watermarks removed."""
    forms = {xref for page in doc for xref, *_ in page.get_xobjects()}
    mark = ("name", "/Watermark")
    watermarks = [xref for xref in forms if doc.xref_get_key(xref, "PieceInfo/ADBE_CompoundType/Private") == mark]
    for xref in watermarks:
        doc.update_stream(xref, b"")
    return len(watermarks)


def png_part(pix: pymupdf.Pixmap, ink_level: int = 200, margin: int = 10) -> dict:
    """A pixmap as a PNG image message part, trimmed to its printed content so the model sees it at full size.

    Gray levels darker than ``ink_level`` count as printed content.
    """
    if pix.colorspace.n == 4:  # CMYK
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    pix.set_origin(0, 0)  # a clipped rendering starts at the clip's position, but the trim below counts from 0
    gray = pix if pix.n == 1 else pymupdf.Pixmap(pymupdf.csGRAY, pix)
    ink = np.frombuffer(gray.samples, np.uint8).reshape(gray.height, gray.width) < ink_level
    rows, columns = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    trim = pymupdf.IRect(columns[0], rows[0], columns[-1] + 1, rows[-1] + 1) + (-margin, -margin, margin, margin)
    trim &= pymupdf.IRect(0, 0, pix.width, pix.height)
    trimmed = pymupdf.Pixmap(pix.colorspace, trim, False)
    trimmed.copy(pix, trim)
    png = base64.b64encode(trimmed.tobytes("png")).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}


def page_area_part(page: pymupdf.Page, clip: pymupdf.Rect, dpi: int = 300, **kwargs) -> dict:
    """An area of a page, rendered at ``dpi``, as a PNG image message part (see :func:`png_part`)."""
    return png_part(page.get_pixmap(dpi=dpi, clip=clip), **kwargs)


def embedded_image_part(doc: pymupdf.Document, xref: int, **kwargs) -> dict:
    """An image embedded in the PDF, as stored, as a PNG image message part (see :func:`png_part`)."""
    return png_part(pymupdf.Pixmap(doc, xref), **kwargs)


def text_part(content: str) -> dict:
    return {"type": "text", "text": content}


def unit_batches(items: Sequence[dict], size: int, unit: Callable[[dict], str]) -> list[list[dict]]:
    """Items in order, in batches of at most ``size`` that never split the items of one ``unit`` (such as the parts
    of a multi-part item)."""
    batches: list[list[dict]] = []
    for _, group in groupby(items, key=unit):
        group = list(group)
        if batches and len(batches[-1]) + len(group) <= size:
            batches[-1] += group
        else:
            batches.append(group)
    return batches


class Transcriber:
    """Transcribe items with a model on OpenRouter, a batch of items per request, and cache each usable result.

    ``schema`` is the Pydantic model of one item's transcription, with an ``item_id`` field. Each request asks for
    a list of them. A transcription is kept, as ``<cache_dir>/<item_id>.json``, only if ``problems(item, result,
    returned)`` finds nothing wrong with it; ``returned`` maps the item IDs of the whole response to their results.
    Cached transcriptions are checked again on every run, against all cached ones, so a transcription that fails
    checks added or tightened since it was made is requested again. Nothing is retried: the first failed request, or
    a request without a single usable result, stops the run, and the next run requests only the items that still
    lack a usable transcription.
    """

    def __init__(self, *, api_key: str, model: str, cache_dir: Path, schema: type[pydantic.BaseModel], instructions: str):
        if not api_key:
            raise RuntimeError("No OpenRouter API key: load OPENROUTER_API_KEY from .env")
        self.client = openai.OpenAI(base_url=OPENROUTER_URL, api_key=api_key, max_retries=0, timeout=600)
        self.model = model
        self.cache_dir = cache_dir
        self.schema = schema
        self.batch_schema = pydantic.create_model(f"{schema.__name__}Batch", items=(list[schema], ...))
        self.instructions = instructions
        self.rejected: dict[str, tuple[pydantic.BaseModel | None, list[str]]] = {}

    def path(self, item_id: str) -> Path:
        return self.cache_dir / f"{item_id}.json"

    def request(self, content: list[dict]) -> tuple[dict[str, pydantic.BaseModel], float]:
        """One request: the transcriptions by item ID (empty if the response could not be parsed) and its cost in
        USD as OpenRouter reports it."""
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[{"role": "system", "content": self.instructions}, {"role": "user", "content": content}],
            response_format=self.batch_schema,
            # Route only to providers that support structured output, and report each request's cost.
            extra_body={"provider": {"require_parameters": True}, "usage": {"include": True}},
        )
        cost = getattr(completion.usage, "cost", None) or 0.0
        choice = completion.choices[0]
        if choice.message.parsed is None:
            # OpenRouter reports some upstream failures as HTTP 200 with finish_reason="error" and an error object.
            print(f"  No parsed response (finish_reason={choice.finish_reason}, error={getattr(choice, 'error', None)})")
            return {}, cost
        return {t.item_id: t for t in choice.message.parsed.items}, cost

    def run(
        self,
        items: Sequence[dict],
        *,
        request_content: Callable[[list[dict]], list[dict]],
        problems: Callable[[dict, pydantic.BaseModel, dict], list[str]],
        items_per_request: int,
        unit: Callable[[dict], str] = lambda item: item["item_id"],
        max_requests: int | None = None,
    ) -> dict[str, pydantic.BaseModel]:
        """Transcribe the items without a usable cached transcription, then return every item's cached transcription.

        Raises if some items still lack one, so a rerun picks them up.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stale = self.stale(items, problems)
        for item_id, found in stale.items():
            print(f"The cached transcription of {item_id} fails the checks ({'; '.join(found)}), so it is requested again")
        pending = [item for item in items if not self.path(item["item_id"]).exists() or item["item_id"] in stale]
        requests_ = unit_batches(pending, items_per_request, unit)[:max_requests]
        print(
            f"Transcribing {len(pending)} of {len(items)} items with {self.model} through OpenRouter, "
            f"{len(requests_)} requests in this run"
        )
        total = 0.0
        for number, batch in enumerate(requests_, 1):
            label = f"Request {number}/{len(requests_)} ({batch[0]['item_id']} to {batch[-1]['item_id']})"
            try:
                returned, cost = self.request(request_content(batch))
            except (openai.OpenAIError, pydantic.ValidationError) as error:
                print(f"{label} failed, so no further requests were sent:\n{error}")
                break
            total += cost
            kept = 0
            for item in batch:
                result = returned.get(item["item_id"])
                found = problems(item, result, returned) if result else ["missing from the response"]
                if found:
                    self.rejected[item["item_id"]] = (result, found)
                    print(f"  Rejected {item['item_id']}: {'; '.join(found)}")
                else:
                    self.path(item["item_id"]).write_text(result.model_dump_json(indent=2))
                    stale.pop(item["item_id"], None)
                    kept += 1
            print(f"{label}: kept {kept} of {len(batch)} transcriptions (${cost:.4f})")
            if not kept:
                print("No transcription in this request was usable, so no further requests were sent.")
                break
        print(f"OpenRouter cost of this run: ${total:.4f}")

        missing = [item["item_id"] for item in items if not self.path(item["item_id"]).exists() or item["item_id"] in stale]
        if missing:
            raise RuntimeError(
                f"{len(missing)} items lack a usable transcription: {', '.join(missing)}. Rerun to request only those."
            )
        return self.cached(items)

    def cached(self, items: Sequence[dict]) -> dict[str, pydantic.BaseModel]:
        """The cached transcriptions of the items that have one, by item ID."""
        return {
            item["item_id"]: self.schema.model_validate_json(self.path(item["item_id"]).read_text())
            for item in items
            if self.path(item["item_id"]).exists()
        }

    def stale(
        self, items: Sequence[dict], problems: Callable[[dict, pydantic.BaseModel, dict], list[str]]
    ) -> dict[str, list[str]]:
        """The problems of each cached transcription that fails the checks, by item ID."""
        cached = self.cached(items)
        stale = {}
        for item in items:
            if item["item_id"] in cached and (found := problems(item, cached[item["item_id"]], cached)):
                stale[item["item_id"]] = found
        return stale


def tidy(value: str | None) -> str | None:
    """Even out the model's answer blanks as ____. Runs of spaces are kept, since they can align the digits of
    calculations written vertically."""
    return None if value is None else re.sub(r"_{2,}", "____", value)


def normalized(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", value.lower())


def coverage(drawn: dict[str, str], transcribed: dict[str, str], min_length: int = 3) -> pd.DataFrame:
    """Words drawn on the page that the transcription lacks, by item: ``drawn`` and ``transcribed`` map item IDs (or
    groups of them) to text. Misses are usually labels inside figures."""
    rows = []
    for key, drawn_text in drawn.items():
        words = [w for w in map(normalized, drawn_text.split()) if len(w) >= min_length]
        have = normalized(transcribed[key])
        missing = [w for w in words if w not in have]
        if missing:
            rows.append({"item_id": key, "coverage": 1 - len(missing) / len(words), "missing_words": " ".join(missing)})
    return pd.DataFrame(rows, columns=["item_id", "coverage", "missing_words"]).sort_values("coverage")


def nces_2007_percent_correct(path: Path, sheet: str) -> dict[str, dict]:
    """Percent correct of the TIMSS 2007 released items from an NCES item-statistics workbook, by item ID.

    The sheet has one block of up to seven items after another. Each block starts with the "Item classification (and
    item number)" row, whose cells end in the item ID, and lists the international average, the countries, and, after
    "Benchmarking participants", the benchmarking participants, one row each. "N/A", "—" (not available), and a
    lone "." become None. The workbook compares each result with the U.S. average, by cell color, not with the international
    average, so ``vs_intl_avg`` is None.
    """
    sheet_rows = pd.read_excel(path, sheet_name=sheet, header=None)
    labels = sheet_rows[0].astype(str).str.strip()
    starts = labels.index[labels.str.startswith("Item classification")].tolist()
    results: dict[str, dict] = {}
    for block_start, block_end in zip(starts, [*starts[1:], len(sheet_rows)]):
        block = sheet_rows.iloc[block_start:block_end]
        block_labels = labels.iloc[block_start:block_end]
        first = block_labels.index[block_labels == "International average"][0]
        benchmark = block_labels.index[block_labels.str.startswith("Benchmarking participants")][0]
        last = block_labels.index[block_labels.str.startswith("Average score not compared")][0]
        for column in block.columns[1:]:
            match = re.search(r"\((M\w+)\)\s*$", str(block.at[block_start, column]))
            if not match:
                continue

            def value(row: int, column: int = column) -> float | None:
                cell = sheet_rows.at[row, column]
                return None if pd.isna(cell) or str(cell).strip() in {"N/A", "—", "."} else float(cell)

            results[match[1]] = {
                "pct_correct_intl_avg": value(first),
                "pct_correct_usa": value(block_labels.index[block_labels == "United States"][0]),
                "pct_correct": [
                    {
                        "education_system": labels[row],
                        "pct_correct": value(row),
                        "vs_intl_avg": None,
                        "benchmarking": row > benchmark,
                    }
                    for row in range(first + 1, last)
                    if row != benchmark
                ],
            }
    return results


# TIMSS 2007 mathematics topic areas by grade and content domain (TIMSS 2007 Assessment Frameworks), with the
# spellings of each in the item information files of the TIMSS 2007 International Database.
TIMSS_2007_TOPIC_AREAS = {
    4: {
        "Number": {
            "Whole Numbers": ["Whole Numbers"],
            "Fractions and Decimals": ["Fractions and Decimals", "Fraction and Decimal"],
            "Number Sentences with Whole Numbers": ["Number Sentences", "Number Sentence"],
            "Patterns and Relationships": ["Patterns and Relationships", "Pattern & Relationships"],
        },
        "Geometric Shapes and Measures": {
            "Lines and Angles": ["Lines and Angles"],
            "Two- and Three-dimensional Shapes": ["2-and 3-dimensional shapes", "2- & 3- Dimensional"],
            "Location and Movement": ["Location and Movements"],
        },
        "Data Display": {
            "Reading and Interpreting": ["Reading and Interpreting"],
            "Organizing and Representing": ["Organizing and Representing", "Organizing & Representing"],
        },
    },
    8: {
        "Number": {
            "Whole Numbers": ["Whole Numbers", "Whole Number"],
            "Fractions and Decimals": ["Fractions and Decimals"],
            "Integers": ["Integers"],
            "Ratio, Proportion, and Percent": ["Ratio, Proportion and Percent"],
        },
        "Algebra": {
            "Patterns": ["Patterns"],
            "Algebraic Expressions": ["Algebraic Expressions", "Algebraic Expression"],
            "Equations/Formulas and Functions": ["Equations/Formulas and Functions", "Equations/ Formulas and Functions"],
        },
        "Geometry": {
            "Geometric Shapes": ["Geometric Shapes"],
            "Geometric Measurement": ["Geometric Measurement", "Geomteric Measurement"],
            "Location and Movement": ["Location and Movement"],
        },
        "Data and Chance": {
            "Data Organization and Representation": ["Data Organization and Representation"],
            "Data Interpretation": ["Data Interpretation"],
            "Chance": ["Chance"],
        },
    },
}


def timss_2007_math_item_information(items_zip: Path, grade: int) -> pd.DataFrame:
    """The mathematics items of ``T07_G<grade>_ItemInformation.xls`` in ``T07_Items.zip`` (the "Items" link on
    https://timssandpirls.bc.edu/TIMSS2007/idb_ug.html), indexed by item ID.

    Headings are put on one line and values stripped of stray spaces; ``Topic Area`` holds the framework name of each
    topic area (:data:`TIMSS_2007_TOPIC_AREAS`) instead of the file's spellings.
    """
    with zipfile.ZipFile(items_zip) as archive:
        information = pd.read_excel(archive.open(f"T07_Items/T07_G{grade}_ItemInformation.xls"), dtype=str)
    information.columns = [" ".join(str(column).split()) for column in information.columns]
    information = information.apply(lambda column: column.str.strip())
    information = information[information["Subject"] == "M"].set_index("Item ID", verify_integrity=True)
    topic_names = {
        (domain, spelling): topic
        for domain, topics in TIMSS_2007_TOPIC_AREAS[grade].items()
        for topic, spellings in topics.items()
        for spelling in spellings
    }
    pairs = list(zip(information["Content Domain"], information["Topic Area"]))
    unknown = set(pairs) - set(topic_names)
    if unknown:
        raise ValueError(f"topic areas not in TIMSS_2007_TOPIC_AREAS[{grade}]: {sorted(unknown)}")
    information["Topic Area"] = [topic_names[pair] for pair in pairs]
    return information


def timss_2003_math_item_information(items_zip: Path, grade: int) -> pd.DataFrame:
    """The mathematics rows of the TIMSS 2003 item information file of a grade (``ITEMS/Aitinfm3.xls`` or
    ``ITEMS/Bitinfm3.xls`` in ``t03_items.zip``, the "Items" link on
    https://timssandpirls.bc.edu/timss2003i/userguide.html), with headings on one line and values stripped of stray
    spaces. Each row is an item as scaled, under its "Unique ID": M0 rows for the regular items, MF rows for their
    copies in end-of-session block positions, and, at grade 8, MC rows for five items listed twice."""
    member = {4: "ITEMS/Aitinfm3.xls", 8: "ITEMS/Bitinfm3.xls"}[grade]
    with zipfile.ZipFile(items_zip) as archive:
        information = pd.read_excel(archive.open(member), dtype=str)
    information.columns = [" ".join(str(column).split()) for column in information.columns]
    information = information.apply(lambda column: column.str.strip())
    information = information[information["Subject"] == "M"].reset_index(drop=True)
    if not information["Unique ID"].is_unique:
        raise ValueError(f"{member}: repeated unique IDs")
    return information


# Sidebar labels of an item page in IEA's TIMSS 2003 Released Items.
IEA_2003_FIELDS = ["Content Domain", "Main Topic", "Cognitive Domain", "Key"]


def iea_2003_item(doc: pymupdf.Document, item_id: str) -> tuple[dict[str, str], list[int]]:
    """An item of IEA's TIMSS 2003 Released Items (``T03_RELEASED_M4.pdf`` and ``T03_RELEASED_M8.pdf``, linked from
    https://timssandpirls.bc.edu/timss2003i/released.html): the documentation in the sidebar of its page, as {label:
    value}, and the xrefs of its embedded images, top to bottom.

    Each page shows one item as one or more embedded images, under a header with the item's unique ID; the red
    copyright watermark is separate page content, so the images are clean. Each sidebar label is a text block of its
    own, with its value in the next block below it.
    """
    for page in doc:
        header = page.get_text(clip=pymupdf.Rect(0, 0, page.rect.width, 50))
        if not re.search(rf"^{re.escape(item_id)}$", header, re.MULTILINE):
            continue
        images = sorted(page.get_image_info(xrefs=True), key=lambda info: info["bbox"][1])
        right = max(info["bbox"][2] for info in images)  # the sidebar lies right of the images
        blocks = sorted((b for b in page.get_text("blocks") if b[0] > right and b[1] > 50), key=lambda b: b[1])
        fields = {
            label[4].strip(): " ".join(value[4].split())
            for label, value in pairwise(blocks)
            if label[4].strip() in IEA_2003_FIELDS
        }
        return fields, [info["xref"] for info in images]
    raise ValueError(f"{item_id} is not in {doc.name}")


def write_jsonl(records: Sequence[dict], path: Path) -> pd.DataFrame:
    """Write records with exactly :data:`RECORD_FIELDS`, in order, one JSON object per line, and read them back."""
    for record in records:
        if list(record) != RECORD_FIELDS:
            raise ValueError(f"{record.get('item_id')}: fields differ from RECORD_FIELDS")
    ids = [record["item_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item IDs")
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return pd.read_json(path, lines=True)
