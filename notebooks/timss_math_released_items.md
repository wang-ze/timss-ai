# TIMSS released mathematics items and their IRT parameters

This note documents how `notebooks/scratch/get_data.ipynb` turns every mathematics set on the NCES page [TIMSS released assessment questions](https://nces.ed.gov/timss/released-questions.asp) into item-level JSONL files with one shared schema, combines them, and finds the official IRT item parameters of every item.
The notebook is kept locally and is not in the repository; its helpers, imported as `ri`, are in `src/released_items.py`.
Paths are relative to `notebooks/`, the parent of the notebook's working directory, `notebooks/scratch/`.
Facts are as of 2026-10-03.

## Outputs

| File | Contents |
| --- | --- |
| `data/timssYY_gG_math_released_items.jsonl` | One file per TIMSS set, such as `timss99_g8_math_released_items.jsonl`; one item per line. |
| `data/timssadvYY_g12_math_released_items.jsonl` | The two TIMSS Advanced sets, in the same schema. |
| `data/timss_math_released_items.jsonl` | All sets, oldest assessment first, then by grade. |
| `data/timss_g4_math_released_items.jsonl` | The grade 4 items alone, as before the other grades were added. |
| `data/timss_math_released_item_parameters.xlsx` | IRT parameters of every item, with their sources (sheets `item_parameters`, `sources`, `crosswalk_1995_g4`, `crosswalk_1999_g8`). |
| `data/timss_g4_math_released_item_parameters.xlsx` | The grade 4 rows of the same workbook. |
| `../src/timss_math/item_parameter_prediction/data/` | Copies of `timss_math_released_items.jsonl` and of the `item_parameters` sheet as `timss_math_released_item_parameters.csv`, the inputs of the `timss_math.item_parameter_prediction` package. |

## The sets

| Assessment | Grade | Items (MC / CR) | Booklet | Notebook section |
| --- | --- | --- | --- | --- |
| TIMSS 1995 | 4 | 70 (42 / 28) | NCES `TIMSS1995_G4_Math.pdf` | TIMSS 1995 grade 4 |
| TIMSS Advanced 1995 | 12 | 37 (23 / 14) | `CitemAdM.pdf` from timss.bc.edu, linked from the NCES page | TIMSS Advanced 1995 |
| TIMSS 1999 | 8 | 82 (65 / 17) | NCES `TIMSS1999_G8_Math.pdf` | TIMSS 1999 grade 8 |
| TIMSS 2003 | 4 | 79 (51 / 28) | NCES `TIMSS2003_G4_Math.pdf` | TIMSS 2003 grade 4 |
| TIMSS 2003 | 8 | 99 (72 / 27) | NCES `TIMSS2003_G8_Math.pdf` | TIMSS 2003 grade 8 |
| TIMSS 2007 | 4 | 74 (36 / 38) | `T07_G4_Released_Items_MAT.zip` from timssandpirls.bc.edu | TIMSS 2007 grade 4 |
| TIMSS 2007 | 8 | 89 (51 / 38) | `T07_G8_Released_Items_MAT.zip` from timssandpirls.bc.edu | TIMSS 2007 grade 8 |
| TIMSS Advanced 2008 | 12 | 40 (25 / 15) | `TA08_MAT_Released_Items.pdf` in `TA08_Items.zip` from timssandpirls.bc.edu | TIMSS Advanced 2008 |
| TIMSS 2011 | 4 | 73 (36 / 37) | NCES `TIMSS2011_G4_Math.pdf` | TIMSS 2011 grade 4 |
| TIMSS 2011 | 8 | 88 (48 / 40) | NCES `TIMSS2011_G8_Math.pdf` | TIMSS 2011 grade 8 |

The 731 items are distinct, but short item IDs repeat across sets: K1 to L9 name different items in TIMSS 1995 grade 4 and TIMSS Advanced 1995, and L10 to L18 in TIMSS Advanced 1995 and TIMSS 1999.
`(assessment, grade, item_id)` identifies an item.
TIMSS Advanced records have `grade` 12 (the final year of secondary school) and `subject` "Advanced Mathematics".

## Status on 2026-10-03

All 731 items are transcribed and pass their checks, and every output file above is written.
A sample of each set's transcriptions was compared with the item images.
For every set with a text layer, the printed words of each item were compared with its stem, options, and figure description; the only words missing were figure labels that the description leaves out and page furniture.

## Record schema

Every JSONL file has these fields, in this order (`ri.RECORD_FIELDS` in `../src/released_items.py`, the fields of the first file, `timss11_g4_math_released_items.jsonl`):

| Field | Meaning |
| --- | --- |
| `item_id` | ID as printed in the booklet: permanent IDs such as M032064 and MA13001, or the short 1995 and 1999 IDs such as I3, V4A, B08, and T02A |
| `item_label` | Short title; null for TIMSS Advanced 1995, whose booklet has none |
| `assessment`, `grade`, `subject` | Such as "TIMSS 1999", 8, "Mathematics" |
| `content_domain`, `main_topic`, `cognitive_domain` | The classification in the framework of the item's assessment; `main_topic` is null for 1995 and 1999, whose frameworks have no topic areas |
| `item_type`, `max_points` | `multiple_choice` or `constructed_response`, and the score points of the scoring guide |
| `stem`, `options`, `correct_option` | Model transcription of the item, and the answer key from the booklet |
| `scoring_guide`, `scoring_notes` | Model transcription of a constructed-response item's scoring guide: categories with their descriptors (score codes are used only for checks) |
| `has_figure`, `figure_description` | Model description of every picture in the item |
| `accessibility_text` | The official NCES screen-reader transcription; only the 2011 booklets have one |
| `pct_correct_intl_avg`, `pct_correct_usa`, `pct_correct` | International average, U.S., and per-education-system percent correct, with significance markers where printed |
| `booklet_page`, `pdf_page`, `source_url` | Where the item is |
| `transcription_model` | `google/gemini-3.8-flash` |

Set-specific gaps:

- TIMSS 2007: the booklets have no labels, topic areas, or statistics.
  `item_label` and `main_topic` come from the item information files in `T07_Items.zip`, and percent correct from the NCES item statistics workbooks, which compare results with the U.S. average, so `vs_intl_avg` is null.
  M042273 (grade 8) has no statistics.
- TIMSS Advanced 2008: the booklet has no statistics, so the percent-correct fields are null.
- TIMSS Advanced 1995: only the international average percent correct is printed.

## Extraction

Each set's section has three code cells: a deterministic parse, the transcription, and the write cell.

- **Deterministic, from the PDFs and official files:** IDs, labels, classifications, item types, keys, maximum points, scoring-guide category headings and score codes, response-option letters, statistics, and pages.
  Every parse accepts only known classifications, keys, and headings, and, where the booklet or the international database offers a second source (an item index, an item table, or an item information file), checks the items against it; it stops on any disagreement.
- **`gemini-3.8-flash` through OpenRouter:** `stem`, `options`, `scoring_guide`, `scoring_notes`, `has_figure`, and `figure_description`, from images of the item and its scoring guide (rendered page areas, or embedded images where the booklet stores the item as a scan), a few items per request.

`ri.Transcriber` sends the requests, without retries, and keeps a transcription only if it passes the section's checks against the deterministic parse (option letters, answer key, scoring categories and score codes, part labels of multi-part items).
Kept transcriptions are cached per item under `data/cache/item_transcriptions/<set>/gemini-3.8-flash-v<prompt version>/`, and cached transcriptions are checked again on every run, so checks added later apply to them too.
Where the booklet has a text layer, the write cell reports printed words of each item box that its transcription lacks; misses are usually figure labels.
For TIMSS Advanced 1995 this is a hard check, since its first transcriptions showed that a miss there can be a wrong option.
`figure_description` is model-generated, and model descriptions can miscount grid units.

## Booklet quirks and corrections

TIMSS 2011 grade 8:

- M032047's page prints the content domain Number over an Algebra topic; the record follows the item index (Algebra).
- M042300Z is a derived 2-point item scored from parts A and B, with a scoring guide per part.

TIMSS 2003 (both grades):

- The NCES page of M032670 (grade 8) shows the item box of M032760 (Red and Black Tiles, parts A-C) under M032670's label, classification, and key, and the NCES page of M011013 (grade 4) leaves out the response options.
  Both items come from IEA's [TIMSS 2003 Released Items](https://timssandpirls.bc.edu/timss2003i/released.html) (`T03_RELEASED_M4.pdf` and `T03_RELEASED_M8.pdf`) instead, whose pages must classify them as the NCES booklet does.
  The 2003 item information gives their number of options.
- Every item's type, key, maximum points, and number of options agree with the 2003 item information (`t03_items.zip`).
- Small capitals in the headers extract in mixed case.
  The grade 4 booklet also has fonts without a Unicode map and placeholder text hidden under white boxes.

TIMSS 1999 grade 8:

- Item IDs are booklet positions (B08, T02A); many items are scans with the option letters inside the picture.
- The percent-correct markers are Wingdings arrows coded as control characters, read span by span.

TIMSS 2007 grade 8:

- MP32753 and MP32754 are shared introduction pages, not items.
- M032691's scoring guide prints "Non response" for "Nonresponse", so the check matches category headings by their letters; the record keeps the printed heading.

TIMSS Advanced 2008:

- The item information marks four more items as released (MA13005, MA13010, MA13022, MA13023); they were not used in scaling and are not in the booklet.

TIMSS Advanced 1995:

- Items and coding guides carry a red copyright watermark that Acrobat stamped over the text.
  The notebook removes it from the opened PDF (`ri.remove_watermarks`), so neither the parse nor the images sent to the model contain it.
- Whitening the watermark's pixels, the first approach, also erased the text under it, and the model filled in the gaps wrongly: option B of L4 ("11C8 = 165") became "11C9 = 55", and option D of L11 ("none need have told the truth") became "none of the others told the truth".
  The transcription check now requires every printed word of an item in its transcription, and the set was transcribed again from clean images.
- Coding guides use "Minimal Response" and 3-point codes.

The grade 4 quirks are documented in the notebook sections and in `released_items_2007_vs_2011.md`.

## Item parameters

The parameters are the slope, location, guessing, and step parameters (with standard errors) of the 3PL model for multiple-choice items, the 2PL model for 1-point constructed-response items, and the generalized partial credit model for items worth 2 or 3 points.

| Set | Source | Where | IDs |
| --- | --- | --- | --- |
| TIMSS 2011 grade 4 and 8 | `T11_ItemParameters.zip` ([TIMSS 2011 International Database](https://timssandpirls.bc.edu/timss2011/international-database.html)) | Sheet `MAT` of `T11_G4_ItemParameters.xlsx` and `T11_G8_ItemParameters.xlsx` | Item IDs |
| TIMSS 2007 grade 4 and 8 | `T07_Items.zip` ([TIMSS 2007 International Database](https://timssandpirls.bc.edu/TIMSS2007/idb_ug.html)) | Sheets `G4-MAT` and `G8-MAT` of `T07_ItemParameters_Overall.xls`, rows with `MZ` IDs | MZ31286 is M031286 |
| TIMSS 2003 grade 4 | `T03_TR_AppD.pdf` ([TIMSS 2003 Technical Report](https://timssandpirls.bc.edu/timss2003i/technicalD.html), Appendix D) | Exhibit D.3, `M0` rows | Item IDs |
| TIMSS 2003 grade 8 | Same | Exhibit D.1, `M0` rows | Item IDs |
| TIMSS 1999 grade 8 | Same | Exhibit D.1 | Permanent IDs from the 1999 achievement codebook `BSACBKM2.CDT` (`bm2_cdbks.zip`, [TIMSS 1999 International Database](https://timssandpirls.bc.edu/timss1999i/database.html)) |
| TIMSS 1995 grade 4 | Same | Exhibit D.3 | 2003-style IDs from a derived crosswalk |
| TIMSS Advanced 2008 and 1995 | `TA08_Items.zip` ([TIMSS Advanced 2008 International Database](https://timssandpirls.bc.edu/timss_advanced/idb.html)) | Sheet `Adv. Math` of `TA08_Item_Parameters.xls` (Exhibit C.3 of the [Technical Report](https://timssandpirls.bc.edu/timss_advanced/tr.html)) | 2008 item IDs; 1995 items by their 1995 block position (K01 for K1) |

Notes:

- **Calibrations.** Parameters from different calibrations are on different scales: the 1995 and 2003 grade 4 items share the joint 1995-2003 calibration, the 1999 and 2003 grade 8 items the joint 1999-2003 calibration, the TIMSS Advanced items the 1995-2008 concurrent scaling, and the 2007 and 2011 items of each grade have their own (joint 2003-2007 and 2007-2011).
  Compare parameters across calibrations only after linking them.
- **TIMSS 1999.** The 1999 Technical Report's own exhibit, Exhibit E.3 of `T99_TR_AppenE.pdf` (joint 1995-1999 calibration), lists no items between M022070 and M022165 and so lacks 17 released items.
  Exhibit D.1 is used for all 82, and Exhibit E.3 is a check: for the 65 items in both, the IRT models agree and the slopes and locations correlate 0.96 and 0.97.
  The codebook's answer keys match the booklet's for every item.
- **TIMSS 1995 grade 4.** No official crosswalk from the 1995 IDs to the IDs of Exhibit D.3 exists; the notebook derives it from the 1995 user guide, the 1995 released population 2 items, and the 2003 item information, and checks it (sheet `crosswalk_1995_g4`).
- **TIMSS 2007.** The 2007 workbook's `M0`, `MF`, and `MC` rows are 2003 booklet items.
  M042273 (grade 8) was not used in scaling and has no parameters.
- **TIMSS Advanced 1995.** K8, K9, L10, L11, L15A, and L15B have no parameters: the 2008 rescaling of the 1995 data left out 11 of the 68 advanced mathematics items of 1995, 10 because they did not fit the 2008 framework (Technical Report, section 8.4.1).
  K14 and L18 are 2-point items in the coding guides but have 1-point (2PL) parameters; the Technical Report counts the 31 scaled released 1995 items as worth 40 points, which holds only with these two at 1 point.
  The `note` column explains each case.
- **Fixed guessing.** The exhibits print 0.000 with a standard error of 0.000 as the guessing parameter of constructed-response items; it is left blank, as in the workbooks.

Checks in the last cell: every item has parameters or a documented reason; the IRT model matches the item type and maximum points, with one step per point; the steps of every GPCM item add up to 0; and, per set, the location's rank correlation with the international average percent correct is reported.

| Set | 2PL | 3PL | GPCM | None | Rank correlation, all items | Multiple choice |
| --- | --- | --- | --- | --- | --- | --- |
| TIMSS 1995 grade 4 | 22 | 42 | 6 | 0 | -0.88 | -0.97 |
| TIMSS Advanced 1995 | 6 | 19 | 6 | 6 | -0.77 | -0.91 |
| TIMSS 1999 grade 8 | 14 | 65 | 3 | 0 | -0.87 | -0.92 |
| TIMSS 2003 grade 4 | 25 | 51 | 3 | 0 | -0.90 | -0.95 |
| TIMSS 2003 grade 8 | 21 | 72 | 6 | 0 | -0.87 | -0.89 |
| TIMSS 2007 grade 4 | 35 | 36 | 3 | 0 | -0.86 | -0.96 |
| TIMSS 2007 grade 8 | 27 | 50 | 11 | 1 | -0.86 | -0.92 |
| TIMSS Advanced 2008 | 13 | 25 | 2 | 0 | (no statistics) | |
| TIMSS 2011 grade 4 | 31 | 36 | 6 | 0 | -0.87 | -0.93 |
| TIMSS 2011 grade 8 | 30 | 48 | 10 | 0 | -0.80 | -0.87 |

## Costs

OpenRouter bills `google/gemini-3.8-flash` per token, about $0.01-0.05 per request of up to 4 items.
The whole sets cost about $0.48 (2003 grade 4), $0.61 (1995 grade 4), $0.61 (2003 grade 8), $0.65 (2011 grade 8), $0.22 (TIMSS Advanced 2008), $0.47 (1999 grade 8), $0.7 (2007 grade 8), and $0.26 (TIMSS Advanced 1995, from images without the watermark).
About $0.6 more went to TIMSS Advanced 1995 transcriptions from whitened images, which were discarded.
