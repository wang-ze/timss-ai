# TIMSS 2007 vs. 2011 grade 4 mathematics released items: data structure differences

This note compares the two released-item sources behind `notebooks/get_data.ipynb` and the JSONL files built from them.
Facts come from parsing both PDFs and from `data/timss11_g4_math_item_parameters.xlsx` on 2026-10-02.
All 74 TIMSS 2007 items are transcribed, but this note covers structure, not transcribed wording.

| | TIMSS 2011 | TIMSS 2007 |
| --- | --- | --- |
| Source | `TIMSS2011_G4_Math.pdf` from [NCES](https://nces.ed.gov/timss/released-questions.asp) | `T07_G4_Released_Items_MAT.pdf` in a zip from [TIMSS & PIRLS, Boston College](https://timssandpirls.bc.edu/timss2007/items.html) |
| Output | `data/timss11_g4_math_released_items.jsonl` | `data/timss07_g4_math_released_items.jsonl` |
| Items | 73 | 74, plus 1 shared introduction page (MP31350) |
| Item ID prefixes | M03 (33), M04 (28), M05 (12) | M03 (46), M04 (28) |

No item ID appears in both sets.

## 1. How the item content is stored in the PDF

| | TIMSS 2011 | TIMSS 2007 |
| --- | --- | --- |
| Item text | Text layer, but much of the math (fractions, number sentences) is drawn as vector outlines | No text at all: each item is one embedded raster image (grayscale; one is CMYK) |
| Accessibility text | Official NCES transcription (`ActualText`) for every item, including figure descriptions | None; the PDF is tagged but has no `ActualText` or `Alt` text |
| Overlays | None | A red "Copyright protected by IEA" watermark drawn over the item as separate page content; the embedded image itself is clean |
| Text that is machine-readable | Item text (partly), classification header, answer key, percent-correct table | Only the item documentation: header (item ID, subject, grade, block, block sequence) and sidebar (content domain, cognitive domain, maximum points, key) |

Consequence: the 2011 stems can be checked word by word against the PDF's text layer, but the 2007 stems can only be checked structurally (option labels, part labels, scoring-guide item ID, and score codes).
The `[Image: ...]` descriptions in 2011 stems follow official NCES wording; in 2007 they are written by the model.

## 2. Item documentation fields

| Field | TIMSS 2011 | TIMSS 2007 |
| --- | --- | --- |
| Item ID | Yes | Yes |
| Item label (short title, e.g. "How many pages needed altogether") | Yes | **Missing** |
| Content domain | Yes, printed in upper case (e.g. "NUMBER") | Yes, title case; same three domains |
| Main topic | Yes (8 topics across the three domains) | **Missing** |
| Cognitive domain | Yes | Yes; same three domains |
| Item type | Inferred from the "Correct Response: X" box | Inferred from the key ("See scoring guide" means constructed response) |
| Maximum points | **Not printed**; inferred from whether the scoring guide has a "Partially Correct Response" category | Printed |
| Multiple-choice key | Printed in a box below the item | Printed in the sidebar and in the item table |
| Block and position in block | **Missing** | Printed (e.g. block M01, sequence 01) |
| Subject and grade codes | Not printed | Printed ("M", "4") |
| Item index | PDF page 3 lists each item's ID, label, and page, grouped by content domain | PDF pages 4-5 hold a full table of the documentation, which matches every item page |
| Percent correct | 57 education systems (50 participants and 7 benchmarking participants), international average, and higher/lower significance markers | **Missing** |
| Printed page number | `pdf_page - 3` | `pdf_page - 2` |

Both sets have four options, A-D, on every multiple-choice item.

## 3. Scoring guides

| | TIMSS 2011 | TIMSS 2007 |
| --- | --- | --- |
| Location | On the item page, below the item; some items add continuation pages with sample student responses | On a separate page after each constructed-response item, as one image |
| Format | Category headings with bulleted descriptors | Table with a two-digit score code per row, headed "Item: <ID>" |
| Score codes | None | 1x (correct on 1-point items, partial on 2-point items), 2x (correct on 2-point items), 7x (incorrect), 99 (blank); for example, M031227 has 10 and 19 (other correct) |
| Categories | "Correct Response", "Partially Correct Response" (2-point items only), "Incorrect Response" | "Correct Response", a partial-credit category (2-point items only), "Incorrect Response", "Nonresponse"; the partial-credit heading is printed "Partial Response" on M031282 and M031247 but "Partially Correct Response" on M041275 |
| Catch-all incorrect row | "Other incorrect (including crossed out, erased, stray marks, illegible, or off task)" | Code 79, worded inconsistently: "crossed out/erased" on M031286, "crossed out, erased" on M041300A |
| 2-point items | 6 | 3 |

## 4. Multi-part items

| | TIMSS 2011 | TIMSS 2007 |
| --- | --- | --- |
| Parts | M031079B-C, M031346A-C, M041115A-B, M041160A-B, M051064A-B (11 parts) | M041300A-D, M031350A-C, M041258A-B, M031242A-C (12 parts) |
| Layout | Each part has its own page, and the shared stimulus is repeated on every part's page (it is in each part's accessibility text) | The parts share booklet pages: the shared introduction is on part A's page or on a page of its own (MP31350, an ID with no domain, points, or key), and later parts may share one page (e.g. M041300B-D) |
| Incomplete sets | M031079A is not released; only parts B and C are | None |
| Mixed parts | All parts are constructed response and share a content domain | Parts can differ in content domain and item type: M031242A is Number, while M031242B and M031242C are Data Display, and M031242C is multiple choice |

In both JSONL files, a part's stem is the shared stimulus followed by that part only.
For 2007, that requires sending every booklet page of the item with each part, because the stimulus is not on every part's page.

## 5. Item counts

| Content domain | 2011 CR | 2011 MC | 2007 CR | 2007 MC |
| --- | --- | --- | --- | --- |
| Number | 23 | 17 | 18 | 20 |
| Geometric Shapes and Measures | 12 | 12 | 14 | 10 |
| Data Display | 2 | 7 | 6 | 6 |
| Total | 37 | 36 | 38 | 36 |

| Cognitive domain | 2011 | 2007 |
| --- | --- | --- |
| Knowing | 29 | 24 |
| Applying | 29 | 30 |
| Reasoning | 15 | 20 |

## 6. How the shared JSONL schema absorbs the differences

Both files have the same 25 fields in the same order; the 2007 notebook cell asserts this against the 2011 file.

| Field | 2011 value | 2007 value |
| --- | --- | --- |
| `item_label`, `main_topic`, `accessibility_text` | From the PDF | `null` |
| `pct_correct_intl_avg`, `pct_correct_usa`, `pct_correct` | From the PDF | `null` |
| `max_points` | Inferred from the scoring categories | Printed in the PDF |
| `scoring_guide` | Categories as printed | Categories as printed, except that "Partially Correct Response" is written "Partial Response" so all three 2-point items match (the transcription prompt asks for this); "Nonresponse" appears; descriptors without their score codes |
| `has_figure`, `figure_description` | Model output, guided by the official accessibility text | Model output from the image alone |
| `source_url` | The PDF URL | The zip URL |
| `transcription_model` | `google/gemini-3.8-flash` (through OpenRouter) | `google/gemini-3.8-flash` (through OpenRouter), or `gemini-3.8-flash` if the Gemini API cell is used instead |

Not kept in the 2007 JSONL, although the PDF has them: block, position in block, and score codes.
They can be added as extra fields if the schema may differ from 2011.

## 7. Link to the TIMSS 2011 item parameters

All 74 TIMSS 2007 released items are in `T11_G4_ItemParameters.xlsx`, in the group "Items Released in 2007".
Their `block_position` there matches the 2007 block and sequence for every item, with the part letter appended for multi-part items (e.g. `M02_08A`).
Their IRT models match the 2007 documentation exactly: 36 multiple-choice items are 3PL, 35 one-point constructed-response items are 2PL, and 3 two-point items are GPCM.
So the 2007 items can join the 2011 items as calibrated reference items for parameter prediction, on the same scale.

## 8. What this means when combining the two sets

- Analyses or filters by `main_topic`, `item_label`, or percent correct cover only the 2011 items; for example, the 2011 crosstab of content domain by main topic would drop every 2007 row.
- Comparisons of scoring categories need a mapping: "Partial Response" (2007) means "Partially Correct Response" (2011), and "Nonresponse" exists only in 2007.
- The parameter-prediction prompt shows the percent correct of similar items; 2007 items would appear without it.
- Stem wording for figures differs in origin: official NCES descriptions for 2011, model-written descriptions for 2007.
- The item labels and topic areas missing from the 2007 PDF are in `T07_G4_ItemInformation.xls` (in `T07_Items.zip` from the TIMSS 2007 international database), and `data/timss07_g4_math_released_items_analysis.xlsx` fills `item_label` and `main_topic` from it; the 2007 JSONL itself keeps them null.
- Percent correct for 2007 items would need yet another source, such as the TIMSS 2007 achievement almanacs.
