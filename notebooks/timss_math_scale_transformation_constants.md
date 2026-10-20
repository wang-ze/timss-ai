# TIMSS reporting-scale transformation constants

The IRT parameters of the released items (`data/timss_math_released_item_parameters.xlsx`) come from seven calibrations, and each calibration has a θ metric of its own.
TIMSS reports achievement on reporting scales instead, and puts the θ of a calibration on its reporting scale with a linear transformation, one per plausible value i:

scale score = A<sub>i</sub> + B<sub>i</sub> θ

`data/timss_math_scale_transformation_constants.csv` holds these constants for all seven calibrations, with the source of every row.
Five of the seven are published in the official TIMSS technical documents.
The two TIMSS 2003 calibrations (1995-2003 at grade 4 and 1999-2003 at grade 8) were never published, so they are derived by common-item linking and checked against published constants.

The constants were collected on 2026-10-17 by a script, `scratch/timss_scale_constants.py`, which is kept locally and is not in the repository: it downloads the documents to `data/raw/`, reads the exhibits, checks every row, derives the 2003 constants, and writes the CSV.
It also writes a copy to `../src/timss_math/data/`, with which the package puts the θ of LLMs that answer the released items on the reporting scales; `tests/` checks that the two copies are equal.
Paths are relative to `notebooks/`.

## The reporting scales

The TIMSS mathematics scales of grades 4 and 8 were set in 1995 with a mean of 500 and a standard deviation of 100 across the participating countries, and each later assessment was linked to them by a concurrent calibration and a linear transformation.
The TIMSS Advanced mathematics scale was set when the 1995 data were rescaled with the 2- and 3-parameter models in 2008: the average of the mean scores of the countries in that calibration is 500 and the standard deviation 100.

Each transformation matches the mean and standard deviation of a group of students on the new metric (μ\*, σ\*) to their mean and standard deviation on the reporting scale (μ, σ), so B = σ / σ\* and A = μ - B μ\*.
TIMSS computes one pair for each of the five plausible values; to transform one θ, such as an LLM's, use the mean of the five A and the mean of the five B.

## Constants by calibration

Means of the five plausible values:

| Calibration | Items of this project | A | B | Constants | Source |
| --- | --- | --- | --- | --- | --- |
| Joint 1995-2003, grade 4 | TIMSS 1995 and 2003, grade 4 | 507.76 | 90.43 | Derived | Stocking-Lord linking and Exhibit 11.14 of the TIMSS 2007 Technical Report (see below) |
| Joint 2003-2007, grade 4 | TIMSS 2007, grade 4 | 507.46 | 102.80 | Published | TIMSS 2007 Technical Report, Chapter 11, Exhibit 11.15 (page 262) |
| Joint 2007-2011, grade 4 | TIMSS 2011, grade 4 | 516.31 | 93.95 | Published | Methods and Procedures in TIMSS and PIRLS 2011, Exhibit 13 (page 25) |
| Joint 1999-2003, grade 8 | TIMSS 1999 and 2003, grade 8 | 498.75 | 99.45 | Derived | Stocking-Lord linking and Exhibit 11.14 of the TIMSS 2007 Technical Report (see below) |
| Joint 2003-2007, grade 8 | TIMSS 2007, grade 8 | 476.01 | 108.81 | Published | TIMSS 2007 Technical Report, Chapter 11, Exhibit 11.15 (page 262) |
| Joint 2007-2011, grade 8 | TIMSS 2011, grade 8 | 477.13 | 112.86 | Published | Methods and Procedures in TIMSS and PIRLS 2011, Exhibit 14 (page 26) |
| TIMSS Advanced 1995-2008 | TIMSS Advanced 1995 and 2008 | 487.83 | 112.21 | Published | TIMSS Advanced 2008 Technical Report, Chapter 8, Exhibit 8.22 (page 189) |

## Sources

| Document | URL | Used for |
| --- | --- | --- |
| TIMSS 2007 Technical Report, Chapter 11: Scaling the Data from the TIMSS 2007 Mathematics and Science Assessments (Foy, Galia, and Li) | https://timssandpirls.bc.edu/TIMSS2007/PDF/T07_TR_Chapter11.pdf | Exhibits 11.14 and 11.15 |
| Methods and Procedures in TIMSS and PIRLS 2011: Scaling the TIMSS and PIRLS 2011 Achievement Data | https://timssandpirls.bc.edu/methods/pdf/TP11_Scaling_Achievement.pdf | Exhibits 13 and 14 |
| TIMSS 2011 International Database, item parameters (`T11_ItemParameters.zip`) | https://timssandpirls.bc.edu/timss2011/downloads/T11_ItemParameters.zip | `T11_G4_Transform.xlsx` and `T11_G8_Transform.xlsx`, a check of Exhibits 13 and 14; the 2007-2011 item parameters for the check of the linking |
| TIMSS Advanced 2008 Technical Report, Chapter 8: Scaling the Data from the TIMSS Advanced 2008 Assessments | https://timssandpirls.bc.edu/timss_advanced/downloads/T08_TR_Chapter8.pdf | Exhibit 8.22 |
| TIMSS 2003 Technical Report, Chapter 11: Scaling Methods and Procedures for the TIMSS 2003 Mathematics and Science Scales (Gonzalez, Galia, and Li) | https://timssandpirls.bc.edu/PDF/t03_download/T03_TR_Chap11.pdf | Section 11.3.6, which describes the 2003 transformation without its constants |
| TIMSS 2003 Technical Report, Appendix D: Item Parameters | https://timssandpirls.bc.edu/PDF/t03_download/T03_TR_AppD.pdf | Exhibits D.1 and D.3, the parameters of the 1999-2003 and 1995-2003 calibrations |
| TIMSS 2007 International Database, items (`T07_Items.zip`) | https://timssandpirls.bc.edu/TIMSS2007/PDF/T07_Items.zip | `T07_ItemParameters_Overall.xls`, the parameters of the 2003-2007 calibrations |

### TIMSS 2007: two steps

TIMSS 2007 put its scores on the reporting scale in two steps (Technical Report, section 11.3.7).
First, the 2003 data, scaled with the 2003-2007 concurrent calibration, were matched to the 2003 scores as published in 2003 (Exhibit 11.14); these constants were applied to the 2007 bridging data.
Second, the 2007 assessment data were matched to the transformed bridging data (Exhibit 11.15), and these constants were applied to all 2007 proficiency scores.
The θ of the 2007 items is the θ of the 2007 proficiency scores, so the CSV uses Exhibit 11.15 for the 2003-2007 calibrations (`use_for_scoring` true).
It also records Exhibit 11.14 (`use_for_scoring` false), which puts the 2003 data on the 2003-2007 metric on the reporting scale, and which the derivation of the 2003 constants uses.
The two steps differ by a few scale points: at grade 4, Exhibit 11.14 gives A = 505.19 and B = 104.74.

### TIMSS 2011 and TIMSS Advanced 2008: one step

TIMSS 2011 matched the TIMSS 2007 data, rescaled with the 2007-2011 concurrent calibration, to the 2007 scores as published (Exhibits 13 and 14), and applied the constants to the 2011 data.
The constants of the exhibits equal those of `T11_G4_Transform.xlsx` and `T11_G8_Transform.xlsx` to 0.00006.

TIMSS Advanced 2008 matched the 1995 data, scaled with the 1995-2008 concurrent calibration, to the 1995 data as rescaled in 2008 (Exhibit 8.22, section 8.4.7), and applied the constants to the 2008 data.

### TIMSS 2003: not published, so derived

The TIMSS 2003 Technical Report says that the 2003 scores were put on the 1995 and 1999 metrics by matching the means and standard deviations of the "bridge scores" (the 1995 grade 4 and 1999 grade 8 data, scaled with the 2003 calibrations) to the scores published in 1995 and 1999 (section 11.3.6), but it publishes neither the constants nor the bridge scores' means and standard deviations on the θ metric.
Neither the full Technical Report nor the TIMSS 2003 User Guide for the International Database has them.

So the constants are derived from what is published:

1. The 2003 items have parameters in their own 2003 calibration (Appendix D of the 2003 Technical Report: Exhibit D.3 for grade 4, D.1 for grade 8) and in the 2003-2007 calibration (`T07_ItemParameters_Overall.xls`, rows with `M0` IDs), because the 2003-2007 calibration used the 2003 data.
   Every common item is scored with the same model and points in both.
2. Stocking-Lord linking of the common items finds θ(2003-2007) = k θ(2003) + m, the line that brings the test characteristic curve of the items on the 2003 metric closest to their curve on the 2003-2007 metric (weighted by a standard normal on the 2003-2007 metric, from -4 to 4).
3. Exhibit 11.14 puts the 2003 data on the 2003-2007 metric on the reporting scale (the 2003 scores as published), so for each plausible value, A = A<sub>11.14</sub> + B<sub>11.14</sub> m and B = B<sub>11.14</sub> k.

| Grade | Calibration | Common items | Correlation of locations | k (Stocking-Lord) | m (Stocking-Lord) | k, m (mean-sigma) | k, m (Haebara-type) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | Joint 1995-2003 | 159 | 0.961 | 0.8634 | 0.0246 | 0.8593, -0.0309 | 0.8621, 0.0256 |
| 8 | Joint 1999-2003 | 194 | 0.984 | 0.9200 | 0.2132 | 0.8942, 0.2326 | 0.9158, 0.2107 |

The Haebara-type criterion compares the items' expected scores rather than their category probabilities, which differs from Haebara's only for GPCM items.

**The check.**
The same procedure, run where both sides are published, must give back the published constants.
The 2007 items have parameters in the 2003-2007 calibration (`MZ` rows of `T07_ItemParameters_Overall.xls`) and in the 2007-2011 calibration (`T11_G<G>_ItemParameters.xlsx`); linking them and applying Exhibit 13 or 14 (the 2007 data on the 2007-2011 metric, matched to the published 2007 scores) derives the 2007 constants, which Exhibit 11.15 publishes.

| Grade | Common 2007 items | Correlation of locations | Derived A, B | Published A, B (Exhibit 11.15) | Derived minus published, θ from -2 to 2 |
| --- | --- | --- | --- | --- | --- |
| 4 | 177 | 0.983 | 508.21, 102.66 | 507.46, 102.80 | +0.47 to +1.03 scale points |
| 8 | 210 | 0.993 | 476.50, 108.20 | 476.01, 108.81 | -0.74 to +1.71 scale points |

So the derived 2003 constants should be good to about 2 scale points where the items are, and less good far from them; the script raises if the check differs by more than 3 points.
The 2003 items' locations agree less across the two grade 4 calibrations (correlation 0.961) than in the check (0.983), so the grade 4 constants are the least certain.
The derived constants apply to the whole calibration: the TIMSS 1995 grade 4 and TIMSS 1999 grade 8 items share the metric of the 2003 items.

## The CSV

One row per calibration, transformation, and plausible value (45 rows):

| Column | Meaning |
| --- | --- |
| `calibration` | The calibration, as the `calibration` column of the item parameters names it |
| `grade`, `scale` | The grade (12 for TIMSS Advanced) and the scale: overall mathematics, or advanced mathematics |
| `applies_to` | Whose θ the row transforms |
| `use_for_scoring` | True for the five rows that put the θ of the calibration's items on the reporting scale; false for the bridge step of TIMSS 2007 |
| `plausible_value` | PV1 to PV5 |
| `target_mean`, `target_sd` | The mean and standard deviation that the transformation matches, on the reporting scale (`target_description`) |
| `theta_mean`, `theta_sd` | The same students' mean and standard deviation on the calibration's θ metric (`theta_description`); derived for the 2003 calibrations |
| `A`, `B` | The constants: scale score = A + B θ |
| `method` | `published`, or how the constants were derived |
| `source`, `source_url`, `exhibit`, `report_page`, `pdf_page` | Where the row comes from; for the derived rows, the exhibit of the bridge constants |

## Checks

- Every printed row satisfies B = σ / σ\* and A = μ - B μ\* to 0.01, so no value was misread.
- The 2011 exhibits equal the 2011 workbooks.
- Every calibration of the item parameters has five rows for scoring.
- The derivation reproduces the published 2007 constants within 3 scale points from θ = -2 to 2 (within 1.71).
