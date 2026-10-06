"""Graphs of the leave-one-out prediction errors.

Every function takes the ``errors`` table of ``PredictionResults`` (one row per item and predicted parameter, with
``predicted``, ``calibrated``, ``error``, and the baselines ``calibration_mean`` and ``reference_mean``) and returns a
matplotlib Figure, one panel per parameter:

- ``predicted_vs_calibrated``: each item's predicted against its calibrated value, by grade, with the identity line.
- ``error_distributions``: box plots of the errors (predicted minus calibrated) by grade or assessment.
- ``mae_vs_baselines``: the mean absolute error of the LLM against the two baselines that need no LLM.
- ``mean_error_by_group``: the bias (mean error) with its 95% confidence interval, by assessment or grade, which
  shows whether the items of one calibration are predicted too easy or too hard.
- ``compare_runs``: the mean absolute errors of several runs (such as different LLMs or search methods) on the items
  that all of them predicted.
- ``factorial_mae``: the mean absolute error of each condition of a factorial experiment (``FactorialResults.errors``),
  one line per level of one factor across the levels of another, which shows their main effects and interaction.

``save_all`` writes the first four. In a notebook, a returned Figure displays inline once the inline backend is on
(``%matplotlib inline``; these Figures do not use pyplot, which would turn it on). From the command line::

    uv run timss-plot-errors notebooks/data/parameter_prediction_runs/<settings>/predictions.xlsx
    uv run timss-plot-errors run_a/predictions.xlsx run_b/predictions.xlsx --names "hybrid" "semantic"
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from .items import PARAMETER_NAMES

# Chart surface, ink, and the categorical series colors in their fixed order (each validated for color-vision
# deficiencies against the surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Grades keep one color and marker in every graph; the marker is a second channel besides the color.
GRADE_STYLES = {4: (SERIES[0], "o"), 8: (SERIES[1], "s"), 12: (SERIES[2], "^")}
GRADE_NAMES = {4: "Grade 4", 8: "Grade 8", 12: "TIMSS Advanced (grade 12)"}
GRADE_SHORT_NAMES = {4: "Grade 4", 8: "Grade 8", 12: "Advanced"}
PARAMETER_TITLES = {
    "slope": "Slope (a)",
    "location": "Location (b)",
    "guessing": "Guessing (c)",
    "step1": "Step 1 (d₁)",
    "step2": "Step 2 (d₂)",
    "step3": "Step 3 (d₃)",
}
BASELINES = {
    "predicted": ("LLM prediction", SERIES[0]),
    "calibration_mean": ("Calibration-summary mean", SERIES[1]),
    "reference_mean": ("Reference-item mean", SERIES[2]),
}
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    # Family names rather than "sans-serif", which matplotlib would resolve when it draws, outside this style; glyphs
    # that Arial lacks, such as subscripts, fall back to DejaVu Sans.
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def _parameters(errors: pd.DataFrame, parameters: Sequence[str] | None) -> list[str]:
    """The parameters to plot, in PARAMETER_NAMES order: those given, else every parameter with errors."""
    present = set(errors["parameter"].astype(str))
    chosen = [name for name in PARAMETER_NAMES if name in present and (parameters is None or name in parameters)]
    if not chosen:
        raise ValueError(f"No errors of {parameters or 'any parameter'}")
    return chosen


def _panels(count: int, panel_size: tuple[float, float], columns: int = 3) -> tuple[Figure, list[Axes]]:
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    figure = Figure(figsize=(panel_size[0] * columns, panel_size[1] * rows + 0.6), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).ravel().tolist()
    for unused in axes[count:]:
        unused.set_visible(False)
    return figure, axes[:count]


def _title(figure: Figure, title: str) -> None:
    figure.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)


def _group_order(values: pd.Series) -> list:
    return sorted(values.unique(), key=lambda value: (isinstance(value, str), value))


def _group_label(by: str, value, short: bool = False) -> str:
    if value == "all":
        return "All" if short else "All items"
    if by != "grade":
        return str(value)
    return (GRADE_SHORT_NAMES if short else GRADE_NAMES).get(value, f"Grade {value}")


def _group_rows(rows: pd.DataFrame, by: str, group) -> pd.DataFrame:
    return rows if group == "all" else rows[rows[by] == group]


def _levels(values: pd.Series) -> list:
    """The levels of a factor: its categories in order if it is categorical, else its sorted values."""
    return list(values.cat.categories) if isinstance(values.dtype, pd.CategoricalDtype) else _group_order(values)


def _ci_half_width(values: pd.Series) -> float:
    """Half the width of the t-based 95% confidence interval of the mean of ``values``."""
    if len(values) < 2:
        return np.nan
    return float(stats.t.ppf(0.975, len(values) - 1)) * values.std() / math.sqrt(len(values))


def _with_counts(by: str, groups: list, counts: list[int]) -> list[str]:
    """Tick labels of groups on two lines, the group and its number of items, so they take little width."""
    return [f"{_group_label(by, group, short=True)}\nn = {count}" for group, count in zip(groups, counts)]


def predicted_vs_calibrated(errors: pd.DataFrame, parameters: Sequence[str] | None = None) -> Figure:
    """Each item's predicted against its calibrated value, one panel per parameter, colored by grade."""
    with matplotlib.rc_context(STYLE):
        names = _parameters(errors, parameters)
        figure, axes = _panels(len(names), (3.6, 3.5))
        grades = _group_order(errors["grade"])
        for name, ax in zip(names, axes):
            rows = errors[errors["parameter"] == name]
            low = min(rows["predicted"].min(), rows["calibrated"].min())
            high = max(rows["predicted"].max(), rows["calibrated"].max())
            pad = (high - low) * 0.05 or 0.05
            limits = (low - pad, high + pad)
            ax.plot(limits, limits, color=INK_MUTED, linewidth=1, zorder=1)
            for grade in grades:
                grade_rows = rows[rows["grade"] == grade]
                color, marker = GRADE_STYLES.get(grade, (INK_SECONDARY, "o"))
                ax.scatter(
                    grade_rows["calibrated"],
                    grade_rows["predicted"],
                    s=20,
                    marker=marker,
                    color=color,
                    alpha=0.7,
                    edgecolors=SURFACE,
                    linewidths=0.6,
                    zorder=2,
                )
            ax.set(xlim=limits, ylim=limits, xlabel="Calibrated", ylabel="Predicted", aspect="equal")
            ax.set_title(PARAMETER_TITLES[name])
            error = rows["error"]
            correlation = rows["predicted"].corr(rows["calibrated"]) if len(rows) > 2 else np.nan
            ax.text(
                0.03,
                0.97,
                f"n = {len(rows)}\nMAE {error.abs().mean():.3f}\nr {correlation:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color=INK_SECONDARY,
            )
        handles = [
            Line2D(
                [], [], linestyle="", marker=GRADE_STYLES[grade][1], color=GRADE_STYLES[grade][0], label=GRADE_NAMES[grade]
            )
            for grade in grades
            if grade in GRADE_STYLES
        ]
        handles.append(Line2D([], [], color=INK_MUTED, linewidth=1, label="Predicted = calibrated"))
        figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
        _title(figure, "Predicted and calibrated item parameters, one point per item")
    return figure


def error_distributions(errors: pd.DataFrame, by: str = "grade", parameters: Sequence[str] | None = None) -> Figure:
    """Box plots of the errors (predicted minus calibrated) by ``by``, one panel per parameter."""
    with matplotlib.rc_context(STYLE):
        names = _parameters(errors, parameters)
        groups = _group_order(errors[by])
        figure, axes = _panels(len(names), (3.8, 0.5 * len(groups) + 1.3))
        for name, ax in zip(names, axes):
            rows = errors[errors["parameter"] == name]
            present = [group for group in groups if (rows[by] == group).any()]
            data = [rows.loc[rows[by] == group, "error"].to_numpy() for group in present]
            positions = np.arange(len(present))[::-1]  # first group on top
            ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
            ax.boxplot(
                data,
                positions=positions,
                orientation="horizontal",
                widths=0.55,
                patch_artist=True,
                boxprops={"facecolor": matplotlib.colors.to_rgba(SERIES[0], 0.15), "edgecolor": SERIES[0], "linewidth": 1},
                medianprops={"color": INK, "linewidth": 1.5},
                whiskerprops={"color": SERIES[0], "linewidth": 1},
                capprops={"color": SERIES[0], "linewidth": 1},
                flierprops={"marker": "o", "markersize": 3, "markerfacecolor": INK_MUTED, "markeredgecolor": "none"},
            )
            ax.set_yticks(positions, _with_counts(by, present, [len(values) for values in data]))
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Predicted − calibrated")
            ax.set_title(PARAMETER_TITLES[name])
        _title(figure, f"Prediction errors by {by.replace('_', ' ')}: boxes span the middle half of the items")
    return figure


def mae_vs_baselines(errors: pd.DataFrame, by: str = "grade", parameters: Sequence[str] | None = None) -> Figure:
    """Mean absolute error of the LLM and of the two baselines, by ``by`` and over all items, one panel per parameter.

    The calibration-summary mean predicts each item by the mean of the calibrated items of its grade with its model and
    points; the reference-item mean by the mean over its reference items with its model and points. A useful LLM beats
    both.
    """
    with matplotlib.rc_context(STYLE):
        names = _parameters(errors, parameters)
        groups = [*_group_order(errors[by]), "all"]
        figure, axes = _panels(len(names), (max(3.6, 0.75 * len(groups) + 1), 3.2))
        width = 0.26
        for name, ax in zip(names, axes):
            rows = errors[errors["parameter"] == name]
            present = [group for group in groups if group == "all" or (rows[by] == group).any()]
            x = np.arange(len(present))
            for offset, (column, (_, color)) in zip((-width, 0, width), BASELINES.items()):
                mae = [
                    (_group_rows(rows, by, group)[column] - _group_rows(rows, by, group)["calibrated"]).abs().mean()
                    for group in present
                ]
                ax.bar(x + offset, mae, width, color=color, edgecolor=SURFACE, linewidth=1, zorder=2)
            labels = [_group_label(by, group, short=True) for group in present]
            ax.set_xticks(x, labels, rotation=30 if len(present) > 4 else 0, ha="right" if len(present) > 4 else "center")
            ax.grid(axis="x", visible=False)
            ax.set_ylabel("Mean absolute error")
            ax.set_title(PARAMETER_TITLES[name])
        handles = [Patch(color=color, label=label) for label, color in BASELINES.values()]
        figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
        _title(figure, "Mean absolute error of the LLM and of two baselines without an LLM (lower is better)")
    return figure


def mean_error_by_group(errors: pd.DataFrame, by: str = "assessment", parameters: Sequence[str] | None = None) -> Figure:
    """Mean error (bias) with its 95% confidence interval by ``by`` and over all items, one panel per parameter.

    A mean error above 0 means the items are predicted too high: for the location, too hard.
    """
    with matplotlib.rc_context(STYLE):
        names = _parameters(errors, parameters)
        groups = [*_group_order(errors[by]), "all"]
        figure, axes = _panels(len(names), (3.8, 0.5 * len(groups) + 1.3))
        for name, ax in zip(names, axes):
            rows = errors[errors["parameter"] == name]
            present = [group for group in groups if group == "all" or (rows[by] == group).any()]
            group_errors = [_group_rows(rows, by, group)["error"] for group in present]
            means = np.array([values.mean() for values in group_errors])
            half_widths = np.array([_ci_half_width(values) for values in group_errors])
            positions = np.arange(len(present))[::-1]
            ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
            ax.errorbar(means, positions, xerr=half_widths, fmt="none", ecolor=SERIES[0], elinewidth=2, capsize=0, zorder=2)
            ax.scatter(means, positions, s=40, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.axhline(0.5, color=AXIS, linewidth=0.8)  # sets "All items" apart
            ax.set_yticks(positions, _with_counts(by, present, [len(values) for values in group_errors]))
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("Mean error")
            ax.set_title(PARAMETER_TITLES[name])
        group = by.replace("_", " ")
        _title(figure, f"Mean prediction error (predicted − calibrated) by {group}, with 95% confidence intervals")
    return figure


def compare_runs(
    errors_by_run: dict[str, pd.DataFrame], parameters: Sequence[str] | None = None, value_labels: bool = True
) -> Figure:
    """Mean absolute error of each run, on the items that every run predicted, one panel per parameter.

    The calibration-summary baseline of those items is drawn as a line, since it is the same for every run.
    """
    if not 1 <= len(errors_by_run) <= len(SERIES):
        raise ValueError(f"Compare 1 to {len(SERIES)} runs")
    common = set.intersection(*(set(errors["item_key"]) for errors in errors_by_run.values()))
    if not common:
        raise ValueError("The runs have no predicted item in common")
    runs = {name: errors[errors["item_key"].isin(common)] for name, errors in errors_by_run.items()}
    with matplotlib.rc_context(STYLE):
        first = next(iter(runs.values()))
        names = _parameters(first, parameters)
        figure, axes = _panels(len(names), (max(2.8, 0.55 * len(runs) + 1.4), 3.2))
        for name, ax in zip(names, axes):
            maes = []
            for position, (run, color) in enumerate(zip(runs, SERIES)):
                rows = runs[run][runs[run]["parameter"] == name]
                maes.append(rows["error"].abs().mean())
                ax.bar(position, maes[-1], 0.5, color=color, edgecolor=SURFACE, linewidth=1, zorder=2)
            if value_labels and len(runs) <= 4:
                for position, mae in enumerate(maes):
                    ax.text(position, mae, f"{mae:.3f}", ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)
            rows = first[first["parameter"] == name]
            baseline = (rows["calibration_mean"] - rows["calibrated"]).abs().mean()
            ax.axhline(baseline, color=INK, linewidth=1.2, zorder=3)
            ax.set_ylim(0, max(*maes, baseline) * 1.12)
            ax.set_xlim(-0.6, len(runs) - 0.4)
            ax.set_xticks([])
            ax.grid(axis="x", visible=False)
            ax.set_ylabel("Mean absolute error")
            ax.set_title(f"{PARAMETER_TITLES[name]} (n = {len(rows)})")
        handles = [Patch(color=color, label=run) for run, color in zip(runs, SERIES)]
        handles.append(Line2D([], [], color=INK, linewidth=1.2, label="Calibration-summary mean (baseline)"))
        figure.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 2))
        _title(figure, f"Mean absolute error by run, on the {len(common)} items every run predicted")
    return figure


def factorial_mae(errors: pd.DataFrame, x: str, lines: str, parameters: Sequence[str] | None = None) -> Figure:
    """Mean absolute error of each condition of a factorial experiment, with its 95% confidence interval, on the items
    that every condition predicted, one panel per parameter.

    ``errors`` is ``FactorialResults.errors``, and ``x`` and ``lines`` are two of its factors: the levels of ``x`` go
    on the x axis and each level of ``lines`` is a line, so the slope of a line is the effect of ``x`` at that level,
    the gap between the lines is the effect of ``lines``, and lines that are not parallel show an interaction. A third
    factor would be averaged over. The calibration-summary baseline of those items is drawn as a black line.
    """
    compared = errors[errors["in_all_conditions"]]
    if compared.empty:
        raise ValueError("No item has a prediction in every condition")
    x_levels, line_levels = _levels(compared[x]), _levels(compared[lines])
    if len(line_levels) > len(SERIES):
        raise ValueError(f"Plot at most {len(SERIES)} lines")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    with matplotlib.rc_context(STYLE):
        names = _parameters(compared, parameters)
        figure, axes = _panels(len(names), (max(2.8, 0.9 * len(x_levels) + 1.2), 3.3))
        positions = np.arange(len(x_levels))
        dodge = 0.08
        for name, ax in zip(names, axes):
            rows = compared[compared["parameter"] == name]
            for number, (level, color, marker) in enumerate(zip(line_levels, SERIES, markers)):
                cells = [rows.loc[(rows[x] == x_level) & (rows[lines] == level), "abs_error"] for x_level in x_levels]
                offset = (number - (len(line_levels) - 1) / 2) * dodge
                ax.errorbar(
                    positions + offset,
                    [cell.mean() for cell in cells],
                    yerr=[_ci_half_width(cell) for cell in cells],
                    color=color,
                    linewidth=2,
                    elinewidth=1.2,
                    capsize=0,
                    marker=marker,
                    markersize=7,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.2,
                    zorder=3,
                )
            first = rows[(rows[x] == x_levels[0]) & (rows[lines] == line_levels[0])]
            ax.axhline((first["calibration_mean"] - first["calibrated"]).abs().mean(), color=INK, linewidth=1.2, zorder=2)
            ax.set_xticks(positions, x_levels)
            ax.set_xlim(-0.5, len(x_levels) - 0.5)
            ax.set_ylim(bottom=0)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel("Mean absolute error")
            ax.set_title(f"{PARAMETER_TITLES[name]} (n = {first['item_key'].nunique()})")
        handles = [
            Line2D([], [], color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE, label=level)
            for level, color, marker in zip(line_levels, SERIES, markers)
        ]
        handles.append(Line2D([], [], color=INK, linewidth=1.2, label="Calibration-summary mean (baseline)"))
        figure.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 3))
        items = compared["item_key"].nunique()
        _title(figure, f"Mean absolute error with 95% confidence intervals, on the {items} items every condition predicted")
    return figure


def save_all(errors: pd.DataFrame, directory: str | Path, image_format: str = "png", dpi: int = 200) -> list[Path]:
    """Write the graphs of one run's errors to ``directory`` and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "predicted_vs_calibrated": predicted_vs_calibrated(errors),
        "error_distribution_by_grade": error_distributions(errors, "grade"),
        "error_distribution_by_assessment": error_distributions(errors, "assessment"),
        "mae_vs_baselines_by_grade": mae_vs_baselines(errors, "grade"),
        "mean_error_by_assessment": mean_error_by_group(errors, "assessment"),
    }
    paths = []
    for name, figure in figures.items():
        path = directory / f"{name}.{image_format}"
        figure.savefig(path, dpi=dpi)
        paths.append(path)
    return paths


def _run_name(workbook: Path) -> str:
    """A workbook's run name: its directory for a run's predictions.xlsx, else its file name."""
    return workbook.parent.name if workbook.name == "predictions.xlsx" else workbook.stem


def main(argv: list[str] | None = None) -> None:
    from .evaluation import PredictionResults

    parser = argparse.ArgumentParser(
        prog="timss-plot-errors",
        description=(
            "Graph the prediction errors of a leave-one-out workbook (written by timss-predict-loo or the notebook). "
            "With several workbooks, also compare their mean absolute errors."
        ),
    )
    parser.add_argument("workbooks", type=Path, nargs="+", help="workbooks with a prediction_errors sheet")
    parser.add_argument("--names", nargs="+", help="a name for each workbook in the comparison")
    parser.add_argument("--output-dir", type=Path, help="default: a plots directory next to the first workbook")
    parser.add_argument("--format", default="png", help="image format, such as png, svg, or pdf (default: png)")
    args = parser.parse_args(argv)
    if args.names and len(args.names) != len(args.workbooks):
        parser.error("Give one name per workbook")

    names = args.names or [_run_name(workbook) for workbook in args.workbooks]
    errors_by_run = {name: PredictionResults.read_excel(workbook).errors for name, workbook in zip(names, args.workbooks)}
    output_dir = args.output_dir or args.workbooks[0].parent / "plots"
    if len(errors_by_run) == 1:
        paths = save_all(next(iter(errors_by_run.values())), output_dir, args.format)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = [output_dir / f"compare_runs.{args.format}"]
        compare_runs(errors_by_run).savefig(paths[0], dpi=200)
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
