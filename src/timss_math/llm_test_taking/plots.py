"""Graphs of the scores of the conditions, one panel per calibration: a row per grade, its calibrations side by side,
with the conditions on the y axes and the TIMSS reporting scale of the calibration on the top axis.

- ``theta_figure``: each condition's θ (EAP, with ±1 posterior standard deviation) on each calibration's scale, where 0
  is roughly the average student of the calibration, with the Advanced International Benchmark (625) dashed.
- ``item_scores_figure``: each item score on the IRT scale: every item a condition answered, at its location b, by
  whether the condition earned full, partial, or no credit, with the condition's θ (EAP) marked. A condition whose θ
  fits its answers earns credit on most items left of its mark and misses most items right of it.

Both take a ``ScoringResults`` with at least one scored answer and return a matplotlib Figure in the style of the
package's other graphs; in a notebook, it displays inline once the inline backend is on (``%matplotlib inline``).
``save_all`` writes both, as ``timss-llm-test`` does.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..plot_style import (
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    STYLE,
    SURFACE,
)
from .results import ScoringResults
from .scales import GRADE_ROWS

SCORE_STYLES = {  # full, partial, and no credit; the marker is a second channel besides the color
    "full credit": (SERIES[0], "o"),
    "partial credit": (SERIES[2], "D"),
    "no credit": (SERIES[1], "X"),
}
ADVANCED_BENCHMARK = 625


def _suptitle(figure: Figure, title: str) -> None:
    figure.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)


def _timss_axis(ax: Axes, a: float, b: float) -> None:
    """The TIMSS reporting scale (A + B θ) on the top axis, and its Advanced benchmark (625) dashed."""
    top = ax.secondary_xaxis("top", functions=(lambda theta: a + b * theta, lambda score: (score - a) / b))
    top.set_xlabel("TIMSS scale", color=INK_SECONDARY)
    top.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY)
    ax.axvline((ADVANCED_BENCHMARK - a) / b, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)


def _calibration_axes(results: ScoringResults, height_per_condition: float) -> tuple[Figure, dict[str, Axes]]:
    """One panel per calibration: a row per grade, its calibrations side by side, with conditions on the y axes."""
    if results.overall.empty:
        raise ValueError("No condition has a scored answer to draw")
    conditions, calibrations = list(results.conditions.index), list(results.calibrations.index)
    columns = max(sum(label.startswith(prefix) for label in calibrations) for prefix, _ in GRADE_ROWS)
    figure = Figure(
        figsize=(3.3 * columns + 2.6, (height_per_condition * len(conditions) + 1.1) * len(GRADE_ROWS) + 0.6),
        layout="constrained",
    )
    grid = figure.subplots(len(GRADE_ROWS), columns, sharey=True, squeeze=False)
    axes = {}
    for row, (prefix, _) in enumerate(GRADE_ROWS):
        labels = [label for label in calibrations if label.startswith(prefix)]
        for column, ax in enumerate(grid[row]):
            if column >= len(labels):
                ax.set_visible(False)
                continue
            axes[labels[column]] = ax
            ax.set_title(labels[column])
            ax.set_yticks(range(len(conditions)), conditions)
            ax.set_ylim(len(conditions) - 0.5, -0.5)
            ax.grid(axis="y", visible=False)
            ax.set_xlabel("θ")
            _timss_axis(ax, *results.calibrations.loc[labels[column], ["timss_scale_a", "timss_scale_b"]])
    return figure, axes


def _benchmark_legend(figure: Figure, handles: list) -> None:
    label = f"Advanced International Benchmark ({ADVANCED_BENCHMARK})"
    handles.append(Line2D([], [], color=INK_MUTED, linewidth=1, linestyle="--", label=label))
    figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))


def theta_figure(results: ScoringResults) -> Figure:
    """Each condition's θ (EAP) ± 1 posterior SD on each calibration's scale, with the TIMSS scale on top."""
    overall, conditions = results.overall, list(results.conditions.index)
    with matplotlib.rc_context(STYLE):
        figure, axes = _calibration_axes(results, 0.32)
        low = min(-3.0, np.floor((overall["theta_eap"] - overall["theta_eap_se"]).min()))
        high = max(3.0, np.ceil((overall["theta_eap"] + overall["theta_eap_se"]).max()))
        for label, ax in axes.items():
            ax.set_xlim(low, high)
            ax.axvline(0, color=GRID, linewidth=1, zorder=1)
            rows = overall[overall["calibration"] == label]
            positions = [conditions.index(condition) for condition in rows["condition"]]
            ax.errorbar(
                rows["theta_eap"],
                positions,
                xerr=rows["theta_eap_se"],
                fmt="none",
                ecolor=SERIES[0],
                elinewidth=2,
                capsize=0,
                zorder=2,
            )
            ax.scatter(rows["theta_eap"], positions, s=40, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        handle = Line2D([], [], marker="o", color=SERIES[0], linewidth=2, label="θ (EAP) ± 1 posterior SD")
        _benchmark_legend(figure, [handle])
        _suptitle(figure, "Ability of each condition on each calibration's scale")
    return figure


def item_scores_figure(results: ScoringResults) -> Figure:
    """Every item score at the item's location b, by full, partial, or no credit, with each condition's θ (EAP)."""
    item_scores, overall, conditions = results.item_scores, results.overall, list(results.conditions.index)
    with matplotlib.rc_context(STYLE):
        figure, axes = _calibration_axes(results, 0.42)
        jitter = np.random.default_rng(2).uniform(-0.22, 0.22, len(item_scores))
        credit = np.where(
            item_scores["score"] == item_scores["scaled_points"],
            "full credit",
            np.where(item_scores["score"] == 0, "no credit", "partial credit"),
        )
        positions = item_scores["condition"].map(conditions.index).to_numpy() + jitter
        values = pd.concat([results.item_scores_wide["location"], overall["theta_eap"]])
        for label, ax in axes.items():
            ax.set_xlim(np.floor(values.min()), np.ceil(values.max()))
            ax.set_xlabel("θ; items at their location b")
            in_panel = (item_scores["calibration"] == label).to_numpy()
            for name, (color, marker) in SCORE_STYLES.items():
                chosen = in_panel & (credit == name)
                ax.scatter(
                    item_scores["location"][chosen],
                    positions[chosen],
                    s=14,
                    color=color,
                    marker=marker,
                    alpha=0.75,
                    linewidths=0,
                    zorder=2,
                )
            for row in overall[overall["calibration"] == label].itertuples():
                position = conditions.index(row.condition)
                ax.plot([row.theta_eap] * 2, [position - 0.38, position + 0.38], color=INK, linewidth=2, zorder=3)
        handles = [
            Line2D([], [], linestyle="", marker=marker, color=color, label=name)
            for name, (color, marker) in SCORE_STYLES.items()
        ]
        handles.append(Line2D([], [], color=INK, linewidth=2, label="θ of the condition (EAP)"))
        _benchmark_legend(figure, handles)
        _suptitle(figure, "Item scores on the IRT scale: each item at its location b, by the credit each condition earned")
    return figure


def save_all(results: ScoringResults, directory: str | Path, image_format: str = "png", dpi: int = 200) -> list[Path]:
    """Write both graphs to ``directory`` and return their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in {"theta": theta_figure(results), "item_scores": item_scores_figure(results)}.items():
        path = directory / f"{name}.{image_format}"
        figure.savefig(path, dpi=dpi)
        paths.append(path)
    return paths
