"""The IRT models of TIMSS, the ability estimates of a response pattern, and checks of both.

TIMSS scales the items with three models, where θ is ability and 1.7 the scaling constant, as in the TIMSS technical
reports and the prediction prompts:

- 3PL, for multiple-choice items: P(correct | θ) = c + (1 - c) / (1 + exp(-1.7 a (θ - b))).
- 2PL, for 1-point constructed-response items: the same with c = 0.
- GPCM, for 2- and 3-point items: P(score = k | θ) is proportional to exp(Σ_{v = 0..k} 1.7 a (θ - b + d_v)), with
  d_0 = 0, so that scores k - 1 and k are equally likely at θ = b - d_k.

``estimate_theta`` scores a response pattern on the items of one calibration in two ways:

- EAP, the mean of the posterior under a standard normal prior (on each calibration's θ scale, its students have
  roughly mean 0 and standard deviation 1), computed on the grid ``THETA`` from -6 to 6, with the posterior standard
  deviation as its standard error. It is finite for every pattern, but a perfect or near-perfect pattern says only that
  θ is above most of the items, so the prior pulls it down: it understates an LLM that answers (almost) everything
  right.
- MLE, the maximum-likelihood estimate, with the standard error 1 / √(test information). It is infinite (±∞) for a
  perfect or a zero score.

The calibrations are on different θ scales, so a pattern is scored on one calibration's items at a time, never pooled.

The checks verify the models and estimators before they are used, and raise if any fails: ``check_item_models`` (the
probabilities of every item sum to 1, the 3PL passes (1 + c) / 2 at θ = b, the GPCM steps cross where TIMSS's
definition puts them, and the analytic information equals a numeric one), ``implied_percent_correct`` (the percent
correct that the parameters imply for the students of a calibration follows the international percent correct of the
items), and ``recover_theta`` (response patterns simulated on every calibration's items give back their θ).
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from ..items import STEP_NAMES

D = 1.7
THETA = np.linspace(-6, 6, 1201)
"""The grid of the EAP and of the first MLE search, 0.01 apart."""
PRIOR = np.exp(-(THETA**2) / 2)
"""The standard normal prior on THETA, unnormalized."""


@dataclass(frozen=True)
class ItemModel:
    """An item's IRT model and calibrated parameters."""

    irt_model: str
    slope: float
    location: float
    guessing: float
    steps: tuple[float, ...]

    @classmethod
    def from_parameters(cls, row: pd.Series) -> ItemModel:
        """The model of a row of calibrated parameters (``irt_model``, ``slope``, ``location``, ``guessing``, and the
        steps), with no guessing for the 2PL and GPCM and no steps for the 3PL and 2PL."""
        guessing = 0.0 if pd.isna(row["guessing"]) else float(row["guessing"])
        steps = tuple(float(row[name]) for name in STEP_NAMES if pd.notna(row[name])) if row["irt_model"] == "GPCM" else ()
        return cls(row["irt_model"], float(row["slope"]), float(row["location"]), guessing, steps)

    @property
    def points(self) -> int:
        return len(self.steps) if self.irt_model == "GPCM" else 1

    def probabilities(self, theta) -> np.ndarray:
        """P(score = k | θ), one row per θ and one column per score from 0 to the item's points."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if self.irt_model == "GPCM":
            terms = np.column_stack([D * self.slope * (theta - self.location + step) for step in self.steps])
            cumulative = np.column_stack([np.zeros_like(theta), np.cumsum(terms, axis=1)])
            weights = np.exp(cumulative - cumulative.max(axis=1, keepdims=True))
            return weights / weights.sum(axis=1, keepdims=True)
        correct = self.guessing + (1 - self.guessing) / (1 + np.exp(-D * self.slope * (theta - self.location)))
        return np.column_stack([1 - correct, correct])

    @functools.cached_property
    def log_probabilities(self) -> np.ndarray:
        """log P(score = k | θ) on THETA, with probabilities below 1e-300 raised to it."""
        return np.log(np.clip(self.probabilities(THETA), 1e-300, None))

    def expected_score(self, theta) -> np.ndarray:
        return self.probabilities(theta) @ np.arange(self.points + 1)

    def information(self, theta) -> np.ndarray:
        """Fisher information: (1.7 a)² Var(score | θ) for the GPCM, and its 3PL form for the other models."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if self.irt_model == "GPCM":
            probabilities = self.probabilities(theta)
            scores = np.arange(self.points + 1)
            mean = probabilities @ scores
            return (D * self.slope) ** 2 * (probabilities @ scores**2 - mean**2)
        logistic = 1 / (1 + np.exp(-D * self.slope * (theta - self.location)))
        correct = self.guessing + (1 - self.guessing) * logistic
        return (D * self.slope) ** 2 * (1 - self.guessing) * logistic**2 * (1 - logistic) / correct


def item_models(parameters: pd.DataFrame) -> dict[str, ItemModel]:
    """The model of every item of a table of calibrated parameters, by its index (item_key)."""
    return {key: ItemModel.from_parameters(row) for key, row in parameters.iterrows()}


def total_information(models: Sequence[ItemModel], theta) -> np.ndarray:
    """The test information of the items at θ."""
    return sum(model.information(theta) for model in models)


def estimate_theta(models: Sequence[ItemModel], scores: Sequence[int]) -> dict:
    """The EAP (standard normal prior) and MLE of θ from the scores of the items, with their standard errors, and
    ``mle_note``, why an MLE is infinite ("zero score", "perfect score", or "beyond ±6"), or None."""
    loglik = sum(model.log_probabilities[:, score] for model, score in zip(models, scores))
    posterior = np.exp(loglik - loglik.max()) * PRIOR
    posterior /= posterior.sum()
    eap = float(THETA @ posterior)
    estimate = {"theta_eap": eap, "theta_eap_se": float(np.sqrt((THETA - eap) ** 2 @ posterior))}
    if all(score == 0 for score in scores):
        return estimate | {"theta_mle": -np.inf, "theta_mle_se": np.nan, "mle_note": "zero score"}
    if all(score == model.points for model, score in zip(models, scores)):
        return estimate | {"theta_mle": np.inf, "theta_mle_se": np.nan, "mle_note": "perfect score"}
    start = THETA[loglik.argmax()]
    if abs(start) == THETA[-1]:
        return estimate | {"theta_mle": np.sign(start) * np.inf, "theta_mle_se": np.nan, "mle_note": "beyond ±6"}

    def negative_loglik(theta: float) -> float:
        return -sum(np.log(max(model.probabilities(theta)[0, score], 1e-300)) for model, score in zip(models, scores))

    mle = optimize.minimize_scalar(negative_loglik, bounds=(start - 0.02, start + 0.02), method="bounded").x
    standard_error = float(1 / np.sqrt(total_information(models, mle)[0]))
    return estimate | {"theta_mle": float(mle), "theta_mle_se": standard_error, "mle_note": None}


def check_item_models(models: Mapping[str, ItemModel]) -> None:
    """Raise if a model departs from its definition: probabilities that do not sum to 1, a 3PL that does not pass
    (1 + c) / 2 at θ = b, GPCM steps that do not cross at θ = b - d_k, or information that differs from the numeric
    information, the sum over scores of P'(θ)² / P(θ) with P' by central differences."""
    grid = np.linspace(-4, 4, 81)
    for key, model in models.items():
        probabilities = model.probabilities(THETA)
        if not np.allclose(probabilities.sum(axis=1), 1) or (probabilities < 0).any():
            raise ValueError(f"The probabilities of {key} do not sum to 1")
        if model.irt_model == "GPCM":
            for step, d in enumerate(model.steps, 1):
                at = model.probabilities(model.location - d)[0]
                if not np.isclose(at[step - 1], at[step]):
                    raise ValueError(f"Scores {step - 1} and {step} of {key} are not equally likely at θ = b - d{step}")
        elif not np.isclose(model.probabilities(model.location)[0, 1], (1 + model.guessing) / 2):
            raise ValueError(f"P(correct | θ = b) of {key} is not (1 + c) / 2")
        slope = (model.probabilities(grid + 1e-5) - model.probabilities(grid - 1e-5)) / 2e-5
        numeric = (slope**2 / model.probabilities(grid)).sum(axis=1)
        if not np.allclose(model.information(grid), numeric, rtol=1e-4, atol=1e-8):
            raise ValueError(f"The information of {key} does not match the numeric information")


def implied_percent_correct(
    models: Mapping[str, ItemModel],
    percent_correct: Mapping[str, float | None],
    calibration_items: Mapping[str, Sequence[str]],
    min_correlation: float = 0.8,
) -> pd.DataFrame:
    """The percent correct that each dichotomous item's parameters imply for students with θ ~ N(0, 1), against its
    international average percent correct, summarized by calibration: the items, their correlation, and both means.
    Raises if a correlation is below ``min_correlation``; a sign or scale error in the models would break it."""
    weights = PRIOR / PRIOR.sum()
    rows = []
    for label, keys in calibration_items.items():
        implied, international = [], []
        for key in keys:
            if models[key].points == 1 and percent_correct.get(key) is not None:
                implied.append(100 * models[key].probabilities(THETA)[:, 1] @ weights)
                international.append(percent_correct[key])
        implied, international = pd.Series(implied, dtype=float), pd.Series(international, dtype=float)
        rows.append(
            {
                "calibration": label,
                "items": len(implied),
                "correlation": implied.corr(international),
                "mean_implied": implied.mean(),
                "mean_international": international.mean(),
            }
        )
    summary = pd.DataFrame(rows).set_index("calibration")
    if (summary["correlation"] < min_correlation).any():
        raise ValueError("The implied percent correct does not follow the international percent correct")
    return summary


def recover_theta(
    models: Mapping[str, ItemModel],
    calibration_items: Mapping[str, Sequence[str]],
    true_thetas: Sequence[float] = (-1.0, 0.0, 1.0, 2.0),
    replications: int = 500,
    seed: int = 1,
) -> pd.DataFrame:
    """Simulated response patterns on every calibration's items: the median grid MLE, its spread (the standard
    deviation of a normal with its interquartile range), the standard error 1 / √(test information), and the mean EAP,
    for each true θ. Raises unless the median MLE is within 0.1 of the true θ and the spread within 0.8 to 1.25 times
    the standard error; the mean EAP is close to the true θ but pulled toward 0 by the prior."""
    rng = np.random.default_rng(seed)
    rows = []
    for label, keys in calibration_items.items():
        for true_theta in true_thetas:
            patterns = np.column_stack(
                [
                    rng.choice(models[key].points + 1, size=replications, p=models[key].probabilities(true_theta)[0])
                    for key in keys
                ]
            )
            loglik = sum(models[key].log_probabilities[:, patterns[:, column]] for column, key in enumerate(keys))
            mle = THETA[loglik.argmax(axis=0)]
            posterior = np.exp(loglik - loglik.max(axis=0)) * PRIOR[:, None]
            eap = THETA @ (posterior / posterior.sum(axis=0))
            quartiles = np.percentile(mle, [25, 75])
            rows.append(
                {
                    "calibration": label,
                    "true_theta": true_theta,
                    "median_mle": np.median(mle),
                    "mle_spread": (quartiles[1] - quartiles[0]) / 1.349,
                    "standard_error": 1 / np.sqrt(total_information([models[key] for key in keys], true_theta)[0]),
                    "mean_eap": eap.mean(),
                }
            )
    simulation = pd.DataFrame(rows)
    simulation["spread_ratio"] = simulation["mle_spread"] / simulation["standard_error"]
    off = (simulation["median_mle"] - simulation["true_theta"]).abs() > 0.1
    if off.any() or not simulation["spread_ratio"].between(0.8, 1.25).all():
        raise ValueError("Simulated response patterns do not give back their θ")
    return simulation
