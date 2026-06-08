"""Expected-value and selective-k targeting (alternative to flat top-k curve max)."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from ..dataset.utils import (
    BREAK_EVEN_PROBABILITY,
    DEFAULT_MAX_TARGETS,
    business_score_from_binary,
)
from .dataclasses import TargetingSelectionResult
from .modeling import build_profit_curve

KSelectionStrategy = Literal["ev_threshold", "elbow", "nested_cv", "combined"]


@dataclass(frozen=True, slots=True)
class TargetingConfig:
    """Hyperparameters for EV-based contact selection."""

    max_targets: int = DEFAULT_MAX_TARGETS
    break_even_probability: float = BREAK_EVEN_PROBABILITY
    tp_reward: float = 10.0
    fp_cost: float = 5.0
    k_selection_strategy: KSelectionStrategy = "combined"
    elbow_score_fraction: float = 0.95
    nested_cv_k_grid: tuple[int, ...] = (
        50,
        100,
        150,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    )
    calibrate_probabilities: bool = True
    min_probability_floor: float | None = None
    cv_folds: int = 5
    random_state: int = 42


def expected_value_per_contact(
    probabilities: np.ndarray,
    tp_reward: float = 10.0,
    fp_cost: float = 5.0,
) -> np.ndarray:
    """Per-contact expected profit: p * tp_reward - (1 - p) * fp_cost."""
    probabilities = np.asarray(probabilities, dtype=float)
    return probabilities * tp_reward - (1.0 - probabilities) * fp_cost


def calibrate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, IsotonicRegression]:
    """Isotonic calibration fit on training labels (OOF or in-sample)."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(probabilities, y_true)
    return calibrator.predict(probabilities), calibrator


def business_score_at_k(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    k: int,
    feature_count: int,
) -> tuple[float, int, int]:
    """Business score when contacting top-k by probability."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    k = max(0, min(int(k), len(probabilities)))
    ranking = np.argsort(probabilities)[::-1]
    y_pred = np.zeros(len(y_true), dtype=int)
    if k > 0:
        y_pred[ranking[:k]] = 1
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    score = float(business_score_from_binary(y_true, y_pred, feature_count))
    return score, tp, fp


def select_k_ev_threshold(
    probabilities: np.ndarray,
    min_prob: float = BREAK_EVEN_PROBABILITY,
    max_k: int = DEFAULT_MAX_TARGETS,
    min_probability_floor: float | None = None,
) -> int:
    """Stop at first top-ranked customer below break-even (or optional floor)."""
    probabilities = np.asarray(probabilities, dtype=float)
    threshold = min_prob if min_probability_floor is None else max(min_prob, min_probability_floor)
    ranking = np.argsort(probabilities)[::-1]
    limit = min(max_k, len(ranking))
    k = 0
    for idx in ranking[:limit]:
        if probabilities[idx] >= threshold:
            k += 1
        else:
            break
    return k


def select_k_elbow(
    curve: pd.DataFrame,
    score_fraction: float = 0.95,
) -> int:
    """Smallest k reaching ``score_fraction`` of the best cumulative business score."""
    if curve.empty:
        return 0
    max_score = float(curve["score"].max())
    target = score_fraction * max_score
    eligible = curve.loc[curve["score"] >= target]
    return int(eligible["k"].min())


def select_k_nested_cv(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    feature_count: int,
    k_grid: tuple[int, ...],
    cv: int = 5,
    random_state: int = 42,
) -> int:
    """Pick k maximizing mean fold business score on OOF ranks (inner loop)."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    k_list = sorted({max(1, min(k, len(probabilities))) for k in k_grid})
    mean_scores: dict[int, float] = dict.fromkeys(k_list, 0.0)

    for _train_idx, val_idx in splitter.split(probabilities, y_true):
        fold_proba = probabilities[val_idx]
        fold_y = y_true[val_idx]
        for k in k_list:
            score, _, _ = business_score_at_k(fold_y, fold_proba, k, feature_count)
            mean_scores[k] += score

    for k in mean_scores:
        mean_scores[k] /= cv

    return max(mean_scores, key=mean_scores.get)


def _resolve_final_k(
    strategy: KSelectionStrategy,
    k_ev: int,
    k_elbow: int,
    k_nested: int,
    k_curve_max: int,  # noqa: ARG001
    max_targets: int,
) -> tuple[int, str]:
    """Combine component k estimates according to strategy."""
    if strategy == "ev_threshold":
        k = k_ev
    elif strategy == "elbow":
        k = k_elbow
    elif strategy == "nested_cv":
        k = k_nested
    else:
        selective = [k_ev, k_elbow, k_nested]
        positive = [k for k in selective if k > 0]
        k = min(positive) if positive else 0

    k = max(0, min(k, max_targets))
    return k, strategy


def choose_targeting_k(
    y_true: pd.Series,
    oof_probabilities: np.ndarray,
    feature_count: int,
    config: TargetingConfig | None = None,
) -> TargetingSelectionResult:
    """Select contact count from OOF probabilities using EV / elbow / nested CV."""
    config = config or TargetingConfig()
    y_array = y_true.to_numpy()
    probabilities = np.asarray(oof_probabilities, dtype=float)
    calibrated = False

    if config.calibrate_probabilities:
        probabilities, _ = calibrate_probabilities(y_array, probabilities)
        calibrated = True

    profit_curve = build_profit_curve(
        y_true,
        probabilities,
        feature_count=feature_count,
        max_targets=config.max_targets,
    )
    k_curve_max = int(profit_curve.curve.loc[profit_curve.curve["score"].idxmax(), "k"])
    k_ev = select_k_ev_threshold(
        probabilities,
        min_prob=config.break_even_probability,
        max_k=config.max_targets,
        min_probability_floor=config.min_probability_floor,
    )
    k_elbow = select_k_elbow(
        profit_curve.curve,
        score_fraction=config.elbow_score_fraction,
    )
    k_nested = select_k_nested_cv(
        y_array,
        probabilities,
        feature_count,
        config.nested_cv_k_grid,
        cv=config.cv_folds,
        random_state=config.random_state,
    )

    selected_k, strategy = _resolve_final_k(
        strategy=config.k_selection_strategy,
        k_ev=k_ev,
        k_elbow=k_elbow,
        k_nested=k_nested,
        k_curve_max=k_curve_max,
        max_targets=config.max_targets,
    )

    oof_score, oof_tp, oof_fp = business_score_at_k(
        y_array,
        probabilities,
        selected_k,
        feature_count,
    )

    ranking = np.argsort(probabilities)[::-1]
    min_prob = float(probabilities[ranking[selected_k - 1]]) if selected_k > 0 else float("nan")
    n_above_break_even = int(np.sum(probabilities >= config.break_even_probability))

    return TargetingSelectionResult(
        selected_k=selected_k,
        strategy=strategy,
        break_even_probability=config.break_even_probability,
        k_ev_threshold=k_ev,
        k_elbow=k_elbow,
        k_nested_cv=k_nested,
        k_profit_curve_max=k_curve_max,
        oof_business_score=oof_score,
        oof_tp=oof_tp,
        oof_fp=oof_fp,
        min_probability_in_selection=min_prob,
        n_above_break_even=n_above_break_even,
        calibrated=calibrated,
    )


def rank_test_indices(
    test_probabilities: np.ndarray,
    k: int,
    min_prob: float | None = None,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select up to k test rows by probability with optional EV floor."""
    test_probabilities = np.asarray(test_probabilities, dtype=float)
    cap = min(max_targets, len(test_probabilities))
    ranking = np.argsort(test_probabilities)[::-1]
    selected: list[int] = []

    for idx in ranking:
        if len(selected) >= k or len(selected) >= cap:
            break
        if min_prob is None or test_probabilities[idx] >= min_prob:
            selected.append(int(idx))

    diagnostics = {
        "requested_k": float(k),
        "exported_k": float(len(selected)),
        "n_test_above_break_even": float(
            np.sum(test_probabilities >= (min_prob or BREAK_EVEN_PROBABILITY))
        ),
        "min_prob_in_export": float(test_probabilities[selected[-1]]) if selected else float("nan"),
        "max_prob_in_export": float(test_probabilities[selected[0]]) if selected else float("nan"),
    }
    return np.asarray(selected, dtype=int), diagnostics
