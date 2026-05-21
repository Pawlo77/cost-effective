import numpy as np
import pandas as pd
import pytest

from cost_effective.models.profit_targeting import (
    TargetingConfig,
    business_score_at_k,
    choose_targeting_k,
    expected_value_per_contact,
    rank_test_indices,
    select_k_ev_threshold,
)


def test_expected_value_break_even_at_one_third() -> None:
    ev = expected_value_per_contact(np.array([1.0 / 3.0, 0.2]))
    assert ev[0] == pytest.approx(0.0)
    assert ev[1] < 0.0


def test_select_k_ev_threshold_stops_below_break_even() -> None:
    proba = np.array([0.9, 0.8, 0.4, 0.2, 0.1])
    k = select_k_ev_threshold(proba, min_prob=1.0 / 3.0, max_k=10)
    assert k == 3


def test_choose_targeting_k_combined_more_selective_than_cap() -> None:
    rng = np.random.default_rng(42)
    n = 200
    y = rng.binomial(1, 0.5, size=n)
    proba = np.where(y == 1, rng.uniform(0.55, 0.9, n), rng.uniform(0.1, 0.45, n))
    result = choose_targeting_k(
        pd.Series(y),
        proba,
        feature_count=3,
        config=TargetingConfig(
            max_targets=100,
            k_selection_strategy="combined",
            calibrate_probabilities=False,
        ),
    )
    assert result.selected_k <= 100
    assert result.k_ev_threshold <= result.n_above_break_even


def test_rank_test_indices_respects_k_and_floor() -> None:
    proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1])
    indices, diag = rank_test_indices(proba, k=2, min_prob=0.5)
    assert len(indices) == 2
    assert diag["exported_k"] == 2.0
    assert all(proba[i] >= 0.5 for i in indices)


def test_business_score_at_k_zero_when_k_zero() -> None:
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.2])
    score, tp, fp = business_score_at_k(y, proba, 0, feature_count=1)
    assert score == -200.0
    assert tp == 0
    assert fp == 0
