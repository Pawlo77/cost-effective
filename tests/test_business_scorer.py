import numpy as np
from sklearn.dummy import DummyClassifier

from cost_effective.dataset.utils import (
    BREAK_EVEN_PROBABILITY,
    DEFAULT_MAX_TARGETS,
    best_k_break_even,
    business_score_from_proba,
    custom_scorer,
)


def test_business_score_prefers_top_k_over_threshold() -> None:
    y_true = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.1])

    top_k_score = business_score_from_proba(y_true, y_prob, n_features=1, max_k=2)
    threshold_score = business_score_from_proba(
        y_true,
        y_prob,
        n_features=1,
        max_k=DEFAULT_MAX_TARGETS,
        threshold=0.5,
    )

    assert top_k_score == (2 * 10) - (0 * 5) - 200
    assert threshold_score == (2 * 10) - (1 * 5) - 200


def test_custom_scorer_respects_max_k_cap() -> None:
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3))
    y = rng.integers(0, 2, size=n)

    clf = DummyClassifier(strategy="uniform", random_state=42)
    clf.fit(X, y)

    score_full = custom_scorer(clf, X, y, max_k=n)
    score_capped = custom_scorer(clf, X, y, max_k=10)

    assert score_capped <= score_full


def test_best_k_break_even_stops_below_threshold() -> None:
    y_prob = np.array([0.9, 0.8, 0.34, 0.2, 0.1])
    assert best_k_break_even(y_prob, max_k=5, min_prob=BREAK_EVEN_PROBABILITY) == 3
