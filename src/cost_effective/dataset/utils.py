"""Utility functions for feature selection and business-metric scoring."""

import numpy as np
from lightgbm import LGBMClassifier

DEFAULT_MAX_TARGETS = 1000
BREAK_EVEN_PROBABILITY = 1.0 / 3.0


def business_score_from_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> float:
    """Compute business score from binary predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))

    return (tp * 10) - (fp * 5) - (n_features * 200)


def best_k_break_even(
    y_prob: np.ndarray,
    max_k: int = DEFAULT_MAX_TARGETS,
    min_prob: float = BREAK_EVEN_PROBABILITY,
) -> int:
    """Count top-ranked samples with probability at or above break-even (plan §2.1)."""
    y_prob = np.asarray(y_prob)
    sorted_prob = np.sort(y_prob)[::-1]
    limit = min(max_k, len(sorted_prob))
    k = 0
    for prob in sorted_prob[:limit]:
        if prob < min_prob:
            break
        k += 1
    return k


def business_score_from_proba(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_features: int,
    max_k: int = DEFAULT_MAX_TARGETS,
    threshold: float | None = None,
) -> float:
    """Compute business score from probabilities via top-k ranking (task metric).

    When ``threshold`` is None, selects k in ``1..max_k`` that maximizes score.
    When ``threshold`` is set, keeps samples above threshold and caps at ``max_k``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if threshold is not None:
        y_pred = (y_prob >= threshold).astype(int)
        if y_pred.sum() > max_k:
            above_thresh = np.where(y_pred == 1)[0]
            sorted_above = above_thresh[np.argsort(y_prob[above_thresh])[::-1]]
            y_pred = np.zeros(len(y_prob), dtype=int)
            y_pred[sorted_above[:max_k]] = 1
        return business_score_from_binary(y_true, y_pred, n_features)

    sorted_idx = np.argsort(y_prob)[::-1]
    best_score = -np.inf
    best_k = 0
    cumulative_tp = 0
    cumulative_fp = 0

    for k in range(1, min(max_k, len(y_prob)) + 1):
        idx = sorted_idx[k - 1]
        if y_true[idx] == 1:
            cumulative_tp += 1
        else:
            cumulative_fp += 1
        score_k = (cumulative_tp * 10) - (cumulative_fp * 5) - (n_features * 200)
        if score_k > best_score:
            best_score = score_k
            best_k = k

    y_pred = np.zeros(len(y_prob), dtype=int)
    if best_k > 0:
        y_pred[sorted_idx[:best_k]] = 1

    return business_score_from_binary(y_true, y_pred, n_features)


def business_score_no_var_from_proba(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_k: int = DEFAULT_MAX_TARGETS,
    threshold: float | None = None,
) -> float:
    """Business score without the per-variable penalty (TP/FP only)."""
    return business_score_from_proba(
        y_true,
        y_prob,
        n_features=0,
        max_k=max_k,
        threshold=threshold,
    )


def custom_scorer(
    estimator,
    X: np.ndarray,
    y_true: np.ndarray,
    max_k: int = DEFAULT_MAX_TARGETS,
) -> float:
    """Sklearn scorer: top-k profit curve with feature-cost penalty."""
    y_prob = estimator.predict_proba(X)[:, 1]
    return business_score_from_proba(
        y_true,
        y_prob,
        n_features=X.shape[1],
        max_k=max_k,
    )


def business_scorer_no_var_penalty(
    estimator,
    X: np.ndarray,
    y_true: np.ndarray,
    max_k: int = DEFAULT_MAX_TARGETS,
) -> float:
    """Sklearn scorer: top-k profit without variable penalty."""
    y_prob = estimator.predict_proba(X)[:, 1]
    return business_score_no_var_from_proba(y_true, y_prob, max_k=max_k)


def f1_at_optimal_top_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_k: int = DEFAULT_MAX_TARGETS,
) -> float:
    """F1 at the top-k cutoff that maximizes F1 (diagnostic only)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    ranking = np.argsort(y_prob)[::-1]
    ranked_targets = y_true[ranking]

    total_positives = int((y_true == 1).sum())
    if total_positives == 0:
        return 0.0

    limit = min(max_k, len(ranked_targets))
    best_f1 = 0.0
    cumulative_tp = 0
    cumulative_fp = 0

    for k in range(1, limit + 1):
        if ranked_targets[k - 1] == 1:
            cumulative_tp += 1
        else:
            cumulative_fp += 1
        precision = cumulative_tp / k
        recall = cumulative_tp / total_positives
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)

    return float(best_f1)


def f1_scorer_wrapper(
    estimator,
    X: np.ndarray,
    y_true: np.ndarray,
    max_k: int = DEFAULT_MAX_TARGETS,
) -> float:
    """Sklearn scorer: F1 at top-k that maximizes F1 within max_k."""
    y_prob = estimator.predict_proba(X)[:, 1]
    return f1_at_optimal_top_k(y_true, y_prob, max_k=max_k)


def get_classifier(y: np.ndarray) -> LGBMClassifier:
    """Get a LightGBM classifier with specified class imbalance handling."""
    scale_pos_weight = get_scale_pos_weight(y)
    return LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        verbose=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )


def get_scale_pos_weight(y: np.ndarray) -> float:
    """Calculate scale_pos_weight for LightGBM based on class distribution."""
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()
    return n_negative / n_positive if n_positive > 0 else 1.0
