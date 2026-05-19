"""Utility functions for feature selection."""

import numpy as np
from lightgbm import LGBMClassifier


def custom_scorer(estimator, X: np.ndarray, y_true: np.ndarray) -> float:
    """Compute business metric for feature selection."""
    y_pred_proba = estimator.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > (1.0 / 3.0)).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    n_features = X.shape[1]

    return (tp * 10) - (fp * 5) - (n_features * 200)


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
