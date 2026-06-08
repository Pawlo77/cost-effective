"""Shared top-k business scoring helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from cost_effective.dataset.utils import DEFAULT_MAX_TARGETS


def exact_business_score_at_k(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    feature_count: int,
    k: int,
) -> dict[str, float | int]:
    """Compute the project business score at an exact top-k cutoff."""
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    k = int(min(max(k, 0), len(score_array)))
    if k == 0:
        return {"k": 0, "tp": 0, "fp": 0, "business_score": -feature_count * 200}
    chosen = np.argsort(score_array)[::-1][:k]
    tp = int((y_array[chosen] == 1).sum())
    fp = int(k - tp)
    return {
        "k": k,
        "tp": tp,
        "fp": fp,
        "business_score": (tp * 10) - (fp * 5) - (feature_count * 200),
    }


def exact_top_k_business_curve(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    feature_count: int,
    k_values: Sequence[int],
) -> pd.DataFrame:
    """Compute exact top-k business scores for diagnostics."""
    rows = [
        exact_business_score_at_k(y_true, scores, feature_count=feature_count, k=k)
        for k in sorted(dict.fromkeys(k_values))
    ]
    return pd.DataFrame(rows)


def topk_business_curve(
    y_true: Sequence[int] | pd.Series,
    scores: Sequence[float] | np.ndarray,
    *,
    feature_count: int,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> pd.DataFrame:
    """Business curve for k=0..max_targets using the final unscaled objective."""
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    if len(y_array) != len(score_array):
        raise ValueError("y_true and scores must have the same length")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    feature_penalty = float(int(feature_count) * 200)
    rows: list[dict[str, Any]] = [
        {
            "k": 0,
            "tp": 0,
            "fp": 0,
            "gross_score": 0.0,
            "business_score": -feature_penalty,
            "threshold": np.nan,
            "feature_penalty": feature_penalty,
        }
    ]
    limit = min(max(int(max_targets), 0), len(score_array))
    if limit > 0:
        order = np.argsort(score_array)[::-1]
        ranked_y = y_array[order][:limit]
        ranked_scores = score_array[order][:limit]
        cumulative_tp = np.cumsum(ranked_y == 1)
        cumulative_fp = np.cumsum(ranked_y == 0)
        gross = cumulative_tp * 10 - cumulative_fp * 5
        rows.extend(
            {
                "k": int(k),
                "tp": int(tp),
                "fp": int(fp),
                "gross_score": float(gross_score),
                "business_score": float(gross_score - feature_penalty),
                "threshold": float(threshold),
                "feature_penalty": feature_penalty,
            }
            for k, tp, fp, gross_score, threshold in zip(
                range(1, limit + 1), cumulative_tp, cumulative_fp, gross, ranked_scores, strict=True
            )
        )
    return pd.DataFrame(rows)


def feature_cost_scale(
    evaluation_row_count: int,
    *,
    reference_row_count: int = DEFAULT_MAX_TARGETS,
) -> float:
    """Scale feature-acquisition cost to the number of rows being evaluated."""
    reference = int(reference_row_count)
    if reference <= 0:
        raise ValueError("reference_row_count must be positive")
    return float(evaluation_row_count) / float(reference)


def scaled_business_curve(
    y_true: pd.Series,
    scores: np.ndarray,
    *,
    feature_count: int,
    max_targets: int = DEFAULT_MAX_TARGETS,
    feature_cost_reference_row_count: int = DEFAULT_MAX_TARGETS,
    feature_cost_per_feature: float = 200.0,
    true_positive_value: float = 10.0,
    false_positive_cost: float = 5.0,
) -> pd.DataFrame:
    """Build a top-k business curve with the full objective scaled by row count."""
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    if len(y_array) != len(score_array):
        raise ValueError("y_true and scores must have the same length")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite for final inner-OOF scoring")

    target_limit = min(max(int(max_targets), 0), len(y_array))
    scale = feature_cost_scale(
        len(y_array),
        reference_row_count=feature_cost_reference_row_count,
    )
    scaled_tp_value = float(true_positive_value) * scale
    scaled_fp_cost = float(false_positive_cost) * scale
    scaled_feature_cost = float(feature_count) * float(feature_cost_per_feature) * scale
    unscaled_feature_cost = float(feature_count) * float(feature_cost_per_feature)

    if target_limit == 0:
        return pd.DataFrame([
            {
                "k": 0,
                "tp": 0,
                "fp": 0,
                "business_value_before_feature_cost": 0.0,
                "business_score": -scaled_feature_cost,
                "business_score_unscaled_feature_cost": -unscaled_feature_cost,
                "threshold": np.nan,
                "evaluation_row_count": len(y_array),
                "target_limit": target_limit,
                "feature_cost_scale": scale,
                "scaled_true_positive_value": scaled_tp_value,
                "scaled_false_positive_cost": scaled_fp_cost,
                "scaled_feature_cost": scaled_feature_cost,
                "unscaled_feature_cost": unscaled_feature_cost,
            }
        ])

    ranking = np.argsort(score_array)[::-1]
    ranked_targets = y_array[ranking]
    ranked_scores = score_array[ranking]
    cumulative_tp = np.cumsum(ranked_targets[:target_limit] == 1)
    cumulative_fp = np.cumsum(ranked_targets[:target_limit] == 0)
    business_value = cumulative_tp * scaled_tp_value - cumulative_fp * scaled_fp_cost
    return pd.DataFrame({
        "k": np.arange(1, target_limit + 1),
        "tp": cumulative_tp.astype(int),
        "fp": cumulative_fp.astype(int),
        "business_value_before_feature_cost": business_value.astype(float),
        "business_score": business_value.astype(float) - scaled_feature_cost,
        "business_score_unscaled_feature_cost": business_value.astype(float)
        - unscaled_feature_cost,
        "threshold": ranked_scores[:target_limit],
        "evaluation_row_count": len(y_array),
        "target_limit": target_limit,
        "feature_cost_scale": scale,
        "scaled_true_positive_value": scaled_tp_value,
        "scaled_false_positive_cost": scaled_fp_cost,
        "scaled_feature_cost": scaled_feature_cost,
        "unscaled_feature_cost": unscaled_feature_cost,
    })
