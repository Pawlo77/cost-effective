"""Sparse top-k feature-selection helpers for the alternative notebook."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

from cost_effective.dataset.utils import DEFAULT_MAX_TARGETS
from cost_effective.models.modeling import (
    build_f1_curve,
    build_profit_curve,
    compute_oof_probabilities,
)

PRESCREEN_METHODS: tuple[str, ...] = ("mutual_info", "abs_corr", "logistic_l1")
EXTENDED_PRESCREEN_METHODS: tuple[str, ...] = (
    *PRESCREEN_METHODS,
    "lightgbm_gain",
    "sparse_gam_spam",
    "lambdamart",
)
OPTIONAL_PRESCREEN_METHODS: tuple[str, ...] = ("abess_logistic", "ebm")

PRESCREEN_METHOD_LABELS: dict[str, str] = {
    "mutual_info": "Mutual information",
    "abs_corr": "Absolute correlation",
    "logistic_l1": "Sparse L1 logistic",
    "lightgbm_gain": "LightGBM gain",
    "sparse_gam_spam": "Sparse GAM / SpAM proxy",
    "lambdamart": "LambdaMART gain",
    "abess_logistic": "ABESS best-subset logistic",
    "ebm": "Explainable Boosting Machine",
}

PRESCREEN_OPTIONAL_PACKAGES: dict[str, str] = {
    "abess_logistic": "abess",
    "ebm": "interpret",
}


@dataclass(frozen=True, slots=True)
class BinaryCandidates:
    """Binarized train/test matrices and mapping to original paid variables."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    source_map: dict[str, tuple[str, ...]]
    thresholds: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PairwiseSparseRanker:
    """Sparse linear ranker trained on positive-negative pair differences."""

    features: tuple[str, ...]
    scaler: StandardScaler
    model: LogisticRegression


def prescreen_method_availability(
    methods: Sequence[str] = (*EXTENDED_PRESCREEN_METHODS, *OPTIONAL_PRESCREEN_METHODS),
) -> pd.DataFrame:
    """Return package availability for known prescreen ranking methods."""
    import importlib.util

    rows: list[dict[str, Any]] = []
    for method in methods:
        package = PRESCREEN_OPTIONAL_PACKAGES.get(method)
        rows.append({
            "method": method,
            "label": PRESCREEN_METHOD_LABELS.get(method, method),
            "package": package or "project dependencies",
            "available": package is None or importlib.util.find_spec(package) is not None,
        })
    return pd.DataFrame(rows)


def rank_features_by_method(
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank all input features with a single prescreening signal."""
    scores = _scores_for_method(method, X, y, random_state)
    return _normalized_rank_frame(method, X.columns, scores)


def collect_prescreen_rankings(
    X: pd.DataFrame,
    y: pd.Series,
    methods: Sequence[str] = EXTENDED_PRESCREEN_METHODS,
    *,
    random_state: int = 42,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run multiple all-variable prescreen methods and collect failures separately."""
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for method in methods:
        try:
            frames.append(rank_features_by_method(X, y, method, random_state=random_state))
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append({
                "method": method,
                "label": PRESCREEN_METHOD_LABELS.get(method, method),
                "error": repr(exc),
            })

    if frames:
        rankings = pd.concat(frames, ignore_index=True)
    else:
        rankings = pd.DataFrame(columns=["feature", "method", "raw_score", "rank", "rank_score"])
    return rankings, pd.DataFrame(failures)


def combine_prescreen_rankings(
    rankings: pd.DataFrame,
    methods: Sequence[str],
    *,
    name: str | None = None,
) -> pd.DataFrame:
    """Average normalized feature ranks for a selected set of prescreen methods."""
    method_tuple = tuple(dict.fromkeys(methods))
    if not method_tuple:
        raise ValueError("At least one prescreen method is required")

    subset = rankings.loc[rankings["method"].isin(method_tuple)].copy()
    present = set(subset["method"].drop_duplicates())
    missing = [method for method in method_tuple if method not in present]
    if missing:
        raise ValueError(f"Missing rankings for prescreen methods: {missing}")

    scores_wide = subset.pivot(index="feature", columns="method", values="rank_score")
    ranks_wide = subset.pivot(index="feature", columns="method", values="rank")
    combo_name = name or prescreen_combo_name(method_tuple)
    ranking = pd.DataFrame({
        "feature": scores_wide.index,
        "prescreen_name": combo_name,
        "prescreen_methods": " + ".join(method_tuple),
        "prescreen_method_count": len(method_tuple),
        "ensemble_score": scores_wide.mean(axis=1),
        "mean_rank": ranks_wide.mean(axis=1),
    }).reset_index(drop=True)
    ranking = ranking.sort_values(
        ["ensemble_score", "mean_rank", "feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking["order"] = np.arange(1, len(ranking) + 1)
    return ranking.merge(scores_wide.add_prefix("score__").reset_index(), on="feature")


def build_prescreen_ensembles(
    rankings: pd.DataFrame,
    methods: Sequence[str] | None = None,
    *,
    combination_sizes: Sequence[int] = (1, 2, 3),
    include_all: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build single-method and multi-method prescreen rankings from long rankings."""
    method_list = list(dict.fromkeys(methods or rankings["method"].drop_duplicates().tolist()))
    sizes = [size for size in dict.fromkeys(combination_sizes) if 0 < size <= len(method_list)]
    if include_all and method_list and len(method_list) not in sizes:
        sizes.append(len(method_list))

    ensembles: dict[str, pd.DataFrame] = {}
    seen: set[tuple[str, ...]] = set()
    for size in sizes:
        for combo in combinations(method_list, size):
            if combo in seen:
                continue
            seen.add(combo)
            name = prescreen_combo_name(combo, all_count=len(method_list))
            ensembles[name] = combine_prescreen_rankings(rankings, combo, name=name)
    return ensembles


def prescreen_combo_name(methods: Sequence[str], *, all_count: int | None = None) -> str:
    """Create stable names for single, pair, triple, and all-method prescreens."""
    method_tuple = tuple(methods)
    joined = "+".join(method_tuple)
    if all_count is not None and len(method_tuple) == all_count and all_count > 3:
        return f"all__{joined}"
    if len(method_tuple) == 1:
        return f"single__{joined}"
    if len(method_tuple) == 2:
        return f"pair__{joined}"
    if len(method_tuple) == 3:
        return f"triple__{joined}"
    return f"ensemble_{len(method_tuple)}__{joined}"


def build_all_k_prescreen_ensembles(
    rankings: pd.DataFrame,
    methods: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build prescreen ensembles for every non-empty method combination."""
    method_list = list(dict.fromkeys(methods or rankings["method"].drop_duplicates().tolist()))
    if not method_list:
        return {}
    return build_prescreen_ensembles(
        rankings,
        methods=method_list,
        combination_sizes=tuple(range(1, len(method_list) + 1)),
        include_all=False,
    )


def combine_prescreen_rankings_weighted(
    rankings: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    name: str,
) -> pd.DataFrame:
    """Combine prescreen rankings with explicit method weights."""
    clean_weights = {
        method: float(weight) for method, weight in weights.items() if float(weight) > 0
    }
    if not clean_weights:
        raise ValueError("At least one positive prescreen weight is required")

    methods = tuple(clean_weights)
    subset = rankings.loc[rankings["method"].isin(methods)].copy()
    present = set(subset["method"].drop_duplicates())
    missing = [method for method in methods if method not in present]
    if missing:
        raise ValueError(f"Missing rankings for weighted prescreen methods: {missing}")

    weight_sum = sum(clean_weights.values())
    normalized_weights = {method: weight / weight_sum for method, weight in clean_weights.items()}
    scores_wide = subset.pivot(index="feature", columns="method", values="rank_score")
    ranks_wide = subset.pivot(index="feature", columns="method", values="rank")
    score = sum(scores_wide[method] * normalized_weights[method] for method in methods)
    mean_rank = sum(ranks_wide[method] * normalized_weights[method] for method in methods)

    ranking = pd.DataFrame({
        "feature": scores_wide.index,
        "prescreen_name": name,
        "prescreen_methods": " + ".join(methods),
        "prescreen_method_count": len(methods),
        "ensemble_score": score.to_numpy(),
        "mean_rank": mean_rank.to_numpy(),
    }).reset_index(drop=True)
    ranking = ranking.sort_values(
        ["ensemble_score", "mean_rank", "feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking["order"] = np.arange(1, len(ranking) + 1)
    ranking["prescreen_weights"] = ";".join(
        f"{method}={normalized_weights[method]:.6f}" for method in methods
    )
    return ranking.merge(scores_wide.add_prefix("score__").reset_index(), on="feature")


def prescreen_weights_from_single_scores(
    scores: pd.DataFrame,
    *,
    method_column: str = "prescreen_methods",
    score_column: str = "oof_business_score",
    min_weight: float = 0.05,
) -> dict[str, float]:
    """Derive positive prescreen weights from each single method's best score."""
    if scores.empty or method_column not in scores or score_column not in scores:
        return {}
    single = scores.loc[scores.get("prescreen_method_count", 1).eq(1)].copy()
    if single.empty:
        return {}
    best = single.groupby(method_column, as_index=False)[score_column].max()
    values = best[score_column].astype(float)
    shifted = values - values.min()
    if float(shifted.sum()) <= 0:
        raw = np.ones(len(best), dtype=float)
    else:
        raw = shifted.to_numpy(dtype=float)
    raw = raw + float(min_weight) * max(float(raw.mean()), 1.0)
    weights = raw / raw.sum()
    return dict(zip(best[method_column], weights, strict=False))


def evaluate_prescreen_rankings(
    X: pd.DataFrame,
    y: pd.Series,
    ranking_map: Mapping[str, pd.DataFrame],
    *,
    sizes: Sequence[int] = (1, 2, 3, 5, 8, 10, 15, 20, 30),
    estimator_factory: Any | None = None,
    cv: int = 5,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> pd.DataFrame:
    """Score top-k feature sets from each prescreen ranking with OOF probabilities."""
    rows: list[dict[str, Any]] = []
    for ranking_name, ranking in ranking_map.items():
        ordered_features = ranking.sort_values("order")["feature"].tolist()
        feature_sets = build_top_feature_sets(ordered_features, sizes=sizes)
        for feature_set_name, features in feature_sets.items():
            oof = compute_oof_probabilities(
                X[list(features)],
                y,
                estimator_factory=estimator_factory,
                cv=cv,
            )
            row = score_oof_predictions(
                y,
                oof,
                features,
                method=f"{ranking_name}::{feature_set_name}",
                candidate_column_count=len(features),
                max_targets=max_targets,
            )
            row.update({
                "prescreen_name": ranking_name,
                "prescreen_methods": ranking["prescreen_methods"].iloc[0],
                "prescreen_method_count": int(ranking["prescreen_method_count"].iloc[0]),
                "feature_set_name": feature_set_name,
            })
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["oof_business_score", "source_feature_count"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def candidate_pools_from_prescreen_scores(
    ranking_map: Mapping[str, pd.DataFrame],
    scores: pd.DataFrame,
    *,
    top_n: int = 60,
    max_pools: int = 6,
    include_names: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Create downstream candidate pools from the best evaluated prescreen rankings."""
    if scores.empty:
        names = list(include_names)
    else:
        best_names = (
            scores.sort_values(
                ["oof_business_score", "source_feature_count"],
                ascending=[False, True],
            )["prescreen_name"]
            .drop_duplicates()
            .head(max_pools)
        )
        names = [*include_names, *best_names.tolist()]

    pools: dict[str, list[str]] = {}
    for name in dict.fromkeys(names):
        ranking = ranking_map.get(name)
        if ranking is None:
            continue
        pools[name] = ranking.sort_values("order")["feature"].head(top_n).tolist()
    return pools


def rank_features_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    methods: Sequence[str] = PRESCREEN_METHODS,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Average normalized ranks from simple model-agnostic and sparse-linear signals."""
    method_frames = []
    for method in methods:
        scores = _scores_for_method(method, X, y, random_state)
        method_frames.append(_normalized_rank_frame(method, X.columns, scores))

    long = pd.concat(method_frames, ignore_index=True)
    scores_wide = long.pivot(index="feature", columns="method", values="rank_score")
    ranks_wide = long.pivot(index="feature", columns="method", values="rank")
    ranking = pd.DataFrame({
        "feature": scores_wide.index,
        "ensemble_score": scores_wide.mean(axis=1),
        "mean_rank": ranks_wide.mean(axis=1),
    }).reset_index(drop=True)
    ranking = ranking.sort_values(
        ["ensemble_score", "mean_rank", "feature"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking["order"] = np.arange(1, len(ranking) + 1)
    return ranking.merge(scores_wide.add_prefix("score__").reset_index(), on="feature")


def build_top_feature_sets(
    ranked_features: Sequence[str],
    sizes: Sequence[int] = (1, 2, 3, 5, 8, 10, 15, 20, 30),
) -> dict[str, list[str]]:
    """Create top-k feature sets from a ranked list."""
    features = list(dict.fromkeys(ranked_features))
    return {
        f"top_{size:02d}": features[:size]
        for size in sorted(set(sizes))
        if 0 < size <= len(features)
    }


def make_quantile_binary_candidates(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: Sequence[str],
    *,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
) -> BinaryCandidates:
    """Binarize continuous features for rule-list/tree/risk-score methods."""
    train_out = pd.DataFrame(index=X_train.index)
    test_out = pd.DataFrame(index=X_test.index)
    source_map: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, Any]] = []

    for feature in features:
        if feature not in X_train.columns or feature not in X_test.columns:
            continue
        values = X_train[feature]
        for quantile in quantiles:
            threshold = float(values.quantile(quantile))
            name = f"{feature}__ge_q{int(100 * quantile):02d}"
            train_out[name] = (X_train[feature] >= threshold).astype(int)
            test_out[name] = (X_test[feature] >= threshold).astype(int)
            source_map[name] = (feature,)
            rows.append({"feature": name, "source_feature": feature, "threshold": threshold})

    return BinaryCandidates(train_out, test_out, source_map, pd.DataFrame(rows))


def source_features(
    features: Iterable[str],
    source_map: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return original paid variables needed by raw or derived columns."""
    sources: set[str] = set()
    for feature in features:
        sources.update(source_map.get(feature, (feature,)))
    return tuple(sorted(sources, key=_var_sort_key))


def score_oof_predictions(
    y: pd.Series,
    oof_scores: np.ndarray,
    source_feature_names: Sequence[str],
    *,
    method: str,
    candidate_column_count: int | None = None,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> dict[str, Any]:
    """Score arbitrary OOF ranking scores with the project top-k objective."""
    unique_sources = tuple(dict.fromkeys(source_feature_names))
    score_array = np.asarray(oof_scores, dtype=float)
    profit = build_profit_curve(
        y,
        score_array,
        feature_count=len(unique_sources),
        max_targets=max_targets,
    )
    try:
        f1 = build_f1_curve(y, score_array, max_targets=max_targets)
        f1_score = f1.best_f1
        roc_auc_score = f1.roc_auc
    except ValueError:
        f1_score = np.nan
        roc_auc_score = np.nan

    best_curve_rows = profit.curve.loc[profit.curve["k"].eq(profit.best_k)]
    best_curve_row = best_curve_rows.iloc[0] if not best_curve_rows.empty else None
    tp_at_optimal_k = int(best_curve_row["tp"]) if best_curve_row is not None else 0
    fp_at_optimal_k = int(best_curve_row["fp"]) if best_curve_row is not None else 0
    feature_penalty = float(len(unique_sources) * 200)
    return {
        "method": method,
        "candidate_column_count": candidate_column_count or len(unique_sources),
        "source_feature_count": len(unique_sources),
        "oof_business_score": profit.best_score,
        "oof_optimal_k": profit.best_k,
        "oof_tp_at_optimal_k": tp_at_optimal_k,
        "oof_fp_at_optimal_k": fp_at_optimal_k,
        "oof_n_predictions": len(y),
        "oof_selected_rate": float(profit.best_k / len(y)) if len(y) else 0.0,
        "feature_penalty": feature_penalty,
        "gross_score": float(profit.best_score + feature_penalty),
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
        "source_features": list(unique_sources),
    }


def train_test_shift_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    """Compare train/test feature distributions without using test labels."""
    rows = []
    for feature in features:
        if feature not in X_train.columns or feature not in X_test.columns:
            continue
        train = X_train[feature].astype(float)
        test = X_test[feature].astype(float)
        train_std = float(train.std(ddof=0))
        test_std = float(test.std(ddof=0))
        pooled = max(np.sqrt((train_std**2 + test_std**2) / 2), 1e-12)
        q01 = float(train.quantile(0.01))
        q99 = float(train.quantile(0.99))
        rows.append({
            "feature": feature,
            "train_mean": float(train.mean()),
            "test_mean": float(test.mean()),
            "standardized_mean_diff": abs(float(train.mean() - test.mean())) / pooled,
            "train_std": train_std,
            "test_std": test_std,
            "test_outside_train_1_99_rate": float(((test < q01) | (test > q99)).mean()),
        })
    return pd.DataFrame(rows).sort_values(
        ["standardized_mean_diff", "test_outside_train_1_99_rate"],
        ascending=False,
    )


def selected_features_from_coef(model: Any, features: Sequence[str]) -> tuple[str, ...]:
    """Extract non-zero coefficient features from linear sparse models."""
    coef = np.asarray(getattr(model, "coef_", []), dtype=float).reshape(-1)
    if coef.size != len(features):
        return tuple(features)
    return tuple(
        feature for feature, value in zip(features, coef, strict=False) if abs(value) > 1e-12
    )


def fit_pairwise_sparse_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    C: float = 0.1,
    max_pairs: int = 20000,
    random_state: int = 42,
) -> PairwiseSparseRanker:
    """Fit a sparse pairwise ranker on positive-negative differences."""
    rng = np.random.default_rng(random_state)
    y_array = y.to_numpy()
    positives = np.flatnonzero(y_array == 1)
    negatives = np.flatnonzero(y_array == 0)
    n_pairs = min(max_pairs, max(len(positives), 1) * max(len(negatives), 1))
    pos_idx = rng.choice(positives, size=n_pairs, replace=True)
    neg_idx = rng.choice(negatives, size=n_pairs, replace=True)

    matrix = X[list(features)].to_numpy(dtype=float)
    diffs = matrix[pos_idx] - matrix[neg_idx]
    pair_x = np.vstack([diffs, -diffs])
    pair_y = np.r_[np.ones(n_pairs, dtype=int), np.zeros(n_pairs, dtype=int)]

    scaler = StandardScaler().fit(pair_x)
    model = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=C,
        class_weight="balanced",
        random_state=random_state,
        max_iter=1000,
    )
    model.fit(scaler.transform(pair_x), pair_y)
    return PairwiseSparseRanker(tuple(features), scaler, model)


def score_pairwise_ranker(ranker: PairwiseSparseRanker, X: pd.DataFrame) -> np.ndarray:
    """Score rows for a fitted pairwise sparse ranker."""
    matrix = ranker.scaler.transform(X[list(ranker.features)].to_numpy(dtype=float))
    return matrix @ ranker.model.coef_.reshape(-1) + float(ranker.model.intercept_[0])


def fit_lambdamart_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    random_state: int = 42,
    n_estimators: int = 180,
) -> Any:
    """Fit a LambdaMART ranker treating the full sample as one ranking group."""
    model = _make_lambdamart(random_state=random_state, n_estimators=n_estimators)
    model.fit(X[list(features)], y.astype(int), group=[len(y)])
    return model


def score_lambdamart_ranker(model: Any, X: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    """Score rows with a fitted LambdaMART model."""
    return np.asarray(model.predict(X[list(features)]), dtype=float)


def pairwise_sparse_oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    C: float = 0.1,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """OOF scores for the sparse pairwise ranker."""
    oof = np.zeros(len(y), dtype=float)
    selected: set[str] = set()
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        ranker = fit_pairwise_sparse_ranker(
            X.iloc[train_idx],
            y.iloc[train_idx],
            features,
            C=C,
            random_state=random_state + fold,
        )
        oof[val_idx] = score_pairwise_ranker(ranker, X.iloc[val_idx])
        selected.update(selected_features_from_coef(ranker.model, ranker.features))
    return oof, tuple(sorted(selected, key=_var_sort_key))


def lambdamart_oof_scores(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    *,
    cv: int = 5,
    random_state: int = 42,
    n_estimators: int = 180,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """OOF scores and non-zero-importance features for LambdaMART."""
    oof = np.zeros(len(y), dtype=float)
    selected: set[str] = set()
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    feature_list = list(features)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        model = fit_lambdamart_ranker(
            X.iloc[train_idx],
            y.iloc[train_idx],
            feature_list,
            random_state=random_state + fold,
            n_estimators=n_estimators,
        )
        oof[val_idx] = score_lambdamart_ranker(model, X.iloc[val_idx], feature_list)
        selected.update(
            feature
            for feature, importance in zip(feature_list, model.feature_importances_, strict=False)
            if float(importance) > 0
        )
    if not selected:
        selected.update(feature_list)
    return oof, tuple(sorted(selected, key=_var_sort_key))


def rank_test_customers(test_scores: np.ndarray, *, k: int = DEFAULT_MAX_TARGETS) -> pd.DataFrame:
    """Return test row indices ranked by descending score."""
    scores = np.asarray(test_scores, dtype=float)
    order = np.argsort(scores)[::-1][: min(k, len(scores))]
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "sample_index": order,
        "score": scores[order],
    })


def _scores_for_method(
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> np.ndarray:
    if method == "mutual_info":
        return mutual_info_classif(X, y, random_state=random_state)
    if method == "abs_corr":
        return X.corrwith(y).abs().to_numpy()
    if method == "logistic_l1":
        return _logistic_l1_scores(X, y, random_state)
    if method == "lightgbm_gain":
        return _lightgbm_gain_scores(X, y, random_state)
    if method == "sparse_gam_spam":
        return _sparse_gam_spam_scores(X, y, random_state)
    if method == "lambdamart":
        return _lambdamart_gain_scores(X, y, random_state)
    if method == "abess_logistic":
        return _abess_logistic_scores(X, y)
    if method == "ebm":
        return _ebm_scores(X, y, random_state)
    raise ValueError(f"Unknown ranking method: {method}")


def _logistic_l1_scores(X: pd.DataFrame, y: pd.Series, random_state: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(X)
    for c_value in (0.02, 0.05, 0.1, 0.5, 1.0):
        model = LogisticRegression(
            solver="liblinear",
            l1_ratio=1.0,
            C=c_value,
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
        )
        model.fit(scaled, y)
        scores = np.abs(model.coef_[0])
        if np.any(scores > 0):
            return scores
    return scores


def _lightgbm_gain_scores(X: pd.DataFrame, y: pd.Series, random_state: int) -> np.ndarray:
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        scale_pos_weight=_scale_pos_weight(y),
        importance_type="gain",
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X, y)
    return np.asarray(model.feature_importances_, dtype=float)


def _lambdamart_gain_scores(X: pd.DataFrame, y: pd.Series, random_state: int) -> np.ndarray:
    model = _make_lambdamart(random_state=random_state)
    model.fit(X, y.astype(int), group=[len(y)])
    scores = np.asarray(model.feature_importances_, dtype=float)
    if np.any(scores > 0):
        return scores
    return _lightgbm_gain_scores(X, y, random_state)


def _sparse_gam_spam_scores(X: pd.DataFrame, y: pd.Series, random_state: int) -> np.ndarray:
    matrix = X.to_numpy(dtype=float)
    try:
        transformer = SplineTransformer(
            n_knots=4,
            degree=3,
            include_bias=False,
            extrapolation="constant",
            sparse_output=True,
        )
    except TypeError:
        transformer = SplineTransformer(
            n_knots=4,
            degree=3,
            include_bias=False,
            extrapolation="constant",
        )

    basis = transformer.fit_transform(matrix)
    scaled = StandardScaler(with_mean=False).fit_transform(basis)
    scores = np.zeros(X.shape[1], dtype=float)
    for c_value in (0.01, 0.03, 0.1, 0.3):
        model = LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            C=c_value,
            class_weight="balanced",
            random_state=random_state,
            max_iter=700,
            tol=1e-3,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(scaled, y)
        coef = np.abs(model.coef_[0])
        scores = _collapse_basis_scores(coef, X.shape[1])
        if np.any(scores > 0):
            return scores
    return scores


def _abess_logistic_scores(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    try:
        from abess import LogisticRegression as AbessLogisticRegression
    except ImportError:
        from abess.linear import LogisticRegression as AbessLogisticRegression

    support_limit = min(30, X.shape[1])
    model = AbessLogisticRegression(support_size=range(support_limit + 1))
    model.fit(X, y)
    coef = np.asarray(getattr(model, "coef_", []), dtype=float).reshape(-1)
    if coef.size != X.shape[1]:
        return np.zeros(X.shape[1], dtype=float)
    return np.abs(coef)


def _ebm_scores(X: pd.DataFrame, y: pd.Series, random_state: int) -> np.ndarray:
    from interpret.glassbox import ExplainableBoostingClassifier

    model = ExplainableBoostingClassifier(
        interactions=0,
        max_bins=64,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X, y)
    scores = np.zeros(X.shape[1], dtype=float)
    importances = np.asarray(model.term_importances(), dtype=float)
    term_features = getattr(model, "term_features_", [(idx,) for idx in range(len(importances))])
    for features, importance in zip(term_features, importances, strict=False):
        if not features:
            continue
        share = float(abs(importance)) / len(features)
        for feature_idx in features:
            if feature_idx < len(scores):
                scores[feature_idx] += share
    return scores


def _collapse_basis_scores(coef: np.ndarray, n_features: int) -> np.ndarray:
    if n_features == 0:
        return np.array([], dtype=float)
    if coef.size % n_features != 0:
        return np.resize(coef, n_features)
    basis_per_feature = coef.size // n_features
    return coef.reshape(n_features, basis_per_feature).sum(axis=1)


def _scale_pos_weight(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return negatives / positives if positives > 0 else 1.0


def _make_lambdamart(*, random_state: int, n_estimators: int = 180) -> Any:
    from lightgbm import LGBMRanker

    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        random_state=random_state,
        importance_type="gain",
        verbose=-1,
        n_jobs=1,
    )


def _normalized_rank_frame(
    method: str,
    features: Sequence[str],
    scores: np.ndarray,
) -> pd.DataFrame:
    clean = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ranks = pd.Series(clean, index=features).rank(ascending=False, method="average")
    n_features = len(features)
    if n_features == 1:
        rank_score = np.ones(n_features)
    else:
        rank_score = 1 - ((ranks.to_numpy() - 1) / (n_features - 1))
    return pd.DataFrame({
        "feature": list(features),
        "method": method,
        "raw_score": clean,
        "rank": ranks.to_numpy(),
        "rank_score": rank_score,
    })


def _var_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("var_") and name[4:].isdigit():
        return int(name[4:]), name
    return 10**9, name
