"""High-level steps shared by modeling notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from .dataset.utils import best_k_break_even
from .models import (
    ClusterSplitConfig,
    ClusterSplitTargetingResult,
    FusionResult,
    SegmentConfig,
    SegmentTargetingResult,
    TargetingConfig,
    attach_best_hyperparams,
    build_f1_curve,
    build_model_factories,
    build_profit_curve,
    choose_targeting_k,
    compare_models_on_feature_sets,
    compute_cluster_split_oof_probabilities,
    experts_summary_frame,
    fit_final_model_and_predict,
    predict_cluster_split_test_probabilities,
    predict_fusion_test_probabilities,
    predict_segment_test_probabilities,
    rank_test_indices,
    run_fusion_committee,
)
from .models.cluster_k_selection import kmeans_k_diagnostics
from .models.cluster_split_targeting import build_cluster_y_profile, fit_train_cluster_labels
from .models.dataclasses import F1CurveResult, ProfitCurveResult
from .models.profit_targeting import TargetingSelectionResult, expected_value_per_contact
from .models.rank_fusion import FusionExpertConfig
from .models.segment_targeting import compute_segment_oof_probabilities
from .notebook_setup import ModelingNotebookContext
from .utils import (
    MODELING_CV_FOLDS,
    feature_set_candidates_from_selection_results,
    run_winner_hyperparameter_search,
    write_submission_files,
)


@dataclass(frozen=True, slots=True)
class WinnerSelection:
    model_name: str
    feature_set_name: str
    features: list[str]
    cv_score_mean: float


@dataclass(frozen=True, slots=True)
class TopkOofEvaluation:
    business_curve: ProfitCurveResult
    f1_curve: F1CurveResult
    optimal_k: int
    profit_k: int
    breakeven_k: int
    confusion_matrix: pd.DataFrame
    summary_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvOofEvaluation:
    targeting: TargetingSelectionResult
    profit_curve: ProfitCurveResult
    confusion_matrix: pd.DataFrame
    roc_auc: float
    summary_lines: tuple[str, ...]


def stage_matrices(
    ctx: ModelingNotebookContext,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Unpack common matrices from context."""
    stage = ctx.stage
    return stage.X_train, stage.y_train, stage.X_test, stage.X_stage2


def run_model_comparison(
    ctx: ModelingNotebookContext,
    estimator_factories: dict[str, Any],
    feature_set_candidates: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Cross-validate model families on Stage 3 feature subsets."""
    _, y_train, _, x_stage2 = stage_matrices(ctx)
    candidates = feature_set_candidates or ctx.stage.feature_set_candidates
    return compare_models_on_feature_sets(
        x_stage2,
        y_train,
        candidates,
        estimator_factories=estimator_factories,
        cv=MODELING_CV_FOLDS,
        max_targets=ctx.max_targets,
    )


def build_estimator_factories(
    ctx: ModelingNotebookContext,
    model_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return model factories, optionally filtered by name."""
    _, y_train, _, _ = stage_matrices(ctx)
    factories = build_model_factories(y_train)
    if model_names is None:
        return factories
    return {name: factories[name] for name in model_names if name in factories}


def select_winner_from_candidates(
    comparison: pd.DataFrame,
    feature_set_candidates: dict[str, list[str]],
) -> WinnerSelection:
    """Pick winner and attach feature list from candidates dict."""
    row = comparison.loc[comparison["cv_score_mean"].idxmax()]
    feature_set_name = str(row["feature_set_name"])
    return WinnerSelection(
        model_name=str(row["model_name"]),
        feature_set_name=feature_set_name,
        features=feature_set_candidates[feature_set_name],
        cv_score_mean=float(row["cv_score_mean"]),
    )


def tune_winner(
    ctx: ModelingNotebookContext,
    winner: WinnerSelection,
    estimator_factory: Any,
    x_best: pd.DataFrame,
    hpo_n_iter: int | None = None,
) -> tuple[Any, dict[str, Any], float, Any]:
    """HPO on winner feature matrix; persist JSON under approach outputs."""
    _, y_train, _, _ = stage_matrices(ctx)
    kwargs: dict[str, Any] = {"cv_folds": MODELING_CV_FOLDS}
    if hpo_n_iter is not None:
        kwargs["hpo_n_iter"] = hpo_n_iter
    return run_winner_hyperparameter_search(
        winner.model_name,
        estimator_factory,
        x_best,
        y_train,
        ctx.outputs_path,
        **kwargs,
    )


def save_tuned_comparison(
    ctx: ModelingNotebookContext,
    winner: WinnerSelection,
    tuned_factory: Any,
    best_params: dict[str, Any],
    feature_set_candidates: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Evaluate tuned estimator and write ``model_comparison.csv``."""
    _, y_train, _, x_stage2 = stage_matrices(ctx)
    candidates = feature_set_candidates or {winner.feature_set_name: winner.features}
    comparison = compare_models_on_feature_sets(
        x_stage2,
        y_train,
        candidates,
        estimator_factories={winner.model_name: tuned_factory},
        cv=MODELING_CV_FOLDS,
        max_targets=ctx.max_targets,
    )
    comparison = attach_best_hyperparams(
        comparison,
        model_name=winner.model_name,
        best_params=best_params,
    )
    comparison.to_csv(ctx.outputs_path / "model_comparison.csv", index=False)
    return comparison


def oof_confusion_matrix(y_true: pd.Series, y_prob: np.ndarray, k: int) -> pd.DataFrame:
    """Confusion matrix for top-k OOF predictions (Jupyter-displayable)."""
    ranking = np.argsort(y_prob)[::-1]
    y_pred = np.zeros(len(y_true), dtype=int)
    y_pred[ranking[:k]] = 1
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )


def evaluate_topk_oof(
    y_train: pd.Series,
    oof_probabilities: np.ndarray,
    feature_count: int,
    max_targets: int,
) -> TopkOofEvaluation:
    """Build profit/F1 curves, optimal k, and display tables for top-k approach."""
    business_curve = build_profit_curve(
        y_train,
        oof_probabilities,
        feature_count=feature_count,
        max_targets=max_targets,
    )
    f1_curve = build_f1_curve(
        y_train,
        oof_probabilities,
        max_targets=max_targets,
    )
    business_best = business_curve.curve.loc[business_curve.curve["score"].idxmax()]
    f1_best = f1_curve.curve.loc[f1_curve.curve["f1"].idxmax()]
    profit_k = int(business_best["k"])
    breakeven_k = best_k_break_even(oof_probabilities, max_k=max_targets)

    lines = (
        f"Business-optimal: k={business_curve.best_k}, "
        f"threshold={business_curve.best_threshold:.4f}, score={business_curve.best_score:.2f}",
        f"F1-optimal: k={f1_curve.best_k}, threshold={f1_curve.best_threshold:.4f}, "
        f"f1={f1_curve.best_f1:.4f}, roc_auc={f1_curve.roc_auc:.4f}, "
        f"avg_precision={f1_curve.average_precision:.4f}",
        f"Business curve TP={int(business_best['tp'])}, FP={int(business_best['fp'])}",
        f"F1 curve TP={int(f1_best['tp'])}, FP={int(f1_best['fp'])}",
        (
            f"OOF profit-optimal k={profit_k}, break-even k={breakeven_k}, "
            f"effective k={business_curve.best_k}"
        ),
    )

    return TopkOofEvaluation(
        business_curve=business_curve,
        f1_curve=f1_curve,
        optimal_k=business_curve.best_k,
        profit_k=profit_k,
        breakeven_k=breakeven_k,
        confusion_matrix=oof_confusion_matrix(y_train, oof_probabilities, business_curve.best_k),
        summary_lines=lines,
    )


def evaluate_ev_oof(
    y_train: pd.Series,
    oof_probabilities: np.ndarray,
    feature_count: int,
    config: TargetingConfig,
) -> EvOofEvaluation:
    """Targeting selection, profit curve, and OOF diagnostics for EV approach."""
    targeting = choose_targeting_k(
        y_train,
        oof_probabilities,
        feature_count=feature_count,
        config=config,
    )
    profit_curve = build_profit_curve(
        y_train,
        oof_probabilities,
        feature_count=feature_count,
        max_targets=config.max_targets,
    )
    lines = (
        f"Strategy: {targeting.strategy}",
        f"Selected k (OOF): {targeting.selected_k}",
        (
            f"  k_ev={targeting.k_ev_threshold}, k_elbow={targeting.k_elbow}, "
            f"k_nested={targeting.k_nested_cv}, k_curve_max={targeting.k_profit_curve_max}"
        ),
        (
            f"OOF business={targeting.oof_business_score:.0f} "
            f"(TP={targeting.oof_tp}, FP={targeting.oof_fp})"
        ),
        (
            f"OOF n(p>=1/3)={targeting.n_above_break_even}, "
            f"min p in top-k={targeting.min_probability_in_selection:.4f}"
        ),
        f"OOF ROC-AUC (uncalibrated): {roc_auc_score(y_train, oof_probabilities):.4f}",
    )
    return EvOofEvaluation(
        targeting=targeting,
        profit_curve=profit_curve,
        confusion_matrix=oof_confusion_matrix(
            y_train,
            oof_probabilities,
            targeting.selected_k,
        ),
        roc_auc=float(roc_auc_score(y_train, oof_probabilities)),
        summary_lines=lines,
    )


def export_topk_test_predictions(
    ctx: ModelingNotebookContext,
    winner: WinnerSelection,
    x_stage2: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    tuned_factory: Any,
    optimal_k: int,
    cv_business_mean: float,
    hpo_cv: float,
    oof_eval: TopkOofEvaluation,
) -> pd.DataFrame:
    """Fit final model, write predictions CSV, summary JSON, and submission files."""
    result = fit_final_model_and_predict(
        X_train=x_stage2,
        y_train=y_train,
        X_test=x_test,
        selected_features=winner.features,
        estimator_factory=tuned_factory,
        n_targets=optimal_k,
    )
    frame = pd.DataFrame({
        "rank": np.arange(1, len(result.ranked_test_indices) + 1),
        "sample_index": result.ranked_test_indices,
        "probability": result.probabilities[result.ranked_test_indices],
    })
    frame.to_csv(ctx.outputs_path / "model_predictions.csv", index=False)

    summary = {
        "approach": ctx.approach,
        "best_model": winner.model_name,
        "best_feature_set": winner.feature_set_name,
        "features": winner.features,
        "cv_business_mean": cv_business_mean,
        "hpo_cv_business_mean": hpo_cv,
        "oof_profit_optimal_k": oof_eval.profit_k,
        "oof_breakeven_k": oof_eval.breakeven_k,
        "oof_optimal_k": oof_eval.optimal_k,
        "oof_business_score": float(oof_eval.business_curve.best_score),
        "test_targets": len(result.ranked_test_indices),
    }
    with (ctx.outputs_path / "modeling_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    write_submission_files(
        ctx.outputs_path,
        ctx.submission_prefix,
        result.ranked_test_indices,
        winner.features,
    )
    return frame


def export_ev_test_predictions(
    ctx: ModelingNotebookContext,
    winner: WinnerSelection,
    x_stage2: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    tuned_factory: Any,
    oof_eval: EvOofEvaluation,
    cv_business_mean: float,
    hpo_cv: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit, rank test set with EV k, persist artifacts and submission files."""
    targeting = oof_eval.targeting
    result = fit_final_model_and_predict(
        X_train=x_stage2,
        y_train=y_train,
        X_test=x_test,
        selected_features=winner.features,
        estimator_factory=tuned_factory,
        n_targets=targeting.selected_k,
    )
    test_indices, test_diag = rank_test_indices(
        result.probabilities,
        targeting.selected_k,
        min_prob=targeting.break_even_probability,
        max_targets=ctx.max_targets,
    )

    frame = pd.DataFrame({
        "rank": np.arange(1, len(test_indices) + 1),
        "sample_index": test_indices,
        "probability": result.probabilities[test_indices],
        "expected_value": expected_value_per_contact(result.probabilities[test_indices]),
    })
    frame.to_csv(ctx.outputs_path / "model_predictions.csv", index=False)

    summary = {
        "approach": ctx.approach,
        "best_model": winner.model_name,
        "best_feature_set": winner.feature_set_name,
        "features": winner.features,
        "cv_business_mean": cv_business_mean,
        "hpo_cv_business_mean": hpo_cv,
        "targeting_strategy": targeting.strategy,
        "oof_selected_k": targeting.selected_k,
        "k_ev_threshold": targeting.k_ev_threshold,
        "k_elbow": targeting.k_elbow,
        "k_nested_cv": targeting.k_nested_cv,
        "k_profit_curve_max": targeting.k_profit_curve_max,
        "oof_business_score": targeting.oof_business_score,
        "oof_tp": targeting.oof_tp,
        "oof_fp": targeting.oof_fp,
        "oof_n_above_break_even": targeting.n_above_break_even,
        "test_exported_k": len(test_indices),
        "test_diagnostics": test_diag,
        "calibrated": targeting.calibrated,
    }
    with (ctx.outputs_path / "modeling_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    write_submission_files(
        ctx.outputs_path,
        ctx.submission_prefix,
        test_indices,
        winner.features,
    )
    return frame, test_diag


def run_rank_fusion_pipeline(
    ctx: ModelingNotebookContext,
    expert_configs: tuple[FusionExpertConfig, ...] | None = None,
) -> FusionResult:
    """Build committee OOF probabilities on Stage 2 matrix."""
    _, y_train, _, x_stage2 = stage_matrices(ctx)
    return run_fusion_committee(
        x_stage2,
        y_train,
        ctx.stage.feature_set_candidates,
        expert_configs=expert_configs,
        cv=MODELING_CV_FOLDS,
    )


def export_fusion_test_predictions(
    ctx: ModelingNotebookContext,
    fusion: FusionResult,
    x_stage2: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    oof_eval: TopkOofEvaluation,
) -> pd.DataFrame:
    """Fused test ranking, expert summary CSV, modeling summary JSON, submissions."""
    test_prob = predict_fusion_test_probabilities(x_stage2, y_train, x_test, fusion)
    k = oof_eval.optimal_k
    ranking = np.argsort(test_prob)[::-1]
    test_indices = ranking[: min(k, len(ranking))]

    frame = pd.DataFrame({
        "rank": np.arange(1, len(test_indices) + 1),
        "sample_index": test_indices,
        "probability": test_prob[test_indices],
    })
    frame.to_csv(ctx.outputs_path / "model_predictions.csv", index=False)
    experts_summary_frame(fusion).to_csv(
        ctx.outputs_path / "fusion_experts.csv",
        index=False,
    )

    summary = {
        "approach": ctx.approach,
        "submission_features": list(fusion.submission_features),
        "feature_count": fusion.feature_count,
        "fusion_weights": fusion.weights,
        "experts": [
            {
                "name": e.config.name,
                "model": e.config.model_name,
                "feature_set": e.config.feature_set_name,
                "n_features": len(e.features),
                "cv_business_mean": e.cv_business_mean,
                "weight": e.weight,
            }
            for e in fusion.experts
        ],
        "oof_optimal_k": oof_eval.optimal_k,
        "oof_business_score": float(oof_eval.business_curve.best_score),
        "test_targets": len(test_indices),
    }
    with (ctx.outputs_path / "modeling_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    write_submission_files(
        ctx.outputs_path,
        ctx.submission_prefix,
        test_indices,
        list(fusion.submission_features),
    )
    return frame


def run_segment_targeting_pipeline(
    ctx: ModelingNotebookContext,
    config: SegmentConfig,
) -> SegmentTargetingResult:
    """Segment-aware OOF probabilities."""
    _, y_train, _, x_stage2 = stage_matrices(ctx)
    candidates = ctx.stage.feature_set_candidates
    segment_features = candidates[config.segment_feature_set]
    model_features = candidates[config.model_feature_set]
    return compute_segment_oof_probabilities(
        x_stage2,
        y_train,
        segment_features,
        model_features,
        config,
    )


def export_segment_test_predictions(
    ctx: ModelingNotebookContext,
    segment_result: SegmentTargetingResult,
    x_stage2: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    oof_eval: TopkOofEvaluation,
    config: SegmentConfig,
) -> pd.DataFrame:
    """Test export for segment-aware approach."""
    candidates = ctx.stage.feature_set_candidates
    test_prob = predict_segment_test_probabilities(
        x_stage2,
        y_train,
        x_test,
        candidates[config.segment_feature_set],
        candidates[config.model_feature_set],
        config,
    )
    k = oof_eval.optimal_k
    ranking = np.argsort(test_prob)[::-1]
    test_indices = ranking[: min(k, len(ranking))]

    frame = pd.DataFrame({
        "rank": np.arange(1, len(test_indices) + 1),
        "sample_index": test_indices,
        "probability": test_prob[test_indices],
    })
    frame.to_csv(ctx.outputs_path / "model_predictions.csv", index=False)

    summary = {
        "approach": ctx.approach,
        "segment_feature_set": config.segment_feature_set,
        "model_feature_set": config.model_feature_set,
        "submission_features": list(segment_result.model_features),
        "feature_count": segment_result.feature_count,
        "n_clusters": segment_result.n_clusters,
        "cluster_counts_train": segment_result.cluster_counts_train,
        "oof_optimal_k": oof_eval.optimal_k,
        "oof_business_score": float(oof_eval.business_curve.best_score),
        "test_targets": len(test_indices),
    }
    with (ctx.outputs_path / "modeling_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    write_submission_files(
        ctx.outputs_path,
        ctx.submission_prefix,
        test_indices,
        list(segment_result.model_features),
    )
    return frame


def cluster_split_feature_sets(
    ctx: ModelingNotebookContext,
    feature_set_candidates: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Top-k subsets from ``feature_selection_results.csv`` (Stage 3 ranking)."""
    return feature_set_candidates or feature_set_candidates_from_selection_results(
        ctx.feature_selection_outputs
    )


def resolve_cluster_n_clusters(
    feature_set: str,
    n_clusters_by_feature_set: dict[str, int | None],
    *,
    default_n_clusters: int | None = None,
) -> int:
    """Resolve k from per-set map (notebook placeholders) or optional global default."""
    if feature_set in n_clusters_by_feature_set:
        k = n_clusters_by_feature_set[feature_set]
        if k is not None:
            return int(k)
    if default_n_clusters is not None:
        return int(default_n_clusters)
    msg = (
        f"Set N_CLUSTERS_BY_FEATURE_SET[{feature_set!r}] after elbow/silhouette diagnostics "
        "(or set DEFAULT_N_CLUSTERS)."
    )
    raise ValueError(msg)


def analyze_cluster_k_for_feature_sets(
    ctx: ModelingNotebookContext,
    feature_set_candidates: dict[str, list[str]] | None = None,
    k_values: range | list[int] | None = None,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elbow + silhouette for each top-k name; write ``cluster_k_*.csv`` under approach outputs."""
    candidates = cluster_split_feature_sets(ctx, feature_set_candidates)
    _, _, _, x_stage2 = stage_matrices(ctx)
    k_range = range(2, 11) if k_values is None else k_values

    detail_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for name in sorted(candidates):
        features = candidates[name]
        diag = kmeans_k_diagnostics(
            x_stage2,
            features,
            k_range,
            random_state=random_state,
        )
        diag = diag.assign(feature_set=name, n_features=len(features))
        detail_parts.append(diag)
        summary_rows.append({
            "feature_set": name,
            "n_features": len(features),
            "k_min": int(diag["k"].min()) if not diag.empty else None,
            "k_max": int(diag["k"].max()) if not diag.empty else None,
        })

    details = pd.concat(detail_parts, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)
    details.to_csv(ctx.outputs_path / "cluster_k_details.csv", index=False)
    summary_frame.to_csv(ctx.outputs_path / "cluster_k_summary.csv", index=False)
    return summary_frame, details


def analyze_cluster_y_distributions(
    ctx: ModelingNotebookContext,
    feature_set_candidates: dict[str, list[str]] | None = None,
    n_clusters_by_feature_set: dict[str, int | None] | None = None,
    *,
    default_n_clusters: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train KMeans at chosen k; profile y by cluster for each configured feature set."""
    candidates = cluster_split_feature_sets(ctx, feature_set_candidates)
    _, y_train, _, x_stage2 = stage_matrices(ctx)
    k_map = n_clusters_by_feature_set or {}

    parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for name in sorted(candidates):
        try:
            n_clusters = resolve_cluster_n_clusters(
                name,
                k_map,
                default_n_clusters=default_n_clusters,
            )
        except ValueError:
            continue

        features = candidates[name]
        labels = fit_train_cluster_labels(
            x_stage2,
            features,
            n_clusters,
            random_state=random_state,
        )
        profile = build_cluster_y_profile(y_train, labels)
        profile = profile.assign(feature_set=name, n_clusters=n_clusters)
        parts.append(profile)

        sizes = profile["n"]
        summary_rows.append({
            "feature_set": name,
            "n_clusters": n_clusters,
            "global_positive_rate": float(profile["global_positive_rate"].iloc[0]),
            "min_cluster_n": int(sizes.min()),
            "max_cluster_n": int(sizes.max()),
            "size_imbalance": float(sizes.max() / sizes.min()) if sizes.min() else float("inf"),
            "max_positive_rate": float(profile["positive_rate"].max()),
            "min_positive_rate": float(profile["positive_rate"].min()),
            "max_lift": float(profile["lift_vs_global"].max()),
        })

    if not parts:
        msg = "No feature sets with k set — fill N_CLUSTERS_BY_FEATURE_SET first"
        raise ValueError(msg)

    long_frame = pd.concat(parts, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)
    long_frame.to_csv(ctx.outputs_path / "cluster_y_profiles.csv", index=False)
    summary_frame.to_csv(ctx.outputs_path / "cluster_y_summary.csv", index=False)
    return long_frame


def compare_cluster_split_feature_sets(
    ctx: ModelingNotebookContext,
    feature_set_candidates: dict[str, list[str]] | None = None,
    *,
    base_config: ClusterSplitConfig | None = None,
    n_clusters_by_feature_set: dict[str, int | None] | None = None,
    default_n_clusters: int | None = None,
) -> tuple[pd.DataFrame, ClusterSplitConfig, ClusterSplitTargetingResult, TopkOofEvaluation]:
    """Run cluster-split OOF for every top-k name; save CSV; return best run."""
    candidates = cluster_split_feature_sets(ctx, feature_set_candidates)
    _, y_train, _, _ = stage_matrices(ctx)
    base = base_config or ClusterSplitConfig(n_clusters=2)
    k_map = n_clusters_by_feature_set or {}

    rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_config = base
    best_result: ClusterSplitTargetingResult | None = None
    best_eval: TopkOofEvaluation | None = None

    for name in sorted(candidates):
        n_clusters = resolve_cluster_n_clusters(
            name,
            k_map,
            default_n_clusters=default_n_clusters,
        )
        config = ClusterSplitConfig(
            feature_set=name,
            n_clusters=n_clusters,
            min_cluster_samples=base.min_cluster_samples,
            max_targets=base.max_targets,
            cv_folds=base.cv_folds,
            random_state=base.random_state,
        )
        result = run_cluster_split_pipeline(ctx, config, candidates)
        oof_eval = evaluate_topk_oof(
            y_train,
            result.oof_probabilities,
            result.feature_count,
            ctx.max_targets,
        )
        score = float(oof_eval.business_curve.best_score)
        rows.append({
            "feature_set": name,
            "n_features": result.feature_count,
            "n_clusters": n_clusters,
            "oof_k": oof_eval.optimal_k,
            "oof_business_score": score,
            "cluster_counts_train": str(result.cluster_counts_train),
        })
        if score > best_score:
            best_score = score
            best_config = config
            best_result = result
            best_eval = oof_eval

    if best_result is None or best_eval is None:
        msg = "No feature sets to compare"
        raise ValueError(msg)

    frame = (
        pd.DataFrame(rows)
        .sort_values(["oof_business_score", "n_features"], ascending=[False, True])
        .reset_index(drop=True)
    )
    frame.to_csv(ctx.outputs_path / "feature_set_comparison.csv", index=False)
    return frame, best_config, best_result, best_eval


def run_cluster_split_pipeline(
    ctx: ModelingNotebookContext,
    config: ClusterSplitConfig,
    feature_set_candidates: dict[str, list[str]] | None = None,
) -> ClusterSplitTargetingResult:
    """Two-cluster KMeans split with per-cluster feature sets."""
    _, y_train, _, x_stage2 = stage_matrices(ctx)
    candidates = cluster_split_feature_sets(ctx, feature_set_candidates)
    features = candidates[config.feature_set]
    return compute_cluster_split_oof_probabilities(
        x_stage2,
        y_train,
        features,
        config,
    )


def export_cluster_split_test_predictions(
    ctx: ModelingNotebookContext,
    split_result: ClusterSplitTargetingResult,
    x_stage2: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    oof_eval: TopkOofEvaluation,
    config: ClusterSplitConfig,
    feature_set_candidates: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Test export for two-cluster split approach."""
    candidates = cluster_split_feature_sets(ctx, feature_set_candidates)
    features = candidates[config.feature_set]
    test_prob = predict_cluster_split_test_probabilities(
        x_stage2,
        y_train,
        x_test,
        features,
        config,
    )
    k = oof_eval.optimal_k
    ranking = np.argsort(test_prob)[::-1]
    test_indices = ranking[: min(k, len(ranking))]

    frame = pd.DataFrame({
        "rank": np.arange(1, len(test_indices) + 1),
        "sample_index": test_indices,
        "probability": test_prob[test_indices],
    })
    frame.to_csv(ctx.outputs_path / "model_predictions.csv", index=False)

    summary = {
        "approach": ctx.approach,
        "feature_set": config.feature_set,
        "submission_features": list(split_result.features),
        "feature_count": split_result.feature_count,
        "n_clusters": split_result.n_clusters,
        "cluster_counts_train": split_result.cluster_counts_train,
        "oof_optimal_k": oof_eval.optimal_k,
        "oof_business_score": float(oof_eval.business_curve.best_score),
        "test_targets": len(test_indices),
    }
    with (ctx.outputs_path / "modeling_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    write_submission_files(
        ctx.outputs_path,
        ctx.submission_prefix,
        test_indices,
        list(split_result.features),
    )
    return frame
