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
    TargetingConfig,
    attach_best_hyperparams,
    build_f1_curve,
    build_model_factories,
    build_profit_curve,
    choose_targeting_k,
    compare_models_on_feature_sets,
    fit_final_model_and_predict,
    rank_test_indices,
)
from .models.dataclasses import F1CurveResult, ProfitCurveResult
from .models.profit_targeting import TargetingSelectionResult, expected_value_per_contact
from .notebook_setup import ModelingNotebookContext
from .utils import (
    MODELING_CV_FOLDS,
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
