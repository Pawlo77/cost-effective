import json

import pandas as pd

from cost_effective.models.nested_cv_outer_evaluation import (
    aggregate_outer_scores,
    build_rank_mean_ensemble,
    complete_outer_prediction_pairs,
    filter_outer_recipe_pairs,
    normalize_pipeline_recipes,
    refit_final_configurations,
    rescore_outer_predictions_at_targets,
    run_outer_fold_evaluation,
    score_topk_predictions,
    select_cost_aware_configurations,
    select_global_stage02_pipeline_recipes,
    select_stable_configurations,
)


def _toy_data() -> tuple[pd.DataFrame, pd.Series]:
    x = pd.DataFrame({
        "var_1": [0.0, 0.1, 0.9, 1.0, 0.2, 0.3, 0.8, 1.1],
        "var_2": [1.0, 0.8, 0.1, 0.0, 0.7, 0.6, 0.2, 0.1],
    })
    y = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    return x, y


def _prescreen_recipes() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "prescreen_recipe_id": "ps_abs",
            "prescreen_name": "single__abs_corr",
            "prescreen_methods": json.dumps(["abs_corr"]),
            "prescreen_method_count": 1,
            "rank_aggregation": "mean_rank_score",
            "adaptive": False,
            "adaptive_method_count": None,
            "leakage_rule": "fit ranking only on the active training partition",
        }
    ])


def _selected_recipes() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "outer_fold": 1,
            "pipeline_recipe_id": "ps_abs__k_001__logistic_smoke",
            "feature_recipe_id": "ps_abs__k_001",
            "prescreen_recipe_id": "ps_abs",
            "prescreen_name": "single__abs_corr",
            "prescreen_methods": json.dumps(["abs_corr"]),
            "rank_aggregation": "mean_rank_score",
            "feature_size": 1,
            "model_spec_id": "logistic_smoke",
            "base_model_family": "logistic_baseline",
            "model_kind": "classifier",
            "model_params": json.dumps({
                "solver": "saga",
                "C": 1.0,
                "l1_ratio": 0.0,
                "class_weight": None,
                "max_iter": 500,
            }),
            "inner_selection_rank": 1,
            "inner_oof_business_score": 10.0,
        }
    ])


def test_score_topk_predictions_allows_zero_contacts() -> None:
    scored = score_topk_predictions([0, 0], [0.9, 0.8], feature_count=1, max_targets=2)

    assert scored["outer_optimal_k"] == 0
    assert scored["outer_business_score"] == -200.0
    assert scored["outer_tp_at_optimal_k"] == 0
    assert scored["outer_fp_at_optimal_k"] == 0


def test_run_outer_fold_evaluation_refits_prescreen_on_outer_train() -> None:
    x, y = _toy_data()
    outer_assignments = pd.DataFrame({
        "sample_index": range(len(y)),
        "outer_fold": [1, 1, 2, 2, 2, 2, 2, 2],
    })

    scores, predictions, selected_features = run_outer_fold_evaluation(
        x,
        y,
        _selected_recipes(),
        outer_assignments,
        _prescreen_recipes(),
        random_state=7,
        max_targets=2,
    )

    assert scores["status"].tolist() == ["ok"]
    assert predictions["sample_index"].tolist() == [0, 1]
    assert len(selected_features) == 1
    assert selected_features.iloc[0]["feature"] in {"var_1", "var_2"}
    assert scores.iloc[0]["selected_features_text"] == selected_features.iloc[0]["feature"]


def test_aggregate_and_select_stable_configurations_rank_by_outer_score() -> None:
    scores = pd.DataFrame({
        "status": ["ok", "ok", "ok"],
        "outer_fold": [1, 2, 1],
        "recipe_config_id": ["a", "a", "b"],
        "prescreen_recipe_id": ["p", "p", "p"],
        "prescreen_name": ["p", "p", "p"],
        "feature_size": [1, 1, 2],
        "model_spec_id": ["m", "m", "m"],
        "base_model_family": ["fam", "fam", "fam"],
        "model_kind": ["classifier", "classifier", "classifier"],
        "model_params": ["{}", "{}", "{}"],
        "outer_business_score": [10.0, 20.0, 100.0],
        "outer_optimal_k": [1, 1, 2],
        "outer_f1_score": [0.1, 0.2, 0.3],
        "outer_roc_auc_score": [0.5, 0.6, 0.7],
        "inner_oof_business_score": [1.0, 1.0, 2.0],
        "inner_selection_rank": [1, 1, 2],
        "selected_feature_count": [1, 1, 2],
        "selected_features_text": ["var_1", "var_1", "var_1,var_2"],
    })

    summary = aggregate_outer_scores(scores)
    stable = select_stable_configurations(summary, top_n=1, min_outer_folds=2)

    assert summary.iloc[0]["recipe_config_id"] == "b"
    assert stable.iloc[0]["recipe_config_id"] == "a"
    assert stable.iloc[0]["outer_fold_count"] == 2


def test_rate_matched_rescore_and_cost_aware_selection_use_saved_predictions() -> None:
    predictions = pd.DataFrame({
        "outer_fold": [1, 1, 1, 1, 1, 1],
        "recipe_config_id": ["cheap"] * 3 + ["expensive"] * 3,
        "pipeline_recipe_id": ["cheap"] * 3 + ["expensive"] * 3,
        "prescreen_recipe_id": ["p"] * 6,
        "prescreen_name": ["p"] * 6,
        "feature_size": [1] * 3 + [3] * 3,
        "model_spec_id": ["m"] * 6,
        "base_model_family": ["fam"] * 6,
        "model_kind": ["classifier"] * 6,
        "model_params": ["{}"] * 6,
        "sample_index": [0, 1, 2, 0, 1, 2],
        "y_true": [1, 1, 0, 1, 1, 0],
        "score": [0.9, 0.8, 0.1, 0.95, 0.85, 0.2],
        "status": ["ok"] * 6,
        "error": [""] * 6,
    })
    original_scores = pd.DataFrame({
        "outer_fold": [1, 1],
        "recipe_config_id": ["cheap", "expensive"],
        "selected_features_text": ["var_1", "var_1,var_2,var_3"],
        "selected_feature_count": [1, 3],
    })

    rescored = rescore_outer_predictions_at_targets(
        predictions,
        max_targets_by_outer_fold={1: 2},
        gross_score_scale_by_outer_fold={1: 3.0},
        original_scores=original_scores,
    )
    summary = aggregate_outer_scores(rescored)
    selected = select_cost_aware_configurations(
        summary,
        top_n=1,
        min_outer_folds=1,
        max_feature_size=1,
    )

    assert rescored.set_index("recipe_config_id").loc["cheap", "outer_optimal_k"] == 2
    assert rescored.set_index("recipe_config_id").loc["cheap", "outer_business_score"] == -140.0
    assert (
        rescored.set_index("recipe_config_id").loc[
            "cheap", "outer_business_score_unscaled_rate_fold"
        ]
        == -180.0
    )
    assert selected.iloc[0]["recipe_config_id"] == "cheap"
    assert selected.iloc[0]["feature_size"] == 1


def test_global_stage02_candidates_expand_to_all_outer_folds(tmp_path) -> None:
    stage_two_dir = tmp_path / "stage02"
    stage_two_dir.mkdir()
    rows = []
    for outer_fold, score in [(1, 100.0), (2, 90.0)]:
        rows.append({
            "outer_fold": outer_fold,
            "round_name": "round2_model_hpo_selection",
            "pipeline_recipe_id": "ps_abs__k_001__m",
            "feature_recipe_id": "ps_abs__k_001",
            "prescreen_recipe_id": "ps_abs",
            "prescreen_name": "single__abs_corr",
            "prescreen_methods": json.dumps(["abs_corr"]),
            "rank_aggregation": "mean_rank_score",
            "feature_size": 1,
            "model_spec_id": "m",
            "base_model_family": "fam",
            "model_kind": "classifier",
            "model_params": "{}",
            "inner_oof_business_score": score,
            "inner_oof_optimal_k": 2,
            "inner_oof_f1_score": 0.5,
            "inner_oof_roc_auc_score": 0.7,
            "inner_fold_count": 5,
            "missing_inner_oof_predictions": 0,
            "missing_inner_fold_count": 0,
            "status": "ok",
        })
    pd.DataFrame(rows).to_csv(stage_two_dir / "round2_inner_oof_scores.csv", index=False)

    selected = select_global_stage02_pipeline_recipes(
        stage_two_dir,
        outer_folds=(1, 2, 3),
        top_n=1,
        min_inner_fold_count=5,
        max_feature_size=2,
    )

    assert selected["outer_fold"].tolist() == [1, 2, 3]
    assert selected["recipe_config_id"].nunique() == 1
    assert selected["stage02_outer_fold_count"].tolist() == [2, 2, 2]
    assert selected["inner_selection_rank"].tolist() == [1, 1, 1]


def test_complete_outer_pairs_and_exact_finalist_filter_reject_incomplete_configs() -> None:
    outer_assignments = pd.DataFrame({
        "outer_fold": [1, 1, 2, 2],
        "sample_index": [0, 1, 2, 3],
    })
    scores = pd.DataFrame({
        "outer_fold": [1, 2],
        "recipe_config_id": ["complete", "complete"],
        "status": ["ok", "ok"],
    })
    predictions = pd.DataFrame({
        "outer_fold": [1, 1, 2],
        "recipe_config_id": ["complete", "complete", "complete"],
        "sample_index": [0, 1, 2],
        "status": ["ok", "ok", "ok"],
    })

    complete_pairs = complete_outer_prediction_pairs(scores, predictions, outer_assignments)
    filtered = filter_outer_recipe_pairs(scores, complete_pairs)

    assert complete_pairs == {(1, "complete")}
    assert filtered["outer_fold"].tolist() == [1]

    outer_summary = pd.DataFrame({
        "recipe_config_id": ["complete", "partial"],
        "outer_fold_count": [5, 4],
        "mean_outer_business_score": [10.0, 100.0],
        "min_outer_business_score": [5.0, 100.0],
        "feature_size": [1, 1],
    })
    selected = select_cost_aware_configurations(
        outer_summary,
        top_n=2,
        min_outer_folds=5,
        max_feature_size=1,
        fallback_min_outer_folds=5,
        require_exact_outer_folds=True,
        allow_incomplete_fallback=False,
    )

    assert selected["recipe_config_id"].tolist() == ["complete"]
    assert selected["outer_fold_count"].tolist() == [5]

    no_complete_selected = select_cost_aware_configurations(
        outer_summary.loc[outer_summary["recipe_config_id"].eq("partial")],
        top_n=2,
        min_outer_folds=5,
        max_feature_size=1,
        fallback_min_outer_folds=5,
        require_exact_outer_folds=True,
        allow_incomplete_fallback=False,
    )

    assert no_complete_selected.empty
    assert "recipe_config_id" in no_complete_selected.columns
    assert "finalist_rank" in no_complete_selected.columns


def test_final_refit_and_rank_mean_ensemble_use_notebook_three_finalists() -> None:
    x, y = _toy_data()
    x_test = x.copy()
    finalists = normalize_pipeline_recipes(_selected_recipes()).assign(
        finalist_rank=1,
        mean_outer_business_score=10.0,
        median_outer_business_score=10.0,
        outer_fold_count=1,
    )

    manifest, predictions, features = refit_final_configurations(
        x,
        y,
        x_test,
        finalists,
        _prescreen_recipes(),
        random_state=7,
        top_n=3,
    )

    assert manifest["status"].tolist() == ["ok"]
    assert predictions["sample_index"].nunique() == len(x_test)
    assert int(predictions["selected_for_top_n"].sum()) == 3
    assert features["feature"].tolist() == ["var_1"]

    doubled_manifest = pd.concat(
        [
            manifest,
            manifest.assign(recipe_config_id="second_config", finalist_rank=2),
        ],
        ignore_index=True,
    )
    doubled_predictions = pd.concat(
        [
            predictions,
            predictions.assign(recipe_config_id="second_config", finalist_rank=2),
        ],
        ignore_index=True,
    )
    ensemble_manifest, ensemble_ranking = build_rank_mean_ensemble(
        doubled_predictions,
        doubled_manifest,
        top_n=3,
    )

    assert ensemble_manifest.iloc[0]["member_count"] == 2
    assert len(ensemble_ranking) == 3
