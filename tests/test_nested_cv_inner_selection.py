import json

import numpy as np
import pandas as pd
import pytest

from cost_effective.models.nested_cv_inner_selection import (
    aggregate_inner_scores,
    build_fold_feature_candidates,
    evaluate_feature_candidates,
    evaluate_feature_candidates_oof_blocks,
    evaluate_feature_candidates_predictions,
    feature_cost_scale,
    filter_model_specs,
    filter_prescreen_recipes,
    has_inner_prediction_columns,
    inner_fold_indices,
    materialize_prescreen_ranking,
    outer_fold_indices,
    parse_json_dict,
    parse_json_list,
    restrict_feature_candidates_to_selected_pairs,
    score_inner_oof_baselines,
    score_inner_oof_prediction_blocks,
    score_inner_oof_predictions,
    score_inner_oof_predictions_csv,
    score_validation_predictions_scaled,
    train_only_prescreen_weights,
)


def _toy_rankings() -> pd.DataFrame:
    return pd.DataFrame({
        "feature": ["var_1", "var_2", "var_3", "var_1", "var_2", "var_3"],
        "method": ["m1", "m1", "m1", "m2", "m2", "m2"],
        "raw_score": [0.9, 0.7, 0.1, 0.2, 0.8, 0.4],
        "rank": [1.0, 2.0, 3.0, 3.0, 1.0, 2.0],
        "rank_score": [1.0, 0.5, 0.0, 0.0, 1.0, 0.5],
    })


def test_outer_and_inner_indices_do_not_overlap_for_same_outer_fold() -> None:
    outer = pd.DataFrame({
        "sample_index": np.arange(10),
        "outer_fold": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    })
    inner = pd.DataFrame({
        "outer_fold": [1] * 8,
        "sample_index": [2, 3, 4, 5, 6, 7, 8, 9],
        "inner_fold": [1, 1, 2, 2, 3, 3, 4, 4],
    })

    outer_train, outer_val = outer_fold_indices(outer, 1)
    inner_train, inner_val = inner_fold_indices(inner, 1, 1)

    assert set(outer_val) == {0, 1}
    assert set(outer_train) == {2, 3, 4, 5, 6, 7, 8, 9}
    assert set(inner_val) == {2, 3}
    assert set(inner_train) == {4, 5, 6, 7, 8, 9}
    assert set(outer_val).isdisjoint(inner_train)
    assert set(outer_val).isdisjoint(inner_val)


def test_json_parsers_accept_json_and_python_feature_literals() -> None:
    assert parse_json_list('["a", "b"]') == ["a", "b"]
    assert parse_json_list("['var_1', 'var_2']") == ["var_1", "var_2"]
    assert parse_json_dict('{"C": 0.5}') == {"C": 0.5}


def test_materialize_mean_and_weighted_prescreen_ranking() -> None:
    rankings = _toy_rankings()
    mean_recipe = pd.Series({
        "prescreen_recipe_id": "ps1",
        "prescreen_name": "pair__m1+m2",
        "prescreen_methods": json.dumps(["m1", "m2"]),
        "rank_aggregation": "mean_rank_score",
    })
    weighted_recipe = mean_recipe.copy()
    weighted_recipe["prescreen_name"] = "weighted"
    weighted_recipe["rank_aggregation"] = "train_only_single_score_weighted"

    mean_ranking = materialize_prescreen_ranking(rankings, mean_recipe)
    weights = train_only_prescreen_weights(rankings, ["m1", "m2"])
    weighted_ranking = materialize_prescreen_ranking(rankings, weighted_recipe)

    assert mean_ranking is not None
    assert weighted_ranking is not None
    assert set(weights) == {"m1", "m2"}
    assert mean_ranking["order"].is_monotonic_increasing
    assert weighted_ranking["order"].is_monotonic_increasing


def test_build_fold_feature_candidates_uses_local_rankings_and_caps() -> None:
    rankings = _toy_rankings()
    recipes = pd.DataFrame([
        {
            "prescreen_recipe_id": "ps1",
            "prescreen_name": "single__m1",
            "prescreen_methods": json.dumps(["m1"]),
            "rank_aggregation": "mean_rank_score",
        },
        {
            "prescreen_recipe_id": "ps2",
            "prescreen_name": "single__missing",
            "prescreen_methods": json.dumps(["missing"]),
            "rank_aggregation": "mean_rank_score",
        },
    ])

    candidates, failures = build_fold_feature_candidates(
        rankings,
        recipes,
        feature_sizes=(1, 2),
        max_feature_candidates=1,
    )

    assert len(candidates) == 1
    assert candidates.iloc[0]["selected_features"] == ["var_1"]
    assert failures.iloc[0]["prescreen_recipe_id"] == "ps2"


def test_aggregate_inner_scores_ranks_by_mean_then_median() -> None:
    scores = pd.DataFrame({
        "status": ["ok", "ok", "ok", "error"],
        "outer_fold": [1, 1, 1, 1],
        "inner_fold": [1, 2, 1, 2],
        "prescreen_recipe_id": ["a", "a", "b", "b"],
        "feature_size": [3, 3, 3, 3],
        "model_spec_id": ["m", "m", "m", "m"],
        "inner_business_score": [10.0, 20.0, 25.0, np.nan],
        "inner_optimal_k": [5, 6, 7, np.nan],
        "selected_features_text": ["x", "x", "y", "y"],
        "prescreen_name": ["a", "a", "b", "b"],
        "base_model_family": ["fam", "fam", "fam", "fam"],
    })

    aggregated = aggregate_inner_scores(
        scores,
        group_cols=("outer_fold", "prescreen_recipe_id", "feature_size", "model_spec_id"),
    )

    assert aggregated.iloc[0]["prescreen_recipe_id"] == "b"
    assert aggregated.iloc[1]["prescreen_recipe_id"] == "a"


def test_filter_helpers_keep_requested_rows() -> None:
    models = pd.DataFrame({"model_spec_id": ["a", "b", "c"]})
    prescreens = pd.DataFrame({
        "prescreen_recipe_id": ["p1", "p2"],
        "adaptive": [False, True],
    })

    assert filter_model_specs(models, ["b"])["model_spec_id"].tolist() == ["b"]
    assert filter_model_specs(models, max_specs=2)["model_spec_id"].tolist() == ["a", "b"]
    assert filter_prescreen_recipes(prescreens, include_adaptive=False)[
        "prescreen_recipe_id"
    ].tolist() == ["p1"]


def test_restrict_feature_candidates_to_selected_pairs_keeps_exact_pairs() -> None:
    candidates = pd.DataFrame({
        "prescreen_recipe_id": ["p1", "p1", "p2"],
        "feature_size": [3, 5, 3],
        "value": [1, 2, 3],
    })
    selected = pd.DataFrame({
        "prescreen_recipe_id": ["p1", "p2"],
        "feature_size": [5, 3],
    })

    restricted = restrict_feature_candidates_to_selected_pairs(candidates, selected)

    assert restricted["value"].tolist() == [2, 3]


def test_scaled_validation_score_scales_feature_cost_by_eval_rows() -> None:
    y = pd.Series([1, 1, 0, 0, 0])
    scores = np.array([0.95, 0.90, 0.40, 0.30, 0.20])

    scored = score_validation_predictions_scaled(
        y,
        scores,
        ["var_1", "var_2"],
        method="toy",
        max_targets=3,
        feature_cost_reference_row_count=10,
    )

    assert feature_cost_scale(5, reference_row_count=10) == 0.5
    assert scored["feature_cost_scale"] == 0.5
    assert scored["scaled_true_positive_value"] == 5.0
    assert scored["scaled_false_positive_cost"] == 2.5
    assert scored["scaled_feature_cost"] == 200.0
    assert scored["unscaled_feature_cost"] == 400.0
    assert scored["oof_optimal_k"] == 2
    assert scored["oof_tp_at_optimal_k"] == 2
    assert scored["oof_fp_at_optimal_k"] == 0
    assert scored["oof_business_value_before_feature_cost"] == 10.0
    assert scored["oof_business_score"] == -190.0
    assert scored["oof_business_score_unscaled_feature_cost"] == -390.0


def test_evaluate_feature_candidates_parallel_matches_sequential() -> None:
    x = pd.DataFrame({
        "var_1": [0.0, 0.2, 0.8, 1.0, 1.1, 1.3],
        "var_2": [1.0, 0.9, 0.3, 0.1, 0.0, -0.2],
    })
    y = pd.Series([0, 0, 1, 1, 1, 0])
    train_idx = [0, 1, 2, 3]
    val_idx = [4, 5]
    feature_candidates = pd.DataFrame({
        "feature_recipe_id": ["fr1"],
        "prescreen_recipe_id": ["ps1"],
        "prescreen_name": ["single__toy"],
        "prescreen_methods": [json.dumps(["toy"])],
        "rank_aggregation": ["mean_rank_score"],
        "feature_size": [1],
        "selected_features": [["var_1"]],
        "selected_features_text": ["var_1"],
        "candidate_heuristic_score": [1.0],
    })
    model_specs = pd.DataFrame({
        "model_spec_id": ["logistic_smoke"],
        "base_model_family": ["logistic_baseline"],
        "model_kind": ["classifier"],
        "model_params": [
            json.dumps({
                "solver": "saga",
                "C": 1.0,
                "l1_ratio": 0.0,
                "class_weight": "balanced",
                "max_iter": 300,
            })
        ],
    })

    kwargs = {
        "X_train": x.iloc[train_idx],
        "y_train": y.iloc[train_idx],
        "X_val": x.iloc[val_idx],
        "y_val": y.iloc[val_idx],
        "feature_candidates": feature_candidates,
        "model_specs": model_specs,
        "outer_fold": 1,
        "inner_fold": 1,
        "round_name": "test",
        "random_state": 42,
        "max_targets": 2,
        "log_every": 0,
    }

    sequential = evaluate_feature_candidates(**kwargs, n_jobs=1)
    parallel = evaluate_feature_candidates(**kwargs, n_jobs=2, parallel_prefer="threads")

    assert sequential["status"].tolist() == ["ok"]
    assert parallel["status"].tolist() == ["ok"]
    assert sequential["inner_business_score"].tolist() == parallel["inner_business_score"].tolist()


def _toy_prediction_rows() -> pd.DataFrame:
    base = {
        "outer_fold": 1,
        "round_name": "round1",
        "feature_recipe_id": "fr1",
        "prescreen_recipe_id": "ps1",
        "prescreen_name": "single__toy",
        "prescreen_methods": json.dumps(["toy"]),
        "rank_aggregation": "mean_rank_score",
        "feature_size": 1,
        "selected_features": ["var_1"],
        "selected_features_text": "var_1",
        "candidate_heuristic_score": 1.0,
        "model_spec_id": "m1",
        "base_model_family": "toy_model",
        "model_kind": "classifier",
        "model_params": "{}",
        "status": "ok",
        "error": "",
        "duration_seconds": 0.01,
    }
    rows = []
    for sample_index, inner_fold, y_true, score in [
        (0, 1, 1, 0.90),
        (1, 1, 0, 0.80),
        (2, 2, 1, 0.70),
        (3, 2, 0, 0.10),
    ]:
        rows.append({
            **base,
            "pipeline_recipe_id": "fr1__m1",
            "inner_fold": inner_fold,
            "sample_index": sample_index,
            "y_true": y_true,
            "score": score,
        })

    better = {**base, "feature_recipe_id": "fr2", "pipeline_recipe_id": "fr2__m1"}
    for sample_index, inner_fold, y_true, score in [
        (0, 1, 1, 0.90),
        (1, 1, 0, 0.20),
        (2, 2, 1, 0.80),
        (3, 2, 0, 0.10),
    ]:
        rows.append({
            **better,
            "inner_fold": inner_fold,
            "sample_index": sample_index,
            "y_true": y_true,
            "score": score,
        })
    return pd.DataFrame(rows)


def test_score_inner_oof_predictions_scores_combined_outer_train_once() -> None:
    predictions = _toy_prediction_rows()

    scored = score_inner_oof_predictions(predictions, max_targets=3)

    assert scored.iloc[0]["pipeline_recipe_id"] == "fr2__m1"
    assert scored.iloc[0]["inner_oof_business_score"] == -180.0
    assert scored.iloc[0]["inner_oof_optimal_k"] == 2
    assert scored.iloc[0]["inner_oof_n_predictions"] == 4
    assert scored.iloc[0]["inner_oof_selected_rate"] == 0.5
    assert scored.iloc[0]["feature_penalty"] == 200.0
    assert scored.iloc[0]["gross_score"] == 20.0
    assert scored.iloc[0]["baseline_select_all_capped_features_0_business_score"] == 15.0
    assert scored.iloc[0]["baseline_select_all_capped_features_1_business_score"] == -185.0
    assert bool(scored.iloc[0]["beats_baseline_select_all_capped_features_1"])


def test_score_inner_oof_predictions_rejects_duplicate_recipe_sample() -> None:
    predictions = _toy_prediction_rows()
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate inner-OOF predictions"):
        score_inner_oof_predictions(duplicated, max_targets=3)


def test_score_inner_oof_baselines_use_same_universe() -> None:
    baselines = score_inner_oof_baselines(
        _toy_prediction_rows(),
        max_targets=3,
        random_repeats=2,
        random_state=123,
    )

    select_all = baselines.loc[baselines["baseline_type"].eq("select_all_capped_no_model")]
    assert set(select_all["baseline_name"]) == {
        "baseline_select_all_capped_features_0",
        "baseline_select_all_capped_features_1",
    }
    assert (
        select_all.set_index("baseline_name").loc[
            "baseline_select_all_capped_features_0", "baseline_business_score"
        ]
        == 15.0
    )
    assert len(baselines.loc[baselines["baseline_type"].eq("random_score_baseline")]) == 2


def test_aggregate_inner_scores_can_rank_by_inner_oof_business_score() -> None:
    scored = score_inner_oof_predictions(_toy_prediction_rows(), max_targets=3)

    aggregated = aggregate_inner_scores(
        scored,
        group_cols=("outer_fold", "round_name", "pipeline_recipe_id"),
    )

    assert aggregated.iloc[0]["pipeline_recipe_id"] == "fr2__m1"
    assert aggregated.iloc[0]["inner_oof_business_score"] == -180.0


def test_evaluate_feature_candidates_predictions_returns_sample_level_rows() -> None:
    x = pd.DataFrame({
        "var_1": [0.0, 0.2, 0.8, 1.0, 1.1, 1.3],
        "var_2": [1.0, 0.9, 0.3, 0.1, 0.0, -0.2],
    })
    y = pd.Series([0, 0, 1, 1, 1, 0])
    train_idx = [0, 1, 2, 3]
    val_idx = [4, 5]
    feature_candidates = pd.DataFrame({
        "feature_recipe_id": ["fr1"],
        "prescreen_recipe_id": ["ps1"],
        "prescreen_name": ["single__toy"],
        "prescreen_methods": [json.dumps(["toy"])],
        "rank_aggregation": ["mean_rank_score"],
        "feature_size": [1],
        "selected_features": [["var_1"]],
        "selected_features_text": ["var_1"],
        "candidate_heuristic_score": [1.0],
    })
    model_specs = pd.DataFrame({
        "model_spec_id": ["logistic_smoke"],
        "base_model_family": ["logistic_baseline"],
        "model_kind": ["classifier"],
        "model_params": [
            json.dumps({
                "solver": "saga",
                "C": 1.0,
                "l1_ratio": 0.0,
                "class_weight": "balanced",
                "max_iter": 300,
            })
        ],
    })

    predictions = evaluate_feature_candidates_predictions(
        x.iloc[train_idx],
        y.iloc[train_idx],
        x.iloc[val_idx],
        y.iloc[val_idx],
        feature_candidates,
        model_specs,
        outer_fold=1,
        inner_fold=1,
        round_name="test",
        random_state=42,
        log_every=0,
        n_jobs=1,
    )

    assert predictions["status"].tolist() == ["ok", "ok"]
    assert predictions["sample_index"].tolist() == val_idx
    assert predictions["y_true"].tolist() == [1, 0]
    assert predictions["score"].notna().all()
    assert "inner_business_score" not in predictions.columns


def test_compact_oof_blocks_match_sample_level_inner_oof_scoring() -> None:
    x = pd.DataFrame({
        "var_1": [0.0, 0.2, 0.8, 1.0, 1.1, 1.3],
        "var_2": [1.0, 0.9, 0.3, 0.1, 0.0, -0.2],
    })
    y = pd.Series([0, 0, 1, 1, 1, 0])
    train_idx = [0, 1, 2, 3]
    val_idx = [4, 5]
    feature_candidates = pd.DataFrame({
        "feature_recipe_id": ["fr1"],
        "prescreen_recipe_id": ["ps1"],
        "prescreen_name": ["single__toy"],
        "prescreen_methods": [json.dumps(["toy"])],
        "rank_aggregation": ["mean_rank_score"],
        "feature_size": [1],
        "selected_features": [["var_1"]],
        "selected_features_text": ["var_1"],
        "candidate_heuristic_score": [1.0],
    })
    model_specs = pd.DataFrame({
        "model_spec_id": ["logistic_smoke"],
        "base_model_family": ["logistic_baseline"],
        "model_kind": ["classifier"],
        "model_params": [
            json.dumps({
                "solver": "saga",
                "C": 1.0,
                "l1_ratio": 0.0,
                "class_weight": "balanced",
                "max_iter": 300,
            })
        ],
    })

    predictions = evaluate_feature_candidates_predictions(
        x.iloc[train_idx],
        y.iloc[train_idx],
        x.iloc[val_idx],
        y.iloc[val_idx],
        feature_candidates,
        model_specs,
        outer_fold=1,
        inner_fold=1,
        round_name="test",
        random_state=42,
        log_every=0,
        n_jobs=1,
    )
    blocks = evaluate_feature_candidates_oof_blocks(
        x.iloc[train_idx],
        y.iloc[train_idx],
        x.iloc[val_idx],
        y.iloc[val_idx],
        feature_candidates,
        model_specs,
        outer_fold=1,
        inner_fold=1,
        round_name="test",
        random_state=42,
        log_every=0,
        n_jobs=1,
    )

    sample_scored = score_inner_oof_predictions(
        predictions,
        max_targets=2,
        include_baseline_comparison=False,
    )
    block_scored, baselines = score_inner_oof_prediction_blocks(
        blocks,
        max_targets=2,
        include_baseline_comparison=False,
    )

    assert baselines.empty
    assert len(blocks) == 1
    assert len(blocks[0]["score"]) == len(val_idx)
    assert block_scored.iloc[0]["pipeline_recipe_id"] == sample_scored.iloc[0]["pipeline_recipe_id"]
    assert (
        block_scored.iloc[0]["inner_oof_business_score"]
        == sample_scored.iloc[0]["inner_oof_business_score"]
    )
    assert (
        block_scored.iloc[0]["inner_oof_optimal_k"] == sample_scored.iloc[0]["inner_oof_optimal_k"]
    )


def test_score_inner_oof_predictions_csv_scores_chunks(tmp_path) -> None:
    predictions = _toy_prediction_rows()
    csv_path = tmp_path / "round1_inner_scores.csv"
    output_path = tmp_path / "round1_inner_oof_scores.csv"
    baseline_path = tmp_path / "round1_inner_oof_baselines.csv"
    predictions.to_csv(csv_path, index=False)

    scored = score_inner_oof_predictions_csv(
        csv_path,
        max_targets=3,
        chunksize=3,
        random_repeats=1,
        random_state=123,
        output_path=output_path,
        baseline_output_path=baseline_path,
    )

    assert has_inner_prediction_columns(csv_path)
    assert output_path.exists()
    assert baseline_path.exists()
    assert scored.iloc[0]["pipeline_recipe_id"] == "fr2__m1"
    assert scored.iloc[0]["inner_oof_business_score"] == -180.0
    assert scored.iloc[0]["inner_oof_n_predictions"] == 4
    baselines = pd.read_csv(baseline_path)
    assert set(baselines["baseline_type"]) == {
        "select_all_capped_no_model",
        "select_all_full_illegal_diagnostic",
        "random_score_baseline",
    }


def test_inner_oof_scoring_allows_k_zero_when_all_targets_hurt_objective() -> None:
    predictions = _toy_prediction_rows().iloc[:2].copy()
    predictions["pipeline_recipe_id"] = "bad__m1"
    predictions["y_true"] = [0, 0]
    predictions["score"] = [0.9, 0.8]
    predictions["feature_size"] = 1

    scored = score_inner_oof_predictions(predictions, max_targets=2)

    assert scored.iloc[0]["inner_oof_business_score"] == -200.0
    assert scored.iloc[0]["inner_oof_optimal_k"] == 0
    assert scored.iloc[0]["gross_score"] == 0.0


def test_inner_oof_feature_size_is_cost_source_not_selected_feature_text() -> None:
    predictions = _toy_prediction_rows().iloc[:4].copy()
    predictions["selected_features"] = [["a", "b", "c"]] * len(predictions)
    predictions["selected_features_text"] = "a,b,c"
    predictions["feature_size"] = 1

    scored = score_inner_oof_predictions(predictions, max_targets=3)

    assert scored.iloc[0]["feature_size"] == 1
    assert scored.iloc[0]["feature_penalty"] == 200.0
    assert scored.iloc[0]["inner_oof_business_score"] == -185.0


def test_inner_prediction_csv_requires_feature_size_and_inner_fold(tmp_path) -> None:
    predictions = _toy_prediction_rows()
    missing_feature_size = tmp_path / "missing_feature_size.csv"
    missing_inner_fold = tmp_path / "missing_inner_fold.csv"
    predictions.drop(columns=["feature_size"]).to_csv(missing_feature_size, index=False)
    predictions.drop(columns=["inner_fold"]).to_csv(missing_inner_fold, index=False)

    assert not has_inner_prediction_columns(missing_feature_size)
    assert not has_inner_prediction_columns(missing_inner_fold)
    with pytest.raises(ValueError, match="feature_size"):
        score_inner_oof_predictions_csv(missing_feature_size, chunksize=2)
    with pytest.raises(ValueError, match="inner_fold"):
        score_inner_oof_predictions_csv(missing_inner_fold, chunksize=2)


def test_inner_oof_csv_and_in_memory_scoring_are_equivalent_with_manifest(tmp_path) -> None:
    predictions = _toy_prediction_rows()
    manifest = pd.DataFrame({
        "outer_fold": [1, 1, 1, 1],
        "inner_fold": [1, 1, 2, 2],
        "sample_index": [0, 1, 2, 3],
    })
    y = pd.Series([1, 0, 1, 0])
    csv_path = tmp_path / "predictions.csv"
    predictions.to_csv(csv_path, index=False)

    in_memory = (
        score_inner_oof_predictions(
            predictions,
            max_targets=3,
            inner_assignments=manifest,
            y=y,
            inner_folds=(1, 2),
        )
        .sort_values("pipeline_recipe_id")
        .reset_index(drop=True)
    )
    csv_scored = (
        score_inner_oof_predictions_csv(
            csv_path,
            max_targets=3,
            chunksize=3,
            random_repeats=0,
            outer_folds=(1,),
            round_names=("round1",),
            inner_assignments=manifest,
            y=y,
            inner_folds=(1, 2),
            verbose=False,
        )
        .sort_values("pipeline_recipe_id")
        .reset_index(drop=True)
    )

    cols = [
        "pipeline_recipe_id",
        "feature_size",
        "inner_oof_business_score",
        "inner_oof_optimal_k",
        "inner_oof_n_predictions",
        "missing_inner_oof_predictions",
        "missing_inner_fold_count",
        "manifest_checked",
    ]
    pd.testing.assert_frame_equal(in_memory[cols], csv_scored[cols], check_dtype=False)


def test_manifest_missing_inner_fold_marks_incomplete_and_selection_excludes_it() -> None:
    predictions = _toy_prediction_rows().loc[lambda frame: frame["inner_fold"].eq(1)].copy()
    manifest = pd.DataFrame({
        "outer_fold": [1, 1, 1, 1],
        "inner_fold": [1, 1, 2, 2],
        "sample_index": [0, 1, 2, 3],
    })
    y = pd.Series([1, 0, 1, 0])

    scored = score_inner_oof_predictions(
        predictions,
        max_targets=3,
        inner_assignments=manifest,
        y=y,
        inner_folds=(1, 2),
    )

    assert set(scored["status"]) == {"incomplete"}
    assert scored["missing_inner_fold_count"].eq(1).all()
    assert scored["missing_inner_oof_predictions"].eq(2).all()
    aggregated = aggregate_inner_scores(
        scored,
        group_cols=("outer_fold", "round_name", "pipeline_recipe_id"),
    )
    assert aggregated.empty
