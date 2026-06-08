import numpy as np
import pandas as pd

from cost_effective.models.model_family_validation import (
    candidate_feature_sets_from_rankings,
    feature_set_key,
    merge_candidate_feature_sets,
    parse_feature_list,
    rank_validation_scores,
    select_unique_feature_sets,
)


def test_parse_feature_list_and_key_are_stable() -> None:
    assert parse_feature_list("['var_2', 'var_1']") == ["var_2", "var_1"]
    assert parse_feature_list(["var_2", 1]) == ["var_2", "1"]
    assert parse_feature_list("not a list") == []
    assert feature_set_key(["var_2", "var_1", "var_2"]) == ("var_1", "var_2")


def test_select_unique_feature_sets_keeps_best_unique_valid_sets() -> None:
    experiment_results = pd.DataFrame({
        "status": ["ok", "ok", "error", "ok"],
        "source_features": [
            "['var_2', 'var_1']",
            "['var_1', 'var_2']",
            "['var_3']",
            "['var_4', 'missing']",
        ],
        "comparison_rank": [2, 1, 3, 4],
        "source_feature_count": [2, 2, 1, 2],
        "experiment_stage": ["a", "b", "c", "d"],
        "method_family": ["fam_a", "fam_b", "fam_c", "fam_d"],
        "method": ["m_a", "m_b", "m_c", "m_d"],
        "oof_business_score": [10.0, 11.0, 12.0, 4.0],
    })

    selected = select_unique_feature_sets(
        experiment_results,
        available_features=["var_1", "var_2", "var_4"],
        top_n=5,
        max_features=3,
    )

    assert selected["source_features"].tolist() == [["var_1", "var_2"], ["var_4"]]
    assert selected["origin_method"].tolist() == ["m_b", "m_d"]
    assert selected["candidate_source"].tolist() == ["previous_experiment", "previous_experiment"]


def test_candidate_feature_sets_from_rankings_deduplicates_top_k_sets() -> None:
    ranking_a = pd.DataFrame({
        "feature": ["var_1", "var_2", "var_3"],
        "order": [1, 2, 3],
        "ensemble_score": [0.9, 0.8, 0.2],
    })
    ranking_b = pd.DataFrame({
        "feature": ["var_2", "var_1", "var_4"],
        "order": [1, 2, 3],
        "ensemble_score": [0.7, 0.6, 0.5],
    })

    candidates = candidate_feature_sets_from_rankings(
        {"a": ranking_a, "b": ranking_b},
        sizes=(1, 2),
        max_features=5,
    )

    keys = {tuple(features) for features in candidates["source_features"]}
    assert keys == {("var_1",), ("var_2",), ("var_1", "var_2")}
    assert candidates["feature_set_id"].str.startswith("gen_").all()


def test_merge_candidate_feature_sets_preserves_order_and_relabels() -> None:
    previous = pd.DataFrame({
        "feature_set_id": ["old_1"],
        "source_features": [["var_2", "var_1"]],
        "source_feature_count": [2],
    })
    generated = pd.DataFrame({
        "feature_set_id": ["gen_1", "gen_2"],
        "source_features": [["var_1", "var_2"], ["var_3"]],
        "source_feature_count": [2, 1],
    })

    merged = merge_candidate_feature_sets([previous, generated], top_n=3)

    assert merged["feature_set_id"].tolist() == ["fs_001", "fs_002"]
    assert merged["source_features"].tolist() == [["var_1", "var_2"], ["var_3"]]


def test_rank_validation_scores_ranks_only_successful_rows() -> None:
    scores = pd.DataFrame({
        "status": ["ok", "error", "ok"],
        "validation_oof_business_score": [100.0, np.nan, 100.0],
        "source_feature_count": [5, 2, 3],
        "model_family": ["a", "b", "c"],
        "feature_set_id": ["fs_1", "fs_2", "fs_3"],
    })

    ranked = rank_validation_scores(scores)

    assert ranked.loc[0, "feature_set_id"] == "fs_3"
    assert ranked.loc[0, "validation_rank"] == 1
    assert ranked.loc[1, "validation_rank"] == 2
    assert ranked.loc[2, "status"] == "error"
