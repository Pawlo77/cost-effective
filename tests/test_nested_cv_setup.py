import numpy as np
import pandas as pd

from cost_effective.models.nested_cv_setup import (
    DEFAULT_HONEST_FEATURE_SIZES,
    DEFAULT_HONEST_PRESCREEN_METHODS,
    assert_nested_split_integrity,
    build_feature_size_grid,
    build_model_spec_space,
    build_pipeline_recipe_space,
    build_prescreen_recipe_space,
    leakage_contract_table,
    make_inner_fold_assignments,
    make_outer_fold_assignments,
    summarize_inner_folds,
    summarize_outer_folds,
)


def _toy_y() -> pd.Series:
    return pd.Series(np.r_[np.zeros(80, dtype=int), np.ones(20, dtype=int)])


def test_outer_and_inner_assignments_are_nested_without_overlap() -> None:
    y = _toy_y()
    outer = make_outer_fold_assignments(y, n_splits=5, random_state=7)
    inner = make_inner_fold_assignments(y, outer, n_splits=4, random_state=11)

    assert_nested_split_integrity(
        outer,
        inner,
        n_samples=len(y),
        outer_splits=5,
        inner_splits=4,
    )
    assert outer["sample_index"].nunique() == len(y)

    for outer_fold in sorted(outer["outer_fold"].unique()):
        outer_val = set(outer.loc[outer["outer_fold"].eq(outer_fold), "sample_index"])
        inner_samples = set(inner.loc[inner["outer_fold"].eq(outer_fold), "sample_index"])
        assert outer_val.isdisjoint(inner_samples)
        assert len(inner_samples) == len(y) - len(outer_val)


def test_fold_summaries_have_expected_shape_and_positive_rates() -> None:
    y = _toy_y()
    outer = make_outer_fold_assignments(y, n_splits=5, random_state=7)
    inner = make_inner_fold_assignments(y, outer, n_splits=4, random_state=11)

    outer_summary = summarize_outer_folds(y, outer)
    inner_summary = summarize_inner_folds(y, inner)

    assert outer_summary.shape[0] == 5
    assert inner_summary.shape[0] == 20
    assert outer_summary["val_count"].sum() == len(y)
    assert outer_summary["val_positive_rate"].between(0, 1).all()
    assert inner_summary["inner_val_positive_rate"].between(0, 1).all()


def test_prescreen_recipe_space_contains_all_mean_combinations_plus_adaptive_weights() -> None:
    methods = ("mutual_info", "abs_corr", "lightgbm_gain")
    recipes = build_prescreen_recipe_space(methods)

    assert recipes.shape[0] == (2 ** len(methods) - 1) + 2
    assert recipes["prescreen_recipe_id"].is_unique
    assert recipes["adaptive"].sum() == 2
    assert "weighted_all__train_only_single_scores" in set(recipes["prescreen_name"])


def test_pipeline_recipe_space_is_cartesian_declaration() -> None:
    prescreen = build_prescreen_recipe_space(
        ("mutual_info", "abs_corr"),
        include_adaptive_weighted=False,
    )
    sizes = build_feature_size_grid((1, 3))
    models = build_model_spec_space(
        include_xgboost=False,
        include_ebm=False,
        include_lambdamart=False,
    ).head(2)

    pipeline = build_pipeline_recipe_space(prescreen, sizes, models)

    assert pipeline.shape[0] == len(prescreen) * len(sizes) * len(models)
    assert pipeline["pipeline_recipe_id"].is_unique
    assert set(pipeline["feature_size"]) == {1, 3}


def test_default_recipe_space_is_large_but_declarative() -> None:
    prescreen = build_prescreen_recipe_space(DEFAULT_HONEST_PRESCREEN_METHODS)
    sizes = build_feature_size_grid(DEFAULT_HONEST_FEATURE_SIZES)
    models = build_model_spec_space(
        include_xgboost=False,
        include_ebm=False,
        include_lambdamart=False,
    )
    pipeline = build_pipeline_recipe_space(prescreen, sizes, models)

    assert len(prescreen) == 257
    assert len(sizes) == len(DEFAULT_HONEST_FEATURE_SIZES)
    assert len(pipeline) == len(prescreen) * len(sizes) * len(models)


def test_leakage_contract_mentions_inner_and_outer_forbidden_data() -> None:
    contract = leakage_contract_table()

    assert not contract.empty
    assert contract["forbidden_data"].str.contains("outer_val").any()
    assert contract["stage"].str.contains("inner").any()
