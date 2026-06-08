"""Setup helpers for the alternative feature-selection workflow."""

from cost_effective.models.nested_cv_setup import (
    DEFAULT_NESTED_CV_FEATURE_SIZES as DEFAULT_ALTERNATIVE_FEATURE_SIZES,
)
from cost_effective.models.nested_cv_setup import (
    DEFAULT_NESTED_CV_PRESCREEN_METHODS as DEFAULT_ALTERNATIVE_PRESCREEN_METHODS,
)
from cost_effective.models.nested_cv_setup import (
    assert_nested_split_integrity as assert_split_integrity,
)
from cost_effective.models.nested_cv_setup import (
    build_feature_size_grid,
    build_model_spec_space,
    build_pipeline_recipe_space,
    build_prescreen_recipe_space,
    leakage_contract_table,
    make_inner_fold_assignments,
    make_outer_fold_assignments,
    summarize_inner_folds,
    summarize_outer_folds,
    write_stage_one_outputs,
)

__all__ = [
    "DEFAULT_ALTERNATIVE_FEATURE_SIZES",
    "DEFAULT_ALTERNATIVE_PRESCREEN_METHODS",
    "assert_split_integrity",
    "build_feature_size_grid",
    "build_model_spec_space",
    "build_pipeline_recipe_space",
    "build_prescreen_recipe_space",
    "leakage_contract_table",
    "make_inner_fold_assignments",
    "make_outer_fold_assignments",
    "summarize_inner_folds",
    "summarize_outer_folds",
    "write_stage_one_outputs",
]
