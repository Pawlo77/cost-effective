"""Rank-fusion audit model specification grids."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AuditModelSpec:
    """Seedable model specification for final audit experiments."""

    model_name: str
    base_model_family: str
    kind: str
    params: Mapping[str, Any]


def expanded_hpo_model_specs(
    *,
    include_xgboost: bool = True,
    include_ebm: bool = False,
    include_lambdamart: bool = True,
) -> list[AuditModelSpec]:
    """Return a deeper but still curated final-stage HPO grid."""
    grid: dict[str, list[dict[str, Any]]] = {
        "extra_trees_small": [
            {
                "n_estimators": 320,
                "max_depth": 3,
                "min_samples_leaf": 8,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 320,
                "max_depth": 4,
                "min_samples_leaf": 12,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 400,
                "max_depth": 5,
                "min_samples_leaf": 16,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 400,
                "max_depth": 6,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 500,
                "max_depth": 7,
                "min_samples_leaf": 24,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 500,
                "max_depth": 8,
                "min_samples_leaf": 30,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 35,
                "class_weight": "balanced",
            },
            {"n_estimators": 500, "max_depth": 5, "min_samples_leaf": 20, "class_weight": None},
        ],
        "random_forest_small": [
            {
                "n_estimators": 320,
                "max_depth": 4,
                "min_samples_leaf": 10,
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 400,
                "max_depth": 5,
                "min_samples_leaf": 16,
                "class_weight": "balanced_subsample",
            },
            {
                "n_estimators": 400,
                "max_depth": 6,
                "min_samples_leaf": 20,
                "class_weight": "balanced",
            },
            {
                "n_estimators": 500,
                "max_depth": 7,
                "min_samples_leaf": 24,
                "class_weight": "balanced",
            },
            {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 30, "class_weight": None},
            {
                "n_estimators": 600,
                "max_depth": None,
                "min_samples_leaf": 40,
                "class_weight": "balanced_subsample",
            },
        ],
        "lightgbm_classifier": [
            {
                "n_estimators": 160,
                "learning_rate": 0.05,
                "num_leaves": 7,
                "min_child_samples": 10,
                "scale_pos_weight_multiplier": 0.75,
            },
            {
                "n_estimators": 220,
                "learning_rate": 0.04,
                "num_leaves": 11,
                "min_child_samples": 15,
                "scale_pos_weight_multiplier": 0.9,
            },
            {
                "n_estimators": 260,
                "learning_rate": 0.035,
                "num_leaves": 15,
                "min_child_samples": 20,
                "scale_pos_weight_multiplier": 1.0,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 23,
                "min_child_samples": 25,
                "scale_pos_weight_multiplier": 1.1,
            },
            {
                "n_estimators": 360,
                "learning_rate": 0.025,
                "num_leaves": 31,
                "min_child_samples": 30,
                "scale_pos_weight_multiplier": 1.25,
            },
            {
                "n_estimators": 420,
                "learning_rate": 0.02,
                "num_leaves": 39,
                "min_child_samples": 35,
                "scale_pos_weight_multiplier": 1.4,
            },
            {
                "n_estimators": 260,
                "learning_rate": 0.04,
                "num_leaves": 15,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "n_estimators": 320,
                "learning_rate": 0.03,
                "num_leaves": 23,
                "min_child_samples": 50,
                "reg_lambda": 3.0,
            },
        ],
        "logistic_baseline": [
            {
                "solver": "saga",
                "C": 0.05,
                "l1_ratio": 0.0,
                "class_weight": "balanced",
                "max_iter": 2500,
            },
            {
                "solver": "saga",
                "C": 0.1,
                "l1_ratio": 0.2,
                "class_weight": "balanced",
                "max_iter": 2500,
            },
            {
                "solver": "saga",
                "C": 0.2,
                "l1_ratio": 0.5,
                "class_weight": "balanced",
                "max_iter": 2500,
            },
            {
                "solver": "saga",
                "C": 0.5,
                "l1_ratio": 0.8,
                "class_weight": "balanced",
                "max_iter": 2500,
            },
            {
                "solver": "saga",
                "C": 1.0,
                "l1_ratio": 1.0,
                "class_weight": "balanced",
                "max_iter": 2500,
            },
            {"solver": "saga", "C": 2.0, "l1_ratio": 0.5, "class_weight": None, "max_iter": 2500},
        ],
    }
    if include_xgboost:
        grid["xgboost_classifier"] = [
            {
                "n_estimators": 160,
                "learning_rate": 0.05,
                "max_depth": 2,
                "min_child_weight": 1.0,
                "scale_pos_weight_multiplier": 0.75,
            },
            {
                "n_estimators": 220,
                "learning_rate": 0.04,
                "max_depth": 2,
                "min_child_weight": 2.0,
                "scale_pos_weight_multiplier": 1.0,
            },
            {
                "n_estimators": 260,
                "learning_rate": 0.035,
                "max_depth": 3,
                "min_child_weight": 3.0,
                "scale_pos_weight_multiplier": 1.0,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 5.0,
                "scale_pos_weight_multiplier": 1.2,
            },
            {
                "n_estimators": 360,
                "learning_rate": 0.025,
                "max_depth": 4,
                "min_child_weight": 5.0,
                "scale_pos_weight_multiplier": 1.3,
            },
            {
                "n_estimators": 420,
                "learning_rate": 0.02,
                "max_depth": 4,
                "min_child_weight": 8.0,
                "scale_pos_weight_multiplier": 1.5,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "max_depth": 2,
                "min_child_weight": 8.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            {
                "n_estimators": 360,
                "learning_rate": 0.025,
                "max_depth": 3,
                "min_child_weight": 10.0,
                "reg_lambda": 3.0,
            },
        ]
    if include_ebm:
        grid["ebm_additive"] = [
            {"interactions": 0, "max_bins": 64, "outer_bags": 4, "max_rounds": 3000},
            {"interactions": 0, "max_bins": 128, "outer_bags": 6, "max_rounds": 4000},
            {"interactions": 2, "max_bins": 64, "outer_bags": 4, "max_rounds": 3000},
        ]

    specs: list[AuditModelSpec] = []
    for family, params_list in grid.items():
        for idx, params in enumerate(params_list, start=1):
            specs.append(
                AuditModelSpec(
                    model_name=f"{family}_deep{idx:02d}",
                    base_model_family=family,
                    kind="classifier",
                    params=dict(params),
                )
            )
    if include_lambdamart:
        for idx, params in enumerate(
            [
                {"n_estimators": 160, "learning_rate": 0.05, "num_leaves": 15},
                {"n_estimators": 240, "learning_rate": 0.04, "num_leaves": 23},
                {"n_estimators": 320, "learning_rate": 0.03, "num_leaves": 31},
            ],
            start=1,
        ):
            specs.append(
                AuditModelSpec(
                    model_name=f"lambdamart_ranker_deep{idx:02d}",
                    base_model_family="lambdamart_ranker",
                    kind="lambdamart",
                    params=dict(params),
                )
            )
    return specs
