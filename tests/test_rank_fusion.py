"""Tests for rank-fusion committee."""

import numpy as np
import pandas as pd
import pytest

from cost_effective.models.rank_fusion import (
    BordaRankClassifier,
    FusionExpertConfig,
    FusionExpertResult,
    fuse_probabilities,
    fusion_weights_from_cv,
    run_fusion_committee,
)


def _toy_frame(n: int = 120, p: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"var_{i}" for i in range(p)])
    y = pd.Series((x["var_0"] + rng.normal(scale=0.5, size=n) > 0).astype(int))
    return x, y


def test_borda_rank_classifier_proba() -> None:
    x, y = _toy_frame()
    clf = BordaRankClassifier(["var_0", "var_1"]).fit(x, y)
    proba = clf.predict_proba(x)
    assert proba.shape == (len(x), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_fusion_weights_normalize() -> None:
    experts = [
        FusionExpertResult(
            config=FusionExpertConfig("a", "top_01", "logistic_regression"),
            features=("var_0",),
            oof_probabilities=np.array([0.1, 0.9]),
            cv_business_mean=100.0,
            cv_business_std=0.0,
            weight=0.0,
        ),
        FusionExpertResult(
            config=FusionExpertConfig("b", "top_02", "logistic_regression"),
            features=("var_1",),
            oof_probabilities=np.array([0.2, 0.8]),
            cv_business_mean=300.0,
            cv_business_std=0.0,
            weight=0.0,
        ),
    ]
    weights = fusion_weights_from_cv(experts)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["b"] > weights["a"]


def test_fuse_probabilities_weighted() -> None:
    experts = [
        FusionExpertResult(
            config=FusionExpertConfig("a", "top_01", "logistic_regression"),
            features=("var_0",),
            oof_probabilities=np.array([0.0, 1.0]),
            cv_business_mean=1.0,
            cv_business_std=0.0,
            weight=0.25,
        ),
        FusionExpertResult(
            config=FusionExpertConfig("b", "top_02", "logistic_regression"),
            features=("var_1",),
            oof_probabilities=np.array([1.0, 0.0]),
            cv_business_mean=3.0,
            cv_business_std=0.0,
            weight=0.75,
        ),
    ]
    fused = fuse_probabilities(experts, {"a": 0.25, "b": 0.75})
    assert fused[0] == pytest.approx(0.75)
    assert fused[1] == pytest.approx(0.25)


def test_run_fusion_committee_small() -> None:
    x, y = _toy_frame(n=80, p=6)
    candidates = {
        "top_01": ["var_0"],
        "top_03": ["var_0", "var_1", "var_2"],
        "top_05": ["var_0", "var_1", "var_2", "var_3", "var_4"],
    }
    configs = (
        FusionExpertConfig("logistic_top01", "top_01", "logistic_regression"),
        FusionExpertConfig("borda_top03", "top_03", "borda_rank"),
    )
    fusion = run_fusion_committee(x, y, candidates, expert_configs=configs, cv=3)
    assert len(fusion.fused_oof_probabilities) == len(y)
    assert fusion.feature_count >= 1
    assert len(fusion.experts) == 2
