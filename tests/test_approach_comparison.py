"""Tests for cross-approach comparison helpers."""

import json
from pathlib import Path

from cost_effective.approach_comparison import (
    build_approaches_comparison,
    load_all_approach_summaries,
    missing_approaches,
)
from cost_effective.paths import APPROACH_TOPK_PROFIT_CURVE


def test_build_approaches_comparison_from_temp_summaries(tmp_path: Path) -> None:
    out = tmp_path / "outputs" / APPROACH_TOPK_PROFIT_CURVE
    out.mkdir(parents=True)
    summary = {
        "approach": APPROACH_TOPK_PROFIT_CURVE,
        "best_model": "logistic_regression",
        "best_feature_set": "top_01",
        "features": ["var_1"],
        "cv_business_mean": 100.0,
        "oof_optimal_k": 50,
        "oof_business_score": 200.0,
        "test_targets": 50,
    }
    (out / "modeling_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    frame = build_approaches_comparison(tmp_path, approaches=(APPROACH_TOPK_PROFIT_CURVE,))
    assert len(frame) == 1
    assert frame.loc[0, "oof_business_score"] == 200.0
    assert frame.loc[0, "n_features"] == 1


def test_missing_approaches(tmp_path: Path) -> None:
    missing = missing_approaches(tmp_path)
    assert APPROACH_TOPK_PROFIT_CURVE in missing


def test_load_all_skips_missing(tmp_path: Path) -> None:
    rows = load_all_approach_summaries(tmp_path, approaches=(APPROACH_TOPK_PROFIT_CURVE,))
    assert rows == []
