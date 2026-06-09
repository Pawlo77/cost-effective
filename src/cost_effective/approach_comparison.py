"""Load and compare summaries from all modeling approach output folders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import (
    APPROACH_CLUSTER_SPLIT,
    APPROACH_EV_PROFIT_TARGETING,
    APPROACH_RANK_FUSION,
    APPROACH_SEGMENT_TARGETING,
    APPROACH_TOPK_PROFIT_CURVE,
    SHARED_OUTPUTS,
    approach_outputs_dir,
)

ALL_MODELING_APPROACHES: tuple[str, ...] = (
    APPROACH_TOPK_PROFIT_CURVE,
    APPROACH_EV_PROFIT_TARGETING,
    APPROACH_RANK_FUSION,
    APPROACH_SEGMENT_TARGETING,
    APPROACH_CLUSTER_SPLIT,
)

APPROACH_LABELS: dict[str, str] = {
    APPROACH_TOPK_PROFIT_CURVE: "Top-k profit curve",
    APPROACH_EV_PROFIT_TARGETING: "EV profit targeting",
    APPROACH_RANK_FUSION: "Rank fusion committee",
    APPROACH_SEGMENT_TARGETING: "Segment-aware targeting",
    APPROACH_CLUSTER_SPLIT: "Cluster split (k=4)",
}


def _summary_path(project_root: Path, approach: str) -> Path:
    return approach_outputs_dir(project_root, approach) / "modeling_summary.json"


def load_approach_summary(path: Path) -> dict[str, Any]:
    """Read one ``modeling_summary.json``."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_summary(approach: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Map heterogeneous summary JSON to a flat comparison row."""
    features = raw.get("features") or raw.get("submission_features") or []
    feature_count = raw.get("feature_count", len(features))
    oof_k = raw.get("oof_optimal_k") or raw.get("oof_selected_k") or raw.get("test_targets")
    model = raw.get("best_model")
    if approach == APPROACH_RANK_FUSION:
        model = f"committee ({len(raw.get('experts', []))} experts)"
    elif approach in (APPROACH_SEGMENT_TARGETING, APPROACH_CLUSTER_SPLIT):
        model = "per-cluster logistic"

    row: dict[str, Any] = {
        "approach": approach,
        "label": APPROACH_LABELS.get(approach, approach),
        "model": model,
        "feature_set": raw.get("best_feature_set")
        or raw.get("model_feature_set")
        or raw.get("segment_feature_set")
        or raw.get("feature_set")
        or raw.get("cluster_feature_set"),
        "n_features": int(feature_count),
        "features": ", ".join(features) if features else "",
        "cv_business_mean": raw.get("cv_business_mean"),
        "hpo_cv_business_mean": raw.get("hpo_cv_business_mean"),
        "oof_k": oof_k,
        "oof_business_score": raw.get("oof_business_score"),
        "test_targets": raw.get("test_targets") or raw.get("test_exported_k"),
        "targeting_notes": raw.get("targeting_strategy"),
    }
    if approach == APPROACH_EV_PROFIT_TARGETING:
        row["targeting_notes"] = (
            f"{raw.get('targeting_strategy')}; k_ev={raw.get('k_ev_threshold')}, "
            f"k_elbow={raw.get('k_elbow')}, k_nested={raw.get('k_nested_cv')}"
        )
    if approach == APPROACH_RANK_FUSION and raw.get("fusion_weights"):
        top = sorted(raw["fusion_weights"].items(), key=lambda x: -x[1])[:2]
        row["targeting_notes"] = "weights: " + ", ".join(f"{n}={w:.2f}" for n, w in top)
    if approach == APPROACH_SEGMENT_TARGETING:
        row["targeting_notes"] = (
            f"GMM k={raw.get('n_clusters')}; clusters={raw.get('cluster_counts_train')}"
        )
    if approach == APPROACH_CLUSTER_SPLIT:
        row["targeting_notes"] = (
            f"KMeans k={raw.get('n_clusters')}; feature_set={raw.get('feature_set')}; "
            f"counts={raw.get('cluster_counts_train')}"
        )
    return row


def load_all_approach_summaries(
    project_root: Path,
    approaches: tuple[str, ...] = ALL_MODELING_APPROACHES,
) -> list[dict[str, Any]]:
    """Load available ``modeling_summary.json`` files (skip missing)."""
    rows: list[dict[str, Any]] = []
    for approach in approaches:
        path = _summary_path(project_root, approach)
        if not path.exists():
            continue
        rows.append(_normalize_summary(approach, load_approach_summary(path)))
    return rows


def build_approaches_comparison(
    project_root: Path,
    approaches: tuple[str, ...] = ALL_MODELING_APPROACHES,
) -> pd.DataFrame:
    """Comparison table of all completed modeling runs."""
    rows = load_all_approach_summaries(project_root, approaches)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["oof_business_score", "cv_business_mean"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def load_baseline_reference_rows(project_root: Path) -> pd.DataFrame:
    """Key baseline rows from ``baseline_results.json`` (CV means)."""
    path = project_root / SHARED_OUTPUTS / "baseline_results.json"
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as fh:
        summary = json.load(fh).get("summary", {})

    rows: list[dict[str, Any]] = []
    k0 = summary.get("k0_reference", {})
    if k0:
        rows.append({
            "approach": "baseline_k0",
            "label": "Baseline: k=0",
            "model": "—",
            "n_features": 0,
            "cv_business_mean": k0.get("business_with_var_penalty"),
            "oof_business_score": None,
            "oof_k": 0,
            "test_targets": 0,
        })

    pick = (
        "prior_dummy",
        "logistic_1feat_var_389",
        "logistic_500feat_C=0.0003",
    )
    for name in pick:
        metrics = summary.get(name)
        if not isinstance(metrics, dict):
            continue
        pen = metrics.get("business_with_var_penalty", {})
        nop = metrics.get("business_no_var_penalty", {})
        n_feat = 0 if "dummy" in name or name == "prior_dummy" else (500 if "500" in name else 1)
        rows.append({
            "approach": f"baseline_{name}",
            "label": f"Baseline: {name}",
            "model": "reference",
            "n_features": n_feat,
            "cv_business_mean": pen.get("mean") if isinstance(pen, dict) else pen,
            "cv_business_no_var": nop.get("mean") if isinstance(nop, dict) else nop,
            "oof_business_score": None,
            "oof_k": None,
            "test_targets": None,
        })
    return pd.DataFrame(rows)


def missing_approaches(
    project_root: Path,
    approaches: tuple[str, ...] = ALL_MODELING_APPROACHES,
) -> list[str]:
    """Approach folder names with no ``modeling_summary.json`` yet."""
    return [
        approach for approach in approaches if not _summary_path(project_root, approach).exists()
    ]
