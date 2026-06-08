"""Inner-OOF scoring and aggregation helpers for nested-CV selection."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cost_effective.dataset.utils import DEFAULT_MAX_TARGETS
from cost_effective.models.modeling import build_f1_curve
from cost_effective.models.parsing import parse_feature_names


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _first_non_null(frame: pd.DataFrame, column: str, default: Any = "") -> Any:
    if column not in frame.columns:
        return default
    values = frame[column].dropna()
    if values.empty:
        return default
    return values.iloc[0]


def _coerce_feature_names(value: Any) -> list[str]:
    return parse_feature_names(value)


def _prediction_group_features(frame: pd.DataFrame) -> list[str]:
    """Return feature metadata for display only; never for feature cost."""
    feature_values: list[str] = []
    source_column = "source_features" if "source_features" in frame.columns else "selected_features"
    if source_column in frame.columns:
        for value in frame[source_column].dropna():
            feature_values.extend(_coerce_feature_names(value))
    if not feature_values and "selected_features_text" in frame.columns:
        for value in frame["selected_features_text"].dropna():
            feature_values.extend(_coerce_feature_names(value))
    return list(dict.fromkeys(feature_values))


def _normalize_feature_size(value: Any) -> int:
    if value is None or pd.isna(value):
        raise ValueError("feature_size is required for final inner-OOF scoring")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature_size must be an integer-like value, got {value!r}") from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"feature_size must be an integer-like value, got {value!r}")
    size = int(numeric_value)
    if size < 0:
        raise ValueError("feature_size must be non-negative")
    return size


def _consistent_feature_size(frame: pd.DataFrame) -> int:
    if "feature_size" not in frame.columns:
        raise ValueError("feature_size is required for final inner-OOF scoring")
    values = frame["feature_size"].dropna().map(_normalize_feature_size).drop_duplicates()
    if values.empty:
        raise ValueError("feature_size is required for final inner-OOF scoring")
    if len(values) != 1:
        raise ValueError(f"inconsistent feature_size values: {values.tolist()}")
    return int(values.iloc[0])


def _expected_inner_oof_context(
    inner_assignments: pd.DataFrame | None,
    outer_fold: int,
    *,
    inner_folds: Sequence[int] | None = None,
    y: pd.Series | None = None,
) -> tuple[set[Any] | None, set[int] | None, dict[Any, int] | None]:
    if inner_assignments is None:
        return None, None, None
    required = {"outer_fold", "inner_fold", "sample_index"}
    missing = required.difference(inner_assignments.columns)
    if missing:
        raise ValueError(f"inner_assignments missing required columns: {sorted(missing)}")

    rows = inner_assignments.loc[inner_assignments["outer_fold"].eq(int(outer_fold))].copy()
    if inner_folds is not None:
        allowed_folds = {int(fold) for fold in inner_folds}
        rows = rows.loc[rows["inner_fold"].isin(allowed_folds)].copy()
    if rows.empty:
        raise ValueError(f"no manifest rows for outer_fold={outer_fold}")

    expected_samples = set(rows["sample_index"].map(_python_scalar).tolist())
    expected_inner_folds = set(rows["inner_fold"].astype(int).tolist())
    labels: dict[Any, int] | None = None
    if y is not None:
        sample_indices = rows["sample_index"].astype(int).to_numpy()
        labels = {
            _python_scalar(sample_index): int(label)
            for sample_index, label in zip(sample_indices, y.iloc[sample_indices], strict=True)
        }
    return expected_samples, expected_inner_folds, labels


def _labels_from_prediction_rows(frame: pd.DataFrame) -> dict[Any, int]:
    labels: dict[Any, int] = {}
    for sample_index, y_true in (
        frame[["sample_index", "y_true"]].drop_duplicates().itertuples(index=False)
    ):
        sample_key = _python_scalar(sample_index)
        label = int(y_true)
        previous = labels.get(sample_key)
        if previous is not None and previous != label:
            raise ValueError(f"conflicting y_true values for sample_index={sample_key}")
        labels[sample_key] = label
    return labels


def _score_pooled_inner_oof(
    y_true: Sequence[int] | pd.Series,
    scores: Sequence[float] | np.ndarray,
    *,
    feature_size: int,
    method: str,
    max_targets: int = DEFAULT_MAX_TARGETS,
) -> dict[str, Any]:
    """Canonical honest-02 objective: 10*TP - 5*FP - 200*feature_size.

    This is intentionally unscaled and allows k=0. It does not use break-even caps,
    selected feature names, or source-feature unions.
    """
    feature_count = _normalize_feature_size(feature_size)
    y_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    if len(y_array) != len(score_array):
        raise ValueError("y_true and scores must have the same length")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite for final inner-OOF scoring")

    target_limit = min(max(int(max_targets), 0), len(y_array))
    feature_penalty = float(200 * feature_count)
    rows: list[dict[str, Any]] = [
        {
            "k": 0,
            "tp": 0,
            "fp": 0,
            "score": -feature_penalty,
            "gross_score": 0.0,
            "threshold": np.nan,
        }
    ]

    if target_limit > 0:
        ranking = np.argsort(score_array)[::-1]
        ranked_targets = y_array[ranking]
        ranked_scores = score_array[ranking]
        cumulative_tp = np.cumsum(ranked_targets[:target_limit] == 1)
        cumulative_fp = np.cumsum(ranked_targets[:target_limit] == 0)
        gross_scores = cumulative_tp * 10 - cumulative_fp * 5
        rows.extend(
            {
                "k": int(k),
                "tp": int(tp),
                "fp": int(fp),
                "score": float(gross - feature_penalty),
                "gross_score": float(gross),
                "threshold": float(threshold),
            }
            for k, tp, fp, gross, threshold in zip(
                range(1, target_limit + 1),
                cumulative_tp,
                cumulative_fp,
                gross_scores,
                ranked_scores[:target_limit],
                strict=True,
            )
        )

    curve = pd.DataFrame(rows)
    best_row = curve.iloc[int(curve["score"].idxmax())]
    try:
        f1 = build_f1_curve(pd.Series(y_array), score_array, max_targets=max_targets)
        f1_score = f1.best_f1
        roc_auc_score = f1.roc_auc
    except ValueError:
        f1_score = np.nan
        roc_auc_score = np.nan

    return {
        "method": method,
        "source_feature_count": feature_count,
        "candidate_column_count": feature_count,
        "oof_business_score": float(best_row["score"]),
        "oof_optimal_k": int(best_row["k"]),
        "oof_tp_at_optimal_k": int(best_row["tp"]),
        "oof_fp_at_optimal_k": int(best_row["fp"]),
        "oof_n_predictions": len(y_array),
        "oof_selected_rate": float(best_row["k"] / len(y_array)) if len(y_array) else 0.0,
        "feature_penalty": feature_penalty,
        "gross_score": float(best_row["gross_score"]),
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
        "score_curve": curve,
    }


def _ok_prediction_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    required = {
        "outer_fold",
        "inner_fold",
        "round_name",
        "pipeline_recipe_id",
        "sample_index",
        "y_true",
        "score",
        "feature_size",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing required columns: {sorted(missing)}")
    status = predictions["status"] if "status" in predictions.columns else "ok"
    return predictions.loc[
        pd.Series(status, index=predictions.index).eq("ok")
        & predictions["sample_index"].notna()
        & predictions["y_true"].notna()
        & predictions["score"].notna()
        & predictions["feature_size"].notna()
        & predictions["inner_fold"].notna()
    ].copy()


def _baseline_rows_from_inner_oof_labels(
    labels_by_sample: dict[Any, int],
    *,
    outer_fold: int,
    round_name: str,
    max_targets: int,
    random_repeats: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    sample_items = sorted(labels_by_sample.items(), key=lambda item: item[0])
    y_values = np.asarray([label for _sample, label in sample_items], dtype=int)
    n_predictions = len(y_values)
    target_limit = min(max(int(max_targets), 0), n_predictions)
    rows: list[dict[str, Any]] = []

    for n_features in (0, 1):
        selected = y_values[:target_limit]
        tp = int((selected == 1).sum())
        fp = int((selected == 0).sum())
        feature_penalty = float(200 * n_features)
        gross_score = float(10 * tp - 5 * fp)
        rows.append({
            "outer_fold": int(outer_fold),
            "round_name": round_name,
            "baseline_name": f"baseline_select_all_capped_features_{n_features}",
            "baseline_type": "select_all_capped_no_model",
            "random_repeat": np.nan,
            "baseline_business_score": gross_score - feature_penalty,
            "baseline_optimal_k": target_limit,
            "baseline_f1_score": np.nan,
            "baseline_roc_auc_score": np.nan,
            "baseline_n_predictions": n_predictions,
            "baseline_selected_rate": float(target_limit / n_predictions) if n_predictions else 0.0,
            "baseline_n_features": n_features,
            "feature_penalty": feature_penalty,
            "gross_score": gross_score,
            "tp": tp,
            "fp": fp,
            "is_legal_under_max_targets": True,
            "status": "ok",
        })
        if n_predictions > target_limit:
            full_tp = int((y_values == 1).sum())
            full_fp = int((y_values == 0).sum())
            full_gross = float(10 * full_tp - 5 * full_fp)
            rows.append({
                "outer_fold": int(outer_fold),
                "round_name": round_name,
                "baseline_name": f"baseline_select_all_illegal_full_features_{n_features}",
                "baseline_type": "select_all_full_illegal_diagnostic",
                "random_repeat": np.nan,
                "baseline_business_score": full_gross - feature_penalty,
                "baseline_optimal_k": n_predictions,
                "baseline_f1_score": np.nan,
                "baseline_roc_auc_score": np.nan,
                "baseline_n_predictions": n_predictions,
                "baseline_selected_rate": 1.0 if n_predictions else 0.0,
                "baseline_n_features": n_features,
                "feature_penalty": feature_penalty,
                "gross_score": full_gross,
                "tp": full_tp,
                "fp": full_fp,
                "is_legal_under_max_targets": False,
                "status": "illegal_max_targets_exceeded",
            })

    for repeat in range(int(random_repeats)):
        scored = _score_pooled_inner_oof(
            y_values,
            rng.random(n_predictions),
            feature_size=0,
            method=(
                f"baseline_random_score__outer{int(outer_fold):02d}__"
                f"{round_name}__repeat{repeat:03d}"
            ),
            max_targets=max_targets,
        )
        rows.append({
            "outer_fold": int(outer_fold),
            "round_name": round_name,
            "baseline_name": "baseline_random_score_features_0",
            "baseline_type": "random_score_baseline",
            "random_repeat": int(repeat),
            "baseline_business_score": scored.get("oof_business_score", np.nan),
            "baseline_optimal_k": scored.get("oof_optimal_k", np.nan),
            "baseline_f1_score": scored.get("f1_score", np.nan),
            "baseline_roc_auc_score": scored.get("roc_auc_score", np.nan),
            "baseline_n_predictions": scored.get("oof_n_predictions", n_predictions),
            "baseline_selected_rate": scored.get("oof_selected_rate", np.nan),
            "baseline_n_features": 0,
            "feature_penalty": scored.get("feature_penalty", 0.0),
            "gross_score": scored.get("gross_score", np.nan),
            "tp": scored.get("oof_tp_at_optimal_k", np.nan),
            "fp": scored.get("oof_fp_at_optimal_k", np.nan),
            "is_legal_under_max_targets": True,
            "status": "ok",
        })
    return rows


def _merge_baseline_scores(score_frame: pd.DataFrame, baseline_frame: pd.DataFrame) -> pd.DataFrame:
    if score_frame.empty or baseline_frame.empty:
        return score_frame
    baseline_scores = baseline_frame.pivot_table(
        index=["outer_fold", "round_name"],
        columns="baseline_name",
        values="baseline_business_score",
        aggfunc="first",
    ).reset_index()
    baseline_scores.columns = [
        f"{column}_business_score" if str(column).startswith("baseline_") else column
        for column in baseline_scores.columns
    ]
    out = score_frame.merge(baseline_scores, on=["outer_fold", "round_name"], how="left")
    for column in baseline_scores.columns:
        if str(column).startswith("baseline_select_all"):
            out[f"beats_{column.removesuffix('_business_score')}"] = (
                out["inner_oof_business_score"] > out[column]
            )
    return out


def score_inner_oof_baselines(
    predictions: pd.DataFrame,
    *,
    max_targets: int = DEFAULT_MAX_TARGETS,
    random_repeats: int = 0,
    random_state: int = 42,
    inner_assignments: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    inner_folds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Score legal capped select-all and optional random baselines."""
    ok_predictions = _ok_prediction_rows(predictions)
    if ok_predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_state)
    for (outer_fold, round_name), context in ok_predictions.groupby(
        ["outer_fold", "round_name"], sort=False
    ):
        expected_samples, _expected_folds, expected_labels = _expected_inner_oof_context(
            inner_assignments,
            int(outer_fold),
            inner_folds=inner_folds,
            y=y,
        )
        labels_by_sample = expected_labels or _labels_from_prediction_rows(context)
        if expected_samples is not None:
            missing_labels = expected_samples.difference(labels_by_sample)
            if missing_labels:
                raise ValueError(
                    "baseline labels missing for manifest samples: "
                    f"outer_fold={outer_fold}, examples={sorted(missing_labels)[:5]}"
                )
        rows.extend(
            _baseline_rows_from_inner_oof_labels(
                labels_by_sample,
                outer_fold=int(outer_fold),
                round_name=str(round_name),
                max_targets=max_targets,
                random_repeats=random_repeats,
                rng=rng,
            )
        )
    return pd.DataFrame(rows)


def score_inner_oof_predictions(
    predictions: pd.DataFrame,
    *,
    max_targets: int = DEFAULT_MAX_TARGETS,
    include_baseline_comparison: bool = True,
    inner_assignments: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    inner_folds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Score each recipe once on combined inner-OOF predictions per outer fold."""
    ok_predictions = _ok_prediction_rows(predictions)
    if ok_predictions.empty:
        return pd.DataFrame()

    group_keys = ["outer_fold", "round_name", "pipeline_recipe_id"]
    duplicates = ok_predictions.duplicated([*group_keys, "sample_index"], keep=False)
    if duplicates.any():
        duplicate_rows = ok_predictions.loc[
            duplicates, [*group_keys, "inner_fold", "sample_index"]
        ].head(10)
        raise ValueError(
            "duplicate inner-OOF predictions for the same recipe/sample: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    labels_by_context = {
        key: _labels_from_prediction_rows(group)
        for key, group in ok_predictions.groupby(["outer_fold", "round_name"], sort=False)
    }

    rows: list[dict[str, Any]] = []
    for (outer_fold, round_name, pipeline_recipe_id), group in ok_predictions.groupby(
        group_keys, sort=False
    ):
        group = group.sort_values("sample_index").copy()
        feature_size = _consistent_feature_size(group)
        expected_samples, expected_folds, _expected_labels = _expected_inner_oof_context(
            inner_assignments,
            int(outer_fold),
            inner_folds=inner_folds,
            y=y,
        )
        if expected_samples is None:
            expected_samples = set(labels_by_context[(outer_fold, round_name)])
        if expected_folds is None:
            expected_folds = set(
                ok_predictions.loc[
                    ok_predictions["outer_fold"].eq(outer_fold)
                    & ok_predictions["round_name"].eq(round_name),
                    "inner_fold",
                ].astype(int)
            )

        sample_set = set(group["sample_index"].map(_python_scalar).tolist())
        extra_samples = sample_set.difference(expected_samples)
        if extra_samples:
            raise ValueError(
                "prediction rows contain samples outside the expected inner-OOF manifest: "
                f"outer_fold={outer_fold}, examples={sorted(extra_samples)[:5]}"
            )
        observed_folds = set(group["inner_fold"].astype(int).tolist())
        extra_folds = observed_folds.difference(expected_folds)
        if extra_folds:
            raise ValueError(
                "prediction rows contain inner folds outside the expected manifest: "
                f"outer_fold={outer_fold}, examples={sorted(extra_folds)[:5]}"
            )
        missing_samples = expected_samples.difference(sample_set)
        missing_folds = expected_folds.difference(observed_folds)
        status = "ok" if not missing_samples and not missing_folds else "incomplete"
        selected_features_text = _first_non_null(group, "selected_features_text")
        if not selected_features_text:
            selected_features_text = ",".join(_prediction_group_features(group))
        method = f"inner_oof__outer{int(outer_fold):02d}__{round_name}__{pipeline_recipe_id}"
        scored = _score_pooled_inner_oof(
            group["y_true"].astype(int).reset_index(drop=True),
            group["score"].astype(float).to_numpy(),
            feature_size=feature_size,
            method=method,
            max_targets=max_targets,
        )
        n_predictions = int(scored.get("oof_n_predictions", len(group)))
        optimal_k = int(scored.get("oof_optimal_k", 0))
        rows.append({
            "outer_fold": int(outer_fold),
            "round_name": round_name,
            "pipeline_recipe_id": pipeline_recipe_id,
            "feature_recipe_id": _first_non_null(group, "feature_recipe_id"),
            "prescreen_recipe_id": _first_non_null(group, "prescreen_recipe_id"),
            "prescreen_name": _first_non_null(group, "prescreen_name"),
            "prescreen_methods": _first_non_null(group, "prescreen_methods"),
            "rank_aggregation": _first_non_null(group, "rank_aggregation"),
            "feature_size": feature_size,
            "selected_features_text": selected_features_text,
            "model_spec_id": _first_non_null(group, "model_spec_id"),
            "base_model_family": _first_non_null(group, "base_model_family"),
            "model_kind": _first_non_null(group, "model_kind"),
            "model_params": _first_non_null(group, "model_params"),
            "inner_oof_business_score": scored.get("oof_business_score", np.nan),
            "inner_oof_optimal_k": optimal_k,
            "inner_oof_f1_score": scored.get("f1_score", np.nan),
            "inner_oof_roc_auc_score": scored.get("roc_auc_score", np.nan),
            "inner_oof_n_predictions": n_predictions,
            "inner_oof_selected_rate": float(optimal_k / n_predictions) if n_predictions else 0.0,
            "inner_oof_tp_at_optimal_k": scored.get("oof_tp_at_optimal_k", np.nan),
            "inner_oof_fp_at_optimal_k": scored.get("oof_fp_at_optimal_k", np.nan),
            "feature_penalty": scored.get("feature_penalty", float(200 * feature_size)),
            "gross_score": scored.get("gross_score", np.nan),
            "expected_inner_oof_predictions": len(expected_samples),
            "missing_inner_oof_predictions": len(missing_samples),
            "inner_fold_count": len(observed_folds),
            "expected_inner_fold_count": len(expected_folds),
            "missing_inner_fold_count": len(missing_folds),
            "missing_inner_folds": ",".join(map(str, sorted(missing_folds))),
            "manifest_checked": inner_assignments is not None,
            "status": status,
        })

    scored_frame = pd.DataFrame(rows)
    if include_baseline_comparison and not scored_frame.empty:
        baselines = score_inner_oof_baselines(
            predictions,
            max_targets=max_targets,
            random_repeats=0,
            inner_assignments=inner_assignments,
            y=y,
            inner_folds=inner_folds,
        )
        scored_frame = _merge_baseline_scores(scored_frame, baselines)

    sort_cols = ["inner_oof_business_score"]
    ascending = [False]
    if "inner_oof_roc_auc_score" in scored_frame.columns:
        sort_cols.append("inner_oof_roc_auc_score")
        ascending.append(False)
    sort_cols.append("feature_size")
    ascending.append(True)
    return scored_frame.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _block_metadata(block: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"sample_index", "y_true", "score", "status", "error", "duration_seconds"}
    return {key: value for key, value in block.items() if key not in excluded}


def _labels_from_oof_blocks(blocks: Sequence[Mapping[str, Any]]) -> dict[Any, int]:
    labels: dict[Any, int] = {}
    for block in blocks:
        if str(block.get("status", "ok")) != "ok":
            continue
        for sample_index, y_true in zip(
            np.asarray(block.get("sample_index", [])),
            np.asarray(block.get("y_true", []), dtype=int),
            strict=False,
        ):
            sample_key = _python_scalar(sample_index)
            label = int(y_true)
            previous = labels.get(sample_key)
            if previous is not None and previous != label:
                raise ValueError(f"conflicting y_true values for sample_index={sample_key}")
            labels[sample_key] = label
    return labels


def score_inner_oof_prediction_blocks(
    blocks: Sequence[Mapping[str, Any]],
    *,
    max_targets: int = DEFAULT_MAX_TARGETS,
    include_baseline_comparison: bool = True,
    random_repeats: int = 0,
    random_state: int = 42,
    inner_assignments: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    inner_folds: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score compact OOF prediction blocks without writing sample-level predictions."""
    if not blocks:
        return pd.DataFrame(), pd.DataFrame()

    states: dict[tuple[int, str, str], dict[str, Any]] = {}
    error_rows: list[dict[str, Any]] = []
    labels_by_context: dict[tuple[int, str], dict[Any, int]] = {}

    for block in blocks:
        outer_fold = int(block["outer_fold"])
        inner_fold = int(block["inner_fold"])
        round_name = str(block["round_name"])
        pipeline_recipe_id = str(block["pipeline_recipe_id"])
        metadata = _block_metadata(block)
        status = str(block.get("status", "ok"))
        if status != "ok":
            error_rows.append({
                "outer_fold": outer_fold,
                "round_name": round_name,
                "pipeline_recipe_id": pipeline_recipe_id,
                "feature_recipe_id": metadata.get("feature_recipe_id", ""),
                "prescreen_recipe_id": metadata.get("prescreen_recipe_id", ""),
                "prescreen_name": metadata.get("prescreen_name", ""),
                "prescreen_methods": metadata.get("prescreen_methods", ""),
                "rank_aggregation": metadata.get("rank_aggregation", ""),
                "feature_size": metadata.get("feature_size", np.nan),
                "selected_features_text": metadata.get("selected_features_text", ""),
                "candidate_heuristic_score": metadata.get("candidate_heuristic_score", np.nan),
                "model_spec_id": metadata.get("model_spec_id", ""),
                "base_model_family": metadata.get("base_model_family", ""),
                "model_kind": metadata.get("model_kind", ""),
                "model_params": metadata.get("model_params", ""),
                "status": "error",
                "error": block.get("error", ""),
                "inner_fold_count": 0,
            })
            continue

        sample_index = np.asarray(block.get("sample_index", []))
        y_values = np.asarray(block.get("y_true", []), dtype=int)
        score_values = np.asarray(block.get("score", []), dtype=float)
        if not (len(sample_index) == len(y_values) == len(score_values)):
            raise ValueError("OOF block arrays must have equal lengths")
        if len(sample_index) == 0:
            continue

        context_labels = labels_by_context.setdefault((outer_fold, round_name), {})
        for sample, label in zip(sample_index, y_values, strict=True):
            sample_key = _python_scalar(sample)
            previous = context_labels.get(sample_key)
            if previous is not None and previous != int(label):
                raise ValueError(f"conflicting y_true values for sample_index={sample_key}")
            context_labels[sample_key] = int(label)

        key = (outer_fold, round_name, pipeline_recipe_id)
        feature_size = _normalize_feature_size(metadata.get("feature_size"))
        state = states.setdefault(
            key,
            {
                "sample_index": [],
                "y_true": [],
                "score": [],
                "inner_folds": set(),
                "feature_size": feature_size,
                "metadata": metadata,
                "duration_seconds_sum": 0.0,
                "duration_seconds_count": 0,
            },
        )
        if int(state["feature_size"]) != feature_size:
            raise ValueError(f"inconsistent feature_size values for {pipeline_recipe_id}")
        state["sample_index"].append(sample_index)
        state["y_true"].append(y_values)
        state["score"].append(score_values)
        state["inner_folds"].add(inner_fold)
        state["duration_seconds_sum"] += float(block.get("duration_seconds", 0.0) or 0.0)
        state["duration_seconds_count"] += 1

    score_rows: list[dict[str, Any]] = []
    baseline_frames: list[pd.DataFrame] = []
    rng = np.random.default_rng(random_state)

    contexts = sorted({(outer, round_name) for outer, round_name, _pipeline in states})
    contexts.extend(context for context in labels_by_context if context not in set(contexts))
    for outer_fold, round_name in contexts:
        expected_samples, expected_inner_folds, expected_labels = _expected_inner_oof_context(
            inner_assignments,
            int(outer_fold),
            inner_folds=inner_folds,
            y=y,
        )
        labels_by_sample = expected_labels or labels_by_context.get((outer_fold, round_name), {})
        if expected_samples is None:
            expected_samples = set(labels_by_sample)
        if expected_inner_folds is None:
            expected_inner_folds = set()

        missing_labels = expected_samples.difference(labels_by_sample)
        if missing_labels:
            raise ValueError(
                "labels missing for expected manifest samples: "
                f"outer_fold={outer_fold}, examples={sorted(missing_labels)[:5]}"
            )

        if include_baseline_comparison:
            baseline_frame = pd.DataFrame(
                _baseline_rows_from_inner_oof_labels(
                    labels_by_sample,
                    outer_fold=int(outer_fold),
                    round_name=str(round_name),
                    max_targets=max_targets,
                    random_repeats=random_repeats,
                    rng=rng,
                )
            )
            baseline_frames.append(baseline_frame)

    for (outer_fold, round_name, pipeline_recipe_id), state in states.items():
        sample_index = np.concatenate(state["sample_index"])
        duplicate_samples = pd.Series(sample_index).duplicated(keep=False)
        if bool(duplicate_samples.any()):
            examples = pd.Series(sample_index).loc[duplicate_samples].head(5).tolist()
            raise ValueError(
                "duplicate inner-OOF predictions for the same recipe/sample in blocks: "
                f"outer_fold={outer_fold}, round_name={round_name}, "
                f"pipeline_recipe_id={pipeline_recipe_id}, examples={examples}"
            )

        expected_samples, expected_inner_folds, _expected_labels = _expected_inner_oof_context(
            inner_assignments,
            int(outer_fold),
            inner_folds=inner_folds,
            y=y,
        )
        if expected_samples is None:
            expected_samples = set(labels_by_context.get((outer_fold, round_name), {}))
        if expected_inner_folds is None:
            expected_inner_folds = set(state["inner_folds"])

        group_samples = set(sample_index.tolist())
        extra_samples = group_samples.difference(expected_samples)
        if extra_samples:
            raise ValueError(
                "OOF blocks contain samples outside the expected inner-OOF manifest: "
                f"outer_fold={outer_fold}, examples={sorted(extra_samples)[:5]}"
            )
        observed_folds = {int(fold) for fold in state["inner_folds"]}
        extra_folds = observed_folds.difference(expected_inner_folds)
        if extra_folds:
            raise ValueError(
                "OOF blocks contain inner folds outside the expected manifest: "
                f"outer_fold={outer_fold}, examples={sorted(extra_folds)[:5]}"
            )
        missing_samples = expected_samples.difference(group_samples)
        missing_folds = expected_inner_folds.difference(observed_folds)
        status = "ok" if not missing_samples and not missing_folds else "incomplete"

        y_values = np.concatenate(state["y_true"])
        score_values = np.concatenate(state["score"])
        metadata = state["metadata"]
        feature_size = int(state["feature_size"])
        selected_features_text = metadata.get("selected_features_text", "")
        if not selected_features_text:
            selected_features_text = ",".join(_prediction_group_features(pd.DataFrame([metadata])))
        scored = _score_pooled_inner_oof(
            y_values,
            score_values,
            feature_size=feature_size,
            method=f"inner_oof__outer{int(outer_fold):02d}__{round_name}__{pipeline_recipe_id}",
            max_targets=max_targets,
        )
        n_predictions = int(scored.get("oof_n_predictions", len(y_values)))
        optimal_k = int(scored.get("oof_optimal_k", 0))
        score_rows.append({
            "outer_fold": int(outer_fold),
            "round_name": round_name,
            "pipeline_recipe_id": pipeline_recipe_id,
            "feature_recipe_id": metadata.get("feature_recipe_id", ""),
            "prescreen_recipe_id": metadata.get("prescreen_recipe_id", ""),
            "prescreen_name": metadata.get("prescreen_name", ""),
            "prescreen_methods": metadata.get("prescreen_methods", ""),
            "rank_aggregation": metadata.get("rank_aggregation", ""),
            "feature_size": feature_size,
            "selected_features_text": selected_features_text,
            "candidate_heuristic_score": metadata.get("candidate_heuristic_score", np.nan),
            "model_spec_id": metadata.get("model_spec_id", ""),
            "base_model_family": metadata.get("base_model_family", ""),
            "model_kind": metadata.get("model_kind", ""),
            "model_params": metadata.get("model_params", ""),
            "inner_oof_business_score": scored.get("oof_business_score", np.nan),
            "inner_oof_optimal_k": optimal_k,
            "inner_oof_f1_score": scored.get("f1_score", np.nan),
            "inner_oof_roc_auc_score": scored.get("roc_auc_score", np.nan),
            "inner_oof_n_predictions": n_predictions,
            "inner_oof_selected_rate": float(optimal_k / n_predictions) if n_predictions else 0.0,
            "inner_oof_tp_at_optimal_k": scored.get("oof_tp_at_optimal_k", np.nan),
            "inner_oof_fp_at_optimal_k": scored.get("oof_fp_at_optimal_k", np.nan),
            "feature_penalty": scored.get("feature_penalty", float(200 * feature_size)),
            "gross_score": scored.get("gross_score", np.nan),
            "expected_inner_oof_predictions": len(expected_samples),
            "missing_inner_oof_predictions": len(missing_samples),
            "inner_fold_count": len(observed_folds),
            "expected_inner_fold_count": len(expected_inner_folds),
            "missing_inner_fold_count": len(missing_folds),
            "missing_inner_folds": ",".join(map(str, sorted(missing_folds))),
            "mean_duration_seconds": (
                float(state["duration_seconds_sum"] / state["duration_seconds_count"])
                if state["duration_seconds_count"]
                else np.nan
            ),
            "manifest_checked": inner_assignments is not None,
            "status": status,
        })

    score_frame = pd.DataFrame([*score_rows, *error_rows])
    baseline_frame = (
        pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    )
    if include_baseline_comparison and not score_frame.empty and not baseline_frame.empty:
        score_frame = _merge_baseline_scores(score_frame, baseline_frame)
    if not score_frame.empty and "inner_oof_business_score" in score_frame.columns:
        score_frame = score_frame.sort_values(
            ["inner_oof_business_score", "inner_oof_roc_auc_score", "feature_size"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
    return score_frame, baseline_frame


INNER_PREDICTION_REQUIRED_COLUMNS = frozenset({
    "outer_fold",
    "inner_fold",
    "round_name",
    "pipeline_recipe_id",
    "sample_index",
    "y_true",
    "score",
    "feature_size",
})


def has_inner_prediction_columns(path: Path | str) -> bool:
    """Return whether a CSV has the minimum columns needed for inner-OOF scoring."""
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    return INNER_PREDICTION_REQUIRED_COLUMNS.issubset(columns)


def _inner_prediction_csv_contexts(
    path: Path,
    *,
    chunksize: int,
    outer_folds: Sequence[int] | None,
    round_names: Sequence[str] | None,
    log_every_chunks: int,
    verbose: bool,
) -> list[tuple[int, str]]:
    allowed_outer_folds = {int(fold) for fold in outer_folds} if outer_folds is not None else None
    allowed_round_names = {str(name) for name in round_names} if round_names is not None else None
    if allowed_outer_folds is not None and allowed_round_names is not None:
        contexts = sorted(
            (outer_fold, round_name)
            for outer_fold in allowed_outer_folds
            for round_name in allowed_round_names
        )
        if verbose:
            print(
                f"[inner-oof-csv] using explicit contexts from outer_folds/round_names: {contexts}",
                flush=True,
            )
        return contexts

    contexts: set[tuple[int, str]] = set()
    started = time.perf_counter()
    if verbose:
        print(
            f"[inner-oof-csv] scanning contexts in {path} with chunksize={chunksize}",
            flush=True,
        )
    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=["outer_fold", "round_name"], chunksize=chunksize),
        start=1,
    ):
        raw_rows = len(chunk)
        if allowed_outer_folds is not None:
            chunk = chunk.loc[chunk["outer_fold"].isin(allowed_outer_folds)]
        if allowed_round_names is not None:
            chunk = chunk.loc[chunk["round_name"].isin(allowed_round_names)]
        if not chunk.empty:
            contexts.update(
                (int(row.outer_fold), str(row.round_name))
                for row in chunk.drop_duplicates().itertuples(index=False)
            )
        if verbose and log_every_chunks > 0 and chunk_index % log_every_chunks == 0:
            print(
                f"[inner-oof-csv] context scan chunks={chunk_index} "
                f"last_chunk_rows={raw_rows} contexts_found={len(contexts)} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    if verbose:
        print(
            f"[inner-oof-csv] context scan done "
            f"chunks={chunk_index if 'chunk_index' in locals() else 0} "
            f"contexts_found={len(contexts)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return sorted(contexts)


def _score_inner_oof_context_from_csv(
    path: Path,
    *,
    outer_fold: int,
    round_name: str,
    columns: set[str],
    chunksize: int,
    max_targets: int,
    random_repeats: int,
    random_state: int,
    expected_samples: set[Any] | None,
    expected_inner_folds: set[int] | None,
    expected_labels: dict[Any, int] | None,
    log_every_chunks: int,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_columns = [
        "inner_fold",
        "feature_recipe_id",
        "prescreen_recipe_id",
        "prescreen_name",
        "prescreen_methods",
        "rank_aggregation",
        "feature_size",
        "selected_features",
        "selected_features_text",
        "candidate_heuristic_score",
        "model_spec_id",
        "base_model_family",
        "model_kind",
        "model_params",
        "status",
        "error",
        "duration_seconds",
    ]
    usecols = sorted(
        INNER_PREDICTION_REQUIRED_COLUMNS.union(set(metadata_columns)).intersection(columns)
    )
    states: dict[str, dict[str, Any]] = {}
    labels_by_sample: dict[Any, int] = dict(expected_labels or {})
    manifest_checked = expected_samples is not None and expected_inner_folds is not None
    started = time.perf_counter()
    raw_rows_seen = 0
    matched_rows_seen = 0
    if verbose:
        print(
            f"[inner-oof-csv] start outer_fold={outer_fold} round={round_name} "
            f"chunksize={chunksize}",
            flush=True,
        )

    for chunk_index, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1
    ):
        raw_rows_seen += len(chunk)
        chunk = chunk.loc[
            chunk["outer_fold"].eq(int(outer_fold)) & chunk["round_name"].eq(str(round_name))
        ]
        if "status" in chunk.columns:
            chunk = chunk.loc[chunk["status"].eq("ok")]
        chunk = chunk.loc[
            chunk["sample_index"].notna()
            & chunk["y_true"].notna()
            & chunk["score"].notna()
            & chunk["feature_size"].notna()
            & chunk["inner_fold"].notna()
        ]
        if chunk.empty:
            if verbose and log_every_chunks > 0 and chunk_index % log_every_chunks == 0:
                print(
                    f"[inner-oof-csv] outer_fold={outer_fold} round={round_name} "
                    f"chunks={chunk_index} raw_rows={raw_rows_seen} "
                    f"matched_rows={matched_rows_seen} recipes={len(states)} "
                    f"samples={len(labels_by_sample)} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            continue

        matched_rows_seen += len(chunk)
        for sample_index, y_true in (
            chunk[["sample_index", "y_true"]].drop_duplicates().itertuples(index=False)
        ):
            sample_key = _python_scalar(sample_index)
            label = int(y_true)
            previous = labels_by_sample.get(sample_key)
            if previous is not None and previous != label:
                raise ValueError(
                    "conflicting y_true values in inner-OOF prediction CSV for "
                    f"outer_fold={outer_fold}, round_name={round_name}, sample_index={sample_key}"
                )
            labels_by_sample[sample_key] = label

        for pipeline_recipe_id, group in chunk.groupby("pipeline_recipe_id", sort=False):
            key = str(pipeline_recipe_id)
            group_feature_size = _consistent_feature_size(group)
            state = states.setdefault(
                key,
                {
                    "sample_index": [],
                    "y_true": [],
                    "score": [],
                    "inner_folds": set(),
                    "feature_size": group_feature_size,
                    "metadata": {},
                    "duration_seconds_sum": 0.0,
                    "duration_seconds_count": 0,
                },
            )
            if int(state["feature_size"]) != group_feature_size:
                raise ValueError(
                    "inconsistent feature_size values in CSV for "
                    f"pipeline_recipe_id={pipeline_recipe_id}"
                )
            state["sample_index"].append(group["sample_index"].to_numpy())
            state["y_true"].append(group["y_true"].to_numpy(dtype=int))
            state["score"].append(group["score"].to_numpy(dtype=float))
            state["inner_folds"].update(group["inner_fold"].dropna().astype(int).tolist())
            if "duration_seconds" in group.columns:
                durations = group["duration_seconds"].dropna().astype(float)
                state["duration_seconds_sum"] += float(durations.sum())
                state["duration_seconds_count"] += len(durations)
            if not state["metadata"]:
                row = group.iloc[0]
                state["metadata"] = {
                    column: row[column]
                    for column in metadata_columns
                    if column in group.columns
                    and column not in {"status", "error", "duration_seconds"}
                }
        if verbose and log_every_chunks > 0 and chunk_index % log_every_chunks == 0:
            print(
                f"[inner-oof-csv] outer_fold={outer_fold} round={round_name} "
                f"chunks={chunk_index} raw_rows={raw_rows_seen} "
                f"matched_rows={matched_rows_seen} recipes={len(states)} "
                f"samples={len(labels_by_sample)} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if verbose:
        print(
            f"[inner-oof-csv] finished reading outer_fold={outer_fold} round={round_name} "
            f"chunks={chunk_index if 'chunk_index' in locals() else 0} raw_rows={raw_rows_seen} "
            f"matched_rows={matched_rows_seen} recipes={len(states)} "
            f"samples={len(labels_by_sample)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    if expected_samples is None:
        expected_samples = set(labels_by_sample)
    if expected_inner_folds is None:
        expected_inner_folds = (
            set().union(*(state["inner_folds"] for state in states.values())) if states else set()
        )

    missing_labels = expected_samples.difference(labels_by_sample)
    if missing_labels:
        raise ValueError(
            "labels missing for expected manifest samples: "
            f"outer_fold={outer_fold}, examples={sorted(missing_labels)[:5]}"
        )

    score_rows: list[dict[str, Any]] = []
    for pipeline_recipe_id, state in states.items():
        sample_index = np.concatenate(state["sample_index"])
        duplicate_samples = pd.Series(sample_index).duplicated(keep=False)
        if bool(duplicate_samples.any()):
            examples = pd.Series(sample_index).loc[duplicate_samples].head(5).tolist()
            raise ValueError(
                "duplicate inner-OOF predictions for the same recipe/sample in CSV: "
                f"outer_fold={outer_fold}, round_name={round_name}, "
                f"pipeline_recipe_id={pipeline_recipe_id}, examples={examples}"
            )

        y_values = np.concatenate(state["y_true"])
        score_values = np.concatenate(state["score"])
        group_samples = set(sample_index.tolist())
        extra_samples = group_samples.difference(expected_samples)
        if extra_samples:
            raise ValueError(
                "prediction CSV contains samples outside the expected inner-OOF manifest: "
                f"outer_fold={outer_fold}, examples={sorted(extra_samples)[:5]}"
            )
        observed_folds = {int(fold) for fold in state["inner_folds"]}
        extra_folds = observed_folds.difference(expected_inner_folds)
        if extra_folds:
            raise ValueError(
                "prediction CSV contains inner folds outside the expected manifest: "
                f"outer_fold={outer_fold}, examples={sorted(extra_folds)[:5]}"
            )
        missing_samples = expected_samples.difference(group_samples)
        missing_folds = expected_inner_folds.difference(observed_folds)
        status = "ok" if not missing_samples and not missing_folds else "incomplete"
        metadata = state["metadata"]
        feature_size = int(state["feature_size"])
        selected_features_text = metadata.get("selected_features_text", "")
        if not selected_features_text:
            selected_features_text = ",".join(_prediction_group_features(pd.DataFrame([metadata])))
        scored = _score_pooled_inner_oof(
            y_values,
            score_values,
            feature_size=feature_size,
            method=f"inner_oof__outer{int(outer_fold):02d}__{round_name}__{pipeline_recipe_id}",
            max_targets=max_targets,
        )
        n_predictions = int(scored.get("oof_n_predictions", len(y_values)))
        optimal_k = int(scored.get("oof_optimal_k", 0))
        score_rows.append({
            "outer_fold": int(outer_fold),
            "round_name": round_name,
            "pipeline_recipe_id": pipeline_recipe_id,
            "feature_recipe_id": metadata.get("feature_recipe_id", ""),
            "prescreen_recipe_id": metadata.get("prescreen_recipe_id", ""),
            "prescreen_name": metadata.get("prescreen_name", ""),
            "prescreen_methods": metadata.get("prescreen_methods", ""),
            "rank_aggregation": metadata.get("rank_aggregation", ""),
            "feature_size": feature_size,
            "selected_features_text": selected_features_text,
            "candidate_heuristic_score": metadata.get("candidate_heuristic_score", np.nan),
            "model_spec_id": metadata.get("model_spec_id", ""),
            "base_model_family": metadata.get("base_model_family", ""),
            "model_kind": metadata.get("model_kind", ""),
            "model_params": metadata.get("model_params", ""),
            "inner_oof_business_score": scored.get("oof_business_score", np.nan),
            "inner_oof_optimal_k": optimal_k,
            "inner_oof_f1_score": scored.get("f1_score", np.nan),
            "inner_oof_roc_auc_score": scored.get("roc_auc_score", np.nan),
            "inner_oof_n_predictions": n_predictions,
            "inner_oof_selected_rate": float(optimal_k / n_predictions) if n_predictions else 0.0,
            "inner_oof_tp_at_optimal_k": scored.get("oof_tp_at_optimal_k", np.nan),
            "inner_oof_fp_at_optimal_k": scored.get("oof_fp_at_optimal_k", np.nan),
            "feature_penalty": scored.get("feature_penalty", float(200 * feature_size)),
            "gross_score": scored.get("gross_score", np.nan),
            "expected_inner_oof_predictions": len(expected_samples),
            "missing_inner_oof_predictions": len(missing_samples),
            "inner_fold_count": len(observed_folds),
            "expected_inner_fold_count": len(expected_inner_folds),
            "missing_inner_fold_count": len(missing_folds),
            "missing_inner_folds": ",".join(map(str, sorted(missing_folds))),
            "mean_duration_seconds": (
                float(state["duration_seconds_sum"] / state["duration_seconds_count"])
                if state["duration_seconds_count"]
                else np.nan
            ),
            "manifest_checked": manifest_checked,
            "status": status,
        })

    score_frame = pd.DataFrame(score_rows)
    baseline_frame = pd.DataFrame(
        _baseline_rows_from_inner_oof_labels(
            labels_by_sample,
            outer_fold=outer_fold,
            round_name=round_name,
            max_targets=max_targets,
            random_repeats=random_repeats,
            rng=np.random.default_rng(random_state),
        )
    )
    score_frame = _merge_baseline_scores(score_frame, baseline_frame)
    if not score_frame.empty:
        score_frame = score_frame.sort_values(
            ["inner_oof_business_score", "inner_oof_roc_auc_score", "feature_size"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    if verbose:
        ok_count = int(score_frame["status"].eq("ok").sum()) if not score_frame.empty else 0
        print(
            f"[inner-oof-csv] scored outer_fold={outer_fold} round={round_name} "
            f"recipes={len(score_frame)} ok={ok_count} baselines={len(baseline_frame)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return score_frame, baseline_frame


def score_inner_oof_predictions_csv(
    prediction_path: Path | str,
    *,
    max_targets: int = DEFAULT_MAX_TARGETS,
    chunksize: int = 1_000_000,
    random_repeats: int = 0,
    random_state: int = 42,
    outer_folds: Sequence[int] | None = None,
    round_names: Sequence[str] | None = None,
    output_path: Path | str | None = None,
    baseline_output_path: Path | str | None = None,
    overwrite: bool = True,
    inner_assignments: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    inner_folds: Sequence[int] | None = None,
    log_every_chunks: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Score a large inner-prediction CSV without loading it fully into memory."""
    path = Path(prediction_path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    started = time.perf_counter()
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = INNER_PREDICTION_REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(
            f"{path} is not a sample-level prediction CSV; missing columns: {sorted(missing)}"
        )

    output_csv = Path(output_path) if output_path is not None else None
    baseline_csv = Path(baseline_output_path) if baseline_output_path is not None else None
    if overwrite:
        for csv_path in (output_csv, baseline_csv):
            if csv_path is not None and csv_path.exists():
                csv_path.unlink()

    if verbose:
        print(
            f"[inner-oof-csv] scoring file={path} "
            f"size_gb={path.stat().st_size / 1_000_000_000:.2f} "
            f"chunksize={chunksize} max_targets={max_targets}",
            flush=True,
        )
    effective_outer_folds = outer_folds
    if effective_outer_folds is None and inner_assignments is not None:
        effective_outer_folds = sorted(
            inner_assignments["outer_fold"].dropna().astype(int).unique()
        )

    contexts = _inner_prediction_csv_contexts(
        path,
        chunksize=chunksize,
        outer_folds=effective_outer_folds,
        round_names=round_names,
        log_every_chunks=log_every_chunks,
        verbose=verbose,
    )
    if verbose:
        print(f"[inner-oof-csv] contexts={contexts}", flush=True)
    score_frames: list[pd.DataFrame] = []
    for context_index, (outer_fold, round_name) in enumerate(contexts):
        if verbose:
            print(
                f"[inner-oof-csv] context {context_index + 1}/{len(contexts)} "
                f"outer_fold={outer_fold} round={round_name}",
                flush=True,
            )
        expected_samples, expected_inner_folds, expected_labels = _expected_inner_oof_context(
            inner_assignments,
            int(outer_fold),
            inner_folds=inner_folds,
            y=y,
        )
        score_frame, baseline_frame = _score_inner_oof_context_from_csv(
            path,
            outer_fold=outer_fold,
            round_name=round_name,
            columns=columns,
            chunksize=chunksize,
            max_targets=max_targets,
            random_repeats=random_repeats,
            random_state=random_state + context_index,
            expected_samples=expected_samples,
            expected_inner_folds=expected_inner_folds,
            expected_labels=expected_labels,
            log_every_chunks=log_every_chunks,
            verbose=verbose,
        )
        if output_csv is not None:
            append_csv(score_frame, output_csv)
            if verbose:
                print(f"[inner-oof-csv] appended scores to {output_csv}", flush=True)
        if baseline_csv is not None:
            append_csv(baseline_frame, baseline_csv)
            if verbose:
                print(f"[inner-oof-csv] appended baselines to {baseline_csv}", flush=True)
        score_frames.append(score_frame)

    if not score_frames:
        return pd.DataFrame()
    result = pd.concat(score_frames, ignore_index=True)
    if verbose:
        print(
            f"[inner-oof-csv] all contexts done rows={len(result)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return result


def aggregate_inner_scores(
    scores: pd.DataFrame,
    *,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    """Aggregate inner validation scores across folds or inner-OOF recipe scores."""
    if scores.empty:
        return pd.DataFrame()
    ok_scores = (
        scores.loc[scores["status"].eq("ok")].copy() if "status" in scores else scores.copy()
    )
    if ok_scores.empty:
        return pd.DataFrame()

    if "inner_oof_business_score" in ok_scores.columns:
        for completeness_col in ("missing_inner_oof_predictions", "missing_inner_fold_count"):
            if completeness_col in ok_scores.columns:
                ok_scores = ok_scores.loc[
                    pd.to_numeric(ok_scores[completeness_col], errors="coerce").eq(0)
                ].copy()
        if {"inner_fold_count", "expected_inner_fold_count"}.issubset(ok_scores.columns):
            observed_fold_count = pd.to_numeric(ok_scores["inner_fold_count"], errors="coerce")
            expected_fold_count = pd.to_numeric(
                ok_scores["expected_inner_fold_count"], errors="coerce"
            )
            ok_scores = ok_scores.loc[observed_fold_count.ge(expected_fold_count)].copy()
        if ok_scores.empty:
            return pd.DataFrame()

        aggregations: dict[str, tuple[str, str]] = {
            "inner_oof_business_score": ("inner_oof_business_score", "mean"),
            "inner_oof_optimal_k": ("inner_oof_optimal_k", "mean"),
            "inner_oof_f1_score": ("inner_oof_f1_score", "mean"),
            "inner_oof_roc_auc_score": ("inner_oof_roc_auc_score", "mean"),
            "inner_oof_n_predictions": ("inner_oof_n_predictions", "min"),
            "inner_oof_selected_rate": ("inner_oof_selected_rate", "mean"),
            "selected_features_text": ("selected_features_text", "first"),
            "feature_size": ("feature_size", "first"),
            "prescreen_name": ("prescreen_name", "first"),
            "base_model_family": ("base_model_family", "first"),
            "model_spec_id": ("model_spec_id", "first"),
        }
        optional_aggregations: dict[str, tuple[str, str]] = {
            "inner_fold_count": ("inner_fold_count", "min"),
            "expected_inner_oof_predictions": ("expected_inner_oof_predictions", "min"),
            "missing_inner_oof_predictions": ("missing_inner_oof_predictions", "max"),
            "expected_inner_fold_count": ("expected_inner_fold_count", "min"),
            "missing_inner_fold_count": ("missing_inner_fold_count", "max"),
            "manifest_checked": ("manifest_checked", "min"),
            "feature_recipe_id": ("feature_recipe_id", "first"),
            "pipeline_recipe_id": ("pipeline_recipe_id", "first"),
            "prescreen_recipe_id": ("prescreen_recipe_id", "first"),
            "prescreen_methods": ("prescreen_methods", "first"),
            "rank_aggregation": ("rank_aggregation", "first"),
            "model_kind": ("model_kind", "first"),
            "model_params": ("model_params", "first"),
            "feature_penalty": ("feature_penalty", "mean"),
            "gross_score": ("gross_score", "mean"),
            "inner_oof_tp_at_optimal_k": ("inner_oof_tp_at_optimal_k", "mean"),
            "inner_oof_fp_at_optimal_k": ("inner_oof_fp_at_optimal_k", "mean"),
            "baseline_select_all_capped_features_0_business_score": (
                "baseline_select_all_capped_features_0_business_score",
                "first",
            ),
            "baseline_select_all_capped_features_1_business_score": (
                "baseline_select_all_capped_features_1_business_score",
                "first",
            ),
            "baseline_select_all_illegal_full_features_0_business_score": (
                "baseline_select_all_illegal_full_features_0_business_score",
                "first",
            ),
            "baseline_select_all_illegal_full_features_1_business_score": (
                "baseline_select_all_illegal_full_features_1_business_score",
                "first",
            ),
            "beats_baseline_select_all_capped_features_0": (
                "beats_baseline_select_all_capped_features_0",
                "mean",
            ),
            "beats_baseline_select_all_capped_features_1": (
                "beats_baseline_select_all_capped_features_1",
                "mean",
            ),
            "beats_baseline_select_all_illegal_full_features_0": (
                "beats_baseline_select_all_illegal_full_features_0",
                "mean",
            ),
            "beats_baseline_select_all_illegal_full_features_1": (
                "beats_baseline_select_all_illegal_full_features_1",
                "mean",
            ),
        }
        for output_col, spec in optional_aggregations.items():
            source_col, _func = spec
            if source_col in ok_scores.columns and output_col not in group_cols:
                aggregations[output_col] = spec

        grouped = (
            ok_scores.groupby(list(group_cols), as_index=False)
            .agg(**aggregations)
            .sort_values(
                ["inner_oof_business_score", "inner_oof_roc_auc_score", "feature_size"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
        grouped["inner_selection_rank"] = np.arange(1, len(grouped) + 1)
        return grouped

    aggregations = {
        "inner_fold_count": ("inner_fold", "nunique"),
        "mean_inner_business_score": ("inner_business_score", "mean"),
        "median_inner_business_score": ("inner_business_score", "median"),
        "min_inner_business_score": ("inner_business_score", "min"),
        "max_inner_business_score": ("inner_business_score", "max"),
        "std_inner_business_score": ("inner_business_score", "std"),
        "mean_inner_optimal_k": ("inner_optimal_k", "mean"),
        "selected_features_text": ("selected_features_text", "first"),
        "feature_size": ("feature_size", "first"),
        "prescreen_name": ("prescreen_name", "first"),
        "base_model_family": ("base_model_family", "first"),
        "model_spec_id": ("model_spec_id", "first"),
    }
    optional_aggregations: dict[str, tuple[str, str]] = {
        "scoring_version": ("scoring_version", "first"),
        "mean_inner_evaluation_row_count": ("inner_evaluation_row_count", "mean"),
        "mean_inner_feature_cost_scale": ("inner_feature_cost_scale", "mean"),
        "mean_inner_scaled_true_positive_value": (
            "inner_scaled_true_positive_value",
            "mean",
        ),
        "mean_inner_scaled_false_positive_cost": (
            "inner_scaled_false_positive_cost",
            "mean",
        ),
        "mean_inner_scaled_feature_cost": ("inner_scaled_feature_cost", "mean"),
        "mean_inner_unscaled_feature_cost": ("inner_unscaled_feature_cost", "mean"),
        "mean_inner_business_value_before_feature_cost": (
            "inner_business_value_before_feature_cost",
            "mean",
        ),
        "mean_inner_business_score_unscaled_feature_cost": (
            "inner_business_score_unscaled_feature_cost",
            "mean",
        ),
        "mean_inner_tp_at_optimal_k": ("inner_tp_at_optimal_k", "mean"),
        "mean_inner_fp_at_optimal_k": ("inner_fp_at_optimal_k", "mean"),
    }
    for output_col, spec in optional_aggregations.items():
        source_col, _func = spec
        if source_col in ok_scores.columns:
            aggregations[output_col] = spec

    grouped = (
        ok_scores.groupby(list(group_cols), as_index=False)
        .agg(**aggregations)
        .sort_values(
            [
                "mean_inner_business_score",
                "median_inner_business_score",
                "min_inner_business_score",
                "feature_size",
            ],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )
    grouped["inner_selection_rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def select_top_feature_recipes(
    round1_scores: pd.DataFrame,
    *,
    top_n: int,
    min_inner_fold_count: int = 1,
) -> pd.DataFrame:
    """Select feature recipes after round-1 inner CV screening."""
    group_cols: list[str] = ["outer_fold"]
    if "round_name" in round1_scores.columns:
        group_cols.append("round_name")
    group_cols.extend(["prescreen_recipe_id", "feature_size"])
    grouped = aggregate_inner_scores(round1_scores, group_cols=tuple(group_cols))
    if grouped.empty:
        return grouped
    grouped = grouped.loc[grouped["inner_fold_count"] >= min_inner_fold_count].copy()
    return grouped.head(top_n).reset_index(drop=True)


def select_top_pipeline_recipes(
    round2_scores: pd.DataFrame,
    *,
    top_n: int,
    min_inner_fold_count: int = 1,
) -> pd.DataFrame:
    """Select full pipeline recipes after round-2 model/HPO inner CV."""
    group_cols: list[str] = ["outer_fold"]
    if "round_name" in round2_scores.columns:
        group_cols.append("round_name")
    if "pipeline_recipe_id" in round2_scores.columns:
        group_cols.append("pipeline_recipe_id")
    else:
        group_cols.extend([
            "prescreen_recipe_id",
            "feature_size",
            "model_spec_id",
        ])
    grouped = aggregate_inner_scores(round2_scores, group_cols=tuple(group_cols))
    if grouped.empty:
        return grouped
    grouped = grouped.loc[grouped["inner_fold_count"] >= min_inner_fold_count].copy()
    return grouped.head(top_n).reset_index(drop=True)


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    """Append a frame to CSV with stable header behavior."""
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def filter_model_specs(
    model_specs: pd.DataFrame,
    model_spec_ids: Sequence[str] | None = None,
    *,
    max_specs: int | None = None,
) -> pd.DataFrame:
    """Filter model specs by explicit ids and optional head cap."""
    out = model_specs.copy()
    if model_spec_ids:
        out = out.loc[out["model_spec_id"].isin(model_spec_ids)].copy()
    if max_specs is not None:
        out = out.head(max_specs).copy()
    return out.reset_index(drop=True)


def filter_prescreen_recipes(
    prescreen_recipes: pd.DataFrame,
    prescreen_recipe_ids: Sequence[str] | None = None,
    *,
    include_adaptive: bool = True,
    max_recipes: int | None = None,
) -> pd.DataFrame:
    """Filter prescreen recipes for execution."""
    out = prescreen_recipes.copy()
    if not include_adaptive and "adaptive" in out:
        out = out.loc[~out["adaptive"].astype(bool)].copy()
    if prescreen_recipe_ids:
        out = out.loc[out["prescreen_recipe_id"].isin(prescreen_recipe_ids)].copy()
    if max_recipes is not None:
        out = out.head(max_recipes).copy()
    return out.reset_index(drop=True)


def _inner_score_row(
    candidate: pd.Series,
    model_spec: pd.Series,
    scored: dict[str, Any],
    *,
    outer_fold: int,
    inner_fold: int,
    round_name: str,
    pipeline_recipe_id: str,
    status: str,
    error: str,
    duration_seconds: float,
) -> dict[str, Any]:
    return {
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "round_name": round_name,
        "pipeline_recipe_id": pipeline_recipe_id,
        "feature_recipe_id": candidate.get("feature_recipe_id", ""),
        "prescreen_recipe_id": candidate.get("prescreen_recipe_id", ""),
        "prescreen_name": candidate.get("prescreen_name", ""),
        "prescreen_methods": candidate.get("prescreen_methods", ""),
        "rank_aggregation": candidate.get("rank_aggregation", ""),
        "feature_size": int(candidate.get("feature_size", 0)),
        "selected_features": scored.get("source_features", candidate.get("selected_features", [])),
        "selected_features_text": ",".join(
            scored.get("source_features", candidate.get("selected_features", []))
        ),
        "candidate_heuristic_score": candidate.get("candidate_heuristic_score", np.nan),
        "model_spec_id": model_spec.get("model_spec_id", ""),
        "base_model_family": model_spec.get("base_model_family", ""),
        "model_kind": model_spec.get("model_kind", ""),
        "model_params": model_spec.get("model_params", ""),
        "status": status,
        "error": error,
        "scoring_version": scored.get("scoring_version", ""),
        "inner_evaluation_row_count": scored.get("evaluation_row_count", np.nan),
        "inner_target_limit": scored.get("target_limit", np.nan),
        "inner_feature_cost_reference_row_count": scored.get(
            "feature_cost_reference_row_count", np.nan
        ),
        "inner_feature_cost_scale": scored.get("feature_cost_scale", np.nan),
        "inner_scaled_true_positive_value": scored.get("scaled_true_positive_value", np.nan),
        "inner_scaled_false_positive_cost": scored.get("scaled_false_positive_cost", np.nan),
        "inner_scaled_feature_cost": scored.get("scaled_feature_cost", np.nan),
        "inner_unscaled_feature_cost": scored.get("unscaled_feature_cost", np.nan),
        "inner_business_value_before_feature_cost": scored.get(
            "oof_business_value_before_feature_cost", np.nan
        ),
        "inner_business_score": scored.get("oof_business_score", np.nan),
        "inner_business_score_unscaled_feature_cost": scored.get(
            "oof_business_score_unscaled_feature_cost", np.nan
        ),
        "inner_optimal_k": scored.get("oof_optimal_k", np.nan),
        "inner_tp_at_optimal_k": scored.get("oof_tp_at_optimal_k", np.nan),
        "inner_fp_at_optimal_k": scored.get("oof_fp_at_optimal_k", np.nan),
        "inner_f1_score": scored.get("f1_score", np.nan),
        "inner_roc_auc_score": scored.get("roc_auc_score", np.nan),
        "duration_seconds": float(duration_seconds),
    }
