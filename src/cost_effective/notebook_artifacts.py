"""Cache-aware artifact helpers for experiment notebooks."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

LEGACY_OUTPUT_ALIASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("outputs", "feature_selection_alternative", "01_fold_plan_and_recipe_space"): (
        "outputs",
        "honest_nested_selection",
        "01_outer_folds_and_recipe_space",
    ),
    ("outputs", "feature_selection_alternative", "02_inner_recipe_selection"): (
        "outputs",
        "honest_nested_selection",
        "02_inner_cv_recipe_selection",
    ),
    ("outputs", "feature_selection_alternative", "03_fold_evaluation"): (
        "outputs",
        "honest_nested_selection",
        "03_outer_fold_evaluation",
    ),
    ("outputs", "feature_selection_alternative", "04_refit_and_test_ranking"): (
        "outputs",
        "honest_nested_selection",
        "04_final_refit_and_test_ranking",
    ),
    ("outputs", "feature_selection_prescreening_ensemble"): (
        "outputs",
        "feature_selection_broad_ensemble",
    ),
    ("outputs", "modeling_feature_set_optimization"): (
        "outputs",
        "feature_selection_model_family_optimization",
    ),
    ("outputs", "modeling_feature_set_validation"): (
        "outputs",
        "feature_set_model_family_validation",
    ),
    ("outputs", "modeling_rank_fusion_audit"): (
        "outputs",
        "final_repeated_oof_rank_fusion_audit",
    ),
    ("outputs", "nested_cv_selection", "01_outer_folds_and_recipe_space"): (
        "outputs",
        "honest_nested_selection",
        "01_outer_folds_and_recipe_space",
    ),
    ("outputs", "nested_cv_selection", "02_inner_cv_recipe_selection"): (
        "outputs",
        "honest_nested_selection",
        "02_inner_cv_recipe_selection",
    ),
    ("outputs", "nested_cv_selection", "03_outer_fold_evaluation"): (
        "outputs",
        "honest_nested_selection",
        "03_outer_fold_evaluation",
    ),
    ("outputs", "nested_cv_selection", "04_refit_and_test_ranking"): (
        "outputs",
        "honest_nested_selection",
        "04_final_refit_and_test_ranking",
    ),
}


def read_csv_if_available(
    path: Path | str,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV artifact when present; otherwise return an empty frame."""
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns or ()))
    return pd.read_csv(csv_path)


def read_json_if_available[T](
    path: Path | str,
    *,
    default: T | None = None,
) -> Any | T | None:
    """Read a JSON artifact when present; otherwise return ``default``."""
    json_path = Path(path)
    if not json_path.exists() or json_path.stat().st_size == 0:
        return default
    return json.loads(json_path.read_text())


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write a JSON artifact with stable formatting."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True))


def should_use_cached_outputs(
    *paths: Path | str,
    use_existing_outputs: bool = True,
    force_rerun: bool = False,
) -> bool:
    """Return whether all requested artifacts can be reused."""
    if force_rerun or not use_existing_outputs:
        return False
    return all(Path(path).exists() and Path(path).stat().st_size > 0 for path in paths)


def resolve_output_dir(
    project_root: Path | str,
    *parts: str,
    legacy_parts: Sequence[str] | None = None,
    use_existing_outputs: bool = True,
    force_rerun: bool = False,
) -> Path:
    """Resolve an output dir, falling back to a populated legacy directory."""
    root = Path(project_root)
    output_parts = tuple(parts)
    output_dir = root.joinpath(*output_parts)
    resolved_legacy_parts = (
        LEGACY_OUTPUT_ALIASES.get(output_parts) if legacy_parts is None else tuple(legacy_parts)
    )
    legacy_dir = root.joinpath(*resolved_legacy_parts) if resolved_legacy_parts else None
    if (
        legacy_dir is not None
        and use_existing_outputs
        and not force_rerun
        and legacy_dir.exists()
        and any(legacy_dir.iterdir())
        and not output_dir.exists()
    ):
        return legacy_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_or_build_csv(
    path: Path | str,
    builder: Callable[[], pd.DataFrame],
    *,
    use_existing_outputs: bool = True,
    force_rerun: bool = False,
) -> pd.DataFrame:
    """Load a CSV artifact if available, otherwise build and persist it."""
    csv_path = Path(path)
    if should_use_cached_outputs(
        csv_path,
        use_existing_outputs=use_existing_outputs,
        force_rerun=force_rerun,
    ):
        return pd.read_csv(csv_path)
    frame = builder()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return frame


def load_or_build_json(
    path: Path | str,
    builder: Callable[[], Mapping[str, Any]],
    *,
    use_existing_outputs: bool = True,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """Load a JSON artifact if available, otherwise build and persist it."""
    json_path = Path(path)
    if should_use_cached_outputs(
        json_path,
        use_existing_outputs=use_existing_outputs,
        force_rerun=force_rerun,
    ):
        return dict(json.loads(json_path.read_text()))
    payload = dict(builder())
    write_json(json_path, payload)
    return payload
