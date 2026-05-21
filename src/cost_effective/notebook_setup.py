"""One-shot notebook environment and data loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataset import DEFAULT_MAX_TARGETS, find_project_root
from .paths import approach_outputs_dir, feature_selection_outputs_dir
from .utils import (
    DEFAULT_SUBMISSION_PREFIX,
    ModelingStageData,
    configure_notebook_style,
    load_modeling_stage_data,
)


@dataclass(frozen=True, slots=True)
class ModelingNotebookContext:
    """Paths and Stage 3 data for a modeling notebook run."""

    approach: str
    project_root: Path
    data_path: Path
    outputs_path: Path
    feature_selection_outputs: Path
    submission_prefix: str
    max_targets: int
    stage: ModelingStageData


def setup_modeling_notebook(
    approach: str,
    exclude_top_26: bool = False,
    random_state: int = 42,
) -> ModelingNotebookContext:
    """Configure plots and load train/test plus feature-selection artifacts."""
    configure_notebook_style(random_state=random_state)
    project_root = find_project_root()
    feature_selection_outputs = feature_selection_outputs_dir(project_root)
    stage = load_modeling_stage_data(
        project_root / "data",
        feature_selection_outputs,
        exclude_top_26=exclude_top_26,
    )
    return ModelingNotebookContext(
        approach=approach,
        project_root=project_root,
        data_path=project_root / "data",
        outputs_path=approach_outputs_dir(project_root, approach),
        feature_selection_outputs=feature_selection_outputs,
        submission_prefix=DEFAULT_SUBMISSION_PREFIX,
        max_targets=DEFAULT_MAX_TARGETS,
        stage=stage,
    )
