"""Output directory names for notebook approaches."""

from pathlib import Path

APPROACH_TOPK_PROFIT_CURVE = "topk_profit_curve"
APPROACH_EV_PROFIT_TARGETING = "ev_profit_targeting"
APPROACH_RANK_FUSION = "rank_fusion"
APPROACH_SEGMENT_TARGETING = "segment_targeting"
APPROACH_CLUSTER_SPLIT = "cluster_split"

SHARED_OUTPUTS = "outputs"


def approach_outputs_dir(project_root: Path, approach: str) -> Path:
    """Return ``outputs/<approach>/`` and ensure it exists."""
    path = project_root / SHARED_OUTPUTS / approach
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_selection_outputs_dir(project_root: Path) -> Path:
    """Stage 1-3 artifacts shared by all modeling notebooks."""
    return project_root / SHARED_OUTPUTS
