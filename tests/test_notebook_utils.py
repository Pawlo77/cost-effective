from pathlib import Path

import numpy as np

from cost_effective.utils import (
    feature_set_candidates_from_selection_results,
    var_name_to_index,
    write_submission_files,
)


def test_feature_set_candidates_from_selection_results(tmp_path) -> None:
    csv = tmp_path / "feature_selection_results.csv"
    csv.write_text(
        "feature,cv_score_if_dropped,delta,order\n"
        "var_0,0,1,1\n"
        "var_1,0,1,2\n"
        "var_2,0,1,3\n"
        "var_3,0,1,4\n"
        "var_4,0,1,5\n",
        encoding="utf-8",
    )
    candidates = feature_set_candidates_from_selection_results(
        tmp_path,
        sizes=(1, 3, 5),
    )
    assert candidates["top_01"] == ["var_0"]
    assert candidates["top_03"] == ["var_0", "var_1", "var_2"]
    assert candidates["top_05"] == ["var_0", "var_1", "var_2", "var_3", "var_4"]


def test_var_name_to_index() -> None:
    assert var_name_to_index("var_389") == 389


def test_write_submission_files(tmp_path: Path) -> None:
    obs_path, vars_path = write_submission_files(
        tmp_path,
        "team_test",
        np.array([10, 20]),
        ["var_1", "var_42"],
    )
    assert obs_path.read_text(encoding="utf-8") == "10\n20\n"
    assert vars_path.read_text(encoding="utf-8") == "1\n42\n"
