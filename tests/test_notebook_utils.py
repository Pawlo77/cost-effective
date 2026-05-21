from pathlib import Path

import numpy as np

from cost_effective.utils import (
    var_name_to_index,
    write_submission_files,
)


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
