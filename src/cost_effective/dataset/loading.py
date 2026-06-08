"""Shared train/test loading used by all notebooks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_project_root() -> Path:
    """Locate repo root by presence of ``data/x_train.txt``."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "data" / "x_train.txt").exists():
            return candidate
    package_root = Path(__file__).resolve().parents[3]
    if (package_root / "data" / "x_train.txt").exists():
        return package_root
    raise FileNotFoundError("Could not find project root containing data/x_train.txt")


DATA_DIR = find_project_root() / "data"


def load_training_data(data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load aligned feature matrix and binary target (5000 rows)."""
    root = data_dir or DATA_DIR

    x = pd.read_csv(
        root / "x_train.txt",
        sep=r"\s+",
        header=None,
        low_memory=False,
    ).apply(pd.to_numeric, errors="coerce")
    y_raw = pd.to_numeric(
        pd.read_csv(root / "y_train.txt", sep=r"\s+", header=None).iloc[:, 0],
        errors="coerce",
    )

    valid = y_raw.notna() & y_raw.isin([0, 1])
    x = x.loc[valid].reset_index(drop=True)
    y = y_raw.loc[valid].astype(int).reset_index(drop=True)

    x.columns = [f"var_{i}" for i in range(x.shape[1])]
    x = x.fillna(0.0)
    y.name = "target"
    return x, y


def load_test_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load test feature matrix (5000 rows, same columns as train)."""
    root = data_dir or DATA_DIR
    x_test = pd.read_csv(
        root / "x_test.txt",
        sep=r"\s+",
        header=None,
        low_memory=False,
    ).apply(pd.to_numeric, errors="coerce")
    # drop the header row (becomes all-NaN after numeric coercion)
    x_test = x_test.dropna(how="all").reset_index(drop=True)
    x_test.columns = [f"var_{i}" for i in range(x_test.shape[1])]
    return x_test.fillna(0.0)
