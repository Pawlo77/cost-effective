"""Tests for KMeans k diagnostics."""

import numpy as np
import pandas as pd

from cost_effective.models.cluster_k_selection import kmeans_k_diagnostics


def test_kmeans_k_diagnostics_columns() -> None:
    rng = np.random.default_rng(0)
    cols = [f"var_{i}" for i in range(5)]
    x = pd.DataFrame(rng.normal(size=(120, 5)), columns=cols)
    diag = kmeans_k_diagnostics(x, cols, range(2, 5), random_state=0)
    assert set(diag.columns) >= {"k", "inertia", "silhouette"}
    assert diag["k"].min() >= 2
