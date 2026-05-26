"""KMeans k diagnostics: elbow (inertia) and silhouette per feature set."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from .cluster_split_targeting import _scale_cluster_matrix


def kmeans_k_diagnostics(
    X: pd.DataFrame,
    features: list[str],
    k_values: range | list[int],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """Inertia and silhouette for each k on scaled features."""
    k_list = sorted({int(k) for k in k_values if int(k) >= 2})
    n_samples = len(X)
    k_list = [k for k in k_list if k < n_samples]

    scaler = RobustScaler()
    matrix = _scale_cluster_matrix(X, features, scaler, fit=True)

    rows: list[dict[str, float | int]] = []
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(matrix)
        n_labels = len(np.unique(labels))
        sil = (
            float(silhouette_score(matrix, labels))
            if n_labels > 1 and n_labels < n_samples
            else float("nan")
        )
        rows.append({"k": k, "inertia": float(kmeans.inertia_), "silhouette": sil})

    return pd.DataFrame(rows)
