"""Stage 2 multicollinearity filtering and L1 regularization (80 -> ~25, data-driven)."""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .utils import custom_scorer


class MulticollinearityFilter:
    """Remove redundant features and apply L1-driven sparsity.

    Pipeline steps:
        1. Correlation pruning.
        2. VIF filtering.
        3. LASSO path feature selection.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.85,
        vif_threshold: float = 5.0,
        random_state: int = 42,
    ):
        """Initialize stage 2 filtering configuration.

        Args:
            correlation_threshold: Absolute correlation threshold for pairwise
                pruning.
            vif_threshold: VIF threshold used to remove multicollinear
                features.
            random_state: Random seed used by stochastic components.
        """
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.random_state = random_state

        self.selected_features_ = None
        self.filter_history_ = {}
        self.lasso_path_ = None
        self.mi_scores_ = None

    def _correlation_pruning(self, X: pd.DataFrame, y: pd.Series) -> tuple[list[str], dict]:
        """Prune highly correlated feature pairs.

        For each pair above threshold, the feature with lower mutual
        information with the target is removed.

        Args:
            X: Input feature matrix.
            y: Target vector.

        Returns:
            A tuple `(selected_features, metadata)`.
        """
        # Compute MI with target
        X_np = X.values
        mi_scores = mutual_info_classif(X_np, y, random_state=self.random_state)
        mi_dict = dict(zip(X.columns, mi_scores, strict=False))
        self.mi_scores_ = mi_dict

        # Compute correlation matrix
        corr_matrix = X.corr(method="pearson").abs()

        # Find high-correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= self.correlation_threshold:
                    feat_i = corr_matrix.columns[i]
                    feat_j = corr_matrix.columns[j]
                    high_corr_pairs.append({
                        "feature_1": feat_i,
                        "feature_2": feat_j,
                        "correlation": corr_matrix.iloc[i, j],
                        "mi_1": mi_dict[feat_i],
                        "mi_2": mi_dict[feat_j],
                    })

        # Greedy removal: for each pair, remove the lower-MI feature
        removed = set()
        for pair in high_corr_pairs:
            f1, f2 = pair["feature_1"], pair["feature_2"]
            if f1 in removed or f2 in removed:
                continue

            if pair["mi_1"] >= pair["mi_2"]:
                removed.add(f2)
            else:
                removed.add(f1)

        selected = [f for f in X.columns if f not in removed]

        print(f"[Stage 2a] Correlation Pruning (threshold={self.correlation_threshold}):")
        print(f"  High-correlation pairs found: {len(high_corr_pairs)}")
        print(f"  Features removed: {len(removed)}")
        print(f"  {len(X.columns)} → {len(selected)} features\n")

        return selected, {"removed": list(removed), "pairs": high_corr_pairs}

    def _vif_filtering(self, X: pd.DataFrame) -> tuple[list[str], dict]:
        """Remove features whose VIF exceeds the configured threshold.

        Args:
            X: Input feature matrix.

        Returns:
            A tuple `(selected_features, metadata)` including VIF scores.
        """
        X_scaled = StandardScaler().fit_transform(X)

        vif_scores = []
        for i in range(X.shape[1]):
            try:
                vif = variance_inflation_factor(X_scaled, i)
            except Exception:
                vif = np.inf
            vif_scores.append({"feature": X.columns[i], "vif": vif})

        vif_df = pd.DataFrame(vif_scores).sort_values("vif", ascending=False)

        removed = vif_df[vif_df["vif"] > self.vif_threshold]["feature"].tolist()
        selected = [f for f in X.columns if f not in removed]

        print(f"[Stage 2b] VIF Filtering (threshold={self.vif_threshold}):")
        print(f"  Features with VIF > threshold: {len(removed)}")
        if len(vif_df) > 0:
            print(f"  Top 10 VIF scores:\n{vif_df.head(10).to_string()}\n")
        print(f"  {len(X.columns)} → {len(selected)} features\n")

        return selected, {"vif_scores": vif_df.to_dict("records")}

    def _lasso_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_target: int | None = None,
        n_features_band: tuple[int, int] = (20, 30),
    ) -> tuple[list[str], dict]:
        """Select sparse features with a logistic L1 regularization path.

        Args:
            X: Input feature matrix.
            y: Target vector.
            n_features_target: Optional target number of non-zero features.
                When `None`, selection is data-driven within `n_features_band`.
            n_features_band: Preferred non-zero feature count band used when
                `n_features_target` is `None`.

        Returns:
            A tuple `(selected_features, metadata)` including the explored
            coefficient path and optimal regularization strength.
        """
        X_scaled = StandardScaler().fit_transform(X)

        # Logistic Regression with L1 (LASSO)

        # Try different C values (inverse regularization strength)
        C_values = np.logspace(-4, 2, 50)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        coef_paths = []

        for C in C_values:
            lr_C = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=C,
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
            lr_C.fit(X_scaled, y)
            n_nonzero = (lr_C.coef_[0] != 0).sum()
            cv_score = cross_val_score(
                lr_C,
                X_scaled,
                y,
                cv=cv,
                scoring=custom_scorer,
                n_jobs=-1,
            ).mean()
            coef_paths.append({
                "C": C,
                "n_nonzero": n_nonzero,
                "cv_score": cv_score,
                "coefficients": lr_C.coef_[0].copy(),
            })

        coef_df = pd.DataFrame(coef_paths)
        print("[Stage 2c] LASSO Path Regularization:")
        print(f"  C range: [{C_values[0]:.6f}, {C_values[-1]:.6f}]")
        nonzero_range = f"{coef_df['n_nonzero'].min()} to {coef_df['n_nonzero'].max()}"
        print(f"  Non-zero features range: {nonzero_range}")

        # Select C in a data-driven manner.
        # 1) If an explicit target is given, choose closest feature count.
        # 2) Otherwise, maximize CV business score within a preferred sparsity band.
        if n_features_target is not None:
            idx = (coef_df["n_nonzero"] - n_features_target).abs().idxmin()
        else:
            lower, upper = n_features_band
            in_band = coef_df[
                (coef_df["n_nonzero"] >= lower)
                & (coef_df["n_nonzero"] <= upper)
                & (coef_df["n_nonzero"] > 0)
            ]
            candidate_pool = in_band if not in_band.empty else coef_df[coef_df["n_nonzero"] > 0]
            idx = candidate_pool["cv_score"].idxmax()

        optimal_C = coef_df.loc[idx, "C"]
        optimal_coefs = coef_df.loc[idx, "coefficients"]

        selected = X.columns[optimal_coefs != 0].tolist()

        print(f"  Optimal C: {optimal_C:.6f} → {len(selected)} non-zero features")
        print(f"  Cross-validated business score at optimal C: {coef_df.loc[idx, 'cv_score']:.4f}")
        print(f"  Selected {len(selected)} features\n")

        self.lasso_path_ = coef_df

        return selected, {
            "coef_path": coef_df.to_dict("records"),
            "optimal_C": optimal_C,
            "optimal_coefs": optimal_coefs,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_target: int | None = None,
        n_features_band: tuple[int, int] = (20, 30),
    ) -> "MulticollinearityFilter":
        """Apply stage 2 filtering in sequence.

        Args:
            X: Feature matrix output from stage 1.
            y: Target vector.
            n_features_target: Optional target number of features after LASSO.
                When `None`, a data-driven count is selected in `n_features_band`.
            n_features_band: Preferred approximate band for selected features.

        Returns:
            The fitted filter instance.
        """
        print("=" * 70)
        print("STAGE 2: MULTICOLLINEARITY & L1 REGULARIZATION FILTERING")
        print("=" * 70 + "\n")

        # Step 1: Correlation Pruning
        features_after_corr, corr_info = self._correlation_pruning(X, y)
        self.filter_history_["correlation"] = corr_info

        # Step 2: VIF Filtering
        X_after_corr = X[features_after_corr]
        features_after_vif, vif_info = self._vif_filtering(X_after_corr)
        self.filter_history_["vif"] = vif_info

        # Step 3: LASSO
        X_after_vif = X[features_after_vif]
        features_after_lasso, lasso_info = self._lasso_feature_selection(
            X_after_vif,
            y,
            n_features_target=n_features_target,
            n_features_band=n_features_band,
        )
        self.filter_history_["lasso"] = lasso_info

        self.selected_features_ = features_after_lasso

        print(f"STAGE 2 SUMMARY: {len(X.columns)} → {len(self.selected_features_)} features")
        print("=" * 70 + "\n")

        return self

    def transform(self, X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Select previously fitted features.

        Args:
            X: Feature matrix to transform.

        Returns:
            A tuple `(X_selected, selected_features)`.

        Raises:
            ValueError: If the filter has not been fitted.
        """
        if self.selected_features_ is None:
            raise ValueError("Fit the filter first using .fit()")

        return X[self.selected_features_], self.selected_features_

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features_target: int | None = None,
        n_features_band: tuple[int, int] = (20, 30),
    ) -> tuple[pd.DataFrame, list[str]]:
        """Fit the filter and transform data in one step.

        Args:
            X: Feature matrix.
            y: Target vector.
            n_features_target: Optional target number of features after stage 2.
                When `None`, selection is data-driven in `n_features_band`.
            n_features_band: Preferred approximate band for selected features.

        Returns:
            A tuple `(X_selected, selected_features)`.
        """
        self.fit(X, y, n_features_target=n_features_target, n_features_band=n_features_band)
        return self.transform(X)
