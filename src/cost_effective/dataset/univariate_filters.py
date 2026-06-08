"""Stage 1 univariate feature filtering (500 -> ~80)."""

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif


class UnivariateFeatureFilter:
    """Apply broad univariate filtering to reduce 500 -> ~80 features.

    The filter combines three signals:
        - Variance thresholding.
        - Mutual information ranking.
        - LightGBM impurity-based importance.
    """

    def __init__(self, variance_threshold: float = 0.01):
        """Initialize the stage 1 univariate filter.

        Args:
            variance_threshold: Minimum variance to retain a feature. Features
                with variance lower than this value are removed.
        """
        self.variance_threshold = variance_threshold
        self.selected_features_ = None
        self.feature_rankings_ = {}
        self.mi_scores_ = None
        self.lgbm_importances_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "UnivariateFeatureFilter":
        """Fit the univariate filter on training data.

        Args:
            X: Feature matrix of shape `(n_samples, n_features)`.
            y: Target vector of shape `(n_samples,)`.

        Returns:
            The fitted filter instance.
        """
        # Stage 1a: Variance Threshold
        var_selector = VarianceThreshold(threshold=self.variance_threshold)
        var_selector.fit(X)
        features_after_var = X.columns[var_selector.get_support()].tolist()
        X_var = X[features_after_var]

        print(
            f"[Stage 1a] Variance Threshold: {len(X.columns)} → {len(features_after_var)} features"
        )

        # Stage 1b: Mutual Information
        self.mi_scores_ = mutual_info_classif(X_var, y, random_state=42)
        mi_ranking = pd.DataFrame({
            "feature": features_after_var,
            "mi_score": self.mi_scores_,
        }).sort_values("mi_score", ascending=False)

        print(f"[Stage 1b] Mutual Information computed for {len(features_after_var)} features")
        print(f"  Top 10 MI scores:\n{mi_ranking.head(10).to_string()}\n")

        # Stage 1c: LightGBM Tree-Based Importance
        lgbm = LGBMClassifier(
            n_estimators=100, num_leaves=31, learning_rate=0.05, verbose=-1, random_state=42
        )
        lgbm.fit(X_var, y, eval_metric="auc")

        lgbm_ranking = pd.DataFrame({
            "feature": features_after_var,
            "lgbm_importance": lgbm.feature_importances_,
        }).sort_values("lgbm_importance", ascending=False)

        self.lgbm_importances_ = lgbm.feature_importances_

        print("[Stage 1c] LightGBM Importance computed")
        print(f"  Top 10 LGBM Importances:\n{lgbm_ranking.head(10).to_string()}\n")

        # Combined ranking: Average normalized ranks from MI and LGBM
        mi_rank = (
            mi_ranking.reset_index(drop=True).reset_index().rename(columns={"index": "mi_rank"})
        )
        lgbm_rank = (
            lgbm_ranking.reset_index(drop=True).reset_index().rename(columns={"index": "lgbm_rank"})
        )

        combined = mi_rank.merge(lgbm_rank, on="feature", how="inner")
        combined["combined_rank"] = combined["mi_rank"] + combined["lgbm_rank"]
        combined = combined.sort_values("combined_rank")

        # Store feature rankings
        self.feature_rankings_["variance_filtered"] = features_after_var
        self.feature_rankings_["mi_ranking"] = mi_ranking.to_dict("records")
        self.feature_rankings_["lgbm_ranking"] = lgbm_ranking.to_dict("records")
        self.feature_rankings_["combined"] = combined

        return self

    def transform(self, X: pd.DataFrame, n_features: int = 80) -> tuple[pd.DataFrame, list[str]]:
        """Select top features using combined MI and LightGBM ranking.

        Args:
            X: Feature matrix to transform.
            n_features: Number of top-ranked features to retain.

        Returns:
            A tuple `(X_selected, selected_features)` where `X_selected`
            contains only retained columns.

        Raises:
            ValueError: If the filter has not been fitted.
        """
        if self.feature_rankings_ is None:
            raise ValueError("Fit the filter first using .fit()")

        combined = self.feature_rankings_["combined"]
        selected = combined.head(n_features)["feature"].tolist()
        self.selected_features_ = selected

        print(f"[Stage 1 Output] Selected {len(selected)} features out of {len(combined)}")
        return X[selected], selected

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, n_features: int = 80
    ) -> tuple[pd.DataFrame, list[str]]:
        """Fit the filter and transform data in one step.

        Args:
            X: Feature matrix.
            y: Target vector.
            n_features: Number of top-ranked features to retain.

        Returns:
            A tuple `(X_selected, selected_features)`.
        """
        self.fit(X, y)
        return self.transform(X, n_features=n_features)

    def get_feature_rankings(self) -> pd.DataFrame:
        """Return the combined feature ranking table.

        Returns:
            DataFrame with combined ranking information.

        Raises:
            ValueError: If the filter has not been fitted.
        """
        if self.feature_rankings_ is None:
            raise ValueError("Fit the filter first using .fit()")
        return self.feature_rankings_["combined"]
