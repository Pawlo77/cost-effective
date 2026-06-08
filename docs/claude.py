"""
Custom Business Scorer — ML Group Project
Score = (TP * 10) - (FP * 5) - (NoVariables * 200)
Compatible with: sklearn GridSearchCV, cross_val_score, RFECV, and Optuna.
"""

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ─────────────────────────────────────────────────────────────────────────────
# 1. CORE SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────


def business_score_from_binary(y_true, y_pred, n_features: int) -> float:
    """
    Compute the exact business score given binary predictions.

    Args:
        y_true:    Ground-truth binary labels (array-like).
        y_pred:    Binary predictions — 0 or 1 (array-like).
        n_features: Number of features used by the model (int).

    Returns:
        Business score (float). Higher is better.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))

    return (tp * 10) - (fp * 5) - (n_features * 200)


def business_score_from_proba(
    y_true,
    y_prob,
    n_features: int,
    max_k: int = 1000,
    threshold: float | None = None,
) -> float:
    """
    Compute business score from probability scores.

    Strategy:
      - If threshold is None: select the top-k by probability, where k <= max_k.
        k is chosen to maximise the score (profit curve approach).
      - If threshold is given: use that fixed threshold, then clip to max_k.

    Args:
        y_true:     Ground-truth binary labels.
        y_prob:     Predicted probabilities for class 1.
        n_features: Number of features used.
        max_k:      Maximum number of positive predictions (default 1000).
        threshold:  Fixed probability threshold (optional). If None, uses
                    the optimal top-k from the profit curve.

    Returns:
        Business score (float).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if threshold is not None:
        # Fixed threshold — apply and enforce max_k cap
        y_pred = (y_prob >= threshold).astype(int)
        if y_pred.sum() > max_k:
            # Keep only the top-max_k among those above threshold
            above_thresh = np.where(y_pred == 1)[0]
            sorted_above = above_thresh[np.argsort(y_prob[above_thresh])[::-1]]
            y_pred = np.zeros(len(y_prob), dtype=int)
            y_pred[sorted_above[:max_k]] = 1
    else:
        # Profit-curve sweep: find the optimal k
        sorted_idx = np.argsort(y_prob)[::-1]  # descending probability
        best_score = -np.inf
        best_k = 0
        cumulative_tp = 0
        cumulative_fp = 0

        for k in range(1, min(max_k, len(y_prob)) + 1):
            idx = sorted_idx[k - 1]
            if y_true[idx] == 1:
                cumulative_tp += 1
            else:
                cumulative_fp += 1
            score_k = (cumulative_tp * 10) - (cumulative_fp * 5) - (n_features * 200)
            if score_k > best_score:
                best_score = score_k
                best_k = k

        y_pred = np.zeros(len(y_prob), dtype=int)
        y_pred[sorted_idx[:best_k]] = 1

    return business_score_from_binary(y_true, y_pred, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SKLEARN MAKE_SCORER WRAPPER
# Wraps business_score_from_proba so it works with cross_val_score,
# GridSearchCV, and RFECV. The n_features count is captured via closure.
# ─────────────────────────────────────────────────────────────────────────────


def make_business_scorer(n_features: int, max_k: int = 1000):
    """
    Factory that returns a sklearn-compatible scorer for a fixed feature count.

    Usage:
        scorer = make_business_scorer(n_features=12)
        scores = cross_val_score(model, X, y, scoring=scorer, cv=5)

    Args:
        n_features: Number of features in the dataset passed to the model.
                    Update this as you reduce features.
        max_k:      Maximum number of positive predictions.

    Returns:
        A sklearn scorer object.
    """

    def _scorer(estimator, X, y_true):
        y_prob = estimator.predict_proba(X)[:, 1]
        return business_score_from_proba(y_true, y_prob, n_features, max_k)

    return _scorer  # pass directly to scoring= parameter


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROFIT CURVE — find optimal k and threshold from OOF predictions
# ─────────────────────────────────────────────────────────────────────────────


def profit_curve(
    y_true,
    y_prob,
    n_features: int,
    max_k: int = 1000,
) -> pd.DataFrame:
    """
    Compute the full profit curve: business score at every k from 1 to max_k.

    Usage:
        oof_probs = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:,1]
        df = profit_curve(y, oof_probs, n_features=10)
        k_star = df.loc[df['score'].idxmax(), 'k']
        print(f"Optimal predictions: {k_star}, Score: {df['score'].max():.0f}")

    Returns:
        DataFrame with columns: k, tp, fp, score, precision, threshold
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    sorted_idx = np.argsort(y_prob)[::-1]
    records = []
    tp = fp = 0

    for k in range(1, min(max_k, len(y_prob)) + 1):
        idx = sorted_idx[k - 1]
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1

        score = (tp * 10) - (fp * 5) - (n_features * 200)
        precision = tp / k
        threshold = y_prob[sorted_idx[k - 1]]

        records.append({
            "k": k,
            "tp": tp,
            "fp": fp,
            "score": score,
            "precision": round(precision, 4),
            "threshold": round(threshold, 4),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPTUNA OBJECTIVE FUNCTION
# Simultaneously tunes hyperparameters and validates with the business score.
# ─────────────────────────────────────────────────────────────────────────────


def make_optuna_objective(X_train, y_train, n_features: int, max_k: int = 1000, cv_folds: int = 5):
    """
    Returns an Optuna objective function that optimises LightGBM hyperparameters
    using the custom business scorer.

    Usage:
        objective = make_optuna_objective(X_train, y_train, n_features=10)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        # Retrieve best params
        print(study.best_params)
        print(f"Best CV business score: {study.best_value:.0f}")
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),  # L1
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),  # L2
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        }

        model = lgb.LGBMClassifier(**params)
        scorer = make_business_scorer(n_features=n_features, max_k=max_k)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            fold_score = scorer(model, X_val, y_val)
            fold_scores.append(fold_score)

        return float(np.mean(fold_scores))

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# 5. GREEDY FORWARD FEATURE SELECTOR using the business scorer
# ─────────────────────────────────────────────────────────────────────────────


def greedy_feature_selection(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    candidate_features: list[int],
    max_k: int = 1000,
    cv_folds: int = 5,
    verbose: bool = True,
) -> list[int]:
    """
    Greedily adds features one at a time.
    A feature is kept only if it improves the CV business score by > 200 pts
    (i.e., it pays for its own 200-pt cost).

    Args:
        model:               Unfitted sklearn-compatible classifier.
        X_train:             Full training matrix (all candidate features).
        y_train:             Binary labels.
        candidate_features:  Column indices of features to consider (ordered
                             by importance — most important first).
        max_k:               Max predictions allowed.
        cv_folds:            Number of CV folds.
        verbose:             Print progress.

    Returns:
        List of selected feature indices.
    """
    selected = []
    best_score = -np.inf
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for feat_idx in candidate_features:
        trial_features = [*selected, feat_idx]
        n_feats = len(trial_features)
        X_trial = X_train[:, trial_features]

        scorer = make_business_scorer(n_features=n_feats, max_k=max_k)
        fold_scores = []

        for tr_idx, val_idx in cv.split(X_trial, y_train):
            model.fit(X_trial[tr_idx], y_train[tr_idx])
            fold_scores.append(scorer(model, X_trial[val_idx], y_train[val_idx]))

        trial_score = np.mean(fold_scores)

        # Keep the feature only if improvement > cost of adding it (200 pts)
        improvement = trial_score - best_score
        kept = improvement > 200

        if kept:
            selected.append(feat_idx)
            best_score = trial_score

        if verbose:
            status = "✓ KEPT" if kept else "✗ dropped"
            print(
                f"Feature {feat_idx:>4} | n_feats={n_feats:>2} | "
                f"CV score={trial_score:>8.1f} | Δ={improvement:>+8.1f} | {status}"
            )

    if verbose:
        print(f"\nFinal selection: {len(selected)} features | Best CV score: {best_score:.1f}")

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# USAGE EXAMPLE (run as script)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # --- Synthetic dataset ---
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=8, weights=[0.85, 0.15], random_state=42
    )
    N_FEATURES = 10  # pretend we selected 10 features

    # --- 1. Quick cross_val_score with custom scorer ---
    model = LogisticRegression(max_iter=500, random_state=42)
    scorer = make_business_scorer(n_features=N_FEATURES)
    cv_scores = cross_val_score(model, X, y, scoring=scorer, cv=5)
    print(f"CV Business Scores: {cv_scores.round(1)}")
    print(f"Mean: {cv_scores.mean():.1f}  ±  {cv_scores.std():.1f}")

    # --- 2. Profit curve from OOF predictions ---
    from sklearn.model_selection import cross_val_predict

    oof_probs = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    df_curve = profit_curve(y, oof_probs, n_features=N_FEATURES)
    k_star = int(df_curve.loc[df_curve["score"].idxmax(), "k"])
    best = df_curve["score"].max()
    thresh_star = float(df_curve.loc[df_curve["score"].idxmax(), "threshold"])

    print(f"\nOptimal k*: {k_star} predictions")
    print(f"Optimal threshold: {thresh_star:.4f}")
    print(f"Expected business score at k*: {best:.1f}")

    # Plot profit curve
    plt.figure(figsize=(8, 4))
    plt.plot(df_curve["k"], df_curve["score"], color="#185FA5", lw=1.5)
    plt.axvline(k_star, color="#A32D2D", ls="--", lw=1, label=f"k*={k_star}")
    plt.axhline(0, color="gray", ls=":", lw=0.8)
    plt.xlabel("Number of predictions (k)")
    plt.ylabel("Business score")
    plt.title("Profit curve — business score vs predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("profit_curve.png", dpi=150)
    print("Profit curve saved to profit_curve.png")

    # --- 3. Optuna study (5 quick trials as demo) ---
    objective = make_optuna_objective(X, y, n_features=N_FEATURES, cv_folds=3)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, show_progress_bar=False)
    print(f"\nOptuna best CV score: {study.best_value:.1f}")
    print(f"Best params: {study.best_params}")
