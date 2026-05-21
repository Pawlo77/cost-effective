# Cost-Effective Predictive Modeling — Project Plan

## 1. Executive Summary

This project challenges us to build a **profit-maximizing** predictive model, not an accuracy-maximizing one. The business metric is explicit:

$$\text{Score} = (TP \times 10) - (FP \times 5) - (\text{NoVariables} \times 200)$$

This equation has three competing forces that fundamentally reshape traditional ML optimization:

1. **TP Reward:** +10 EUR per correct prediction (high immediate payoff).
2. **FP Penalty:** −5 EUR per incorrect prediction (precision matters).
3. **Variable Cost:** −200 EUR per feature (extreme sparsity pressure).

**The Core Insight:** A $200$-point penalty per variable means we must be ruthless about dimensionality. A model with $50$ features starts with a **−10,000 point deficit** regardless of predictive power.

---

## 2. Mathematical Framework

### 2.1 Break-Even Analysis

**Question:** What probability threshold should trigger a positive prediction?

**Derivation:** The expected value of predicting positive for a customer is:

$$E[Score] = p(convert) \times 10 + (1 - p(convert)) \times (-5)$$
$$E[Score] = 10p - 5 + 5p = 15p - 5$$

Setting this equal to zero (break-even):

$$15p - 5 = 0 \implies p = \frac{1}{3} \approx 0.333$$

**Implication:** Only target customers with **$p > 0.333$** ($33.3\%$ conversion probability). The standard ML threshold of $0.5$ is overly conservative—it leaves profitable predictions on the table.

### 2.2 Feature Cost Recovery

Each feature must pay for itself. How many True Positives are needed to justify one variable?

$$200 = \text{TP gain} \times 10 \implies \text{TP gain} \geq 20$$

**Rule of thumb:** A feature is net-positive only if it generates at least **20 additional True Positives** (or prevents $40$ False Positives) in cross-validation relative to a baseline.

---

## 3. Feature Selection Strategy

The $500$-variable input space contains massive noise. We must apply a **multi-stage funnel** that progressively filters down to the essential predictors.

### 3.1 Stage 1: Broad Univariate Filtering ($500$ → $\sim 80$)

**Goal:** Eliminate obvious noise and near-zero-variance predictors.

**Methods:**
- **Variance Threshold:** Remove features with variance $< 0.01$ (constants and near-constants).
- **Mutual Information:** Compute mutual information between each feature and target. Retain only top $50$–$80$ by MI.
- **Tree-Based Importance:** Train a shallow LightGBM on all features, rank by impurity-based importance, keep top $80$.

**Output:** $\sim 80$ candidate features.

### 3.2 Stage 2: Multicollinearity & Linear Regularization ($80$ → $\sim 25$)

**Goal:** Remove redundant features and apply penalty-based sparsity.

**Methods:**
- **Correlation Pruning:** Compute Pearson/Spearman correlations. For correlated pairs ($|r| \geq 0.85$), retain only the feature with highest univariate association with target.
- **VIF Filtering:** Calculate Variance Inflation Factor. Remove features with VIF $> 5$.
- **L1 Regularization (LASSO):** Fit Logistic Regression with L1 penalty. Sweep regularization strength parameter (C) from weak to strong, then select the model with best cross-validated performance within an approximate sparsity band (about $20–30$ non-zero features).

**Output:** Data-driven candidate feature set (typically around $20–30$ features) with reduced collinearity.

### 3.3 Stage 3: Order Features by importance

**Goal:** Use RFECV to rank the Stage 2-selected features so that downstream modeling can compare models on multiple feature-set sizes, with smaller subsets constrained to the most important variables.

**Rationale:**
- Stage 2 already removes most noise and collinearity, so Stage 3 should focus on ordering rather than aggressively refitting the feature space.
- RFECV gives a cross-validated importance ranking under the custom scorer, which is useful for comparing top-k feature subsets in modeling.
- This lets us evaluate whether the best business score comes from the full Stage 2 set or from a smaller importance-limited subset.

**Process:**
1. Fit RFECV on the Stage 2 feature matrix using the custom business scorer and the weighted classifier.
2. Use the RFECV ranking to order Stage 2 features from most to least useful.
3. Build candidate feature sets by size, such as top-k prefixes of the ranked list, for model comparison.
4. Keep the ranked Stage 2 feature pool as the source for hyperparameter optimization and final model selection.

**Output:** **Ranked Stage 2 feature list** plus a set of top-k feature subsets for modeling and business-score comparison.

**Note:** RFECV is used here as an ordering step inside cross-validation, so the feature ranking remains aligned with the custom scorer while limiting leakage across folds.

---

## 4. Modeling Strategy

### 4.1 Recommended Algorithms

| Algorithm | Strengths | Weaknesses | Use Case |
|---|---|---|---|
| **LightGBM** | Fast, handles imbalance via `scale_pos_weight`, leaf-wise growth, Gradient-based One-Side Sampling (GOSS) | Requires careful hyperparameter tuning | Primary model; use for Optuna HPO |
| **XGBoost** | Robust to outliers, excellent calibration, level-wise growth reduces overfitting | Slower than LightGBM | Ensemble component; secondary model |
| **Logistic Regression** | Interpretable, well-calibrated, immune to high-variance overfitting on sparse features | No non-linear interactions | Baseline comparison; interpretability |

### 4.2 Class Imbalance Handling

Train data likely has imbalanced classes. Counter this with:

- **`scale_pos_weight` in LGBM/XGBoost:** Set to ratio of negative to positive samples.
  ```
  scale_pos_weight = n_negatives / n_positives
  ```
  This forces the gradient descent to penalize minority-class errors more heavily.

- **SMOTE (Optional):** Synthetic Minority Over-Sampling, but only in cross-validation, never on test data.

- **Class Weights:** Assign higher weights to positive class during training.

### 4.3 Threshold Tuning via Profit Curve

Standard models output probabilities. The question: which cutoff maximizes profit on exactly 1,000 targets?

**Process:**

1. **Fit Model:** Train LightGBM/XGBoost on training data with selected features.
2. **Get Test Probabilities:** `proba = model.predict_proba(X_test)[:, 1]`
3. **Sort Descending:** Order test set by predicted probability, highest first.
4. **Sweep Thresholds:** For each possible top-k (k = 100 to 1,000):
   - Count predicted positives in top-k.
   - Compute hypothetical TP, FP using validation ground truth (if available via cross-val).
   - Calculate score contribution: $(TP \times 10) - (FP \times 5)$.
5. **Optimal k\*:** Select k where $(TP \times 10) - (FP \times 5)$ is maximized, capped at 1,000.
6. **Extract Threshold:** Convert k* back to a probability threshold and apply to final test set.

**Alternative (if no validation labels):** Use the mathematical break-even threshold **p > 0.333** as a floor, then refine via cross-validation profit curves.

---

## 5. Hyperparameter Optimization

Use **Optuna** with the custom scorer as the objective function.

### 5.1 LGBM Search Space

```python
def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": scale_pos_weight,  # Set from data
    }

    model = LGBMClassifier(**params, n_estimators=200, verbose=-1)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=custom_scorer)
    return cv_scores.mean()
```

- **Direction:** `maximize`
- **Trials:** 100–200 (wall time: 1–3 hours on laptop)
- **CV:** 5-fold stratified

### 5.2 Key Hyperparameters to Tune

- `num_leaves`: Controls tree complexity (balance bias–variance).
- `learning_rate`: Step size (smaller = more stable, slower).
- `max_depth`: Tree depth limit (prevent overfitting on sparse features).
- `min_child_samples`: Minimum samples per leaf (regularization).
- `subsample` / `colsample_bytree`: Row/column subsampling (robustness).

---

## 6. Implementation Checklist

### Week 1 — Baseline & Infrastructure

- [x] **Data Loading & EDA**
  - Load `x_train.txt`, `y_train.txt`, `x_test.txt`.
  - Compute class balance, missing values, feature statistics.
  - Document baseline conversion rate in training set.

- [x] **Custom Scorer Implementation**
  - Implement `custom_scorer(estimator, X, y)` function.
  - Validate on simple baseline (all features, logistic regression).
  - Verify that scorer correctly computes TP, FP, and penalizes feature count.

- [x] **Baseline Model (Logistic Regression, All Features)**
  - Fitted LR (grid search over `C`) on full feature set (500 vars).
  - Computed 5-fold CV with the custom scorer and saved results to `outputs/baseline_results.json`.
  - Recorded baseline business score; this establishes the floor and validates the scorer implementation.

- [x] **Shared Infrastructure**
  - Set up repository (Git with shared branch).
  - Create `src/custom_scorer.py` module.
  - Create `src/feature_selection.py` skeleton.
  - Create `src/models.py` skeleton.

### Week 2 — Feature Selection Stages 1 & 2

- [x] **Stage 1: Univariate Filtering (500 → 80)**
  - Compute mutual information for each feature.
  - Compute tree-based importance (LightGBM on all features).
  - Select top 80 by combined ranking.
  - Document removal decisions.

- [x] **Stage 2: Multicollinearity & L1 Regularization (80 → ~25, data-driven)**
  - Correlation matrix, identify correlated clusters.
  - VIF analysis, remove high-VIF features.
  - LASSO path: sweep C parameter, select a data-driven feature count (typically 20–30 non-zero coefficients).
  - Compare with baseline score using this Stage 2-selected set.

- [x] **Stage 3: RFECV Ordering of Stage 2 Features**
  - Fit RFECV with the custom business scorer on the Stage 2 feature set.
  - Rank Stage 2 features from most to least useful.
  - Build top-k feature subsets for downstream model comparison.
  - Save ranked features and subset results for modeling.

- [x] **Interim Checkpoint**
  - Confirm Stage 2 feature set (approximately 20–30 total, data-driven) improves CV score relative to all 500.
  - Document feature engineering pipeline.
  - Stage 3 ordering complete: ready for modeling sprint.

### Week 3 — Modeling, Feature-Set Comparison & Threshold Tuning

- [x] **Hyperparameter Optimization (all models)**
  - Perform systematic hyperparameter sweeps for every candidate model (LightGBM, XGBoost, Logistic Regression).
  - Tree models (LightGBM, XGBoost): use Optuna with 100–200 trials per model, 5-fold stratified CV using the ranked Stage 2 feature sets.
  - Linear models (Logistic Regression): run a deterministic grid sweep over penalties and `C` (e.g. [0.001, 0.01, 0.1, 1, 10, 100]) and evaluate with 5-fold CV under the custom scorer.
  - Record best hyperparameters per model, persist Optuna study artifacts and CV result summaries to `outputs/hpo/` for reproducibility.
  - optimize by ROC-AUC to speed up HPO, as custom scorer is dependent on TP/FP counts which are roc-auc optimized.

- [x] **Feature-Set Comparison**
  - Evaluate models on multiple top-k subsets derived from RFECV ranking.
  - Compare business scores across feature-set sizes.
  - Select the best-ranked subset for final model tuning.

- [x] **Threshold Tuning**
  - Generated cross-validated probabilities and swept thresholds (0.0–1.0) to compute profit curve.
  - Selected optimal threshold and recorded associated TP/FP/F1 and business score.
  - Saved optimal threshold configuration to `outputs/optimal_threshold.json` and applied it to derive target predictions.

### Week 4 — Report, Presentation & Submission

- [ ] **Report (5 pages LaTeX)**
  - **§1 (0.5 pg):** Introduction & problem statement. (High-level framing; cost-sensitive setup.)
  - **§2 (1 pg):** Methodology. (Custom scorer, feature selection pipeline, algorithms, threshold tuning.)
  - **§3 (1.5 pg):** Results & experiments. (Feature selection funnel table, profit curve, model comparison, CV scores.)
  - **§4 (1 pg):** Discussion. (Design decisions, trade-offs, challenges, insights.)
  - **§5 (0.75 pg):** Conclusion & reproducibility notes.
  - **§6 (0.25 pg):** References.

- [ ] **Required Figures**
- [ ] **Fig 1:** Feature Selection Funnel (500 → 80 → ~25 data-driven).
  - **Fig 2:** Profit Curve (Score vs. Target Count k).
  - **Fig 3:** Model Comparison (LGBM, XGBoost, LR business scores).
- [ ] **Fig 4:** Feature Importance (mutual information or LightGBM importance of final Stage 2-selected features).
  - **Fig 5:** Class Distribution (baseline imbalance ratio).

- [ ] **Presentation (7 min max)**
  - **Slide 1 (0:00–0:30):** Title, team, problem framing. "Why is this problem different from standard ML?"
  - **Slide 2 (0:30–1:15):** Scoring function breakdown & break-even threshold (p = 0.333).
  - **Slide 3 (1:15–2:00):** Feature selection funnel & final feature set.
  - **Slide 4 (2:00–3:00):** Model selection & hyperparameter tuning results.
  - **Slide 5 (3:00–4:00):** Threshold tuning & profit curve.
  - **Slide 6 (4:00–5:30):** Final model performance, business score, test predictions.
  - **Slide 7 (5:30–7:00):** Key insights, reproducibility, lessons learned.

- [ ] **Submission Package**
  - `*_obs.txt`: Test indices (up to 1,000) in plain text, one per line.
  - `*_vars.txt`: Variable indices (0-indexed) used by model, one per line.
  - `report.pdf`: Final LaTeX report.
  - `presentation.pdf` or `.pptx`: Slides.
  - `code/`: Source code directory with:
    - `src/custom_scorer.py`
    - `src/feature_selection.py`
    - `src/models.py`
    - `notebooks/analysis.ipynb` (optional; analysis & figures)
    - `README.md` (reproduction steps)
  - ZIP archive: `STUDENT1ID_STUDENT2ID_STUDENT3ID.zip`

- [ ] **Final Deliverable**

---

## 7. Team Role Assignments

### Student A — Data & Baseline
- EDA, data cleaning, imputation.
- Baseline LR model on all features.
- Stage 1 feature selection (univariate).
- Coordination on data pipeline.

### Student B — Feature Engineering & Modeling
- Stage 2 multicollinearity filtering, L1 regularization.
- Stage 3 RFECV ordering and top-k feature-set comparison.
- Optuna hyperparameter optimization.
- Primary modeling (LGBM, XGBoost).

### Student C — Metrics, Threshold, Report & Presentation
- Custom scorer implementation & validation.
- Profit curve analysis and threshold tuning.
- Report writing (all sections).
- Presentation slide design and narrative.

---

## 8. Critical Success Factors

1. **Custom Scorer is Non-Negotiable:** All decisions (feature selection, HPO, threshold) must use the business metric, not ROC-AUC or F1.

2. **Feature Count Discipline:** Every variable must justify its 200-point cost. Be ruthless in removal.

3. **Threshold Tuning:** The mathematical break-even (p > 0.333) is a **lower bound**, not a target. Empirical profit curves will be more informative.

4. **Avoid Leakage:** Feature selection happens **inside cross-validation**. Never fit feature selectors on the full training set.

5. **Class Imbalance:** Use `scale_pos_weight` in LGBM/XGBoost to adjust for imbalance. This is critical for precision.

6. **Reproducibility:** Document every step. The code must be runnable from scratch to regenerate all results.

---

## 9. Key References & Resources

- **Scoring Derivation:** See [claude.html](claude.html) for detailed break-even math.
- **Feature Selection Workflow:** See [gemini.md](gemini.md) for the 5-stage funnel and univariate/wrapper/embedded method combinations.
- **Timeline & Milestones:** Checkpoint meetings during Week 1 and Week 3.

---

## 10. Contingency & Risk Mitigation

| Risk | Mitigation |
|---|---|
| Model overfits on Stage 2-selected features | Regularize aggressively (high `min_child_samples`, low `num_leaves`). Test on held-out validation fold. |
| Threshold tuning produces <100 targets | Check that break-even threshold (0.333) isn't filtering out all predictions. Investigate calibration. |
| Report deadline pressure (Week 4) | Prepare figures and table templates in Week 3. Write methodology in parallel during modeling. |
| Presentation time overruns | Rehearse the full 7-minute talk ahead of presentation. Have a 5-minute "quick version" as backup. |
