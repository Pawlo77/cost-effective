# Cost-Effective Predictive Modeling — Plan

## 1. Problem and objective

Build a model that maximizes **profit**, not classification accuracy. The leaderboard score is:

$$\text{Score} = (TP \times 10) - (FP \times 5) - (N_{\text{vars}} \times 200)$$

- **TP:** selected customer accepts the offer (+10).
- **FP:** selected customer does not accept (−5).
- **$N_{\text{vars}}$:** number of features listed in the submission (−200 each).

We may contact at most **1,000** test customers. Indices must be ranked by predicted probability; only the first 1,000 count. Variables used must be listed separately.

Data: 5,000 training rows (500 features, binary label), 5,000 test rows (labels hidden).

Three forces drive every design choice:

1. High reward for true positives pushes recall among the top of the ranking.
2. False positives are cheap relative to a missed TP but add up if we mail too many people.
3. Each feature costs 200 points up front — a 50-feature model starts at −10,000 before a single correct contact.

---

## 2. Mathematical framework

### 2.1 Expected value per contact

If we contact one customer with estimated conversion probability $p$:

$$E[\text{contact}] = p \cdot 10 + (1-p) \cdot (-5) = 15p - 5$$

Contact only when $E > 0$:

$$15p - 5 > 0 \quad \Rightarrow \quad p > \frac{1}{3} \approx 0.333$$

A default classifier threshold of 0.5 is conservative under this cost matrix. Ranking by $p$ or by $15p-5$ is equivalent for ordering (affine map).

### 2.2 Feature cost recovery

One extra feature costs 200 score points. It must earn back at least 200 via contacts:

- **20 additional TPs** at +10 each, or
- **40 fewer FPs** at −5 each, or
- a mix with the same net.

Use this as a sanity check when adding variables after Stage 3, not as a hard rule (correlated features share lift).

### 2.3 Top-$k$ selection under a cap

Let $N = 5000$ training rows. Sort customers by $\hat{p}$ descending. For each $k \in \{1,\ldots,\min(1000,N)\}$:

- Take the top $k$ indices as predicted positive.
- Count $TP_k$, $FP_k$ on labeled train (OOF probabilities for honest evaluation).
- **Profit at $k$:** $\text{Score}_k = 10 \cdot TP_k - 5 \cdot FP_k - 200 \cdot N_{\text{vars}}$.

Submission uses one $k \leq 1000$ and the corresponding top-$k$ test indices. The business scorer in CV must use the same top-$k$ logic with `max_k=1000`, not a full-column threshold at $p=1/3`.

### 2.4 Capped campaign identity

If we always fill the list to $k$ contacts, $TP_k + FP_k = k$, so

$$\text{Score}_k = 10 \cdot TP_k - 5(k - TP_k) - 200 N_{\text{vars}} = 15 \cdot TP_k - 5k - 200 N_{\text{vars}}.$$

Marginal customer at rank $k$ (probability $p_k$) contributes expected $15p_k - 5$ before feature cost. That links the EV rule to the profit curve.

---

## 3. Pipeline and artifacts

```
notebooks/feature_selection.ipynb
    → outputs/feature_selection_results.csv      (Stage 3 ranking)
    → outputs/stage3_feature_set_scores.csv      (top-k subset CV scores)

notebooks/baseline.ipynb                         (optional; 500 features)

notebooks/modeling_topk_profit_curve.ipynb
    → outputs/topk_profit_curve/

notebooks/modeling_ev_profit_targeting.ipynb
    → outputs/ev_profit_targeting/

notebooks/modeling_rank_fusion.ipynb
    → outputs/rank_fusion/

notebooks/modeling_segment_targeting.ipynb
    → outputs/segment_targeting/

notebooks/modeling_cluster_split.ipynb
    → outputs/cluster_split/

notebooks/modeling_compare_all.ipynb
    → outputs/approaches_comparison.csv   (reads all modeling_summary.json)
```

Implementation lives in [src/cost_effective/](src/cost_effective/). Notebooks orchestrate; they are not the source of truth for metric definitions.

**Leakage:** feature ranking, HPO, and $k$ selection use cross-validation on train only. Final model fits all train labels once; test receives ranked indices only.

---

## 4. Feature selection

500 inputs are mostly noise. Reduce in three stages, then expose ranked subsets for modeling.

### 4.1 Stage 1 — univariate filter (500 → ~80)

**Goal:** drop dead columns and keep signal for downstream steps.

**Steps:**

1. **Variance threshold** — remove near-constant features (e.g. variance $< 0.01$).
2. **Mutual information** — score each feature vs. label; keep top ~50–80 by MI.
3. **LightGBM importance** — shallow tree on all remaining columns; keep top ~80 by split gain.

**Output:** union or intersect of top lists (~80 columns). Implemented in [notebooks/feature_selection.ipynb](notebooks/feature_selection.ipynb).

### 4.2 Stage 2 — redundancy and sparsity (80 → ~26)

**Goal:** remove collinearity and enforce a sparse linear structure before ranking.

**Steps:**

1. **Correlation pruning** — among pairs with $|r| \geq 0.85$, drop the member with weaker univariate association to the label.
2. **VIF** — iteratively remove features with VIF $> 5$.
3. **LASSO path** — logistic regression with L1 penalty; sweep $C$; pick a solution in a sensible sparsity band (~20–30 non-zero coefficients) using cross-validated business score, not ROC-AUC alone.

**Output:** Stage 2 feature list (~26).

### 4.3 Stage 3 — order and top-$k$ subsets (~26 ranked)

**Goal:** rank Stage 2 features for profit and precompute subset scores.

**Method:** **drop-column CV ranking** with the business scorer (`max_k=1000`), implemented as `rank_features_drop_column_cv`. For each feature, measure how much mean CV business drops when that column is removed from the Stage 2 matrix. Low drop ⇒ important. This runs inside CV and avoids fitting RFECV on the full training matrix in one shot (which risks optimistic bias).

**Subsets:** prefixes of the ranked list — `top_01`, `top_03`, `top_05`, …, `top_26`. For each subset, run model comparison CV and store means in `stage3_feature_set_scores.csv`.

**Output:** `feature_selection_results.csv` (rank table) and subset score table for modeling notebooks.

### 4.4 Optional ranking methods (notebook-only)

After Stage 3, [notebooks/feature_selection.ipynb](notebooks/feature_selection.ipynb) can compare extra rankings on the Stage 2 matrix:

- **MDI** — Random Forest mean decrease in impurity
- **Permutation importance** — sklearn on a fitted RF
- **Boruta** — `boruta_py` (`boruta` in `pyproject.toml`)

Sections compare top-1 / top-5 / top-10 overlap across methods and CV business score for each method × $k$. Does not replace `feature_selection_results.csv` (Stage 3 drop-column rank remains the modeling default).

---

## 5. Business metric implementation

Defined in [src/cost_effective/dataset/utils.py](src/cost_effective/dataset/utils.py) and used everywhere for model selection.

| Function | Role |
|----------|------|
| `business_score_from_proba` | Sort by $\hat{p}$, take top $k \leq 1000$, compute TP/FP, subtract $200 \cdot N_{\text{vars}}$ |
| `custom_scorer` | sklearn CV scorer wrapping the above on validation folds |
| `business_scorer_no_var_penalty` | TP/FP only (diagnostic) |
| `f1_at_optimal_top_k` / `f1_scorer_wrapper` | F1 at the $k$ that maximizes F1 within cap — diagnostic only, not the objective |

**Profit curve** (`build_profit_curve` in [src/cost_effective/models/modeling.py](src/cost_effective/models/modeling.py)): from OOF $\hat{p}$ on train, build cumulative TP/FP vs. $k$, compute $\text{Score}_k$ at each $k$, record argmax $k$, threshold at rank $k$, and break-even count of OOF rows with $p \geq 1/3$.

---

## 6. Modeling

### 6.1 Algorithms

| Model | Role |
|-------|------|
| **Logistic regression** | Sparse linear baseline; stable on few features; `RobustScaler` + L2/L1 in pipeline |
| **LightGBM** | Main nonlinear candidate; `scale_pos_weight = n_- / n_+` |
| **XGBoost** | Secondary tree model; same imbalance handling |

Compare families on each top-$k$ subset from Stage 3 via `compare_models_on_feature_sets` (5-fold stratified CV, business scorer, `max_k=1000`).

### 6.2 Class imbalance

Training is roughly balanced but trees still benefit from:

- `scale_pos_weight` in LightGBM/XGBoost,
- `class_weight='balanced'` for logistic regression,
- optional SMOTE only inside CV folds if experimented with — never on test.

### 6.3 Hyperparameter optimization

Run on the **winning feature matrix only** (not on full 500 columns).

- **Logistic:** grid over `C` and penalty (`LOGISTIC_PARAM_GRID` in [src/cost_effective/utils.py](src/cost_effective/utils.py)).
- **LightGBM / XGBoost:** randomized search over depth, leaves, learning rate, subsampling (`LGB_PARAM_DIST`, `XGB_PARAM_DIST`).
- **Refit metric:** CV business score from [src/cost_effective/dataset/utils.py](src/cost_effective/dataset/utils.py) `custom_scorer`, not ROC-AUC.
- **Persistence:** `outputs/<approach>/hpo/` JSON per model.

`run_winner_hyperparameter_search` in [src/cost_effective/utils.py](src/cost_effective/utils.py) standardizes this for both notebooks.

### 6.4 Out-of-fold probabilities

Before choosing $k$ or exporting test indices:

1. `compute_oof_probabilities` — 5-fold stratified OOF $\hat{p}$ on train for the tuned model on the selected features.
2. All $k$ rules below are computed from these OOF scores so we do not tune $k$ on the same rows used to fit the final model.

Final step: `fit_final_model_and_predict` retrains on all train data, scores test, returns top-$k$ indices by test $\hat{p}$.

---

## 7. Modeling approaches

Single-model notebooks (A–B) share compare → HPO → OOF → export. Committee/segment/cluster notebooks (C–E) build OOF scores differently, then use the same top-$k$ profit curve for $k$ (cluster split pools per-cluster OOF). Outputs go to separate folders (`paths.py`).

| ID | Folder | Module |
|----|--------|--------|
| A | `topk_profit_curve` | `modeling.py` |
| B | `ev_profit_targeting` | `profit_targeting.py` |
| C | `rank_fusion` | `rank_fusion.py` |
| D | `segment_targeting` | `segment_targeting.py` |
| E | `cluster_split` | `cluster_split_targeting.py`, `cluster_k_selection.py` |

### 7.1 Approach A — top-$k$ profit curve maximum

**Notebook:** [notebooks/modeling_topk_profit_curve.ipynb](notebooks/modeling_topk_profit_curve.ipynb)

1. Load Stage 3 artifacts via `setup_modeling_notebook`.
2. Compare models/subsets; pick CV business winner.
3. HPO; build `tuned_factory`.
4. OOF probabilities.
5. **`evaluate_topk_oof`:** `build_profit_curve` + `build_f1_curve`; set $k^* = \arg\max_k \text{Score}_k$ on the OOF curve (subject to `best_k_break_even` cap inside `build_profit_curve`).
6. Plot: `plot_topk_oof_profit_curve`.
7. Export: `export_topk_test_predictions` with `n_targets=k^*`.

Use when the OOF curve has a clear peak below 1000. If the curve is flat near the cap, this approach tends to push $k \rightarrow 1000$.

### 7.2 Approach B — EV profit targeting (selective $k$)

**Notebook:** [notebooks/modeling_ev_profit_targeting.ipynb](notebooks/modeling_ev_profit_targeting.ipynb)

Same steps through OOF, then `choose_targeting_k` in [src/cost_effective/models/profit_targeting.py](src/cost_effective/models/profit_targeting.py) with `TargetingConfig`:

| Rule | Definition |
|------|------------|
| **EV threshold** `k_ev` | Walk down the OOF ranking; include customers while $\hat{p} \geq 1/3$ (or optional higher floor); stop at first fail; cap 1000 |
| **Elbow** `k_elbow` | Smallest $k$ whose OOF $\text{Score}_k \geq \alpha \cdot \max_k \text{Score}_k$ (default $\alpha=0.95$) |
| **Nested CV** `k_nested` | On each fold, evaluate a grid of $k$ values; pick $k$ maximizing mean fold business score; default grid 50–1000 |
| **Combined** (default) | $k = \min(k_{ev}, k_{elbow}, k_{nested})$ — conservative when EV and nested still want ~1000 |

Optional **isotonic calibration** on OOF $(\hat{p}, y)$ before $k$ rules so thresholds are less distorted by uncalibrated logits.

Test export: `rank_test_indices` applies train-chosen $k$ and the same break-even floor to test probabilities.

Plot: `plot_ev_targeting_dashboard` (profit curve + top-200 expected value).

Prefer this approach when the ranking is weak and the profit curve plateaus: elbow and combined shrink $k$ without relying on a single argmax at the cap.

### 7.3 Approach C — rank fusion committee

**Notebook:** [notebooks/modeling_rank_fusion.ipynb](notebooks/modeling_rank_fusion.ipynb)

**Module:** [src/cost_effective/models/rank_fusion.py](src/cost_effective/models/rank_fusion.py)

Default experts (diverse by design):

| Expert | Features | Model |
|--------|----------|--------|
| `logistic_top03` | `top_03` | Logistic pipeline |
| `lightgbm_top05` | `top_05` | LightGBM |
| `logistic_top08` | `top_08` | Logistic pipeline |
| `borda_top05` | `top_05` | `BordaRankClassifier` (rank fusion, density-lab style) |

Steps:

1. `run_rank_fusion_pipeline` → per-expert CV business + OOF $\hat{p}_e$.
2. Weights $w_e \propto \max(0, \text{CV business}_e)$; fused $\hat{p} = \sum_e w_e \hat{p}_e$.
3. `evaluate_topk_oof` on fused scores; feature penalty uses **union** of expert columns (`submission_features`).
4. `export_fusion_test_predictions` — per-expert refit on full train, fused test ranking.

Inspired by [machine-learning/ensambles](machine-learning/ensambles) (weighted combination) and [machine-learning/features_selection](machine-learning/features_selection) (multiple ranked views).

### 7.4 Approach D — segment-aware targeting

**Notebook:** [notebooks/modeling_segment_targeting.ipynb](notebooks/modeling_segment_targeting.ipynb)

**Module:** [src/cost_effective/models/segment_targeting.py](src/cost_effective/models/segment_targeting.py)

1. Fit **GMM** on scaled (`top_10`, optional PCA) features inside each CV fold.
2. Per cluster: logistic on `top_03` if $n_{\text{cluster}} \geq 80$, else fold fallback logistic.
3. Pool OOF $\hat{p}$; `evaluate_topk_oof` + export with **`top_03` vars only** (segments are latent; `top_10` used only for clustering).

Inspired by [machine-learning/density](machine-learning/density) (mixture structure) and [machine-learning/semi_supervised](machine-learning/semi_supervised) (structure + labels), framed as supervised segmentation rather than pure clustering.

### 7.5 Approach E — cluster split targeting

**Notebook:** [notebooks/modeling_cluster_split.ipynb](notebooks/modeling_cluster_split.ipynb)

**Modules:** [src/cost_effective/models/cluster_split_targeting.py](src/cost_effective/models/cluster_split_targeting.py), [src/cost_effective/models/cluster_k_selection.py](src/cost_effective/models/cluster_k_selection.py)

Same top-$k$ feature sets as other notebooks (`top_01` … from Stage 3). For each candidate set:

1. **K diagnostics** — `kmeans_k_diagnostics`: inertia + silhouette vs. $k$ on scaled features (no PCA for clustering). Plots in `plot_cluster_k_selection_grid`. **No automatic elbow pick** — user sets `N_CLUSTERS_BY_FEATURE_SET` / `DEFAULT_N_CLUSTERS` after inspecting curves.
2. **Cluster y profiles** — `analyze_cluster_y_distributions`: size, $P(y=1)$, lift vs. global rate; `plot_cluster_y_profiles_grid` / `plot_cluster_y_profile`.
3. **OOF sweep** — `compare_cluster_split_feature_sets`: KMeans on scaled top-$k$, per-cluster logistic (fallback model for small clusters), pooled OOF $\hat{p}$; pick best feature set by OOF business score.
4. **Export** — `export_cluster_split_test_predictions`; submission vars = that run’s top-$k$ list.

Outputs under `outputs/cluster_split/` (`cluster_k_*.csv`, `cluster_y_*.csv`, `feature_set_comparison.csv`, `modeling_summary.json`, submission files). Included in `modeling_compare_all.ipynb` via `approach_comparison.py`.

### 7.6 Diagnostics (all approaches)

- OOF confusion matrix at chosen $k$ (`oof_confusion_matrix`).
- F1 curve vs. business curve — report separately; F1-optimal $k$ is not the business optimum.
- `business_score_no_var_penalty` in model tables to see TP/FP tradeoff before feature tax.

---

## 8. Baseline notebook

`baseline.ipynb` runs reference floors (k=0, prior/stratified/uniform dummies, random rankers, 1-feature LR, 500-feature LR) with top-$k$ scorers and per-fold scaling. Useful for scorer validation and feature-tax illustration. Outputs: [outputs/baseline_results.json](outputs/baseline_results.json), [outputs/optimal_threshold.json](outputs/optimal_threshold.json) (500-feature LR OOF only). Not used for submission.

**Added:** optional **top-$k$ raw vs PCA** comparison for $k \in \{1, 3, 5, 10, 15, 25\}$ — logistic on ranked raw features vs. first $k$ PCA components (500 features scaled, PCA fit once with 25 components). Metrics: **top-$k$ F1 and ROC AUC only** (no business / variable penalty in that table).

---

## 9. Code structure

| Module | Contents |
|--------|----------|
| [src/cost_effective/dataset/loading.py](src/cost_effective/dataset/loading.py) | Train/test loaders, project root |
| [src/cost_effective/dataset/utils.py](src/cost_effective/dataset/utils.py) | Scorers, break-even helpers |
| [src/cost_effective/models/modeling.py](src/cost_effective/models/modeling.py) | Feature ranking, CV compare, OOF, curves, final predict |
| [src/cost_effective/models/profit_targeting.py](src/cost_effective/models/profit_targeting.py) | `TargetingConfig`, EV/$k$ selection, calibration |
| [src/cost_effective/models/rank_fusion.py](src/cost_effective/models/rank_fusion.py) | Committee experts, `BordaRankClassifier`, fusion weights |
| [src/cost_effective/models/segment_targeting.py](src/cost_effective/models/segment_targeting.py) | GMM segments, per-cluster OOF |
| [src/cost_effective/models/cluster_k_selection.py](src/cost_effective/models/cluster_k_selection.py) | KMeans inertia / silhouette diagnostics |
| [src/cost_effective/models/cluster_split_targeting.py](src/cost_effective/models/cluster_split_targeting.py) | KMeans split, per-cluster logistic OOF and test |
| [src/cost_effective/approach_comparison.py](src/cost_effective/approach_comparison.py) | Load `modeling_summary.json` from all approaches |
| [src/cost_effective/utils.py](src/cost_effective/utils.py) | HPO grids, submission files, `load_modeling_stage_data` |
| [src/cost_effective/notebook_setup.py](src/cost_effective/notebook_setup.py) | `setup_modeling_notebook`, `ModelingNotebookContext` |
| [src/cost_effective/notebook_workflows.py](src/cost_effective/notebook_workflows.py) | Compare, tune, `evaluate_topk_oof`, `evaluate_ev_oof`, export helpers |
| [src/cost_effective/plots.py](src/cost_effective/plots.py) | Profit/EV curves, cluster k-grid, cluster y-profile plots |
| [src/cost_effective/paths.py](src/cost_effective/paths.py) | `APPROACH_*` constants and per-approach output dirs |

Tests under `tests/` lock scorer behavior, hyperparam attachment, cluster split OOF, and k diagnostics.

---

## 10. Principles

1. **One metric for decisions** — feature subsets, HPO refit, and $k$ use the business score (top-$k$, cap 1000).
2. **Pay for every variable** — prefer smaller top-$k$ feature sets unless CV business clearly gains enough to cover 200 per added column.
3. **OOF before test** — $k$ and thresholds come from OOF train scores; test only sees final ranking.
4. **Do not confuse diagnostics with the objective** — ROC-AUC, F1 at F1-optimal $k$, and “no var penalty” scores are supporting plots only.
5. **Reproducibility** — fixed seeds, saved CSV/JSON under approach-specific output dirs, notebooks cleared and rerun after code changes.

---

## 11. Deliverables

- `STUDENT1_STUDENT2_STUDENT3_obs.txt` — up to 1000 test indices, one per line, best first
- `STUDENT1_STUDENT2_STUDENT3_vars.txt` — 0-based feature indices used
- `report.pdf`, presentation slides
- `code/` — repository with notebooks and `src/`

Submission prefix configured in `utils.DEFAULT_SUBMISSION_PREFIX`.

---

## 12. References

- Formal task statement: [docs/task.md](docs/task.md)
- Scorer reference implementation: [docs/claude.py](docs/claude.py)
- Extended background notes: [docs/gemini.md](docs/gemini.md) (may lag the codebase)
