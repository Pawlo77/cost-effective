# Cost‑effective Predictive Modeling

Lightweight research code for cost-aware predictive modeling, feature selection, and hyperparameter tuning with custom business metrics.

## Notebook pipeline

All experiments are orchestrated from `notebooks/`; there are no `scripts/` runners. The classic workflow lives directly in `notebooks/`, while the stricter alternative feature-selection workflow lives in `notebooks/alternative_approach/`.

| Step | Notebook | Purpose | Outputs |
|------|----------|---------|---------|
| 1 | `notebooks/feature_selection.ipynb` | Stages 1–3: filter 500→~26 features, CV drop-column ranking, top-k CV scores; optional MDI, permutation importance, and Boruta rankings with top-1/5/10 overlap and CV profit comparison | `outputs/` (shared) |
| 2 | `notebooks/baseline.ipynb` | Optional reference baselines on all 500 features (`RobustScaler` inside CV); **top-k raw vs PCA** sweep (`k ∈ {1,3,5,10,15,25}`, F1@top-k and ROC AUC only) | `outputs/baseline_results.json`, `outputs/optimal_threshold.json` |
| 2 alt | `notebooks/alternative_approach/modeling_feature_selection_alternative_setup.ipynb` | Alternative feature-selection path: define reusable assessment/tuning folds and a declarative recipe space before any prescreening or model fitting | `outputs/feature_selection_alternative/01_fold_plan_and_recipe_space/` |
| 3a | `notebooks/modeling_topk_profit_curve.ipynb` | Single model, HPO, **max top-k profit curve** on OOF | `outputs/topk_profit_curve/` |
| 3b | `notebooks/modeling_ev_profit_targeting.ipynb` | EV break-even, calibration, combined selective k | `outputs/ev_profit_targeting/` |
| 3c | `notebooks/modeling_rank_fusion.ipynb` | **Committee:** diverse experts, weighted OOF fusion, union of features | `outputs/rank_fusion/` |
| 3d | `notebooks/modeling_segment_targeting.ipynb` | **Segments:** GMM on `top_10`, per-cluster logistic on `top_03` | `outputs/segment_targeting/` |
| 3e | `notebooks/modeling_cluster_split.ipynb` | **Cluster split:** inertia/silhouette k per top-k set (manual k), cluster y profiles, OOF sweep over feature sets, export best | `outputs/cluster_split/` |
| 4 | `notebooks/modeling_compare_all.ipynb` | Compare all approaches (reads each `modeling_summary.json`) | `outputs/approaches_comparison.csv` |


### Alternative Feature-Selection Workflow

The classic path starts with `notebooks/feature_selection.ipynb`: it is fast, compact, and useful for narrowing the 500 raw variables into practical top-k feature sets. The alternative path is stricter. It first writes a reusable fold plan and a declarative recipe space, then later stages can fit prescreening methods and models only inside the data partition allowed for that evaluation step.

Start this path with `notebooks/alternative_approach/modeling_feature_selection_alternative_setup.ipynb`. The notebook does not train models and does not select variables. It only persists the fold assignments, feature-size grid, prescreening recipes, model spec space, and pipeline recipe space under `outputs/feature_selection_alternative/01_fold_plan_and_recipe_space/`.

Use this workflow when validation reliability matters more than iteration speed. Existing heavy artifacts are cache-aware: notebook helpers can reuse previous output directories when present, so rerunning the setup or later stages does not force expensive training from scratch.

Alternative path notebooks, in order:

1. `notebooks/alternative_approach/modeling_feature_selection_alternative_setup.ipynb`
2. `notebooks/alternative_approach/modeling_feature_selection_alternative_inner_selection.ipynb`
3. `notebooks/alternative_approach/modeling_feature_selection_alternative_evaluation.ipynb`
4. `notebooks/alternative_approach/modeling_feature_selection_alternative_refit_ranking.ipynb`

Re-run notebooks after code changes so artifacts stay in sync.

### Which modeling notebook?

- **`modeling_topk_profit_curve`** — picks k ≈ 1000 when the OOF profit curve is flat near the cap (weak ranking → mails almost everyone allowed).
- **`modeling_ev_profit_targeting`** — uses expected value per contact (15p−5), isotonic calibration on OOF, and a conservative inner-CV k estimate so k is usually **much smaller** when the model is unselective.
- **`modeling_rank_fusion`** — fuses logistic/LightGBM/Borda experts on different top-k subsets; weights from CV business score.
- **`modeling_segment_targeting`** — GMM customer segments, per-segment rankers, global top-k on OOF.
- **`modeling_cluster_split`** — KMeans on scaled top-k features (no PCA); elbow/silhouette plots per feature set; you set `N_CLUSTERS_BY_FEATURE_SET`, inspect cluster target rates, then OOF-sweep all top-k sets and export the best run.

All modeling notebooks read Stage 3 outputs from `outputs/` and write **only** into their own subdirectory.

## Quick Start

```bash
make install
make test
```

```text
# Classic path
notebooks/feature_selection.ipynb
  → notebooks/modeling_ev_profit_targeting.ipynb
  → notebooks/modeling_compare_all.ipynb

# Alternative feature-selection path
notebooks/alternative_approach/modeling_feature_selection_alternative_setup.ipynb
  → notebooks/alternative_approach/modeling_feature_selection_alternative_inner_selection.ipynb
  → notebooks/alternative_approach/modeling_feature_selection_alternative_evaluation.ipynb
  → notebooks/alternative_approach/modeling_feature_selection_alternative_refit_ranking.ipynb
```

## Output layout

**Shared (feature selection):**

- `outputs/feature_selection_results.csv`
- `outputs/stage3_feature_set_scores.csv`

**`outputs/topk_profit_curve/`** (notebook 3a):

- `model_comparison_base.csv`, `model_comparison.csv`
- `hpo/hpo_summary.json`
- `model_predictions.csv`, `modeling_summary.json`
- `{STUDENT_PREFIX}_obs.txt`, `{STUDENT_PREFIX}_vars.txt`

**`outputs/ev_profit_targeting/`** (notebook 3b):

- Same filenames as above, independent copies for this approach

Set `SUBMISSION_PREFIX` in each modeling notebook before export.

## Library code

Shared logic lives in `src/cost_effective/`:

- [src/cost_effective/models/modeling.py](src/cost_effective/models/modeling.py) — CV comparison, OOF probabilities, profit curve, final predict
- [src/cost_effective/models/profit_targeting.py](src/cost_effective/models/profit_targeting.py) — EV k selection, calibration, test export helpers (`TargetingConfig`, `choose_targeting_k`, …)
- [src/cost_effective/models/cluster_k_selection.py](src/cost_effective/models/cluster_k_selection.py) — KMeans inertia/silhouette diagnostics per k
- [src/cost_effective/models/cluster_split_targeting.py](src/cost_effective/models/cluster_split_targeting.py) — per-cluster logistic OOF and test scoring
- [src/cost_effective/paths.py](src/cost_effective/paths.py) — approach output dirs (`APPROACH_*`, `approach_outputs_dir()`)
- [src/cost_effective/approach_comparison.py](src/cost_effective/approach_comparison.py) — load/compare `modeling_summary.json` across approaches
- [src/cost_effective/utils.py](src/cost_effective/utils.py) — param grids, HPO, submission I/O, feature name parsing
- [src/cost_effective/notebook_setup.py](src/cost_effective/notebook_setup.py) — `setup_modeling_notebook()` (style + paths + data in one call)
- [src/cost_effective/notebook_workflows.py](src/cost_effective/notebook_workflows.py) — compare/tune/OOF/export steps used by modeling notebooks
- [src/cost_effective/notebook_artifacts.py](src/cost_effective/notebook_artifacts.py) — cache-aware CSV/JSON and output-directory helpers used by heavier notebooks
- [src/cost_effective/models/feature_selection_alternative_setup.py](src/cost_effective/models/feature_selection_alternative_setup.py) — public setup aliases for the alternative feature-selection workflow
- [src/cost_effective/plots.py](src/cost_effective/plots.py) — profit-curve, EV dashboard, and cluster k / y-profile figures

Notebooks import these modules; they are not CLI scripts.

## Contributing

1. Run `make pre-commit-all` before committing.
2. Keep code style consistent (project uses `ruff`).
