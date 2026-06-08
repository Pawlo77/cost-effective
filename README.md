# Cost-effective Predictive Modeling

Cost-aware customer targeting. Select which observations to contact given the business objective:

```
score = 10·TP − 5·FP − 200·|features|
```

---

## Best configuration

Produced by the nested-CV alternative pipeline.

| | |
|---|---|
| Prescreening | sparse GAM SPAM (`ps_mean_0007`) |
| Features | `var_175  var_190  var_214  var_341  var_379  var_482` |
| Model | ExtraTreesClassifier — balanced, max_depth=5, min_samples_leaf=16, n_estimators=400 |
| Nested-CV OOF score | **5 425** &nbsp;(pooled 5×1 000 disjoint outer-val predictions) |
| In-sample upper bound | 5 790 &nbsp;(train = eval) |
| Submission | top-997 observations &nbsp;(≈ 20% contact rate, k optimised on pooled OOF) |

To reproduce predictions from `X_test` only — run `notebooks/alternative_approach/final_submission_pipeline.ipynb`.

---

## Pipelines

### Classic

Standard CV. Fast iteration, no strict data-access contract.

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `feature_selection.ipynb` | Filter 500 → ~26 features; drop-column CV ranking |
| 2 | `baseline.ipynb` | Reference baselines; top-k raw vs PCA |
| 3a | `modeling_topk_profit_curve.ipynb` | Top-k profit curve, HPO |
| 3b | `modeling_ev_profit_targeting.ipynb` | EV break-even, isotonic calibration |
| 3c | `modeling_rank_fusion.ipynb` | Ensemble of diverse experts, weighted OOF fusion |
| 3d | `modeling_segment_targeting.ipynb` | GMM segments, per-segment rankers |
| 3e | `modeling_cluster_split.ipynb` | KMeans on top-k features, OOF sweep |
| 4 | `modeling_compare_all.ipynb` | Compare all approaches |

### Alternative (nested CV)

Strict data-access contract. Fold plan and recipe space are fixed before any fitting. Prescreening is refitted inside each outer fold. Finalist selection uses rate-matched, population-scaled scores.

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `alternative_approach/…_setup.ipynb` | Outer/inner folds, prescreening recipes, model spec space — no fitting |
| 02 | `alternative_approach/…_inner_selection.ipynb` | Inner nested-CV: score all pipeline recipes |
| 03 | `alternative_approach/…_evaluation.ipynb` | Outer nested-CV: rate-matched finalist selection, pooled OOF scores |
| 04 | `alternative_approach/…_refit_ranking.ipynb` | Refit on full X_train, score X_test, write submission files |
| — | `alternative_approach/score_comparison_oof_vs_test.ipynb` | In-sample / pooled OOF / per-fold / test score diagnostics |
| — | `alternative_approach/final_submission_pipeline.ipynb` | Self-contained: load X_test → submission files |

---

## Project structure

```
├── data/                   x_train.txt, y_train.txt, x_test.txt (5 000 × 500)
├── notebooks/
│   ├── *.ipynb             classic pipeline
│   └── alternative_approach/
│       ├── modeling_feature_selection_alternative_{setup,inner_selection,evaluation,refit_ranking}.ipynb
│       ├── score_comparison_oof_vs_test.ipynb
│       └── final_submission_pipeline.ipynb
├── src/cost_effective/
│   ├── dataset/            loading, SPAM prescreening, feature filters
│   ├── models/             business scoring, nested-CV, modeling helpers
│   └── *.py                plots, notebook utilities, paths
├── outputs/
│   ├── feature_selection_alternative/   01–04 stage artefacts + final_submission/
│   └── {ev_profit_targeting,rank_fusion,segment_targeting,cluster_split,topk_profit_curve}/
├── tests/                  pytest suite
├── report/                 LaTeX source + PDF
├── docs/                   task description, planning notes
├── Makefile
└── pyproject.toml
```

---

## Quick start

```bash
make install && make test
```

One-shot submission (best config already embedded):

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/alternative_approach/final_submission_pipeline.ipynb
```

Full alternative pipeline from scratch — run notebooks 01 → 02 → 03 → 04 in order.

---

## Outputs

```
outputs/feature_selection_alternative/
  01_fold_plan_and_recipe_space/     fold assignments, recipe space
  02_inner_recipe_selection/         inner-CV scores
  03_fold_evaluation/                outer-CV scores, stable_outer_configurations.csv
  04_refit_and_test_ranking/         final_test_predictions.csv, _obs.txt, _vars.txt
  final_submission/                  output of final_submission_pipeline.ipynb
```

---

## Contributing

Run `make pre-commit-all` before committing. Code style enforced by `ruff`.
