# Cost‑effective Predictive Modeling

Lightweight research code for cost-aware predictive modeling, feature selection, and hyperparameter tuning with custom business metrics.

## Notebook pipeline (run in order)

All experiments are orchestrated from `notebooks/` — there are no `scripts/` runners.

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `notebooks/feature_selection.ipynb` | Stages 1–3: filter 500→~26 features, CV drop-column ranking, top-k CV scores |
| 2 | `notebooks/baseline.ipynb` | Optional reference baseline on full scaled feature space |
| 3 | `notebooks/modeling.ipynb` | Model comparison, HPO on winner, OOF profit curve, test export, submission files |

Re-run notebooks top-to-bottom after code changes so outputs under `outputs/` stay in sync.

## Quick Start

```bash
make install
make test
```

Open Jupyter and execute the three notebooks in the order above.

## Outputs

- `outputs/feature_selection_results.csv` — ranked Stage 2 features
- `outputs/stage3_feature_set_scores.csv` — top-k subset CV business scores
- `outputs/model_comparison_base.csv`, `outputs/model_comparison.csv`
- `outputs/model_predictions.csv`, `outputs/modeling_summary.json`
- `outputs/{STUDENT_PREFIX}_obs.txt`, `outputs/{STUDENT_PREFIX}_vars.txt` (set prefix in modeling notebook)

## Library code

Shared logic lives in `src/cost_effective/` (scorers, loaders, modeling helpers). Notebooks import these modules; they are not meant to be run as CLI scripts.

## Contributing

1. Run `make pre-commit-all` before committing.
2. Keep code style consistent (project uses `ruff`).
