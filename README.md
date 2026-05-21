# Cost‑effective Predictive Modeling

Lightweight research code for cost-aware predictive modeling, feature selection, and hyperparameter tuning with custom business metrics.

## Quick Start

Prerequisites: Python 3.10+ and a working virtual environment. From the project root:

```bash
# install dependencies into the active environment
make install

# prepare datasets (downloads / preprocessing)
make datasets

# run the test suite
make test

# run all pre-commit hooks locally
make pre-commit-all
```

## Notebooks & Examples

- Interactive exploration and experiments live in the `notebooks/` folder.
- Recommended to open `notebooks/baseline.ipynb` and `notebooks/modeling.ipynb` for baseline experiments and modeling flows.

## Outputs

- `outputs/` contains experiment results, HPO summaries and model predictions.
- Key files: `outputs/baseline_results.json`, `outputs/model_predictions.csv`.

## Contributing

1. Run `make pre-commit-all` before committing.
2. Keep code style consistent (project uses `ruff` + black formatting).
3. Add tests under `tests/` for new logic.

## License

See the `LICENSE` file in the repository root.
