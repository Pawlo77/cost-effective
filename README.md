# template

Template repository for python projects

## Quick Start

Run the main checks and pipeline entry points:

```bash
make install
make datasets
make test
make pre-commit-all
```

## MLflow

Local MLflow runs are stored under `mlruns/` by default. Launch the UI with:

```bash
make mlflow

# use a different port if 5000 is occupied
make mlflow MLFLOW_PORT=5001

# equivalent direct command
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
```
