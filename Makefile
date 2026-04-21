ifneq ("$(wildcard .env)","")
	include .env
	export
endif

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1

.PHONY: help install clean test pre-commit pre-commit-all datasets mlflow mlflow-stop

MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5005
MLFLOW_WORKERS ?= 1

############################
# Repo Maintenance Targets #
############################

help:
	@echo "Available targets:"
	@echo "  make help                   - Show this help message"
	@echo "  make install                - Install dependencies and hooks"
	@echo "  make clean                  - Clean virtual environment and lockfile"
	@echo "  make test                   - Run tests"
	@echo "  make pre-commit             - Run pre-commit checks on changed files"
	@echo "  make pre-commit-all         - Run pre-commit checks on all files"
	@echo "  make datasets               - Download/build all dataset variants (default, small, extended)"
	@echo "  make mlflow                 - Launch MLflow UI for local runs"
	@echo "  make mlflow-stop            - Stop local MLflow UI processes"

# install dependencies and pre-commit hooks
install:
	uv sync --all-groups
	uv run pre-commit install

# clean up virtual environment and lockfile
clean:
	rm -rf .venv
	rm -rf uv.lock

# run tests with pytest
test:
	uv run pytest -v

# pre-commit checks on changed files only
pre-commit:
	uv run pre-commit run

# pre-commit checks (linting, formatting, type checking)
pre-commit-all:
	uv run pre-commit run --all-files

# Download/build all dataset variants used by the project.
datasets:
	echo "Note implemented."

#########################
# Orchestration Targets #
#########################

#################
# Other Targets #
#################

# Launch MLflow UI for local runs
mlflow:
	@PORT="$(MLFLOW_PORT)"; \
	if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $$PORT is already in use. Run 'make mlflow-stop' or use another port: make mlflow MLFLOW_PORT=5001"; \
		lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		exit 1; \
	fi; \
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host "$(MLFLOW_HOST)" --port "$$PORT" --workers "$(MLFLOW_WORKERS)"

# Stop local MLflow UI processes
mlflow-stop:
	@PORT="$(MLFLOW_PORT)"; \
	PIDS="$$( ( \
		pgrep -f 'mlflow.server.fastapi_app' || true; \
		pgrep -f 'python -m mlflow' || true; \
		pgrep -f 'mlflow ui' || true; \
		pgrep -f 'mlflow server' || true \
	) | sort -u )"; \
	if [ -z "$$PIDS" ] && command -v lsof >/dev/null 2>&1; then \
		PORT_PIDS="$$(lsof -tiTCP:"$$PORT" -sTCP:LISTEN 2>/dev/null || true)"; \
		if [ -n "$$PORT_PIDS" ]; then \
			for PID in $$PORT_PIDS; do \
				ARGS="$$(ps -p $$PID -o args= 2>/dev/null || true)"; \
				case "$$ARGS" in \
					*mlflow*|*fastapi_app:app*) PIDS="$$PIDS $$PID" ;; \
				esac; \
			done; \
			PIDS="$$(printf '%s\n' $$PIDS | awk 'NF' | sort -u | tr '\n' ' ')"; \
		fi; \
	fi; \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping MLflow UI processes: $$PIDS"; \
		kill $$PIDS; \
	else \
		echo "No MLflow UI process found."; \
		if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
			echo "Note: port $$PORT is occupied by a non-MLflow process:"; \
			lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		fi; \
	fi
