ifneq ("$(wildcard .env)","")
	include .env
	export
endif

export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1

.PHONY: help install clean test pre-commit-all

############################
# Repo Maintenance Targets #
############################

help:
	@echo "Available targets:"
	@echo "  make help                   - Show this help message"
	@echo "  make install                - Install dependencies and hooks"
	@echo "  make clean                  - Clean virtual environment and lockfile"
	@echo "  make test                   - Run tests"
	@echo "  make pre-commit-all         - Run pre-commit checks on all files"

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

# pre-commit checks (linting, formatting, type checking)
pre-commit-all:
	uv run pre-commit run --all-files
