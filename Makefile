# EditGPT — one command per thing you actually want to do.
#
# `make check` is the contract: if it is green, the branch is mergeable. It runs the
# same steps CI runs, in the same order, so a green local run means a green pipeline.

SHELL := /bin/bash

# pnpm is resolved absolutely rather than through PATH: it is commonly installed to a
# user-local prefix that an interactive shell exports but `make` does not.
UV    ?= uv
PNPM  ?= $(shell command -v pnpm 2>/dev/null || echo $(HOME)/.npm-global/bin/pnpm)
PY    := $(UV) run
WEB   := $(PNPM) --filter @editgpt/web

.PHONY: help setup check check-fast lint types harness test fmt fmt-check \
        bench-grounding bench-removal bench-tune bench-classifier \
        web-lint web-types web-test models eval memory dev dev-lite \
        compose-up compose-down clean

help:  ## Show the targets worth knowing
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup

setup:  ## Install every dependency, Python and Node
	$(UV) sync --all-extras
	$(PNPM) install

models:  ## Download the model weights (~285 MB) into the local cache
	$(PY) python -m editgpt_models.download

# ---------------------------------------------------------------- the contract

check: lint types harness test web-lint web-types web-test  ## Everything CI runs
	@echo ""
	@echo "  ✔ check passed"

check-fast: lint harness test  ## Python only, for the inner loop
	@echo "  ✔ check-fast passed"

lint:  ## Ruff lint + format check
	$(PY) ruff check packages apps evals benchmarks scripts
	$(PY) ruff format --check packages apps evals benchmarks scripts

types:  ## mypy, strict
	$(PY) mypy packages apps

# Must stay in .PHONY: a directory of the same name exists, and without it make
# considers the target up to date and silently skips the check.
harness:  ## Validate harness integrity: links, paths, commands, secrets
	$(PY) python scripts/check_harness.py

test:  ## pytest with coverage, excluding the slow and networked tiers
	$(PY) pytest packages apps evals benchmarks -m "not slow and not live" \
	  --cov --cov-report=term-missing

memory:  ## The RSS regression tier, including the real-model run
	$(PY) pytest packages -m memory -p no:randomly -v

eval:  ## Run the golden image set and print the quality table
	$(PY) python -m evals.run

bench-grounding:  ## Held-out grounding on RefCOCOg (real mask IoU)
	$(PY) python -m benchmarks.grounding --limit 250

bench-removal:  ## Held-out removal on RemovalBench (paired ground truth)
	$(PY) python -m benchmarks.removal --limit 69

bench-tune:  ## Fit thresholds on one split, report them on another
	$(PY) python -m benchmarks.tune --limit 300

bench-classifier:  ## Cross-validate the learned eraser chooser against honest baselines
	$(PY) python -m benchmarks.classifier --folds 5

web-lint:  ## ESLint
	$(WEB) lint

web-types:  ## tsc --noEmit
	$(WEB) typecheck

web-test:  ## Vitest
	$(WEB) test

fmt:  ## Format everything in place
	$(PY) ruff check --fix packages apps evals
	$(PY) ruff format packages apps evals
	$(PNPM) format

fmt-check:  ## Prettier check across the repo
	$(PNPM) format:check

# ---------------------------------------------------------------- running things

dev:  ## Full stack: redis, postgres, gateway, web
	$(MAKE) compose-up
	$(PNPM) --filter @editgpt/web dev

dev-lite:  ## Gateway only. Use this when a model benchmark is also running.
	$(PY) uvicorn editgpt_gateway.app:app --reload --port 8000

compose-up:  ## Start redis and postgres
	docker compose up -d redis postgres

compose-down:
	docker compose down

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache coverage.xml htmlcov .coverage
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
