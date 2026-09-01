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
        bench-grounding bench-grounding-clipseg bench-removal bench-tune bench-classifier \
        bench-ambiguity \
        web-lint web-types web-test web-build models eval memory dev dev-lite worker \
        migrate migration compose-up compose-s3 compose-down clean \
        e2e load beat sweep

help:  ## Show the targets worth knowing
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup

setup:  ## Install every dependency, Python and Node
	$(UV) sync --all-extras
	$(PNPM) install

models:  ## Download the model weights (~527 MB) into the local cache
	$(PY) python -m editgpt_models.download

# ---------------------------------------------------------------- the contract

check: lint types harness test web-lint web-types web-test web-build  ## Everything CI runs
	@echo ""
	@echo "  ✔ check passed"

check-fast: lint harness test  ## Python only, for the inner loop
	@echo "  ✔ check-fast passed"

lint:  ## Ruff lint + format check
	$(PY) ruff check packages apps evals benchmarks scripts tests
	$(PY) ruff format --check packages apps evals benchmarks scripts tests

types:  ## mypy, strict
	$(PY) mypy packages apps tests

# Must stay in .PHONY: a directory of the same name exists, and without it make
# considers the target up to date and silently skips the check.
harness:  ## Validate harness integrity: links, paths, commands, secrets
	$(PY) python scripts/check_harness.py

# COV_ARGS lets CI ask for an XML report without duplicating the command here. Keeping
# one definition is what makes AGENTS.md's "make check runs exactly what CI runs" true.
COV_ARGS ?= --cov-report=term-missing

test:  ## pytest with coverage, excluding the slow and networked tiers
	$(PY) pytest packages apps evals benchmarks tests -m "not slow and not live" \
	  --cov $(COV_ARGS)

memory:  ## The RSS regression tier, including the real-model run
	$(PY) pytest packages -m memory -p no:randomly -v

eval:  ## Run the golden image set and print the quality table
	$(PY) python -m evals.run

eval-local:  ## The golden set minus the cases that spend provider quota
	$(PY) python -m evals.run --local-only

eval-diff:  ## Compare the last eval run against the recorded baseline
	$(PY) python -m evals.diff

eval-baseline:  ## Record the last eval run as the baseline, after looking at the images
	$(PY) python -m evals.diff --update

bench-grounding:  ## Held-out grounding on RefCOCOg (real mask IoU)
	$(PY) python -m benchmarks.grounding --limit 250 --path detector

bench-grounding-clipseg:  ## The same benchmark down the CLIPSeg path, for comparison
	$(PY) python -m benchmarks.grounding --limit 250 --path clipseg

bench-removal:  ## Held-out removal on both paired datasets, with both quality proxies
	$(PY) python -m benchmarks.removal --limit 69 --dataset both

bench-ambiguity:  ## Could disambiguation help? recall@K and the margin signal
	$(PY) python -m benchmarks.ambiguity --limit 250 --top-k 5

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

# In `check` because it is the only step that *renders* a page. Clerk Core 3 replaced the
# control components with stubs that throw at render time, so lint, typecheck and test all
# passed against an API that could not produce a working site — for as long as Clerk had
# been installed. A check that claims to run what CI runs has to actually run it.
web-build:  ## next build
	$(WEB) build

# Not in `check`: it needs Postgres, Redis, a gateway, a worker and ~550 MB of weights, and
# the signed-in half needs a Clerk user. The smoke half runs with none of that beyond the
# gateway and is what catches "the site does not build" and "the site does not run".
e2e:  ## Full-stack browser tests. Needs the stack up; see docs/RUNBOOK.md
	$(WEB) exec playwright test

# The same paths `make lint` checks. They drifted once, and a file outside the fmt set
# but inside the lint set fails CI with no local way to fix it.
fmt:  ## Format everything in place
	$(PY) ruff check --fix packages apps evals benchmarks scripts tests
	$(PY) ruff format packages apps evals benchmarks scripts tests
	$(PNPM) format

fmt-check:  ## Prettier check across the repo
	$(PNPM) format:check

# ---------------------------------------------------------------- running things

dev:  ## Full stack: redis, postgres, gateway, web
	$(MAKE) compose-up
	$(PNPM) --filter @editgpt/web dev

dev-lite:  ## Gateway only. Use this when a model benchmark is also running.
	$(PY) uvicorn editgpt_gateway.app:app --reload --port 8000

# Concurrency is 1 by design: two edits at once hold two heavy models resident and
# breach the 8 GB budget before either finishes. See apps/worker/AGENTS.md.
worker:  ## Run the Celery worker
	$(PY) celery -A editgpt_worker worker --loglevel=info --concurrency=1

# Not in `make check`: it needs a running stack and takes minutes.
load:  ## Load test the gateway. USERS=10 TIME=60s HOST=http://localhost:8000
	$(UV) run --group load locust -f benchmarks/load/locustfile.py --headless \
		--users $(or $(USERS),10) --spawn-rate 2 --run-time $(or $(TIME),60s) \
		--host $(or $(HOST),http://localhost:8000)

beat:  ## Run the Celery beat scheduler (housekeeping; a worker does not do this alone)
	$(PY) celery -A editgpt_worker beat --loglevel=info

sweep:  ## Report which stored objects a sweep would delete. APPLY=1 to delete them.
	$(PY) python -m editgpt_worker.housekeeping $(if $(APPLY),--apply,)

migrate:  ## Apply database migrations
	cd packages/store && $(UV) run alembic upgrade head

migration:  ## Generate a migration from the models. NAME=... describes the change.
	cd packages/store && $(UV) run alembic revision --autogenerate -m "$(NAME)"

compose-up:  ## Start redis and postgres
	docker compose up -d redis postgres

compose-s3:  ## Also start MinIO, so the object-storage path can be exercised locally
	docker compose --profile s3 up -d minio
	@echo "  MinIO console: http://localhost:9001  (editgpt / editgpt-dev-secret)"

compose-down:
	docker compose down

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache coverage.xml htmlcov .coverage
	find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
