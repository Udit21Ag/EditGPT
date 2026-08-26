
.PHONY: setup setup-all models cf-check report verify clean \
        bench-sam bench-lama bench-migan bench-erasers bench-pipeline \
        bench-text2edit bench-fullres bench-clipseg bench-remote bench-final

setup:
	uv sync

setup-all:
	uv sync --extra torch --extra remote

models:
	uv run python scripts/fetch_models.py

bench-lama:
	uv run python -m bench.harness lama

bench-sam:
	uv run python -m bench.harness mobilesam

bench-pipeline:
	uv run python -m bench.harness pipeline

bench-final:
	uv run python -m bench.harness final

bench-migan:
	uv run python -m bench.harness migan

bench-erasers:
	uv run python -m bench.harness erasers

bench-fullres:
	uv run python -m bench.harness fullres

bench-text2edit:
	uv run python -m bench.harness text2edit

bench-clipseg:
	uv sync --extra torch --extra remote
	uv run python -m bench.harness clipseg

cf-check:
	uv sync --extra torch --extra remote
	uv run python scripts/check_cloudflare.py

bench-remote:
	uv sync --extra torch --extra remote
	uv run python -m bench.harness remote

# Everything except bench-remote, which spends Cloudflare's daily neuron budget.
verify: bench-sam bench-lama bench-migan bench-erasers bench-pipeline \
        bench-clipseg bench-text2edit bench-fullres report

report:
	uv run python -m bench.report

clean:
	rm -rf out/*.json out/*.png
