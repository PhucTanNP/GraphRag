.PHONY: dev run test lint shell clean

# ── Development ───────────────────────────────────────────
dev:        ## Run FastAPI with hot-reload
	KMP_DUPLICATE_LIB_OK=TRUE uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run:        ## Run FastAPI (production mode)
	KMP_DUPLICATE_LIB_OK=TRUE uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

shell:      ## Open Python shell with project deps
	uv run python

# ── Code quality ──────────────────────────────────────────
test:       ## Run tests
	uv run pytest tests -q -x --timeout=30

lint:       ## Lint with ruff
	uv run ruff check app/

clean:      ## Remove __pycache__
	rm -rf app/*/__pycache__ app/__pycache__ tests/__pycache__

# ── Help ──────────────────────────────────────────────────
help:       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'
