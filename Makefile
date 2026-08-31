.PHONY: help install install-dev test lint fmt run docker-up docker-down clean check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

install-dev:  ## Install runtime + development dependencies
	pip install -r requirements-dev.txt

test:  ## Run the test suite
	pytest

test-cov:  ## Run tests with a coverage report
	pytest --cov=pdfchat --cov-report=term-missing

lint:  ## Check formatting and lint rules
	ruff check .
	ruff format --check .

fmt:  ## Auto-format and fix what can be fixed
	ruff format .
	ruff check --fix .

eval:  ## Measure retrieval quality against the labelled gold set (offline)
	python evals/run_eval.py --sweep-chunk-size

eval-real:  ## Same, using the configured embedding provider (costs money)
	python evals/run_eval.py --real-embeddings --sweep-chunk-size

check: lint test  ## Everything CI runs

run:  ## Start the app locally
	streamlit run app.py

config:  ## Validate the current environment
	python -m pdfchat.cli check-config

password:  ## Generate an APP_PASSWORD_HASH
	python -m pdfchat.cli hash-password

docker-up:  ## Build and start the self-hosted stack
	docker compose up -d --build

docker-down:  ## Stop the stack (volumes are preserved)
	docker compose down

docker-logs:  ## Follow application logs
	docker compose logs -f app

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov .pdfchat_cache
