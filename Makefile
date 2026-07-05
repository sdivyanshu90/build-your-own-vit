# =============================================================================
# Build Your Own ViT — developer task runner
# =============================================================================
.DEFAULT_GOAL := help
SHELL := /bin/bash
PY ?= python3
PKG := vit
IMAGE ?= build-your-own-vit
TAG ?= latest

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install the package with dev + serve + export extras (editable)
	$(PY) -m pip install -e ".[all]"

.PHONY: hooks
hooks: ## Install pre-commit git hooks
	pre-commit install

.PHONY: format
format: ## Auto-format the codebase
	ruff format src tests
	ruff check --fix src tests

.PHONY: lint
lint: ## Run linters (no changes)
	ruff check src tests
	ruff format --check src tests

.PHONY: typecheck
typecheck: ## Run static type checking
	mypy src

.PHONY: test
test: ## Run the full test suite with coverage
	pytest --cov=$(PKG) --cov-report=term-missing --cov-report=xml

.PHONY: test-fast
test-fast: ## Run tests, skipping slow/integration markers
	pytest -m "not slow and not integration"

.PHONY: cov-html
cov-html: ## Generate an HTML coverage report in htmlcov/
	pytest --cov=$(PKG) --cov-report=html
	@echo "Open htmlcov/index.html"

.PHONY: check
check: lint typecheck test ## Run lint + typecheck + tests (CI gate)

.PHONY: train
train: ## Smoke-train a tiny model on CIFAR-10 (few steps)
	$(PY) -m vit train --config configs/vit_tiny_cifar10.yaml --smoke

.PHONY: serve
serve: ## Run the inference API locally
	$(PY) -m vit serve --host 0.0.0.0 --port 8080

.PHONY: docker-build
docker-build: ## Build the runtime Docker image
	docker build -t $(IMAGE):$(TAG) .

.PHONY: docker-run
docker-run: ## Run the serving image
	docker run --rm -p 8080:8080 --env-file .env $(IMAGE):$(TAG)

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov \
		.coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
