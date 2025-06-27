.PHONY: help install test lint format type-check clean run-tests dev-setup sync

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install dependencies with UV"
	@echo "  dev-setup     Set up development environment"
	@echo "  sync          Sync dependencies (create uv.lock)"
	@echo "  test          Run basic tests"
	@echo "  test-comprehensive  Run comprehensive test suite"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and ruff"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean up cache files"
	@echo "  run-tests     Run comprehensive tests (alias)"

# Install dependencies
install:
	uv sync

# Sync dependencies (create/update uv.lock)
sync:
	uv lock

# Development setup
dev-setup:
	@echo "Setting up development environment with UV..."
	uv sync --dev
	cp .env.example .env || true
	@echo "Please edit .env file with your API keys"

# Run tests
test:
	@echo "Running AI Feedback Pipeline Tests..."
	uv run python tests/test_simple.py

# Run comprehensive test suite
test-comprehensive:
	@echo "Running Comprehensive AI Feedback Pipeline Test Suite..."
	uv run python test_comprehensive_suite.py

# Run tests with coverage
run-tests: test-comprehensive

# Lint code
lint:
	uv run ruff check .
	uv run ruff format --check .

# Format code
format:
	uv run black .
	uv run ruff format .

# Type checking
type-check:
	uv run mypy .

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Full check (lint, type-check, test)
check: lint type-check test

# Quick development cycle
dev: format lint type-check test

# Run the pipeline with sample data
demo:
	python main.py process-transcript data/transcripts/sample_customer_call.txt

# Start the API server
serve:
	python server.py

# Docker commands
docker-build:
	docker build -t ai-feedback-pipeline .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f feedback-pipeline
