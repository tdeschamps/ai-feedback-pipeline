.PHONY: help install test lint format type-check clean run-tests dev-setup

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install dependencies"
	@echo "  dev-setup     Set up development environment"
	@echo "  test          Run all tests"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean up cache files"
	@echo "  run-tests     Run tests with coverage"

# Install dependencies
install:
	pip install -r requirements.txt

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	pip install -e .
	cp .env.example .env
	@echo "Please edit .env file with your API keys"

# Run tests
test:
	python -m pytest tests/ -v

# Run tests with coverage
run-tests:
	python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# Lint code
lint:
	python -m ruff check .
	python -m ruff format --check .

# Format code
format:
	python -m black .
	python -m ruff format .

# Type checking
type-check:
	python -m mypy .

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
