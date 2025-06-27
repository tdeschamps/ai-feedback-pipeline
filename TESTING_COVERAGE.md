# Test Coverage Status

This document tracks the current test coverage status and identifies areas that need improvement.

## Current Coverage Issues (as of last run)

### Files with Low Coverage (< 50%)

- `main.py` (22%) - Command-line interface and main application logic
- `pipeline.py` (15%) - Core pipeline orchestration
- `embed.py` (38%) - Vector embedding functionality
- `extract.py` (38%) - Data extraction logic
- `rag.py` (36%) - RAG (Retrieval Augmented Generation) implementation

### Files with Medium Coverage (50-75%)

- `llm_client.py` (50%) - LLM client interactions

### Files with Good Coverage (> 75%)

- `notion.py` (76%) - Notion API integration
- `config.py` (96%) - Configuration management

## Priority Areas for Test Improvement

1. **pipeline.py** - Critical component with only 15% coverage
2. **main.py** - Entry point with only 22% coverage
3. **embed.py** - Vector operations with 38% coverage
4. **extract.py** - Data processing with 38% coverage
5. **rag.py** - Core RAG functionality with 36% coverage

## Coverage Configuration

The project now uses proper coverage configuration in `pyproject.toml` that:

- Excludes test files from coverage reporting
- Focuses on core source files only
- Sets minimum coverage threshold at 50%
- Generates HTML reports for detailed analysis
- Sorts by missing lines to highlight gaps

## Recommendations

1. Add integration tests for `pipeline.py`
2. Add CLI tests for `main.py`
3. Add more edge cases for `embed.py`
4. Improve `extract.py` test scenarios
5. Add end-to-end tests for `rag.py`

## Running Coverage Locally

```bash
# Run tests with coverage
uv run pytest tests/ -v --cov --cov-config=pyproject.toml --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

## CI/CD Integration

The CI pipeline now:

- Excludes test files from coverage reporting
- Generates both XML and HTML coverage reports
- Shows coverage summary in GitHub Actions
- Reports coverage to Codecov for tracking trends
