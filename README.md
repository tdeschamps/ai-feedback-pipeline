# 🧠 AI-Powered Feedback Categorization & RAG Pipeline

[![CI/CD](https://github.com/tdeschamps/ai-feedback-pipeline/workflows/CI/badge.svg)](https://github.com/tdeschamps/ai-feedback-pipeline/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen)](https://github.com/tdeschamps/ai-feedback-pipeline)
[![Coverage](https://img.shields.io/badge/coverage-44%25-orange)](https://github.com/tdeschamps/ai-feedback-pipeline)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)

A Python-based pipeline that uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to automatically extract, categorize, and match customer feedback from Circleback transcripts to existing problems in Notion databases.

## 🎯 Features

- **Multi-LLM Support**: OpenAI GPT-4, Claude, Ollama, Groq, HuggingFace
- **Feedback Extraction**: Automatically identifies feature requests and customer pains
- **RAG Matching**: Semantically matches feedback to existing Notion problems
- **Vector Storage**: ChromaDB (local) and Pinecone (cloud) support
- **Auto-Updates**: Updates Notion problems with matched feedback
- **CLI & API**: Command-line interface and FastAPI web server
- **Docker Ready**: Full containerization with docker-compose

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-feedback-pipeline
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file with your API keys:

```bash
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=your_openai_key_here

# Vector Store
VECTOR_STORE=chromadb
CHROMADB_PERSIST_DIRECTORY=./chroma_db

# Notion Integration
NOTION_API_KEY=your_notion_api_key
NOTION_DATABASE_ID=your_notion_database_id
```

### 3. Install Dependencies

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync

# For development dependencies
uv sync --dev
```

### 4. Run Pipeline

```bash
# Process a single transcript
uv run python main.py process-transcript data/transcripts/sample_customer_call.txt

# Process all transcripts in a directory
uv run python main.py batch-process data/transcripts/

# Sync Notion problems
uv run python main.py sync-problems

# Check status
uv run python main.py status
```

## 🧰 Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Circleback    │───▶│   Extract    │───▶│    Embed &      │
│   Transcripts   │    │   Feedback   │    │  Vector Store   │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                      │
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│     Notion      │◀───│     RAG      │◀───│   Semantic      │
│   Problems      │    │   Matching   │    │    Search       │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

### Components

- **`extract.py`**: LLM-based feedback extraction and classification
- **`embed.py`**: Embedding generation and vector store operations
- **`rag.py`**: RAG-based matching with confidence scoring
- **`notion.py`**: Notion API integration for reading/updating problems
- **`llm_client.py`**: Unified LLM interface supporting multiple providers
- **`pipeline.py`**: High-level orchestration and workflow management

## 📖 Usage Examples

### CLI Usage

```bash
# Process single transcript with custom ID
uv run python main.py process-transcript transcript.txt --transcript-id "call-2024-001"

# Batch process with pattern matching
uv run python main.py batch-process transcripts/ --pattern "*.txt" --output results.json

# Show recent feedbacks
uv run python main.py show-feedbacks --limit 5

# View configuration
uv run python main.py status
```

### API Usage

Start the FastAPI server:

```bash
uv run python server.py
# or
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Process transcript via API:

```bash
curl -X POST "http://localhost:8000/process/transcript" \
  -H "Content-Type: application/json" \
  -d '{
    "transcript_id": "call-001",
    "content": "Customer mentioned they need Excel export..."
  }'
```

### Docker Usage

```bash
# Build and run with docker-compose
docker-compose up -d

# Run CLI commands in container
docker-compose exec feedback-pipeline uv run python main.py status

# View logs
docker-compose logs -f feedback-pipeline
```

## 🔧 Configuration

### LLM Providers

#### OpenAI
```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=your_key
```

#### Claude (Anthropic)
```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
ANTHROPIC_API_KEY=your_key
```

#### Ollama (Local)
```bash
LLM_PROVIDER=ollama
LLM_MODEL=mistral
LLM_BASE_URL=http://localhost:11434
```

#### Groq
```bash
LLM_PROVIDER=groq
LLM_MODEL=mixtral-8x7b-32768
GROQ_API_KEY=your_key
```

### Vector Stores

#### ChromaDB (Local - Recommended)

```bash
VECTOR_STORE=chromadb
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
CHROMADB_COLLECTION_NAME=feedback_embeddings
CHROMADB_PERSIST_DIRECTORY=./chroma_db
```

#### Pinecone (Cloud)

```bash
VECTOR_STORE=pinecone
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=feedback-pipeline
```

### Installation and Setup

#### Local ChromaDB Setup (Recommended for Development)

```bash
# Install ChromaDB
uv add chromadb

# ChromaDB will create a local database automatically
VECTOR_STORE=chromadb
CHROMADB_PERSIST_DIRECTORY=./chroma_db
```

#### Cloud Pinecone Setup (Recommended for Production)

```bash
# Install Pinecone
uv add pinecone-client

# Get API key from https://www.pinecone.io/
VECTOR_STORE=pinecone
PINECONE_API_KEY=your_api_key
PINECONE_INDEX_NAME=feedback-pipeline
```

#### Remote ChromaDB Setup (Advanced)

```bash
# For remote ChromaDB server
VECTOR_STORE=chromadb
CHROMADB_HOST=your-chromadb-host.com
CHROMADB_PORT=8000
```

### Pipeline Settings

```bash
CONFIDENCE_THRESHOLD=0.7      # Minimum confidence for auto-updates
MAX_MATCHES=5                 # Number of candidates to consider
RERANK_ENABLED=true          # Use LLM for reranking matches
```

## 📊 Data Flow

### 1. Feedback Extraction

Input transcript → LLM analysis → Structured feedback:

```json
[
  {
    "type": "feature_request",
    "summary": "Need Excel export functionality",
    "verbatim": "I really wish you had an Excel export feature",
    "confidence": 0.9
  },
  {
    "type": "customer_pain",
    "summary": "Search is too slow with large datasets",
    "verbatim": "The search is incredibly slow",
    "confidence": 0.85
  }
]
```

### 2. Vector Matching

Feedback embeddings → Semantic search → Problem candidates → LLM reranking → Best match

### 3. Notion Updates

```
Problem + Feedback → Notion API → Updated problem with:
- Appended feedback verbatim
- Incremented feedback count
- Updated timestamp
- Match confidence logged
```

## 🧪 Testing

### Current Test Coverage: 44%

The project includes comprehensive test suites with recent major improvements:

```bash
# Run tests
uv run pytest tests/

# Run with coverage (recommended)
uv run pytest --cov --cov-config=pyproject.toml --cov-report=html --cov-report=term-missing tests/

# View HTML coverage report
open htmlcov/index.html

# Analyze coverage gaps
./analyze_coverage.py

# Run specific test
uv run pytest tests/test_pipeline.py::TestFeedbackExtractor::test_extract_feedback
```

### Recent Test Improvements

- ✅ **Comprehensive CLI testing** (`test_main_comprehensive.py`)
- ✅ **Complete pipeline orchestration** (`test_pipeline.py`)
- ✅ **Vector store & embedding testing** (`test_embed_comprehensive.py`)
- ✅ **LLM extraction testing** (`test_extract_comprehensive.py`)
- ✅ **RAG matching & metrics** (`test_rag_comprehensive.py`)

See [`TESTING_COVERAGE.md`](./TESTING_COVERAGE.md) for detailed coverage analysis and [`TEST_COVERAGE_IMPROVEMENTS.md`](./TEST_COVERAGE_IMPROVEMENTS.md) for recent improvements.

## 📈 Monitoring & Metrics

The pipeline tracks:

- **Extraction Rate**: Feedbacks extracted per transcript
- **Match Rate**: Percentage of feedbacks matched to problems
- **Confidence Distribution**: High/medium/low confidence matches
- **Processing Time**: Performance metrics per component

View metrics:

```bash
# CLI
uv run python main.py show-feedbacks

# API
curl http://localhost:8000/metrics
```

## 🔍 Troubleshooting

### Common Issues

1. **No feedbacks extracted**: Check LLM provider configuration and API keys
2. **No matches found**: Verify Notion problems are synced and vector store is populated
3. **Low confidence matches**: Adjust `CONFIDENCE_THRESHOLD` or improve problem descriptions
4. **Slow processing**: Consider using local LLMs or optimizing batch sizes

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG uv run python main.py process-transcript transcript.txt

# Check vector store status
uv run python main.py sync-problems --dry-run
```

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# Configuration check
curl http://localhost:8000/config

# Docker health
docker-compose ps
```

## 🚢 Deployment

### Local Development

```bash
uv run python server.py
```

### Docker Production

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Cloud Deployment

The pipeline can be deployed on:

- **Fly.io**: `fly deploy`
- **Render**: Connect GitHub repo
- **Railway**: One-click deploy
- **AWS/GCP/Azure**: Use container services

### Environment Variables for Production

```bash
# Production settings
LOG_LEVEL=INFO
CONFIDENCE_THRESHOLD=0.8
RERANK_ENABLED=true

# Security
CORS_ORIGINS=https://yourdomain.com
API_KEY_REQUIRED=true
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `uv run pytest`
5. Run linting: `uv run ruff . && uv run black .`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@yourcompany.com

---

Built with ❤️ using Python, LangChain, and modern AI technologies.
