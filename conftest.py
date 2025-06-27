"""
Shared test configuration and fixtures.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest


# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with required environment variables."""
    # Set required environment variables for testing
    test_env = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "NOTION_API_KEY": "test-notion-key",
        "PINECONE_API_KEY": "test-pinecone-key",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "gpt-4o",
        "VECTOR_STORE": "chromadb",
        "CONFIDENCE_THRESHOLD": "0.7",
        "MAX_MATCHES": "5",
        "LOG_LEVEL": "INFO",
        "LOG_FILE": "test_pipeline.log",
    }

    with patch.dict(os.environ, test_env):
        yield


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies like LangChain, ChromaDB, etc."""
    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
        "chromadb": Mock(),
        "pinecone": Mock(),
        "notion_client": Mock(),
        "fastapi": Mock(),
        "uvicorn": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        yield mock_modules


@pytest.fixture
def mock_settings():
    """Create a mock settings object for testing."""
    mock = Mock()

    # Set default values for all settings attributes
    mock.llm_provider = "openai"
    mock.llm_model = "gpt-4o"
    mock.llm_base_url = None
    mock.openai_api_key = "test-openai-key"
    mock.anthropic_api_key = "test-anthropic-key"
    mock.groq_api_key = None
    mock.huggingface_api_token = None

    mock.embedding_provider = "openai"
    mock.embedding_model = "text-embedding-3-small"
    mock.embedding_dimensions = 1536

    mock.vector_store = "chromadb"
    mock.chromadb_host = "localhost"
    mock.chromadb_port = 8000
    mock.chromadb_collection_name = "test_collection"
    mock.chromadb_persist_directory = "./test_chroma"
    mock.pinecone_api_key = "test-pinecone-key"
    mock.pinecone_environment = "test-env"
    mock.pinecone_index_name = "test-index"

    mock.notion_api_key = "test-notion-key"
    mock.notion_database_id = "test-database-id"

    mock.confidence_threshold = 0.7
    mock.max_matches = 5
    mock.rerank_enabled = True

    mock.log_level = "INFO"
    mock.log_file = "test_pipeline.log"

    return mock


@pytest.fixture
def sample_embedding_documents():
    """Sample embedding documents for testing."""
    # Import here to avoid circular imports
    from embed import EmbeddingDocument

    return [
        EmbeddingDocument(
            id="test-1",
            content="This is a test feedback about feature requests",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={"type": "feedback", "category": "feature_request"},
            doc_type="feedback"
        ),
        EmbeddingDocument(
            id="test-2",
            content="This is a test problem description",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
            metadata={"type": "problem", "priority": "high"},
            doc_type="problem"
        )
    ]
