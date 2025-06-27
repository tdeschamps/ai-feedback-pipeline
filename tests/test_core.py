"""
Simplified tests for the AI Feedback Pipeline core functionality.
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import pytest


# Ensure project is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set required environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def setup_mocks():
    """Set up mocks for external dependencies."""
    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
        "chromadb": Mock(),
        "pinecone": Mock(),
        "notion_client": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        yield


class TestBasicFunctionality:
    """Test basic application functionality."""

    def test_config_loads(self):
        """Test that configuration loads successfully."""
        import config

        assert config.settings is not None
        assert hasattr(config.settings, "llm_provider")

    def test_feedback_dataclass(self):
        """Test Feedback dataclass creation."""
        import extract

        feedback = extract.Feedback(
            type="feature_request",
            summary="Test feedback",
            verbatim="This is a test feedback",
            confidence=0.9,
            transcript_id="test_transcript",
            timestamp=datetime.now(),
            context="test context",
        )

        assert feedback.type == "feature_request"
        assert feedback.summary == "Test feedback"
        assert feedback.confidence == 0.9

    def test_embedding_document_dataclass(self):
        """Test EmbeddingDocument dataclass creation."""
        import embed

        doc = embed.EmbeddingDocument(
            id="test-doc-1",
            content="Test document content",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={"type": "test", "category": "unit_test"},
            doc_type="feedback",
        )

        assert doc.id == "test-doc-1"
        assert doc.content == "Test document content"
        assert len(doc.embedding) == 5
        assert doc.doc_type == "feedback"


class TestVectorStores:
    """Test vector store functionality with mocked dependencies."""

    def test_chromadb_initialization(self):
        """Test ChromaDB vector store initialization."""
        with patch("config.settings") as mock_settings:
            mock_settings.vector_store = "chromadb"
            mock_settings.chromadb_host = "localhost"
            mock_settings.chromadb_port = 8000
            mock_settings.chromadb_persist_directory = "./test_chroma"
            mock_settings.chromadb_collection_name = "test_collection"

            # Mock ChromaDB components
            mock_client = Mock()
            mock_collection = Mock()

            with patch("embed.chromadb") as mock_chromadb:
                mock_chromadb.PersistentClient.return_value = mock_client
                mock_client.get_or_create_collection.return_value = mock_collection

                import embed

                store = embed.ChromaDBVectorStore()

                assert store.client == mock_client
                assert store.collection == mock_collection
                mock_chromadb.PersistentClient.assert_called_once()
                mock_client.get_or_create_collection.assert_called_once()

    def test_pinecone_initialization(self):
        """Test Pinecone vector store initialization."""
        with patch("config.settings") as mock_settings:
            mock_settings.vector_store = "pinecone"
            mock_settings.pinecone_api_key = "test-api-key"
            mock_settings.pinecone_index_name = "test-index"
            mock_settings.embedding_dimensions = 1536

            # Mock Pinecone components
            mock_pinecone_instance = Mock()
            mock_index = Mock()

            with patch("embed.Pinecone") as mock_pinecone_class:
                mock_pinecone_class.return_value = mock_pinecone_instance
                mock_pinecone_instance.list_indexes.return_value = []
                mock_pinecone_instance.Index.return_value = mock_index

                import embed

                store = embed.PineconeVectorStore()

                assert store.pc == mock_pinecone_instance
                assert store.index == mock_index
                mock_pinecone_class.assert_called_once_with(api_key="test-api-key")


class TestPipelineComponents:
    """Test pipeline components work together."""

    def test_embedding_manager_initialization(self):
        """Test EmbeddingManager can be initialized."""
        with patch("config.settings") as mock_settings:
            mock_settings.vector_store = "chromadb"
            mock_settings.chromadb_host = "localhost"
            mock_settings.chromadb_port = 8000
            mock_settings.chromadb_persist_directory = "./test_chroma"
            mock_settings.chromadb_collection_name = "test_collection"

            with patch("embed.chromadb"), patch("embed.get_llm_client") as mock_get_client:
                mock_client = Mock()
                mock_client.embed.return_value = [[0.1, 0.2, 0.3]]
                mock_get_client.return_value = mock_client

                import embed

                manager = embed.EmbeddingManager()

                assert manager is not None
                assert hasattr(manager, "vector_store")
                assert hasattr(manager, "llm_client")


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v"])
