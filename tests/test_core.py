"""
Simplified tests for the AI Feedback Pipeline core functionality.
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest


# Ensure project is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set required environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def setup_mocks():
    """Set up comprehensive mocks for external dependencies."""
    # Mock LangChain core components that cause Pydantic discriminator issues
    mock_ai_message = MagicMock()
    mock_ai_message.content = "test response"

    mock_human_message = MagicMock()
    mock_system_message = MagicMock()

    # Mock embedding functions
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_embeddings.aembed_documents.return_value = [[0.1, 0.2, 0.3]]

    # Mock LLM clients
    mock_chat_openai = MagicMock()
    mock_chat_openai.ainvoke.return_value = mock_ai_message

    mock_modules = {
        # Core LangChain modules
        "langchain": MagicMock(),
        "langchain.schema": MagicMock(
            HumanMessage=mock_human_message,
            SystemMessage=mock_system_message,
            AIMessage=mock_ai_message,
        ),
        "langchain_core": MagicMock(),
        "langchain_core.messages": MagicMock(
            HumanMessage=mock_human_message,
            SystemMessage=mock_system_message,
            AIMessage=mock_ai_message,
        ),
        "langchain_openai": MagicMock(
            ChatOpenAI=lambda **kwargs: mock_chat_openai,
            OpenAIEmbeddings=lambda **kwargs: mock_embeddings,
        ),
        "langchain_anthropic": MagicMock(
            ChatAnthropic=lambda **kwargs: mock_chat_openai,
        ),
        "langchain_huggingface": MagicMock(
            HuggingFaceEmbeddings=lambda **kwargs: mock_embeddings,
        ),
        "langchain_ollama": MagicMock(
            ChatOllama=lambda **kwargs: mock_chat_openai,
        ),
        "langchain_community": MagicMock(),
        "langchain_community.embeddings": MagicMock(),
        "langchain_community.llms": MagicMock(),
        # Vector stores and databases
        "chromadb": MagicMock(),
        "pinecone": MagicMock(),
        "notion_client": MagicMock(),
        # Other dependencies
        "sentence_transformers": MagicMock(),
        "groq": MagicMock(),
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
        # Import after mocking to avoid Pydantic issues
        with patch("extract.get_llm_client") as mock_get_client:
            mock_get_client.return_value = Mock()
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
        # Import after mocking to avoid Pydantic issues
        with patch("embed.get_llm_client") as mock_get_client:
            mock_get_client.return_value = Mock()
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

            with (
                patch("embed.chromadb") as mock_chromadb,
                patch("embed.get_llm_client") as mock_get_client,
            ):
                mock_chromadb.PersistentClient.return_value = mock_client
                mock_client.get_or_create_collection.return_value = mock_collection
                mock_get_client.return_value = Mock()

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

            with (
                patch("embed.Pinecone") as mock_pinecone_class,
                patch("embed.get_llm_client") as mock_get_client,
            ):
                mock_pinecone_class.return_value = mock_pinecone_instance
                mock_pinecone_instance.list_indexes.return_value = []
                mock_pinecone_instance.Index.return_value = mock_index
                mock_get_client.return_value = Mock()

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

            with (
                patch("embed.chromadb"),
                patch("embed.get_llm_client") as mock_get_client,
            ):
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
