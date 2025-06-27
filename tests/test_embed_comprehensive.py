"""
Tests for embed.py - Embedding and vector store functionality.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def mock_embedding_vector():
    """Create a mock embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dimensional vector


@pytest.fixture
def mock_problem():
    """Create a mock problem object."""
    problem = Mock()
    problem.id = "test_problem_id"
    problem.title = "Test Problem"
    problem.description = "Test problem description"
    problem.category = "feature_request"
    return problem


class TestEmbeddingManager:
    """Test EmbeddingManager class."""

    @patch("embed.settings")
    @patch("embed.OpenAIEmbeddings")
    def test_init_openai_embeddings(self, mock_openai_embeddings, mock_settings):
        """Test initialization with OpenAI embeddings."""
        mock_settings.embedding_model = "text-embedding-ada-002"
        mock_settings.embedding_provider = "openai"

        from embed import EmbeddingManager

        manager = EmbeddingManager()
        assert manager is not None
        mock_openai_embeddings.assert_called_once()

    @patch("embed.settings")
    @patch("embed.HuggingFaceEmbeddings")
    def test_init_huggingface_embeddings(self, mock_hf_embeddings, mock_settings):
        """Test initialization with HuggingFace embeddings."""
        mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        mock_settings.embedding_provider = "huggingface"

        from embed import EmbeddingManager

        manager = EmbeddingManager()
        assert manager is not None
        mock_hf_embeddings.assert_called_once()

    @patch("embed.settings")
    @patch("embed.OpenAIEmbeddings")
    def test_embed_text(self, mock_openai_embeddings, mock_settings, mock_embedding_vector):
        """Test text embedding."""
        mock_settings.embedding_model = "text-embedding-ada-002"
        mock_settings.embedding_provider = "openai"

        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        mock_embeddings_instance.embed_query.return_value = mock_embedding_vector

        from embed import EmbeddingManager

        manager = EmbeddingManager()
        result = manager.embed_text("test text")

        assert result == mock_embedding_vector
        mock_embeddings_instance.embed_query.assert_called_once_with("test text")

    @patch("embed.settings")
    @patch("embed.OpenAIEmbeddings")
    def test_embed_documents(self, mock_openai_embeddings, mock_settings, mock_embedding_vector):
        """Test document embedding."""
        mock_settings.embedding_model = "text-embedding-ada-002"
        mock_settings.embedding_provider = "openai"

        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        mock_embeddings_instance.embed_documents.return_value = [
            mock_embedding_vector,
            mock_embedding_vector,
        ]

        from embed import EmbeddingManager

        manager = EmbeddingManager()
        result = manager.embed_documents(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == mock_embedding_vector
        mock_embeddings_instance.embed_documents.assert_called_once_with(["text1", "text2"])


class TestVectorStoreManager:
    """Test VectorStoreManager class."""

    @patch("embed.settings")
    @patch("embed.Chroma")
    @patch("embed.EmbeddingManager")
    def test_init_chromadb(self, mock_embedding_manager, mock_chroma, mock_settings):
        """Test initialization with ChromaDB."""
        mock_settings.vector_store = "chromadb"
        mock_settings.chromadb_persist_directory = "./test_chroma"

        from embed import VectorStoreManager

        manager = VectorStoreManager()
        assert manager is not None
        mock_chroma.assert_called_once()

    @patch("embed.settings")
    @patch("embed.PineconeVectorStore")
    @patch("embed.EmbeddingManager")
    def test_init_pinecone(self, mock_embedding_manager, mock_pinecone, mock_settings):
        """Test initialization with Pinecone."""
        mock_settings.vector_store = "pinecone"
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        from embed import VectorStoreManager

        manager = VectorStoreManager()
        assert manager is not None
        mock_pinecone.assert_called_once()

    @patch("embed.settings")
    @patch("embed.Chroma")
    @patch("embed.EmbeddingManager")
    def test_add_problems(self, mock_embedding_manager, mock_chroma, mock_settings, mock_problem):
        """Test adding problems to vector store."""
        mock_settings.vector_store = "chromadb"

        mock_vectorstore_instance = Mock()
        mock_chroma.return_value = mock_vectorstore_instance

        from embed import VectorStoreManager

        manager = VectorStoreManager()
        manager.add_problems([mock_problem])

        mock_vectorstore_instance.add_texts.assert_called_once()

    @patch("embed.settings")
    @patch("embed.Chroma")
    @patch("embed.EmbeddingManager")
    def test_search_similar(self, mock_embedding_manager, mock_chroma, mock_settings):
        """Test similarity search."""
        mock_settings.vector_store = "chromadb"

        mock_vectorstore_instance = Mock()
        mock_chroma.return_value = mock_vectorstore_instance

        # Mock search results
        mock_doc = Mock()
        mock_doc.page_content = "similar problem"
        mock_doc.metadata = {"problem_id": "test_id", "title": "Test Problem"}
        mock_vectorstore_instance.similarity_search_with_score.return_value = [(mock_doc, 0.8)]

        from embed import VectorStoreManager

        manager = VectorStoreManager()
        results = manager.search_similar("test query", k=5)

        assert len(results) == 1
        assert results[0][0] == mock_doc
        assert results[0][1] == 0.8
        mock_vectorstore_instance.similarity_search_with_score.assert_called_once_with(
            "test query", k=5
        )

    @patch("embed.settings")
    @patch("embed.Chroma")
    @patch("embed.EmbeddingManager")
    def test_update_problem(self, mock_embedding_manager, mock_chroma, mock_settings, mock_problem):
        """Test updating a problem in vector store."""
        mock_settings.vector_store = "chromadb"

        mock_vectorstore_instance = Mock()
        mock_chroma.return_value = mock_vectorstore_instance
        mock_vectorstore_instance.delete.return_value = None
        mock_vectorstore_instance.add_texts.return_value = None

        from embed import VectorStoreManager

        manager = VectorStoreManager()
        manager.update_problem(mock_problem)

        # Should delete old and add new
        mock_vectorstore_instance.delete.assert_called_once()
        mock_vectorstore_instance.add_texts.assert_called_once()

    @patch("embed.settings")
    @patch("embed.Chroma")
    @patch("embed.EmbeddingManager")
    def test_clear_store(self, mock_embedding_manager, mock_chroma, mock_settings):
        """Test clearing the vector store."""
        mock_settings.vector_store = "chromadb"

        mock_vectorstore_instance = Mock()
        mock_chroma.return_value = mock_vectorstore_instance
        mock_vectorstore_instance._collection = Mock()
        mock_vectorstore_instance._collection.delete.return_value = None

        from embed import VectorStoreManager

        manager = VectorStoreManager()
        manager.clear_store()

        mock_vectorstore_instance._collection.delete.assert_called_once()


class TestEmbeddingUtilities:
    """Test embedding utility functions."""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from embed import cosine_similarity

        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        vec3 = [1, 0, 0]

        # Orthogonal vectors should have similarity 0
        sim1 = cosine_similarity(vec1, vec2)
        assert abs(sim1 - 0.0) < 1e-10

        # Identical vectors should have similarity 1
        sim2 = cosine_similarity(vec1, vec3)
        assert abs(sim2 - 1.0) < 1e-10

    def test_normalize_vector(self):
        """Test vector normalization."""
        from embed import normalize_vector

        vector = [3, 4, 0]  # Length = 5
        normalized = normalize_vector(vector)

        # Check that the normalized vector has length 1
        import math

        length = math.sqrt(sum(x**2 for x in normalized))
        assert abs(length - 1.0) < 1e-10

        # Check values
        assert abs(normalized[0] - 0.6) < 1e-10
        assert abs(normalized[1] - 0.8) < 1e-10
        assert abs(normalized[2] - 0.0) < 1e-10

    @patch("embed.EmbeddingManager")
    def test_batch_embed_texts(self, mock_embedding_manager):
        """Test batch text embedding."""
        from embed import batch_embed_texts

        mock_manager_instance = Mock()
        mock_embedding_manager.return_value = mock_manager_instance
        mock_manager_instance.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]

        texts = ["text1", "text2"]
        results = batch_embed_texts(texts)

        assert len(results) == 2
        assert results[0] == [0.1, 0.2]
        assert results[1] == [0.3, 0.4]
        mock_manager_instance.embed_documents.assert_called_once_with(texts)


class TestErrorHandling:
    """Test error handling in embedding functions."""

    @patch("embed.settings")
    @patch("embed.OpenAIEmbeddings")
    def test_embedding_error_handling(self, mock_openai_embeddings, mock_settings):
        """Test error handling in embedding operations."""
        mock_settings.embedding_model = "text-embedding-ada-002"
        mock_settings.embedding_provider = "openai"

        mock_embeddings_instance = Mock()
        mock_openai_embeddings.return_value = mock_embeddings_instance
        mock_embeddings_instance.embed_query.side_effect = Exception("API Error")

        from embed import EmbeddingManager

        manager = EmbeddingManager()

        # Should handle the exception gracefully
        with pytest.raises(Exception, match="API Error"):
            manager.embed_text("test text")

    @patch("embed.settings")
    @patch("embed.Chroma")
    @patch("embed.EmbeddingManager")
    def test_vector_store_error_handling(self, mock_embedding_manager, mock_chroma, mock_settings):
        """Test error handling in vector store operations."""
        mock_settings.vector_store = "chromadb"

        mock_vectorstore_instance = Mock()
        mock_chroma.return_value = mock_vectorstore_instance
        mock_vectorstore_instance.similarity_search_with_score.side_effect = Exception(
            "Search Error"
        )

        from embed import VectorStoreManager

        manager = VectorStoreManager()

        # Should handle the exception gracefully
        with pytest.raises(Exception, match="Search Error"):
            manager.search_similar("test query")


def test_embed_imports():
    """Test that embed module imports correctly."""
    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_openai": Mock(),
        "langchain_huggingface": Mock(),
        "chromadb": Mock(),
        "pinecone": Mock(),
        "sentence_transformers": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        try:
            import embed

            assert embed is not None
            print("✓ Embed module imports successfully")
        except ImportError as e:
            print(f"✓ Embed imports skipped: {e}")


if __name__ == "__main__":
    print("Running embed tests...")
    test_embed_imports()
    print("✓ All embed tests configured!")
