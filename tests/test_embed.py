"""
Unit tests for embed.py - Embedding generation and vector store operations.
"""

import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from embed import (
    ChromaDBVectorStore,
    EmbeddingDocument,
    EmbeddingManager,
    PineconeVectorStore,
    VectorStore,
)
from extract import Feedback
from notion import NotionProblem


class TestEmbeddingDocument:
    """Test EmbeddingDocument dataclass."""

    def test_embedding_document_creation(self):
        """Test creating an EmbeddingDocument instance."""
        doc = EmbeddingDocument(
            id="test_id",
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"type": "test"},
            doc_type="feedback"
        )

        assert doc.id == "test_id"
        assert doc.content == "test content"
        assert doc.embedding == [0.1, 0.2, 0.3]
        assert doc.metadata == {"type": "test"}
        assert doc.doc_type == "feedback"


class TestVectorStoreInterface:
    """Test VectorStore abstract interface."""

    def test_vector_store_is_abstract(self):
        """Test that VectorStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorStore()

    @pytest.mark.asyncio
    async def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        # Create a minimal implementation to test abstract methods
        class TestVectorStore(VectorStore):
            pass

        with pytest.raises(TypeError):
            TestVectorStore()


class TestChromaDBVectorStore:
    """Test ChromaDBVectorStore implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_documents = [
            EmbeddingDocument(
                id="doc1",
                content="test content 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"type": "feedback"},
                doc_type="feedback"
            ),
            EmbeddingDocument(
                id="doc2",
                content="test content 2",
                embedding=[0.4, 0.5, 0.6],
                metadata={"type": "problem"},
                doc_type="problem"
            )
        ]

    @patch('embed.chromadb')
    @patch('embed.settings')
    def test_chromadb_init_local(self, mock_settings, mock_chromadb):
        """Test ChromaDB initialization with local settings."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()

        assert store.client == mock_client
        assert store.collection == mock_collection
        mock_chromadb.PersistentClient.assert_called_once()

    @patch('embed.chromadb')
    @patch('embed.settings')
    def test_chromadb_init_remote(self, mock_settings, mock_chromadb):
        """Test ChromaDB initialization with remote settings."""
        mock_settings.chromadb_host = "remote-host"
        mock_settings.chromadb_port = 9000
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()

        assert store.client == mock_client
        assert store.collection == mock_collection
        mock_chromadb.HttpClient.assert_called_once()

    @patch('embed.chromadb', None)
    def test_chromadb_init_not_available(self):
        """Test ChromaDB initialization when package not available."""
        with pytest.raises(ImportError, match="ChromaDB not available"):
            ChromaDBVectorStore()

    @patch('embed.chromadb')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_add_documents_success(self, mock_settings, mock_chromadb):
        """Test successful document addition to ChromaDB."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        result = await store.add_documents(self.sample_documents)

        assert result is True
        mock_collection.upsert.assert_called_once()

        # Verify the upsert call arguments
        call_args = mock_collection.upsert.call_args
        assert len(call_args[1]["ids"]) == 2
        assert len(call_args[1]["embeddings"]) == 2
        assert len(call_args[1]["metadatas"]) == 2
        assert len(call_args[1]["documents"]) == 2

    @patch('embed.chromadb')
    @patch('embed.settings')
    @patch('embed.logger')
    @pytest.mark.asyncio
    async def test_add_documents_failure(self, mock_logger, mock_settings, mock_chromadb):
        """Test document addition failure handling."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.upsert.side_effect = Exception("Database error")
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        result = await store.add_documents(self.sample_documents)

        assert result is False
        mock_logger.error.assert_called_with("Error adding documents to ChromaDB: Database error")

    @patch('embed.chromadb')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_search_success(self, mock_settings, mock_chromadb):
        """Test successful document search in ChromaDB."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["content1", "content2"]],
            "metadatas": [[{"type": "feedback"}, {"type": "problem"}]],
            "distances": [[0.1, 0.2]]
        }
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        results = await store.search([0.1, 0.2, 0.3], limit=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["content"] == "content1"
        assert results[0]["score"] == 0.9  # 1 - 0.1
        assert results[1]["id"] == "doc2"
        assert results[1]["score"] == 0.8  # 1 - 0.2

    @patch('embed.chromadb')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_search_empty_results(self, mock_settings, mock_chromadb):
        """Test search with empty results."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {"ids": [], "documents": [], "metadatas": [], "distances": []}
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        results = await store.search([0.1, 0.2, 0.3])

        assert results == []

    @patch('embed.chromadb')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_delete_documents_success(self, mock_settings, mock_chromadb):
        """Test successful document deletion."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        result = await store.delete_documents(["doc1", "doc2"])

        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2"])


class TestPineconeVectorStore:
    """Test PineconeVectorStore implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_documents = [
            EmbeddingDocument(
                id="doc1",
                content="test content 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"type": "feedback"},
                doc_type="feedback"
            )
        ]

    @patch('embed.settings')
    def test_pinecone_init_no_api_key(self, mock_settings):
        """Test Pinecone initialization without API key."""
        mock_settings.pinecone_api_key = None

        with pytest.raises(ValueError, match="Pinecone API key not configured"):
            PineconeVectorStore()

    @patch('embed.Pinecone', None)
    @patch('embed.settings')
    def test_pinecone_init_not_available(self, mock_settings):
        """Test Pinecone initialization when package not available."""
        mock_settings.pinecone_api_key = "test-key"

        with pytest.raises(ImportError, match="Pinecone not available"):
            PineconeVectorStore()

    @patch('embed.Pinecone')
    @patch('embed.settings')
    def test_pinecone_init_success(self, mock_settings, mock_pinecone_class):
        """Test successful Pinecone initialization."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"
        mock_settings.embedding_dimensions = 1536

        mock_pc = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()

        assert store.pc == mock_pc
        assert store.index == mock_index
        mock_pinecone_class.assert_called_once_with(api_key="test-key")

    @patch('embed.Pinecone')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_add_documents_success(self, mock_settings, mock_pinecone_class):
        """Test successful document addition to Pinecone."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        mock_pc = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()
        result = await store.add_documents(self.sample_documents)

        assert result is True
        mock_index.upsert.assert_called_once()

    @patch('embed.Pinecone')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_search_success(self, mock_settings, mock_pinecone_class):
        """Test successful document search in Pinecone."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        mock_pc = Mock()
        mock_index = Mock()
        mock_match = Mock()
        mock_match.id = "doc1"
        mock_match.score = 0.95
        mock_match.metadata = {"content": "test content", "doc_type": "feedback"}
        mock_index.query.return_value.matches = [mock_match]

        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()
        results = await store.search([0.1, 0.2, 0.3])

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.95

    @patch('embed.Pinecone')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_delete_documents_success(self, mock_settings, mock_pinecone_class):
        """Test successful document deletion from Pinecone."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        mock_pc = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()
        result = await store.delete_documents(["doc1", "doc2"])

        assert result is True
        mock_index.delete.assert_called_once_with(ids=["doc1", "doc2"])

    @patch('embed.Pinecone')
    @patch('embed.settings')
    @patch('embed.logger')
    @pytest.mark.asyncio
    async def test_delete_documents_failure(self, mock_logger, mock_settings, mock_pinecone_class):
        """Test document deletion failure in Pinecone."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        mock_pc = Mock()
        mock_index = Mock()
        mock_index.delete.side_effect = Exception("Delete error")
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()
        result = await store.delete_documents(["doc1"])

        assert result is False
        mock_logger.error.assert_called_with("Error deleting documents from Pinecone: Delete error")

    @patch('embed.Pinecone')
    @patch('embed.settings')
    @patch('embed.logger')
    @pytest.mark.asyncio
    async def test_search_failure(self, mock_logger, mock_settings, mock_pinecone_class):
        """Test search failure in Pinecone."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        mock_pc = Mock()
        mock_index = Mock()
        mock_index.query.side_effect = Exception("Search error")
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()
        results = await store.search([0.1, 0.2, 0.3])

        assert results == []
        mock_logger.error.assert_called_with("Error searching Pinecone: Search error")

    @patch('embed.Pinecone')
    @patch('embed.settings')
    @patch('embed.logger')
    @pytest.mark.asyncio
    async def test_add_documents_failure(self, mock_logger, mock_settings, mock_pinecone_class):
        """Test document addition failure in Pinecone."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        mock_pc = Mock()
        mock_index = Mock()
        mock_index.upsert.side_effect = Exception("Upsert error")
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()
        result = await store.add_documents(self.sample_documents)

        assert result is False
        mock_logger.error.assert_called_with("Error adding documents to Pinecone: Upsert error")

    @patch('embed.Pinecone')
    @patch('embed.settings')
    def test_ensure_index_exists_create_new(self, mock_settings, mock_pinecone_class):
        """Test creating a new Pinecone index when it doesn't exist."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "new-index"
        mock_settings.embedding_dimensions = 1536

        mock_pc = Mock()
        mock_pc.list_indexes.return_value = []  # No existing indexes
        mock_pinecone_class.return_value = mock_pc

        store = PineconeVectorStore()

        mock_pc.create_index.assert_called_once_with(
            name="new-index",
            dimension=1536,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )

    @patch('embed.Pinecone')
    @patch('embed.settings')
    def test_ensure_index_exists_use_existing(self, mock_settings, mock_pinecone_class):
        """Test using existing Pinecone index."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "existing-index"

        mock_pc = Mock()
        mock_pc.list_indexes.return_value = [Mock(name="existing-index")]
        mock_pinecone_class.return_value = mock_pc

        store = PineconeVectorStore()

        mock_pc.create_index.assert_not_called()

    @patch('embed.Pinecone')
    @patch('embed.settings')
    def test_ensure_index_exists_error(self, mock_settings, mock_pinecone_class):
        """Test error handling in _ensure_index_exists."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        mock_pc = Mock()
        mock_pc.list_indexes.side_effect = Exception("API error")
        mock_pinecone_class.return_value = mock_pc

        with pytest.raises(Exception, match="API error"):
            PineconeVectorStore()

    @patch('embed.Pinecone')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_add_documents_large_batch(self, mock_settings, mock_pinecone_class):
        """Test adding documents in batches when there are many documents."""
        mock_settings.pinecone_api_key = "test-key"
        mock_settings.pinecone_index_name = "test-index"

        # Create 150 documents to test batching (batch_size = 100)
        large_doc_list = []
        for i in range(150):
            doc = EmbeddingDocument(
                id=f"doc{i}",
                content=f"content {i}",
                embedding=[0.1, 0.2, 0.3],
                metadata={"index": i},
                doc_type="test"
            )
            large_doc_list.append(doc)

        mock_pc = Mock()
        mock_index = Mock()
        mock_pinecone_class.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test-index")]
        mock_pc.Index.return_value = mock_index

        store = PineconeVectorStore()
        result = await store.add_documents(large_doc_list)

        assert result is True
        # Should be called twice: once for first 100, once for remaining 50
        assert mock_index.upsert.call_count == 2


class TestChromaDBVectorStoreErrorHandling:
    """Test error handling for ChromaDBVectorStore."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_documents = [
            EmbeddingDocument(
                id="doc1",
                content="test content",
                embedding=[0.1, 0.2, 0.3],
                metadata={"type": "test"},
                doc_type="test"
            )
        ]

    @patch('embed.chromadb')
    @patch('embed.settings')
    @patch('embed.logger')
    @pytest.mark.asyncio
    async def test_search_failure(self, mock_logger, mock_settings, mock_chromadb):
        """Test search failure handling in ChromaDB."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Query error")
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        results = await store.search([0.1, 0.2, 0.3])

        assert results == []
        mock_logger.error.assert_called_with("Error searching ChromaDB: Query error")

    @patch('embed.chromadb')
    @patch('embed.settings')
    @patch('embed.logger')
    @pytest.mark.asyncio
    async def test_delete_documents_failure(self, mock_logger, mock_settings, mock_chromadb):
        """Test delete documents failure handling in ChromaDB."""
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test_collection"

        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.delete.side_effect = Exception("Delete error")
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        result = await store.delete_documents(["doc1"])

        assert result is False
        mock_logger.error.assert_called_with("Error deleting documents from ChromaDB: Delete error")


class TestEmbeddingManager:
    """Test EmbeddingManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_feedback = Feedback(
            type="feature_request",
            summary="Test feedback",
            verbatim="Customer wants a new feature",
            confidence=0.8,
            context="Test context",
            transcript_id="test_transcript",
            timestamp=datetime.datetime.now()
        )

        self.sample_problem = NotionProblem(
            id="test_problem_id",
            title="Test Problem",
            description="Test problem description",
            status="Open",
            priority="High",
            tags=["bug", "ui"],
            feedback_count=5,
            feedbacks=["Feedback 1", "Feedback 2"],
            last_updated=datetime.datetime.now()
        )

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    def test_embedding_manager_init(self, mock_settings, mock_get_llm_client):
        """Test EmbeddingManager initialization."""
        mock_settings.vector_store = "chromadb"
        mock_llm_client = Mock()
        mock_get_llm_client.return_value = mock_llm_client

        with patch('embed.ChromaDBVectorStore') as mock_chroma:
            mock_vector_store = Mock()
            mock_chroma.return_value = mock_vector_store

            manager = EmbeddingManager()

            assert manager.llm_client == mock_llm_client
            assert manager.vector_store == mock_vector_store

    @patch('embed.get_llm_client')
    @patch('embed.settings', None)
    def test_embedding_manager_init_no_settings(self, mock_get_llm_client):
        """Test EmbeddingManager initialization without settings (test mode)."""
        mock_llm_client = Mock()
        mock_get_llm_client.return_value = mock_llm_client

        manager = EmbeddingManager()

        assert manager.llm_client == mock_llm_client
        # Should return a mock vector store for test environments

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    def test_get_vector_store_unsupported(self, mock_settings, mock_get_llm_client):
        """Test _get_vector_store with unsupported vector store."""
        mock_settings.vector_store = "unsupported"
        mock_get_llm_client.return_value = Mock()

        with pytest.raises(ValueError, match="Unsupported vector store: unsupported"):
            EmbeddingManager()

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    @patch('embed.uuid.uuid4')
    @pytest.mark.asyncio
    async def test_embed_feedbacks(self, mock_uuid, mock_settings, mock_get_llm_client):
        """Test embedding feedback documents."""
        mock_settings.vector_store = "chromadb"
        mock_llm_client = AsyncMock()
        mock_llm_client.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_get_llm_client.return_value = mock_llm_client
        mock_uuid.return_value.hex = "abcd1234"

        with patch('embed.ChromaDBVectorStore'):
            manager = EmbeddingManager()
            documents = await manager.embed_feedbacks([self.sample_feedback])

            assert len(documents) == 1
            doc = documents[0]
            assert doc.doc_type == "feedback"
            assert doc.embedding == [0.1, 0.2, 0.3]
            assert "Test feedback" in doc.content
            assert "Customer wants a new feature" in doc.content
            assert doc.metadata["type"] == "feature_request"
            assert doc.metadata["transcript_id"] == "test_transcript"

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_embed_problems(self, mock_settings, mock_get_llm_client):
        """Test embedding problem documents."""
        mock_settings.vector_store = "chromadb"
        mock_llm_client = AsyncMock()
        mock_llm_client.embed.return_value = [[0.4, 0.5, 0.6]]
        mock_get_llm_client.return_value = mock_llm_client

        with patch('embed.ChromaDBVectorStore'):
            manager = EmbeddingManager()
            documents = await manager.embed_problems([self.sample_problem])

            assert len(documents) == 1
            doc = documents[0]
            assert doc.doc_type == "problem"
            assert doc.id == "problem_test_problem_id"
            assert doc.embedding == [0.4, 0.5, 0.6]
            assert "Test Problem" in doc.content
            assert "Test problem description" in doc.content
            assert "Customer Feedbacks:" in doc.content
            assert "Feedback 1" in doc.content
            assert doc.metadata["notion_id"] == "test_problem_id"
            assert doc.metadata["status"] == "Open"
            assert doc.metadata["tags"] == "bug,ui"

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_embed_problems_no_feedbacks(self, mock_settings, mock_get_llm_client):
        """Test embedding problems without customer feedbacks."""
        problem_without_feedbacks = NotionProblem(
            id="test_id",
            title="Test",
            description="Description",
            status="Open",
            feedbacks=None
        )

        mock_settings.vector_store = "chromadb"
        mock_llm_client = AsyncMock()
        mock_llm_client.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_get_llm_client.return_value = mock_llm_client

        with patch('embed.ChromaDBVectorStore'):
            manager = EmbeddingManager()
            documents = await manager.embed_problems([problem_without_feedbacks])

            assert len(documents) == 1
            doc = documents[0]
            assert "Customer Feedbacks:" not in doc.content

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_store_embeddings(self, mock_settings, mock_get_llm_client):
        """Test storing embedding documents."""
        mock_settings.vector_store = "chromadb"
        mock_get_llm_client.return_value = Mock()

        mock_vector_store = AsyncMock()
        mock_vector_store.add_documents.return_value = True

        with patch('embed.ChromaDBVectorStore', return_value=mock_vector_store):
            manager = EmbeddingManager()

            test_docs = [
                EmbeddingDocument(
                    id="test", content="test", embedding=[0.1],
                    metadata={}, doc_type="test"
                )
            ]

            result = await manager.store_embeddings(test_docs)

            assert result is True
            mock_vector_store.add_documents.assert_called_once_with(test_docs)

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    @pytest.mark.asyncio
    async def test_search_similar_problems(self, mock_settings, mock_get_llm_client):
        """Test searching for similar problems."""
        mock_settings.vector_store = "chromadb"
        mock_llm_client = AsyncMock()
        mock_llm_client.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_get_llm_client.return_value = mock_llm_client

        mock_vector_store = AsyncMock()
        mock_vector_store.search.return_value = [
            {"id": "problem_1", "doc_type": "problem", "score": 0.9},
            {"id": "feedback_1", "doc_type": "feedback", "score": 0.8},
            {"id": "problem_2", "metadata": {"notion_id": "abc"}, "score": 0.7}
        ]

        with patch('embed.ChromaDBVectorStore', return_value=mock_vector_store):
            manager = EmbeddingManager()

            results = await manager.search_similar_problems("test feedback")

            # Should filter to only return problems
            assert len(results) == 2
            assert results[0]["id"] == "problem_1"
            assert results[1]["id"] == "problem_2"

            mock_llm_client.embed.assert_called_once_with(["test feedback"])
            mock_vector_store.search.assert_called_once_with([0.1, 0.2, 0.3], 5)

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    @patch('embed.datetime')
    @pytest.mark.asyncio
    async def test_refresh_problem_embeddings_success(self, mock_datetime, mock_settings, mock_get_llm_client):
        """Test successful refresh of problem embeddings."""
        mock_settings.vector_store = "chromadb"
        mock_llm_client = AsyncMock()
        mock_llm_client.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_get_llm_client.return_value = mock_llm_client

        mock_now = datetime.datetime(2024, 1, 1, 12, 0, 0)
        mock_datetime.datetime.now.return_value = mock_now

        mock_vector_store = AsyncMock()
        mock_vector_store.delete_documents.return_value = True
        mock_vector_store.add_documents.return_value = True

        with patch('embed.ChromaDBVectorStore', return_value=mock_vector_store):
            manager = EmbeddingManager()

            result = await manager.refresh_problem_embeddings([self.sample_problem])

            assert result is True
            mock_vector_store.delete_documents.assert_called_once_with(["problem_test_problem_id"])
            mock_vector_store.add_documents.assert_called_once()

            # Check that additional metadata was added
            call_args = mock_vector_store.add_documents.call_args[0][0]
            doc = call_args[0]
            assert doc.metadata["refreshed_at"] == mock_now.isoformat()
            assert doc.metadata["source"] == "notion_refresh"

    @patch('embed.get_llm_client')
    @patch('embed.settings')
    @patch('embed.logger')
    @pytest.mark.asyncio
    async def test_refresh_problem_embeddings_failure(self, mock_logger, mock_settings, mock_get_llm_client):
        """Test refresh problem embeddings with failure."""
        mock_settings.vector_store = "chromadb"
        mock_get_llm_client.return_value = Mock()

        mock_vector_store = AsyncMock()
        mock_vector_store.delete_documents.side_effect = Exception("Database error")

        with patch('embed.ChromaDBVectorStore', return_value=mock_vector_store):
            manager = EmbeddingManager()

            result = await manager.refresh_problem_embeddings([self.sample_problem])

            assert result is False
            mock_logger.error.assert_called_with("Error refreshing problem embeddings: Database error")
