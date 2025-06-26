"""
Tests for the embedding and vector store functionality.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from embed import EmbeddingManager, ChromaDBVectorStore, PineconeVectorStore, EmbeddingDocument
from extract import Feedback
from notion import NotionProblem


class TestVectorStores:
    """Test vector store implementations."""

    def test_chromadb_vector_store_init_local(self) -> None:
        """Test ChromaDB initialization for local storage."""
        with patch('embed.chromadb') as mock_chromadb, \
             patch('embed.settings') as mock_settings:

            mock_settings.chromadb_host = "localhost"
            mock_settings.chromadb_port = 8000
            mock_settings.chromadb_persist_directory = "./chroma_db"
            mock_settings.chromadb_collection_name = "test_collection"

            mock_client = Mock()
            mock_collection = Mock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection

            store = ChromaDBVectorStore()

            assert store.client == mock_client
            assert store.collection == mock_collection
            mock_chromadb.PersistentClient.assert_called_once()

    def test_chromadb_vector_store_init_remote(self) -> None:
        """Test ChromaDB initialization for remote storage."""
        with patch('embed.chromadb') as mock_chromadb, \
             patch('embed.settings') as mock_settings:

            mock_settings.chromadb_host = "remote-host"
            mock_settings.chromadb_port = 8001
            mock_settings.chromadb_collection_name = "test_collection"

            mock_client = Mock()
            mock_collection = Mock()
            mock_chromadb.HttpClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection

            store = ChromaDBVectorStore()

            assert store.client == mock_client
            assert store.collection == mock_collection
            mock_chromadb.HttpClient.assert_called_once()

    def test_pinecone_vector_store_init(self) -> None:
        """Test Pinecone initialization."""
        with patch('embed.Pinecone') as mock_pinecone_class, \
             patch('embed.settings') as mock_settings:

            mock_settings.pinecone_api_key = "test-api-key"
            mock_settings.pinecone_index_name = "test-index"
            mock_settings.embedding_dimensions = 1536

            mock_pinecone = Mock()
            mock_pinecone_class.return_value = mock_pinecone
            mock_pinecone.list_indexes.return_value = [Mock(name="existing-index")]
            mock_pinecone.create_index = Mock()

            mock_index = Mock()
            mock_pinecone.Index.return_value = mock_index

            store = PineconeVectorStore()

            assert store.pc == mock_pinecone
            assert store.index == mock_index
            mock_pinecone_class.assert_called_once_with(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_chromadb_add_documents(self) -> None:
        """Test adding documents to ChromaDB."""
        with patch('embed.chromadb') as mock_chromadb, \
             patch('embed.settings') as mock_settings:

            mock_settings.chromadb_host = "localhost"
            mock_settings.chromadb_port = 8000
            mock_settings.chromadb_persist_directory = "./chroma_db"
            mock_settings.chromadb_collection_name = "test_collection"

            mock_client = Mock()
            mock_collection = Mock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection

            store = ChromaDBVectorStore()

            documents = [
                EmbeddingDocument(
                    id="test-1",
                    content="Test content",
                    embedding=[0.1, 0.2, 0.3],
                    metadata={"type": "test"},
                    doc_type="feedback"
                )
            ]

            result = await store.add_documents(documents)

            assert result is True
            mock_collection.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_chromadb_search(self) -> None:
        """Test searching documents in ChromaDB."""
        with patch('embed.chromadb') as mock_chromadb, \
             patch('embed.settings') as mock_settings:

            mock_settings.chromadb_host = "localhost"
            mock_settings.chromadb_port = 8000
            mock_settings.chromadb_persist_directory = "./chroma_db"
            mock_settings.chromadb_collection_name = "test_collection"

            mock_client = Mock()
            mock_collection = Mock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection

            # Mock search results
            mock_collection.query.return_value = {
                "ids": [["doc-1", "doc-2"]],
                "documents": [["content 1", "content 2"]],
                "metadatas": [[{"doc_type": "problem"}, {"doc_type": "feedback"}]],
                "distances": [[0.1, 0.2]]
            }

            store = ChromaDBVectorStore()

            results = await store.search([0.1, 0.2, 0.3], limit=2)

            assert len(results) == 2
            assert results[0]["id"] == "doc-1"
            assert results[0]["content"] == "content 1"
            assert results[0]["score"] == 0.9  # 1 - 0.1
            mock_collection.query.assert_called_once()


class TestEmbeddingManager:
    """Test embedding manager functionality."""

    @pytest.fixture
    def sample_feedbacks(self) -> list[Feedback]:
        return [
            Feedback(
                type="feature_request",
                summary="Excel export needed",
                verbatim="We need Excel export",
                confidence=0.9,
                transcript_id="transcript1",
                timestamp=datetime.now(),
                context="Customer call"
            )
        ]

    @pytest.fixture
    def sample_problems(self) -> list[NotionProblem]:
        return [
            NotionProblem(
                id="problem1",
                title="Export Features",
                description="Add export capabilities",
                status="New",
                priority="High",
                tags=["export", "feature"],
                feedback_count=5,
                last_updated=datetime.now()
            )
        ]

    def test_get_vector_store_chromadb(self) -> None:
        """Test getting ChromaDB vector store."""
        with patch('embed.settings') as mock_settings, \
             patch('embed.ChromaDBVectorStore') as mock_store, \
             patch('embed.get_llm_client') as mock_get_client:

            mock_settings.vector_store = "chromadb"
            mock_instance = Mock()
            mock_store.return_value = mock_instance
            mock_get_client.return_value = Mock()

            manager = EmbeddingManager()

            assert manager.vector_store == mock_instance

    def test_get_vector_store_pinecone(self) -> None:
        """Test getting Pinecone vector store."""
        with patch('embed.settings') as mock_settings, \
             patch('embed.PineconeVectorStore') as mock_store, \
             patch('embed.get_llm_client') as mock_get_client:

            mock_settings.vector_store = "pinecone"
            mock_instance = Mock()
            mock_store.return_value = mock_instance
            mock_get_client.return_value = Mock()

            manager = EmbeddingManager()

            assert manager.vector_store == mock_instance

    def test_get_vector_store_unsupported(self) -> None:
        """Test getting unsupported vector store raises error."""
        with patch('embed.settings') as mock_settings, \
             patch('embed.get_llm_client') as mock_get_client:

            mock_settings.vector_store = "unsupported"
            mock_get_client.return_value = Mock()

            with pytest.raises(ValueError, match="Unsupported vector store"):
                EmbeddingManager()

    @pytest.mark.asyncio
    async def test_embed_feedbacks(self, sample_feedbacks: list[Feedback]) -> None:
        """Test embedding generation for feedbacks."""
        with patch('embed.get_llm_client') as mock_get_client, \
             patch('embed.settings') as mock_settings, \
             patch('embed.ChromaDBVectorStore'):

            mock_settings.vector_store = "chromadb"
            mock_llm_client = Mock()
            mock_llm_client.embed = AsyncMock(return_value=[[0.1, 0.2]])
            mock_get_client.return_value = mock_llm_client

            manager = EmbeddingManager()
            documents = await manager.embed_feedbacks(sample_feedbacks)

            assert len(documents) == 1
            assert documents[0].doc_type == "feedback"
            assert documents[0].embedding == [0.1, 0.2]
            assert "Excel export needed" in documents[0].content
            assert documents[0].metadata["type"] == "feature_request"
            assert documents[0].metadata["transcript_id"] == "transcript1"

    @pytest.mark.asyncio
    async def test_embed_problems(self, sample_problems: list[NotionProblem]) -> None:
        """Test embedding generation for problems."""
        with patch('embed.get_llm_client') as mock_get_client, \
             patch('embed.settings') as mock_settings, \
             patch('embed.ChromaDBVectorStore'):

            mock_settings.vector_store = "chromadb"
            mock_llm_client = Mock()
            mock_llm_client.embed = AsyncMock(return_value=[[0.3, 0.4]])
            mock_get_client.return_value = mock_llm_client

            manager = EmbeddingManager()
            documents = await manager.embed_problems(sample_problems)

            assert len(documents) == 1
            assert documents[0].doc_type == "problem"
            assert documents[0].embedding == [0.3, 0.4]
            assert "Export Features" in documents[0].content
            assert documents[0].metadata["notion_id"] == "problem1"
            assert documents[0].metadata["tags"] == "export,feature"

    @pytest.mark.asyncio
    async def test_search_similar_problems(self) -> None:
        """Test searching for similar problems."""
        with patch('embed.get_llm_client') as mock_get_client, \
             patch('embed.settings') as mock_settings, \
             patch('embed.ChromaDBVectorStore') as mock_vector_store:

            mock_settings.vector_store = "chromadb"
            mock_llm_client = Mock()
            mock_llm_client.embed = AsyncMock(return_value=[[0.5, 0.6]])
            mock_get_client.return_value = mock_llm_client

            mock_store_instance = Mock()
            mock_store_instance.search = AsyncMock(return_value=[
                {
                    "id": "problem_1",
                    "content": "Export functionality",
                    "doc_type": "problem",
                    "metadata": {"notion_id": "123"},
                    "score": 0.95
                }
            ])
            mock_vector_store.return_value = mock_store_instance

            manager = EmbeddingManager()
            results = await manager.search_similar_problems("Need export feature")

            assert len(results) == 1
            assert results[0]["doc_type"] == "problem"
            assert results[0]["metadata"]["notion_id"] == "123"
            mock_llm_client.embed.assert_called_once_with(["Need export feature"])
