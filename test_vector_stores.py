#!/usr/bin/env python3
"""
Simple test script to verify ChromaDB and Pinecone vector stores work.
"""

import asyncio
import os
from unittest.mock import Mock, patch


# Set environment variables to avoid API key errors
os.environ["OPENAI_API_KEY"] = "test-key"

from embed import ChromaDBVectorStore, EmbeddingDocument, PineconeVectorStore


async def test_chromadb_basic():
    """Test ChromaDB basic functionality with mocks."""
    print("Testing ChromaDB...")

    with patch("embed.chromadb") as mock_chromadb, patch("embed.settings") as mock_settings:
        # Configure settings
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./chroma_db"
        mock_settings.chromadb_collection_name = "test_collection"

        # Mock ChromaDB client and collection
        mock_client = Mock()
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        # Test initialization
        store = ChromaDBVectorStore()
        print("✓ ChromaDB store initialized")

        # Test adding documents
        documents = [
            EmbeddingDocument(
                id="test-doc-1",
                content="This is a test document",
                embedding=[0.1, 0.2, 0.3],
                metadata={"type": "test"},
                doc_type="feedback",
            )
        ]

        result = await store.add_documents(documents)
        print(f"✓ Add documents result: {result}")

        # Test search
        mock_collection.query.return_value = {
            "ids": [["test-doc-1"]],
            "documents": [["This is a test document"]],
            "metadatas": [[{"type": "test", "doc_type": "feedback"}]],
            "distances": [[0.1]],
        }

        search_results = await store.search([0.1, 0.2, 0.3], limit=1)
        print(f"✓ Search results: {len(search_results)} documents found")

        print("ChromaDB test completed successfully!\n")


async def test_pinecone_basic():
    """Test Pinecone basic functionality with mocks."""
    print("Testing Pinecone...")

    with patch("embed.Pinecone") as mock_pinecone_class, patch("embed.settings") as mock_settings:
        # Configure settings
        mock_settings.pinecone_api_key = "test-api-key"
        mock_settings.pinecone_index_name = "test-index"
        mock_settings.embedding_dimensions = 1536

        # Mock Pinecone client and index
        mock_pinecone = Mock()
        mock_pinecone_class.return_value = mock_pinecone
        mock_pinecone.list_indexes.return_value = []  # No existing indexes
        mock_pinecone.create_index = Mock()

        mock_index = Mock()
        mock_pinecone.Index.return_value = mock_index

        # Test initialization
        store = PineconeVectorStore()
        print("✓ Pinecone store initialized")

        # Test adding documents
        documents = [
            EmbeddingDocument(
                id="test-doc-1",
                content="This is a test document",
                embedding=[0.1] * 1536,  # Match embedding dimensions
                metadata={"type": "test"},
                doc_type="problem",
            )
        ]

        result = await store.add_documents(documents)
        print(f"✓ Add documents result: {result}")

        # Test search
        mock_match = Mock()
        mock_match.id = "test-doc-1"
        mock_match.score = 0.95
        mock_match.metadata = {
            "type": "test",
            "doc_type": "problem",
            "content": "This is a test document",
        }

        mock_response = Mock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response

        search_results = await store.search([0.1] * 1536, limit=1)
        print(f"✓ Search results: {len(search_results)} documents found")

        print("Pinecone test completed successfully!\n")


async def main():
    """Run all tests."""
    print("=== Vector Store Integration Tests ===\n")

    try:
        await test_chromadb_basic()
        await test_pinecone_basic()
        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
