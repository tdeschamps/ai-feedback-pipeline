"""
Minimal test suite to verify basic functionality works.
"""

import os
import sys
from unittest.mock import Mock, patch


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


def test_config_loads():
    """Test that config loads without errors."""
    import config

    assert config.settings is not None
    print("✓ Config loads successfully")


def test_basic_dataclasses():
    """Test basic dataclasses can be created."""
    from datetime import datetime

    # Mock external dependencies
    with patch.dict(
        "sys.modules",
        {
            "langchain.schema": Mock(),
            "langchain_anthropic": Mock(),
            "langchain_community.embeddings": Mock(),
            "langchain_community.llms": Mock(),
            "langchain_openai": Mock(),
            "chromadb": Mock(),
            "pinecone": Mock(),
            "notion_client": Mock(),
        },
    ):
        # Test extract module
        import extract

        feedback = extract.Feedback(
            type="feature_request",
            summary="Test feedback",
            verbatim="This is a test",
            confidence=0.9,
            transcript_id="test_id",
            timestamp=datetime.now(),
            context="test context",
        )
        assert feedback.type == "feature_request"
        print("✓ Feedback dataclass works")

        # Test embed module
        import embed

        doc = embed.EmbeddingDocument(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": "data"},
            doc_type="feedback",
        )
        assert doc.id == "test-1"
        print("✓ EmbeddingDocument dataclass works")


def test_vector_store_interfaces():
    """Test that vector store interfaces can be instantiated."""
    with (
        patch.dict(
            "sys.modules",
            {
                "langchain.schema": Mock(),
                "langchain_anthropic": Mock(),
                "langchain_community.embeddings": Mock(),
                "langchain_community.llms": Mock(),
                "langchain_openai": Mock(),
                "chromadb": Mock(),
                "pinecone": Mock(),
            },
        ),
        patch("config.settings") as mock_settings,
    ):
        mock_settings.vector_store = "chromadb"
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test"

        # Mock chromadb client and collection
        mock_client = Mock()
        mock_collection = Mock()

        with patch("embed.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection

            import embed

            store = embed.ChromaDBVectorStore()
            assert store.client == mock_client
            assert store.collection == mock_collection
            print("✓ ChromaDBVectorStore can be instantiated")


def run_all_tests():
    """Run all tests."""
    print("Running minimal test suite...")

    try:
        test_config_loads()
        test_basic_dataclasses()
        test_vector_store_interfaces()
        print("\n✅ All minimal tests passed!")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
