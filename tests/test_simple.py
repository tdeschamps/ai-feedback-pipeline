"""
Simple test runner without pytest to verify functionality.
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


def test_config_loads():
    """Test configuration loads."""
    print("Testing config loading...")
    import config

    # In test environment, settings might be None, which is expected
    assert hasattr(config, "settings")  # Check the attribute exists
    print("✓ Config module loads successfully")


def test_dataclasses():
    """Test dataclasses work."""
    print("Testing dataclasses...")

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
        import embed
        import extract

        # Test Feedback
        feedback = extract.Feedback(
            type="feature_request",
            summary="Test",
            verbatim="Test feedback",
            confidence=0.9,
            transcript_id="test",
            timestamp=datetime.now(),
            context="test",
        )
        assert feedback.type == "feature_request"
        print("✓ Feedback dataclass works")

        # Test EmbeddingDocument
        doc = embed.EmbeddingDocument(
            id="test-1",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": "data"},
            doc_type="feedback",
        )
        assert doc.id == "test-1"
        print("✓ EmbeddingDocument dataclass works")


def test_vector_stores():
    """Test vector store initialization."""
    print("Testing vector stores...")

    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
        "chromadb": Mock(),
        "pinecone": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.vector_store = "chromadb"
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./test_db"
        mock_settings.chromadb_collection_name = "test"

        mock_client = Mock()
        mock_collection = Mock()

        with patch("embed.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection

            import embed

            store = embed.ChromaDBVectorStore()
            assert store.client == mock_client
            print("✓ ChromaDB vector store works")


def run_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running AI Feedback Pipeline Tests")
    print("=" * 50)

    tests = [
        test_config_loads,
        test_dataclasses,
        test_vector_stores,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
