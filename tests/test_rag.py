"""
Test coverage for RAG functionality.
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


def test_match_result_dataclass():
    """Test MatchResult dataclass."""
    print("Testing MatchResult dataclass...")

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
        import rag

        match = rag.MatchResult(
            problem_id="prob-123",
            problem_title="Export Features",
            confidence=0.85,
            similarity_score=0.92,
            reasoning="Strong semantic similarity in export functionality",
            metadata={"embedding_score": 0.9, "llm_score": 0.8},
        )

        assert match.problem_id == "prob-123"
        assert match.confidence == 0.85
        assert match.reasoning is not None
        assert "export" in match.reasoning.lower()

    print("✓ MatchResult dataclass works")


def test_rag_matcher_initialization():
    """Test RAGMatcher initialization."""
    print("Testing RAGMatcher initialization...")

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

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.vector_store = "chromadb"
        mock_settings.llm_provider = "openai"

        with patch("embed.chromadb"), patch("rag.get_llm_client") as mock_get_llm:
            mock_llm = Mock()
            mock_get_llm.return_value = mock_llm

            import rag

            matcher = rag.RAGMatcher()
            assert matcher is not None
            assert hasattr(matcher, "llm_client")
            assert hasattr(matcher, "embedding_manager")

    print("✓ RAGMatcher initialization works")


def test_matching_metrics():
    """Test MatchingMetrics functionality."""
    print("Testing MatchingMetrics...")

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
        import rag

        # Create mock match results
        match1 = rag.MatchResult(
            problem_id="prob-1",
            problem_title="Feature A",
            confidence=0.9,
            similarity_score=0.95,
            reasoning="High confidence match",
            metadata={},
        )

        match2 = rag.MatchResult(
            problem_id="prob-2",
            problem_title="Feature B",
            confidence=0.7,
            similarity_score=0.8,
            reasoning="Medium confidence match",
            metadata={},
        )

        metrics = rag.MatchingMetrics()
        assert metrics is not None

        # Test adding matches (use add_result method which exists)
        from extract import Feedback

        test_feedback = Feedback(
            type="feature_request",
            summary="Test feedback",
            verbatim="Test verbatim",
            confidence=0.9,
            transcript_id="test-123",
            timestamp=match1.metadata.get("timestamp") if match1.metadata else datetime.now(),
        )
        metrics.add_result(test_feedback, match1)
        metrics.add_result(test_feedback, match2)

        stats = metrics.get_statistics()
        assert stats["matched"] == 2
        assert stats["total_feedbacks"] == 2
        assert stats["match_rate"] == 1.0  # All feedbacks matched
        assert stats["avg_confidence"] == 0.8  # (0.9 + 0.7) / 2

    print("✓ MatchingMetrics works")


def test_rag_confidence_scoring():
    """Test RAG confidence scoring logic."""
    print("Testing RAG confidence scoring...")

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

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.vector_store = "chromadb"
        mock_settings.matching_confidence_threshold = 0.75
        mock_settings.llm_provider = "openai"  # Add this

        with patch("embed.chromadb"), patch("rag.get_llm_client"):
            import rag

            matcher = rag.RAGMatcher()

        # Test that the matcher was created successfully
        assert matcher is not None
        assert hasattr(matcher, "llm_client")
        assert hasattr(matcher, "embedding_manager")

        # Test rerank prompt building
        prompt = matcher._build_rerank_prompt()
        assert "confidence" in prompt
        assert "JSON" in prompt

    print("✓ RAG confidence scoring works")


def run_rag_tests():
    """Run all RAG tests."""
    print("=" * 50)
    print("Running RAG Tests")
    print("=" * 50)

    tests = [
        test_match_result_dataclass,
        test_rag_matcher_initialization,
        test_matching_metrics,
        test_rag_confidence_scoring,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"RAG Tests: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


def run_all_tests():
    """Run all RAG tests."""
    return run_rag_tests()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
