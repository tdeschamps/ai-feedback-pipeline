"""
Test coverage for RAG functionality.
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch
import pytest

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def mock_modules():
    """Fixture for mocking external modules."""
    return {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
        "chromadb": Mock(),
        "pinecone": Mock(),
        "notion_client": Mock(),
    }


def test_match_result_dataclass(mock_modules):
    """Test MatchResult dataclass."""
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


def test_rag_matcher_initialization(mock_modules):
    """Test RAGMatcher initialization."""
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


def test_matching_metrics(mock_modules):
    """Test MatchingMetrics functionality."""
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


def test_rag_confidence_scoring(mock_modules):
    """Test RAG confidence scoring logic."""
    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.vector_store = "chromadb"
        mock_settings.matching_confidence_threshold = 0.75
        mock_settings.llm_provider = "openai"

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
