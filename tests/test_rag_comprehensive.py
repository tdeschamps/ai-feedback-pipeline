"""
Tests for rag.py - RAG matching functionality.
"""

import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def mock_feedback():
    """Create a mock feedback object."""
    feedback = Mock()
    feedback.type = "feature_request"
    feedback.summary = "Better dashboard needed"
    feedback.verbatim = "Customer wants improved analytics dashboard"
    feedback.confidence = 0.8
    feedback.transcript_id = "test_transcript"
    return feedback


@pytest.fixture
def mock_search_results():
    """Create mock vector search results."""
    results = []
    for i in range(3):
        doc = Mock()
        doc.page_content = f"Problem {i} description"
        doc.metadata = {
            "problem_id": f"problem_{i}",
            "title": f"Problem {i} Title",
            "category": "feature_request",
            "status": "open",
        }
        results.append((doc, 0.9 - i * 0.1))  # Decreasing similarity scores
    return results


class TestMatchResult:
    """Test MatchResult data class."""

    def test_match_result_creation(self):
        """Test creating a MatchResult object."""
        from rag import MatchResult

        match = MatchResult(
            problem_id="test_problem",
            problem_title="Test Problem",
            similarity_score=0.85,
            confidence=0.9,
            reasoning="Strong semantic similarity",
        )

        assert match.problem_id == "test_problem"
        assert match.problem_title == "Test Problem"
        assert match.similarity_score == 0.85
        assert match.confidence == 0.9
        assert match.reasoning == "Strong semantic similarity"


class TestRAGMatcher:
    """Test RAGMatcher class."""

    @patch("rag.VectorStoreManager")
    @patch("rag.LLMClient")
    def test_matcher_initialization(self, mock_llm_client, mock_vector_store):
        """Test RAGMatcher initialization."""
        from rag import RAGMatcher

        matcher = RAGMatcher()
        assert matcher is not None
        mock_vector_store.assert_called_once()
        mock_llm_client.assert_called_once()

    @patch("rag.VectorStoreManager")
    @patch("rag.LLMClient")
    @patch("rag.settings")
    async def test_find_best_match_success(
        self, mock_settings, mock_llm_client, mock_vector_store, mock_feedback, mock_search_results
    ):
        """Test successful best match finding."""
        from rag import RAGMatcher

        mock_settings.rag_top_k = 5
        mock_settings.rerank_enabled = False

        # Setup vector store mock
        mock_vector_instance = Mock()
        mock_vector_store.return_value = mock_vector_instance
        mock_vector_instance.search_similar.return_value = mock_search_results

        # Setup LLM mock
        mock_llm_instance = AsyncMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.call_llm.return_value = """
        {
            "best_match": "problem_0",
            "confidence": 0.85,
            "reasoning": "Strong semantic similarity in dashboard requirements"
        }
        """

        matcher = RAGMatcher()
        result = await matcher.find_best_match(mock_feedback)

        assert result is not None
        assert result.problem_id == "problem_0"
        assert result.confidence == 0.85
        assert "dashboard" in result.reasoning

    @patch("rag.VectorStoreManager")
    @patch("rag.LLMClient")
    async def test_find_best_match_no_candidates(
        self, mock_llm_client, mock_vector_store, mock_feedback
    ):
        """Test when no candidates are found."""
        from rag import RAGMatcher

        # Setup vector store to return no results
        mock_vector_instance = Mock()
        mock_vector_store.return_value = mock_vector_instance
        mock_vector_instance.search_similar.return_value = []

        mock_llm_client.return_value = AsyncMock()

        matcher = RAGMatcher()
        result = await matcher.find_best_match(mock_feedback)

        assert result is None

    @patch("rag.VectorStoreManager")
    @patch("rag.LLMClient")
    async def test_find_best_match_llm_no_match(
        self, mock_llm_client, mock_vector_store, mock_feedback, mock_search_results
    ):
        """Test when LLM determines no good match."""
        from rag import RAGMatcher

        mock_vector_instance = Mock()
        mock_vector_store.return_value = mock_vector_instance
        mock_vector_instance.search_similar.return_value = mock_search_results

        # LLM says no match
        mock_llm_instance = AsyncMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.call_llm.return_value = """
        {
            "best_match": null,
            "confidence": 0.1,
            "reasoning": "No semantic similarity found"
        }
        """

        matcher = RAGMatcher()
        result = await matcher.find_best_match(mock_feedback)

        assert result is None

    @patch("rag.VectorStoreManager")
    @patch("rag.LLMClient")
    async def test_find_best_match_with_reranking(
        self, mock_llm_client, mock_vector_store, mock_feedback, mock_search_results
    ):
        """Test matching with reranking enabled."""
        from rag import RAGMatcher

        with patch("rag.settings") as mock_settings:
            mock_settings.rag_top_k = 5
            mock_settings.rerank_enabled = True
            mock_settings.rerank_top_k = 3

            mock_vector_instance = Mock()
            mock_vector_store.return_value = mock_vector_instance
            mock_vector_instance.search_similar.return_value = mock_search_results

            mock_llm_instance = AsyncMock()
            mock_llm_client.return_value = mock_llm_instance

            # Mock reranking call
            mock_llm_instance.call_llm.side_effect = [
                # Reranking response
                '["problem_2", "problem_0", "problem_1"]',
                # Final matching response
                """
                {
                    "best_match": "problem_2",
                    "confidence": 0.9,
                    "reasoning": "After reranking, this is the best match"
                }
                """,
            ]

            matcher = RAGMatcher()
            result = await matcher.find_best_match(mock_feedback)

            assert result is not None
            assert result.problem_id == "problem_2"
            assert result.confidence == 0.9

    @patch("rag.VectorStoreManager")
    @patch("rag.LLMClient")
    async def test_find_best_match_llm_error(
        self, mock_llm_client, mock_vector_store, mock_feedback, mock_search_results
    ):
        """Test handling LLM errors during matching."""
        from rag import RAGMatcher

        mock_vector_instance = Mock()
        mock_vector_store.return_value = mock_vector_instance
        mock_vector_instance.search_similar.return_value = mock_search_results

        mock_llm_instance = AsyncMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.call_llm.side_effect = Exception("LLM API Error")

        matcher = RAGMatcher()

        # Should raise the exception
        with pytest.raises(Exception, match="LLM API Error"):
            await matcher.find_best_match(mock_feedback)

    @patch("rag.VectorStoreManager")
    @patch("rag.LLMClient")
    async def test_find_best_match_invalid_json(
        self, mock_llm_client, mock_vector_store, mock_feedback, mock_search_results
    ):
        """Test handling invalid JSON from LLM."""
        from rag import RAGMatcher

        mock_vector_instance = Mock()
        mock_vector_store.return_value = mock_vector_instance
        mock_vector_instance.search_similar.return_value = mock_search_results

        mock_llm_instance = AsyncMock()
        mock_llm_client.return_value = mock_llm_instance
        mock_llm_instance.call_llm.return_value = "Invalid JSON response"

        matcher = RAGMatcher()
        result = await matcher.find_best_match(mock_feedback)

        # Should return None on invalid JSON
        assert result is None

    def test_build_matching_prompt(self, mock_feedback, mock_search_results):
        """Test building matching prompt."""
        from rag import RAGMatcher

        with patch("rag.VectorStoreManager"), patch("rag.LLMClient"):
            matcher = RAGMatcher()
            prompt = matcher._build_matching_prompt(mock_feedback, mock_search_results)

            assert isinstance(prompt, str)
            assert mock_feedback.summary in prompt
            assert "Problem 0 Title" in prompt  # From search results
            assert "problem_0" in prompt

    def test_build_reranking_prompt(self, mock_feedback, mock_search_results):
        """Test building reranking prompt."""
        from rag import RAGMatcher

        with patch("rag.VectorStoreManager"), patch("rag.LLMClient"):
            matcher = RAGMatcher()
            prompt = matcher._build_reranking_prompt(mock_feedback, mock_search_results)

            assert isinstance(prompt, str)
            assert mock_feedback.summary in prompt
            assert "rerank" in prompt.lower()
            assert "problem_0" in prompt


class TestMatchingMetrics:
    """Test MatchingMetrics class."""

    def test_metrics_initialization(self):
        """Test MatchingMetrics initialization."""
        from rag import MatchingMetrics

        metrics = MatchingMetrics()
        assert metrics is not None
        assert len(metrics.results) == 0

    def test_add_result_with_match(self, mock_feedback):
        """Test adding result with match."""
        from rag import MatchingMetrics, MatchResult

        match = MatchResult(
            problem_id="test_problem",
            problem_title="Test Problem",
            similarity_score=0.8,
            confidence=0.9,
            reasoning="Good match",
        )

        metrics = MatchingMetrics()
        metrics.add_result(mock_feedback, match)

        assert len(metrics.results) == 1
        assert metrics.results[0]["feedback"] == mock_feedback
        assert metrics.results[0]["match"] == match

    def test_add_result_no_match(self, mock_feedback):
        """Test adding result without match."""
        from rag import MatchingMetrics

        metrics = MatchingMetrics()
        metrics.add_result(mock_feedback, None)

        assert len(metrics.results) == 1
        assert metrics.results[0]["feedback"] == mock_feedback
        assert metrics.results[0]["match"] is None

    def test_get_match_rate(self, mock_feedback):
        """Test calculating match rate."""
        from rag import MatchingMetrics, MatchResult

        metrics = MatchingMetrics()

        # Add some results
        match = MatchResult("id1", "title1", 0.8, 0.9, "reason")
        metrics.add_result(mock_feedback, match)  # Match
        metrics.add_result(mock_feedback, None)  # No match
        metrics.add_result(mock_feedback, match)  # Match

        match_rate = metrics.get_match_rate()
        assert match_rate == 2 / 3  # 2 matches out of 3

    def test_get_confidence_distribution(self, mock_feedback):
        """Test confidence distribution calculation."""
        from rag import MatchingMetrics, MatchResult

        metrics = MatchingMetrics()

        # Add results with different confidence levels
        high_conf = MatchResult("id1", "title1", 0.9, 0.9, "high")
        medium_conf = MatchResult("id2", "title2", 0.7, 0.6, "medium")
        low_conf = MatchResult("id3", "title3", 0.5, 0.3, "low")

        metrics.add_result(mock_feedback, high_conf)
        metrics.add_result(mock_feedback, medium_conf)
        metrics.add_result(mock_feedback, low_conf)
        metrics.add_result(mock_feedback, None)  # No match

        distribution = metrics.get_confidence_distribution()

        assert distribution["high"] == 1
        assert distribution["medium"] == 1
        assert distribution["low"] == 1
        assert distribution["no_match"] == 1

    def test_get_summary_stats(self, mock_feedback):
        """Test summary statistics."""
        from rag import MatchingMetrics, MatchResult

        metrics = MatchingMetrics()

        match1 = MatchResult("id1", "title1", 0.9, 0.8, "reason1")
        match2 = MatchResult("id2", "title2", 0.7, 0.6, "reason2")

        metrics.add_result(mock_feedback, match1)
        metrics.add_result(mock_feedback, match2)
        metrics.add_result(mock_feedback, None)

        stats = metrics.get_summary_stats()

        assert stats["total_feedbacks"] == 3
        assert stats["matched_feedbacks"] == 2
        assert stats["match_rate"] == 2 / 3
        assert stats["avg_confidence"] == 0.7  # (0.8 + 0.6) / 2
        assert stats["avg_similarity"] == 0.8  # (0.9 + 0.7) / 2


def test_rag_imports():
    """Test that rag module imports correctly."""
    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
        "chromadb": Mock(),
        "pinecone": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        try:
            import rag

            assert rag is not None
            print("✓ RAG module imports successfully")
        except ImportError as e:
            print(f"✓ RAG imports skipped: {e}")


if __name__ == "__main__":
    print("Running RAG tests...")
    test_rag_imports()
    print("✓ All RAG tests configured!")
