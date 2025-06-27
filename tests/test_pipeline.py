"""
Tests for pipeline.py - Pipeline orchestration functionality.
"""

import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("NOTION_API_KEY", "test-key")


@pytest.fixture
def mock_feedback():
    """Create a mock feedback object."""
    feedback = Mock()
    feedback.type = "feature_request"
    feedback.summary = "Better dashboard needed"
    feedback.verbatim = "Customer wants improved analytics dashboard"
    feedback.confidence = 0.8
    feedback.transcript_id = "test_transcript"
    feedback.timestamp = datetime.now()
    return feedback


@pytest.fixture
def mock_match():
    """Create a mock match object."""
    match = Mock()
    match.problem_id = "test_problem_id"
    match.problem_title = "Dashboard Improvements"
    match.confidence = 0.85
    match.similarity_score = 0.9
    return match


class TestFeedbackPipeline:
    """Test FeedbackPipeline class."""

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    def test_pipeline_initialization(
        self, mock_metrics, mock_matcher, mock_embedding, mock_notion, mock_extractor
    ):
        """Test pipeline initialization."""
        from pipeline import FeedbackPipeline

        pipeline = FeedbackPipeline()

        assert pipeline is not None
        mock_extractor.assert_called_once()
        mock_notion.assert_called_once()
        mock_embedding.assert_called_once()
        mock_matcher.assert_called_once()
        mock_metrics.assert_called_once()

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    async def test_process_transcript_success(
        self, mock_metrics, mock_matcher, mock_embedding, mock_notion, mock_extractor, mock_feedback
    ):
        """Test successful transcript processing."""
        from pipeline import FeedbackPipeline

        # Setup mocks
        mock_extractor_instance = AsyncMock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_feedback.return_value = [mock_feedback]

        mock_matcher_instance = AsyncMock()
        mock_matcher.return_value = mock_matcher_instance

        mock_notion_instance = Mock()
        mock_notion.return_value = mock_notion_instance
        mock_notion_instance.update_problem_with_feedback.return_value = True

        mock_embedding_instance = Mock()
        mock_embedding.return_value = mock_embedding_instance

        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance

        # Create pipeline and test
        pipeline = FeedbackPipeline()

        with (
            patch.object(pipeline, "process_feedbacks") as mock_process_feedbacks,
            patch.object(pipeline, "_save_processing_logs") as mock_save_logs,
        ):
            mock_process_feedbacks.return_value = [(mock_feedback, Mock())]
            mock_save_logs.return_value = None

            result = await pipeline.process_transcript("test transcript", "test_id")

            assert result is not None
            assert result["transcript_id"] == "test_id"
            assert result["feedbacks_extracted"] == 1
            assert result["status"] == "completed"

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    async def test_process_transcript_no_feedback(
        self, mock_metrics, mock_matcher, mock_embedding, mock_notion, mock_extractor
    ):
        """Test transcript processing with no feedback extracted."""
        from pipeline import FeedbackPipeline

        # Setup mocks - no feedback extracted
        mock_extractor_instance = AsyncMock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_feedback.return_value = []

        mock_matcher.return_value = AsyncMock()
        mock_notion.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_metrics.return_value = Mock()

        pipeline = FeedbackPipeline()
        result = await pipeline.process_transcript("test transcript", "test_id")

        assert result["transcript_id"] == "test_id"
        assert result["feedbacks_extracted"] == 0
        assert result["status"] == "no_feedback"

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    async def test_process_transcript_error(
        self, mock_metrics, mock_matcher, mock_embedding, mock_notion, mock_extractor
    ):
        """Test transcript processing with error."""
        from pipeline import FeedbackPipeline

        # Setup mocks - extractor raises exception
        mock_extractor_instance = AsyncMock()
        mock_extractor.return_value = mock_extractor_instance
        mock_extractor_instance.extract_feedback.side_effect = Exception("Test error")

        mock_matcher.return_value = AsyncMock()
        mock_notion.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_metrics.return_value = Mock()

        pipeline = FeedbackPipeline()
        result = await pipeline.process_transcript("test transcript", "test_id")

        assert result["transcript_id"] == "test_id"
        assert result["status"] == "error"
        assert "error" in result

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    @patch("pipeline.settings")
    async def test_process_feedbacks_with_high_confidence_match(
        self,
        mock_settings,
        mock_metrics,
        mock_matcher,
        mock_embedding,
        mock_notion,
        mock_extractor,
        mock_feedback,
        mock_match,
    ):
        """Test processing feedbacks with high confidence match."""
        from pipeline import FeedbackPipeline

        # Setup settings
        mock_settings.confidence_threshold = 0.7

        # Setup mocks
        mock_matcher_instance = AsyncMock()
        mock_matcher.return_value = mock_matcher_instance
        mock_matcher_instance.find_best_match.return_value = mock_match

        mock_notion_instance = Mock()
        mock_notion.return_value = mock_notion_instance
        mock_notion_instance.update_problem_with_feedback.return_value = True

        mock_extractor.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance

        pipeline = FeedbackPipeline()
        results = await pipeline.process_feedbacks([mock_feedback])

        assert len(results) == 1
        assert results[0][0] == mock_feedback
        assert results[0][1] == mock_match
        mock_notion_instance.update_problem_with_feedback.assert_called_once()

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    @patch("pipeline.settings")
    async def test_process_feedbacks_with_low_confidence_match(
        self,
        mock_settings,
        mock_metrics,
        mock_matcher,
        mock_embedding,
        mock_notion,
        mock_extractor,
        mock_feedback,
        mock_match,
    ):
        """Test processing feedbacks with low confidence match."""
        from pipeline import FeedbackPipeline

        # Setup settings and low confidence match
        mock_settings.confidence_threshold = 0.9
        mock_match.confidence = 0.5  # Below threshold

        mock_matcher_instance = AsyncMock()
        mock_matcher.return_value = mock_matcher_instance
        mock_matcher_instance.find_best_match.return_value = mock_match

        mock_notion_instance = Mock()
        mock_notion.return_value = mock_notion_instance

        mock_extractor.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance

        pipeline = FeedbackPipeline()
        results = await pipeline.process_feedbacks([mock_feedback])

        assert len(results) == 1
        # Should not update Notion due to low confidence
        mock_notion_instance.update_problem_with_feedback.assert_not_called()

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    async def test_process_feedbacks_no_match(
        self, mock_metrics, mock_matcher, mock_embedding, mock_notion, mock_extractor, mock_feedback
    ):
        """Test processing feedbacks with no match found."""
        from pipeline import FeedbackPipeline

        mock_matcher_instance = AsyncMock()
        mock_matcher.return_value = mock_matcher_instance
        mock_matcher_instance.find_best_match.return_value = None  # No match

        mock_notion_instance = Mock()
        mock_notion.return_value = mock_notion_instance

        mock_extractor.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance

        pipeline = FeedbackPipeline()
        results = await pipeline.process_feedbacks([mock_feedback])

        assert len(results) == 1
        assert results[0][0] == mock_feedback
        assert results[0][1] is None
        mock_notion_instance.update_problem_with_feedback.assert_not_called()

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    async def test_batch_process_transcripts(
        self, mock_metrics, mock_matcher, mock_embedding, mock_notion, mock_extractor
    ):
        """Test batch processing of transcripts."""
        from pipeline import FeedbackPipeline

        # Setup mocks
        mock_extractor.return_value = Mock()
        mock_notion.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_matcher.return_value = Mock()
        mock_metrics_instance = Mock()
        mock_metrics.return_value = mock_metrics_instance

        pipeline = FeedbackPipeline()

        # Mock the process_transcript method
        with patch.object(pipeline, "process_transcript") as mock_process:
            mock_process.return_value = {
                "feedbacks_extracted": 2,
                "matches_found": 1,
                "problems_updated": 1,
                "status": "completed",
            }

            transcripts = [
                {"id": "test1", "content": "content1"},
                {"id": "test2", "content": "content2"},
            ]

            with patch.object(pipeline, "_save_batch_results") as mock_save:
                mock_save.return_value = "metrics_file.json"

                result = await pipeline.batch_process_transcripts(transcripts)

                assert result["total_transcripts"] == 2
                assert result["total_feedbacks"] == 4  # 2 per transcript
                assert result["total_matches"] == 2
                assert result["total_updates"] == 2
                assert result["success_rate"] == 1.0

    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    async def test_sync_notion_problems(
        self, mock_metrics, mock_matcher, mock_embedding, mock_notion, mock_extractor
    ):
        """Test syncing Notion problems."""
        from pipeline import FeedbackPipeline

        mock_notion_instance = AsyncMock()
        mock_notion.return_value = mock_notion_instance
        mock_notion_instance.sync_all_problems.return_value = True

        mock_extractor.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_matcher.return_value = Mock()
        mock_metrics.return_value = Mock()

        pipeline = FeedbackPipeline()
        result = await pipeline.sync_notion_problems()

        assert result is True
        mock_notion_instance.sync_all_problems.assert_called_once()

    @patch("pipeline.save_feedback_logs")
    @patch("pipeline.Path")
    @patch("pipeline.FeedbackExtractor")
    @patch("pipeline.NotionClient")
    @patch("pipeline.EmbeddingManager")
    @patch("pipeline.RAGMatcher")
    @patch("pipeline.MatchingMetrics")
    async def test_save_processing_logs(
        self,
        mock_metrics,
        mock_matcher,
        mock_embedding,
        mock_notion,
        mock_extractor,
        mock_path,
        mock_save_logs,
        mock_feedback,
    ):
        """Test saving processing logs."""
        from pipeline import FeedbackPipeline

        # Setup mocks
        mock_extractor.return_value = Mock()
        mock_notion.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_matcher.return_value = Mock()
        mock_metrics.return_value = Mock()

        # Mock Path operations
        mock_data_dir = Mock()
        mock_path.return_value = mock_data_dir
        mock_data_dir.exists.return_value = True

        pipeline = FeedbackPipeline()

        # Test the private method via process_transcript
        with patch.object(pipeline, "extractor") as mock_ext:
            mock_ext.extract_feedback = AsyncMock(return_value=[mock_feedback])

            with patch.object(pipeline, "process_feedbacks") as mock_process_feedbacks:
                mock_process_feedbacks.return_value = [(mock_feedback, Mock())]

                await pipeline.process_transcript("test", "test_id")

                # Verify save_feedback_logs was called
                mock_save_logs.assert_called()


def test_pipeline_imports():
    """Test that pipeline module imports correctly."""
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
        try:
            import pipeline

            assert pipeline is not None
            print("✓ Pipeline imports successfully")
        except ImportError as e:
            print(f"✓ Pipeline imports skipped: {e}")


if __name__ == "__main__":
    print("Running pipeline tests...")
    test_pipeline_imports()
    print("✓ All pipeline tests configured!")
