"""
Comprehensive pytest tests for pipeline.py
"""

import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("NOTION_TOKEN", "test-token")


@pytest.fixture
def mock_modules():
    """Get standardized mock modules to prevent import issues."""
    return {
        "notion_client": Mock(),
        "langchain_core": Mock(),
        "langchain_core.messages": Mock(),
        "langchain_openai": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_ollama": Mock(),
        "langchain_huggingface": Mock(),
        "pydantic": Mock(),
        "pydantic._internal": Mock(),
        "pydantic._internal._config": Mock(),
        "pydantic_settings": Mock(),
        "pydantic_settings.main": Mock(),
        "chromadb": Mock(),
        "chromadb.config": Mock(),
        "openai": Mock(),
        "anthropic": Mock(),
    }


@pytest.fixture
def mock_feedback():
    """Create a mock feedback object for testing."""
    import extract

    return extract.Feedback(
        type="feature_request",
        summary="Test feedback",
        verbatim="This is test verbatim",
        confidence=0.9,
        transcript_id="test-transcript",
        timestamp=datetime.now(),
        context="Test context",
    )


@pytest.fixture
def mock_match():
    """Create a mock match object for testing."""
    mock = Mock()
    mock.confidence = 0.8
    mock.problem_id = "problem-123"
    mock.problem_title = "Test Problem"
    mock.similarity_score = 0.85
    mock.reasoning = "Test reasoning"
    return mock


class TestFeedbackPipelineInitialization:
    """Test the FeedbackPipeline class initialization."""

    def test_pipeline_initialization(self, mock_modules):
        """Test basic FeedbackPipeline initialization."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor") as mock_extractor,
            patch("pipeline.NotionClient") as mock_notion,
            patch("pipeline.EmbeddingManager") as mock_embedding,
            patch("pipeline.RAGMatcher") as mock_matcher,
            patch("pipeline.MatchingMetrics") as mock_metrics,
        ):
            import pipeline

            # Create pipeline instance
            pipe = pipeline.FeedbackPipeline()

            # Verify all components are initialized
            assert pipe.extractor is not None
            assert pipe.notion_client is not None
            assert pipe.embedding_manager is not None
            assert pipe.matcher is not None
            assert pipe.metrics is not None

            # Verify constructors were called
            mock_extractor.assert_called_once()
            mock_notion.assert_called_once()
            mock_embedding.assert_called_once()
            mock_matcher.assert_called_once()
            mock_metrics.assert_called_once()


class TestProcessTranscript:
    """Test the process_transcript method."""

    @pytest.mark.asyncio
    async def test_process_transcript_no_feedback(self, mock_modules):
        """Test process_transcript that extracts no feedback."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor") as mock_extractor,
            patch("pipeline.NotionClient"),
            patch("pipeline.EmbeddingManager"),
            patch("pipeline.RAGMatcher"),
            patch("pipeline.MatchingMetrics"),
        ):
            import pipeline

            # Setup mocks
            mock_extractor_instance = Mock()
            mock_extractor_instance.extract_feedback = AsyncMock(return_value=[])
            mock_extractor.return_value = mock_extractor_instance

            pipe = pipeline.FeedbackPipeline()

            # Test processing empty feedback
            result = await pipe.process_transcript("Test transcript", "test-id")

            # Verify result structure
            assert result["transcript_id"] == "test-id"
            assert result["feedbacks_extracted"] == 0
            assert result["matches_found"] == 0
            assert result["problems_updated"] == 0
            assert result["status"] == "no_feedback"

            # Verify extractor was called
            mock_extractor_instance.extract_feedback.assert_called_once_with(
                "Test transcript", "test-id"
            )

    @pytest.mark.asyncio
    async def test_process_transcript_with_error(self, mock_modules):
        """Test process_transcript that encounters an error."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor") as mock_extractor,
            patch("pipeline.NotionClient"),
            patch("pipeline.EmbeddingManager"),
            patch("pipeline.RAGMatcher"),
            patch("pipeline.MatchingMetrics"),
        ):
            import pipeline

            # Setup mocks to raise exception
            mock_extractor_instance = Mock()
            mock_extractor_instance.extract_feedback = AsyncMock(
                side_effect=Exception("Extraction failed")
            )
            mock_extractor.return_value = mock_extractor_instance

            pipe = pipeline.FeedbackPipeline()

            # Test processing with error
            result = await pipe.process_transcript("Test transcript", "error-id")

            # Verify error handling
            assert result["transcript_id"] == "error-id"
            assert result["status"] == "error"
            assert "error" in result
            assert "Extraction failed" in result["error"]

    @pytest.mark.asyncio
    async def test_process_transcript_with_feedback_success(
        self, mock_modules, mock_feedback, mock_match
    ):
        """Test process_transcript with successful feedback processing."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("config.settings") as mock_settings,
        ):
            # Setup settings first
            mock_settings.confidence_threshold = 0.7

            with (
                patch("pipeline.FeedbackExtractor") as mock_extractor,
                patch("pipeline.NotionClient") as mock_notion,
                patch("pipeline.EmbeddingManager"),
                patch("pipeline.RAGMatcher") as mock_matcher,
                patch("pipeline.MatchingMetrics") as mock_metrics,
                patch("pipeline.save_feedback_logs"),
                patch("builtins.open", mock_open()),
                patch("json.dump"),
            ):
                import pipeline

                # Setup mocks
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_feedback = AsyncMock(return_value=[mock_feedback])
                mock_extractor.return_value = mock_extractor_instance

                mock_notion_instance = Mock()
                mock_notion_instance.update_problem_with_feedback = Mock(return_value=True)
                mock_notion.return_value = mock_notion_instance

                mock_matcher_instance = Mock()
                mock_matcher_instance.find_best_match = AsyncMock(return_value=mock_match)
                mock_matcher.return_value = mock_matcher_instance

                mock_metrics_instance = Mock()
                mock_metrics.return_value = mock_metrics_instance

                pipe = pipeline.FeedbackPipeline()

                # Test processing with successful feedback
                result = await pipe.process_transcript("Test transcript", "success-id")

                # Verify result structure
                assert result["transcript_id"] == "success-id"
                assert result["feedbacks_extracted"] == 1
                assert result["matches_found"] == 1
                assert result["problems_updated"] == 1
                assert result["status"] == "completed"
                assert "results" in result

                # Verify calls
                mock_extractor_instance.extract_feedback.assert_called_once()
                mock_matcher_instance.find_best_match.assert_called_once_with(mock_feedback)
                mock_notion_instance.update_problem_with_feedback.assert_called_once_with(
                    "problem-123", mock_feedback, 0.8
                )
                mock_metrics_instance.add_result.assert_called_once_with(mock_feedback, mock_match)


class TestSyncNotionProblems:
    """Test the sync_notion_problems method."""

    @pytest.mark.asyncio
    async def test_sync_notion_problems_success(self, mock_modules):
        """Test successful sync of Notion problems."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor"),
            patch("pipeline.NotionClient") as mock_notion,
            patch("pipeline.EmbeddingManager") as mock_embedding,
            patch("pipeline.RAGMatcher"),
            patch("pipeline.MatchingMetrics"),
        ):
            import notion
            import pipeline

            # Create mock problems
            mock_problem = notion.NotionProblem(
                id="problem-1",
                title="Test Problem",
                description="Test description",
                status="Open",
                feedback_count=0,
            )

            # Setup mocks
            mock_notion_instance = Mock()
            mock_notion_instance.get_all_problems = Mock(return_value=[mock_problem])
            mock_notion.return_value = mock_notion_instance

            mock_embedding_instance = Mock()
            mock_embedding_instance.refresh_problem_embeddings = AsyncMock(return_value=True)
            mock_embedding.return_value = mock_embedding_instance

            pipe = pipeline.FeedbackPipeline()

            # Test successful sync
            result = await pipe.sync_notion_problems()

            # Verify result
            assert result is True

            # Verify calls
            mock_notion_instance.get_all_problems.assert_called_once()
            mock_embedding_instance.refresh_problem_embeddings.assert_called_once_with(
                [mock_problem]
            )

    @pytest.mark.asyncio
    async def test_sync_notion_problems_no_problems(self, mock_modules):
        """Test sync with no problems returned."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor"),
            patch("pipeline.NotionClient") as mock_notion,
            patch("pipeline.EmbeddingManager"),
            patch("pipeline.RAGMatcher"),
            patch("pipeline.MatchingMetrics"),
        ):
            import pipeline

            # Setup mocks
            mock_notion_instance = Mock()
            mock_notion_instance.get_all_problems = Mock(return_value=[])  # Empty list
            mock_notion.return_value = mock_notion_instance

            pipe = pipeline.FeedbackPipeline()

            # Test sync with no problems
            result = await pipe.sync_notion_problems()

            # Verify result
            assert result is False

            # Verify call
            mock_notion_instance.get_all_problems.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_notion_problems_error(self, mock_modules):
        """Test sync with error handling."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor"),
            patch("pipeline.NotionClient") as mock_notion,
            patch("pipeline.EmbeddingManager"),
            patch("pipeline.RAGMatcher"),
            patch("pipeline.MatchingMetrics"),
        ):
            import pipeline

            # Setup mocks
            mock_notion_instance = Mock()
            mock_notion_instance.get_all_problems = Mock(side_effect=Exception("Notion API Error"))
            mock_notion.return_value = mock_notion_instance

            pipe = pipeline.FeedbackPipeline()

            # Test sync with error
            result = await pipe.sync_notion_problems()

            # Verify result
            assert result is False


class TestBatchProcessTranscripts:
    """Test the batch_process_transcripts method."""

    @pytest.mark.asyncio
    async def test_batch_process_transcripts_success(self, mock_modules):
        """Test successful batch processing of transcripts."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor"),
            patch("pipeline.NotionClient") as mock_notion,
            patch("pipeline.EmbeddingManager") as mock_embedding,
            patch("pipeline.RAGMatcher"),
            patch("pipeline.MatchingMetrics") as mock_metrics,
            patch("pathlib.Path.mkdir"),
            patch("builtins.open", mock_open()),
        ):
            import pipeline

            # Setup mocks for sync
            mock_notion_instance = Mock()
            mock_notion_instance.get_all_problems = Mock(return_value=[])
            mock_notion.return_value = mock_notion_instance

            mock_embedding_instance = Mock()
            mock_embedding_instance.refresh_problem_embeddings = AsyncMock(return_value=True)
            mock_embedding.return_value = mock_embedding_instance

            mock_metrics_instance = Mock()
            mock_metrics_instance.save_metrics = Mock()
            mock_metrics.return_value = mock_metrics_instance

            # Mock process_transcript method
            pipe = pipeline.FeedbackPipeline()
            pipe.process_transcript = AsyncMock(
                side_effect=[
                    {
                        "transcript_id": "test-1",
                        "feedbacks_extracted": 2,
                        "matches_found": 1,
                        "problems_updated": 1,
                        "status": "completed",
                    },
                    {
                        "transcript_id": "test-2",
                        "feedbacks_extracted": 1,
                        "matches_found": 0,
                        "problems_updated": 0,
                        "status": "completed",
                    },
                ]
            )

            # Test data
            transcripts = [
                {"id": "test-1", "content": "First transcript content"},
                {"id": "test-2", "content": "Second transcript content"},
            ]

            # Test batch processing
            result = await pipe.batch_process_transcripts(transcripts)

            # Verify result structure
            assert result["total_transcripts"] == 2
            assert result["total_feedbacks"] == 3
            assert result["total_matches"] == 1
            assert result["total_updates"] == 1
            assert result["success_rate"] == 1 / 3  # 1 match out of 3 feedbacks
            assert len(result["results"]) == 2
            assert "metrics_file" in result

            # Verify calls
            assert pipe.process_transcript.call_count == 2
            mock_metrics_instance.save_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_process_transcripts_empty_content(self, mock_modules):
        """Test batch processing with empty transcript content."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("pipeline.FeedbackExtractor"),
            patch("pipeline.NotionClient") as mock_notion,
            patch("pipeline.EmbeddingManager") as mock_embedding,
            patch("pipeline.RAGMatcher"),
            patch("pipeline.MatchingMetrics") as mock_metrics,
        ):
            import pipeline

            # Setup mocks for sync
            mock_notion_instance = Mock()
            mock_notion_instance.get_all_problems = Mock(return_value=[])
            mock_notion.return_value = mock_notion_instance

            mock_embedding_instance = Mock()
            mock_embedding_instance.refresh_problem_embeddings = AsyncMock(return_value=True)
            mock_embedding.return_value = mock_embedding_instance

            mock_metrics_instance = Mock()
            mock_metrics_instance.save_metrics = Mock()
            mock_metrics.return_value = mock_metrics_instance

            pipe = pipeline.FeedbackPipeline()
            pipe.process_transcript = AsyncMock(
                return_value={
                    "transcript_id": "test-1",
                    "feedbacks_extracted": 1,
                    "matches_found": 0,
                    "problems_updated": 0,
                    "status": "completed",
                }
            )

            # Test data with empty content
            transcripts = [
                {"id": "test-1", "content": "Valid content"},
                {"id": "test-2", "content": ""},  # Empty content - should be skipped
            ]

            # Test batch processing
            await pipe.batch_process_transcripts(transcripts)

            # Only one transcript should be processed (the valid one)
            assert pipe.process_transcript.call_count == 1
            pipe.process_transcript.assert_called_with("Valid content", "test-1")


class TestProcessFeedbacks:
    """Test the process_feedbacks method."""

    @pytest.mark.asyncio
    async def test_process_feedbacks_low_confidence_match(self, mock_modules):
        """Test process_feedbacks with low confidence match."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("config.settings") as mock_settings,
        ):
            # Setup settings first
            mock_settings.confidence_threshold = 0.8  # High threshold

            with (
                patch("pipeline.FeedbackExtractor"),
                patch("pipeline.NotionClient") as mock_notion,
                patch("pipeline.EmbeddingManager"),
                patch("pipeline.RAGMatcher") as mock_matcher,
                patch("pipeline.MatchingMetrics") as mock_metrics,
            ):
                import extract
                import pipeline

                # Create mock feedback
                mock_feedback = extract.Feedback(
                    type="customer_pain",
                    summary="Low confidence feedback",
                    verbatim="This is low confidence verbatim",
                    confidence=0.9,
                    transcript_id="low-conf-transcript",
                    timestamp=datetime.now(),
                    context="Test context",
                )

                # Create mock match with low confidence
                mock_match = Mock()
                mock_match.confidence = 0.6  # Below threshold
                mock_match.problem_id = "problem-456"
                mock_match.problem_title = "Low Confidence Problem"

                # Setup mocks
                mock_matcher_instance = Mock()
                mock_matcher_instance.find_best_match = AsyncMock(return_value=mock_match)
                mock_matcher.return_value = mock_matcher_instance

                mock_notion_instance = Mock()
                mock_notion_instance.update_problem_with_feedback = Mock(return_value=True)
                mock_notion.return_value = mock_notion_instance

                mock_metrics_instance = Mock()
                mock_metrics.return_value = mock_metrics_instance

                pipe = pipeline.FeedbackPipeline()

                # Test processing with low confidence match
                results = await pipe.process_feedbacks([mock_feedback])

                # Verify results
                assert len(results) == 1
                feedback, match = results[0]
                assert feedback == mock_feedback
                assert match == mock_match

                # Verify no update was made due to low confidence
                mock_notion_instance.update_problem_with_feedback.assert_not_called()
                mock_metrics_instance.add_result.assert_called_once_with(mock_feedback, mock_match)

    @pytest.mark.asyncio
    async def test_process_feedbacks_no_match(self, mock_modules):
        """Test process_feedbacks with no match found."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("config.settings") as mock_settings,
        ):
            # Setup settings first
            mock_settings.confidence_threshold = 0.7

            with (
                patch("pipeline.FeedbackExtractor"),
                patch("pipeline.NotionClient") as mock_notion,
                patch("pipeline.EmbeddingManager"),
                patch("pipeline.RAGMatcher") as mock_matcher,
                patch("pipeline.MatchingMetrics") as mock_metrics,
            ):
                import extract
                import pipeline

                # Create mock feedback
                mock_feedback = extract.Feedback(
                    type="feature_request",
                    summary="No match feedback",
                    verbatim="This feedback has no match",
                    confidence=0.8,
                    transcript_id="no-match-transcript",
                    timestamp=datetime.now(),
                    context="Test context",
                )

                # Setup mocks for no match
                mock_matcher_instance = Mock()
                mock_matcher_instance.find_best_match = AsyncMock(return_value=None)
                mock_matcher.return_value = mock_matcher_instance

                mock_notion_instance = Mock()
                mock_notion.return_value = mock_notion_instance

                mock_metrics_instance = Mock()
                mock_metrics.return_value = mock_metrics_instance

                pipe = pipeline.FeedbackPipeline()

                # Test processing with no match
                results = await pipe.process_feedbacks([mock_feedback])

                # Verify results
                assert len(results) == 1
                feedback, match = results[0]
                assert feedback == mock_feedback
                assert match is None

                # Verify calls
                mock_metrics_instance.add_result.assert_called_once_with(mock_feedback, None)

    @pytest.mark.asyncio
    async def test_process_feedbacks_error_handling(self, mock_modules):
        """Test process_feedbacks with error during processing."""
        with (
            patch.dict("sys.modules", mock_modules),
            patch("config.settings") as mock_settings,
        ):
            # Setup settings first
            mock_settings.confidence_threshold = 0.7

            with (
                patch("pipeline.FeedbackExtractor"),
                patch("pipeline.NotionClient"),
                patch("pipeline.EmbeddingManager"),
                patch("pipeline.RAGMatcher") as mock_matcher,
                patch("pipeline.MatchingMetrics") as mock_metrics,
            ):
                import extract
                import pipeline

                # Create mock feedback
                mock_feedback = extract.Feedback(
                    type="customer_pain",
                    summary="Error feedback",
                    verbatim="This feedback causes an error",
                    confidence=0.8,
                    transcript_id="error-transcript",
                    timestamp=datetime.now(),
                    context="Test context",
                )

                # Setup mocks to raise exception
                mock_matcher_instance = Mock()
                mock_matcher_instance.find_best_match = AsyncMock(
                    side_effect=Exception("Matching error")
                )
                mock_matcher.return_value = mock_matcher_instance

                mock_metrics_instance = Mock()
                mock_metrics.return_value = mock_metrics_instance

                pipe = pipeline.FeedbackPipeline()

                # Test processing with error
                results = await pipe.process_feedbacks([mock_feedback])

                # Verify results
                assert len(results) == 1
                feedback, match = results[0]
                assert feedback == mock_feedback
                assert match is None  # Error should result in None match
