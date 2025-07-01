"""
Comprehensive step-by-step tests for pipeline.py
"""

import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, mock_open, patch


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("NOTION_TOKEN", "test-token")


def get_mock_modules():
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


class TestFeedbackPipelineInitialization:
    """Test the FeedbackPipeline class initialization step by step."""

    def test_pipeline_initialization(self):
        """Test 1: Basic FeedbackPipeline initialization."""
        print("Test 1: Testing FeedbackPipeline initialization...")

        with (
            patch.dict("sys.modules", get_mock_modules()),
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

            print("✓ Test 1 passed: FeedbackPipeline initialization works")


class TestProcessTranscript:
    """Test the process_transcript method step by step."""

    def test_process_transcript_no_feedback(self):
        """Test 2: Process transcript that extracts no feedback."""
        print("Test 2: Testing process_transcript with no feedback...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                print("✓ Test 2 passed: process_transcript with no feedback works")

        # Run async test
        asyncio.run(async_test())

    def test_process_transcript_with_error(self):
        """Test 3: Process transcript that encounters an error."""
        print("Test 3: Testing process_transcript with error...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                print("✓ Test 3 passed: process_transcript error handling works")

        # Run async test
        asyncio.run(async_test())

    def test_process_transcript_with_feedback_success(self):
        """Test 4: Process transcript with successful feedback processing."""
        print("Test 4: Testing process_transcript with successful feedback processing...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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
                    import extract
                    import pipeline

                    # Create mock feedback
                    mock_feedback = extract.Feedback(
                        type="feature_request",
                        summary="Test feedback",
                        verbatim="This is test verbatim",
                        confidence=0.9,
                        transcript_id="test-transcript",
                        timestamp=datetime.now(),
                        context="Test context",
                    )

                    # Create mock match result
                    mock_match = Mock()
                    mock_match.confidence = 0.8
                    mock_match.problem_id = "problem-123"
                    mock_match.problem_title = "Test Problem"
                    mock_match.similarity_score = 0.85
                    mock_match.reasoning = "Test reasoning"

                    # Setup mocks
                    mock_extractor_instance = Mock()
                    mock_extractor_instance.extract_feedback = AsyncMock(
                        return_value=[mock_feedback]
                    )
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
                    mock_metrics_instance.add_result.assert_called_once_with(
                        mock_feedback, mock_match
                    )

                    print(
                        "✓ Test 4 passed: process_transcript with successful feedback processing works"
                    )

        # Run async test
        asyncio.run(async_test())


class TestSyncNotionProblems:
    """Test the sync_notion_problems method step by step."""

    def test_sync_notion_problems_success(self):
        """Test 5: Successful sync of Notion problems."""
        print("Test 5: Testing sync_notion_problems success...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                print("✓ Test 5 passed: sync_notion_problems success works")

        # Run async test
        asyncio.run(async_test())

    def test_sync_notion_problems_no_problems(self):
        """Test 6: Sync with no problems returned."""
        print("Test 6: Testing sync_notion_problems with no problems...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                print("✓ Test 6 passed: sync_notion_problems with no problems works")

        # Run async test
        asyncio.run(async_test())

    def test_sync_notion_problems_error(self):
        """Test 7: Sync with error handling."""
        print("Test 7: Testing sync_notion_problems error handling...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
                patch("pipeline.FeedbackExtractor"),
                patch("pipeline.NotionClient") as mock_notion,
                patch("pipeline.EmbeddingManager"),
                patch("pipeline.RAGMatcher"),
                patch("pipeline.MatchingMetrics"),
            ):
                import pipeline

                # Setup mocks
                mock_notion_instance = Mock()
                mock_notion_instance.get_all_problems = Mock(
                    side_effect=Exception("Notion API Error")
                )
                mock_notion.return_value = mock_notion_instance

                pipe = pipeline.FeedbackPipeline()

                # Test sync with error
                result = await pipe.sync_notion_problems()

                # Verify result
                assert result is False

                print("✓ Test 7 passed: sync_notion_problems error handling works")

        # Run async test
        asyncio.run(async_test())


class TestBatchProcessTranscripts:
    """Test the batch_process_transcripts method step by step."""

    def test_batch_process_transcripts_success(self):
        """Test 8: Successful batch processing of transcripts."""
        print("Test 8: Testing batch_process_transcripts success...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                print("✓ Test 8 passed: batch_process_transcripts success works")

        # Run async test
        asyncio.run(async_test())

    def test_batch_process_transcripts_empty_content(self):
        """Test 9: Batch processing with empty transcript content."""
        print("Test 9: Testing batch_process_transcripts with empty content...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                print("✓ Test 9 passed: batch_process_transcripts with empty content works")

        # Run async test
        asyncio.run(async_test())


class TestProcessFeedbacks:
    """Test the process_feedbacks method step by step."""

    def test_process_feedbacks_low_confidence_match(self):
        """Test 10: Process feedbacks with low confidence match."""
        print("Test 10: Testing process_feedbacks with low confidence match...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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
                    mock_metrics_instance.add_result.assert_called_once_with(
                        mock_feedback, mock_match
                    )

                    print("✓ Test 10 passed: process_feedbacks with low confidence match works")

        # Run async test
        asyncio.run(async_test())

    def test_process_feedbacks_no_match(self):
        """Test 11: Process feedbacks with no match found."""
        print("Test 11: Testing process_feedbacks with no match...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                    print("✓ Test 11 passed: process_feedbacks with no match works")

        # Run async test
        asyncio.run(async_test())

    def test_process_feedbacks_error_handling(self):
        """Test 12: Process feedbacks with error during processing."""
        print("Test 12: Testing process_feedbacks error handling...")

        async def async_test():
            with (
                patch.dict("sys.modules", get_mock_modules()),
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

                    print("✓ Test 12 passed: process_feedbacks error handling works")

        # Run async test
        asyncio.run(async_test())


if __name__ == "__main__":
    print("Running comprehensive Pipeline tests step by step...")

    try:
        # Test pipeline initialization
        init_tests = TestFeedbackPipelineInitialization()
        init_tests.test_pipeline_initialization()

        # Test process_transcript method
        transcript_tests = TestProcessTranscript()
        transcript_tests.test_process_transcript_no_feedback()
        transcript_tests.test_process_transcript_with_error()
        transcript_tests.test_process_transcript_with_feedback_success()

        # Test sync_notion_problems method
        sync_tests = TestSyncNotionProblems()
        sync_tests.test_sync_notion_problems_success()
        sync_tests.test_sync_notion_problems_no_problems()
        sync_tests.test_sync_notion_problems_error()

        # Test batch_process_transcripts method
        batch_tests = TestBatchProcessTranscripts()
        batch_tests.test_batch_process_transcripts_success()
        batch_tests.test_batch_process_transcripts_empty_content()

        # Test process_feedbacks method
        feedback_tests = TestProcessFeedbacks()
        feedback_tests.test_process_feedbacks_low_confidence_match()
        feedback_tests.test_process_feedbacks_no_match()
        feedback_tests.test_process_feedbacks_error_handling()

        print("\n🎉 ALL 12 PIPELINE TESTS PASSED! 🎉")
        print("✅ Pipeline initialization: 1 test")
        print("✅ process_transcript method: 3 tests")
        print("✅ sync_notion_problems method: 3 tests")
        print("✅ batch_process_transcripts method: 2 tests")
        print("✅ process_feedbacks method: 3 tests")
        print("\nPipeline functionality is being tested step by step!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
