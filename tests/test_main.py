"""
Comprehensive pytest tests for main.py CLI application
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("NOTION_API_KEY", "test-key")


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
        "chromadb": Mock(),
        "chromadb.config": Mock(),
        "openai": Mock(),
        "anthropic": Mock(),
        "click": Mock(),
    }


class TestCLIGroup:
    """Test the main CLI group and setup."""

    def test_cli_import(self, mock_modules):
        """Test CLI module imports successfully."""
        with patch.dict("sys.modules", mock_modules):
            import main

            assert hasattr(main, "cli"), "CLI group not found"

    def test_cli_setup_with_click_mocking(self, mock_modules):
        """Test CLI setup with proper Click mocking."""
        with patch.dict("sys.modules", mock_modules):
            # Mock click more specifically
            mock_click = Mock()
            mock_context = Mock()
            mock_context.ensure_object.return_value = None
            mock_context.obj = {}

            with patch("main.click", mock_click), patch("main.setup_logging"):
                import main

                # Test that the CLI function exists
                assert hasattr(main, "cli"), "CLI function not found"
                assert hasattr(main, "process_transcript"), "process_transcript command not found"
                assert hasattr(main, "batch_process"), "batch_process command not found"
                assert hasattr(main, "sync_problems"), "sync_problems command not found"
                assert hasattr(main, "show_feedbacks"), "show_feedbacks command not found"
                assert hasattr(main, "status"), "status command not found"


class TestProcessTranscriptCommand:
    """Test the process_transcript CLI command."""

    @pytest.mark.asyncio
    async def test_process_transcript_async_wrapper(self, mock_modules):
        """Test async transcript processing wrapper."""
        with patch.dict("sys.modules", mock_modules):
            with patch("main.FeedbackPipeline") as mock_pipeline_class:
                mock_pipeline = AsyncMock()
                mock_pipeline.process_transcript.return_value = {
                    "status": "completed",
                    "feedbacks_extracted": 2,
                    "matches_found": 1,
                    "problems_updated": 1,
                }
                mock_pipeline_class.return_value = mock_pipeline

                import main

                # Test the async wrapper
                result = await main._process_transcript_async("test content", "test-id")

                assert result["status"] == "completed"
                assert result["feedbacks_extracted"] == 2
                mock_pipeline.process_transcript.assert_called_once_with("test content", "test-id")

    def test_display_processing_result_success(self, mock_modules):
        """Test display of successful processing results."""
        with patch.dict("sys.modules", mock_modules), patch("main.click") as mock_click:
            import main

            result = {
                "status": "completed",
                "feedbacks_extracted": 3,
                "matches_found": 2,
                "problems_updated": 1,
            }

            main._display_processing_result(result, "test-transcript")

            # Verify click.echo was called multiple times
            assert mock_click.echo.called, "click.echo should have been called"
            call_count = mock_click.echo.call_count
            assert call_count >= 5, f"Expected multiple echo calls, got {call_count}"

    def test_display_processing_result_error(self, mock_modules):
        """Test display of error processing results."""
        with patch.dict("sys.modules", mock_modules), patch("main.click") as mock_click:
            import main

            result = {"status": "error", "error": "Test error message"}

            main._display_processing_result(result, "test-transcript")

            # Verify error display
            assert mock_click.echo.called, "click.echo should have been called for error"

    def test_display_processing_result_warnings(self, mock_modules):
        """Test display of processing results with warnings."""
        with patch.dict("sys.modules", mock_modules), patch("main.click") as mock_click:
            import main

            # Test no feedbacks extracted warning
            result_no_feedbacks = {
                "status": "completed",
                "feedbacks_extracted": 0,
                "matches_found": 0,
                "problems_updated": 0,
            }

            main._display_processing_result(result_no_feedbacks, "test-transcript")

            # Test no matches found warning
            result_no_matches = {
                "status": "completed",
                "feedbacks_extracted": 2,
                "matches_found": 0,
                "problems_updated": 0,
            }

            main._display_processing_result(result_no_matches, "test-transcript")

            assert mock_click.echo.called, "click.echo should have been called for warnings"


class TestFileOperations:
    """Test file operations and save functions."""

    def test_save_results_success(self, mock_modules):
        """Test save processing results to file."""
        with patch.dict("sys.modules", mock_modules):
            import main

            # Create a temporary file for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "test_results.json"

                result = {
                    "status": "completed",
                    "feedbacks_extracted": 2,
                    "matches_found": 1,
                    "timestamp": "2023-01-01T00:00:00",
                }

                main._save_results(result, output_path)

                # Verify file was created and contains expected content
                assert output_path.exists(), "Results file should have been created"

                with open(output_path) as f:
                    saved_data = json.load(f)

                assert saved_data["status"] == "completed"
                assert saved_data["feedbacks_extracted"] == 2

    def test_save_results_directory_creation(self, mock_modules):
        """Test save results with directory creation."""
        with patch.dict("sys.modules", mock_modules):
            import main

            # Create a temporary directory and test nested path creation
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "nested" / "directory" / "results.json"

                result = {"status": "test"}

                main._save_results(result, output_path)

                # Verify nested directories were created
                assert output_path.exists(), "Nested directory and file should have been created"
                assert output_path.parent.exists(), "Parent directory should exist"


class TestBatchProcessCommand:
    """Test the batch_process CLI command."""

    @pytest.mark.asyncio
    async def test_batch_process_async_wrapper(self, mock_modules):
        """Test async batch processing wrapper."""
        with patch.dict("sys.modules", mock_modules):
            with patch("main.FeedbackPipeline") as mock_pipeline_class:
                mock_pipeline = AsyncMock()
                mock_pipeline.batch_process_transcripts.return_value = {
                    "total_transcripts": 2,
                    "total_feedbacks": 4,
                    "total_matches": 2,
                    "total_updates": 1,
                    "success_rate": 0.5,
                    "metrics_file": "test_metrics.json",
                }
                mock_pipeline_class.return_value = mock_pipeline

                import main

                transcripts = [
                    {"id": "t1", "content": "content1"},
                    {"id": "t2", "content": "content2"},
                ]

                result = await main._batch_process_async(transcripts)

                assert result["total_transcripts"] == 2
                assert result["total_feedbacks"] == 4
                mock_pipeline.batch_process_transcripts.assert_called_once_with(transcripts)


class TestSyncProblemsCommand:
    """Test the sync_problems CLI command."""

    @pytest.mark.asyncio
    async def test_sync_problems_async_wrapper(self, mock_modules):
        """Test async sync problems wrapper."""
        with patch.dict("sys.modules", mock_modules):
            with patch("main.FeedbackPipeline") as mock_pipeline_class:
                mock_pipeline = AsyncMock()
                mock_pipeline.sync_notion_problems.return_value = True
                mock_pipeline_class.return_value = mock_pipeline

                import main

                result = await main._sync_problems_async()

                assert result is True
                mock_pipeline.sync_notion_problems.assert_called_once()


class TestStatusCommand:
    """Test the status CLI command."""

    def test_status_command_structure(self, mock_modules):
        """Test status command structure and settings access."""
        with patch.dict("sys.modules", mock_modules):
            import main

            # Test status command exists
            assert hasattr(main, "status"), "Status command should exist"

            # Mock the necessary dependencies more simply
            with (
                patch("main.settings") as mock_settings,
                patch("main.click"),
                patch("main.Path") as mock_path,
            ):
                # Mock settings attributes
                mock_settings.llm_provider = "openai"
                mock_settings.llm_model = "gpt-4"
                mock_settings.embedding_model = "text-embedding-ada-002"
                mock_settings.vector_store = "chroma"
                mock_settings.notion_database_id = "test-db-id"
                mock_settings.confidence_threshold = 0.8
                mock_settings.rerank_enabled = True

                # Mock Path for data directory
                mock_data_dir = Mock()
                mock_data_dir.exists.return_value = False
                mock_path.return_value = mock_data_dir

                # Call status function
                main.status()


class TestProcessFeedbackCommand:
    """Test the process_feedback CLI command."""

    @pytest.mark.asyncio
    async def test_process_feedback_async_wrapper_feature_request(self, mock_modules):
        """Test async feedback processing wrapper for feature request."""
        with patch.dict("sys.modules", mock_modules):
            with (
                patch("main.FeedbackPipeline") as mock_pipeline_class,
                patch("main.Feedback") as mock_feedback_class,
                patch("main.datetime") as mock_datetime,
            ):
                # Mock pipeline
                mock_pipeline = AsyncMock()
                mock_match = Mock()
                mock_match.problem_id = "problem-123"
                mock_match.problem_title = "Test Problem"
                mock_match.confidence = 0.85
                mock_match.similarity_score = 0.92
                mock_match.reasoning = "Test reasoning"

                mock_feedback = Mock()
                mock_feedback.type = "feature_request"
                mock_feedback.summary = "Test feature request"
                mock_feedback.verbatim = "We need this feature"
                mock_feedback.confidence = 0.8
                mock_feedback.transcript_id = "test-id"
                mock_feedback.context = "test context"

                mock_pipeline.process_feedbacks.return_value = [(mock_feedback, mock_match)]
                mock_pipeline_class.return_value = mock_pipeline

                # Mock Feedback creation
                mock_feedback_class.return_value = mock_feedback

                # Mock datetime
                mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)

                import main

                # Test the async wrapper
                result = await main._process_feedback_async(
                    "feature_request",
                    "Test feature request",
                    "We need this feature",
                    0.8,
                    "test-id",
                    "test context",
                )

                # Verify result structure
                assert result["status"] == "completed"
                assert result["feedback"]["type"] == "feature_request"
                assert result["feedback"]["summary"] == "Test feature request"
                assert result["feedback"]["verbatim"] == "We need this feature"
                assert result["feedback"]["confidence"] == 0.8
                assert result["feedback"]["transcript_id"] == "test-id"
                assert result["feedback"]["context"] == "test context"

                assert result["match"]["problem_id"] == "problem-123"
                assert result["match"]["problem_title"] == "Test Problem"
                assert result["match"]["confidence"] == 0.85
                assert result["match"]["similarity_score"] == 0.92
                assert result["match"]["reasoning"] == "Test reasoning"

                # Verify calls
                mock_pipeline.process_feedbacks.assert_called_once()
                call_args = mock_pipeline.process_feedbacks.call_args[0][0]
                assert len(call_args) == 1, "Should process exactly one feedback"

    @pytest.mark.asyncio
    async def test_process_feedback_async_wrapper_customer_pain(self, mock_modules):
        """Test async feedback processing wrapper for customer pain."""
        with patch.dict("sys.modules", mock_modules):
            with (
                patch("main.FeedbackPipeline") as mock_pipeline_class,
                patch("main.Feedback") as mock_feedback_class,
                patch("main.datetime") as mock_datetime,
            ):
                # Mock pipeline
                mock_pipeline = AsyncMock()
                mock_feedback = Mock()
                mock_feedback.type = "customer_pain"
                mock_feedback.summary = "Test customer pain"
                mock_feedback.verbatim = "This is frustrating"
                mock_feedback.confidence = 0.9
                mock_feedback.transcript_id = "test-pain-id"
                mock_feedback.context = None

                # Test case with no match
                mock_pipeline.process_feedbacks.return_value = [(mock_feedback, None)]
                mock_pipeline_class.return_value = mock_pipeline

                # Mock Feedback creation
                mock_feedback_class.return_value = mock_feedback

                # Mock datetime
                mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)

                import main

                # Test the async wrapper
                result = await main._process_feedback_async(
                    "customer_pain",
                    "Test customer pain",
                    "This is frustrating",
                    0.9,
                    "test-pain-id",
                    None,
                )

                # Verify result structure
                assert result["status"] == "completed"
                assert result["feedback"]["type"] == "customer_pain"
                assert result["feedback"]["summary"] == "Test customer pain"
                assert result["feedback"]["verbatim"] == "This is frustrating"
                assert result["feedback"]["confidence"] == 0.9
                assert result["feedback"]["transcript_id"] == "test-pain-id"
                assert result["feedback"]["context"] is None

                assert result["match"] is None

    @pytest.mark.asyncio
    async def test_process_feedback_async_wrapper_error_handling(self, mock_modules):
        """Test async feedback processing error handling."""
        with patch.dict("sys.modules", mock_modules):
            with patch("main.FeedbackPipeline") as mock_pipeline_class, patch("main.Feedback"):
                # Mock pipeline to raise an exception
                mock_pipeline = AsyncMock()
                mock_pipeline.process_feedbacks.side_effect = Exception("Test pipeline error")
                mock_pipeline_class.return_value = mock_pipeline

                import main

                # Test that exception is properly re-raised
                with pytest.raises(Exception) as exc_info:
                    await main._process_feedback_async(
                        "feature_request",
                        "Test summary",
                        "Test verbatim",
                        0.8,
                        "test-id",
                        None,
                    )

                assert (
                    "Test pipeline error" in str(exc_info.value)
                    or "pipeline processing failed" in str(exc_info.value).lower()
                )

    def test_display_feedback_processing_result_success_with_match(self, mock_modules):
        """Test display of successful feedback processing results with match."""
        with patch.dict("sys.modules", mock_modules):
            with patch("main.click") as mock_click:
                import main

                result = {
                    "status": "completed",
                    "feedback": {
                        "type": "feature_request",
                        "summary": "Test feature request summary",
                        "verbatim": "We really need this feature for our workflow",
                        "confidence": 0.85,
                        "transcript_id": "test-123",
                        "context": "Customer interview",
                    },
                    "match": {
                        "problem_id": "prob-456",
                        "problem_title": "Workflow Improvement",
                        "confidence": 0.92,
                        "similarity_score": 0.88,
                        "reasoning": "Strong semantic similarity detected",
                    },
                }

                main._display_feedback_processing_result(result, "test-123")

                # Verify click.echo was called multiple times for all the information
                assert mock_click.echo.called, "click.echo should have been called"
                call_count = mock_click.echo.call_count
                assert call_count >= 8, (
                    f"Expected multiple echo calls for detailed info, got {call_count}"
                )

    def test_display_feedback_processing_result_success_no_match(self, mock_modules):
        """Test display of successful feedback processing results without match."""
        with patch.dict("sys.modules", mock_modules), patch("main.click") as mock_click:
            import main

            result = {
                "status": "completed",
                "feedback": {
                    "type": "customer_pain",
                    "summary": "Test customer pain summary",
                    "verbatim": "This process is very confusing",
                    "confidence": 0.75,
                    "transcript_id": "test-456",
                    "context": None,
                },
                "match": None,
            }

            main._display_feedback_processing_result(result, "test-456")

            # Verify click.echo was called
            assert mock_click.echo.called, "click.echo should have been called"
            call_count = mock_click.echo.call_count
            assert call_count >= 5, f"Expected multiple echo calls, got {call_count}"

    def test_display_feedback_processing_result_error(self, mock_modules):
        """Test display of error feedback processing results."""
        with patch.dict("sys.modules", mock_modules), patch("main.click") as mock_click:
            import main

            result = {"status": "error", "error": "Failed to process feedback"}

            main._display_feedback_processing_result(result, "test-error")

            # Verify error display
            assert mock_click.echo.called, "click.echo should have been called for error"

    def test_process_feedback_command_validation(self, mock_modules):
        """Test process feedback command validation logic."""
        with patch.dict("sys.modules", mock_modules):
            # Test confidence validation
            with patch("main.sys") as mock_sys:
                import main

                # Mock click context and command
                mock_sys.exit = Mock()

                # Test invalid confidence - too high
                main.process_feedback(
                    feedback_type="feature_request",
                    summary="Test summary",
                    verbatim="Test verbatim",
                    confidence=1.5,  # Invalid
                    transcript_id="test",
                    context=None,
                    output=None,
                )

                # Should call sys.exit(1) for invalid confidence
                mock_sys.exit.assert_called_with(1)

    def test_process_feedback_command_structure(self, mock_modules):
        """Test process feedback command exists and has correct structure."""
        with patch.dict("sys.modules", mock_modules):
            import main

            # Test that process_feedback command exists
            assert hasattr(main, "process_feedback"), "process_feedback command should exist"
            assert hasattr(main, "_process_feedback_async"), (
                "_process_feedback_async helper should exist"
            )
            assert hasattr(main, "_display_feedback_processing_result"), (
                "_display_feedback_processing_result helper should exist"
            )
