"""
Complete unit tests for extract.py - Comprehensive coverage of all classes and methods.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from extract import Feedback, FeedbackExtractor, load_feedback_logs, save_feedback_logs


class TestFeedback:
    """Test the Feedback dataclass."""

    def test_feedback_creation_valid(self):
        """Test creating a valid Feedback instance."""
        feedback = Feedback(
            type="feature_request",
            summary="User wants better dashboard",
            verbatim="I really need a better dashboard for analytics",
            confidence=0.8,
            transcript_id="test_123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            context="Product feedback session",
        )

        assert feedback.type == "feature_request"
        assert feedback.summary == "User wants better dashboard"
        assert feedback.verbatim == "I really need a better dashboard for analytics"
        assert feedback.confidence == 0.8
        assert feedback.transcript_id == "test_123"
        assert feedback.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert feedback.context == "Product feedback session"

    def test_feedback_creation_minimal(self):
        """Test creating Feedback with minimal required fields."""
        feedback = Feedback(
            type="customer_pain",
            summary="App is slow",
            verbatim="The app takes forever to load",
            confidence=0.7,
            transcript_id="test_456",
            timestamp=datetime.now(),
        )

        assert feedback.type == "customer_pain"
        assert feedback.context is None

    def test_feedback_validation_invalid_type(self):
        """Test that invalid feedback type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid feedback type"):
            Feedback(
                type="invalid_type",
                summary="Test summary",
                verbatim="Test verbatim",
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now(),
            )

    def test_feedback_validation_invalid_confidence_low(self):
        """Test that confidence below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Feedback(
                type="feature_request",
                summary="Test summary",
                verbatim="Test verbatim",
                confidence=-0.1,
                transcript_id="test",
                timestamp=datetime.now(),
            )

    def test_feedback_validation_invalid_confidence_high(self):
        """Test that confidence above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Feedback(
                type="feature_request",
                summary="Test summary",
                verbatim="Test verbatim",
                confidence=1.1,
                transcript_id="test",
                timestamp=datetime.now(),
            )

    def test_feedback_validation_empty_summary(self):
        """Test that empty summary raises ValueError."""
        with pytest.raises(ValueError, match="Summary cannot be empty"):
            Feedback(
                type="feature_request",
                summary="",
                verbatim="Test verbatim",
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now(),
            )

    def test_feedback_validation_whitespace_summary(self):
        """Test that whitespace-only summary raises ValueError."""
        with pytest.raises(ValueError, match="Summary cannot be empty"):
            Feedback(
                type="feature_request",
                summary="   ",
                verbatim="Test verbatim",
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now(),
            )

    def test_feedback_validation_empty_verbatim(self):
        """Test that empty verbatim raises ValueError."""
        with pytest.raises(ValueError, match="Verbatim cannot be empty"):
            Feedback(
                type="feature_request",
                summary="Test summary",
                verbatim="",
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now(),
            )

    def test_feedback_to_dict(self):
        """Test converting Feedback to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        feedback = Feedback(
            type="feature_request",
            summary="Test summary",
            verbatim="Test verbatim",
            confidence=0.8,
            transcript_id="test_123",
            timestamp=timestamp,
            context="Test context",
        )

        result = feedback.to_dict()

        expected = {
            "type": "feature_request",
            "summary": "Test summary",
            "verbatim": "Test verbatim",
            "confidence": 0.8,
            "transcript_id": "test_123",
            "timestamp": "2024-01-01T12:00:00",
            "context": "Test context",
        }

        assert result == expected

    def test_feedback_to_dict_no_context(self):
        """Test converting Feedback to dictionary with no context."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        feedback = Feedback(
            type="customer_pain",
            summary="Test summary",
            verbatim="Test verbatim",
            confidence=0.7,
            transcript_id="test_456",
            timestamp=timestamp,
        )

        result = feedback.to_dict()

        assert result["context"] is None


class TestFeedbackExtractor:
    """Test the FeedbackExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_transcript = """
        Customer mentioned they really need better analytics dashboards.
        They said the current reporting is too slow and lacks real-time data.
        The customer expressed frustration with the mobile app loading times.
        They specifically requested push notifications for important alerts.
        """

    @patch("extract.get_llm_client")
    def test_feedback_extractor_init(self, mock_get_llm_client):
        """Test FeedbackExtractor initialization."""
        mock_llm_client = Mock()
        mock_get_llm_client.return_value = mock_llm_client

        extractor = FeedbackExtractor()

        assert extractor.llm_client == mock_llm_client
        assert extractor._min_confidence == 0.3
        assert extractor._min_summary_length == 10
        assert extractor._min_verbatim_length == 5
        assert isinstance(extractor.extraction_prompt, str)
        assert "feature_request" in extractor.extraction_prompt
        assert "customer_pain" in extractor.extraction_prompt

    @patch("extract.get_llm_client")
    def test_build_extraction_prompt(self, mock_get_llm_client):
        """Test that extraction prompt is properly built."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        prompt = extractor._build_extraction_prompt()

        # Check key elements are in the prompt
        assert "feature_request" in prompt
        assert "customer_pain" in prompt
        assert "confidence" in prompt
        assert "verbatim" in prompt
        assert "summary" in prompt

    @patch("extract.get_llm_client")
    @pytest.mark.asyncio
    async def test_extract_feedback_success(self, mock_get_llm_client):
        """Test successful feedback extraction."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """[
            {
                "type": "feature_request",
                "summary": "Customer wants better analytics dashboards",
                "verbatim": "Customer mentioned they really need better analytics dashboards",
                "confidence": 0.9
            },
            {
                "type": "customer_pain",
                "summary": "Reporting is slow and lacks real-time data",
                "verbatim": "They said the current reporting is too slow and lacks real-time data",
                "confidence": 0.8
            }
        ]"""

        mock_llm_client = AsyncMock()
        mock_llm_client.generate.return_value = mock_response
        mock_get_llm_client.return_value = mock_llm_client

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(self.sample_transcript, "test_123")

        assert len(feedbacks) == 2

        # Check first feedback
        assert feedbacks[0].type == "feature_request"
        assert feedbacks[0].summary == "Customer wants better analytics dashboards"
        assert (
            feedbacks[0].verbatim
            == "Customer mentioned they really need better analytics dashboards"
        )
        assert feedbacks[0].confidence == 0.9
        assert feedbacks[0].transcript_id == "test_123"

        # Check second feedback
        assert feedbacks[1].type == "customer_pain"
        assert feedbacks[1].summary == "Reporting is slow and lacks real-time data"
        assert feedbacks[1].confidence == 0.8

    @patch("extract.get_llm_client")
    @pytest.mark.asyncio
    async def test_extract_feedback_invalid_json(self, mock_get_llm_client):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.content = "This is not valid JSON"

        mock_llm_client = AsyncMock()
        mock_llm_client.generate.return_value = mock_response
        mock_get_llm_client.return_value = mock_llm_client

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(self.sample_transcript, "test_123")

        assert feedbacks == []

    @patch("extract.get_llm_client")
    @pytest.mark.asyncio
    async def test_extract_feedback_no_json_array(self, mock_get_llm_client):
        """Test handling of response without JSON array."""
        mock_response = Mock()
        mock_response.content = "Some text without JSON array"

        mock_llm_client = AsyncMock()
        mock_llm_client.generate.return_value = mock_response
        mock_get_llm_client.return_value = mock_llm_client

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(self.sample_transcript, "test_123")

        assert feedbacks == []

    @patch("extract.get_llm_client")
    @patch("extract.logger")
    @pytest.mark.asyncio
    async def test_extract_feedback_exception(self, mock_logger, mock_get_llm_client):
        """Test handling of exceptions during extraction."""
        mock_llm_client = AsyncMock()
        mock_llm_client.generate.side_effect = Exception("LLM error")
        mock_get_llm_client.return_value = mock_llm_client

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(self.sample_transcript, "test_123")

        assert feedbacks == []
        mock_logger.error.assert_called_once()

    @patch("extract.get_llm_client")
    def test_parse_feedback_response_valid_json(self, mock_get_llm_client):
        """Test parsing valid JSON response."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        response = """Some text before [
            {
                "type": "feature_request",
                "summary": "Test summary",
                "verbatim": "Test verbatim",
                "confidence": 0.8
            }
        ] some text after"""

        result = extractor._parse_feedback_response(response)

        assert len(result) == 1
        assert result[0]["type"] == "feature_request"
        assert result[0]["summary"] == "Test summary"

    @patch("extract.get_llm_client")
    def test_parse_feedback_response_invalid_json(self, mock_get_llm_client):
        """Test parsing invalid JSON response."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        response = "Invalid JSON content"

        result = extractor._parse_feedback_response(response)

        assert result == []

    @patch("extract.get_llm_client")
    def test_parse_feedback_response_malformed_json(self, mock_get_llm_client):
        """Test parsing malformed JSON response."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        response = '[{"type": "feature_request", "summary": "Test"'  # Missing closing

        result = extractor._parse_feedback_response(response)

        assert result == []

    @patch("extract.get_llm_client")
    def test_is_valid_feedback_valid(self, mock_get_llm_client):
        """Test validation of valid feedback."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        feedback = Feedback(
            type="feature_request",
            summary="Customer wants better dashboard functionality",
            verbatim="I really need a better dashboard for analytics",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now(),
        )

        assert extractor._is_valid_feedback(feedback) is True

    @patch("extract.get_llm_client")
    def test_is_valid_feedback_low_confidence(self, mock_get_llm_client):
        """Test validation rejects low confidence feedback."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        feedback = Feedback(
            type="feature_request",
            summary="Customer wants better dashboard functionality",
            verbatim="I really need a better dashboard",
            confidence=0.2,  # Below minimum threshold of 0.3
            transcript_id="test",
            timestamp=datetime.now(),
        )

        assert extractor._is_valid_feedback(feedback) is False

    @patch("extract.get_llm_client")
    def test_is_valid_feedback_short_content(self, mock_get_llm_client):
        """Test validation rejects content that's too short."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        feedback = Feedback(
            type="feature_request",
            summary="Short",  # Below minimum length of 10
            verbatim="Hi",  # Below minimum length of 5
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now(),
        )

        assert extractor._is_valid_feedback(feedback) is False

    @patch("extract.get_llm_client")
    def test_is_generic_feedback_generic(self, mock_get_llm_client):
        """Test detection of generic feedback."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        feedback = Feedback(
            type="feature_request",
            summary="Thanks",  # Generic phrase with minimal remaining text
            verbatim="Thanks",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now(),
        )

        assert extractor._is_generic_feedback(feedback) is True

    @patch("extract.get_llm_client")
    def test_is_generic_feedback_specific(self, mock_get_llm_client):
        """Test detection of specific, non-generic feedback."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()
        feedback = Feedback(
            type="feature_request",
            summary="Customer wants specific analytics dashboard with real-time data",
            verbatim="I need an analytics dashboard that shows real-time data",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now(),
        )

        assert extractor._is_generic_feedback(feedback) is False

    @patch("extract.get_llm_client")
    @pytest.mark.asyncio
    async def test_batch_extract_success(self, mock_get_llm_client):
        """Test successful batch extraction."""
        mock_response = Mock()
        mock_response.content = """[
            {
                "type": "feature_request",
                "summary": "Customer wants better analytics",
                "verbatim": "Need better analytics dashboard",
                "confidence": 0.8
            }
        ]"""

        mock_llm_client = AsyncMock()
        mock_llm_client.generate.return_value = mock_response
        mock_get_llm_client.return_value = mock_llm_client

        extractor = FeedbackExtractor()
        transcripts = [
            {"id": "transcript_1", "content": "Customer feedback about analytics"},
            {"id": "transcript_2", "content": "More customer feedback"},
        ]

        feedbacks = await extractor.batch_extract(transcripts)

        assert len(feedbacks) == 2  # One feedback from each transcript
        assert all(f.transcript_id in ["transcript_1", "transcript_2"] for f in feedbacks)

    @patch("extract.get_llm_client")
    @patch("extract.logger")
    @pytest.mark.asyncio
    async def test_batch_extract_empty_transcript(self, mock_logger, mock_get_llm_client):
        """Test batch extraction with empty transcript content."""
        mock_get_llm_client.return_value = AsyncMock()

        extractor = FeedbackExtractor()
        transcripts = [
            {"id": "transcript_1", "content": ""},  # Empty content
            {"id": "transcript_2", "content": "   "},  # Whitespace only
        ]

        feedbacks = await extractor.batch_extract(transcripts)

        assert feedbacks == []
        # Should log warnings about empty content
        assert mock_logger.warning.call_count >= 1

    @patch("extract.get_llm_client")
    @pytest.mark.asyncio
    async def test_batch_extract_missing_fields(self, mock_get_llm_client):
        """Test batch extraction with missing transcript fields."""
        mock_get_llm_client.return_value = AsyncMock()

        extractor = FeedbackExtractor()
        transcripts = [
            {"content": "Some content"},  # Missing id
            {"id": "transcript_2"},  # Missing content
        ]

        feedbacks = await extractor.batch_extract(transcripts)

        assert feedbacks == []


class TestFeedbackExtractorEdgeCases:
    """Test edge cases and error scenarios for FeedbackExtractor."""

    @patch("extract.get_llm_client")
    @pytest.mark.asyncio
    async def test_extract_feedback_with_invalid_items(self, mock_get_llm_client):
        """Test extraction that returns some invalid feedback items."""
        mock_response = Mock()
        mock_response.content = """[
            {
                "type": "feature_request",
                "summary": "Implement automated reporting system with real-time data visualization",
                "verbatim": "The customer requested an automated reporting system with real-time data visualization capabilities",
                "confidence": 0.8
            },
            {
                "type": "invalid_type",
                "summary": "Invalid type feedback",
                "verbatim": "This has invalid type",
                "confidence": 0.8
            },
            {
                "type": "customer_pain",
                "summary": "Low confidence feedback",
                "verbatim": "This has low confidence",
                "confidence": 0.1
            },
            {
                "type": "feature_request",
                "summary": "Short",
                "verbatim": "Hi",
                "confidence": 0.8
            }
        ]"""

        mock_llm_client = AsyncMock()
        mock_llm_client.generate.return_value = mock_response
        mock_get_llm_client.return_value = mock_llm_client

        # Create extractor AFTER setting up the mock
        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback("test transcript", "test_123")

        # Should only return the valid feedback
        assert len(feedbacks) == 1
        assert feedbacks[0].type == "feature_request"
        assert (
            feedbacks[0].summary
            == "Implement automated reporting system with real-time data visualization"
        )

    @patch("extract.get_llm_client")
    def test_is_valid_feedback_exception_handling(self, mock_get_llm_client):
        """Test that validation handles exceptions gracefully."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()

        # Create a feedback object that might cause issues during validation
        feedback = Mock()
        feedback.type = "feature_request"
        feedback.summary = Mock()
        feedback.summary.strip.side_effect = Exception("Mock exception")

        result = extractor._is_valid_feedback(feedback)

        assert result is False

    @patch("extract.get_llm_client")
    def test_parse_feedback_response_edge_cases(self, mock_get_llm_client):
        """Test parsing edge cases."""
        mock_get_llm_client.return_value = Mock()

        extractor = FeedbackExtractor()

        # Test empty response
        assert extractor._parse_feedback_response("") == []

        # Test response with no brackets
        assert extractor._parse_feedback_response("No JSON here") == []

        # Test response with only opening bracket
        assert extractor._parse_feedback_response("Some text [") == []

        # Test response with only closing bracket
        assert extractor._parse_feedback_response("Some text ]") == []

    @patch("extract.get_llm_client")
    @pytest.mark.asyncio
    async def test_extract_feedback_filters_generic(self, mock_get_llm_client):
        """Test that generic feedback is filtered out."""
        mock_response = Mock()
        mock_response.content = """[
            {
                "type": "feature_request",
                "summary": "Thanks",
                "verbatim": "Thanks",
                "confidence": 0.8
            },
            {
                "type": "customer_pain",
                "summary": "Specific issue with loading times affecting productivity",
                "verbatim": "The loading times are really affecting our productivity",
                "confidence": 0.8
            }
        ]"""

        mock_llm_client = AsyncMock()
        mock_llm_client.generate.return_value = mock_response
        mock_get_llm_client.return_value = mock_llm_client

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback("test transcript", "test_123")

        # Should only return the specific, non-generic feedback
        assert len(feedbacks) == 1
        assert feedbacks[0].type == "customer_pain"
        assert "productivity" in feedbacks[0].summary


class TestFeedbackLogging:
    """Test the feedback logging utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sample_feedbacks = [
            Feedback(
                type="feature_request",
                summary="Better analytics dashboard",
                verbatim="Need better analytics dashboard",
                confidence=0.8,
                transcript_id="test_1",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                context="Product feedback",
            ),
            Feedback(
                type="customer_pain",
                summary="App is slow",
                verbatim="The app takes forever to load",
                confidence=0.7,
                transcript_id="test_2",
                timestamp=datetime(2024, 1, 2, 14, 0, 0),
            ),
        ]

    def test_save_feedback_logs_success(self):
        """Test successful saving of feedback logs."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            result = save_feedback_logs(self.sample_feedbacks, temp_file)

            assert result is True

            # Verify file was created and contains correct data
            with open(temp_file) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["type"] == "feature_request"
            assert data[0]["summary"] == "Better analytics dashboard"
            assert data[0]["timestamp"] == "2024-01-01T12:00:00"
            assert data[1]["type"] == "customer_pain"
            assert data[1]["context"] is None

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_save_feedback_logs_empty_list(self):
        """Test saving empty feedback list."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            result = save_feedback_logs([], temp_file)

            assert result is False

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_save_feedback_logs_directory_creation(self):
        """Test that save_feedback_logs creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "subdir" / "logs" / "feedback.json"

            result = save_feedback_logs(self.sample_feedbacks, nested_path)

            assert result is True
            assert nested_path.exists()
            assert nested_path.parent.exists()

    @patch("extract.logger")
    def test_save_feedback_logs_file_error(self, mock_logger):
        """Test handling of file write errors."""
        # Try to write to a path that doesn't exist and can't be created
        invalid_path = "/invalid/path/that/cannot/be/created/feedback.json"

        result = save_feedback_logs(self.sample_feedbacks, invalid_path)

        assert result is False
        mock_logger.error.assert_called_once()

    def test_load_feedback_logs_success(self):
        """Test successful loading of feedback logs."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            # First save some feedbacks
            save_feedback_logs(self.sample_feedbacks, temp_file)

            # Then load them back
            loaded_feedbacks = load_feedback_logs(temp_file)

            assert len(loaded_feedbacks) == 2
            assert loaded_feedbacks[0].type == "feature_request"
            assert loaded_feedbacks[0].summary == "Better analytics dashboard"
            assert loaded_feedbacks[0].timestamp == datetime(2024, 1, 1, 12, 0, 0)
            assert loaded_feedbacks[1].type == "customer_pain"
            assert loaded_feedbacks[1].context is None

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_feedback_logs_file_not_found(self):
        """Test loading from non-existent file."""
        non_existent_file = "/path/that/does/not/exist/feedback.json"

        result = load_feedback_logs(non_existent_file)

        assert result == []

    def test_load_feedback_logs_invalid_json(self):
        """Test loading from file with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("This is not valid JSON")
            temp_file = f.name

        try:
            result = load_feedback_logs(temp_file)

            assert result == []

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_feedback_logs_invalid_format(self):
        """Test loading from file with invalid data format."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump({"not": "a list"}, f)
            temp_file = f.name

        try:
            result = load_feedback_logs(temp_file)

            assert result == []

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def test_load_feedback_logs_malformed_items(self):
        """Test loading with some malformed feedback items."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            # Mix valid and invalid feedback items
            data = [
                {
                    "type": "feature_request",
                    "summary": "Valid feedback",
                    "verbatim": "This is valid",
                    "confidence": 0.8,
                    "transcript_id": "test",
                    "timestamp": "2024-01-01T12:00:00",
                    "context": None,
                },
                {
                    "type": "invalid_type",  # Invalid type
                    "summary": "Invalid feedback",
                    "verbatim": "This is invalid",
                    "confidence": 0.8,
                    "transcript_id": "test",
                    "timestamp": "2024-01-01T12:00:00",
                },
                {
                    # Missing required fields
                    "type": "customer_pain",
                    "summary": "Incomplete feedback",
                },
            ]
            json.dump(data, f)
            temp_file = f.name

        try:
            result = load_feedback_logs(temp_file)

            # Should only load the valid feedback item
            assert len(result) == 1
            assert result[0].summary == "Valid feedback"

        finally:
            Path(temp_file).unlink(missing_ok=True)
