"""
Unit tests specifically for feedback validation and extraction logic.
"""
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from extract import Feedback, FeedbackExtractor
from llm_client import LLMResponse


class TestFeedbackValidation:
    """Test the Feedback dataclass validation."""

    def test_valid_feedback_creation(self) -> None:
        """Test creating a valid feedback object."""
        feedback = Feedback(
            type="feature_request",
            summary="User wants Excel export functionality",
            verbatim="I really wish you had an Excel export feature",
            confidence=0.85,
            transcript_id="transcript_123",
            timestamp=datetime.now()
        )

        assert feedback.type == "feature_request"
        assert feedback.confidence == 0.85
        assert "Excel export" in feedback.summary

    def test_invalid_feedback_type_raises_error(self) -> None:
        """Test that invalid feedback type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid feedback type"):
            Feedback(
                type="invalid_type",
                summary="Some summary",
                verbatim="Some verbatim",
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now()
            )

    def test_invalid_confidence_raises_error(self) -> None:
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between"):
            Feedback(
                type="feature_request",
                summary="Some summary",
                verbatim="Some verbatim",
                confidence=1.5,  # Invalid: > 1.0
                transcript_id="test",
                timestamp=datetime.now()
            )

        with pytest.raises(ValueError, match="Confidence must be between"):
            Feedback(
                type="feature_request",
                summary="Some summary",
                verbatim="Some verbatim",
                confidence=-0.1,  # Invalid: < 0.0
                transcript_id="test",
                timestamp=datetime.now()
            )

    def test_empty_summary_raises_error(self) -> None:
        """Test that empty summary raises ValueError."""
        with pytest.raises(ValueError, match="Summary cannot be empty"):
            Feedback(
                type="feature_request",
                summary="   ",  # Only whitespace
                verbatim="Some verbatim",
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now()
            )

    def test_empty_verbatim_raises_error(self) -> None:
        """Test that empty verbatim raises ValueError."""
        with pytest.raises(ValueError, match="Verbatim cannot be empty"):
            Feedback(
                type="customer_pain",
                summary="Some summary",
                verbatim="",  # Empty
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now()
            )


class TestFeedbackExtractorValidation:
    """Test the FeedbackExtractor validation methods."""

    @pytest.fixture
    def extractor(self) -> FeedbackExtractor:
        """Create a FeedbackExtractor instance for testing."""
        with pytest.MonkeyPatch().context() as m:
            # Mock the LLM client to avoid initialization issues
            mock_client = Mock()
            m.setattr("extract.get_llm_client", lambda: mock_client)
            return FeedbackExtractor()

    def test_valid_feedback_passes_validation(self, extractor: FeedbackExtractor) -> None:
        """Test that valid feedback passes validation."""
        feedback = Feedback(
            type="feature_request",
            summary="Users need better export functionality for data analysis",
            verbatim="I really wish you had an Excel export feature because the current PDF export doesn't work for our data analysis needs",
            confidence=0.85,
            transcript_id="transcript_123",
            timestamp=datetime.now()
        )

        assert extractor._is_valid_feedback(feedback) is True

    def test_low_confidence_fails_validation(self, extractor: FeedbackExtractor) -> None:
        """Test that low confidence feedback fails validation."""
        feedback = Feedback(
            type="feature_request",
            summary="Users need better export functionality",
            verbatim="Maybe some export stuff",
            confidence=0.2,  # Below threshold
            transcript_id="test",
            timestamp=datetime.now()
        )

        assert extractor._is_valid_feedback(feedback) is False

    def test_short_content_fails_validation(self, extractor: FeedbackExtractor) -> None:
        """Test that too-short content fails validation."""
        feedback = Feedback(
            type="customer_pain",
            summary="Bad",  # Too short
            verbatim="No",  # Too short
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )

        assert extractor._is_valid_feedback(feedback) is False

    def test_generic_feedback_fails_validation(self, extractor: FeedbackExtractor) -> None:
        """Test that generic feedback fails validation."""
        feedback = Feedback(
            type="feature_request",
            summary="Good job team, thanks for everything",  # Generic
            verbatim="Thanks, good job",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )

        assert extractor._is_valid_feedback(feedback) is False

    def test_is_generic_feedback_detection(self, extractor: FeedbackExtractor) -> None:
        """Test the generic feedback detection logic."""
        # Generic feedback
        generic_feedback = Feedback(
            type="feature_request",
            summary="Great work",
            verbatim="Great job everyone",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )

        assert extractor._is_generic_feedback(generic_feedback) is True

        # Specific feedback
        specific_feedback = Feedback(
            type="feature_request",
            summary="Need real-time data synchronization capabilities",
            verbatim="We really need the ability to sync data in real-time",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )

        assert extractor._is_generic_feedback(specific_feedback) is False


class TestFeedbackExtractionLogic:
    """Test the feedback extraction and parsing logic."""

    @pytest.fixture
    def extractor_with_mock_client(self) -> FeedbackExtractor:
        """Create extractor with mocked LLM client."""
        mock_client = Mock()
        extractor = FeedbackExtractor()
        extractor.llm_client = mock_client
        return extractor

    @pytest.mark.asyncio
    async def test_successful_feedback_extraction(self, extractor_with_mock_client: FeedbackExtractor) -> None:
        """Test successful feedback extraction from transcript."""
        # Mock LLM response
        mock_response = LLMResponse(
            content='''[
                {
                    "type": "feature_request",
                    "summary": "Need Excel export functionality for better data analysis",
                    "verbatim": "I really wish you had an Excel export feature",
                    "confidence": 0.9
                },
                {
                    "type": "customer_pain",
                    "summary": "Search performance is too slow with large datasets",
                    "verbatim": "The search is incredibly slow with large datasets",
                    "confidence": 0.85
                }
            ]''',
            model="test-model",
            provider="test"
        )

        extractor_with_mock_client.llm_client.generate = AsyncMock(return_value=mock_response)

        transcript = """
        Customer: I really wish you had an Excel export feature.
        Customer: The search is incredibly slow with large datasets.
        """

        feedbacks = await extractor_with_mock_client.extract_feedback(transcript, "test_transcript")

        assert len(feedbacks) == 2
        assert feedbacks[0].type == "feature_request"
        assert feedbacks[1].type == "customer_pain"
        assert "Excel export" in feedbacks[0].summary
        assert "slow" in feedbacks[1].summary

    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(self, extractor_with_mock_client: FeedbackExtractor) -> None:
        """Test handling of invalid JSON response from LLM."""
        mock_response = LLMResponse(
            content="This is not valid JSON at all",
            model="test-model",
            provider="test"
        )

        extractor_with_mock_client.llm_client.generate = AsyncMock(return_value=mock_response)

        feedbacks = await extractor_with_mock_client.extract_feedback("Some transcript", "test_id")

        assert len(feedbacks) == 0

    @pytest.mark.asyncio
    async def test_malformed_json_array_handling(self, extractor_with_mock_client: FeedbackExtractor) -> None:
        """Test handling of malformed JSON array."""
        mock_response = LLMResponse(
            content='[{"type": "feature_request", "summary": "incomplete...',  # Malformed JSON
            model="test-model",
            provider="test"
        )

        extractor_with_mock_client.llm_client.generate = AsyncMock(return_value=mock_response)

        feedbacks = await extractor_with_mock_client.extract_feedback("Some transcript", "test_id")

        assert len(feedbacks) == 0

    @pytest.mark.asyncio
    async def test_empty_transcript_handling(self, extractor_with_mock_client: FeedbackExtractor) -> None:
        """Test handling of empty transcript."""
        feedbacks = await extractor_with_mock_client.extract_feedback("", "test_id")

        # Should still call LLM but likely return no feedbacks
        assert isinstance(feedbacks, list)

    def test_parse_feedback_response_with_nested_json(self, extractor_with_mock_client: FeedbackExtractor) -> None:
        """Test parsing response with JSON embedded in text."""
        response_with_text = '''
        Here are the extracted feedbacks:

        [
            {
                "type": "feature_request",
                "summary": "Need better export options",
                "verbatim": "We need Excel export",
                "confidence": 0.8
            }
        ]

        That's all I found.
        '''

        result = extractor_with_mock_client._parse_feedback_response(response_with_text)

        assert len(result) == 1
        assert result[0]["type"] == "feature_request"
        assert result[0]["summary"] == "Need better export options"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
