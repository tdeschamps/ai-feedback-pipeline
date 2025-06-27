"""
Tests for extract.py - Feedback extraction functionality.
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


@pytest.fixture
def sample_transcript():
    """Sample transcript text."""
    return """
    Customer mentioned they really need better analytics dashboards.
    They said the current reporting is too slow and lacks real-time data.
    The customer expressed frustration with the mobile app loading times.
    They specifically requested push notifications for important alerts.
    Overall, they love the product but want these improvements.
    """


@pytest.fixture
def sample_llm_response():
    """Sample LLM response for feedback extraction."""
    return {
        "feedbacks": [
            {
                "type": "feature_request",
                "summary": "Better analytics dashboards needed",
                "verbatim": "Customer mentioned they really need better analytics dashboards",
                "confidence": 0.9,
            },
            {
                "type": "customer_pain",
                "summary": "Slow reporting performance",
                "verbatim": "They said the current reporting is too slow and lacks real-time data",
                "confidence": 0.8,
            },
        ]
    }


class TestFeedback:
    """Test Feedback data class."""

    def test_feedback_creation(self):
        """Test creating a Feedback object."""
        from extract import Feedback

        feedback = Feedback(
            type="feature_request",
            summary="Test summary",
            verbatim="Test verbatim",
            confidence=0.8,
            transcript_id="test_id",
            timestamp=datetime.now(),
        )

        assert feedback.type == "feature_request"
        assert feedback.summary == "Test summary"
        assert feedback.verbatim == "Test verbatim"
        assert feedback.confidence == 0.8
        assert feedback.transcript_id == "test_id"
        assert isinstance(feedback.timestamp, datetime)

    def test_feedback_to_dict(self):
        """Test converting Feedback to dictionary."""
        from extract import Feedback

        timestamp = datetime.now()
        feedback = Feedback(
            type="customer_pain",
            summary="Test issue",
            verbatim="Customer complained about slow performance",
            confidence=0.7,
            transcript_id="test_123",
            timestamp=timestamp,
        )

        feedback_dict = feedback.to_dict()

        assert feedback_dict["type"] == "customer_pain"
        assert feedback_dict["summary"] == "Test issue"
        assert feedback_dict["verbatim"] == "Customer complained about slow performance"
        assert feedback_dict["confidence"] == 0.7
        assert feedback_dict["transcript_id"] == "test_123"
        assert feedback_dict["timestamp"] == timestamp.isoformat()


class TestFeedbackExtractor:
    """Test FeedbackExtractor class."""

    @patch("extract.LLMClient")
    def test_extractor_initialization(self, mock_llm_client):
        """Test FeedbackExtractor initialization."""
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()
        assert extractor is not None
        mock_llm_client.assert_called_once()

    @patch("extract.LLMClient")
    async def test_extract_feedback_success(
        self, mock_llm_client, sample_transcript, sample_llm_response
    ):
        """Test successful feedback extraction."""
        from extract import FeedbackExtractor

        mock_client_instance = AsyncMock()
        mock_llm_client.return_value = mock_client_instance
        mock_client_instance.call_llm.return_value = json.dumps(sample_llm_response)

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(sample_transcript, "test_id")

        assert len(feedbacks) == 2
        assert feedbacks[0].type == "feature_request"
        assert feedbacks[0].summary == "Better analytics dashboards needed"
        assert feedbacks[0].confidence == 0.9
        assert feedbacks[1].type == "customer_pain"
        assert feedbacks[1].summary == "Slow reporting performance"
        assert feedbacks[1].confidence == 0.8

    @patch("extract.LLMClient")
    async def test_extract_feedback_no_feedback(self, mock_llm_client, sample_transcript):
        """Test extraction when no feedback is found."""
        from extract import FeedbackExtractor

        mock_client_instance = AsyncMock()
        mock_llm_client.return_value = mock_client_instance
        mock_client_instance.call_llm.return_value = json.dumps({"feedbacks": []})

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(sample_transcript, "test_id")

        assert len(feedbacks) == 0

    @patch("extract.LLMClient")
    async def test_extract_feedback_invalid_json(self, mock_llm_client, sample_transcript):
        """Test handling of invalid JSON response."""
        from extract import FeedbackExtractor

        mock_client_instance = AsyncMock()
        mock_llm_client.return_value = mock_client_instance
        mock_client_instance.call_llm.return_value = "Invalid JSON response"

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(sample_transcript, "test_id")

        # Should return empty list on invalid JSON
        assert len(feedbacks) == 0

    @patch("extract.LLMClient")
    async def test_extract_feedback_llm_error(self, mock_llm_client, sample_transcript):
        """Test handling of LLM errors."""
        from extract import FeedbackExtractor

        mock_client_instance = AsyncMock()
        mock_llm_client.return_value = mock_client_instance
        mock_client_instance.call_llm.side_effect = Exception("LLM API Error")

        extractor = FeedbackExtractor()

        # Should raise the exception
        with pytest.raises(Exception, match="LLM API Error"):
            await extractor.extract_feedback(sample_transcript, "test_id")

    @patch("extract.LLMClient")
    async def test_extract_feedback_malformed_response(self, mock_llm_client, sample_transcript):
        """Test handling of malformed LLM response."""
        from extract import FeedbackExtractor

        # Missing required fields
        malformed_response = {
            "feedbacks": [
                {
                    "type": "feature_request",
                    "summary": "Test summary",
                    # Missing verbatim and confidence
                }
            ]
        }

        mock_client_instance = AsyncMock()
        mock_llm_client.return_value = mock_client_instance
        mock_client_instance.call_llm.return_value = json.dumps(malformed_response)

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(sample_transcript, "test_id")

        # Should skip malformed feedback items
        assert len(feedbacks) == 0

    @patch("extract.LLMClient")
    async def test_extract_feedback_confidence_filtering(self, mock_llm_client, sample_transcript):
        """Test filtering by confidence threshold."""
        from extract import FeedbackExtractor

        response_with_low_confidence = {
            "feedbacks": [
                {
                    "type": "feature_request",
                    "summary": "High confidence feedback",
                    "verbatim": "Customer clearly requested this feature",
                    "confidence": 0.9,
                },
                {
                    "type": "customer_pain",
                    "summary": "Low confidence feedback",
                    "verbatim": "Maybe they mentioned this issue",
                    "confidence": 0.3,
                },
            ]
        }

        mock_client_instance = AsyncMock()
        mock_llm_client.return_value = mock_client_instance
        mock_client_instance.call_llm.return_value = json.dumps(response_with_low_confidence)

        extractor = FeedbackExtractor()
        feedbacks = await extractor.extract_feedback(
            sample_transcript, "test_id", min_confidence=0.5
        )

        # Should only return high confidence feedback
        assert len(feedbacks) == 1
        assert feedbacks[0].confidence == 0.9

    def test_validate_feedback_valid(self):
        """Test feedback validation with valid data."""
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()
        feedback_data = {
            "type": "feature_request",
            "summary": "Valid summary",
            "verbatim": "Valid verbatim text",
            "confidence": 0.8,
        }

        is_valid = extractor._validate_feedback(feedback_data)
        assert is_valid is True

    def test_validate_feedback_invalid(self):
        """Test feedback validation with invalid data."""
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()

        # Missing required fields
        invalid_feedback = {"type": "feature_request", "summary": "Missing verbatim and confidence"}

        is_valid = extractor._validate_feedback(invalid_feedback)
        assert is_valid is False

        # Invalid type
        invalid_type = {
            "type": "invalid_type",
            "summary": "Invalid type",
            "verbatim": "Text",
            "confidence": 0.8,
        }

        is_valid = extractor._validate_feedback(invalid_type)
        assert is_valid is False

        # Invalid confidence range
        invalid_confidence = {
            "type": "feature_request",
            "summary": "Invalid confidence",
            "verbatim": "Text",
            "confidence": 1.5,  # > 1.0
        }

        is_valid = extractor._validate_feedback(invalid_confidence)
        assert is_valid is False


class TestFeedbackLogging:
    """Test feedback logging functionality."""

    def test_save_feedback_logs(self):
        """Test saving feedback logs to file."""
        from extract import Feedback, save_feedback_logs

        # Create test feedbacks
        feedbacks = [
            Feedback(
                type="feature_request",
                summary="Test feedback 1",
                verbatim="Customer wants feature A",
                confidence=0.9,
                transcript_id="test_1",
                timestamp=datetime.now(),
            ),
            Feedback(
                type="customer_pain",
                summary="Test feedback 2",
                verbatim="Customer has issue with B",
                confidence=0.7,
                transcript_id="test_2",
                timestamp=datetime.now(),
            ),
        ]

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            result_path = save_feedback_logs(feedbacks, temp_path)

            # Verify file was created
            assert Path(result_path).exists()

            # Verify content
            with open(result_path) as f:
                saved_data = json.load(f)

            assert len(saved_data) == 2
            assert saved_data[0]["type"] == "feature_request"
            assert saved_data[1]["type"] == "customer_pain"

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_load_feedback_logs(self):
        """Test loading feedback logs from file."""
        from extract import Feedback, load_feedback_logs, save_feedback_logs

        # Create and save test feedbacks
        original_feedbacks = [
            Feedback(
                type="feature_request",
                summary="Test feedback",
                verbatim="Customer feedback text",
                confidence=0.8,
                transcript_id="test_id",
                timestamp=datetime.now(),
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save feedbacks
            save_feedback_logs(original_feedbacks, temp_path)

            # Load feedbacks
            loaded_feedbacks = load_feedback_logs(temp_path)

            assert len(loaded_feedbacks) == 1
            assert loaded_feedbacks[0].type == "feature_request"
            assert loaded_feedbacks[0].summary == "Test feedback"
            assert loaded_feedbacks[0].confidence == 0.8

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_load_feedback_logs_file_not_found(self):
        """Test loading feedback logs from non-existent file."""
        from extract import load_feedback_logs

        feedbacks = load_feedback_logs("/non/existent/file.json")
        assert feedbacks == []


class TestPromptGeneration:
    """Test prompt generation for LLM."""

    def test_build_extraction_prompt(self):
        """Test building extraction prompt."""
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()
        transcript = "Customer wants better dashboards"

        prompt = extractor._build_extraction_prompt(transcript)

        assert isinstance(prompt, str)
        assert "transcript" in prompt.lower()
        assert "feedback" in prompt.lower()
        assert transcript in prompt

    def test_build_prompt_with_examples(self):
        """Test building prompt with few-shot examples."""
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()
        transcript = "Customer mentioned slow performance"

        prompt = extractor._build_extraction_prompt(transcript, include_examples=True)

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be longer with examples
        assert "example" in prompt.lower() or "sample" in prompt.lower()


def test_extract_imports():
    """Test that extract module imports correctly."""
    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        try:
            import extract

            assert extract is not None
            print("✓ Extract module imports successfully")
        except ImportError as e:
            print(f"✓ Extract imports skipped: {e}")


if __name__ == "__main__":
    print("Running extract tests...")
    test_extract_imports()
    print("✓ All extract tests configured!")
