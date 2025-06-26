"""
Basic tests for the feedback pipeline.
"""
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from config import settings
from extract import Feedback, FeedbackExtractor


class TestFeedbackExtractor:
    """Test feedback extraction functionality."""

    @pytest.fixture
    def sample_transcript(self) -> str:
        return """
        Customer said: "I really wish you had an Excel export feature.
        The PDF export just doesn't work for our needs."

        Later they mentioned: "The search is incredibly slow with large datasets.
        It takes 30-40 seconds which is frustrating."
        """

    @pytest.fixture
    def mock_llm_client(self) -> Mock:
        """Mock LLM client for testing."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = '''[
            {
                "type": "feature_request",
                "summary": "Need Excel export functionality",
                "verbatim": "I really wish you had an Excel export feature",
                "confidence": 0.9
            },
            {
                "type": "customer_pain",
                "summary": "Search is too slow with large datasets",
                "verbatim": "The search is incredibly slow with large datasets",
                "confidence": 0.85
            }
        ]'''
        mock_client.generate = AsyncMock(return_value=mock_response)
        return mock_client

    @pytest.mark.asyncio
    async def test_extract_feedback(self, sample_transcript: str, mock_llm_client: Mock) -> None:
        """Test basic feedback extraction."""
        extractor = FeedbackExtractor()
        extractor.llm_client = mock_llm_client

        feedbacks = await extractor.extract_feedback(sample_transcript, "test_transcript")

        assert len(feedbacks) == 2
        assert feedbacks[0].type == "feature_request"
        assert feedbacks[1].type == "customer_pain"
        assert "Excel" in feedbacks[0].summary
        assert "slow" in feedbacks[1].summary

    def test_feedback_validation(self) -> None:
        """Test feedback validation logic."""
        extractor = FeedbackExtractor()

        # Valid feedback
        valid_feedback = Feedback(
            type="feature_request",
            summary="Add export functionality",
            verbatim="We need export features",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert extractor._is_valid_feedback(valid_feedback)

        # Invalid type
        invalid_feedback = Feedback(
            type="invalid_type",
            summary="Add export functionality",
            verbatim="We need export features",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert not extractor._is_valid_feedback(invalid_feedback)

        # Low confidence
        low_conf_feedback = Feedback(
            type="feature_request",
            summary="Add export functionality",
            verbatim="We need export features",
            confidence=0.2,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert not extractor._is_valid_feedback(low_conf_feedback)

class TestConfiguration:
    """Test configuration handling."""

    def test_settings_loaded(self) -> None:
        """Test that settings are properly loaded."""
        assert hasattr(settings, 'llm_provider')
        assert hasattr(settings, 'confidence_threshold')
        assert settings.confidence_threshold > 0
        assert settings.confidence_threshold <= 1.0

    def test_llm_config(self) -> None:
        """Test LLM configuration."""
        from config import get_llm_config
        config = get_llm_config()
        assert isinstance(config, dict)

@pytest.mark.asyncio
async def test_pipeline_integration() -> None:
    """Integration test for the full pipeline (mocked)."""
    # This would test the full pipeline with mocked external dependencies

if __name__ == "__main__":
    pytest.main([__file__])
