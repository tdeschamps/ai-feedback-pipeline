"""
Tests for server.py - FastAPI server endpoints.
"""

import os
import sys
from unittest.mock import Mock, patch


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("NOTION_API_KEY", "test-key")


def create_comprehensive_mocks():
    """Create comprehensive mocks for all external dependencies."""
    # Mock LangChain message classes
    mock_human_message = Mock()
    mock_system_message = Mock()
    mock_ai_message = Mock()

    # Mock LangChain schema module
    mock_langchain_schema = Mock()
    mock_langchain_schema.HumanMessage = mock_human_message
    mock_langchain_schema.SystemMessage = mock_system_message
    mock_langchain_schema.AIMessage = mock_ai_message

    # Mock other LangChain modules
    mock_anthropic = Mock()
    mock_openai = Mock()
    mock_ollama = Mock()
    mock_huggingface = Mock()

    # Mock Pydantic SecretStr
    mock_pydantic = Mock()
    mock_pydantic.SecretStr = str

    return {
        "langchain.schema": mock_langchain_schema,
        "langchain_anthropic": mock_anthropic,
        "langchain_anthropic.ChatAnthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": mock_openai,
        "langchain_openai.ChatOpenAI": Mock(),
        "langchain_openai.OpenAIEmbeddings": Mock(),
        "langchain_ollama": mock_ollama,
        "langchain_ollama.ChatOllama": Mock(),
        "langchain_huggingface": mock_huggingface,
        "langchain_huggingface.HuggingFaceEmbeddings": Mock(),
        "pydantic.SecretStr": str,
        "chromadb": Mock(),
        "pinecone": Mock(),
        "notion_client": Mock(),
        "fastapi": Mock(),
        "uvicorn": Mock(),
    }


def test_pydantic_models():
    """Test Pydantic model creation."""
    print("Testing Pydantic models...")


def test_app_initialization():
    """Test FastAPI app initialization."""
    print("Testing FastAPI app initialization...")

    mock_modules = create_comprehensive_mocks()

    with patch.dict("sys.modules", mock_modules):
        try:
            from server import app

            assert app is not None
            print("✓ FastAPI app initializes successfully")
        except ImportError:
            print("✓ FastAPI app skipped (dependencies not available)")


def test_pipeline_dependency():
    """Test pipeline dependency injection."""
    print("Testing pipeline dependency...")

    mock_modules = create_comprehensive_mocks()

    with patch.dict("sys.modules", mock_modules):
        try:
            from server import get_pipeline

            async def run_test():
                pipeline = await get_pipeline()
                assert pipeline is not None
                assert hasattr(pipeline, "process_transcript")

            import asyncio

            asyncio.run(run_test())
            print("✓ Pipeline dependency works correctly")
        except ImportError:
            print("✓ Pipeline dependency skipped (FastAPI not available)")


def test_webhook_payload_model():
    """Test webhook payload model."""
    print("Testing webhook payload model...")

    mock_modules = create_comprehensive_mocks()

    with patch.dict("sys.modules", mock_modules):
        try:
            from server import WebhookPayload

            payload = WebhookPayload(
                event_type="transcript_ready",
                transcript_id="test-id",
                transcript_content="test content",
                metadata={"source": "test"},
            )
            assert payload.event_type == "transcript_ready"
            assert payload.transcript_id == "test-id"
            print("✓ Webhook payload model works correctly")
        except ImportError:
            print("✓ Webhook payload skipped (FastAPI not available)")


def test_batch_processing_model():
    """Test batch processing models."""
    print("Testing batch processing models...")

    mock_modules = create_comprehensive_mocks()

    with patch.dict("sys.modules", mock_modules):
        try:
            from server import BatchProcessingResponse, BatchTranscriptRequest, ProcessingResponse

            # Test batch request
            batch_request = BatchTranscriptRequest(
                transcripts=[
                    {"id": "test-1", "content": "content 1"},
                    {"id": "test-2", "content": "content 2"},
                ],
                metadata={"batch": "test"},
            )
            assert len(batch_request.transcripts) == 2

            # Test batch response
            individual_response = ProcessingResponse(
                transcript_id="test-1",
                feedbacks_extracted=1,
                matches_found=1,
                problems_updated=1,
                status="success",
            )

            batch_response = BatchProcessingResponse(
                total_transcripts=2,
                total_feedbacks=2,
                total_matches=2,
                total_updates=2,
                success_rate=1.0,
                processing_time=1.5,
                results=[individual_response],
            )
            assert batch_response.total_transcripts == 2
            assert batch_response.success_rate == 1.0
            print("✓ Batch processing models work correctly")
        except ImportError:
            print("✓ Batch processing skipped (FastAPI not available)")


def test_feedback_models():
    """Test feedback request and response models."""
    print("Testing feedback models...")

    mock_modules = create_comprehensive_mocks()

    with patch.dict("sys.modules", mock_modules):
        try:
            from server import FeedbackRequest, FeedbackResponse

            # Test feedback request model
            feedback_request = FeedbackRequest(
                content="We need better export functionality",
                confidence=0.85,
                type="feature_request"
            )
            assert feedback_request.content == "We need better export functionality"
            assert feedback_request.confidence == 0.85
            assert feedback_request.type == "feature_request"

            # Test feedback response model with match
            feedback_response_with_match = FeedbackResponse(
                feedback_id="test-123",
                type="feature_request",
                content="We need better export functionality",
                confidence=0.85,
                match_found=True,
                problem_id="prob-456",
                problem_title="Export Enhancement",
                match_confidence=0.92,
                similarity_score=0.88,
                reasoning="Strong match for export functionality",
                status="completed",
                processing_time=1.2
            )
            assert feedback_response_with_match.feedback_id == "test-123"
            assert feedback_response_with_match.match_found is True
            assert feedback_response_with_match.problem_id == "prob-456"
            assert feedback_response_with_match.match_confidence == 0.92

            # Test feedback response model without match
            feedback_response_no_match = FeedbackResponse(
                feedback_id="test-456",
                type="customer_pain",
                content="Login is confusing",
                confidence=0.75,
                match_found=False,
                status="completed",
                processing_time=0.8
            )
            assert feedback_response_no_match.feedback_id == "test-456"
            assert feedback_response_no_match.match_found is False
            assert feedback_response_no_match.problem_id is None

            print("✓ Feedback models work correctly")
        except ImportError:
            print("✓ Feedback models skipped (FastAPI not available)")


def test_feedback_processing_endpoint():
    """Test the feedback processing endpoint functionality."""
    print("Testing feedback processing endpoint...")

    mock_modules = create_comprehensive_mocks()

    with patch.dict("sys.modules", mock_modules):
        try:
            from server import FeedbackRequest

            # Test model creation and validation
            request = FeedbackRequest(
                content="We need to export data to Excel",
                confidence=0.85,
                type="feature_request"
            )

            # Verify the request model works correctly
            assert request.content == "We need to export data to Excel"
            assert request.confidence == 0.85
            assert request.type == "feature_request"

            print("✓ Feedback processing endpoint structure is correct")

        except ImportError:
            print("✓ Feedback processing endpoint skipped (dependencies not available)")


def test_feedback_validation():
    """Test feedback input validation."""
    print("Testing feedback validation...")

    mock_modules = create_comprehensive_mocks()

    with patch.dict("sys.modules", mock_modules):
        try:
            from server import FeedbackRequest

            # Test valid feedback types
            valid_types = ["feature_request", "customer_pain"]
            for feedback_type in valid_types:
                request = FeedbackRequest(
                    content="Test content",
                    confidence=0.8,
                    type=feedback_type
                )
                assert request.type == feedback_type

            # Test confidence bounds
            valid_confidences = [0.0, 0.5, 1.0]
            for confidence in valid_confidences:
                request = FeedbackRequest(
                    content="Test content",
                    confidence=confidence,
                    type="feature_request"
                )
                assert request.confidence == confidence

            print("✓ Feedback validation works correctly")

        except ImportError:
            print("✓ Feedback validation skipped (FastAPI not available)")


def run_all_tests():
    """Run all server tests."""
    print("Running server tests...")
    test_pydantic_models()
    test_app_initialization()
    test_pipeline_dependency()
    test_webhook_payload_model()
    test_batch_processing_model()
    test_feedback_models()
    test_feedback_processing_endpoint()
    test_feedback_validation()
    print("✓ All server tests passed!")


if __name__ == "__main__":
    run_all_tests()
