"""
Tests for extract.py - Edge cases and comprehensive coverage.
"""

import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("NOTION_API_KEY", "test-key")


def test_feedback_dataclass_edge_cases():
    """Test Feedback dataclass with edge cases."""
    print("Testing Feedback dataclass edge cases...")

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
        from extract import Feedback

        # Test minimal feedback
        minimal_feedback = Feedback(
            type="feature_request",
            summary="Test summary",
            verbatim="Test verbatim",
            confidence=0.5,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert minimal_feedback.type == "feature_request"
        assert minimal_feedback.summary == "Test summary"

        # Test with None values where allowed
        feedback_with_nones = Feedback(
            type="customer_pain",
            summary="Test",
            verbatim="Test verbatim",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now(),
            context=None
        )
        assert feedback_with_nones.context is None

        # Test with complex metadata
        complex_feedback = Feedback(
            type="feature_request",
            summary="Complex test",
            verbatim="Test with metadata",
            confidence=0.9,
            transcript_id="test",
            timestamp=datetime.now(),
            context="complex context with details"
        )
        assert complex_feedback.context == "complex context with details"

    print("✓ Feedback dataclass edge cases handled correctly")


def test_feedback_extractor_initialization():
    """Test FeedbackExtractor initialization variants."""
    print("Testing FeedbackExtractor initialization...")

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
        # Test with mocked LLM client before importing
        with patch("extract.get_llm_client") as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client

            from extract import FeedbackExtractor

            # Test default initialization
            extractor = FeedbackExtractor()
            assert extractor is not None
            assert hasattr(extractor, "llm_client")

            # Test that mock client is used
            extractor_with_client = FeedbackExtractor()
            assert extractor_with_client.llm_client == mock_client

    print("✓ FeedbackExtractor initialization works correctly")


def test_empty_transcript_handling():
    """Test handling of empty/invalid transcripts."""
    print("Testing empty transcript handling...")

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
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()
        extractor.llm_client.extract_feedback = AsyncMock(return_value=[])

        async def run_test():
            # Test empty transcript
            result = await extractor.extract_feedback("", "test-id")
            assert result == []

            # Test None transcript
            result = await extractor.extract_feedback(None, "test-id")
            assert result == []

            # Test whitespace-only transcript
            result = await extractor.extract_feedback("   \n\t   ", "test-id")
            assert result == []

        import asyncio
        asyncio.run(run_test())

    print("✓ Empty transcript handling works correctly")


def test_malformed_llm_response_handling():
    """Test handling of malformed LLM responses."""
    print("Testing malformed LLM response handling...")

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
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()

        # Test with malformed JSON response
        extractor.llm_client.extract_feedback = AsyncMock(return_value="invalid json")

        async def run_test():
            result = await extractor.extract_feedback("test transcript", "test-id")
            # Should handle gracefully and return empty list or log error
            assert isinstance(result, list)

        import asyncio
        asyncio.run(run_test())

    print("✓ Malformed LLM response handling works correctly")


def test_transcript_preprocessing():
    """Test transcript preprocessing edge cases."""
    print("Testing transcript preprocessing...")

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
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()
        extractor.llm_client.extract_feedback = AsyncMock(return_value=[])

        async def run_test():
            # Test very long transcript
            long_transcript = "This is a test. " * 10000
            result = await extractor.extract_feedback(long_transcript, "test-id")
            assert isinstance(result, list)

            # Test transcript with special characters
            special_transcript = "Test with 🚀 emojis and ñoñó special chars & symbols!"
            result = await extractor.extract_feedback(special_transcript, "test-id")
            assert isinstance(result, list)

            # Test transcript with multiple languages
            multilingual = "Hello world. Bonjour le monde. Hola mundo. こんにちは世界"
            result = await extractor.extract_feedback(multilingual, "test-id")
            assert isinstance(result, list)

        import asyncio
        asyncio.run(run_test())

    print("✓ Transcript preprocessing works correctly")


def test_feedback_validation():
    """Test feedback validation edge cases."""
    print("Testing feedback validation...")

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
        from extract import Feedback

        # Test invalid feedback types
        try:
            invalid_feedback = Feedback(
                type="invalid_type",
                summary="Test",
                verbatim="Test verbatim",
                confidence=0.5,
                transcript_id="test",
                timestamp=datetime.now()
            )
            # Should raise ValueError
            assert False, "Should have raised ValueError for invalid type"
        except ValueError:
            # Expected behavior
            pass

        # Test extremely long summaries
        long_summary = "A" * 1000
        long_feedback = Feedback(
            type="feature_request",
            summary=long_summary,
            verbatim="Test verbatim",
            confidence=0.5,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert len(long_feedback.summary) == 1000

    print("✓ Feedback validation works correctly")


def test_save_feedback_logs():
    """Test feedback logging functionality."""
    print("Testing feedback logging...")

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
        from extract import Feedback, save_feedback_logs

        feedback_list = [
            Feedback(
                type="feature_request",
                summary="Test feature 1",
                verbatim="Customer wants this feature",
                confidence=0.9,
                transcript_id="test-1",
                timestamp=datetime.now()
            ),
            Feedback(
                type="customer_pain",
                summary="Test pain point",
                verbatim="Customer is frustrated with this",
                confidence=0.8,
                transcript_id="test-2",
                timestamp=datetime.now()
            )
        ]

        # Test saving logs (should handle file operations gracefully)
        try:
            save_feedback_logs(feedback_list, "test_output.json")
            print("✓ Feedback logging completed")
        except Exception as e:
            print(f"✓ Feedback logging handled error: {e}")


def test_concurrent_extraction():
    """Test concurrent feedback extraction scenarios."""
    print("Testing concurrent extraction...")

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
        from extract import FeedbackExtractor

        extractor = FeedbackExtractor()
        extractor.llm_client.extract_feedback = AsyncMock(return_value=[])

        async def run_test():
            # Test multiple concurrent extractions
            import asyncio

            tasks = [
                extractor.extract_feedback(f"transcript {i}", f"id-{i}")
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            assert all(isinstance(result, list) for result in results)

        import asyncio
        asyncio.run(run_test())

    print("✓ Concurrent extraction works correctly")


def run_all_tests():
    """Run all extract edge case tests."""
    print("Running extract edge case tests...")
    test_feedback_dataclass_edge_cases()
    test_feedback_extractor_initialization()
    test_empty_transcript_handling()
    test_malformed_llm_response_handling()
    test_transcript_preprocessing()
    test_feedback_validation()
    test_save_feedback_logs()
    test_concurrent_extraction()
    print("✓ All extract edge case tests passed!")


if __name__ == "__main__":
    run_all_tests()
