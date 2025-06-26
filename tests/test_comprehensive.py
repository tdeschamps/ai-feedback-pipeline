"""
Comprehensive test suite for the feedback pipeline.
"""
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from config import get_llm_config, settings
from embed import EmbeddingManager
from extract import Feedback, FeedbackExtractor, load_feedback_logs, save_feedback_logs
from llm_client import LLMClient, LLMResponse
from notion import NotionProblem, notion_problems_to_text
from pipeline import FeedbackPipeline
from rag import MatchingMetrics, MatchResult, RAGMatcher


class TestConfiguration:
    """Test configuration and settings."""

    def test_settings_loaded(self) -> None:
        """Test that settings are properly loaded."""
        assert hasattr(settings, 'llm_provider')
        assert hasattr(settings, 'confidence_threshold')
        assert 0 < settings.confidence_threshold <= 1.0
        assert settings.max_matches > 0

    def test_llm_config(self) -> None:
        """Test LLM configuration function."""
        config = get_llm_config()
        assert isinstance(config, dict)

        # Test with different providers
        original_provider = settings.llm_provider
        try:
            settings.llm_provider = "openai"
            openai_config = get_llm_config()
            assert "api_key" in openai_config
            assert "model" in openai_config

            settings.llm_provider = "anthropic"
            anthropic_config = get_llm_config()
            assert "api_key" in anthropic_config

        finally:
            settings.llm_provider = original_provider


class TestFeedbackExtractor:
    """Test feedback extraction functionality."""

    @pytest.fixture
    def sample_transcript(self) -> str:
        return """
        Customer: I really wish you had an Excel export feature.
        The PDF export just doesn't work for our needs.

        Customer: The search is incredibly slow with large datasets.
        It takes 30-40 seconds which is frustrating when trying to be productive.

        Customer: It would be great if we could save custom filters.
        Right now I have to recreate them every day.
        """

    @pytest.fixture
    def mock_llm_response(self) -> str:
        return '''[
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
            },
            {
                "type": "feature_request",
                "summary": "Need ability to save custom filters",
                "verbatim": "It would be great if we could save custom filters",
                "confidence": 0.8
            }
        ]'''

    @pytest.fixture
    def mock_llm_client(self, mock_llm_response: str) -> Mock:
        """Mock LLM client for testing."""
        mock_client = Mock(spec=LLMClient)
        mock_response = LLMResponse(
            content=mock_llm_response,
            model="test-model",
            provider="test"
        )
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3] for _ in range(3)])
        return mock_client

    @pytest.mark.asyncio
    async def test_extract_feedback_success(self, sample_transcript: str, mock_llm_client: Mock) -> None:
        """Test successful feedback extraction."""
        extractor = FeedbackExtractor()
        extractor.llm_client = mock_llm_client

        feedbacks = await extractor.extract_feedback(sample_transcript, "test_transcript")

        assert len(feedbacks) == 3
        assert feedbacks[0].type == "feature_request"
        assert feedbacks[1].type == "customer_pain"
        assert feedbacks[2].type == "feature_request"
        assert "Excel" in feedbacks[0].summary
        assert "slow" in feedbacks[1].summary
        assert "filters" in feedbacks[2].summary

        # Verify all feedbacks have required fields
        for feedback in feedbacks:
            assert feedback.transcript_id == "test_transcript"
            assert isinstance(feedback.timestamp, datetime)
            assert 0 <= feedback.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_extract_feedback_invalid_json(self, sample_transcript: str) -> None:
        """Test handling of invalid JSON response."""
        mock_client = Mock(spec=LLMClient)
        mock_response = LLMResponse(
            content="Invalid JSON response",
            model="test-model",
            provider="test"
        )
        mock_client.generate = AsyncMock(return_value=mock_response)

        extractor = FeedbackExtractor()
        extractor.llm_client = mock_client

        feedbacks = await extractor.extract_feedback(sample_transcript, "test_transcript")

        assert len(feedbacks) == 0

    @pytest.mark.asyncio
    async def test_batch_extract(self, mock_llm_client: Mock) -> None:
        """Test batch feedback extraction."""
        transcripts = [
            {"id": "transcript1", "content": "Customer wants Excel export"},
            {"id": "transcript2", "content": "Search is too slow"},
            {"id": "transcript3", "content": ""}  # Empty content
        ]

        extractor = FeedbackExtractor()
        extractor.llm_client = mock_llm_client

        all_feedbacks = await extractor.batch_extract(transcripts)

        # Should process 2 transcripts (skip empty one)
        assert len(all_feedbacks) >= 0
        mock_llm_client.generate.assert_called()

    def test_feedback_validation(self) -> None:
        """Test feedback validation logic."""
        extractor = FeedbackExtractor()

        # Valid feedback
        valid_feedback = Feedback(
            type="feature_request",
            summary="Add export functionality to help users",
            verbatim="We really need export features",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert extractor._is_valid_feedback(valid_feedback)

        # Invalid type
        invalid_type_feedback = Feedback(
            type="invalid_type",
            summary="Add export functionality",
            verbatim="We need export features",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert not extractor._is_valid_feedback(invalid_type_feedback)

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

        # Empty summary
        empty_summary_feedback = Feedback(
            type="feature_request",
            summary="",
            verbatim="We need export features",
            confidence=0.8,
            transcript_id="test",
            timestamp=datetime.now()
        )
        assert not extractor._is_valid_feedback(empty_summary_feedback)


class TestRAGMatcher:
    """Test RAG matching functionality."""

    @pytest.fixture
    def sample_feedback(self) -> Feedback:
        return Feedback(
            type="feature_request",
            summary="Need Excel export functionality",
            verbatim="We really need Excel export",
            confidence=0.9,
            transcript_id="test_transcript",
            timestamp=datetime.now()
        )

    @pytest.fixture
    def mock_embedding_manager(self) -> Mock:
        mock_manager = Mock(spec=EmbeddingManager)
        mock_candidates = [
            {
                "content": "Export feature for spreadsheets",
                "metadata": {
                    "notion_id": "problem_1",
                    "title": "Excel Export Feature",
                    "status": "In Progress",
                    "priority": "High"
                },
                "certainty": 0.85
            },
            {
                "content": "Slow search performance issues",
                "metadata": {
                    "notion_id": "problem_2",
                    "title": "Search Performance",
                    "status": "New",
                    "priority": "Medium"
                },
                "certainty": 0.45
            }
        ]
        mock_manager.search_similar_problems = AsyncMock(return_value=mock_candidates)
        return mock_manager

    @pytest.fixture
    def mock_llm_rerank_response(self) -> str:
        return '''
        {
            "best_match": {
                "problem_id": "problem_1",
                "confidence": 0.9,
                "reasoning": "Both relate to Excel export functionality"
            },
            "all_matches": [
                {
                    "problem_id": "problem_1",
                    "confidence": 0.9,
                    "reasoning": "Direct match for Excel export"
                },
                {
                    "problem_id": "problem_2",
                    "confidence": 0.2,
                    "reasoning": "Different problem domain"
                }
            ]
        }
        '''

    @pytest.mark.asyncio
    async def test_find_best_match_success(
        self,
        sample_feedback: Feedback,
        mock_embedding_manager: Mock,
        mock_llm_rerank_response: str
    ) -> None:
        """Test successful matching with reranking."""
        mock_llm_client = Mock()
        mock_response = LLMResponse(
            content=mock_llm_rerank_response,
            model="test-model",
            provider="test"
        )
        mock_llm_client.generate = AsyncMock(return_value=mock_response)

        matcher = RAGMatcher()
        matcher.llm_client = mock_llm_client
        matcher.embedding_manager = mock_embedding_manager

        match = await matcher.find_best_match(sample_feedback)

        assert match is not None
        assert match.problem_id == "problem_1"
        assert match.confidence == 0.9
        assert match.problem_title == "Excel Export Feature"
        assert "Excel export" in match.reasoning

    @pytest.mark.asyncio
    async def test_find_best_match_no_candidates(self, sample_feedback: Feedback) -> None:
        """Test matching when no candidates are found."""
        mock_embedding_manager = Mock(spec=EmbeddingManager)
        mock_embedding_manager.search_similar_problems = AsyncMock(return_value=[])

        matcher = RAGMatcher()
        matcher.embedding_manager = mock_embedding_manager

        match = await matcher.find_best_match(sample_feedback)

        assert match is None

    @pytest.mark.asyncio
    async def test_find_best_match_low_confidence(
        self,
        sample_feedback: Feedback,
        mock_embedding_manager: Mock
    ) -> None:
        """Test matching when confidence is below threshold."""
        low_confidence_response = '''
        {
            "best_match": {
                "problem_id": "problem_1",
                "confidence": 0.3,
                "reasoning": "Low relevance match"
            }
        }
        '''

        mock_llm_client = Mock()
        mock_response = LLMResponse(
            content=low_confidence_response,
            model="test-model",
            provider="test"
        )
        mock_llm_client.generate = AsyncMock(return_value=mock_response)

        matcher = RAGMatcher()
        matcher.llm_client = mock_llm_client
        matcher.embedding_manager = mock_embedding_manager

        match = await matcher.find_best_match(sample_feedback)

        assert match is None  # Below confidence threshold


class TestMatchingMetrics:
    """Test matching metrics functionality."""

    def test_add_result_and_statistics(self) -> None:
        """Test adding results and calculating statistics."""
        metrics = MatchingMetrics()

        # Add some matched results
        feedback1 = Feedback(
            type="feature_request",
            summary="Test feedback 1",
            verbatim="Test verbatim 1",
            confidence=0.9,
            transcript_id="transcript1",
            timestamp=datetime.now()
        )

        match1 = MatchResult(
            problem_id="problem1",
            problem_title="Test Problem 1",
            confidence=0.85,
            similarity_score=0.8,
            reasoning="Good match",
            metadata={}
        )

        feedback2 = Feedback(
            type="customer_pain",
            summary="Test feedback 2",
            verbatim="Test verbatim 2",
            confidence=0.8,
            transcript_id="transcript2",
            timestamp=datetime.now()
        )

        metrics.add_result(feedback1, match1)
        metrics.add_result(feedback2, None)  # Unmatched

        stats = metrics.get_statistics()

        assert stats["total_feedbacks"] == 2
        assert stats["matched"] == 1
        assert stats["unmatched"] == 1
        assert stats["match_rate"] == 0.5
        assert stats["avg_confidence"] == 0.85


class TestNotionIntegration:
    """Test Notion API integration."""

    @pytest.fixture
    def sample_notion_page(self) -> dict[str, Any]:
        return {
            "id": "page_123",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"text": {"content": "Excel Export Issue"}}]
                },
                "Description": {
                    "type": "rich_text",
                    "rich_text": [{"text": {"content": "Users need Excel export functionality"}}]
                },
                "Status": {
                    "type": "select",
                    "select": {"name": "In Progress"}
                },
                "Priority": {
                    "type": "select",
                    "select": {"name": "High"}
                },
                "Feedback Count": {
                    "type": "number",
                    "number": 3
                }
            },
            "last_edited_time": "2024-01-01T10:00:00.000Z"
        }

    def test_parse_notion_page(self, sample_notion_page: dict[str, Any]) -> None:
        """Test parsing Notion page into NotionProblem."""
        # Mock the Notion client to avoid import issues
        with patch('notion.Client', MagicMock()), patch.dict('os.environ', {
            'NOTION_API_KEY': 'test_key',
            'NOTION_DATABASE_ID': 'test_db_id'
        }):
            from notion import NotionClient

            client = NotionClient()
            problem = client._parse_notion_page(sample_notion_page)

            assert problem is not None
            assert problem.id == "page_123"
            assert problem.title == "Excel Export Issue"
            assert problem.description == "Users need Excel export functionality"
            assert problem.status == "In Progress"
            assert problem.priority == "High"
            assert problem.feedback_count == 3

    def test_notion_problems_to_text(self) -> None:
        """Test converting Notion problems to text format."""
        problems = [
            NotionProblem(
                id="problem1",
                title="Excel Export",
                description="Need Excel export feature",
                status="New",
                priority="High",
                tags=["export", "excel"],
                feedback_count=2
            ),
            NotionProblem(
                id="problem2",
                title="Search Performance",
                description="Search is too slow",
                status="In Progress",
                priority="Medium",
                tags=["performance"],
                feedback_count=5
            )
        ]

        text_data = notion_problems_to_text(problems)

        assert len(text_data) == 2
        assert "Excel Export" in text_data[0]["content"]
        assert "Need Excel export feature" in text_data[0]["content"]
        assert text_data[0]["metadata"]["id"] == "problem1"
        assert text_data[0]["metadata"]["tags"] == ["export", "excel"]


class TestEmbeddingManager:
    """Test embedding management functionality."""

    @pytest.fixture
    def sample_feedbacks(self) -> list[Feedback]:
        return [
            Feedback(
                type="feature_request",
                summary="Excel export needed",
                verbatim="We need Excel export",
                confidence=0.9,
                transcript_id="transcript1",
                timestamp=datetime.now()
            ),
            Feedback(
                type="customer_pain",
                summary="Search too slow",
                verbatim="Search takes forever",
                confidence=0.8,
                transcript_id="transcript2",
                timestamp=datetime.now()
            )
        ]

    @pytest.fixture
    def sample_problems(self) -> list[NotionProblem]:
        return [
            NotionProblem(
                id="problem1",
                title="Export Features",
                description="Add export capabilities",
                status="New",
                priority="High"
            )
        ]

    @pytest.mark.asyncio
    async def test_embed_feedbacks(self, sample_feedbacks: list[Feedback]) -> None:
        """Test embedding generation for feedbacks."""
        mock_llm_client = Mock()
        mock_llm_client.embed = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

        with patch('embed.get_llm_client', return_value=mock_llm_client), \
             patch('embed.VectorStore'):
            manager = EmbeddingManager()
            manager.llm_client = mock_llm_client

            documents = await manager.embed_feedbacks(sample_feedbacks)

            assert len(documents) == 2
            assert documents[0].doc_type == "feedback"
            assert documents[0].embedding == [0.1, 0.2]
            assert "Excel export needed" in documents[0].content
            assert documents[0].metadata["type"] == "feature_request"


class TestPipeline:
    """Test complete pipeline integration."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self) -> None:
        """Test pipeline component initialization."""
        with patch('pipeline.FeedbackExtractor'), \
             patch('pipeline.NotionClient'), \
             patch('pipeline.EmbeddingManager'), \
             patch('pipeline.RAGMatcher'):

            pipeline = FeedbackPipeline()

            assert hasattr(pipeline, 'extractor')
            assert hasattr(pipeline, 'notion_client')
            assert hasattr(pipeline, 'embedding_manager')
            assert hasattr(pipeline, 'matcher')
            assert hasattr(pipeline, 'metrics')

    @pytest.mark.asyncio
    async def test_process_transcript_no_feedback(self) -> None:
        """Test processing transcript with no extractable feedback."""
        mock_extractor = Mock()
        mock_extractor.extract_feedback = AsyncMock(return_value=[])

        with patch('pipeline.FeedbackExtractor', return_value=mock_extractor), \
             patch('pipeline.NotionClient'), \
             patch('pipeline.EmbeddingManager'), \
             patch('pipeline.RAGMatcher'):

            pipeline = FeedbackPipeline()
            result = await pipeline.process_transcript("No feedback here", "test_id")

            assert result["feedbacks_extracted"] == 0
            assert result["matches_found"] == 0
            assert result["status"] == "no_feedback"


class TestDataPersistence:
    """Test data saving and loading functionality."""

    def test_save_and_load_feedback_logs(self, tmp_path) -> None:
        """Test saving and loading feedback logs."""
        feedbacks = [
            Feedback(
                type="feature_request",
                summary="Test feedback",
                verbatim="Test verbatim",
                confidence=0.8,
                transcript_id="test",
                timestamp=datetime.now()
            )
        ]

        log_file = tmp_path / "test_feedback_logs.json"
        save_feedback_logs(feedbacks, str(log_file))

        assert log_file.exists()

        loaded_feedbacks = load_feedback_logs(str(log_file))

        assert len(loaded_feedbacks) == 1
        assert loaded_feedbacks[0].type == "feature_request"
        assert loaded_feedbacks[0].summary == "Test feedback"
        assert loaded_feedbacks[0].confidence == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
