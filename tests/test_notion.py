"""
Unit tests for Notion API integration.
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

# Ensure project is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set required environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def setup_mocks():
    """Set up comprehensive mocks for external dependencies."""
    mock_modules = {
        # Core LangChain modules
        "langchain": MagicMock(),
        "langchain.schema": MagicMock(),
        "langchain_core": MagicMock(),
        "langchain_core.messages": MagicMock(),
        "langchain_openai": MagicMock(),
        "langchain_anthropic": MagicMock(),
        "langchain_huggingface": MagicMock(),
        "langchain_ollama": MagicMock(),
        "langchain_community": MagicMock(),
        "langchain_community.embeddings": MagicMock(),
        "langchain_community.llms": MagicMock(),
        # Vector stores and databases
        "chromadb": MagicMock(),
        "pinecone": MagicMock(),
        "notion_client": MagicMock(),
        # Other dependencies
        "sentence_transformers": MagicMock(),
        "groq": MagicMock(),
    }

    with patch.dict("sys.modules", mock_modules):
        yield


class TestNotionProblem:
    """Test NotionProblem dataclass."""

    def test_notion_problem_creation(self):
        """Test NotionProblem dataclass creation with all fields."""
        import notion

        problem = notion.NotionProblem(
            id="problem-123",
            title="Test Problem",
            description="This is a test problem description",
            status="In Progress",
            feedbacks=["Feedback 1", "Feedback 2"],
            priority="High",
            tags=["bug", "urgent"],
            feedback_count=5,
            last_updated=datetime(2024, 1, 1, 12, 0, 0),
        )

        assert problem.id == "problem-123"
        assert problem.title == "Test Problem"
        assert problem.description == "This is a test problem description"
        assert problem.status == "In Progress"
        assert problem.feedbacks == ["Feedback 1", "Feedback 2"]
        assert problem.priority == "High"
        assert problem.tags == ["bug", "urgent"]
        assert problem.feedback_count == 5
        assert problem.last_updated == datetime(2024, 1, 1, 12, 0, 0)

    def test_notion_problem_minimal(self):
        """Test NotionProblem with minimal required fields."""
        import notion

        problem = notion.NotionProblem(
            id="problem-minimal",
            title="Minimal Problem",
            description="Simple description",
            status="Open",
        )

        assert problem.id == "problem-minimal"
        assert problem.title == "Minimal Problem"
        assert problem.description == "Simple description"
        assert problem.status == "Open"
        assert problem.feedbacks is None
        assert problem.priority is None
        assert problem.tags is None
        assert problem.feedback_count == 0
        assert problem.last_updated is None


class TestNotionClientInitialization:
    """Test NotionClient initialization."""

    def test_notion_client_init_no_settings(self):
        """Test NotionClient initialization with no settings (test mode)."""
        with patch("notion.settings", None):
            import notion

            client = notion.NotionClient()

            assert client.client is None
            assert client.database_id == "test-db"

    def test_notion_client_init_missing_api_key(self):
        """Test NotionClient initialization with missing API key."""
        with patch("notion.settings") as mock_settings:
            mock_settings.notion_api_key = None
            mock_settings.notion_database_id = "test-db"

            import notion

            with pytest.raises(ValueError, match="NOTION_API_KEY not configured"):
                notion.NotionClient()

    def test_notion_client_init_missing_database_id(self):
        """Test NotionClient initialization with missing database ID."""
        with patch("notion.settings") as mock_settings:
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = None

            import notion

            with pytest.raises(ValueError, match="NOTION_DATABASE_ID not configured"):
                notion.NotionClient()

    def test_notion_client_init_success(self):
        """Test successful NotionClient initialization."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()
            mock_notion_class.return_value = mock_client_instance

            import notion

            client = notion.NotionClient()

            assert client.client == mock_client_instance
            assert client.database_id == "test-db"
            mock_notion_class.assert_called_once_with(auth="test-key")

    def test_notion_client_init_import_error(self):
        """Test NotionClient initialization when notion-client not available."""
        with patch("notion.settings") as mock_settings, patch("notion.NotionClientClass", None):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            import notion

            with pytest.raises(ImportError, match="notion-client package not installed"):
                notion.NotionClient()


class TestNotionPageParsing:
    """Test parsing of Notion pages."""

    def test_parse_notion_page_minimal(self):
        """Test parse minimal Notion page with just title."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"
            mock_notion_class.return_value = Mock()

            import notion

            client = notion.NotionClient()

            # Minimal page data
            page_data = {
                "id": "page-123",
                "properties": {
                    "Problème": {
                        "type": "title",
                        "title": [{"text": {"content": "Test Problem Title"}}],
                    }
                },
                "last_edited_time": "2024-01-01T00:00:00.000Z",
            }

            problem = client._parse_notion_page(page_data)

            assert problem is not None
            assert problem.id == "page-123"
            assert problem.title == "Test Problem Title"
            assert problem.description == ""
            assert problem.status == ""
            assert problem.feedbacks == []
            assert problem.priority is None
            assert problem.tags == []
            assert problem.feedback_count == 0

    def test_parse_notion_page_full(self):
        """Test parse complete Notion page with all fields."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"
            mock_notion_class.return_value = Mock()

            import notion

            client = notion.NotionClient()

            # Complete page data
            page_data = {
                "id": "page-456",
                "properties": {
                    "Problème": {
                        "type": "title",
                        "title": [{"text": {"content": "Complete Test Problem"}}],
                    },
                    "🚀 Customer Feedbacks": {
                        "type": "rich_text",
                        "rich_text": [{"text": {"content": "This is a test description"}}],
                    },
                    "Statut": {"type": "select", "select": {"name": "In Progress"}},
                    "Score Impact": {"type": "number", "number": 8},
                    "Thèmes": {
                        "type": "multi_select",
                        "multi_select": [{"name": "bug"}, {"name": "urgent"}],
                    },
                    "Composant": {"type": "select", "select": {"name": "API"}},
                    "# clients": {"type": "number", "number": 5},
                    "🚀 Customer Feedbacks 1": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": "Customer complaint about feature (call-123) Another feedback item (call-456)"
                                }
                            }
                        ]
                    },
                },
                "last_edited_time": "2024-01-01T12:30:00.000Z",
            }

            problem = client._parse_notion_page(page_data)

            assert problem is not None
            assert problem.id == "page-456"
            assert problem.title == "Complete Test Problem"
            assert problem.description == "This is a test description"
            assert problem.status == "In Progress"
            assert problem.priority == "High"  # Score 8 = High priority
            assert "bug" in problem.tags
            assert "urgent" in problem.tags
            assert "Component: API" in problem.tags
            assert problem.feedback_count == 5
            assert len(problem.feedbacks) == 2
            assert "Customer complaint about feature" in problem.feedbacks
            assert "Another feedback item" in problem.feedbacks

    def test_parse_notion_page_customer_feedbacks_parsing(self):
        """Test parsing of customer feedbacks with different formats."""
        import notion

        client = notion.NotionClient()

        # Test page with various customer feedback formats
        page_data = {
            "id": "page-123",
            "last_edited_time": "2024-01-15T10:30:00Z",
            "properties": {
                "Problème": {"type": "title", "title": [{"text": {"content": "Test Problem"}}]},
                "🚀 Customer Feedbacks 1": {
                    "rich_text": [
                        {
                            "text": {
                                "content": "First feedback (2024-01-01) Second feedback without parentheses Third feedback (with extra info)"
                            }
                        }
                    ]
                },
                "🚀 Customer Feedbacks": {
                    "type": "rich_text",
                    "rich_text": [{"text": {"content": "Test description"}}],
                },
                "Statut": {"type": "select", "select": {"name": "Open"}},
            },
        }

        result = client._parse_notion_page(page_data)

        assert result is not None
        assert result.feedbacks == [
            "First feedback",
            "Second feedback without parentheses Third feedback",
        ]  # Properly parsed feedback sections

    def test_parse_notion_page_priority_scoring(self):
        """Test priority calculation from Score Impact."""
        import notion

        client = notion.NotionClient()

        test_cases = [
            (9, "High"),
            (8, "High"),
            (7, "Medium"),
            (5, "Medium"),
            (3, "Low"),
            (1, "Low"),
        ]

        for score, expected_priority in test_cases:
            page_data = {
                "id": "page-priority-test",
                "last_edited_time": "2024-01-01T00:00:00Z",
                "properties": {
                    "Problème": {
                        "type": "title",
                        "title": [{"text": {"content": "Priority Test"}}],
                    },
                    "Score Impact": {"type": "number", "number": score},
                },
            }

            result = client._parse_notion_page(page_data)
            assert result is not None
            assert result.priority == expected_priority

    def test_parse_notion_page_missing_optional_fields(self):
        """Test parsing when optional fields are completely missing."""
        import notion

        client = notion.NotionClient()

        # Minimal page with only required fields
        page_data = {
            "id": "page-minimal",
            "last_edited_time": "2024-01-15T10:30:00Z",
            "properties": {
                "Problème": {"type": "title", "title": [{"text": {"content": "Minimal Problem"}}]}
                # Missing all optional fields: Description, Status, Priority, Tags, etc.
            },
        }

        result = client._parse_notion_page(page_data)

        assert result is not None
        assert result.title == "Minimal Problem"
        assert result.description == ""
        assert result.status == ""
        assert result.priority is None
        assert result.tags == []
        assert result.feedback_count == 0
        assert result.feedbacks == []

    def test_parse_notion_page_error_handling(self):
        """Test error handling in page parsing."""
        import notion

        client = notion.NotionClient()

        # Invalid page data that should trigger exception
        invalid_page_data = {"invalid": "data"}

        result = client._parse_notion_page(invalid_page_data)
        assert result is None


class TestNotionGetAllProblems:
    """Test retrieving all problems from Notion."""

    def test_get_all_problems_single_page(self):
        """Test get all problems with single page response."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()
            mock_client_instance.databases.query.return_value = {
                "results": [
                    {
                        "id": "page-1",
                        "properties": {
                            "Problème": {
                                "type": "title",
                                "title": [{"text": {"content": "Problem 1"}}],
                            }
                        },
                        "last_edited_time": "2024-01-01T00:00:00.000Z",
                    },
                    {
                        "id": "page-2",
                        "properties": {
                            "Problème": {
                                "type": "title",
                                "title": [{"text": {"content": "Problem 2"}}],
                            }
                        },
                        "last_edited_time": "2024-01-01T00:00:00.000Z",
                    },
                ],
                "has_more": False,
                "next_cursor": None,
            }
            mock_notion_class.return_value = mock_client_instance

            import notion

            client = notion.NotionClient()
            problems = client.get_all_problems()

            assert len(problems) == 2
            assert problems[0].id == "page-1"
            assert problems[0].title == "Problem 1"
            assert problems[1].id == "page-2"
            assert problems[1].title == "Problem 2"

            # Verify the query was called correctly
            mock_client_instance.databases.query.assert_called_once_with(
                database_id="test-db", page_size=100
            )

    def test_get_all_problems_pagination(self):
        """Test get all problems with pagination."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()

            # First call returns page 1 with more data
            # Second call returns page 2 with no more data
            mock_client_instance.databases.query.side_effect = [
                {
                    "results": [
                        {
                            "id": "page-1",
                            "properties": {
                                "Problème": {
                                    "type": "title",
                                    "title": [{"text": {"content": "Problem 1"}}],
                                }
                            },
                            "last_edited_time": "2024-01-01T00:00:00.000Z",
                        }
                    ],
                    "has_more": True,
                    "next_cursor": "cursor-123",
                },
                {
                    "results": [
                        {
                            "id": "page-2",
                            "properties": {
                                "Problème": {
                                    "type": "title",
                                    "title": [{"text": {"content": "Problem 2"}}],
                                }
                            },
                            "last_edited_time": "2024-01-01T00:00:00.000Z",
                        }
                    ],
                    "has_more": False,
                    "next_cursor": None,
                },
            ]
            mock_notion_class.return_value = mock_client_instance

            import notion

            client = notion.NotionClient()
            problems = client.get_all_problems()

            assert len(problems) == 2
            assert problems[0].title == "Problem 1"
            assert problems[1].title == "Problem 2"

            # Verify pagination was handled correctly
            assert mock_client_instance.databases.query.call_count == 2

    def test_get_all_problems_error_handling(self):
        """Test error handling in get_all_problems."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()
            mock_client_instance.databases.query.side_effect = Exception("API Error")
            mock_notion_class.return_value = mock_client_instance

            import notion

            client = notion.NotionClient()
            problems = client.get_all_problems()

            # Should return empty list on error
            assert problems == []


class TestNotionUpdateProblem:
    """Test updating problems in Notion."""

    def test_update_problem_with_feedback_success(self):
        """Test successful problem update with feedback."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()

            # Mock the page retrieval
            mock_client_instance.pages.retrieve.return_value = {
                "properties": {
                    "🚀 Customer Feedbacks 1": {
                        "type": "rich_text",
                        "rich_text": [{"text": {"content": "Existing feedback"}}],
                    },
                    "# clients": {"type": "number", "number": 3},
                }
            }

            # Mock successful update
            mock_client_instance.pages.update.return_value = {"id": "updated-page"}
            mock_notion_class.return_value = mock_client_instance

            import notion
            from extract import Feedback

            client = notion.NotionClient()

            # Create a test feedback
            feedback = Feedback(
                type="feature_request",
                summary="Test feedback summary",
                verbatim="This is test verbatim",
                confidence=0.85,
                transcript_id="transcript-123",
                timestamp=datetime(2024, 1, 15, 10, 30, 0),
                context="test context",
            )

            result = client.update_problem_with_feedback("problem-123", feedback, 0.85)

            assert result is True
            mock_client_instance.pages.retrieve.assert_called_once_with(page_id="problem-123")
            mock_client_instance.pages.update.assert_called_once()

            # Check the update call arguments
            update_call = mock_client_instance.pages.update.call_args
            assert update_call[1]["page_id"] == "problem-123"
            update_data = update_call[1]["properties"]

            # Verify feedback was appended
            new_feedback_content = update_data["🚀 Customer Feedbacks 1"]["rich_text"][0]["text"][
                "content"
            ]
            assert "Existing feedback" in new_feedback_content
            assert "Test feedback summary" in new_feedback_content
            assert "transcript-123" in new_feedback_content

            # Verify client count was incremented
            assert update_data["# clients"]["number"] == 4

    def test_update_problem_with_feedback_error(self):
        """Test error handling in problem update."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()
            mock_client_instance.pages.retrieve.side_effect = Exception("API Error")
            mock_notion_class.return_value = mock_client_instance

            import notion
            from extract import Feedback

            client = notion.NotionClient()

            feedback = Feedback(
                type="customer_pain",
                summary="Test bug",
                verbatim="Bug description",
                confidence=0.9,
                transcript_id="transcript-456",
                timestamp=datetime.now(),
                context="test context",
            )

            result = client.update_problem_with_feedback("problem-456", feedback, 0.9)

            assert result is False


class TestNotionCreateProblem:
    """Test creating new problems in Notion."""

    def test_create_new_problem_success(self):
        """Test successful creation of new problem."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()
            mock_client_instance.pages.create.return_value = {"id": "new-problem-123"}
            mock_notion_class.return_value = mock_client_instance

            import notion
            from extract import Feedback

            client = notion.NotionClient()

            feedback = Feedback(
                type="customer_pain",
                summary="New customer pain point",
                verbatim="Customer is frustrated with slow loading",
                confidence=0.75,
                transcript_id="transcript-789",
                timestamp=datetime(2024, 1, 20, 14, 15, 0),
                context="customer call",
            )

            result = client.create_new_problem(feedback)

            assert result == "new-problem-123"
            mock_client_instance.pages.create.assert_called_once()

            # Check the creation call arguments
            create_call = mock_client_instance.pages.create.call_args
            create_data = create_call[1]

            assert create_data["parent"]["database_id"] == "test-db"

            properties = create_data["properties"]
            assert (
                "New Issue: New customer pain point"
                in properties["Problème"]["title"][0]["text"]["content"]
            )
            assert (
                "Customer Pain"
                in properties["🚀 Customer Feedbacks 1"]["rich_text"][0]["text"]["content"]
            )
            assert properties["Statut"]["select"]["name"] == "New"
            assert properties["Score Impact"]["number"] == 5
            assert properties["# clients"]["number"] == 1

    def test_create_new_problem_error(self):
        """Test error handling in new problem creation."""
        with (
            patch("notion.settings") as mock_settings,
            patch("notion.NotionClientClass") as mock_notion_class,
        ):
            mock_settings.notion_api_key = "test-key"
            mock_settings.notion_database_id = "test-db"

            mock_client_instance = Mock()
            mock_client_instance.pages.create.side_effect = Exception("Creation failed")
            mock_notion_class.return_value = mock_client_instance

            import notion
            from extract import Feedback

            client = notion.NotionClient()

            feedback = Feedback(
                type="feature_request",
                summary="Failed creation test",
                verbatim="This should fail",
                confidence=0.8,
                transcript_id="transcript-fail",
                timestamp=datetime.now(),
                context="test",
            )

            result = client.create_new_problem(feedback)

            assert result is None


class TestNotionUtilities:
    """Test utility functions for Notion integration."""

    def test_notion_problems_to_text(self):
        """Test conversion of NotionProblem to text format for embedding."""
        import notion

        problems = [
            notion.NotionProblem(
                id="problem-1",
                title="First Problem",
                description="Description of first problem",
                status="Open",
                priority="High",
                tags=["bug", "urgent"],
                feedback_count=3,
            ),
            notion.NotionProblem(
                id="problem-2",
                title="Second Problem",
                description="Description of second problem",
                status="In Progress",
                priority="Medium",
                tags=["feature"],
                feedback_count=1,
            ),
        ]

        result = notion.notion_problems_to_text(problems)

        assert len(result) == 2

        # Check first problem
        first_result = result[0]
        assert "First Problem" in first_result["content"]
        assert "Description of first problem" in first_result["content"]
        assert first_result["metadata"]["id"] == "problem-1"
        assert first_result["metadata"]["title"] == "First Problem"
        assert first_result["metadata"]["status"] == "Open"
        assert first_result["metadata"]["priority"] == "High"
        assert first_result["metadata"]["tags"] == ["bug", "urgent"]
        assert first_result["metadata"]["feedback_count"] == 3

        # Check second problem
        second_result = result[1]
        assert "Second Problem" in second_result["content"]
        assert "Description of second problem" in second_result["content"]
        assert second_result["metadata"]["id"] == "problem-2"
        assert second_result["metadata"]["tags"] == ["feature"]

    def test_notion_problems_to_text_empty_list(self):
        """Test conversion with empty problem list."""
        import notion

        result = notion.notion_problems_to_text([])
        assert result == []

    def test_notion_problems_to_text_minimal_problem(self):
        """Test conversion with minimal problem data."""
        import notion

        problems = [
            notion.NotionProblem(
                id="minimal-problem",
                title="Minimal",
                description="Simple",
                status="Open",
            )
        ]

        result = notion.notion_problems_to_text(problems)

        assert len(result) == 1
        assert "Minimal" in result[0]["content"]
        assert "Simple" in result[0]["content"]
        assert result[0]["metadata"]["id"] == "minimal-problem"
        assert result[0]["metadata"]["tags"] == []
        assert result[0]["metadata"]["feedback_count"] == 0


if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v"])
