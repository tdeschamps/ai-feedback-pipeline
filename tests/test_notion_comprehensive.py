"""
Comprehensive pytest tests for notion.py
"""
# ruff: noqa: SIM117

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch
import pytest

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("NOTION_TOKEN", "test-token")


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
    "pydantic": Mock(),
    "pydantic._internal": Mock(),
    "pydantic._internal._config": Mock(),
    "pydantic_settings": Mock(),
    "pydantic_settings.main": Mock(),
  }


class TestNotionProblemDataclass:
  """Test the NotionProblem dataclass step by step."""

  def test_notion_problem_creation_minimal(self, mock_modules):
    """Test basic NotionProblem creation with required fields only."""
    with patch.dict("sys.modules", mock_modules):
      import notion

      problem = notion.NotionProblem(
        id="test-123", title="Test Problem", description="Test description", status="New"
      )

      assert problem.id == "test-123"
      assert problem.title == "Test Problem"
      assert problem.description == "Test description"
      assert problem.status == "New"
      assert problem.feedbacks is None
      assert problem.priority is None
      assert problem.tags is None
      assert problem.feedback_count == 0
      assert problem.last_updated is None

  def test_notion_problem_creation_full(self, mock_modules):
    """Test NotionProblem creation with all fields."""
    with patch.dict("sys.modules", mock_modules):
      import notion

      test_date = datetime.now()
      problem = notion.NotionProblem(
        id="test-456",
        title="Full Test Problem",
        description="Full test description",
        status="In Progress",
        feedbacks=["feedback1", "feedback2"],
        priority="High",
        tags=["bug", "urgent"],
        feedback_count=5,
        last_updated=test_date,
      )

      assert problem.id == "test-456"
      assert problem.title == "Full Test Problem"
      assert problem.feedbacks == ["feedback1", "feedback2"]
      assert problem.priority == "High"
      assert problem.tags == ["bug", "urgent"]
      assert problem.feedback_count == 5
      assert problem.last_updated == test_date


class TestNotionClientInitialization:
  """Test NotionClient initialization step by step."""

  def test_notion_client_init_no_settings(self, mock_modules):
    """Test NotionClient initialization with no settings (test mode)."""
    with patch.dict("sys.modules", mock_modules), patch("config.settings", None):
      import notion

      client = notion.NotionClient()

      assert client.client is None
      assert client.database_id == "test-db"

  def test_notion_client_init_missing_api_key(self, mock_modules):
    """Test NotionClient initialization with missing API key."""
    with patch.dict("sys.modules", mock_modules):
      with patch("config.settings") as mock_settings:
        mock_settings.notion_api_key = None
        mock_settings.notion_database_id = "test-db"

        import notion

        with pytest.raises(ValueError, match="NOTION_API_KEY not configured"):
          notion.NotionClient()

  def test_notion_client_init_missing_database_id(self, mock_modules):
    """Test NotionClient initialization with missing database ID."""
    with patch.dict("sys.modules", mock_modules):
      with patch("config.settings") as mock_settings:
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = None

        import notion

        with pytest.raises(ValueError, match="NOTION_DATABASE_ID not configured"):
          notion.NotionClient()

  def test_notion_client_init_missing_notion_client(self, mock_modules):
    """Test NotionClient initialization with missing notion-client package."""
    with patch.dict("sys.modules", mock_modules):
      with patch("config.settings") as mock_settings, patch("notion.NotionClientClass", None):
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db"

        import notion

        with pytest.raises(ImportError, match="notion-client package not installed"):
          notion.NotionClient()

  def test_notion_client_init_success(self, mock_modules):
    """Test successful NotionClient initialization."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
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


class TestNotionPageParsing:
  """Test the _parse_notion_page method step by step."""

  def test_parse_notion_page_minimal(self, mock_modules):
    """Test parse minimal Notion page with just title."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
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
            "Title": {
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
        assert problem.priority is None
        assert problem.tags == []
        assert problem.feedback_count == 0

  def test_parse_notion_page_full(self, mock_modules):
    """Test parse complete Notion page with all fields."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
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
            "Title": {
              "type": "title",
              "title": [{"text": {"content": "Complete Test Problem"}}],
            },
            "Description": {
              "type": "rich_text",
              "rich_text": [{"text": {"content": "This is a test description"}}],
            },
            "Status": {"type": "select", "select": {"name": "In Progress"}},
            "Priority": {"type": "select", "select": {"name": "High"}},
            "Tags": {
              "type": "multi_select",
              "multi_select": [{"name": "bug"}, {"name": "urgent"}],
            },
            "Feedback Count": {"type": "number", "number": 5},
            "🚀 Customer Feedbacks 1": {
              "rich_text": [
                {
                  "text": {
                    "content": "Customer complaint about feature (call-123)"
                  }
                },
                {"text": {"content": "Another feedback item (call-456)"}},
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
        assert problem.priority == "High"
        assert problem.tags == ["bug", "urgent"]
        assert problem.feedback_count == 5
        assert len(problem.feedbacks) == 2
        assert "Customer complaint about feature" in problem.feedbacks
        assert "Another feedback item" in problem.feedbacks

  def test_parse_notion_page_invalid(self, mock_modules):
    """Test parse invalid Notion page data."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
        patch("notion.NotionClientClass") as mock_notion_class,
      ):
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db"
        mock_notion_class.return_value = Mock()

        import notion

        client = notion.NotionClient()

        # Invalid page data (missing required fields)
        invalid_page_data = {
          "id": "page-invalid",
          "properties": {},
          # Missing last_edited_time
        }

        problem = client._parse_notion_page(invalid_page_data)

        # Should return None for invalid data
        assert problem is None

  def test_parse_notion_page_customer_feedbacks_parsing(self, mock_modules):
    """Test parsing of customer feedbacks with different formats."""
    with patch.dict("sys.modules", mock_modules):
      import notion

      client = notion.NotionClient()

      # Test page with various customer feedback formats
      page_data = {
        "id": "page-123",
        "last_edited_time": "2024-01-15T10:30:00Z",
        "properties": {
          "Name": {"type": "title", "title": [{"text": {"content": "Test Problem"}}]},
          "🚀 Customer Feedbacks 1": {
            "rich_text": [
              {"text": {"content": "First feedback (2024-01-01)"}},
              {"text": {"content": "Second feedback without parentheses"}},
              {"text": {"content": ""}},  # Empty feedback
              {"text": {"content": "Third feedback (with extra info)"}},
            ]
          },
          "Description": {
            "type": "rich_text",
            "rich_text": [{"text": {"content": "Test description"}}],
          },
          "Status": {"type": "select", "select": {"name": "Open"}},
        },
      }

      result = client._parse_notion_page(page_data)

      assert result is not None
      assert result.feedbacks == [
        "First feedback",
        "Third feedback",
      ]  # Only those with parentheses, empty excluded

  def test_parse_notion_page_missing_optional_fields(self, mock_modules):
    """Test parsing when optional fields are completely missing."""
    with patch.dict("sys.modules", mock_modules):
      import notion

      client = notion.NotionClient()

      # Minimal page with only required fields
      page_data = {
        "id": "page-minimal",
        "last_edited_time": "2024-01-15T10:30:00Z",
        "properties": {
          "Name": {"type": "title", "title": [{"text": {"content": "Minimal Problem"}}]}
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
      assert result.feedbacks == []
      assert result.feedback_count == 0


class TestNotionGetAllProblems:
  """Test the get_all_problems method step by step."""

  def test_get_all_problems_empty(self, mock_modules):
    """Test get all problems with empty response."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
        patch("notion.NotionClientClass") as mock_notion_class,
      ):
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db"

        mock_client_instance = Mock()
        mock_client_instance.databases.query.return_value = {
          "results": [],
          "has_more": False,
          "next_cursor": None,
        }
        mock_notion_class.return_value = mock_client_instance

        import notion

        client = notion.NotionClient()
        problems = client.get_all_problems()

        assert problems == []
        mock_client_instance.databases.query.assert_called_once()

  def test_get_all_problems_single_page(self, mock_modules):
    """Test get all problems with single page response."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
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
                "Title": {
                  "type": "title",
                  "title": [{"text": {"content": "Problem 1"}}],
                }
              },
              "last_edited_time": "2024-01-01T00:00:00.000Z",
            },
            {
              "id": "page-2",
              "properties": {
                "Title": {
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

  def test_get_all_problems_pagination(self, mock_modules):
    """Test get all problems with pagination."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
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
                  "Title": {
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
                  "Title": {
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

        # Should have been called twice for pagination
        assert mock_client_instance.databases.query.call_count == 2

  def test_get_all_problems_error(self, mock_modules):
    """Test get all problems with error handling."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
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


class TestNotionUpdateOperations:
  """Test update and create operations step by step."""

  def test_update_problem_with_feedback_success(self, mock_modules):
    """Test successfully update problem with feedback."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
        patch("notion.NotionClientClass") as mock_notion_class,
      ):
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db"

        mock_client_instance = Mock()
        mock_client_instance.pages.retrieve.return_value = {
          "properties": {
            "Description": {
              "type": "rich_text",
              "rich_text": [{"text": {"content": "Existing description"}}],
            },
            "Feedback Count": {"number": 2},
          }
        }
        mock_client_instance.pages.update.return_value = {"id": "page-123"}
        mock_notion_class.return_value = mock_client_instance

        import extract
        import notion

        client = notion.NotionClient()

        # Create test feedback
        feedback = extract.Feedback(
          type="feature_request",
          summary="Need better export",
          verbatim="Customer wants Excel export",
          confidence=0.9,
          transcript_id="call-123",
          timestamp=datetime(2024, 1, 1, 12, 0, 0),
          context="Customer call",
        )

        result = client.update_problem_with_feedback("page-123", feedback, 0.85)

        assert result is True
        mock_client_instance.pages.retrieve.assert_called_once_with(page_id="page-123")
        mock_client_instance.pages.update.assert_called_once()

  def test_update_problem_with_feedback_error(self, mock_modules):
    """Test handle error in update_problem_with_feedback."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
        patch("notion.NotionClientClass") as mock_notion_class,
      ):
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db"

        mock_client_instance = Mock()
        mock_client_instance.pages.retrieve.side_effect = Exception("API Error")
        mock_notion_class.return_value = mock_client_instance

        import extract
        import notion

        client = notion.NotionClient()

        feedback = extract.Feedback(
          type="customer_pain",
          summary="Test bug",
          verbatim="Test verbatim",
          confidence=0.8,
          transcript_id="call-456",
          timestamp=datetime.now(),
          context="Test context",
        )

        result = client.update_problem_with_feedback("page-123", feedback, 0.85)

        assert result is False

  def test_create_new_problem_success(self, mock_modules):
    """Test successfully create new problem."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
        patch("notion.NotionClientClass") as mock_notion_class,
      ):
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db"

        mock_client_instance = Mock()
        mock_client_instance.pages.create.return_value = {"id": "new-page-123"}
        mock_notion_class.return_value = mock_client_instance

        import extract
        import notion

        client = notion.NotionClient()

        feedback = extract.Feedback(
          type="feature_request",
          summary="New feature request",
          verbatim="Customer wants new dashboard",
          confidence=0.9,
          transcript_id="call-789",
          timestamp=datetime(2024, 1, 1, 15, 30, 0),
          context="Customer interview",
        )

        result = client.create_new_problem(feedback)

        assert result == "new-page-123"
        mock_client_instance.pages.create.assert_called_once()

  def test_create_new_problem_error(self, mock_modules):
    """Test handle error in create_new_problem."""
    with patch.dict("sys.modules", mock_modules):
      with (
        patch("config.settings") as mock_settings,
        patch("notion.NotionClientClass") as mock_notion_class,
      ):
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db"

        mock_client_instance = Mock()
        mock_client_instance.pages.create.side_effect = Exception("Create Error")
        mock_notion_class.return_value = mock_client_instance

        import extract
        import notion

        client = notion.NotionClient()

        feedback = extract.Feedback(
          type="customer_pain",
          summary="Test bug",
          verbatim="Test verbatim",
          confidence=0.8,
          transcript_id="call-error",
          timestamp=datetime.now(),
          context="Test context",
        )

        result = client.create_new_problem(feedback)

        assert result is None


class TestNotionUtilityFunctions:
  """Test utility functions step by step."""

  def test_notion_problems_to_text_empty(self, mock_modules):
    """Test convert empty problems list to text."""
    with patch.dict("sys.modules", mock_modules):
      import notion

      result = notion.notion_problems_to_text([])

      assert result == []

  def test_notion_problems_to_text_full(self, mock_modules):
    """Test convert problems list to text format."""
    with patch.dict("sys.modules", mock_modules):
      import notion

      problems = [
        notion.NotionProblem(
          id="problem-1",
          title="Export Feature",
          description="Users need export functionality",
          status="New",
          priority="High",
          tags=["export", "feature"],
          feedback_count=3,
        ),
        notion.NotionProblem(
          id="problem-2",
          title="Login Bug",
          description="Login issues with special chars",
          status="In Progress",
          priority="Medium",
          tags=["auth", "bug"],
          feedback_count=1,
        ),
      ]

      result = notion.notion_problems_to_text(problems)

      assert len(result) == 2

      # Check first problem
      first = result[0]
      assert "Export Feature" in first["content"]
      assert "Users need export functionality" in first["content"]
      assert first["metadata"]["id"] == "problem-1"
      assert first["metadata"]["priority"] == "High"
      assert "export" in first["metadata"]["tags"]

      # Check second problem
      second = result[1]
      assert "Login Bug" in second["content"]
      assert second["metadata"]["id"] == "problem-2"
      assert second["metadata"]["feedback_count"] == 1
