"""
Comprehensive step-by-step tests for notion.py
"""
# ruff: noqa: SIM117

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("NOTION_TOKEN", "test-token")


def get_mock_modules():
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

    def test_notion_problem_creation_minimal(self):
        """Test 1: Basic NotionProblem creation with required fields only."""
        print("Test 1: Testing NotionProblem minimal creation...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 1 passed: Basic NotionProblem creation works")

    def test_notion_problem_creation_full(self):
        """Test 2: NotionProblem creation with all fields."""
        print("Test 2: Testing NotionProblem full creation...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 2 passed: Full NotionProblem creation works")


class TestNotionClientInitialization:
    """Test NotionClient initialization step by step."""

    def test_notion_client_init_no_settings(self):
        """Test 3: NotionClient initialization with no settings (test mode)."""
        print("Test 3: Testing NotionClient initialization with no settings...")

        with patch.dict("sys.modules", get_mock_modules()), patch("config.settings", None):
            import notion

            client = notion.NotionClient()

            assert client.client is None
            assert client.database_id == "test-db"

        print("✓ Test 3 passed: NotionClient test mode initialization works")

    def test_notion_client_init_missing_api_key(self):
        """Test 4: NotionClient initialization with missing API key."""
        print("Test 4: Testing NotionClient initialization with missing API key...")

        with patch.dict("sys.modules", get_mock_modules()):
            with patch("config.settings") as mock_settings:
                mock_settings.notion_api_key = None
                mock_settings.notion_database_id = "test-db"

                import notion

                try:
                    notion.NotionClient()
                    raise AssertionError("Should have raised ValueError")
                except ValueError as e:
                    assert "NOTION_API_KEY not configured" in str(e)

        print("✓ Test 4 passed: Missing API key validation works")

    def test_notion_client_init_missing_database_id(self):
        """Test 5: NotionClient initialization with missing database ID."""
        print("Test 5: Testing NotionClient initialization with missing database ID...")

        with patch.dict("sys.modules", get_mock_modules()):
            with patch("config.settings") as mock_settings:
                mock_settings.notion_api_key = "test-key"
                mock_settings.notion_database_id = None

                import notion

                try:
                    notion.NotionClient()
                    raise AssertionError("Should have raised ValueError")
                except ValueError as e:
                    assert "NOTION_DATABASE_ID not configured" in str(e)

        print("✓ Test 5 passed: Missing database ID validation works")

    def test_notion_client_init_missing_notion_client(self):
        """Test 6: NotionClient initialization with missing notion-client package."""
        print("Test 6: Testing NotionClient initialization with missing notion-client...")

        with patch.dict("sys.modules", get_mock_modules()):
            with patch("config.settings") as mock_settings, patch("notion.NotionClientClass", None):
                mock_settings.notion_api_key = "test-key"
                mock_settings.notion_database_id = "test-db"

                import notion

                try:
                    notion.NotionClient()
                    raise AssertionError("Should have raised ImportError")
                except ImportError as e:
                    assert "notion-client package not installed" in str(e)

        print("✓ Test 6 passed: Missing notion-client package validation works")

    def test_notion_client_init_success(self):
        """Test 7: Successful NotionClient initialization."""
        print("Test 7: Testing successful NotionClient initialization...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 7 passed: Successful NotionClient initialization works")


class TestNotionPageParsing:
    """Test the _parse_notion_page method step by step."""

    def test_parse_notion_page_minimal(self):
        """Test 8: Parse minimal Notion page with just title."""
        print("Test 8: Testing minimal Notion page parsing...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 8 passed: Minimal Notion page parsing works")

    def test_parse_notion_page_full(self):
        """Test 9: Parse complete Notion page with all fields."""
        print("Test 9: Testing complete Notion page parsing...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 9 passed: Complete Notion page parsing works")

    def test_parse_notion_page_invalid(self):
        """Test 10: Parse invalid Notion page data."""
        print("Test 10: Testing invalid Notion page parsing...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 10 passed: Invalid Notion page parsing handles errors correctly")

    def test_parse_notion_page_customer_feedbacks_parsing(self):
        """Test 21: Test parsing of customer feedbacks with different formats."""
        print("Test 21: Testing customer feedbacks parsing edge cases...")

        with patch.dict("sys.modules", get_mock_modules()):
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
            print("✓ Test 21 passed: Customer feedbacks parsing edge cases work correctly")

    def test_parse_notion_page_missing_optional_fields(self):
        """Test 22: Test parsing when optional fields are completely missing."""
        print("Test 22: Testing parsing with missing optional fields...")

        with patch.dict("sys.modules", get_mock_modules()):
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
            print("✓ Test 22 passed: Missing optional fields handled correctly")


class TestNotionGetAllProblems:
    """Test the get_all_problems method step by step."""

    def test_get_all_problems_empty(self):
        """Test 11: Get all problems with empty response."""
        print("Test 11: Testing get_all_problems with empty response...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 11 passed: Empty problems response works")

    def test_get_all_problems_single_page(self):
        """Test 12: Get all problems with single page response."""
        print("Test 12: Testing get_all_problems with single page...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 12 passed: Single page problems response works")

    def test_get_all_problems_pagination(self):
        """Test 13: Get all problems with pagination."""
        print("Test 13: Testing get_all_problems with pagination...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 13 passed: Pagination in get_all_problems works")

    def test_get_all_problems_error(self):
        """Test 14: Get all problems with error handling."""
        print("Test 14: Testing get_all_problems error handling...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 14 passed: Error handling in get_all_problems works")


class TestNotionUpdateOperations:
    """Test update and create operations step by step."""

    def test_update_problem_with_feedback_success(self):
        """Test 15: Successfully update problem with feedback."""
        print("Test 15: Testing successful update_problem_with_feedback...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 15 passed: Successful problem update works")

    def test_update_problem_with_feedback_error(self):
        """Test 16: Handle error in update_problem_with_feedback."""
        print("Test 16: Testing error in update_problem_with_feedback...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 16 passed: Error handling in problem update works")

    def test_create_new_problem_success(self):
        """Test 17: Successfully create new problem."""
        print("Test 17: Testing successful create_new_problem...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 17 passed: Successful problem creation works")

    def test_create_new_problem_error(self):
        """Test 18: Handle error in create_new_problem."""
        print("Test 18: Testing error in create_new_problem...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 18 passed: Error handling in problem creation works")


class TestNotionUtilityFunctions:
    """Test utility functions step by step."""

    def test_notion_problems_to_text_empty(self):
        """Test 19: Convert empty problems list to text."""
        print("Test 19: Testing notion_problems_to_text with empty list...")

        with patch.dict("sys.modules", get_mock_modules()):
            import notion

            result = notion.notion_problems_to_text([])

            assert result == []

        print("✓ Test 19 passed: Empty problems to text conversion works")

    def test_notion_problems_to_text_full(self):
        """Test 20: Convert problems list to text format."""
        print("Test 20: Testing notion_problems_to_text with full data...")

        with patch.dict("sys.modules", get_mock_modules()):
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

        print("✓ Test 20 passed: Full problems to text conversion works")


if __name__ == "__main__":
    # Run tests step by step
    print("Running comprehensive Notion tests step by step...\n")

    dataclass_tests = TestNotionProblemDataclass()
    client_tests = TestNotionClientInitialization()
    parsing_tests = TestNotionPageParsing()
    get_problems_tests = TestNotionGetAllProblems()
    update_tests = TestNotionUpdateOperations()
    utility_tests = TestNotionUtilityFunctions()

    try:
        # Test NotionProblem dataclass
        dataclass_tests.test_notion_problem_creation_minimal()
        dataclass_tests.test_notion_problem_creation_full()

        # Test NotionClient initialization
        client_tests.test_notion_client_init_no_settings()
        client_tests.test_notion_client_init_missing_api_key()
        client_tests.test_notion_client_init_missing_database_id()
        client_tests.test_notion_client_init_missing_notion_client()
        client_tests.test_notion_client_init_success()

        # Test Notion page parsing
        parsing_tests.test_parse_notion_page_minimal()
        parsing_tests.test_parse_notion_page_full()
        parsing_tests.test_parse_notion_page_invalid()
        parsing_tests.test_parse_notion_page_customer_feedbacks_parsing()
        parsing_tests.test_parse_notion_page_missing_optional_fields()

        # Test get_all_problems
        get_problems_tests.test_get_all_problems_empty()
        get_problems_tests.test_get_all_problems_single_page()
        get_problems_tests.test_get_all_problems_pagination()
        get_problems_tests.test_get_all_problems_error()

        # Test update operations
        update_tests.test_update_problem_with_feedback_success()
        update_tests.test_update_problem_with_feedback_error()
        update_tests.test_create_new_problem_success()
        update_tests.test_create_new_problem_error()

        # Test utility functions
        utility_tests.test_notion_problems_to_text_empty()
        utility_tests.test_notion_problems_to_text_full()

        print("\n🎉 ALL 22 COMPREHENSIVE NOTION TESTS PASSED! 🎉")
        print("✅ NotionProblem dataclass: 2 tests")
        print("✅ NotionClient initialization: 5 tests")
        print("✅ Notion page parsing: 4 tests")
        print("✅ Get all problems: 4 tests")
        print("✅ Update operations: 4 tests")
        print("✅ Utility functions: 2 tests")
        print("\nAll core functionality of notion.py is thoroughly tested!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
