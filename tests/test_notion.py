"""
Test coverage for Notion integration functionality.
"""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("NOTION_TOKEN", "test-token")


def test_notion_problem_dataclass():
    """Test NotionProblem dataclass."""
    print("Testing NotionProblem dataclass...")

    mock_modules = {
        "notion_client": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        import notion

        problem = notion.NotionProblem(
            id="notion-123",
            title="Customer Export Issues",
            description="Users are having trouble with CSV exports",
            status="In Progress",
            priority="High",
            tags=["export", "csv", "bug"],
            feedback_count=3,
            last_updated=datetime.now(),
        )

        assert problem.id == "notion-123"
        assert problem.title == "Customer Export Issues"
        assert problem.priority == "High"
        assert "export" in problem.tags
        assert problem.feedback_count == 3

    print("✓ NotionProblem dataclass works")


def test_notion_client_initialization():
    """Test NotionClient initialization."""
    print("Testing NotionClient initialization...")

    mock_modules = {
        "notion_client": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db-id"

        with patch("notion.NotionClientClass") as mock_notion_class:
            mock_client_instance = Mock()
            mock_notion_class.return_value = mock_client_instance

            import notion

            client = notion.NotionClient()
            assert client is not None
            assert hasattr(client, "client")

    print("✓ NotionClient initialization works")


def test_notion_client_missing_config():
    """Test NotionClient with missing configuration."""
    print("Testing NotionClient with missing config...")

    mock_modules = {
        "notion_client": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.notion_api_key = None
        mock_settings.notion_database_id = "test-db"

        with patch("notion.NotionClientClass", None):
            import notion

            try:
                notion.NotionClient()
                raise AssertionError("Should have raised ValueError")
            except ValueError as e:
                assert "NOTION_API_KEY not configured" in str(e)

    print("✓ Missing config handling works")


def test_notion_problem_fetching():
    """Test fetching problems from Notion."""
    print("Testing Notion problem fetching...")

    mock_modules = {
        "notion_client": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db-id"

        # Mock Notion API response
        mock_response = {
            "results": [
                {
                    "id": "page-1",
                    "properties": {
                        "Title": {
                            "type": "title",
                            "title": [{"text": {"content": "Export Feature"}}],
                        },
                        "Description": {
                            "type": "rich_text",
                            "rich_text": [{"text": {"content": "Export functionality"}}],
                        },
                        "Status": {"type": "select", "select": {"name": "New"}},
                        "Priority": {"type": "select", "select": {"name": "High"}},
                        "Tags": {
                            "type": "multi_select",
                            "multi_select": [{"name": "export"}, {"name": "feature"}],
                        },
                        "Feedback Count": {"type": "number", "number": 2},
                    },
                    "last_edited_time": "2024-01-01T00:00:00.000Z",
                }
            ],
            "has_more": False,
            "next_cursor": None,
        }

        with patch("notion.NotionClientClass") as mock_notion_class:
            mock_client_instance = Mock()
            mock_client_instance.databases.query.return_value = mock_response
            mock_notion_class.return_value = mock_client_instance

            import notion

            client = notion.NotionClient()
            problems = client.get_all_problems()

            assert len(problems) == 1
            assert problems[0].title == "Export Feature"
            assert problems[0].priority == "High"
            assert "export" in problems[0].tags

    print("✓ Notion problem fetching works")


def test_notion_problem_updating():
    """Test updating problems in Notion."""
    print("Testing Notion problem updating...")

    mock_modules = {
        "notion_client": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.notion_api_key = "test-key"
        mock_settings.notion_database_id = "test-db-id"

        with patch("notion.NotionClientClass") as mock_notion_class:
            mock_client_instance = Mock()
            # Mock the retrieve method to return a page with properties
            mock_client_instance.pages.retrieve.return_value = {
                "properties": {
                    "Description": {
                        "type": "rich_text",
                        "rich_text": [{"text": {"content": "Existing description"}}],
                    },
                    "Feedback Count": {"number": 1},
                }
            }
            # Mock the update method to succeed
            mock_client_instance.pages.update.return_value = {"id": "page-1"}
            mock_notion_class.return_value = mock_client_instance

            import extract
            import notion

            client = notion.NotionClient()

            # Create test feedback
            feedback = extract.Feedback(
                type="feature_request",
                summary="Need Excel export",
                verbatim="Customer really wants Excel export",
                confidence=0.9,
                transcript_id="call-123",
                timestamp=datetime.now(),
                context="Customer call",
            )

            # Test update
            success = client.update_problem_with_feedback("page-1", feedback, 0.85)
            assert success is True

            # Verify the API was called
            mock_client_instance.pages.update.assert_called_once()

    print("✓ Notion problem updating works")


def run_notion_tests():
    """Run all Notion tests."""
    print("=" * 50)
    print("Running Notion Tests")
    print("=" * 50)

    tests = [
        test_notion_problem_dataclass,
        test_notion_client_initialization,
        test_notion_client_missing_config,
        test_notion_problem_fetching,
        test_notion_problem_updating,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("=" * 50)
    print(f"Notion Tests: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


def run_all_tests():
    """Run all Notion tests."""
    return run_notion_tests()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
