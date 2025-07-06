"""
Notion API integration for reading and updating customer problems.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any


try:
    from notion_client import Client as NotionClientClass
except ImportError:
    logging.warning("notion-client not available")
    NotionClientClass = None  # type: ignore

from config import settings
from extract import Feedback


logger = logging.getLogger(__name__)


@dataclass
class NotionProblem:
    """Represents a customer problem from Notion database."""

    id: str
    title: str
    description: str
    status: str
    feedbacks: list[str] | None = None
    priority: str | None = None
    tags: list[str] | None = None
    feedback_count: int = 0
    last_updated: datetime | None = None


class NotionClient:
    """Client for interacting with Notion API."""

    def __init__(self) -> None:
        if settings is None:
            # In test environment, create a mock client
            logger.warning("No settings available, using mock client")
            self.client = None
            self.database_id = "test-db"
            return

        if not settings.notion_api_key:
            raise ValueError("NOTION_API_KEY not configured")

        if not settings.notion_database_id:
            raise ValueError("NOTION_DATABASE_ID not configured")

        if NotionClientClass is None:
            raise ImportError("notion-client package not installed")

        self.client = NotionClientClass(auth=settings.notion_api_key)  # type: ignore
        self.database_id = settings.notion_database_id  # type: ignore

    def get_all_problems(self) -> list[NotionProblem]:
        """Retrieve all customer problems from Notion database."""
        try:
            problems = []
            has_more = True
            start_cursor = None

            while has_more:
                query_params: dict[str, Any] = {"database_id": self.database_id, "page_size": 100}  # type: ignore

                if start_cursor:
                    query_params["start_cursor"] = start_cursor

                response = self.client.databases.query(**query_params)  # type: ignore

                for page in response["results"]:
                    problem = self._parse_notion_page(page)
                    if problem:
                        problems.append(problem)

                has_more = response["has_more"]
                start_cursor = response.get("next_cursor")

            logger.info(f"Retrieved {len(problems)} problems from Notion")
            return problems

        except Exception as e:
            logger.error(f"Error retrieving problems from Notion: {e}")
            return []

    def _parse_notion_page(self, page: dict[str, Any]) -> NotionProblem | None:
        """Parse Notion page into NotionProblem object."""
        try:
            properties = page.get("properties", {})

            # Extract title from "Problème" property
            title = ""
            if "Problème" in properties:
                title_prop = properties["Problème"]
                if title_prop.get("type") == "title":
                    title_array = title_prop.get("title", [])
                    if title_array:
                        title = title_array[0].get("text", {}).get("content", "")

            # Extract existing feedbacks from "🚀 Customer Feedbacks 1"
            feedbacks = []
            customer_feedbacks = properties.get("🚀 Customer Feedbacks 1", {})
            for feedback in customer_feedbacks.get("rich_text", []):
                feedback_text = feedback.get("text", {}).get("content", "")
                if feedback_text:
                    # Split by parentheses and extract clean text parts
                    parts = feedback_text.split("(")

                    # Add the first part (before any parentheses)
                    if parts[0].strip():
                        feedbacks.append(parts[0].strip())

                    # Process remaining parts (after each opening parenthesis)
                    for part in parts[1:]:
                        if ")" in part:
                            # Extract text after the closing parenthesis
                            text_after_paren = part.split(")", 1)[1].strip()
                            if text_after_paren:
                                feedbacks.append(text_after_paren)

            # Extract description from "🚀 Customer Feedbacks" (main feedback column)
            description = ""
            if "🚀 Customer Feedbacks" in properties:
                desc_prop = properties["🚀 Customer Feedbacks"]
                if desc_prop.get("type") == "rich_text":
                    rich_text = desc_prop.get("rich_text", [])
                    if rich_text:
                        description = rich_text[0].get("text", {}).get("content", "")

            # Extract status from "Statut"
            status = ""
            if "Statut" in properties:
                status_prop = properties["Statut"]
                if status_prop.get("type") == "select":
                    select_data = status_prop.get("select")
                    if select_data:
                        status = select_data.get("name", "")

            # Extract priority from "Score Impact" (using this as priority indicator)
            priority = None
            if "Score Impact" in properties:
                priority_prop = properties["Score Impact"]
                if priority_prop.get("type") == "number":
                    score = priority_prop.get("number")
                    if score is not None:
                        # Convert score to priority level
                        if score >= 8:
                            priority = "High"
                        elif score >= 5:
                            priority = "Medium"
                        else:
                            priority = "Low"

            # Extract tags from "Thèmes" and "Composant"
            tags = []

            # Add themes
            if "Thèmes" in properties:
                themes_prop = properties["Thèmes"]
                if themes_prop.get("type") == "multi_select":
                    multi_select = themes_prop.get("multi_select", [])
                    tags.extend([tag.get("name", "") for tag in multi_select])

            # Add component as tag
            if "Composant" in properties:
                component_prop = properties["Composant"]
                if component_prop.get("type") == "select":
                    select_data = component_prop.get("select")
                    if select_data:
                        component = select_data.get("name", "")
                        if component:
                            tags.append(f"Component: {component}")

            # Extract feedback count from "# clients"
            feedback_count = 0
            if "# clients" in properties:
                count_prop = properties["# clients"]
                if count_prop.get("type") == "number":
                    feedback_count = count_prop.get("number", 0) or 0

            return NotionProblem(
                id=page["id"],
                title=title,
                feedbacks=feedbacks,
                description=description,
                status=status,
                priority=priority,
                tags=tags,
                feedback_count=feedback_count,
                last_updated=datetime.fromisoformat(
                    page["last_edited_time"].replace("Z", "+00:00")
                ),
            )

        except Exception as e:
            logger.error(f"Error parsing Notion page: {e}")
            return None

    def update_problem_with_feedback(
        self, problem_id: str, feedback: Feedback, confidence: float
    ) -> bool:
        """Update Notion problem by appending feedback."""
        try:
            # First, get the current page to read existing content
            page = self.client.pages.retrieve(page_id=problem_id)  # type: ignore

            # Prepare feedback entry
            timestamp_str = feedback.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            feedback_entry = f"\n\n**Feedback from {feedback.transcript_id}** ({timestamp_str}) - Confidence: {confidence:.2f}\n"
            feedback_entry += f"**Type:** {feedback.type.replace('_', ' ').title()}\n"
            feedback_entry += f"**Summary:** {feedback.summary}\n"
            feedback_entry += f'**Verbatim:** "{feedback.verbatim}"\n'

            # Get existing feedback content from "🚀 Customer Feedbacks 1"
            existing_feedback = ""
            properties = page.get("properties", {})
            if "🚀 Customer Feedbacks 1" in properties:
                feedback_prop = properties["🚀 Customer Feedbacks 1"]
                if feedback_prop.get("type") == "rich_text":
                    rich_text = feedback_prop.get("rich_text", [])
                    if rich_text:
                        existing_feedback = rich_text[0].get("text", {}).get("content", "")

            # Get current client count
            current_client_count = 0
            if "# clients" in properties:
                count_prop = properties["# clients"]
                if count_prop.get("type") == "number":
                    current_client_count = count_prop.get("number", 0) or 0

            # Update the page with correct column names
            update_data = {
                "properties": {
                    "🚀 Customer Feedbacks 1": {
                        "rich_text": [{"text": {"content": existing_feedback + feedback_entry}}]
                    },
                    "# clients": {"number": current_client_count + 1},
                    # Update the "Date" field (assuming it tracks last feedback date)
                    "Date": {"date": {"start": datetime.now().isoformat()}},
                }
            }

            self.client.pages.update(page_id=problem_id, **update_data)  # type: ignore
            logger.info(
                f"Updated Notion problem {problem_id} with feedback from {feedback.transcript_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating Notion problem {problem_id}: {e}")
            return False

    def create_new_problem(self, feedback: Feedback) -> str | None:
        """Create a new problem in Notion from unmatched feedback."""
        try:
            timestamp_str = feedback.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            new_page_data = {
                "parent": {"database_id": self.database_id},  # type: ignore
                "properties": {
                    "Problème": {  # Title column
                        "title": [{"text": {"content": f"New Issue: {feedback.summary[:50]}..."}}]
                    },
                    "🚀 Customer Feedbacks 1": {
                        "rich_text": [
                            {
                                "text": {
                                    "content": f"**Auto-created from feedback**\n\n"
                                    f"**Type:** {feedback.type.replace('_', ' ').title()}\n"
                                    f"**Summary:** {feedback.summary}\n"
                                    f'**Verbatim:** "{feedback.verbatim}"\n'
                                    f"**Source:** {feedback.transcript_id} ({timestamp_str})\n"
                                    f"**Confidence:** {feedback.confidence:.2f}"
                                }
                            }
                        ]
                    },
                    "Statut": {"select": {"name": "New"}},
                    "Score Impact": {"number": 5},  # Default medium impact
                    "# clients": {"number": 1},
                    "Date": {"date": {"start": datetime.now().isoformat()}},
                },
            }

            response = self.client.pages.create(**new_page_data)  # type: ignore
            logger.info(f"Created new Notion problem for feedback from {feedback.transcript_id}")
            return response["id"]

        except Exception as e:
            logger.error(f"Error creating new Notion problem: {e}")
            return None


def notion_problems_to_text(problems: list[NotionProblem]) -> list[dict[str, Any]]:
    """Convert Notion problems to text format for embedding."""
    texts = []
    for problem in problems:
        # Combine title and description for better semantic matching
        text_content = f"{problem.title}\n\n{problem.description}"

        # Add metadata
        metadata = {
            "id": problem.id,
            "title": problem.title,
            "status": problem.status,
            "priority": problem.priority,
            "tags": problem.tags or [],
            "feedback_count": problem.feedback_count,
        }

        texts.append({"content": text_content, "metadata": metadata})

    return texts
