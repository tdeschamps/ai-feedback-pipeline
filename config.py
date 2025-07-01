"""
Configuration management for the AI Feedback Pipeline.
"""

import logging
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record: logging.LogRecord) -> str:
        # Get the original formatted message
        message = super().format(record)

        # Add color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]

        # Color the entire message
        return f"{color}{message}{reset}"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_base_url: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_key: str | None = None
    huggingface_api_token: str | None = None

    # Embedding Configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Vector Store Configuration
    vector_store: str = "chromadb"
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    chromadb_collection_name: str = "feedback_embeddings"
    chromadb_persist_directory: str = "./chroma_db"
    pinecone_api_key: str | None = None
    pinecone_environment: str | None = None
    pinecone_index_name: str = "feedback-pipeline"

    # Notion Configuration
    notion_api_key: str | None = None
    notion_database_id: str | None = None

    # Pipeline Configuration
    confidence_threshold: float = 0.7
    max_matches: int = 5
    rerank_enabled: bool = True

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "pipeline.log"
    suppress_import_warnings: bool = False

    model_config = {"env_file": ".env", "case_sensitive": False}


# Global settings instance
try:
    settings = Settings()
except Exception:
    # Fallback for test environments without proper config
    settings = None  # type: ignore


def setup_logging() -> None:
    """Setup logging configuration with colored console output."""
    if settings is None:
        return  # Skip logging setup in test environments

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create handlers
    file_handler = logging.FileHandler(settings.log_file)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Clear any existing handlers and add our custom ones
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_llm_config() -> dict[str, Any]:
    """Get LLM configuration based on provider."""
    if settings is None:
        return {}  # Return empty config in test environments

    configs: dict[str, dict[str, Any]] = {
        "openai": {
            "api_key": settings.openai_api_key,
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
        },
        "anthropic": {"api_key": settings.anthropic_api_key, "model": settings.llm_model},
        "groq": {"api_key": settings.groq_api_key, "model": settings.llm_model},
        "ollama": {
            "base_url": settings.llm_base_url or "http://localhost:11434",
            "model": settings.llm_model,
        },
        "huggingface": {"api_token": settings.huggingface_api_token, "model": settings.llm_model},
    }
    return configs.get(settings.llm_provider, {})
