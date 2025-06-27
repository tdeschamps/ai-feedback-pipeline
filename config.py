"""
Configuration management for the AI Feedback Pipeline.
"""

import logging
from typing import Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


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

    # Logging
    log_level: str = "INFO"
    log_file: str = "pipeline.log"

    model_config = {"env_file": ".env", "case_sensitive": False}


# Global settings instance
try:
    settings = Settings()
except Exception:
    # Fallback for test environments without proper config
    settings = None


def setup_logging() -> None:
    """Setup logging configuration."""
    if settings is None:
        return  # Skip logging setup in test environments

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(settings.log_file), logging.StreamHandler()],
    )


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
