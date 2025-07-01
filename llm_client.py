"""
Abstract LLM client interface supporting multiple providers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock


try:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from pydantic import SecretStr
except ImportError as e:
    logging.warning(f"Some dependencies not available: {e}")
    # Set defaults for missing imports
    SecretStr = str  # type: ignore

from config import get_llm_config, settings


logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM with metadata."""

    content: str
    usage: dict[str, Any] | None = None
    model: str | None = None
    provider: str | None = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Generate response from messages."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""


class OpenAIClient(LLMClient):
    """OpenAI LLM client."""

    def __init__(self) -> None:
        config = get_llm_config()
        self.llm = ChatOpenAI(
            model=config.get("model", "gpt-4o"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            temperature=0.1,
        )
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, api_key=config.get("api_key")
        )

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Generate response using OpenAI."""
        try:
            # Convert messages to LangChain format
            lc_messages: list[SystemMessage | HumanMessage] = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))

            response = await self.llm.ainvoke(lc_messages)
            return LLMResponse(
                content=str(response.content), model=self.llm.model_name, provider="openai"
            )
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI."""
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise


class AnthropicClient(LLMClient):
    """Anthropic (Claude) LLM client."""

    def __init__(self) -> None:
        config = get_llm_config()

        self.llm = ChatAnthropic(
            model_name=config.get("model", "claude-3-sonnet-20240229"),
            api_key=config.get("api_key", ""),
            temperature=0.1,
            timeout=30,  # type: ignore
            stop=None,  # type: ignore
        )
        # Use HuggingFace embeddings as fallback
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Generate response using Anthropic."""
        try:
            lc_messages: list[SystemMessage | HumanMessage] = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))

            response = await self.llm.ainvoke(lc_messages)
            return LLMResponse(
                content=str(response.content), model=self.llm.model, provider="anthropic"
            )
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using HuggingFace."""
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise


class OllamaClient(LLMClient):
    """Ollama local LLM client."""

    def __init__(self) -> None:
        config = get_llm_config()
        self.llm = ChatOllama(
            model=config.get("model", "mistral"),
            base_url=config.get("base_url", "http://localhost:11434"),
            temperature=0.1,
            extract_reasoning=True,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Generate response using Ollama."""
        try:
            # Convert messages to LangChain format
            lc_messages: list[SystemMessage | HumanMessage] = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))

            response = await self.llm.ainvoke(lc_messages)
            return LLMResponse(content=response.content, model=self.llm.model, provider="ollama")
        except Exception as fallback_e:
            logger.error(f"Ollama fallback generation error: {fallback_e}")
            raise fallback_e  # Raise the original error with context

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using HuggingFace."""
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise


def get_llm_client() -> LLMClient:
    """Factory function to get LLM client based on configuration."""
    if settings is None:
        # Return a mock client for test environments
        mock_client = Mock()
        mock_client.generate = Mock(return_value="mocked response")
        mock_client.embed = Mock(return_value=[0.1, 0.2, 0.3])
        return mock_client  # type: ignore

    provider = settings.llm_provider.lower()

    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
    }

    if provider not in clients:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    logger.info(f"Initializing {provider} LLM client")
    return clients[provider]()  # type: ignore
