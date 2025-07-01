"""
Comprehensive test coverage for LLM client functionality.
"""

import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "openai")


class TestLLMResponse:
    """Test the LLMResponse dataclass."""

    def test_llm_response_creation_full(self):
        """Test creating LLMResponse with all fields."""
        from llm_client import LLMResponse

        response = LLMResponse(
            content="This is a test response",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model="gpt-4o",
            provider="openai",
        )

        assert response.content == "This is a test response"
        assert response.usage["prompt_tokens"] == 10
        assert response.model == "gpt-4o"
        assert response.provider == "openai"

    def test_llm_response_creation_minimal(self):
        """Test creating LLMResponse with minimal fields."""
        from llm_client import LLMResponse

        response = LLMResponse(content="Test content")

        assert response.content == "Test content"
        assert response.usage is None
        assert response.model is None
        assert response.provider is None


class TestLLMClientFactory:
    """Test the get_llm_client factory function."""

    def test_get_llm_client_no_settings(self):
        """Test getting LLM client when settings is None returns mock."""
        with patch("llm_client.settings", None):
            from llm_client import get_llm_client

            client = get_llm_client()
            assert client is not None
            assert hasattr(client, 'generate')
            assert hasattr(client, 'embed')

    def test_get_llm_client_openai(self):
        """Test getting OpenAI LLM client."""
        mock_config = {"model": "gpt-4o", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings, \
             patch("llm_client.ChatOpenAI"), \
             patch("llm_client.OpenAIEmbeddings"):

            mock_settings.llm_provider = "openai"
            mock_settings.embedding_model = "text-embedding-ada-002"

            from llm_client import get_llm_client, OpenAIClient

            client = get_llm_client()
            assert client is not None
            assert isinstance(client, OpenAIClient)

    def test_get_llm_client_anthropic(self):
        """Test getting Anthropic LLM client."""
        mock_config = {"model": "claude-3-sonnet-20240229", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings, \
             patch("llm_client.ChatAnthropic"), \
             patch("llm_client.HuggingFaceEmbeddings"):

            mock_settings.llm_provider = "anthropic"
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            from llm_client import get_llm_client, AnthropicClient

            client = get_llm_client()
            assert client is not None
            assert isinstance(client, AnthropicClient)

    def test_get_llm_client_ollama(self):
        """Test getting Ollama LLM client."""
        mock_config = {"model": "mistral", "base_url": "http://localhost:11434"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings, \
             patch("llm_client.ChatOllama"), \
             patch("llm_client.HuggingFaceEmbeddings"):

            mock_settings.llm_provider = "ollama"
            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            from llm_client import get_llm_client, OllamaClient

            client = get_llm_client()
            assert client is not None
            assert isinstance(client, OllamaClient)

    def test_get_llm_client_unsupported(self):
        """Test getting unsupported LLM client raises error."""
        with patch("llm_client.settings") as mock_settings:
            mock_settings.llm_provider = "unsupported"

            from llm_client import get_llm_client

            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                get_llm_client()


class TestLLMClientTypes:
    """Test client class inheritance and instantiation."""

    def test_client_inheritance(self):
        """Test that client classes are subclasses of LLMClient."""
        from llm_client import LLMClient, OpenAIClient, AnthropicClient, OllamaClient

        # Test abstract base class
        assert hasattr(LLMClient, 'generate')
        assert hasattr(LLMClient, 'embed')

        # Test that concrete classes are subclasses of LLMClient
        assert issubclass(OpenAIClient, LLMClient)
        assert issubclass(AnthropicClient, LLMClient)
        assert issubclass(OllamaClient, LLMClient)

    def test_client_instantiation_openai(self):
        """Test OpenAI client instantiation."""
        mock_config = {"model": "gpt-4o", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings, \
             patch("llm_client.ChatOpenAI") as mock_openai, \
             patch("llm_client.OpenAIEmbeddings") as mock_openai_emb:

            mock_settings.embedding_model = "text-embedding-ada-002"

            from llm_client import OpenAIClient

            client = OpenAIClient()
            assert client is not None
            mock_openai.assert_called_once()
            mock_openai_emb.assert_called_once()

    def test_client_instantiation_anthropic(self):
        """Test Anthropic client instantiation."""
        mock_config = {"model": "claude-3-sonnet-20240229", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings, \
             patch("llm_client.ChatAnthropic") as mock_anthropic, \
             patch("llm_client.HuggingFaceEmbeddings") as mock_hf_emb:

            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            from llm_client import AnthropicClient

            client = AnthropicClient()
            assert client is not None
            mock_anthropic.assert_called_once()
            mock_hf_emb.assert_called_once()

    def test_client_instantiation_ollama(self):
        """Test Ollama client instantiation."""
        mock_config = {"model": "mistral", "base_url": "http://localhost:11434"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings, \
             patch("llm_client.ChatOllama") as mock_ollama, \
             patch("llm_client.HuggingFaceEmbeddings") as mock_hf_emb:

            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            from llm_client import OllamaClient

            client = OllamaClient()
            assert client is not None
            mock_ollama.assert_called_once()
            mock_hf_emb.assert_called_once()


class TestOpenAIClient:
    """Test OpenAI client functionality."""

    @pytest.mark.asyncio
    async def test_openai_generate_success(self):
        """Test successful OpenAI generation."""
        mock_config = {"model": "gpt-4o", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings:

            mock_settings.embedding_model = "text-embedding-ada-002"

            # Mock the LangChain response
            mock_response = Mock()
            mock_response.content = "This is a test response"

            mock_chat_openai = Mock()
            mock_chat_openai.ainvoke = AsyncMock(return_value=mock_response)
            mock_chat_openai.model_name = "gpt-4o"

            mock_openai_embeddings = Mock()

            with patch("llm_client.ChatOpenAI", return_value=mock_chat_openai), \
                 patch("llm_client.OpenAIEmbeddings", return_value=mock_openai_embeddings):

                from llm_client import OpenAIClient, LLMResponse

                client = OpenAIClient()

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]

                response = await client.generate(messages)

                assert isinstance(response, LLMResponse)
                assert response.content == "This is a test response"
                assert response.model == "gpt-4o"
                assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_openai_generate_error(self):
        """Test OpenAI generation error handling."""
        mock_config = {"model": "gpt-4o", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings:

            mock_settings.embedding_model = "text-embedding-ada-002"

            # Mock the LangChain components to raise an error
            mock_chat_openai = Mock()
            mock_chat_openai.ainvoke = AsyncMock(side_effect=Exception("API Error"))

            mock_openai_embeddings = Mock()

            with patch("llm_client.ChatOpenAI", return_value=mock_chat_openai), \
                 patch("llm_client.OpenAIEmbeddings", return_value=mock_openai_embeddings):

                from llm_client import OpenAIClient

                client = OpenAIClient()
                messages = [{"role": "user", "content": "Hello!"}]

                with pytest.raises(Exception, match="API Error"):
                    await client.generate(messages)

    @pytest.mark.asyncio
    async def test_openai_embed_success(self):
        """Test successful OpenAI embedding."""
        mock_config = {"model": "gpt-4o", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings:

            mock_settings.embedding_model = "text-embedding-ada-002"

            # Mock the embeddings
            mock_embeddings_result = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_openai_embeddings = Mock()
            mock_openai_embeddings.aembed_documents = AsyncMock(return_value=mock_embeddings_result)

            mock_chat_openai = Mock()

            with patch("llm_client.ChatOpenAI", return_value=mock_chat_openai), \
                 patch("llm_client.OpenAIEmbeddings", return_value=mock_openai_embeddings):

                from llm_client import OpenAIClient

                client = OpenAIClient()

                texts = ["Hello world", "Test embedding"]
                embeddings = await client.embed(texts)

                assert embeddings == mock_embeddings_result
                mock_openai_embeddings.aembed_documents.assert_called_once_with(texts)


class TestAnthropicClient:
    """Test Anthropic client functionality."""

    @pytest.mark.asyncio
    async def test_anthropic_generate_success(self):
        """Test successful Anthropic generation."""
        mock_config = {"model": "claude-3-sonnet-20240229", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings:

            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            # Mock the LangChain response
            mock_response = Mock()
            mock_response.content = "This is Claude's response"

            mock_chat_anthropic = Mock()
            mock_chat_anthropic.ainvoke = AsyncMock(return_value=mock_response)
            mock_chat_anthropic.model = "claude-3-sonnet-20240229"

            mock_hf_embeddings = Mock()

            with patch("llm_client.ChatAnthropic", return_value=mock_chat_anthropic), \
                 patch("llm_client.HuggingFaceEmbeddings", return_value=mock_hf_embeddings):

                from llm_client import AnthropicClient, LLMResponse

                client = AnthropicClient()

                messages = [
                    {"role": "system", "content": "You are Claude."},
                    {"role": "user", "content": "Hello!"}
                ]

                response = await client.generate(messages)

                assert isinstance(response, LLMResponse)
                assert response.content == "This is Claude's response"
                assert response.model == "claude-3-sonnet-20240229"
                assert response.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_anthropic_embed_success(self):
        """Test successful Anthropic embedding with HuggingFace."""
        mock_config = {"model": "claude-3-sonnet-20240229", "api_key": "test-key"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings:

            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            # Mock the embeddings
            mock_embeddings_result = [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]
            mock_hf_embeddings = Mock()
            mock_hf_embeddings.aembed_documents = AsyncMock(return_value=mock_embeddings_result)

            mock_chat_anthropic = Mock()

            with patch("llm_client.ChatAnthropic", return_value=mock_chat_anthropic), \
                 patch("llm_client.HuggingFaceEmbeddings", return_value=mock_hf_embeddings):

                from llm_client import AnthropicClient

                client = AnthropicClient()

                texts = ["Hello Claude", "Test embedding"]
                embeddings = await client.embed(texts)

                assert embeddings == mock_embeddings_result


class TestOllamaClient:
    """Test Ollama client functionality."""

    @pytest.mark.asyncio
    async def test_ollama_generate_success(self):
        """Test successful Ollama generation."""
        mock_config = {"model": "mistral", "base_url": "http://localhost:11434"}

        with patch("llm_client.get_llm_config", return_value=mock_config), \
             patch("llm_client.settings") as mock_settings:

            mock_settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            # Mock the LangChain response
            mock_response = Mock()
            mock_response.content = "This is Mistral's response"

            mock_chat_ollama = Mock()
            mock_chat_ollama.ainvoke = AsyncMock(return_value=mock_response)
            mock_chat_ollama.model = "mistral"

            mock_hf_embeddings = Mock()

            with patch("llm_client.ChatOllama", return_value=mock_chat_ollama), \
                 patch("llm_client.HuggingFaceEmbeddings", return_value=mock_hf_embeddings):

                from llm_client import OllamaClient, LLMResponse

                client = OllamaClient()

                messages = [{"role": "user", "content": "Hello Ollama!"}]

                response = await client.generate(messages)

                assert isinstance(response, LLMResponse)
                assert response.content == "This is Mistral's response"
                assert response.model == "mistral"
                assert response.provider == "ollama"


# Legacy test functions for backwards compatibility
def test_llm_response_dataclass():
    """Test LLMResponse dataclass."""
    test_instance = TestLLMResponse()
    test_instance.test_llm_response_creation_full()
    test_instance.test_llm_response_creation_minimal()
    print("✓ LLMResponse dataclass works")
    return True


def test_get_llm_client_factory():
    """Test the LLM client factory function."""
    test_instance = TestLLMClientFactory()
    test_instance.test_get_llm_client_no_settings()
    print("✓ LLM client factory works")
    return True


def test_client_types():
    """Test that client classes can be imported and instantiated."""
    test_instance = TestLLMClientTypes()
    test_instance.test_client_inheritance()
    print("✓ Client classes can be imported")
    return True


def test_client_instantiation_with_mocks():
    """Test client instantiation with mocked dependencies."""
    test_instance = TestLLMClientTypes()
    test_instance.test_client_instantiation_openai()
    test_instance.test_client_instantiation_anthropic()
    test_instance.test_client_instantiation_ollama()
    print("✓ Client instantiation works")
    return True


def test_factory_with_providers():
    """Test factory function with different providers."""
    test_instance = TestLLMClientFactory()
    test_instance.test_get_llm_client_openai()
    test_instance.test_get_llm_client_anthropic()
    test_instance.test_get_llm_client_ollama()

    try:
        test_instance.test_get_llm_client_unsupported()
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    print("✓ Factory provider selection works")
    return True


def run_llm_client_tests():
    """Run all LLM client tests."""
    print("=" * 60)
    print("Running Comprehensive LLM Client Tests")
    print("=" * 60)

    tests = [
        test_llm_response_dataclass,
        test_get_llm_client_factory,
        test_client_types,
        test_client_instantiation_with_mocks,
        test_factory_with_providers,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1

    print("=" * 60)
    print(f"LLM Client Tests: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


def run_all_tests():
    """Run all LLM client tests."""
    return run_llm_client_tests()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
