"""
Test coverage for LLM client functionality.
"""

import os
import sys
from unittest.mock import Mock, patch


# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")


def test_llm_response_dataclass():
    """Test LLMResponse dataclass."""
    print("Testing LLMResponse dataclass...")

    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
    }

    with patch.dict("sys.modules", mock_modules):
        import llm_client

        response = llm_client.LLMResponse(
            content="This is a test response",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model="gpt-4o",
            provider="openai",
        )

        assert response.content == "This is a test response"
        assert response.usage["prompt_tokens"] == 10
        assert response.model == "gpt-4o"
        assert response.provider == "openai"

    print("✓ LLMResponse dataclass works")


def test_get_llm_client_openai():
    """Test getting OpenAI LLM client."""
    print("Testing OpenAI LLM client...")

    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.llm_provider = "openai"
        mock_settings.openai_api_key = "test-key"

        import llm_client

        client = llm_client.get_llm_client()
        assert client is not None
        assert isinstance(client, llm_client.OpenAIClient)

    print("✓ OpenAI client creation works")


def test_get_llm_client_anthropic():
    """Test getting Anthropic LLM client."""
    print("Testing Anthropic LLM client...")

    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.llm_provider = "anthropic"
        mock_settings.anthropic_api_key = "test-key"

        import llm_client

        client = llm_client.get_llm_client()
        assert client is not None
        assert isinstance(client, llm_client.AnthropicClient)

    print("✓ Anthropic client creation works")


def test_get_llm_client_ollama():
    """Test getting Ollama LLM client."""
    print("Testing Ollama LLM client...")

    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.llm_provider = "ollama"
        mock_settings.llm_base_url = "http://localhost:11434"

        import llm_client

        client = llm_client.get_llm_client()
        assert client is not None
        assert isinstance(client, llm_client.OllamaClient)

    print("✓ Ollama client creation works")


def test_get_llm_client_unsupported():
    """Test getting unsupported LLM client raises error."""
    print("Testing unsupported LLM provider...")

    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.llm_provider = "unsupported"

        import llm_client

        try:
            llm_client.get_llm_client()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported LLM provider" in str(e)

    print("✓ Unsupported provider handling works")


def run_llm_client_tests():
    """Run all LLM client tests."""
    print("=" * 50)
    print("Running LLM Client Tests")
    print("=" * 50)

    tests = [
        test_llm_response_dataclass,
        test_get_llm_client_openai,
        test_get_llm_client_anthropic,
        test_get_llm_client_ollama,
        test_get_llm_client_unsupported,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 50)
    print(f"LLM Client Tests: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


def run_all_tests():
    """Run all LLM client tests."""
    return run_llm_client_tests()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
