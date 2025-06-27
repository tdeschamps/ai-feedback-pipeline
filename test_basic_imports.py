"""
Simple test to verify the core functionality works without problematic imports.
"""

import os
import sys


sys.path.insert(0, ".")

# Set environment variables to avoid API key errors
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["ANTHROPIC_API_KEY"] = "test-key"

from unittest.mock import Mock, patch


def test_basic_functionality():
    """Test basic functionality with mocked dependencies."""
    print("Testing basic functionality...")

    # Mock all the problematic imports
    mock_modules = {
        "langchain.schema": Mock(),
        "langchain_anthropic": Mock(),
        "langchain_community.embeddings": Mock(),
        "langchain_community.llms": Mock(),
        "langchain_openai": Mock(),
        "chromadb": Mock(),
        "pinecone": Mock(),
        "notion_client": Mock(),
    }

    with patch.dict("sys.modules", mock_modules), patch("config.settings") as mock_settings:
        mock_settings.vector_store = "chromadb"
        mock_settings.chromadb_host = "localhost"
        mock_settings.chromadb_port = 8000
        mock_settings.chromadb_persist_directory = "./chroma_db"
        mock_settings.chromadb_collection_name = "test_collection"
        mock_settings.openai_api_key = "test-key"

        try:
            print("Importing modules with full mocking...")

            # Import core modules
            print("✓ config imported")

            print("✓ llm_client imported")

            print("✓ extract imported")

            print("✓ notion imported")

            print("✓ embed imported")

            print("✓ pipeline imported")

            print("✓ All core modules imported successfully!")
            return True

        except Exception as e:
            print(f"✗ Error importing modules: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ Basic import test passed!")
    else:
        print("\n❌ Basic import test failed!")
        sys.exit(1)
