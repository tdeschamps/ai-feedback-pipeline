"""
Embedding generation and vector store operations.
"""
import logging
from dataclasses import dataclass
from typing import Any


try:
    from supabase import Client as SupabaseClient
    from supabase import create_client
except ImportError as e:
    logging.warning(f"Vector store dependencies not available: {e}")
    SupabaseClient = None  # type: ignore
    create_client = None  # type: ignore

from config import settings
from extract import Feedback
from llm_client import get_llm_client
from notion import NotionProblem


logger = logging.getLogger(__name__)

@dataclass
class EmbeddingDocument:
    """Document with embedding and metadata."""
    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    doc_type: str  # "feedback" or "problem"

class VectorStore:
    """Abstract vector store interface."""

    async def add_documents(self, documents: list[EmbeddingDocument]) -> bool:
        """Add documents to vector store."""
        raise NotImplementedError

    async def search(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents."""
        raise NotImplementedError

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete documents from vector store."""
        raise NotImplementedError

class SupabaseVectorStore(VectorStore):
    """Supabase (pgvector) implementation."""

    def __init__(self) -> None:
        if not settings.supabase_url or not settings.supabase_key:
            raise ValueError("Supabase credentials not configured")

        if create_client is None:
            raise ImportError("Supabase client not available")

        self.client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        self.table_name = "embeddings"
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Ensure the embeddings table exists."""
        # This would typically be done via SQL migration
        # For now, we'll assume the table exists with the correct schema

    async def add_documents(self, documents: list[EmbeddingDocument]) -> bool:
        """Add documents to Supabase."""
        try:
            data = []
            for doc in documents:
                data.append({
                    "id": doc.id,
                    "content": doc.content,
                    "embedding": doc.embedding,
                    "metadata": doc.metadata,
                    "doc_type": doc.doc_type
                })

            self.client.table(self.table_name).upsert(data).execute()
            logger.info(f"Added {len(documents)} documents to Supabase")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to Supabase: {e}")
            return False

    async def search(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents in Supabase."""
        try:
            # Use Supabase's vector similarity search
            # This assumes you have the pgvector extension and proper RPC function
            result = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.1,
                    "match_count": limit
                }
            ).execute()

            return result.data or []

        except Exception as e:
            logger.error(f"Error searching Supabase: {e}")
            return []

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete documents from Supabase."""
        try:
            self.client.table(self.table_name).delete().in_("id", doc_ids).execute()
            logger.info(f"Deleted {len(doc_ids)} documents from Supabase")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from Supabase: {e}")
            return False

class WeaviateVectorStore(VectorStore):
    """Weaviate implementation."""

    def __init__(self) -> None:
        if not settings.weaviate_url:
            raise ValueError("Weaviate URL not configured")

        # Use weaviate v4 API
        import weaviate
        from weaviate.auth import AuthApiKey

        auth_config = None
        if settings.weaviate_api_key:
            auth_config = AuthApiKey(api_key=settings.weaviate_api_key)

        self.client = weaviate.connect_to_custom(
            http_host=settings.weaviate_url.replace("http://", "").replace("https://", ""),
            http_port=8080,
            http_secure=settings.weaviate_url.startswith("https://"),
            grpc_host=settings.weaviate_url.replace("http://", "").replace("https://", ""),
            grpc_port=50051,
            grpc_secure=settings.weaviate_url.startswith("https://"),
            auth_credentials=auth_config
        )
        self.class_name = "Document"
        self._ensure_schema_exists()

    def _ensure_schema_exists(self) -> None:
        """Ensure Weaviate schema exists."""
        try:
            # For Weaviate v4, collections are managed differently
            # This is a simplified placeholder - in practice you'd need to
            # create collections using the new v4 API
            pass
        except Exception as e:
            logger.warning(f"Schema creation warning: {e}")

    async def add_documents(self, documents: list[EmbeddingDocument]) -> bool:
        """Add documents to Weaviate."""
        try:
            # Placeholder for Weaviate v4 API
            # In practice, you'd use the new collections API
            logger.info(f"Would add {len(documents)} documents to Weaviate")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to Weaviate: {e}")
            return False

    async def search(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents in Weaviate."""
        try:
            # Placeholder for Weaviate v4 API
            logger.info(f"Would search Weaviate with limit {limit}")
            return []

        except Exception as e:
            logger.error(f"Error searching Weaviate: {e}")
            return []

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete documents from Weaviate."""
        try:
            # Placeholder for Weaviate v4 API
            logger.info(f"Would delete {len(doc_ids)} documents from Weaviate")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from Weaviate: {e}")
            return False

class EmbeddingManager:
    """Manages embeddings and vector store operations."""

    def __init__(self) -> None:
        self.llm_client = get_llm_client()
        self.vector_store = self._get_vector_store()

    def _get_vector_store(self) -> VectorStore:
        """Get vector store based on configuration."""
        if settings.vector_store.lower() == "supabase":
            return SupabaseVectorStore()
        elif settings.vector_store.lower() == "weaviate":
            return WeaviateVectorStore()
        else:
            raise ValueError(f"Unsupported vector store: {settings.vector_store}")

    async def embed_feedbacks(self, feedbacks: list[Feedback]) -> list[EmbeddingDocument]:
        """Generate embeddings for feedback documents."""
        texts = []
        metadata_list = []

        for feedback in feedbacks:
            # Combine summary and verbatim for better semantic search
            text = f"{feedback.summary}\n\n{feedback.verbatim}"
            texts.append(text)

            metadata = {
                "type": feedback.type,
                "transcript_id": feedback.transcript_id,
                "timestamp": feedback.timestamp.isoformat(),
                "confidence": feedback.confidence,
                "context": feedback.context
            }
            metadata_list.append(metadata)

        # Generate embeddings
        embeddings = await self.llm_client.embed(texts)

        # Create embedding documents
        documents = []
        for i, feedback in enumerate(feedbacks):
            doc = EmbeddingDocument(
                id=f"feedback_{feedback.transcript_id}_{i}",
                content=texts[i],
                embedding=embeddings[i],
                metadata=metadata_list[i],
                doc_type="feedback"
            )
            documents.append(doc)

        return documents

    async def embed_problems(self, problems: list[NotionProblem]) -> list[EmbeddingDocument]:
        """Generate embeddings for Notion problems."""
        texts = []
        metadata_list = []

        for problem in problems:
            # Combine title and description
            text = f"{problem.title}\n\n{problem.description}"
            texts.append(text)

            metadata = {
                "notion_id": problem.id,
                "title": problem.title,
                "status": problem.status,
                "priority": problem.priority,
                "tags": problem.tags or [],
                "feedback_count": problem.feedback_count,
                "last_updated": problem.last_updated.isoformat() if problem.last_updated else None
            }
            metadata_list.append(metadata)

        # Generate embeddings
        embeddings = await self.llm_client.embed(texts)

        # Create embedding documents
        documents = []
        for i, problem in enumerate(problems):
            doc = EmbeddingDocument(
                id=f"problem_{problem.id}",
                content=texts[i],
                embedding=embeddings[i],
                metadata=metadata_list[i],
                doc_type="problem"
            )
            documents.append(doc)

        return documents

    async def store_embeddings(self, documents: list[EmbeddingDocument]) -> bool:
        """Store embedding documents in vector store."""
        return await self.vector_store.add_documents(documents)

    async def search_similar_problems(self, feedback_text: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search for similar problems to a feedback."""
        # Generate embedding for the feedback
        embeddings = await self.llm_client.embed([feedback_text])
        query_embedding = embeddings[0]

        # Search in vector store
        results = await self.vector_store.search(query_embedding, limit)

        # Filter to only return problems
        problem_results = [
            result for result in results
            if result.get("doc_type") == "problem" or result.get("metadata", {}).get("notion_id")
        ]

        return problem_results

    async def refresh_problem_embeddings(self, problems: list[NotionProblem]) -> bool:
        """Refresh embeddings for all problems."""
        try:
            # Delete existing problem embeddings
            existing_ids = [f"problem_{problem.id}" for problem in problems]
            await self.vector_store.delete_documents(existing_ids)

            # Generate new embeddings
            documents = await self.embed_problems(problems)

            # Store new embeddings
            return await self.store_embeddings(documents)

        except Exception as e:
            logger.error(f"Error refreshing problem embeddings: {e}")
            return False
