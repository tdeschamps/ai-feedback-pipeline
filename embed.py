"""
Embedding generation and vector store operations.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock


try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError as e:
    logging.warning(f"ChromaDB not available: {e}")
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore

try:
    import pinecone
    from pinecone import Pinecone
except ImportError as e:
    logging.warning(f"Pinecone not available: {e}")
    pinecone = None  # type: ignore
    Pinecone = None  # type: ignore

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


class ChromaDBVectorStore(VectorStore):
    """ChromaDB implementation for local vector storage."""

    def __init__(self) -> None:
        if chromadb is None:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")

        # Set up ChromaDB client
        if settings.chromadb_host == "localhost" and settings.chromadb_port == 8000:
            # Use persistent local storage
            client_settings = None
            if ChromaSettings is not None:
                client_settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)

            self.client = chromadb.PersistentClient(
                path=settings.chromadb_persist_directory, settings=client_settings
            )
        else:
            # Use HTTP client for remote ChromaDB
            client_settings = None
            if ChromaSettings is not None:
                client_settings = ChromaSettings(anonymized_telemetry=False)

            self.client = chromadb.HttpClient(
                host=settings.chromadb_host, port=settings.chromadb_port, settings=client_settings
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.chromadb_collection_name, metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, documents: list[EmbeddingDocument]) -> bool:
        """Add documents to ChromaDB."""
        try:
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []

            for doc in documents:
                ids.append(doc.id)
                embeddings.append(doc.embedding)

                # ChromaDB metadata must be JSON serializable
                metadata = {
                    "doc_type": doc.doc_type,
                    "content": doc.content[:1000],  # Truncate for metadata
                    **{
                        k: str(v) if not isinstance(v, str | int | float | bool) else v
                        for k, v in doc.metadata.items()
                    },
                }
                metadatas.append(metadata)
                documents_content.append(doc.content)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,  # type: ignore
                metadatas=metadatas,  # type: ignore
                documents=documents_content,
            )

            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            return False

    async def search(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents in ChromaDB."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],  # type: ignore
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            formatted_results = []
            if results["ids"] and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": 1 - (results["distances"][0][i] if results["distances"] else 0),  # type: ignore
                        "doc_type": (
                            results["metadatas"][0][i].get("doc_type", "")  # type: ignore
                            if results["metadatas"]
                            else ""
                        ),
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete documents from ChromaDB."""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            return False


class PineconeVectorStore(VectorStore):
    """Pinecone implementation for cloud vector storage."""

    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise ValueError("Pinecone API key not configured")

        if Pinecone is None:
            raise ImportError("Pinecone not available. Install with: pip install pinecone-client")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

        # Get or create index
        self.index_name = settings.pinecone_index_name
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self) -> None:
        """Ensure Pinecone index exists."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                # Create index with appropriate dimensions
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.embedding_dimensions,
                    metric="cosine",
                    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Error managing Pinecone index: {e}")
            raise

    async def add_documents(self, documents: list[EmbeddingDocument]) -> bool:
        """Add documents to Pinecone."""
        try:
            vectors = []
            for doc in documents:
                vector = {
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": {
                        "doc_type": doc.doc_type,
                        "content": doc.content[:40000],  # Pinecone metadata limit
                        **{
                            k: str(v) if not isinstance(v, str | int | float | bool) else v
                            for k, v in doc.metadata.items()
                        },
                    },
                }
                vectors.append(vector)

            # Upsert in batches to avoid size limits
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)

            logger.info(f"Added {len(documents)} documents to Pinecone")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}")
            return False

    async def search(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Search for similar documents in Pinecone."""
        try:
            response = self.index.query(
                vector=query_embedding, top_k=limit, include_metadata=True, include_values=False
            )

            formatted_results = []
            for match in response.matches:
                result = {
                    "id": match.id,
                    "content": match.metadata.get("content", ""),
                    "metadata": match.metadata,
                    "score": match.score,
                    "doc_type": match.metadata.get("doc_type", ""),
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching Pinecone: {e}")
            return []

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete documents from Pinecone."""
        try:
            self.index.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from Pinecone")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from Pinecone: {e}")
            return False


class EmbeddingManager:
    """Manages embeddings and vector store operations."""

    def __init__(self) -> None:
        self.llm_client = get_llm_client()
        self.vector_store = self._get_vector_store()

    def _get_vector_store(self) -> VectorStore:
        """Get vector store based on configuration."""
        if settings is None:
            # Return a mock vector store for test environments
            return Mock(spec=VectorStore)

        if settings.vector_store.lower() == "chromadb":
            return ChromaDBVectorStore()
        elif settings.vector_store.lower() == "pinecone":
            return PineconeVectorStore()
        else:
            raise ValueError(
                f"Unsupported vector store: {settings.vector_store}. Use 'chromadb' or 'pinecone'."
            )

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
                "context": feedback.context or "",
            }
            metadata_list.append(metadata)

        # Generate embeddings
        embeddings = await self.llm_client.embed(texts)

        # Create embedding documents
        documents = []
        for i, feedback in enumerate(feedbacks):
            doc = EmbeddingDocument(
                id=f"feedback_{feedback.transcript_id}_{i}_{uuid.uuid4().hex[:8]}",
                content=texts[i],
                embedding=embeddings[i],
                metadata=metadata_list[i],
                doc_type="feedback",
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
                "status": problem.status or "",
                "priority": problem.priority or "",
                "tags": ",".join(problem.tags) if problem.tags else "",
                "feedback_count": problem.feedback_count,
                "last_updated": problem.last_updated.isoformat() if problem.last_updated else "",
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
                doc_type="problem",
            )
            documents.append(doc)

        return documents

    async def store_embeddings(self, documents: list[EmbeddingDocument]) -> bool:
        """Store embedding documents in vector store."""
        return await self.vector_store.add_documents(documents)

    async def search_similar_problems(
        self, feedback_text: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search for similar problems to a feedback."""
        # Generate embedding for the feedback
        embeddings = await self.llm_client.embed([feedback_text])
        query_embedding = embeddings[0]

        # Search in vector store
        results = await self.vector_store.search(query_embedding, limit)

        # Filter to only return problems
        problem_results = [
            result
            for result in results
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
