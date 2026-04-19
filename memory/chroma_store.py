"""
LifeOS Agent - ChromaDB Vector Store
======================================
Persistent vector store for storing user messages and agent responses.
Supports semantic search, user-scoped collections, and auto-cleanup.
"""

import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Default storage directory — uses persistent volume on Railway
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db"))

# ═══════════════════════════════════════════
# CHROMA STORE CLASS
# ═══════════════════════════════════════════


class ChromaStore:
    """
    ChromaDB-based vector store for semantic storage and retrieval.
    Each user gets their own collection for data isolation.
    """

    MAX_ENTRIES_PER_USER = 100

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or CHROMA_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self._client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB persistent client."""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(f"ChromaDB initialized at {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._client = None

    def _get_collection(self, user_id: str):
        """Get or create a collection for the given user."""
        if self._client is None:
            logger.error("ChromaDB client not available")
            return None

        try:
            collection_name = f"user_{str(user_id).replace('-', '_')[:50]}"
            # ChromaDB collection names must be 3-63 chars, starts/ends with alphanumeric
            if len(collection_name) < 3:
                collection_name = f"user_{collection_name}_col"
            collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"user_id": str(user_id)},
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to get collection for user {user_id}: {e}")
            return None

    def store(self, user_id: str, text: str, metadata: Optional[dict] = None) -> bool:
        """
        Store a text entry in the user's vector collection.

        Args:
            user_id: Unique user identifier.
            text: Text content to store.
            metadata: Optional metadata dict.

        Returns:
            True if stored successfully.
        """
        try:
            collection = self._get_collection(user_id)
            if collection is None:
                return False

            # Generate a unique document ID
            doc_id = f"doc_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

            # Build metadata
            meta = {
                "user_id": str(user_id),
                "timestamp": datetime.now().isoformat(),
                "type": "message",
            }
            if metadata:
                meta.update({k: str(v) for k, v in metadata.items()})

            collection.add(
                documents=[text],
                metadatas=[meta],
                ids=[doc_id],
            )

            # Auto-cleanup if over limit
            self._auto_cleanup(user_id, collection)

            logger.info(f"Stored entry for user {user_id}: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store entry for user {user_id}: {e}")
            return False

    def query(self, user_id: str, query_text: str, n_results: int = 5) -> list:
        """
        Semantic search for relevant past entries.

        Args:
            user_id: Unique user identifier.
            query_text: The query string to search for.
            n_results: Maximum number of results to return.

        Returns:
            List of dicts with 'document', 'metadata', and 'distance'.
        """
        try:
            collection = self._get_collection(user_id)
            if collection is None:
                return []

            # Check if collection is empty
            count = collection.count()
            if count == 0:
                return []

            actual_n = min(n_results, count)
            results = collection.query(
                query_texts=[query_text],
                n_results=actual_n,
            )

            entries = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                entries.append(
                    {
                        "document": doc,
                        "metadata": meta,
                        "distance": dist,
                    }
                )

            logger.info(
                f"Query for user {user_id} returned {len(entries)} results"
            )
            return entries

        except Exception as e:
            logger.error(f"Failed to query for user {user_id}: {e}")
            return []

    def get_recent(self, user_id: str, limit: int = 10) -> list:
        """
        Get the most recent stored entries for a user.

        Args:
            user_id: Unique user identifier.
            limit: Maximum number of entries to return.

        Returns:
            List of document strings.
        """
        try:
            collection = self._get_collection(user_id)
            if collection is None:
                return []

            count = collection.count()
            if count == 0:
                return []

            # Get all entries and sort by timestamp
            all_data = collection.get(
                limit=min(limit, count),
                include=["documents", "metadatas"],
            )

            documents = all_data.get("documents", [])
            return documents

        except Exception as e:
            logger.error(f"Failed to get recent entries for user {user_id}: {e}")
            return []

    def _auto_cleanup(self, user_id: str, collection) -> None:
        """Remove oldest entries if over the max per-user limit."""
        try:
            count = collection.count()
            if count <= self.MAX_ENTRIES_PER_USER:
                return

            # Get all entries to find oldest ones
            all_data = collection.get(include=["metadatas"])
            ids = all_data.get("ids", [])
            metadatas = all_data.get("metadatas", [])

            # Sort by timestamp and remove oldest
            entries = list(zip(ids, metadatas))
            entries.sort(key=lambda x: x[1].get("timestamp", ""), reverse=False)

            # Remove excess entries (keep newest MAX_ENTRIES_PER_USER)
            to_remove = len(entries) - self.MAX_ENTRIES_PER_USER
            if to_remove > 0:
                ids_to_remove = [entry[0] for entry in entries[:to_remove]]
                collection.delete(ids=ids_to_remove)
                logger.info(
                    f"Auto-cleanup: removed {to_remove} old entries for user {user_id}"
                )

        except Exception as e:
            logger.error(f"Auto-cleanup failed for user {user_id}: {e}")

    def clear_user_data(self, user_id: str) -> bool:
        """Delete all stored data for a user."""
        try:
            if self._client is None:
                return False

            collection_name = f"user_{str(user_id).replace('-', '_')[:50]}"
            if len(collection_name) < 3:
                collection_name = f"user_{collection_name}_col"

            try:
                self._client.delete_collection(name=collection_name)
                logger.info(f"Cleared all data for user {user_id}")
            except Exception:
                logger.info(f"No collection to delete for user {user_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to clear data for user {user_id}: {e}")
            return False

    def get_context_for_query(self, user_id: str, query: str, max_context: int = 3) -> str:
        """
        Get formatted context string from past entries relevant to a query.

        Args:
            user_id: Unique user identifier.
            query: The current query to find context for.
            max_context: Max number of context entries to include.

        Returns:
            Formatted context string.
        """
        try:
            results = self.query(user_id, query, n_results=max_context)
            if not results:
                return "No relevant past context found."

            context_parts = ["📂 **Relevant Past Context:**\n"]
            for idx, entry in enumerate(results, 1):
                doc = entry["document"][:300]  # Truncate long documents
                timestamp = entry["metadata"].get("timestamp", "N/A")[:10]
                context_parts.append(f"{idx}. [{timestamp}] {doc}")

            return "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Failed to get context for user {user_id}: {e}")
            return "Context retrieval failed."


# ═══════════════════════════════════════════
# GLOBAL SINGLETON INSTANCE
# ═══════════════════════════════════════════

_chroma_instance: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """Get or create the global ChromaStore singleton."""
    global _chroma_instance
    if _chroma_instance is None:
        _chroma_instance = ChromaStore()
    return _chroma_instance
