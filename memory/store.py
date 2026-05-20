"""
MemoryStore: persistent vector store for past matching decisions.

This is the storage backend for Scenario C. It stores completed matching
results (summaries, scores, reasoning) and retrieves the most relevant
past decisions for a new CV-JD pair using embedding similarity.

Architecture:
  - Each memory is a MemoryEntry (pydantic model) stored as JSON
  - Each memory gets a sentence embedding computed from a combined
    text representation (cv_summary + jd_summary + reasoning_summary)
  - Retrieval computes cosine similarity between the query embedding
    and all stored embeddings, returning the top-k matches

Storage format:
  memory_dir/
    memories.json    -- list of MemoryEntry dicts
    embeddings.npy   -- numpy array of shape (n_memories, embedding_dim)

Design decisions:
  - Uses numpy cosine similarity instead of FAISS for MVP simplicity
    (FAISS can be swapped in later for production scale)
  - Embeddings are reused from EmbeddingSimilarity (same model, no extra load)
  - Store is append-only for simplicity (no update/delete needed for thesis)
  - Thread-safe is not needed (single-process pipeline)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from embeddings.similarity import EmbeddingSimilarity
from models.entities import MemoryEntry


# Context Engineering 2.0: Selection Factor 4 (Overlapping Information).
# If a new memory is more than this similar to an existing one, skip adding
# to avoid redundant entries cluttering the memory pool (Hua et al., 2025 §6.3).
DEDUP_SIMILARITY_THRESHOLD = 0.95

# Context Engineering 2.0: Selection Factor 3 (Recency).
# Maximum recency penalty. Oldest memories retain at least (1 - MAX_DECAY)
# of their similarity score; newest memories keep 100%. Decay is linear
# across the position of the memory in the store.
MAX_RECENCY_DECAY = 0.15


class MemoryStore:
    """
    Persistent vector store for past matching decisions.

    Stores MemoryEntry objects alongside their embeddings, and
    supports retrieval of the most similar past decisions for
    a new query.

    Usage:
        store = MemoryStore()
        store.add(memory_entry)       # After a matching run
        store.save()                  # Persist to disk

        results = store.retrieve(     # Before a new run
            query_text="Python dev with ML experience...",
            top_k=3,
        )
    """

    def __init__(
        self,
        memory_dir: str | None = None,
        embedding_similarity: EmbeddingSimilarity | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ):
        self._memory_dir = Path(memory_dir or settings.memory_dir)
        self._embedding_sim = embedding_similarity or EmbeddingSimilarity()
        self._top_k = top_k if top_k is not None else settings.memory_top_k
        self._min_similarity = (
            min_similarity if min_similarity is not None
            else settings.memory_min_similarity
        )

        # In-memory state
        self._memories: list[MemoryEntry] = []
        self._embeddings: np.ndarray | None = None  # shape: (n, dim)

        # Load existing memories if they exist on disk
        self._load()

    # ── Public API ──────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of memories in the store."""
        return len(self._memories)

    def add(self, memory: MemoryEntry) -> bool:
        """
        Add a new memory to the store (in-memory only).

        Call save() to persist to disk.

        The memory's text representation is embedded and stored
        alongside the structured data for later retrieval.

        Context Engineering 2.0: deduplication check. If the new memory
        is near-duplicate of an existing one (cosine similarity above
        DEDUP_SIMILARITY_THRESHOLD), skip adding it. This prevents the
        memory pool from filling with redundant entries.

        Returns:
            True if the memory was added, False if skipped as duplicate.
        """
        # Compute embedding for this memory's text representation
        text = self._memory_to_text(memory)
        embedding = self._embedding_sim.encode([text])  # shape: (1, dim)

        # Deduplication: skip if too similar to an existing memory
        if self._embeddings is not None and len(self._embeddings) > 0:
            sims = cosine_similarity(embedding, self._embeddings)[0]
            max_sim = float(sims.max())
            if max_sim >= DEDUP_SIMILARITY_THRESHOLD:
                return False  # duplicate, skip

        self._memories.append(memory)

        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = embedding
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])

        return True

    def retrieve(
        self,
        query_text: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[MemoryEntry]:
        """
        Retrieve the most similar past decisions for a query.

        Args:
            query_text: Combined text describing the current CV+JD
            top_k: Maximum number of results (defaults to settings)
            min_similarity: Minimum similarity threshold (defaults to settings)

        Returns:
            List of MemoryEntry objects, sorted by similarity (descending).
            Each entry has similarity_to_current populated.
        """
        if self.count == 0 or self._embeddings is None:
            return []

        k = top_k if top_k is not None else self._top_k
        threshold = min_similarity if min_similarity is not None else self._min_similarity

        # Encode the query
        query_embedding = self._embedding_sim.encode([query_text])  # (1, dim)

        # Compute cosine similarity against all stored embeddings
        raw_similarities = cosine_similarity(query_embedding, self._embeddings)[0]  # (n,)

        # Context Engineering 2.0 Selection Factor 3: Recency weighting.
        # Older memories are slightly down-weighted to favor recent decisions.
        # Uses position-based decay: the oldest memory gets (1 - MAX_RECENCY_DECAY),
        # the newest gets 1.0, linear in between.
        n = len(self._memories)
        if n > 1:
            # position 0 (oldest) -> factor = 1 - MAX_RECENCY_DECAY
            # position n-1 (newest) -> factor = 1.0
            recency_factors = 1.0 - MAX_RECENCY_DECAY * (
                1.0 - np.arange(n) / (n - 1)
            )
        else:
            recency_factors = np.ones(n)

        weighted_similarities = raw_similarities * recency_factors

        # Get indices sorted by weighted similarity (descending)
        sorted_indices = np.argsort(weighted_similarities)[::-1]

        # Filter by threshold (applied to raw similarity) and limit to top_k
        results: list[MemoryEntry] = []
        for idx in sorted_indices:
            raw_sim = float(raw_similarities[idx])
            if raw_sim < threshold:
                continue  # weighted sort can put low-sim ahead; check raw
            if len(results) >= k:
                break

            # Create a copy with similarity_to_current populated.
            # Report the raw similarity (not weighted) — downstream code
            # uses this for logging and memory impact analysis.
            memory = self._memories[idx].model_copy()
            memory.similarity_to_current = round(raw_sim, 4)
            results.append(memory)

        return results

    def save(self) -> None:
        """
        Persist memories and embeddings to disk.

        Creates the memory directory if it doesn't exist.
        Overwrites existing files (full dump, not incremental).
        """
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # Save memories as JSON
        memories_path = self._memory_dir / "memories.json"
        memories_data = [m.model_dump() for m in self._memories]
        memories_path.write_text(
            json.dumps(memories_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Save embeddings as numpy array
        if self._embeddings is not None and len(self._embeddings) > 0:
            embeddings_path = self._memory_dir / "embeddings.npy"
            np.save(str(embeddings_path), self._embeddings)

    def clear(self) -> None:
        """Remove all memories (in-memory only; call save() to persist)."""
        self._memories = []
        self._embeddings = None

    # ── Internal helpers ─────────────────────────────────────────

    def _load(self) -> None:
        """Load memories and embeddings from disk if they exist."""
        memories_path = self._memory_dir / "memories.json"
        embeddings_path = self._memory_dir / "embeddings.npy"

        if memories_path.exists():
            data = json.loads(memories_path.read_text(encoding="utf-8"))
            self._memories = [MemoryEntry.model_validate(m) for m in data]

        if embeddings_path.exists():
            self._embeddings = np.load(str(embeddings_path))

        # Validate consistency
        if self._embeddings is not None and len(self._memories) != len(self._embeddings):
            # Data is inconsistent — reset to avoid errors
            self._memories = []
            self._embeddings = None

    def _memory_to_text(self, memory: MemoryEntry) -> str:
        """
        Convert a MemoryEntry to a text string for embedding.

        Combines the CV summary, JD summary, and reasoning summary
        into a single text representation. This is what gets embedded
        and compared during retrieval.
        """
        parts = []
        if memory.cv_summary:
            parts.append(f"CV: {memory.cv_summary}")
        if memory.jd_summary:
            parts.append(f"JD: {memory.jd_summary}")
        if memory.reasoning_summary:
            parts.append(f"Reasoning: {memory.reasoning_summary}")
        parts.append(f"Score: {memory.decision_score}")
        return " | ".join(parts)
