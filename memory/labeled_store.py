"""
LabeledMemoryStore: Tier 2 persistent vector store for labeled past decisions.

Distinct from the legacy MemoryStore (which stored only the system's own
unlabeled decisions). This store holds LabeledMemoryEntry objects, each of
which contains the system's past decision PLUS the human ground-truth label
attached to it.

Usage in the streaming-feedback evaluation protocol:
1. Pair N is evaluated. System commits final_decision (Pass 2) without seeing
   pair N's own label.
2. After commit, pair N's ground-truth label is attached and a LabeledMemoryEntry
   is built.
3. add_labeled() stores it (with dedup check).
4. Pair N+1 may retrieve_labeled() to inform its calibration.

Key differences vs the legacy MemoryStore:
- Each entry tracks its source pair_id (for overlap detection across runs).
- The memory format includes both system_score and ground_truth_label.
- Three memory modes (cold-start, continue-stream, fresh-build) are supported
  via the runner; the store itself just persists the current state.

Storage format: identical to legacy MemoryStore on disk:
- memory_dir/labeled_memories.json
- memory_dir/labeled_embeddings.npy
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from embeddings.similarity import EmbeddingSimilarity
from models.entities import LabeledMemoryEntry


# Reuse the same constants as the legacy store — same theory applies.
DEDUP_SIMILARITY_THRESHOLD = 0.95   # Hua et al. 2025 §6.3 Selection Factor 4
MAX_RECENCY_DECAY = 0.15            # Hua et al. 2025 §6.3 Selection Factor 3


class LabeledMemoryStore:
    """
    Persistent store of LabeledMemoryEntry objects with pair_id-aware overlap detection.

    Designed for the streaming-feedback evaluation protocol. The runner is responsible
    for the higher-level semantics (cold-start clear, continue-stream overlap check);
    the store itself just persists state and offers retrieval.
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
        self._entries: list[LabeledMemoryEntry] = []
        self._embeddings: np.ndarray | None = None  # shape: (n, dim)

        # Auto-load if files exist on disk
        self._load()

    # ── Public API ──────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of labeled entries currently in memory."""
        return len(self._entries)

    def pair_ids(self) -> set[str]:
        """
        Return the set of pair_ids currently in memory.

        Used by the runner's continue-stream mode to detect overlap with the
        input dataset. If any input pair_id is already in memory, re-evaluating
        that pair would leak its known label — so the runner refuses.
        """
        return {e.pair_id for e in self._entries if e.pair_id}

    def add_labeled(self, entry: LabeledMemoryEntry) -> bool:
        """
        Add a labeled memory entry to the store.

        Returns True if added, False if skipped as a near-duplicate (cosine
        similarity to an existing entry above DEDUP_SIMILARITY_THRESHOLD).

        The caller is responsible for ensuring entry.system_score was committed
        BEFORE entry.ground_truth_label was attached (the streaming protocol
        invariant). This method does not enforce that — it just persists.
        """
        text = self._entry_to_text(entry)
        embedding = self._embedding_sim.encode([text])  # shape (1, dim)

        # Dedup against existing
        if self._embeddings is not None and len(self._embeddings) > 0:
            sims = cosine_similarity(embedding, self._embeddings)[0]
            if float(sims.max()) >= DEDUP_SIMILARITY_THRESHOLD:
                return False

        self._entries.append(entry)
        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = embedding
        else:
            self._embeddings = np.vstack([self._embeddings, embedding])
        return True

    def retrieve_labeled(
        self,
        query_text: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[LabeledMemoryEntry]:
        """
        Retrieve top-K most similar labeled past entries.

        Applies recency weighting (older entries lose up to MAX_RECENCY_DECAY
        of their score) but reports raw similarity in similarity_to_current.
        """
        if self.count == 0 or self._embeddings is None:
            return []

        k = top_k if top_k is not None else self._top_k
        threshold = min_similarity if min_similarity is not None else self._min_similarity

        query_embedding = self._embedding_sim.encode([query_text])  # (1, dim)
        raw_sims = cosine_similarity(query_embedding, self._embeddings)[0]

        # Recency weighting
        n = len(self._entries)
        if n > 1:
            recency_factors = 1.0 - MAX_RECENCY_DECAY * (1.0 - np.arange(n) / (n - 1))
        else:
            recency_factors = np.ones(n)
        weighted_sims = raw_sims * recency_factors

        # Sort by weighted similarity, filter by raw similarity threshold
        sorted_indices = np.argsort(weighted_sims)[::-1]
        results: list[LabeledMemoryEntry] = []
        for idx in sorted_indices:
            raw_sim = float(raw_sims[idx])
            if raw_sim < threshold:
                continue
            if len(results) >= k:
                break
            entry = self._entries[idx].model_copy()
            entry.similarity_to_current = round(raw_sim, 4)
            results.append(entry)
        return results

    def save(self) -> None:
        """Persist all entries + embeddings to disk."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        entries_path = self._memory_dir / "labeled_memories.json"
        entries_data = [e.model_dump() for e in self._entries]
        entries_path.write_text(
            json.dumps(entries_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if self._embeddings is not None and len(self._embeddings) > 0:
            embeddings_path = self._memory_dir / "labeled_embeddings.npy"
            np.save(str(embeddings_path), self._embeddings)

    def clear(self) -> None:
        """Wipe in-memory state. Caller must save() to persist the empty state."""
        self._entries = []
        self._embeddings = None

    def clear_files(self) -> None:
        """
        Remove labeled memory files from disk.

        Used by the runner's cold-start mode at the beginning of an evaluation
        to ensure no carry-over from previous runs. Safe to call when files
        don't exist (no-op).
        """
        self.clear()
        for fname in ("labeled_memories.json", "labeled_embeddings.npy"):
            path = self._memory_dir / fname
            if path.exists():
                path.unlink()

    # ── Internal helpers ─────────────────────────────────────────

    def _load(self) -> None:
        """Load entries + embeddings from disk if they exist."""
        entries_path = self._memory_dir / "labeled_memories.json"
        embeddings_path = self._memory_dir / "labeled_embeddings.npy"

        if entries_path.exists():
            data = json.loads(entries_path.read_text(encoding="utf-8"))
            self._entries = [LabeledMemoryEntry.model_validate(e) for e in data]
        if embeddings_path.exists():
            self._embeddings = np.load(str(embeddings_path))

        # Sanity check: counts must match. If not, reset to avoid stale data.
        if self._embeddings is not None and len(self._entries) != len(self._embeddings):
            self._entries = []
            self._embeddings = None

    def _entry_to_text(self, entry: LabeledMemoryEntry) -> str:
        """
        Build the text representation that gets embedded for retrieval.

        Combines profile summaries + role + reasoning to produce a representation
        that captures both the semantic content of the pair AND the system's
        interpretation. Ground-truth fields are NOT in the embedding (those are
        for retrieval-time interpretation, not similarity matching).
        """
        parts = []
        if entry.cv_profile_summary:
            parts.append(f"CV: {entry.cv_profile_summary}")
        if entry.jd_profile_summary:
            parts.append(f"JD: {entry.jd_profile_summary}")
        if entry.detected_role:
            parts.append(f"Role: {entry.detected_role}")
        if entry.system_reasoning_summary:
            parts.append(f"Reasoning: {entry.system_reasoning_summary}")
        return " | ".join(parts)
