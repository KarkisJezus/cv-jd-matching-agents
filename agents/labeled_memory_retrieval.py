"""
LabeledMemoryRetrievalAgent: Tier 2 replacement for the legacy MemoryRetrievalAgent.

Retrieves the top-K most similar LabeledMemoryEntry objects from the
LabeledMemoryStore. Runs in Pass 2 of the new architecture (Scenario C only),
between the DecisionAgent's initial_decision commit and the CalibrationAgent.

Key differences vs the legacy MemoryRetrievalAgent:
- Reads from LabeledMemoryStore (entries include human ground-truth labels).
- Writes to context.labeled_memory_entries (not legacy memory_entries).
- Builds the retrieval query from the cv_profile + jd_profile (richer than raw text).

Pure embedding-based retrieval — no LLM call. Fast and deterministic.
"""

from agents.base import BaseAgent
from memory.labeled_store import LabeledMemoryStore
from models.shared_context import SharedContext


# Tier 2 retrieves more memories than Tier 1's default top_k=3.
# CalibrationAgent needs enough evidence to detect patterns — 5 retrieved
# memories with >=3 supporting required gives meaningful pattern detection.
# Higher than 5 increases prompt length without much benefit per the pilot.
LABELED_TOP_K = 5


class LabeledMemoryRetrievalAgent(BaseAgent):
    """
    Retrieves labeled past pairs from the LabeledMemoryStore.

    Reads:  context.cv_profile, context.jd_profile (or falls back to raw text)
    Writes: context.labeled_memory_entries
    """

    def __init__(self, store: LabeledMemoryStore, top_k: int = LABELED_TOP_K):
        self._store = store
        self._top_k = top_k

    def process(self, context: SharedContext) -> SharedContext:
        # If the store is empty (cold-start, pair 1), skip silently.
        if self._store.count == 0:
            context.add_log(
                self.name,
                "memory_empty",
                "No labeled memories available (cold start). Calibration will skip.",
            )
            return context

        query = self._build_query(context)

        context.add_log(
            self.name,
            "retrieval_started",
            f"Searching {self._store.count} labeled memories",
        )

        results = self._store.retrieve_labeled(query, top_k=self._top_k)

        if results:
            context.labeled_memory_entries = results
            best = results[0]
            context.add_log(
                self.name,
                "retrieval_completed",
                f"Retrieved {len(results)} labeled memories. "
                f"Best similarity={best.similarity_to_current:.3f}, "
                f"top match's role={best.detected_role}, "
                f"top match's outcome={'accepted' if best.ground_truth_label else 'rejected'}",
            )
        else:
            context.add_log(
                self.name,
                "retrieval_no_matches",
                "No labeled memories above similarity threshold",
            )

        return context

    def _build_query(self, context: SharedContext) -> str:
        """
        Build the retrieval query from the available context.

        Prefers profiles (richer signal) but falls back to raw text if profiles
        haven't been built yet. The query is what gets embedded and compared
        against stored memory embeddings.
        """
        # Tier 2 path: use profiles
        if context.cv_profile and context.jd_profile:
            cv_text = (
                context.cv_profile.candidate_archetype
                or context.cv_profile.raw_summary
                or " ".join(context.cv_profile.skills[:10])
            )
            jd_text = (
                context.jd_profile.raw_summary
                or " ".join(context.jd_profile.required_skills[:10])
            )
            role_hint = context.jd_profile.detected_role
            return f"CV: {cv_text} | JD: {jd_text} | Role: {role_hint}"

        # Fallback (Tier 1 — shouldn't happen in normal Tier 2 flow but defensive)
        cv_snippet = context.cv_text[:500] if context.cv_text else ""
        jd_snippet = context.jd_text[:500] if context.jd_text else ""
        return f"CV: {cv_snippet} | JD: {jd_snippet}"
