"""
MemoryRetrievalAgent: retrieves relevant past decisions from memory.

This agent is the first in the Scenario C pipeline. Before any
extraction or matching begins, it searches the memory store for
past matching decisions that are similar to the current CV-JD pair.

The retrieved memories are written to context.memory_entries, where
downstream agents (especially ReasoningAgent) can use them as
reference points. This enables the system to learn from experience:

  "We saw a similar CV-JD pair before, scored it 72, and noted
   that cloud experience was lacking. Let's consider that."

This demonstrates:
1. Agent-driven context enrichment from persistent state
2. The system's ability to accumulate knowledge across runs
3. Non-trivial agent behavior: the agent decides relevance
   based on semantic similarity, not exact matching

The agent does NOT use LLM — it uses pure embedding-based retrieval.
This is intentional: memory retrieval is a deterministic operation
that doesn't benefit from LLM variability.
"""

from agents.base import BaseAgent
from memory.store import MemoryStore
from models.shared_context import SharedContext


class MemoryRetrievalAgent(BaseAgent):
    """
    Retrieves relevant past matching decisions from the memory store.

    Agentic behavior:
    - Autonomously decides which memories are relevant (similarity threshold)
    - Adapts its query based on what input is available
    - Enriches the context with historical reference points
    - Downstream agents see memories and can adjust their analysis

    Unlike most other agents, this one does NOT use an LLM.
    It uses embedding-based vector retrieval, similar to RAG systems.
    """

    def __init__(self, memory_store: MemoryStore):
        self._store = memory_store

    def process(self, context: SharedContext) -> SharedContext:
        """
        Retrieve relevant past decisions and add them to context.

        Reads: cv_text, jd_text
        Writes: memory_entries
        """
        if self._store.count == 0:
            context.add_log(
                self.name,
                "memory_empty",
                "No past decisions in memory store",
            )
            return context

        # Build a query from the current CV and JD
        query = self._build_query(context)

        context.add_log(
            self.name,
            "retrieval_started",
            f"Searching {self._store.count} memories",
        )

        # Retrieve similar past decisions
        memories = self._store.retrieve(query)

        if memories:
            context.memory_entries = memories
            context.add_log(
                self.name,
                "retrieval_completed",
                f"Retrieved {len(memories)} relevant memories. "
                f"Best similarity: {memories[0].similarity_to_current:.3f}, "
                f"Best score: {memories[0].decision_score}",
            )
        else:
            context.add_log(
                self.name,
                "retrieval_no_matches",
                "No memories above similarity threshold",
            )

        return context

    def _build_query(self, context: SharedContext) -> str:
        """
        Build a retrieval query from the current CV and JD.

        Uses a truncated combination of both texts. In a production
        system, you might use the extracted entities or summaries
        instead, but at this point in the pipeline extraction hasn't
        run yet (MemoryRetrievalAgent runs first in the chain).
        """
        # Truncate to keep embedding input reasonable
        cv_snippet = context.cv_text[:500]
        jd_snippet = context.jd_text[:500]
        return f"CV: {cv_snippet} | JD: {jd_snippet}"
