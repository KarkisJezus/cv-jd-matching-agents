"""
SemanticMatchingAgent: computes embedding-based similarity between CV and JD skills.

This agent demonstrates a key agentic property: it adapts its behavior
based on the shared context state. If normalized entities exist (Scenario B+),
it uses those. Otherwise, it falls back to raw extracted entities.

This is NOT an LLM-based agent — it uses sentence embeddings directly.
This is intentional: not every agent needs to use an LLM. The matching
agent uses a deterministic, reproducible similarity computation.
"""

from agents.base import BaseAgent
from embeddings.similarity import EmbeddingSimilarity
from models.shared_context import SharedContext


class SemanticMatchingAgent(BaseAgent):
    """
    Computes semantic similarity between CV and JD skill sets.

    Agentic behavior:
    - Reads context to decide which skill sets to compare
      (normalized if available, raw otherwise)
    - Adapts matching strategy based on what data exists
    - Logs its decision about which data source it used
    """

    def __init__(self, embedding_similarity: EmbeddingSimilarity | None = None):
        self._similarity = embedding_similarity or EmbeddingSimilarity()

    def process(self, context: SharedContext) -> SharedContext:
        """
        Compute skill-level semantic similarity.

        Reads: cv_entities, jd_entities, normalized_entities (if exists)
        Writes: similarity_scores
        """
        # Decide which skills to use — this is context-adaptive behavior
        cv_skills, jd_skills = context.get_skills_for_matching()
        source = "normalized" if context.has_enrichment() else "extracted"

        context.add_log(
            self.name,
            "matching_started",
            f"Using {source} skills: {len(cv_skills)} CV skills vs {len(jd_skills)} JD skills",
        )

        if not cv_skills or not jd_skills:
            context.add_log(
                self.name,
                "matching_skipped",
                "No skills available for matching",
            )
            return context

        # Compute similarity
        scores = self._similarity.find_best_matches(cv_skills, jd_skills)
        context.similarity_scores = scores

        context.add_log(
            self.name,
            "matching_completed",
            f"Overall similarity: {scores.overall_score:.3f}, "
            f"Coverage: {scores.coverage_ratio:.1%} "
            f"({scores.matched_skills_count}/{scores.total_jd_skills} JD skills matched)",
        )

        return context
