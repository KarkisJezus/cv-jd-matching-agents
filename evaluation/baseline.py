"""
Baseline evaluator: pure embedding similarity without agents.

This provides a non-agentic reference point for comparison.
If the agent-based system (Scenarios A/B/C) doesn't outperform
this baseline, the agentic architecture adds no measurable value.

Method:
1. Encode the full CV text and full JD text as single embeddings
2. Compute cosine similarity between them
3. Scale to 0-100 and apply threshold for binary classification

This is intentionally primitive:
- No skill extraction (no LLM)
- No reasoning, reflection, or enrichment
- Same embedding model (all-MiniLM-L6-v2) as the agent system
- Compares entire documents, not individual skills

The baseline answers: "How much does the agentic layer improve
over raw semantic similarity?"
"""

from dataclasses import dataclass

from embeddings.similarity import EmbeddingSimilarity


@dataclass
class BaselineResult:
    """Result from the baseline embedding-only evaluator."""

    pair_id: str
    similarity: float       # Raw cosine similarity (0.0 - 1.0)
    predicted_score: float   # Similarity scaled to 0-100
    predicted_label: bool    # score >= threshold
    threshold: float


class BaselineEvaluator:
    """
    Embedding-only baseline: computes document-level cosine similarity.

    Uses the same sentence-transformer model as the agent system
    (all-MiniLM-L6-v2) for a fair comparison. The only difference
    is that no agents, LLM calls, or structured reasoning are involved.
    """

    def __init__(
        self,
        embedding_similarity: EmbeddingSimilarity | None = None,
        threshold: float = 50.0,
    ):
        self._embedding_sim = embedding_similarity or EmbeddingSimilarity()
        self._threshold = threshold

    def evaluate(self, pair_id: str, cv_text: str, jd_text: str) -> BaselineResult:
        """
        Compute baseline similarity between a CV and JD.

        Encodes both texts as single vectors and computes cosine
        similarity. The similarity is scaled to 0-100 for direct
        comparison with agent scores.
        """
        # Encode both documents as single embeddings
        embeddings = self._embedding_sim.encode([cv_text, jd_text])

        # Cosine similarity between the two vectors
        from sklearn.metrics.pairwise import cosine_similarity
        sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0])

        # Scale to 0-100 range (same scale as agent scores)
        score = round(sim * 100, 1)
        predicted_label = score >= self._threshold

        return BaselineResult(
            pair_id=pair_id,
            similarity=round(sim, 4),
            predicted_score=score,
            predicted_label=predicted_label,
            threshold=self._threshold,
        )

    def evaluate_batch(
        self, pairs: list[tuple[str, str, str]]
    ) -> list[BaselineResult]:
        """
        Evaluate multiple (pair_id, cv_text, jd_text) tuples.

        Returns a list of BaselineResult in the same order.
        """
        return [
            self.evaluate(pair_id, cv_text, jd_text)
            for pair_id, cv_text, jd_text in pairs
        ]
