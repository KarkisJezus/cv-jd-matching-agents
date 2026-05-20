"""
Embedding-based semantic similarity using Sentence-BERT.

This module handles:
- Loading the sentence-transformers model
- Computing embeddings for skill lists
- Calculating pairwise cosine similarity between CV and JD skills
- Finding best matches above a configurable threshold

Design decisions:
- Model is loaded once and cached (expensive to load repeatedly)
- Uses cosine similarity from sklearn (well-tested, standard)
- Returns structured results for the SharedContext
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from models.entities import SkillMatch, SimilarityScores


class EmbeddingSimilarity:
    """
    Compute semantic similarity between skill sets using sentence embeddings.

    The model is loaded lazily on first use and cached for subsequent calls.
    This avoids slow startup when the model isn't needed (e.g., in tests).
    """

    def __init__(self, model_name: str | None = None, threshold: float | None = None):
        self._model_name = model_name or settings.embedding_model
        self._threshold = threshold if threshold is not None else settings.similarity_threshold
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embedding vectors."""
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True)

    def compute_similarity_matrix(
        self, cv_skills: list[str], jd_skills: list[str]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between CV and JD skills.

        Returns a matrix of shape (len(cv_skills), len(jd_skills))
        where entry [i][j] is the cosine similarity between
        cv_skills[i] and jd_skills[j].
        """
        if not cv_skills or not jd_skills:
            return np.array([])

        cv_embeddings = self.encode(cv_skills)
        jd_embeddings = self.encode(jd_skills)
        return cosine_similarity(cv_embeddings, jd_embeddings)

    def find_best_matches(
        self, cv_skills: list[str], jd_skills: list[str]
    ) -> SimilarityScores:
        """
        Find the best CV skill match for each JD skill.

        For each JD skill, finds the most similar CV skill.
        Only includes matches above the similarity threshold.

        Returns a SimilarityScores object with:
        - Individual skill matches
        - Overall similarity score (average of best matches)
        - Coverage ratio (fraction of JD skills matched)
        """
        if not cv_skills or not jd_skills:
            return SimilarityScores(
                overall_score=0.0,
                skill_matches=[],
                matched_skills_count=0,
                total_jd_skills=len(jd_skills),
                coverage_ratio=0.0,
            )

        sim_matrix = self.compute_similarity_matrix(cv_skills, jd_skills)
        matches: list[SkillMatch] = []
        match_scores: list[float] = []

        # For each JD skill, find the best matching CV skill
        for j, jd_skill in enumerate(jd_skills):
            best_cv_idx = int(np.argmax(sim_matrix[:, j]))
            best_score = float(sim_matrix[best_cv_idx, j])

            # Determine match type
            if best_score > 0.95:
                match_type = "exact"
            elif best_score >= self._threshold:
                match_type = "semantic"
            else:
                match_type = "below_threshold"

            matches.append(
                SkillMatch(
                    cv_skill=cv_skills[best_cv_idx],
                    jd_skill=jd_skill,
                    similarity=round(best_score, 4),
                    match_type=match_type,
                )
            )
            match_scores.append(best_score)

        # Count matches above threshold
        matched_count = sum(1 for m in matches if m.match_type != "below_threshold")
        coverage = matched_count / len(jd_skills) if jd_skills else 0.0

        # Overall score: average of all best-match scores
        overall = float(np.mean(match_scores)) if match_scores else 0.0

        return SimilarityScores(
            overall_score=round(overall, 4),
            skill_matches=matches,
            matched_skills_count=matched_count,
            total_jd_skills=len(jd_skills),
            coverage_ratio=round(coverage, 4),
        )
