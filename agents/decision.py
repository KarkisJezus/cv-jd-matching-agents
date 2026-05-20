"""
DecisionAgent: produces the final match decision.

This agent synthesizes all available information into a final score,
confidence level, recommendation category, and explanation.

It demonstrates multi-signal reasoning: the decision is based on
similarity scores, LLM reasoning, and (when available) reflection
confidence. It does NOT simply copy the reasoning agent's score —
it makes its own judgment.
"""

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import FinalDecision
from models.shared_context import SharedContext


DECISION_SYSTEM_PROMPT = """\
You are the final decision agent. Based on all available analysis,
produce a final matching decision as a JSON object:
{
  "score": <float 0-100, final match score>,
  "confidence": <float 0-1, how confident you are in this score>,
  "recommendation": "<one of: strong_match, good_match, partial_match, weak_match, no_match>",
  "explanation": "<2-3 sentence explanation for a human reader>",
  "key_factors": ["list of 3-5 key factors that influenced the decision"]
}

Your goal is an honest, evidence-based judgment — neither overly generous
nor overly harsh. Score the evidence as it actually sits.

SCORING RUBRIC:
- 80-100 (strong_match):  ALL major requirements clearly met with strong
                          evidence. No missing core items. Minimal concerns.
- 60-79  (good_match):    most major requirements met, possibly with one
                          or two partial items. No missing core requirements,
                          or at most one that is minor.
- 40-59  (partial_match): mixed evidence. Some core requirements met, some
                          missing or partial. Borderline case. Use the
                          specific evidence to pick within this range.
- 20-39  (weak_match):    multiple core requirements missing. Candidate
                          would struggle without significant ramp-up.
- 0-19   (no_match):      fundamental mismatch (wrong domain / seniority /
                          core stack absent).

Guidance:
- Prefer the lower half of a range when the evidence is ambiguous, the
  higher half when the evidence is clearer.
- Transferable experience and adjacent skills count as PARTIAL evidence
  (they contribute, but less than directly-matching experience).
- Your score should be your own independent judgment, informed by but
  NOT blindly copying the reasoning agent's suggested score. If reflection
  raised legitimate concerns, adjust the score to address them — either
  direction.
- When strengths and gaps balance out, scores in the 40-55 range are
  typical. Don't round up or down systematically; pick the number the
  evidence actually supports.

Bias avoidance:
- Do not favor candidates with longer or better-formatted CVs
- Do not penalize candidates whose skills are phrased differently from the JD
- Base confidence on evidence quality, not on how assertive the reasoning sounds
- Weight all relevant skills equally regardless of their order in the documents

Return ONLY valid JSON.
"""


class DecisionAgent(BaseAgent):
    """
    Produces the final matching decision.

    Agentic behavior:
    - Weighs multiple signals: similarity scores, reasoning analysis,
      and reflection confidence (when available)
    - Adjusts confidence based on data completeness
    - Makes an independent judgment, not just copying prior scores
    """

    def __init__(self, llm_client: BaseLLMClient, commits_to: str = "final"):
        """
        Args:
            llm_client: the LLM client to use.
            commits_to: which SharedContext field to write the decision to.
                - "final" (default, Tier 1 + Tier 2 Scenarios A/B): writes to context.final_decision
                - "initial" (Tier 2 Scenario C Pass 1): writes to context.initial_decision
                  so CalibrationAgent can review it in Pass 2.
        """
        self._llm = llm_client
        if commits_to not in ("final", "initial"):
            raise ValueError(f"commits_to must be 'final' or 'initial', got {commits_to!r}")
        self._commits_to = commits_to

    def process(self, context: SharedContext) -> SharedContext:
        """
        Produce a decision, written to either final_decision or initial_decision
        depending on the commits_to setting.
        """
        user_prompt = self._build_prompt(context)

        context.add_log(self.name, "decision_started", f"commits_to={self._commits_to}")

        result = self._llm.chat_json(DECISION_SYSTEM_PROMPT, user_prompt)

        # Parse and validate the decision
        score = float(result.get("score", 50.0))
        score = max(0.0, min(100.0, score))

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        recommendation = result.get("recommendation", "partial_match")
        valid_recommendations = {
            "strong_match", "good_match", "partial_match", "weak_match", "no_match",
        }
        if recommendation not in valid_recommendations:
            recommendation = self._score_to_recommendation(score)

        decision = FinalDecision(
            score=round(score, 1),
            confidence=round(confidence, 2),
            recommendation=recommendation,
            explanation=result.get("explanation", ""),
            key_factors=[str(f) if not isinstance(f, str) else f for f in result.get("key_factors", [])],
        )

        # Write to the configured field
        if self._commits_to == "initial":
            context.initial_decision = decision
        else:
            context.final_decision = decision

        context.add_log(
            self.name,
            "decision_completed",
            f"Wrote to {self._commits_to}_decision: "
            f"score={decision.score}, confidence={decision.confidence}, "
            f"recommendation={decision.recommendation}",
        )

        return context

    def _build_prompt(self, context: SharedContext) -> str:
        """Build decision prompt from all available analysis data."""
        sections = []

        # Tier 2 profiles (compact summary — full profiles already informed reasoning)
        if context.cv_profile and context.jd_profile:
            sections.append(
                f"== PROFILES ==\n"
                f"Candidate: {context.cv_profile.candidate_archetype} "
                f"(seniority: {context.cv_profile.seniority_level})\n"
                f"Ideal candidate: {context.jd_profile.raw_summary} "
                f"(role: {context.jd_profile.detected_role}, "
                f"seniority: {context.jd_profile.seniority_required})"
            )

        # Similarity data
        if context.similarity_scores:
            s = context.similarity_scores
            sections.append(
                f"== SIMILARITY ANALYSIS ==\n"
                f"Overall similarity: {s.overall_score:.3f}\n"
                f"Skill coverage: {s.coverage_ratio:.1%} "
                f"({s.matched_skills_count}/{s.total_jd_skills} skills matched)"
            )

        # Reasoning data
        if context.reasoning_output:
            r = context.reasoning_output
            sections.append(
                f"== REASONING ANALYSIS ==\n"
                f"Suggested score: {r.suggested_score}\n"
                f"Strengths: {'; '.join(r.strengths)}\n"
                f"Gaps: {'; '.join(r.gaps)}\n"
                f"Concerns: {'; '.join(r.concerns)}\n"
                f"Assessment: {r.overall_assessment}"
            )

        # Reflection data (if available — shows multi-signal decision-making)
        if context.has_reflection() and context.reflection_output:
            ref = context.reflection_output
            sections.append(
                f"== REFLECTION REVIEW ==\n"
                f"Reasoning consistent: {ref.is_consistent}\n"
                f"Reflection confidence: {ref.confidence}\n"
                f"Issues found: {'; '.join(ref.issues_found) if ref.issues_found else 'None'}\n"
                f"Revision cycles completed: {context.revision_count}"
            )

        return "\n\n".join(sections) if sections else "No analysis data available."

    @staticmethod
    def _score_to_recommendation(score: float) -> str:
        """
        Derive recommendation from score if LLM gave an invalid one.

        Aligned with the balanced rubric in the system prompts (Tier 1.5):
        - 80+  strong_match
        - 60-79  good_match
        - 40-59  partial_match   (threshold=50 cuts mid-"partial")
        - 20-39  weak_match
        - <20  no_match
        """
        if score >= 80:
            return "strong_match"
        elif score >= 60:
            return "good_match"
        elif score >= 40:
            return "partial_match"
        elif score >= 20:
            return "weak_match"
        else:
            return "no_match"
