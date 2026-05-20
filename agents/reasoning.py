"""
ReasoningAgent: uses LLM to produce a structured analysis of the match.

This is the most context-aware agent — it reads EVERYTHING available
in the shared context and builds a comprehensive prompt from it.
This demonstrates the blackboard advantage: the reasoning agent sees
extracted entities, similarity scores, enrichment notes (if any),
and memory entries (if any), all at once.

The agent adapts its prompt based on which data exists, producing
different analyses for different scenarios.
"""

import json

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import ReasoningOutput
from models.shared_context import SharedContext


def _coerce_to_str_list(items: list) -> list[str]:
    """Convert list items to strings. LLMs sometimes return dicts instead of strings."""
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # Convert dict like {"skill": "Python", "similarity_score": 0.85} to a string
            parts = [f"{v}" for v in item.values()]
            result.append(" — ".join(parts))
        else:
            result.append(str(item))
    return result


REASONING_SYSTEM_PROMPT = """\
You are a reasoning agent that analyzes the match between a candidate (CV)
and a job position (job description). You have access to extracted entities
and semantic similarity scores. Your goal is an HONEST, EVIDENCE-BASED
judgment — neither over-generous nor overly harsh.

BEFORE SCORING, perform this audit (this gives you the factual basis to
score fairly rather than guessing):

Step 1. List the core requirements from the JD: must-have skills, required
        years of experience, required domain knowledge, required education
        or certifications.
Step 2. For each core requirement, classify the evidence in the CV as:
          MET       — CV clearly demonstrates this requirement
          PARTIAL   — CV shows related experience, transferable skills, or
                      indirect evidence (counts toward the score, but less
                      than MET)
          MISSING   — CV does not mention this requirement at all
Step 3. Weight gaps heavily: a MISSING core requirement is a strong
        negative signal. PARTIAL is a mild negative. Multiple MISSING items
        on core requirements indicate a weak or no match.

Produce a structured analysis as a JSON object:
{
  "strengths": ["list of MET requirements with concrete evidence from the CV"],
  "gaps": ["list of MISSING core requirements the candidate lacks"],
  "concerns": ["list of PARTIAL requirements or risks to highlight"],
  "overall_assessment": "2-3 sentence overall assessment summarizing the
                        met-vs-missing balance",
  "suggested_score": <float 0-100, following the rubric below>
}

SCORING RUBRIC:
- 80-100: strong match — ALL major requirements MET with clear evidence.
          No MISSING core requirements. At most minor concerns.
- 60-79:  good match — most major requirements MET, possibly with one or
          two PARTIAL items. No MISSING core requirements, or at most one
          that is minor.
- 40-59:  partial match — mixed evidence. Some core requirements MET,
          some MISSING or PARTIAL. Candidate is plausible but has real gaps.
- 20-39:  weak match — multiple core requirements MISSING. Candidate
          would struggle to succeed in the role without significant ramp-up.
- 0-19:   no match — fundamental mismatch: wrong domain, wrong seniority,
          or nearly all core skills absent.

NOTE on threshold: scores of 50 and above are typically interpreted as
"match" downstream. So the 40-59 range is genuinely borderline — use the
specific evidence to pick a number inside that range (closer to 40 for
leaning-reject, closer to 59 for leaning-accept).

Guidance on generosity:
- Prefer lower scores when evidence is ambiguous. A clear "probably not" is
  more useful than an uncertain "maybe." When strengths and gaps are roughly
  balanced, pick a score in the 40-55 range.
- Transferable experience and adjacent skills DO count as PARTIAL evidence,
  but PARTIAL is worth less than MET. Don't dismiss related experience, but
  don't treat it as equivalent either.
- Avoid the extremes: don't default to 70+ just because nothing looks
  disqualifying, and don't default to 20 just because anything is uncertain.
  Score the evidence as it actually sits.

Logical rules for gaps:
- Gaps must ONLY list requirements that the candidate genuinely does not meet.
  Apply strict numeric logic: if a JD requires "5+ years" and the candidate
  has 5 years, that requirement IS met (5 >= 5). Do not list met requirements
  as gaps.
- Carefully distinguish between "does not have" vs "not explicitly mentioned".
  If the CV does not mention a core skill required by the JD, it IS a gap.
  If the CV mentions an equivalent skill by different name, it is a strength.

Format rules:
- Each item in strengths/gaps/concerns must be a plain string (not an object)
- Return ONLY valid JSON

Bias avoidance:
- Do not favor or penalize skills based on their position in the CV or JD
- Do not reward longer CVs or penalize shorter ones — judge substance, not length
- Evaluate based on actual qualifications, not writing style or formatting
- Treat equivalent skills equally regardless of phrasing (e.g., "ML" = "Machine Learning")
"""


class ReasoningAgent(BaseAgent):
    """
    Produces LLM-based reasoning about the match quality.

    Agentic behavior:
    - Reads the full shared context, not just a single input
    - Adapts its prompt based on available data (enrichment, memory, etc.)
    - Generates different analysis depth for different scenarios
    - If revision is requested by ReflectionAgent, incorporates feedback
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._llm = llm_client

    def process(self, context: SharedContext) -> SharedContext:
        """
        Produce a structured reasoning analysis.

        Reads: cv_entities, jd_entities, similarity_scores,
               normalized_entities (optional), memory_entries (optional),
               reflection_output (if revision cycle)
        Writes: reasoning_output
        """
        # Build a comprehensive prompt from ALL available context
        user_prompt = self._build_prompt(context)

        context.add_log(
            self.name,
            "reasoning_started",
            f"Revision cycle: {context.revision_count}, "
            f"Context includes: entities={'yes' if context.cv_entities else 'no'}, "
            f"scores={'yes' if context.similarity_scores else 'no'}, "
            f"enrichment={'yes' if context.has_enrichment() else 'no'}, "
            f"memory={'yes' if context.has_memory() else 'no'}",
        )

        result = self._llm.chat_json(REASONING_SYSTEM_PROMPT, user_prompt)

        context.reasoning_output = ReasoningOutput(
            strengths=_coerce_to_str_list(result.get("strengths", [])),
            gaps=_coerce_to_str_list(result.get("gaps", [])),
            concerns=_coerce_to_str_list(result.get("concerns", [])),
            overall_assessment=str(result.get("overall_assessment", "")),
            suggested_score=max(0.0, min(100.0, float(result.get("suggested_score", 50.0)))),
        )

        context.add_log(
            self.name,
            "reasoning_completed",
            f"Suggested score: {context.reasoning_output.suggested_score}, "
            f"Strengths: {len(context.reasoning_output.strengths)}, "
            f"Gaps: {len(context.reasoning_output.gaps)}",
        )

        return context

    def _build_prompt(self, context: SharedContext) -> str:
        """
        Build the user prompt from all available context data.

        This method demonstrates the blackboard advantage: the agent
        reads whatever is available and adapts its prompt accordingly.
        A pipeline agent would only see its direct predecessor's output.

        Tier 2: prefers cv_profile/jd_profile (richer interpretive fields).
        Tier 1: falls back to cv_entities/jd_entities (raw extraction).
        """
        sections = []

        # Tier 2: profiles (preferred)
        if context.cv_profile:
            cp = context.cv_profile
            sections.append(
                "== CANDIDATE PROFILE (from CV) ==\n"
                f"Archetype: {cp.candidate_archetype}\n"
                f"Seniority: {cp.seniority_level}\n"
                f"Likely role fit: {cp.likely_role_fit or 'unknown'}\n"
                f"Skills: {', '.join(cp.skills)}\n"
                f"Domain expertise: {', '.join(cp.domain_expertise)}\n"
                f"Experience: {'; '.join(cp.experience)}\n"
                f"Education: {'; '.join(cp.education)}\n"
                f"Summary: {cp.raw_summary}"
            )
        elif context.cv_entities:
            sections.append(
                "== CV ENTITIES ==\n"
                f"Skills: {', '.join(context.cv_entities.skills)}\n"
                f"Experience: {'; '.join(context.cv_entities.experience)}\n"
                f"Education: {'; '.join(context.cv_entities.education)}\n"
                f"Summary: {context.cv_entities.raw_summary}"
            )

        if context.jd_profile:
            jp = context.jd_profile
            sections.append(
                "== IDEAL CANDIDATE PROFILE (from JD + ESCO role context) ==\n"
                f"Detected role: {jp.detected_role} (ESCO {jp.esco_code}, confidence {jp.role_confidence:.2f})\n"
                f"Seniority required: {jp.seniority_required}\n"
                f"Required experience: {jp.required_experience_years} years\n"
                f"Required education: {jp.required_education}\n"
                f"REQUIRED skills (JD-explicit): {', '.join(jp.required_skills)}\n"
                f"TYPICAL skills for this role (per ESCO): {', '.join(jp.typical_role_skills)}\n"
                f"Key responsibilities: {'; '.join(jp.key_responsibilities)}\n"
                f"Summary: {jp.raw_summary}\n"
                f"\n"
                f"NOTE: REQUIRED skills are explicitly demanded by the JD. "
                f"TYPICAL skills are typical for this role per ESCO standards even "
                f"if the JD doesn't list them — weigh them lower than REQUIRED."
            )
        elif context.jd_entities:
            sections.append(
                "== JD ENTITIES ==\n"
                f"Required skills: {', '.join(context.jd_entities.skills)}\n"
                f"Requirements: {'; '.join(context.jd_entities.experience)}\n"
                f"Summary: {context.jd_entities.raw_summary}"
            )

        # Include similarity scores if available
        if context.similarity_scores:
            scores = context.similarity_scores
            match_details = "\n".join(
                f"  {m.cv_skill} <-> {m.jd_skill}: {m.similarity:.3f} ({m.match_type})"
                for m in scores.skill_matches
            )
            sections.append(
                f"== SIMILARITY SCORES ==\n"
                f"Overall: {scores.overall_score:.3f}\n"
                f"Coverage: {scores.coverage_ratio:.1%} "
                f"({scores.matched_skills_count}/{scores.total_jd_skills})\n"
                f"Matches:\n{match_details}"
            )

        # Include enrichment notes if available (Scenario B+)
        if context.has_enrichment() and context.enrichment_notes:
            sections.append(
                "== ENRICHMENT NOTES ==\n"
                + "\n".join(f"- {note}" for note in context.enrichment_notes)
            )

        # Include memory if available (Scenario C)
        if context.has_memory():
            memory_text = "\n".join(
                f"- Previous match (score={m.decision_score}): {m.reasoning_summary}"
                for m in context.memory_entries
            )
            sections.append(f"== RELEVANT PAST DECISIONS ==\n{memory_text}")

        # Include reflection feedback if this is a revision cycle
        if context.revision_count > 0 and context.reflection_output:
            ref = context.reflection_output
            sections.append(
                f"== REFLECTION FEEDBACK (revision #{context.revision_count}) ==\n"
                f"Issues: {'; '.join(ref.issues_found) if ref.issues_found else 'None'}\n"
                f"Suggestions: {'; '.join(ref.suggestions) if ref.suggestions else 'None'}\n"
                f"Please address the above feedback in your revised analysis."
            )

        return "\n\n".join(sections)
