"""
ReflectionAgent: reviews the reasoning for consistency and quality.

This is the agent that makes the system truly agentic rather than a
sequential pipeline. It performs metacognition — an agent evaluating
another agent's work — and can trigger re-execution of the reasoning
step if it finds problems.

The reflection loop:
  ReasoningAgent produces analysis
    -> ReflectionAgent reviews it
      -> if consistent: proceed to DecisionAgent
      -> if inconsistent: set needs_revision=True, orchestrator
         re-runs ReasoningAgent with reflection feedback appended
         to the prompt, up to max_revisions times

This demonstrates:
1. Agent autonomy: the reflection agent independently judges quality
2. Inter-agent feedback: its output becomes input for the next
   reasoning iteration
3. Non-linear flow: the orchestrator path depends on agent output
4. Self-improvement: repeated cycles should converge toward better
   reasoning
"""

import json

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import ReflectionOutput
from models.shared_context import SharedContext


REFLECTION_SYSTEM_PROMPT = """\
You are a reflection agent. Your task is to review the reasoning analysis
produced by the Reasoning Agent and check it for quality and consistency.
Your job is to catch errors in BOTH directions — over-generous scoring AND
over-harsh scoring — and flag only meaningful problems.

You receive:
- The reasoning output (strengths, gaps, concerns, assessment, suggested score)
- The underlying data (similarity scores, extracted entities)

SCORING RUBRIC the reasoning should be using:
- 80-100: strong match (ALL core requirements MET)
- 60-79:  good match (most MET, at most one or two PARTIAL items)
- 40-59:  partial match (mixed evidence — some MET, some MISSING/PARTIAL)
- 20-39:  weak match (multiple core requirements MISSING)
- 0-19:   no match (fundamental mismatch)

CHECKS (weighted roughly equally):

1. OVER-GENEROUS SCORING:
   - If suggested_score >= 70 but the gaps list includes a core JD
     requirement marked MISSING, the score is too high. FLAG IT.
   - If suggested_score >= 60 and the reasoning relies entirely on hedges
     like "could learn" / "transferable" / "likely familiar" for core
     requirements (no concrete evidence), FLAG IT.
   - If suggested_score is in the 80-100 range but the concerns list
     mentions uncertainty about a core requirement, FLAG IT.

2. OVER-HARSH SCORING (equally important):
   - If suggested_score is below 30 but the strengths list shows multiple
     MET core requirements, the score is too low. FLAG IT.
   - If the reasoning dismisses legitimate transferable experience as
     "not counting at all," FLAG IT — PARTIAL evidence should count for
     something.
   - If the score is below 40 but the coverage ratio in similarity data
     is above 50%, check for overcorrection. FLAG mismatches.

3. STRENGTHS MUST BE EVIDENCE-BACKED:
   - Each strength must cite something concrete from the CV that maps to a
     JD requirement. Vague strengths like "strong communication" without
     evidence are weak supporting facts.

4. GAPS MUST BE REAL:
   - A gap must be a requirement the candidate genuinely does not meet.
   - Logical errors to catch:
     * Listing "5+ years required" as a gap when the candidate has exactly 5
       years (5 >= 5, so the requirement IS met — this is NOT a gap)
     * Listing a skill as missing when the CV mentions it under a different
       name (e.g., ML vs Machine Learning)
     * Listing something as a gap that the JD does not actually require

5. INTERNAL CONSISTENCY:
   - Score must align with the strengths/gaps balance per the rubric
   - No item may appear as both a strength AND a gap
   - overall_assessment should match the score

6. BIAS:
   - Flag if the reasoning favors skills listed early in the CV (position bias)
   - Flag if the reasoning penalizes a short CV or rewards a long one (length bias)
   - Flag if equivalent skills are treated inconsistently

Produce a JSON object:
{
  "is_consistent": <bool, true if the reasoning is sound overall>,
  "issues_found": ["list of specific problems found, empty if none"],
  "suggestions": ["list of concrete suggestions for improvement"],
  "confidence": <float 0-1, your confidence in the reasoning quality>,
  "revision_reason": "<if is_consistent is false, explain why revision is needed>"
}

Decision rules:
- Set is_consistent=false ONLY when a meaningful check above is violated.
  Do NOT flag borderline or stylistic issues.
- Correct scoring errors in BOTH directions — if reasoning is too harsh,
  request revision to raise the score; if too generous, to lower it.
- confidence < 0.5 means serious doubts about reasoning quality.
- Return ONLY valid JSON.
"""


class ReflectionAgent(BaseAgent):
    """
    Reviews the reasoning analysis for consistency and quality.

    This agent reads the full shared context (reasoning output,
    similarity scores, entities) and evaluates whether the reasoning
    is supported by the evidence.

    If it finds problems, it sets needs_revision=True on the context,
    which signals the orchestrator to re-run the ReasoningAgent. The
    ReasoningAgent's _build_prompt method will then include the
    reflection feedback, creating a feedback loop.

    Agentic behavior:
    - Makes an autonomous judgment about another agent's work
    - Its output directly controls the orchestrator's execution path
    - Adapts its assessment based on the revision cycle count
      (is more lenient on later cycles to prevent infinite loops)
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._llm = llm_client

    def process(self, context: SharedContext) -> SharedContext:
        """
        Review the reasoning output and decide whether revision is needed.

        Reads: reasoning_output, similarity_scores, cv_entities, jd_entities,
               revision_count
        Writes: reflection_output, needs_revision
        """
        user_prompt = self._build_prompt(context)

        context.add_log(
            self.name,
            "reflection_started",
            f"Reviewing reasoning (revision cycle {context.revision_count})",
        )

        result = self._llm.chat_json(REFLECTION_SYSTEM_PROMPT, user_prompt)

        # Parse reflection output
        is_consistent = result.get("is_consistent", True)
        issues = result.get("issues_found", [])
        suggestions = result.get("suggestions", [])
        confidence = float(result.get("confidence", 1.0))
        confidence = max(0.0, min(1.0, confidence))
        revision_reason = result.get("revision_reason", "")

        # Coerce list items to strings (LLM may return dicts)
        issues = [str(i) if not isinstance(i, str) else i for i in issues]
        suggestions = [str(s) if not isinstance(s, str) else s for s in suggestions]

        context.reflection_output = ReflectionOutput(
            is_consistent=is_consistent,
            issues_found=issues,
            suggestions=suggestions,
            confidence=confidence,
            revision_reason=revision_reason,
        )

        # Decide whether to request revision
        # The agent sets the flag; the orchestrator decides whether to act on it
        can_still_revise = context.revision_count < context.max_revisions
        context.needs_revision = (not is_consistent) and can_still_revise

        if context.needs_revision:
            context.add_log(
                self.name,
                "revision_requested",
                f"Reason: {revision_reason}. "
                f"Issues: {len(issues)}. "
                f"Revision {context.revision_count + 1}/{context.max_revisions}",
            )
        else:
            status = "consistent" if is_consistent else "issues found but max revisions reached"
            context.add_log(
                self.name,
                "reflection_completed",
                f"Status: {status}. "
                f"Confidence: {confidence:.2f}. "
                f"Issues: {len(issues)}",
            )

        return context

    def _build_prompt(self, context: SharedContext) -> str:
        """
        Build the review prompt with all relevant data for cross-checking.

        The reflection agent needs to see both the reasoning output AND
        the underlying data so it can verify consistency.
        """
        sections = []

        # The reasoning to review
        if context.reasoning_output:
            r = context.reasoning_output
            sections.append(
                "== REASONING OUTPUT TO REVIEW ==\n"
                f"Suggested score: {r.suggested_score}\n"
                f"Strengths: {json.dumps(r.strengths)}\n"
                f"Gaps: {json.dumps(r.gaps)}\n"
                f"Concerns: {json.dumps(r.concerns)}\n"
                f"Assessment: {r.overall_assessment}"
            )

        # The data to cross-check against
        if context.similarity_scores:
            s = context.similarity_scores
            match_detail = ", ".join(
                f"{m.cv_skill}<->{m.jd_skill}={m.similarity:.3f}"
                for m in s.skill_matches
            )
            sections.append(
                f"== SIMILARITY DATA ==\n"
                f"Overall: {s.overall_score:.3f}, "
                f"Coverage: {s.coverage_ratio:.1%}\n"
                f"Matches: {match_detail}"
            )

        if context.cv_entities:
            sections.append(
                f"== CV SKILLS ==\n{', '.join(context.cv_entities.skills)}"
            )

        if context.jd_entities:
            sections.append(
                f"== JD REQUIREMENTS ==\n{', '.join(context.jd_entities.skills)}"
            )

        # Context about the revision cycle
        sections.append(
            f"== REVISION CONTEXT ==\n"
            f"Current revision cycle: {context.revision_count}\n"
            f"Max allowed revisions: {context.max_revisions}"
        )

        return "\n\n".join(sections)
