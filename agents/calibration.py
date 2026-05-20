"""
CalibrationAgent: Pass 2 of the two-pass decision flow (Scenario C only).

Reviews the initial_decision (committed by DecisionAgent in Pass 1, without
seeing any labels) against retrieved labeled past pairs. Detects systematic
miscalibration patterns and proposes a final adjusted score.

This is the centerpiece of the new Scenario C architecture. It's where
the labeled history actually gets used — the system finally has a chance to
correct its own bias by comparing its initial guess against past decisions
where it knows it was wrong.

Methodological note (Yehudai et al. 2025, StreamBench protocol):
The CalibrationAgent never sees the CURRENT pair's label. It only sees:
- The system's own initial decision
- A list of LABELED PAST pairs (each with system_score + ground_truth_label
  + ground_truth_reason)

Past labels function as in-context learning material, not as direct training data.
"""

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import CalibrationOutput, FinalDecision
from models.shared_context import SharedContext


# Conservative calibration constants (tuned after 20-pair pilot revealed
# the LLM was applying max-magnitude lowering on ~60% of pairs).

# Skip calibration entirely when retrieval returns fewer than this many
# labeled memories. Prevents the early-rejection-bias amplification observed
# in the pilot. With LabeledMemoryRetrievalAgent now using top_k=5, this
# triggers when the store has 0-2 entries (i.e., only pairs 1-3 of any
# cold-start run). Setting this too high means the gate fires forever
# (since retrieval is naturally bounded by top_k).
WARMUP_MIN_MEMORIES = 3

# Maximum allowed adjustment magnitude. The pilot showed the LLM treats whatever
# cap we set as "always pick max", so a smaller cap directly reduces the damage
# of over-eager calibration. ±8 means even max-magnitude lowering can't push a
# 65-scored borderline match below the threshold of 60 (would only reach 57).
MAX_ADJUSTMENT_MAGNITUDE = 8

# Below this many supporting memories, only allow keep (no lower/raise).
# Combined with WARMUP_MIN_MEMORIES, this ensures calibration only fires when
# there's enough evidence to be meaningful.
# Lowered from 3 to 2 (Path A) after the second pilot showed the LLM honestly
# reporting only 0-2 supporting memories per pair, which forced "keep" every
# time. Setting to 2 lets calibration actually fire while the ±8 cap still
# bounds damage from over-eager adjustments.
MIN_MEMORIES_FOR_ADJUSTMENT = 2


def _coerce_to_str(value, default: str = "") -> str:
    """Defensive coercion — LLMs occasionally return objects where strings are expected."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


CALIBRATION_SYSTEM_PROMPT = """\
You are a calibration agent. The system has already committed an INITIAL
DECISION on the current CV-JD pair (Pass 1, made without seeing any labels).
Your job is to review that initial decision against a list of LABELED PAST
PAIRS and decide whether to adjust the score.

DEFAULT POSTURE: KEEP. Calibration adjustments should be RARE and CONSERVATIVE.
Most pairs should keep their initial score. Adjustment is only justified when
the labeled history shows a clear, consistent, BALANCED pattern of system
miscalibration on similar profiles.

You must NOT see the ground-truth label for the CURRENT pair — you only see
the system's initial decision. The labeled past pairs include their system
scores AND the human ground-truth labels, so you can detect calibration
patterns.

STRICT EVIDENCE REQUIREMENTS (failing any one defaults to "keep"):

1. **At least 2 supporting memories** must show the same miscalibration direction.
   A single memory is anecdotal; two memories with consistent direction can
   indicate a pattern when paired with the other requirements below.

2. **Balanced evidence**: at least 1 retrieved memory must have the OPPOSITE
   ground-truth label from the others. If all retrieved memories are rejections,
   you do not have a pattern of "system over-scores" — you have a sampling
   artifact. KEEP.

3. **Profile similarity**: the supporting memories must describe candidates
   with comparable seniority and domain to the current pair. A pattern derived
   from a totally different role is not evidence.

4. **Recent track record**: if the most recent retrieved memories are
   inconsistent with each other (some accepted, some rejected with no
   profile-based reason), KEEP.

What patterns DO justify an adjustment:

- LOWER: 2+ similar-profile past pairs where system scored above the current
  initial score AND humans REJECTED them, AND there is at least one accepted
  case in the retrieval to confirm the pattern is profile-specific (not
  global "always lower").
- RAISE: 2+ similar-profile past pairs where system scored below the current
  initial score AND humans ACCEPTED them, AND there is at least one rejected
  case in the retrieval.
- KEEP: anything else. When in doubt, KEEP.

Return a JSON object:
{
  "calibration_decision": "lower" | "raise" | "keep",
  "adjusted_score": <float 0-100, the calibrated final score>,
  "adjusted_recommendation": "strong_match" | "good_match" | "partial_match" | "weak_match" | "no_match",
  "rationale": "<2-3 sentences citing the SPECIFIC labeled memories that satisfied the strict evidence requirements>",
  "confidence": <float 0-1, your confidence in this calibration>,
  "pattern_observed": "<one sentence describing the pattern, or 'no consistent pattern' if KEEP>",
  "n_supporting_memories": <integer, how many of the retrieved memories supported the calibration decision>
}

Decision rules:
- MAX ADJUSTMENT MAGNITUDE: ±8 points. Larger adjustments are NOT allowed.
  Typical valid adjustments are ±3 to ±6 points. ±8 only when the pattern
  is overwhelming (5+ consistent supporting memories on identical profile types).
- Default to KEEP when retrieved memories are mixed or unbalanced.
- Default to KEEP when fewer than 2 supporting memories agree.
- adjusted_recommendation must align with adjusted_score:
  80-100 → strong_match
  60-79 → good_match
  40-59 → partial_match
  20-39 → weak_match
  0-19 → no_match
- confidence reflects pattern strength: 0.8+ for 4+ balanced supporting memories,
  0.5-0.7 for 2-3 supporting memories with proper balance, below 0.5 means KEEP.
- n_supporting_memories must be EXACTLY the count of retrieved memories that
  satisfy ALL of the strict requirements above. Be honest — if only 2 memories
  technically support the direction, write "n_supporting_memories": 2 and KEEP.
- Return ONLY valid JSON.
"""


class CalibrationAgent(BaseAgent):
    """
    Pass 2 of the two-pass decision flow. Reads initial_decision + labeled
    memories and produces a calibrated final_decision.

    Reads:
        context.initial_decision
        context.reasoning_output (for context)
        context.labeled_memory_entries (the retrieved labeled past pairs)
        context.cv_profile, context.jd_profile (for prompt context)

    Writes:
        context.calibration_output  — full CalibrationOutput from the LLM
        context.final_decision      — synthesized FinalDecision after calibration
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._llm = llm_client

    def process(self, context: SharedContext) -> SharedContext:
        if context.initial_decision is None:
            context.add_log(
                self.name,
                "skipped",
                "No initial_decision in context — Pass 1 must run first",
            )
            return context

        # Cold-start handling: no labeled memories → keep initial decision unchanged.
        # This avoids an unnecessary LLM call when there's nothing to calibrate against.
        if not context.has_labeled_memory():
            context.add_log(
                self.name,
                "cold_start_skip",
                "No labeled memory available — keeping initial_decision as final_decision",
            )
            context.calibration_output = CalibrationOutput(
                calibration_decision="keep",
                adjusted_score=context.initial_decision.score,
                adjusted_recommendation=context.initial_decision.recommendation,
                rationale="Cold start: no labeled past pairs available for calibration.",
                confidence=context.initial_decision.confidence,
                pattern_observed="no history",
                n_supporting_memories=0,
            )
            context.final_decision = self._initial_to_final(context)
            return context

        # Warm-up gate: skip calibration when memory is too small to be representative.
        # The pilot showed pairs 2-5 amplify whatever the first 1-2 entries contain
        # (typically all-rejects in skewed datasets), leading the LLM to learn an
        # "always lower" heuristic. By requiring WARMUP_MIN_MEMORIES (5) before
        # any calibration runs, we let the memory diversify before applying Pass 2.
        n_retrieved = len(context.labeled_memory_entries)
        if n_retrieved < WARMUP_MIN_MEMORIES:
            context.add_log(
                self.name,
                "warmup_skip",
                f"Only {n_retrieved} labeled memories retrieved "
                f"(need >= {WARMUP_MIN_MEMORIES}) — keeping initial_decision unchanged. "
                f"Streaming warm-up phase.",
            )
            context.calibration_output = CalibrationOutput(
                calibration_decision="keep",
                adjusted_score=context.initial_decision.score,
                adjusted_recommendation=context.initial_decision.recommendation,
                rationale=(
                    f"Warm-up phase: only {n_retrieved} similar past pairs available "
                    f"(threshold: {WARMUP_MIN_MEMORIES}). Insufficient evidence "
                    f"to detect a calibration pattern."
                ),
                confidence=context.initial_decision.confidence,
                pattern_observed="warm-up; insufficient memory",
                n_supporting_memories=0,
            )
            context.final_decision = self._initial_to_final(context)
            return context

        # Build the prompt with the initial decision + labeled memories
        user_prompt = self._build_prompt(context)

        context.add_log(
            self.name,
            "calibration_started",
            f"Reviewing initial_decision (score={context.initial_decision.score}) "
            f"against {len(context.labeled_memory_entries)} labeled memories",
        )

        result = self._llm.chat_json(CALIBRATION_SYSTEM_PROMPT, user_prompt)

        # Parse + validate
        decision = _coerce_to_str(result.get("calibration_decision", "keep")).lower().strip()
        if decision not in ("lower", "raise", "keep"):
            decision = "keep"

        adjusted_score = float(result.get("adjusted_score", context.initial_decision.score))
        adjusted_score = max(0.0, min(100.0, adjusted_score))

        n_supporting = int(result.get("n_supporting_memories", 0))
        n_supporting = max(0, n_supporting)

        # Defensive enforcement of the strict evidence requirement.
        # The pilot showed the LLM treats prompt limits as suggestions, so we
        # enforce them in code: insufficient supporting memories → force "keep".
        if decision != "keep" and n_supporting < MIN_MEMORIES_FOR_ADJUSTMENT:
            context.add_log(
                self.name,
                "calibration_overridden",
                f"LLM proposed {decision} with only {n_supporting} supporting memories "
                f"(min: {MIN_MEMORIES_FOR_ADJUSTMENT}). Forcing 'keep'.",
            )
            decision = "keep"
            adjusted_score = context.initial_decision.score

        # Enforce the maximum-adjustment cap. The pilot showed the LLM hits
        # whatever cap we expose 100% of the time; capping in code (not just prompt)
        # is the only reliable way to bound calibration impact.
        if decision != "keep":
            initial_score = context.initial_decision.score
            requested_delta = adjusted_score - initial_score
            if abs(requested_delta) > MAX_ADJUSTMENT_MAGNITUDE:
                # Clamp to allowed magnitude in the LLM-requested direction.
                clamped_delta = (
                    MAX_ADJUSTMENT_MAGNITUDE if requested_delta > 0 else -MAX_ADJUSTMENT_MAGNITUDE
                )
                clamped_score = max(0.0, min(100.0, initial_score + clamped_delta))
                context.add_log(
                    self.name,
                    "calibration_clamped",
                    f"LLM requested {requested_delta:+.1f}pt adjustment "
                    f"(exceeds ±{MAX_ADJUSTMENT_MAGNITUDE} cap). "
                    f"Clamped to {clamped_delta:+.1f}.",
                )
                adjusted_score = clamped_score

        # If keep but score differs, force consistency (LLM sometimes returns "keep" with a different score)
        if decision == "keep":
            adjusted_score = context.initial_decision.score

        adjusted_recommendation = _coerce_to_str(result.get("adjusted_recommendation", ""))
        if adjusted_recommendation not in (
            "strong_match", "good_match", "partial_match", "weak_match", "no_match",
        ):
            adjusted_recommendation = self._score_to_recommendation(adjusted_score)

        confidence = float(result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        context.calibration_output = CalibrationOutput(
            calibration_decision=decision,
            adjusted_score=round(adjusted_score, 1),
            adjusted_recommendation=adjusted_recommendation,
            rationale=_coerce_to_str(result.get("rationale", "")),
            confidence=round(confidence, 2),
            pattern_observed=_coerce_to_str(result.get("pattern_observed", "")),
            n_supporting_memories=n_supporting,
        )

        # Synthesize the final decision from the calibration
        context.final_decision = FinalDecision(
            score=context.calibration_output.adjusted_score,
            confidence=round(confidence, 2),
            recommendation=adjusted_recommendation,
            explanation=(
                f"{context.initial_decision.explanation} "
                f"[Calibration ({decision}): {context.calibration_output.rationale}]"
            ),
            key_factors=list(context.initial_decision.key_factors) + [
                f"Calibrated using {n_supporting} similar past pairs ({decision})",
            ],
        )

        delta = adjusted_score - context.initial_decision.score
        context.add_log(
            self.name,
            "calibration_completed",
            f"decision={decision}, "
            f"score: {context.initial_decision.score} -> {adjusted_score} "
            f"(delta={delta:+.1f}), "
            f"n_supporting_memories={n_supporting}, "
            f"pattern: {context.calibration_output.pattern_observed[:80]}",
        )

        return context

    def _initial_to_final(self, context: SharedContext) -> FinalDecision:
        """Pass-through helper: when calibration is skipped, copy initial_decision."""
        ini = context.initial_decision
        return FinalDecision(
            score=ini.score,
            confidence=ini.confidence,
            recommendation=ini.recommendation,
            explanation=ini.explanation + " [No calibration: cold start.]",
            key_factors=list(ini.key_factors),
        )

    def _build_prompt(self, context: SharedContext) -> str:
        """Build the user prompt: current pair + initial decision + labeled history."""
        sections = []

        # Current pair context
        if context.cv_profile and context.jd_profile:
            sections.append(
                "== CURRENT PAIR (you are calibrating the decision on this pair) ==\n"
                f"CV profile: {context.cv_profile.candidate_archetype}\n"
                f"  Seniority: {context.cv_profile.seniority_level}\n"
                f"  Likely role fit: {context.cv_profile.likely_role_fit or 'unknown'}\n"
                f"JD profile: {context.jd_profile.raw_summary}\n"
                f"  Detected role: {context.jd_profile.detected_role}\n"
                f"  Required experience: {context.jd_profile.required_experience_years} years"
            )

        # Initial decision (what we're calibrating)
        ini = context.initial_decision
        sections.append(
            f"== INITIAL DECISION (Pass 1, no labels seen) ==\n"
            f"Score: {ini.score}\n"
            f"Confidence: {ini.confidence}\n"
            f"Recommendation: {ini.recommendation}\n"
            f"Explanation: {ini.explanation}"
        )

        # Reasoning context (to help the calibrator understand WHY the score was given)
        if context.reasoning_output:
            r = context.reasoning_output
            sections.append(
                f"== REASONING SUMMARY ==\n"
                f"Strengths: {'; '.join(r.strengths[:3])}\n"
                f"Gaps: {'; '.join(r.gaps[:3])}\n"
                f"Assessment: {r.overall_assessment}"
            )

        # The labeled memories — the actual calibration material.
        # Note: ground_truth_reason is intentionally empty in the current
        # methodology because the source dataset's reasons are templated
        # noise. The prompt skips the "Reason: ..." segment when empty so
        # the LLM isn't fed nonsense like 'Lacks cloud experience' on a JD
        # that never mentioned cloud.
        memory_lines = []
        for i, mem in enumerate(context.labeled_memory_entries, 1):
            label_str = "ACCEPTED" if mem.ground_truth_label else "REJECTED"
            correct_marker = "OK" if mem.was_correct else "WRONG"
            reason_segment = (
                f' Reason: "{mem.ground_truth_reason}".' if mem.ground_truth_reason else ""
            )
            memory_lines.append(
                f"[{i}] (similarity {mem.similarity_to_current:.2f}) "
                f"{mem.detected_role}: system scored {mem.system_score:.0f} "
                f"({mem.system_recommendation}). "
                f"HUMAN: {label_str}.{reason_segment} "
                f"System was {correct_marker} ({mem.error_direction})."
            )
            if mem.system_reasoning_summary:
                memory_lines.append(
                    f"    System reasoning: {mem.system_reasoning_summary[:150]}"
                )
        sections.append("== LABELED PAST PAIRS (similar to current) ==\n" + "\n".join(memory_lines))

        return "\n\n".join(sections)

    @staticmethod
    def _score_to_recommendation(score: float) -> str:
        """Mirror DecisionAgent's mapping (Tier 1.5 boundaries: 80/60/40/20)."""
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
