"""
JDProfilingAgent: builds an IdealCandidateProfile from a JD with ESCO role context.

Tier 2 architecture replacement for the JD half of the original ExtractionAgent
PLUS the ContextEnrichmentAgent. Combines extraction + role classification +
ESCO-based contextual enrichment into a single profile of "the ideal candidate
this JD is seeking."

Two-stage flow:
1. Role classification (cheap LLM call): given the JD text and a list of
   curated ESCO role names, classify which role this JD describes.
2. JD profiling (main LLM call): given the JD text AND the ESCO context for
   the detected role, build the IdealCandidateProfile.

This keeps the profiling prompt focused — it only sees ESCO data for ONE role,
not all 100+. Stage 1's prompt is small (just role names + brief summaries),
so the cost overhead is modest.

If role classification confidence is below a threshold, we fall back to the
'generic_professional' profile and the JD-only context. The agent still
produces a usable profile even when the role isn't in the ESCO file.
"""

import json
from pathlib import Path

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import IdealCandidateProfile
from models.shared_context import SharedContext


# Default location of the ESCO occupations data
DEFAULT_ESCO_PATH = Path(__file__).parent.parent / "data" / "esco_occupations.json"

# Below this role-classification confidence, we use the generic fallback.
# Lowered from 0.50 to 0.45 to accept slightly less-confident classifications
# (e.g., when an augmented JD mentions multiple domains). The trade-off: a few
# more wrong classifications in exchange for less reliance on the generic
# fallback profile. Re-tune if the wrong-classification rate climbs.
ROLE_CLASSIFICATION_MIN_CONFIDENCE = 0.45


def _coerce_to_str_list(items) -> list[str]:
    """LLMs sometimes return dicts inside lists; flatten to plain strings."""
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            parts = [str(v) for v in item.values() if v]
            result.append(" — ".join(parts))
        else:
            result.append(str(item))
    return result


# Stage 1: role classification prompt
ROLE_CLASSIFICATION_PROMPT = """\
You are a role classification agent. Given a job description, identify which
role from the provided list it most closely describes.

Return a JSON object:
{
  "detected_role": "<one of the role keys from the list, or empty string if none fit>",
  "role_confidence": <float 0-1, your confidence in the classification>,
  "rationale": "<one sentence explaining the choice>"
}

Available roles (use the snake_case key, NOT the human label):
{role_list}

Rules:
- detected_role MUST be one of the snake_case keys above, or "" if no role fits.
- role_confidence < 0.5 means you are NOT sure — better to return an empty role than guess wrong.
- Pick a role only if the JD's responsibilities and required skills clearly match.
- Return ONLY valid JSON.
"""


# Stage 2: JD profiling prompt (uses ESCO role context)
JD_PROFILING_SYSTEM_PROMPT = """\
You are a JD profiling agent. Your task is to build a profile of the IDEAL
CANDIDATE the job description is seeking. Use both the JD text and the
ESCO role context to produce a richer profile than the JD text alone.

The ESCO role context tells you what is TYPICALLY expected for this role
(per the European ESCO standard). Use it to enrich the profile, but only
include attributes the JD either explicitly mentions or that are clearly
implied by the role.

Return a JSON object with these exact fields:
{
  "required_skills": ["skills the JD EXPLICITLY requires"],
  "typical_role_skills": ["skills typically expected for this role per ESCO context — even if the JD doesn't say so"],
  "required_experience_years": <float, years of experience the JD requires; 0 if not specified>,
  "required_education": "education requirements from the JD as a single string",
  "detected_role": "<the role key passed to you, e.g. 'machine_learning_engineer'>",
  "role_confidence": <float 0-1, propagate from stage 1>,
  "esco_code": "<the ESCO code passed to you>",
  "seniority_required": "junior" | "mid" | "senior" | "lead" | "unknown",
  "key_responsibilities": ["main responsibilities the role entails (combine JD text + ESCO context)"],
  "raw_summary": "2-3 sentence summary of the ideal candidate this JD is seeking"
}

Rules:
- required_skills should ONLY include skills the JD literally mentions. Don't pad it with ESCO-typical skills.
- typical_role_skills should include the ESCO-typical skills (some may overlap with required_skills, that's fine).
- Distinguish required (JD-explicit) from typical (ESCO-derived) so downstream agents can weight them differently.
- Each item in lists must be a plain string, not a dict.
- seniority_required: estimate from JD's years requirement and role title (e.g., 'Senior X' → senior).
- raw_summary should describe the ideal candidate, not just paraphrase the JD.
- Return ONLY valid JSON.
"""

GENERIC_FALLBACK_PROMPT = """\
You are a JD profiling agent. The role could not be confidently classified
into a known ESCO occupation, so you have no role-specific context. Build
the IdealCandidateProfile using ONLY the JD text.

Return a JSON object with these exact fields:
{
  "required_skills": ["skills the JD EXPLICITLY requires"],
  "typical_role_skills": [],
  "required_experience_years": <float; 0 if not specified>,
  "required_education": "education requirements from the JD",
  "detected_role": "generic_professional",
  "role_confidence": 0.0,
  "esco_code": "GENERIC",
  "seniority_required": "junior" | "mid" | "senior" | "lead" | "unknown",
  "key_responsibilities": ["responsibilities mentioned in the JD"],
  "raw_summary": "2-3 sentence summary of the ideal candidate"
}

Rules:
- typical_role_skills must be empty (no ESCO context available).
- Each item in lists must be a plain string.
- Return ONLY valid JSON.
"""


class JDProfilingAgent(BaseAgent):
    """
    Builds an IdealCandidateProfile from the JD with ESCO role context.

    Two-stage execution:
    1. Stage 1 (role classification): classify the JD's role from ESCO list
    2. Stage 2 (profiling): build the profile, optionally with role-specific context

    Reads:  context.cv_text (no), context.jd_text (yes), scenario
    Writes: context.jd_profile

    The agent's behavior depends on the scenario:
    - Scenario A: skip ESCO entirely (use_esco_context=False at construction)
    - Scenario B/C: use ESCO context (use_esco_context=True)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        use_esco_context: bool = True,
        esco_path: Path | str | None = None,
    ):
        self._llm = llm_client
        self._use_esco = use_esco_context
        self._esco_path = Path(esco_path) if esco_path else DEFAULT_ESCO_PATH
        self._esco_data = self._load_esco_data() if use_esco_context else {}

    def _load_esco_data(self) -> dict:
        """Load the ESCO occupations data file."""
        if not self._esco_path.exists():
            return {}
        try:
            data = json.loads(self._esco_path.read_text(encoding="utf-8"))
            return data.get("occupations", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _build_role_list_text(self) -> str:
        """Build the brief role list text shown in Stage 1's prompt."""
        if not self._esco_data:
            return ""
        lines = []
        for key, occ in self._esco_data.items():
            if key == "generic_professional":
                continue  # don't show fallback in classification options
            label = occ.get("preferred_label", key)
            description = occ.get("description", "")
            skills_brief = ", ".join(occ.get("typical_skills", [])[:3])
            lines.append(f"- {key}: {label}. {description} Typical skills: {skills_brief}")
        return "\n".join(lines)

    def _classify_role(self, jd_text: str) -> tuple[str, float]:
        """
        Stage 1: classify the JD's role.

        Returns (detected_role, confidence). detected_role is "" if no good match.
        """
        if not self._esco_data:
            return ("", 0.0)

        role_list_text = self._build_role_list_text()
        prompt = ROLE_CLASSIFICATION_PROMPT.replace("{role_list}", role_list_text)

        result = self._llm.chat_json(prompt, f"Job description:\n\n{jd_text}")
        detected = str(result.get("detected_role", "")).lower().strip().replace(" ", "_")
        confidence = float(result.get("role_confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        # Verify the detected role is actually in our list (LLM hallucination guard)
        if detected and detected not in self._esco_data:
            detected = ""
            confidence = 0.0

        return (detected, confidence)

    def _build_profile_with_esco(
        self,
        jd_text: str,
        detected_role: str,
        role_confidence: float,
    ) -> IdealCandidateProfile:
        """Stage 2 (with ESCO context): build the profile using role-specific context."""
        occ = self._esco_data[detected_role]
        esco_block = (
            f"ESCO ROLE CONTEXT for '{occ.get('preferred_label', detected_role)}':\n"
            f"  ESCO code: {occ.get('esco_code', '')}\n"
            f"  Description: {occ.get('description', '')}\n"
            f"  Typical skills: {', '.join(occ.get('typical_skills', []))}\n"
            f"  Typical experience: {occ.get('typical_experience_years', 0)} years\n"
            f"  Typical education: {occ.get('typical_education', '')}\n"
            f"  Typical responsibilities: {'; '.join(occ.get('typical_responsibilities', []))}\n"
        )
        user_prompt = (
            f"{esco_block}\n"
            f"JOB DESCRIPTION TEXT:\n\n{jd_text}\n\n"
            f"You have classified this role as '{detected_role}' "
            f"with confidence {role_confidence:.2f}. Use the ESCO context above "
            f"to enrich the IdealCandidateProfile, but only include attributes "
            f"that the JD either mentions or are typical for this role."
        )

        result = self._llm.chat_json(JD_PROFILING_SYSTEM_PROMPT, user_prompt)

        return IdealCandidateProfile(
            required_skills=_coerce_to_str_list(result.get("required_skills", [])),
            typical_role_skills=_coerce_to_str_list(result.get("typical_role_skills", [])),
            required_experience_years=float(result.get("required_experience_years", 0.0)),
            required_education=str(result.get("required_education", "")),
            detected_role=detected_role,
            role_confidence=role_confidence,
            esco_code=occ.get("esco_code", ""),
            seniority_required=str(result.get("seniority_required", "unknown")).lower().strip(),
            key_responsibilities=_coerce_to_str_list(result.get("key_responsibilities", [])),
            raw_summary=str(result.get("raw_summary", "")),
        )

    def _build_profile_fallback(self, jd_text: str) -> IdealCandidateProfile:
        """Stage 2 (no ESCO context): build a minimal profile from JD text alone."""
        result = self._llm.chat_json(GENERIC_FALLBACK_PROMPT, f"Job description:\n\n{jd_text}")
        return IdealCandidateProfile(
            required_skills=_coerce_to_str_list(result.get("required_skills", [])),
            typical_role_skills=[],
            required_experience_years=float(result.get("required_experience_years", 0.0)),
            required_education=str(result.get("required_education", "")),
            detected_role="generic_professional",
            role_confidence=0.0,
            esco_code="GENERIC",
            seniority_required=str(result.get("seniority_required", "unknown")).lower().strip(),
            key_responsibilities=_coerce_to_str_list(result.get("key_responsibilities", [])),
            raw_summary=str(result.get("raw_summary", "")),
        )

    def process(self, context: SharedContext) -> SharedContext:
        if not context.jd_text:
            context.add_log(self.name, "skipped", "No jd_text in context")
            return context

        # Scenario A path: no ESCO, just plain JD profiling
        if not self._use_esco or not self._esco_data:
            context.add_log(
                self.name,
                "esco_skipped",
                "ESCO context disabled — using fallback profile",
            )
            context.jd_profile = self._build_profile_fallback(context.jd_text)
        else:
            # Stage 1: classify role
            context.add_log(self.name, "role_classification_started", "Stage 1")
            detected_role, confidence = self._classify_role(context.jd_text)
            context.add_log(
                self.name,
                "role_classified",
                f"detected_role={detected_role or 'unknown'}, confidence={confidence:.2f}",
            )

            # Stage 2: profile (with or without ESCO context)
            if detected_role and confidence >= ROLE_CLASSIFICATION_MIN_CONFIDENCE:
                context.add_log(
                    self.name,
                    "profiling_with_esco",
                    f"Building profile with ESCO context for '{detected_role}'",
                )
                context.jd_profile = self._build_profile_with_esco(
                    context.jd_text, detected_role, confidence,
                )
            else:
                context.add_log(
                    self.name,
                    "profiling_fallback",
                    f"Confidence too low ({confidence:.2f}) — using fallback",
                )
                context.jd_profile = self._build_profile_fallback(context.jd_text)

        context.add_log(
            self.name,
            "profiling_completed",
            f"detected_role={context.jd_profile.detected_role}, "
            f"required_skills={len(context.jd_profile.required_skills)}, "
            f"typical_role_skills={len(context.jd_profile.typical_role_skills)}",
        )

        return context
