"""
CVProfilingAgent: builds a CandidateProfile from the CV text.

Tier 2 architecture replacement for the CV half of the original ExtractionAgent.
This agent goes beyond raw entity extraction — it asks the LLM to interpret the
CV and answer "what kind of candidate is this?" with structured fields like
seniority_level, domain_expertise, candidate_archetype, and likely_role_fit.

The profile is consumed by:
- SemanticMatchingAgent (uses skills for cosine similarity)
- ReasoningAgent (uses the full profile to compare against the JD profile)
- DecisionAgent (uses the archetype for the final explanation)

Why "profile" not "extraction":
A list of skills is data. A profile is interpretation. The interpretive layer
is what gives ReasoningAgent a richer target to match against than just
"skills the candidate listed."
"""

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import CandidateProfile
from models.shared_context import SharedContext


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


CV_PROFILING_SYSTEM_PROMPT = """\
You are a CV profiling agent. Your job is to read a candidate's CV and produce
a structured profile of WHO this candidate is — not just a list of skills, but
an interpretive summary of their seniority, domain expertise, and archetype.

Return a JSON object with these exact fields:
{
  "skills": ["list of technical and soft skills"],
  "experience": ["list of work experience entries, each as a single plain string"],
  "education": ["list of education qualifications, each as a single plain string"],
  "languages": ["list of spoken/written languages"],
  "certifications": ["list of certifications"],
  "seniority_level": "junior" | "mid" | "senior" | "lead" | "unknown",
  "domain_expertise": ["domain areas the candidate appears strong in (e.g., 'machine learning', 'cloud infrastructure', 'frontend web', 'data engineering')"],
  "candidate_archetype": "one-sentence summary of who this candidate is (e.g., 'mid-level ML engineer with strong production deployment experience')",
  "likely_role_fit": "the role this candidate's profile most resembles (e.g., 'machine_learning_engineer', 'data_scientist', 'devops_engineer', 'software_engineer'). Use lowercase snake_case. If unclear, use empty string.",
  "raw_summary": "2-3 sentence freeform summary of the candidate"
}

Rules:
- Each item in skills/experience/education/languages/certifications/domain_expertise must be a plain STRING, never a dict or object.
- For experience entries, format as plain strings like 'Software Developer at TechCorp, 2019-2022'. Do NOT return nested objects.
- seniority_level must be one of: junior, mid, senior, lead, unknown. Estimate from total years of experience and role titles.
- candidate_archetype is the most important interpretive field. It captures the 'shape' of the candidate beyond a skill list.
- likely_role_fit should be lowercase snake_case (e.g., 'data_engineer', not 'Data Engineer').
- Be evidence-based. Don't infer skills the CV doesn't mention. Don't claim seniority not supported by experience.
- Return ONLY valid JSON, no extra text.
"""


class CVProfilingAgent(BaseAgent):
    """
    Builds a CandidateProfile from the CV.

    Reads:  context.cv_text
    Writes: context.cv_profile
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._llm = llm_client

    def process(self, context: SharedContext) -> SharedContext:
        if not context.cv_text:
            context.add_log(self.name, "skipped", "No cv_text in context")
            return context

        user_prompt = f"Build a candidate profile from this CV:\n\n{context.cv_text}"

        context.add_log(self.name, "profiling_started", "Building CandidateProfile from CV")

        result = self._llm.chat_json(CV_PROFILING_SYSTEM_PROMPT, user_prompt)

        # Validate seniority_level
        seniority = str(result.get("seniority_level", "unknown")).lower().strip()
        if seniority not in ("junior", "mid", "senior", "lead", "unknown"):
            seniority = "unknown"

        context.cv_profile = CandidateProfile(
            skills=_coerce_to_str_list(result.get("skills", [])),
            experience=_coerce_to_str_list(result.get("experience", [])),
            education=_coerce_to_str_list(result.get("education", [])),
            languages=_coerce_to_str_list(result.get("languages", [])),
            certifications=_coerce_to_str_list(result.get("certifications", [])),
            seniority_level=seniority,
            domain_expertise=_coerce_to_str_list(result.get("domain_expertise", [])),
            candidate_archetype=str(result.get("candidate_archetype", "")),
            likely_role_fit=str(result.get("likely_role_fit", "")).lower().strip().replace(" ", "_"),
            raw_summary=str(result.get("raw_summary", "")),
        )

        context.add_log(
            self.name,
            "profiling_completed",
            f"seniority={seniority}, "
            f"likely_role={context.cv_profile.likely_role_fit or 'unknown'}, "
            f"skills={len(context.cv_profile.skills)}, "
            f"domains={len(context.cv_profile.domain_expertise)}",
        )

        return context
