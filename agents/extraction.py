"""
ExtractionAgent: extracts structured entities from CV and job description.

This agent reads raw text from the shared context and uses the LLM
to extract structured information: skills, experience, education,
languages, and certifications.

Agentic behavior:
- Extracts from BOTH documents in a single process() call
- Uses LLM to identify and structure entities from free text
"""

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import ExtractedEntities
from models.shared_context import SharedContext


# System prompt template for entity extraction
EXTRACTION_SYSTEM_PROMPT = """\
You are an entity extraction agent. Your task is to extract structured
information from a document (either a CV or a job description).

Extract the following fields and return them as a JSON object:
{
  "skills": ["list of technical and soft skills"],
  "experience": ["list of work experience entries, each as a short summary"],
  "education": ["list of education qualifications"],
  "languages": ["list of languages"],
  "certifications": ["list of certifications"],
  "raw_summary": "a 1-2 sentence summary of the document"
}

Rules:
- Extract skills as individual items (e.g., "Python", not "Python, Java, SQL")
- For experience, write each entry as a single plain string combining role,
  company, and duration. Example: "Software Developer at TechCorp, 2019-2022"
  DO NOT return objects like {"role": "...", "company": "..."} — plain strings only.
- For education, write each entry as a single plain string. Example:
  "BSc Computer Science, Vilnius University, 2016". DO NOT return objects.
- Every item in skills, experience, education, languages, certifications MUST
  be a plain string, not a nested object.
- Be thorough but concise
- Return ONLY valid JSON, no extra text
"""


def _coerce_to_str_list(items) -> list[str]:
    """
    Convert list items to strings. LLMs sometimes return dicts instead of
    strings for structured fields like experience and education entries.
    This helper flattens any dict into a human-readable string so the
    Pydantic model validation doesn't fail.
    """
    if not isinstance(items, list):
        return []
    result = []
    for item in items:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # Join all values into a single string, preserving order
            parts = [str(v) for v in item.values() if v]
            result.append(" — ".join(parts))
        else:
            result.append(str(item))
    return result


class ExtractionAgent(BaseAgent):
    """
    Extracts structured entities from CV and job description texts.

    This agent demonstrates context-awareness: it reads the scenario
    field and could adapt its extraction prompt accordingly (e.g.,
    in Scenario B it might extract with taxonomy hints).
    """

    def __init__(self, llm_client: BaseLLMClient):
        self._llm = llm_client

    def process(self, context: SharedContext) -> SharedContext:
        """
        Extract entities from both CV and JD texts.

        Reads: cv_text, jd_text, scenario
        Writes: cv_entities, jd_entities
        """
        # Extract from CV
        context.add_log(self.name, "extracting_cv", "Starting CV entity extraction")
        cv_data = self._extract_from_text(context.cv_text, "CV")
        context.cv_entities = ExtractedEntities(**cv_data)
        context.add_log(
            self.name,
            "cv_extracted",
            f"Found {len(context.cv_entities.skills)} skills, "
            f"{len(context.cv_entities.experience)} experience entries",
        )

        # Extract from JD
        context.add_log(self.name, "extracting_jd", "Starting JD entity extraction")
        jd_data = self._extract_from_text(context.jd_text, "job description")
        context.jd_entities = ExtractedEntities(**jd_data)
        context.add_log(
            self.name,
            "jd_extracted",
            f"Found {len(context.jd_entities.skills)} skills, "
            f"{len(context.jd_entities.experience)} experience entries",
        )

        return context

    def _extract_from_text(self, text: str, doc_type: str) -> dict:
        """
        Use LLM to extract entities from a single document.

        Args:
            text: The document text
            doc_type: "CV" or "job description" (used in the prompt)

        Returns:
            Parsed JSON dict matching ExtractedEntities schema
        """
        user_prompt = f"Extract entities from this {doc_type}:\n\n{text}"

        result = self._llm.chat_json(EXTRACTION_SYSTEM_PROMPT, user_prompt)

        # Coerce all list fields to list[str] — LLMs sometimes return dicts
        # for structured entries like experience/education despite the prompt.
        return {
            "skills": _coerce_to_str_list(result.get("skills", [])),
            "experience": _coerce_to_str_list(result.get("experience", [])),
            "education": _coerce_to_str_list(result.get("education", [])),
            "languages": _coerce_to_str_list(result.get("languages", [])),
            "certifications": _coerce_to_str_list(result.get("certifications", [])),
            "raw_summary": str(result.get("raw_summary", "")),
        }
