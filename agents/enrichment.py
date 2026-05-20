"""
ContextEnrichmentAgent: normalizes extracted skills using ESCO taxonomy + LLM.

This agent bridges the gap between raw extracted skill strings and
standardized, taxonomy-mapped skill representations. This is what
makes Scenario B more sophisticated than Scenario A:

Before enrichment (Scenario A):
  CV: "Python", "ML"   vs   JD: "Python programming", "Machine Learning"
  → Matching relies entirely on embedding similarity

After enrichment (Scenario B):
  CV: "Python" → NormalizedSkill(normalized="Python programming", esco_code="S1.1.1")
  JD: "Python programming" → NormalizedSkill(normalized="Python programming", esco_code="S1.1.1")
  → Matching can use exact ESCO code comparison + better normalized strings

The agent uses a hybrid approach:
1. Local taxonomy lookup (exact match → synonym match)
2. LLM fallback for unresolved skills (normalizes free-text)

This minimizes LLM calls while handling unknown/ambiguous skill strings.
"""

import json
from pathlib import Path

from agents.base import BaseAgent
from llm.client import BaseLLMClient
from models.entities import NormalizedEntities, NormalizedSkill
from models.shared_context import SharedContext


# ── Taxonomy loader ──────────────────────────────────────────

def load_taxonomy(path: str | None = None) -> list[dict]:
    """
    Load the ESCO skill taxonomy from JSON.

    Returns a list of taxonomy entries, each with:
      - esco_code: str
      - label: str
      - synonyms: list[str]
      - category: str
    """
    if path is None:
        # Default to data/esco_skills.json relative to project root
        path = str(Path(__file__).parent.parent / "data" / "esco_skills.json")

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("skills", [])


# ── LLM prompt for unresolved skills ─────────────────────────

ENRICHMENT_SYSTEM_PROMPT = """\
You are an enrichment agent that normalizes skill names into their canonical form.

You receive a list of skill strings that could not be resolved via taxonomy lookup.
For each skill, produce a normalized version and optionally suggest an ESCO category.

Produce a JSON object:
{
  "normalized_skills": [
    {
      "original": "<the original skill string>",
      "normalized": "<canonical, standardized skill name>",
      "category": "<optional: programming_languages, artificial_intelligence, data_science, software_engineering, cloud_platforms, soft_skills, or 'other'>"
    }
  ],
  "notes": ["<any observations about the skills, e.g., ambiguous terms, potential duplicates>"]
}

Rules:
- Normalize casing, remove abbreviation ambiguity
- Expand well-known abbreviations (e.g., "ML" → "machine learning")
- Keep the meaning identical — do not infer skills the text does not mention
- If a skill is already in canonical form, return it as-is
- Return ONLY valid JSON
"""


class ContextEnrichmentAgent(BaseAgent):
    """
    Normalizes extracted skills using ESCO taxonomy + LLM fallback.

    Agentic behavior:
    - Reads cv_entities and jd_entities from the shared context
    - Uses local taxonomy for known skills (no LLM call needed)
    - Falls back to LLM for unknown/ambiguous skills
    - Writes normalized_entities and enrichment_notes to context
    - Downstream agents (SemanticMatchingAgent) automatically use
      normalized skills via context.get_skills_for_matching()

    This demonstrates two important agentic properties:
    1. Tool selection: the agent decides whether to use local lookup
       or LLM based on what it can resolve locally
    2. Context enrichment: it adds information that wasn't in the
       original extraction, making downstream agents more effective
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        taxonomy_path: str | None = None,
    ):
        self._llm = llm_client
        self._taxonomy = load_taxonomy(taxonomy_path)
        self._build_lookup_index()

    def _build_lookup_index(self) -> None:
        """
        Build fast lookup structures from the taxonomy.

        Creates two indexes:
        - _exact_index: maps exact lowercase strings to taxonomy entries
        - _synonym_index: maps lowercase synonyms to taxonomy entries

        These enable O(1) lookup without LLM calls for known skills.
        """
        self._exact_index: dict[str, dict] = {}
        self._synonym_index: dict[str, dict] = {}

        for entry in self._taxonomy:
            # Index by canonical label
            label_lower = entry["label"].lower()
            self._exact_index[label_lower] = entry

            # Index by each synonym
            for synonym in entry.get("synonyms", []):
                self._synonym_index[synonym.lower()] = entry

    def process(self, context: SharedContext) -> SharedContext:
        """
        Normalize CV and JD skills via taxonomy lookup + LLM fallback.

        Reads: cv_entities, jd_entities
        Writes: normalized_entities, enrichment_notes
        """
        if not context.cv_entities or not context.jd_entities:
            context.add_log(
                self.name,
                "enrichment_skipped",
                "No extracted entities available for enrichment",
            )
            return context

        all_raw_cv = context.cv_entities.skills
        all_raw_jd = context.jd_entities.skills

        context.add_log(
            self.name,
            "enrichment_started",
            f"Normalizing {len(all_raw_cv)} CV skills + {len(all_raw_jd)} JD skills",
        )

        # Phase 1: Local taxonomy lookup
        cv_resolved, cv_unresolved = self._local_lookup(all_raw_cv)
        jd_resolved, jd_unresolved = self._local_lookup(all_raw_jd)

        context.add_log(
            self.name,
            "local_lookup_completed",
            f"Resolved locally: CV={len(cv_resolved)}/{len(all_raw_cv)}, "
            f"JD={len(jd_resolved)}/{len(all_raw_jd)}. "
            f"Unresolved: CV={len(cv_unresolved)}, JD={len(jd_unresolved)}",
        )

        # Phase 2: LLM fallback for unresolved skills
        all_unresolved = list(set(cv_unresolved + jd_unresolved))
        llm_resolved: dict[str, NormalizedSkill] = {}
        notes: list[str] = []

        if all_unresolved:
            llm_resolved, llm_notes = self._llm_normalize(all_unresolved)
            notes.extend(llm_notes)

            context.add_log(
                self.name,
                "llm_normalization_completed",
                f"LLM normalized {len(llm_resolved)} unresolved skills",
            )

        # Combine results: resolved skills + LLM-normalized skills
        cv_skills = cv_resolved + [
            llm_resolved.get(s, self._passthrough(s))
            for s in cv_unresolved
        ]
        jd_skills = jd_resolved + [
            llm_resolved.get(s, self._passthrough(s))
            for s in jd_unresolved
        ]

        # Write to context
        context.normalized_entities = NormalizedEntities(
            cv_skills=cv_skills,
            jd_skills=jd_skills,
        )

        # Add enrichment notes for downstream agents
        total = len(all_raw_cv) + len(all_raw_jd)
        taxonomy_matched = len(cv_resolved) + len(jd_resolved)
        coverage_pct = f"{taxonomy_matched / total:.0%}" if total > 0 else "N/A"
        notes.insert(0, (
            f"Taxonomy coverage: {taxonomy_matched}/{total} skills matched ({coverage_pct})"
        ))

        context.enrichment_notes = notes

        context.add_log(
            self.name,
            "enrichment_completed",
            f"Produced {len(cv_skills)} normalized CV skills, "
            f"{len(jd_skills)} normalized JD skills. "
            f"Notes: {len(notes)}",
        )

        return context

    def _local_lookup(
        self, skills: list[str]
    ) -> tuple[list[NormalizedSkill], list[str]]:
        """
        Attempt to resolve skills using the local taxonomy index.

        Returns:
            - resolved: list of NormalizedSkill objects for matched skills
            - unresolved: list of skill strings that couldn't be matched
        """
        resolved: list[NormalizedSkill] = []
        unresolved: list[str] = []

        for skill in skills:
            skill_lower = skill.lower().strip()

            # Try exact label match first
            if skill_lower in self._exact_index:
                entry = self._exact_index[skill_lower]
                resolved.append(self._taxonomy_to_normalized(skill, entry))
            # Try synonym match
            elif skill_lower in self._synonym_index:
                entry = self._synonym_index[skill_lower]
                resolved.append(self._taxonomy_to_normalized(skill, entry))
            else:
                unresolved.append(skill)

        return resolved, unresolved

    def _taxonomy_to_normalized(
        self, original: str, entry: dict
    ) -> NormalizedSkill:
        """Convert a taxonomy entry to a NormalizedSkill model."""
        return NormalizedSkill(
            original=original,
            normalized=entry["label"],
            esco_code=entry["esco_code"],
            esco_label=entry["label"],
            synonyms=entry.get("synonyms", []),
        )

    def _passthrough(self, skill: str) -> NormalizedSkill:
        """Create a NormalizedSkill with no taxonomy mapping (passthrough)."""
        return NormalizedSkill(
            original=skill,
            normalized=skill,
            esco_code=None,
            esco_label=None,
            synonyms=[],
        )

    def _llm_normalize(
        self, unresolved: list[str]
    ) -> tuple[dict[str, NormalizedSkill], list[str]]:
        """
        Use LLM to normalize skills that couldn't be resolved locally.

        Returns:
            - mapping: dict from original skill string to NormalizedSkill
            - notes: list of observations from the LLM
        """
        user_prompt = (
            "Please normalize these skill strings:\n"
            + "\n".join(f"- {s}" for s in unresolved)
        )

        result = self._llm.chat_json(ENRICHMENT_SYSTEM_PROMPT, user_prompt)

        mapping: dict[str, NormalizedSkill] = {}
        notes: list[str] = result.get("notes", [])

        for item in result.get("normalized_skills", []):
            original = item.get("original", "")
            normalized = item.get("normalized", original)
            mapping[original] = NormalizedSkill(
                original=original,
                normalized=normalized,
                esco_code=None,  # LLM doesn't assign ESCO codes
                esco_label=None,
                synonyms=[],
            )

        return mapping, notes
