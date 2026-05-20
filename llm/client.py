"""
LLM client wrapper with mock support for development.

Provides a unified interface for LLM calls. During development,
you can use MockLLMClient to avoid API costs. Switch to the real
OpenAI client when ready for actual experiments.

Design decisions:
- Simple wrapper, not an abstraction layer for multiple providers
- JSON mode support for structured extraction
- Mock client returns deterministic, realistic results for testing
"""

import ast
import json
from abc import ABC, abstractmethod

from openai import OpenAI

from config.settings import settings


def _parse_llm_json(text: str) -> dict:
    """
    Parse JSON from an LLM response, with multiple fallback strategies.

    Smaller open-weight models (Qwen 7B, Llama 8B, Phi) occasionally produce
    *almost* valid JSON: a missing comma, a trailing comma, single quotes,
    unquoted keys, or extra prose around the JSON block. Strict json.loads()
    fails on these.

    Fallback chain (each step only runs if the previous failed):
        1. Strip markdown code fences and surrounding prose
        2. Strict json.loads()
        3. json-repair library (handles missing commas, trailing commas,
           single quotes, unquoted keys, truncated JSON)
        4. ast.literal_eval (handles Python-style dicts as a last resort)
        5. Raise the original JSONDecodeError if every repair fails

    Returns a dict. If the LLM returned a non-dict (list/str/null), returns
    a dict with the value under the "raw_text" key for downstream agents to
    handle gracefully.
    """
    if not text:
        return {}

    # Step 1: extract from markdown fence + strip surrounding prose
    cleaned = text.strip()
    if "{" in cleaned and "}" in cleaned:
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}") + 1
        cleaned = cleaned[first_brace:last_brace]

    # Step 2: strict JSON
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else {"raw_text": text}
    except json.JSONDecodeError as strict_error:
        original_error = strict_error

    # Step 3: json-repair library
    try:
        from json_repair import repair_json
        repaired = repair_json(cleaned, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
        # Sometimes json-repair returns a list/str — wrap it
        return {"raw_text": text, "repair_result": repaired}
    except (ImportError, ValueError, Exception):
        pass

    # Step 4: ast.literal_eval (handles Python-style dicts)
    try:
        result = ast.literal_eval(cleaned)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    # All fallbacks failed — raise the original error so the runner's
    # try/except can record it and continue
    raise original_error


class BaseLLMClient(ABC):
    """Interface for LLM clients (real and mock)."""

    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the response text."""
        ...

    @abstractmethod
    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a chat completion request and parse the response as JSON."""
        ...


class LLMClient(BaseLLMClient):
    """Real OpenAI-based LLM client."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.base_url = base_url or settings.openai_base_url

        # Token usage tracking for cost analysis
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0

        # Local LLM servers (Ollama, LM Studio) don't need a real API key
        if not self.api_key and not self.base_url:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY in .env or pass api_key parameter. "
                "For local LLMs, set OPENAI_BASE_URL instead."
            )

        client_kwargs = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            # Local servers don't validate API keys, but the SDK requires one
            if not self.api_key:
                self.api_key = "local-llm"
        client_kwargs["api_key"] = self.api_key

        self._client = OpenAI(**client_kwargs)

    def _track_usage(self, response) -> None:
        """Track token usage from an API response."""
        self._total_calls += 1
        if hasattr(response, "usage") and response.usage:
            self._total_prompt_tokens += response.usage.prompt_tokens or 0
            self._total_completion_tokens += response.usage.completion_tokens or 0

    @property
    def usage(self) -> dict:
        """Return cumulative token usage statistics."""
        return {
            "total_calls": self._total_calls,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    def reset_usage(self) -> None:
        """Reset token usage counters (e.g., between evaluation pairs)."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request and return the response as plain text."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        self._track_usage(response)
        return response.choices[0].message.content or ""

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Send a chat request with JSON mode and parse the response."""
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # Some local LLM servers don't support response_format; try with it first
        try:
            kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**kwargs)
        except Exception:
            # Fallback: request JSON without response_format constraint
            kwargs.pop("response_format", None)
            response = self._client.chat.completions.create(**kwargs)

        self._track_usage(response)
        text = response.choices[0].message.content or "{}"
        return _parse_llm_json(text)


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for development and testing.

    Returns deterministic responses that match the exact JSON schemas
    our agents expect. This lets you run the full pipeline without
    API keys or network access.

    The mock detects which agent is calling based on system prompt
    role phrases, and returns an appropriately structured response.

    For reflection testing, the mock tracks call counts so that:
    - First reflection call returns is_consistent=False (triggers revision)
    - Subsequent reflection calls return is_consistent=True (approves)
    This lets you test the full reflection loop deterministically.
    """

    def __init__(self):
        self._reflection_call_count = 0
        self._reasoning_call_count = 0

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Return a mock text response."""
        prompt_lower = system_prompt.lower()

        # Match agent role phrases (order matters — most specific first)
        # Each agent has a distinctive role description in its system prompt

        # Tier 2 profiling agents (must come BEFORE generic 'extraction' check
        # since the profiling prompts may contain related words)
        if "cv profiling agent" in prompt_lower:
            return self._mock_cv_profile_json_str()
        elif "role classification agent" in prompt_lower:
            return self._mock_role_classification_json_str()
        elif "jd profiling agent" in prompt_lower:
            return self._mock_jd_profile_json_str()

        # Tier 2 calibration agent (Pass 2)
        elif "calibration agent" in prompt_lower:
            return self._mock_calibration_json_str()

        # Legacy agents (Tier 1)
        elif "entity extraction" in prompt_lower or "extraction agent" in prompt_lower:
            return self._mock_extraction_json_str()
        elif "enrichment agent" in prompt_lower or "normalizes skill names" in prompt_lower:
            return self._mock_enrichment_json_str(user_prompt)
        elif "final decision" in prompt_lower or "decision agent" in prompt_lower:
            return self._mock_decision_json_str()
        elif "reflection" in prompt_lower or "review the reasoning" in prompt_lower:
            self._reflection_call_count += 1
            return self._mock_reflection_json_str()
        elif "reasoning agent" in prompt_lower or "analyzes the match" in prompt_lower:
            self._reasoning_call_count += 1
            return self._mock_reasoning_json_str()
        else:
            return "Mock LLM response for: " + user_prompt[:100]

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Return a mock JSON response as a parsed dict."""
        text = self.chat(system_prompt, user_prompt)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_text": text}

    # ── Mock responses matching exact agent schemas ──────────────

    def _mock_extraction_json_str(self) -> str:
        """Matches ExtractedEntities schema."""
        return json.dumps({
            "skills": [
                "Python",
                "Machine Learning",
                "Data Analysis",
                "SQL",
                "REST API development",
                "Docker",
                "Git",
            ],
            "experience": [
                "3 years as Software Developer at TechCorp",
                "1 year as Junior Developer at StartupXYZ",
            ],
            "education": [
                "BSc Computer Science, Vilnius University",
            ],
            "languages": ["English", "Lithuanian"],
            "certifications": ["AWS Cloud Practitioner"],
            "raw_summary": (
                "Python developer with 3 years experience in software development, "
                "machine learning, and data analysis. Has built ML models and REST APIs."
            ),
        })

    def _mock_enrichment_json_str(self, user_prompt: str) -> str:
        """
        Matches the enrichment agent's expected schema.

        Parses unresolved skill names from the user prompt and returns
        normalized versions. This simulates what the LLM would do for
        skills not found in the local taxonomy.
        """
        # Extract skill names from the user prompt ("- SkillName" lines)
        unresolved = []
        for line in user_prompt.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                unresolved.append(line[2:].strip())

        # Build normalized versions for common unresolved patterns
        normalized = []
        notes = []
        for skill in unresolved:
            normalized.append({
                "original": skill,
                "normalized": skill.lower().strip(),
                "category": "other",
            })

        if not normalized:
            # Fallback if no skills were parsed
            notes.append("No unresolved skills to normalize")

        return json.dumps({
            "normalized_skills": normalized,
            "notes": notes,
        })

    def _mock_reasoning_json_str(self) -> str:
        """
        Matches ReasoningOutput schema.

        First call returns initial reasoning (score 68).
        Second call returns revised reasoning (score 62) reflecting
        the reflection agent's feedback about overvaluing cloud creds
        and underweighting the deep learning gap.
        """
        if self._reasoning_call_count <= 1:
            # Initial reasoning
            return json.dumps({
                "strengths": [
                    "Strong Python skills matching the core requirement",
                    "Hands-on ML experience with scikit-learn and PyTorch",
                    "Data processing experience with pandas and numpy",
                    "Has AWS certification showing cloud awareness",
                ],
                "gaps": [
                    "No explicit deep learning architecture experience (CNN, RNN, Transformers)",
                    "Limited MLOps experience (no mention of model versioning or CI/CD for ML)",
                    "No TensorFlow experience (only PyTorch)",
                ],
                "concerns": [
                    "Cloud experience is basic (only AWS Cloud Practitioner certification)",
                    "No distributed computing experience mentioned",
                ],
                "overall_assessment": (
                    "The candidate is a solid match for the core technical requirements "
                    "with strong Python and ML skills. The main gaps are in deep learning "
                    "specifics and MLOps practices. The candidate could grow into the role "
                    "but would need ramp-up time on advanced topics."
                ),
                "suggested_score": 68.0,
            })
        else:
            # Revised reasoning after reflection feedback
            return json.dumps({
                "strengths": [
                    "Strong Python skills matching the core requirement",
                    "Hands-on ML experience with scikit-learn and PyTorch",
                    "Data processing experience with pandas and numpy",
                ],
                "gaps": [
                    "No deep learning architecture experience (CNN, RNN, Transformers) - this is a key JD requirement",
                    "Limited MLOps experience (no model versioning or CI/CD for ML)",
                    "No TensorFlow experience (only PyTorch)",
                    "AWS Cloud Practitioner is entry-level; JD requires hands-on cloud platform experience",
                ],
                "concerns": [
                    "The deep learning gap is significant as it is a core JD requirement",
                    "Cloud experience does not meet the practical level required",
                    "No distributed computing experience mentioned",
                ],
                "overall_assessment": (
                    "The candidate has strong foundational skills in Python and classical ML, "
                    "but falls short on the deep learning and cloud requirements that are central "
                    "to this role. The AWS Cloud Practitioner certification is foundational, "
                    "not hands-on. This is a partial match requiring significant upskilling."
                ),
                "suggested_score": 62.0,
            })

    def _mock_reflection_json_str(self) -> str:
        """
        Matches ReflectionOutput schema.

        First call returns is_consistent=False to trigger a revision cycle.
        Subsequent calls return is_consistent=True to approve the reasoning.
        This allows deterministic testing of the full reflection loop.
        """
        if self._reflection_call_count <= 1:
            # First call: find issues, request revision
            return json.dumps({
                "is_consistent": False,
                "issues_found": [
                    "Suggested score of 68 seems too high given the lack of deep learning experience",
                    "AWS Cloud Practitioner is entry-level and should not count as strong cloud experience",
                ],
                "suggestions": [
                    "Lower the suggested score to 55-65 range to reflect the deep learning gap",
                    "Explicitly note that Cloud Practitioner is a foundational certification",
                ],
                "confidence": 0.45,
                "revision_reason": (
                    "The reasoning overvalues the candidate's cloud credentials "
                    "and underweights the deep learning gap relative to the JD requirements"
                ),
            })
        else:
            # Subsequent calls: approve the revised reasoning
            return json.dumps({
                "is_consistent": True,
                "issues_found": [],
                "suggestions": [
                    "Analysis is now well-calibrated with the evidence",
                ],
                "confidence": 0.88,
                "revision_reason": "",
            })

    def _mock_decision_json_str(self) -> str:
        """Matches FinalDecision schema."""
        return json.dumps({
            "score": 70.0,
            "confidence": 0.75,
            "recommendation": "good_match",
            "explanation": (
                "The candidate demonstrates strong alignment with core technical "
                "requirements including Python, ML, and data analysis. Gaps exist "
                "in deep learning specifics and MLOps practices, but the foundation "
                "is solid for growth into the role."
            ),
            "key_factors": [
                "Strong Python and ML skills match core requirements",
                "Data processing experience aligns well",
                "Deep learning and MLOps gaps are notable but addressable",
                "AWS certification partially covers cloud requirement",
            ],
        })

    # ── Tier 2 mock responses ────────────────────────────────────

    def _mock_cv_profile_json_str(self) -> str:
        """Matches CandidateProfile schema (Tier 2)."""
        return json.dumps({
            "skills": ["Python", "Machine Learning", "Data Analysis", "SQL", "Docker", "Git"],
            "experience": [
                "Software Developer at TechCorp, 2019-2022",
                "Junior Developer at StartupXYZ, 2018-2019",
            ],
            "education": ["BSc Computer Science, Vilnius University, 2018"],
            "languages": ["English", "Lithuanian"],
            "certifications": ["AWS Cloud Practitioner"],
            "seniority_level": "mid",
            "domain_expertise": ["machine learning", "data analysis"],
            "candidate_archetype": (
                "Mid-level Python/ML developer with 3 years of experience and "
                "foundational cloud awareness."
            ),
            "likely_role_fit": "machine_learning_engineer",
            "raw_summary": (
                "Mid-level developer with 3 years experience in Python, ML, and "
                "data analysis. Has built ML models and REST APIs."
            ),
        })

    def _mock_role_classification_json_str(self) -> str:
        """Stage 1 of JDProfilingAgent — picks a role from the list."""
        return json.dumps({
            "detected_role": "machine_learning_engineer",
            "role_confidence": 0.85,
            "rationale": (
                "JD describes ML model development, deployment, and Python — "
                "matches the machine_learning_engineer ESCO occupation."
            ),
        })

    def _mock_jd_profile_json_str(self) -> str:
        """Matches IdealCandidateProfile schema (Tier 2, with ESCO context)."""
        return json.dumps({
            "required_skills": ["Python", "Machine Learning", "TensorFlow", "Cloud Platforms"],
            "typical_role_skills": [
                "Python", "scikit-learn", "PyTorch or TensorFlow", "statistics",
                "MLOps fundamentals",
            ],
            "required_experience_years": 3.0,
            "required_education": "BSc or MSc in CS, mathematics, or related quantitative field",
            "detected_role": "machine_learning_engineer",
            "role_confidence": 0.85,
            "esco_code": "S2.5.4",
            "seniority_required": "mid",
            "key_responsibilities": [
                "design and deploy ML models",
                "feature engineering",
                "model performance monitoring",
            ],
            "raw_summary": (
                "Mid-level ML engineer with strong Python/ML fundamentals and "
                "cloud deployment experience. Should be comfortable with the full "
                "ML lifecycle from feature engineering to production monitoring."
            ),
        })

    def _mock_calibration_json_str(self) -> str:
        """Matches CalibrationOutput schema (Tier 2 Pass 2)."""
        return json.dumps({
            "calibration_decision": "lower",
            "adjusted_score": 62.0,
            "adjusted_recommendation": "good_match",
            "rationale": (
                "Past similar profiles (3 retrieved) were system-scored 70-75 "
                "but human-rejected for missing deep learning depth. Lowering "
                "score to reflect this calibration pattern."
            ),
            "confidence": 0.75,
            "pattern_observed": "system over-scores ~10pt for mid-level ML profiles lacking DL depth",
            "n_supporting_memories": 3,
        })
