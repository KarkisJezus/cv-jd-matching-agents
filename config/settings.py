"""
Application settings loaded from environment variables.

Uses python-dotenv to load from a .env file in the project root.
This keeps API keys and configuration out of source code.

Usage:
    from config.settings import settings
    client = OpenAI(api_key=settings.openai_api_key)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """Simple settings container. Reads from environment variables."""

    # ── LLM settings ─────────────────────────────────────────────

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def openai_base_url(self) -> str | None:
        """Custom base URL for OpenAI-compatible APIs (Ollama, LM Studio, etc.)."""
        return os.getenv("OPENAI_BASE_URL", None)

    @property
    def llm_model(self) -> str:
        """Which OpenAI model to use for reasoning."""
        return os.getenv("LLM_MODEL", "gpt-4o-mini")

    @property
    def llm_temperature(self) -> float:
        return float(os.getenv("LLM_TEMPERATURE", "0.3"))

    @property
    def llm_max_tokens(self) -> int:
        return int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # ── Embedding settings ────────────────────────────────────────

    @property
    def embedding_model(self) -> str:
        """Which sentence-transformers model to use."""
        return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Matching settings ────────────────────────────────────────

    @property
    def similarity_threshold(self) -> float:
        """Minimum cosine similarity to consider a skill match."""
        return float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

    # ── Memory settings (Scenario C) ─────────────────────────────

    @property
    def memory_dir(self) -> str:
        """Directory where memory store files are saved."""
        return os.getenv("MEMORY_DIR", str(_project_root / "data" / "memory"))

    @property
    def memory_top_k(self) -> int:
        """How many past memories to retrieve per query."""
        return int(os.getenv("MEMORY_TOP_K", "3"))

    @property
    def memory_min_similarity(self) -> float:
        """Minimum similarity to include a memory in retrieval results."""
        return float(os.getenv("MEMORY_MIN_SIMILARITY", "0.3"))

    # ── Orchestrator settings ────────────────────────────────────

    @property
    def max_revisions(self) -> int:
        """Maximum reflection-revision cycles."""
        return int(os.getenv("MAX_REVISIONS", "2"))

    # ── Judge / auditor settings (provider-agnostic) ─────────────
    # Designed so switching judge providers (DeepSeek <-> Gemini <-> ...) is
    # three env-var changes, not a code change. Both providers expose
    # OpenAI-compatible endpoints, so we reuse LLMClient with a different
    # base_url + api_key.

    @property
    def judge_api_key(self) -> str:
        """API key for the judge provider. Falls back to OPENAI_API_KEY only
        if no JUDGE_API_KEY is set — but using the same provider as the system
        violates the cross-family rule (self-enhancement bias). Prefer to set
        JUDGE_API_KEY explicitly."""
        return os.getenv("JUDGE_API_KEY", "")

    @property
    def judge_base_url(self) -> str:
        """Base URL for the judge provider. Defaults to DeepSeek's OpenAI-compatible
        endpoint. For Gemini, set: https://generativelanguage.googleapis.com/v1beta/openai/"""
        return os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com/v1")

    @property
    def judge_model(self) -> str:
        """Judge model name. DeepSeek default: 'deepseek-chat' (V3.2-style chat).
        For Gemini, set: 'gemini-2.5-pro' or 'gemini-2.5-flash'."""
        return os.getenv("JUDGE_MODEL", "deepseek-chat")

    @property
    def judge_temperature(self) -> float:
        """Judge temperature. Lower than the system's because we want consistent
        verdicts, not creative ones."""
        return float(os.getenv("JUDGE_TEMPERATURE", "0.1"))

    @property
    def judge_max_tokens(self) -> int:
        return int(os.getenv("JUDGE_MAX_TOKENS", "1500"))

    def validate(self) -> list[str]:
        """Check which required settings are missing. Returns list of issues."""
        issues = []
        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY not set (LLM calls will fail)")
        return issues


# Singleton instance
settings = Settings()
