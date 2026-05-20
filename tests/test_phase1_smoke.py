"""
Phase 1 smoke tests.

Verifies that all foundation components work correctly:
- SharedContext can be created and serialized
- BaseAgent interface works
- MockLLMClient returns responses
- EmbeddingSimilarity computes real similarities
"""

import json
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.shared_context import SharedContext
from models.entities import (
    ExtractedEntities,
    SimilarityScores,
    SkillMatch,
    ReasoningOutput,
    FinalDecision,
    LogEntry,
)
from agents.base import BaseAgent
from llm.client import MockLLMClient
from embeddings.similarity import EmbeddingSimilarity


# ──────────────────────────────────────────────
# Test 1: SharedContext creation and serialization
# ──────────────────────────────────────────────

def test_shared_context_creation():
    """SharedContext can be created with minimal input."""
    ctx = SharedContext(
        cv_text="I know Python and ML.",
        jd_text="We need a Python developer with ML experience.",
        scenario="A",
    )
    assert ctx.cv_text == "I know Python and ML."
    assert ctx.scenario == "A"
    assert ctx.cv_entities is None
    assert ctx.needs_revision is False
    assert ctx.logs == []
    print("  [PASS] SharedContext creation")


def test_shared_context_logging():
    """SharedContext logging works."""
    ctx = SharedContext(cv_text="test", jd_text="test")
    ctx.add_log("TestAgent", "did something", details="detail here")
    assert len(ctx.logs) == 1
    assert ctx.logs[0].agent_name == "TestAgent"
    assert ctx.logs[0].action == "did something"
    print("  [PASS] SharedContext logging")


def test_shared_context_serialization():
    """SharedContext can be serialized to JSON and back."""
    ctx = SharedContext(cv_text="test cv", jd_text="test jd", scenario="B")
    ctx.cv_entities = ExtractedEntities(
        skills=["Python", "ML"],
        experience=["3 years"],
        education=["BSc CS"],
    )
    ctx.add_log("Test", "serialization test")

    # Serialize to JSON
    json_str = ctx.model_dump_json(indent=2)
    data = json.loads(json_str)

    assert data["cv_text"] == "test cv"
    assert data["cv_entities"]["skills"] == ["Python", "ML"]
    assert len(data["logs"]) == 1

    # Deserialize back
    ctx2 = SharedContext.model_validate_json(json_str)
    assert ctx2.cv_entities.skills == ["Python", "ML"]
    print("  [PASS] SharedContext serialization round-trip")


def test_shared_context_skill_getter():
    """get_skills_for_matching returns correct skills based on state."""
    ctx = SharedContext(cv_text="test", jd_text="test")

    # No entities yet -> empty
    cv_skills, jd_skills = ctx.get_skills_for_matching()
    assert cv_skills == []
    assert jd_skills == []

    # With raw entities
    ctx.cv_entities = ExtractedEntities(skills=["Python", "ML"])
    ctx.jd_entities = ExtractedEntities(skills=["Java", "AI"])
    cv_skills, jd_skills = ctx.get_skills_for_matching()
    assert cv_skills == ["Python", "ML"]
    assert jd_skills == ["Java", "AI"]

    print("  [PASS] SharedContext skill getter")


# ──────────────────────────────────────────────
# Test 2: BaseAgent interface
# ──────────────────────────────────────────────

class DummyAgent(BaseAgent):
    """A minimal agent for testing the base class."""

    def process(self, context: SharedContext) -> SharedContext:
        context.add_log(self.name, "processed")
        return context


def test_base_agent():
    """BaseAgent execute() wraps process() with logging."""
    agent = DummyAgent()
    ctx = SharedContext(cv_text="test", jd_text="test")

    assert agent.name == "DummyAgent"

    ctx = agent.execute(ctx)

    # Should have: started, processed (from our agent), completed
    assert len(ctx.logs) == 3
    assert ctx.logs[0].action == "started"
    assert ctx.logs[1].action == "processed"
    assert ctx.logs[2].action == "completed"
    print("  [PASS] BaseAgent interface")


# ──────────────────────────────────────────────
# Test 3: MockLLMClient
# ──────────────────────────────────────────────

def test_mock_llm_client():
    """MockLLMClient returns structured responses matching agent schemas."""
    client = MockLLMClient()

    # Extraction response (uses role phrase from ExtractionAgent's prompt)
    text = client.chat("You are an entity extraction agent.", "I know Python.")
    assert isinstance(text, str)
    assert len(text) > 0

    data = client.chat_json("You are an entity extraction agent.", "I know Python.")
    assert isinstance(data, dict)
    assert "skills" in data
    assert "Python" in data["skills"]

    # Reasoning response
    data = client.chat_json("You are a reasoning agent that analyzes the match", "test")
    assert "strengths" in data
    assert "suggested_score" in data

    # Decision response
    data = client.chat_json("You are the final decision agent.", "test")
    assert "score" in data
    assert "recommendation" in data

    print("  [PASS] MockLLMClient responses")


# ──────────────────────────────────────────────
# Test 4: EmbeddingSimilarity
# ──────────────────────────────────────────────

def test_embedding_similarity():
    """EmbeddingSimilarity computes real semantic similarity."""
    sim = EmbeddingSimilarity()

    cv_skills = ["Python programming", "machine learning", "data analysis"]
    jd_skills = ["Python development", "deep learning", "SQL databases"]

    result = sim.find_best_matches(cv_skills, jd_skills)

    assert isinstance(result, SimilarityScores)
    assert 0.0 <= result.overall_score <= 1.0
    assert len(result.skill_matches) == len(jd_skills)
    assert result.total_jd_skills == 3

    # "Python programming" should match "Python development" well
    python_match = next(m for m in result.skill_matches if m.jd_skill == "Python development")
    assert python_match.similarity > 0.7, f"Python match too low: {python_match.similarity}"

    # "machine learning" should match "deep learning" reasonably
    ml_match = next(m for m in result.skill_matches if m.jd_skill == "deep learning")
    assert ml_match.similarity > 0.4, f"ML match too low: {ml_match.similarity}"

    print(f"  [PASS] EmbeddingSimilarity (overall={result.overall_score:.3f})")
    for m in result.skill_matches:
        print(f"         {m.cv_skill:25s} <-> {m.jd_skill:25s} = {m.similarity:.3f} ({m.match_type})")


def test_embedding_empty_input():
    """EmbeddingSimilarity handles empty inputs gracefully."""
    sim = EmbeddingSimilarity()

    result = sim.find_best_matches([], ["Python"])
    assert result.overall_score == 0.0
    assert result.matched_skills_count == 0

    result = sim.find_best_matches(["Python"], [])
    assert result.overall_score == 0.0

    print("  [PASS] EmbeddingSimilarity empty input handling")


# ──────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Phase 1 Smoke Tests ===\n")

    print("SharedContext tests:")
    test_shared_context_creation()
    test_shared_context_logging()
    test_shared_context_serialization()
    test_shared_context_skill_getter()

    print("\nBaseAgent tests:")
    test_base_agent()

    print("\nMockLLMClient tests:")
    test_mock_llm_client()

    print("\nEmbeddingSimilarity tests (will download model on first run):")
    test_embedding_similarity()
    test_embedding_empty_input()

    print("\n=== All Phase 1 tests passed! ===\n")
