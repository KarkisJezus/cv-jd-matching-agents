"""
Phase 2 tests: individual agents and end-to-end orchestration.

Tests each agent in isolation with mock LLM, then runs the full
Scenario A pipeline through the orchestrator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.extraction import ExtractionAgent
from agents.matching import SemanticMatchingAgent
from agents.reasoning import ReasoningAgent
from agents.decision import DecisionAgent
from embeddings.similarity import EmbeddingSimilarity
from llm.client import MockLLMClient
from models.shared_context import SharedContext
from orchestrator.orchestrator import Orchestrator


SAMPLE_CV = "I know Python, Machine Learning, and SQL. 3 years experience."
SAMPLE_JD = "We need Python, Deep Learning, and Docker skills. 2+ years required."


def _make_context() -> SharedContext:
    return SharedContext(cv_text=SAMPLE_CV, jd_text=SAMPLE_JD, scenario="A")


# ──────────────────────────────────────────────
# Individual agent tests
# ──────────────────────────────────────────────

def test_extraction_agent():
    """ExtractionAgent extracts entities from both CV and JD."""
    agent = ExtractionAgent(MockLLMClient())
    ctx = _make_context()

    ctx = agent.execute(ctx)

    assert ctx.cv_entities is not None, "CV entities should be set"
    assert ctx.jd_entities is not None, "JD entities should be set"
    assert len(ctx.cv_entities.skills) > 0, "Should extract CV skills"
    assert len(ctx.jd_entities.skills) > 0, "Should extract JD skills"
    assert ctx.cv_entities.raw_summary != "", "Should have CV summary"

    # Check logging happened
    log_actions = [log.action for log in ctx.logs]
    assert "started" in log_actions
    assert "cv_extracted" in log_actions
    assert "jd_extracted" in log_actions
    print("  [PASS] ExtractionAgent")


def test_matching_agent():
    """SemanticMatchingAgent computes similarity using extracted skills."""
    mock_llm = MockLLMClient()
    ctx = _make_context()

    # First run extraction so we have entities
    ctx = ExtractionAgent(mock_llm).execute(ctx)

    # Then run matching
    agent = SemanticMatchingAgent()
    ctx = agent.execute(ctx)

    assert ctx.similarity_scores is not None, "Similarity scores should be set"
    assert ctx.similarity_scores.overall_score > 0.0, "Should compute non-zero similarity"
    assert len(ctx.similarity_scores.skill_matches) > 0, "Should have individual matches"
    assert ctx.similarity_scores.total_jd_skills > 0, "Should count JD skills"
    print(f"  [PASS] SemanticMatchingAgent (overall={ctx.similarity_scores.overall_score:.3f})")


def test_reasoning_agent():
    """ReasoningAgent produces structured analysis from context."""
    mock_llm = MockLLMClient()
    ctx = _make_context()

    # Build up context through prior agents
    ctx = ExtractionAgent(mock_llm).execute(ctx)
    ctx = SemanticMatchingAgent().execute(ctx)

    # Run reasoning
    agent = ReasoningAgent(mock_llm)
    ctx = agent.execute(ctx)

    assert ctx.reasoning_output is not None, "Reasoning output should be set"
    assert len(ctx.reasoning_output.strengths) > 0, "Should identify strengths"
    assert len(ctx.reasoning_output.gaps) > 0, "Should identify gaps"
    assert ctx.reasoning_output.suggested_score > 0, "Should suggest a score"
    assert ctx.reasoning_output.overall_assessment != "", "Should have assessment"
    print(f"  [PASS] ReasoningAgent (suggested_score={ctx.reasoning_output.suggested_score})")


def test_decision_agent():
    """DecisionAgent produces final decision from all prior analysis."""
    mock_llm = MockLLMClient()
    ctx = _make_context()

    # Build up full context
    ctx = ExtractionAgent(mock_llm).execute(ctx)
    ctx = SemanticMatchingAgent().execute(ctx)
    ctx = ReasoningAgent(mock_llm).execute(ctx)

    # Run decision
    agent = DecisionAgent(mock_llm)
    ctx = agent.execute(ctx)

    assert ctx.final_decision is not None, "Final decision should be set"
    assert 0 <= ctx.final_decision.score <= 100, "Score should be 0-100"
    assert 0 <= ctx.final_decision.confidence <= 1, "Confidence should be 0-1"
    assert ctx.final_decision.recommendation in {
        "strong_match", "good_match", "partial_match", "weak_match", "no_match",
    }, f"Invalid recommendation: {ctx.final_decision.recommendation}"
    assert ctx.final_decision.explanation != "", "Should have explanation"
    assert len(ctx.final_decision.key_factors) > 0, "Should have key factors"
    print(f"  [PASS] DecisionAgent (score={ctx.final_decision.score}, rec={ctx.final_decision.recommendation})")


# ──────────────────────────────────────────────
# End-to-end orchestrator test
# ──────────────────────────────────────────────

def test_orchestrator_scenario_a():
    """Full Scenario A runs end-to-end and produces a valid result."""
    orchestrator = Orchestrator(llm_client=MockLLMClient())
    context = orchestrator.run(SAMPLE_CV, SAMPLE_JD, scenario="A")

    # All outputs should be populated
    assert context.cv_entities is not None, "Should have CV entities"
    assert context.jd_entities is not None, "Should have JD entities"
    assert context.similarity_scores is not None, "Should have similarity scores"
    assert context.reasoning_output is not None, "Should have reasoning"
    assert context.final_decision is not None, "Should have final decision"

    # Scenario-specific: no enrichment in A
    assert context.normalized_entities is None, "Scenario A should not have enrichment"

    # Reflection is now enabled by default (Phase 3)
    assert context.reflection_output is not None, "Should have reflection output"

    # Logs should show the orchestrator flow
    log_agents = [log.agent_name for log in context.logs]
    assert "Orchestrator" in log_agents, "Should have orchestrator logs"
    assert "ExtractionAgent" in log_agents, "Should have extraction logs"
    assert "SemanticMatchingAgent" in log_agents, "Should have matching logs"
    assert "ReasoningAgent" in log_agents, "Should have reasoning logs"
    assert "ReflectionAgent" in log_agents, "Should have reflection logs"
    assert "DecisionAgent" in log_agents, "Should have decision logs"

    print(f"  [PASS] Orchestrator Scenario A end-to-end")
    print(f"         Final score: {context.final_decision.score}/100")
    print(f"         Recommendation: {context.final_decision.recommendation}")
    print(f"         Total log entries: {len(context.logs)}")


def test_context_serialization_after_run():
    """Full context can be serialized to JSON after a complete run."""
    orchestrator = Orchestrator(llm_client=MockLLMClient())
    context = orchestrator.run(SAMPLE_CV, SAMPLE_JD, scenario="A")

    # Serialize to JSON and back
    json_str = context.model_dump_json(indent=2)
    restored = SharedContext.model_validate_json(json_str)

    assert restored.final_decision is not None
    assert restored.final_decision.score == context.final_decision.score
    assert len(restored.logs) == len(context.logs)

    print(f"  [PASS] Context serialization after full run ({len(json_str)} chars)")


# ──────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Phase 2 Agent Tests ===\n")

    print("Individual agents:")
    test_extraction_agent()
    test_matching_agent()
    test_reasoning_agent()
    test_decision_agent()

    print("\nEnd-to-end:")
    test_orchestrator_scenario_a()
    test_context_serialization_after_run()

    print("\n=== All Phase 2 tests passed! ===\n")
