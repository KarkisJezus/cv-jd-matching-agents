"""
Phase 3 tests: ReflectionAgent and the reflection-revision loop.

Tests cover:
1. ReflectionAgent in isolation — reviewing reasoning
2. Reflection that triggers revision (needs_revision=True)
3. Full orchestrator with reflection loop (score changes after revision)
4. Orchestrator with reflection disabled (comparison baseline)
5. Max revisions cap is respected
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.extraction import ExtractionAgent
from agents.matching import SemanticMatchingAgent
from agents.reasoning import ReasoningAgent
from agents.reflection import ReflectionAgent
from agents.decision import DecisionAgent
from llm.client import MockLLMClient
from models.shared_context import SharedContext
from orchestrator.orchestrator import Orchestrator


SAMPLE_CV = "I know Python, Machine Learning, and SQL. 3 years experience."
SAMPLE_JD = "We need Python, Deep Learning, and Docker skills. 2+ years required."


def _build_context_with_reasoning() -> SharedContext:
    """Build a context that has gone through extraction, matching, and reasoning."""
    mock_llm = MockLLMClient()
    ctx = SharedContext(cv_text=SAMPLE_CV, jd_text=SAMPLE_JD, scenario="A")
    ctx = ExtractionAgent(mock_llm).execute(ctx)
    ctx = SemanticMatchingAgent().execute(ctx)
    ctx = ReasoningAgent(mock_llm).execute(ctx)
    return ctx, mock_llm


# ──────────────────────────────────────────────
# Test 1: ReflectionAgent in isolation
# ──────────────────────────────────────────────

def test_reflection_agent_first_call_finds_issues():
    """First reflection call should find issues and request revision."""
    ctx, mock_llm = _build_context_with_reasoning()

    # Verify pre-conditions
    assert ctx.reasoning_output is not None
    assert ctx.reasoning_output.suggested_score == 68.0
    assert ctx.needs_revision is False

    # Run reflection
    agent = ReflectionAgent(mock_llm)
    ctx = agent.execute(ctx)

    assert ctx.reflection_output is not None
    assert ctx.reflection_output.is_consistent is False, "First call should find issues"
    assert len(ctx.reflection_output.issues_found) > 0, "Should list specific issues"
    assert ctx.reflection_output.confidence < 0.5, "Low confidence means serious doubts"
    assert ctx.reflection_output.revision_reason != "", "Should explain why revision needed"
    assert ctx.needs_revision is True, "Should request revision"

    print(f"  [PASS] ReflectionAgent finds issues (confidence={ctx.reflection_output.confidence})")
    for issue in ctx.reflection_output.issues_found:
        print(f"         ! {issue}")


def test_reflection_agent_second_call_approves():
    """Second reflection call (after revision) should approve."""
    ctx, mock_llm = _build_context_with_reasoning()

    agent = ReflectionAgent(mock_llm)

    # First call: reject
    ctx = agent.execute(ctx)
    assert ctx.needs_revision is True

    # Simulate a revision cycle
    ctx.revision_count = 1
    ctx.needs_revision = False

    # Second call: approve
    ctx = agent.execute(ctx)
    assert ctx.reflection_output.is_consistent is True, "Second call should approve"
    assert ctx.reflection_output.confidence > 0.8, "High confidence after revision"
    assert ctx.needs_revision is False, "No more revision needed"

    print(f"  [PASS] ReflectionAgent approves after revision (confidence={ctx.reflection_output.confidence})")


# ──────────────────────────────────────────────
# Test 2: Revision changes reasoning
# ──────────────────────────────────────────────

def test_reasoning_revises_after_reflection():
    """ReasoningAgent produces different output on revision cycle."""
    mock_llm = MockLLMClient()
    ctx = SharedContext(cv_text=SAMPLE_CV, jd_text=SAMPLE_JD, scenario="A")
    ctx = ExtractionAgent(mock_llm).execute(ctx)
    ctx = SemanticMatchingAgent().execute(ctx)

    # First reasoning
    ctx = ReasoningAgent(mock_llm).execute(ctx)
    first_score = ctx.reasoning_output.suggested_score
    first_gaps = len(ctx.reasoning_output.gaps)

    # Simulate reflection feedback
    ctx = ReflectionAgent(mock_llm).execute(ctx)
    assert ctx.needs_revision is True

    # Revision cycle
    ctx.revision_count = 1
    ctx.needs_revision = False
    ctx = ReasoningAgent(mock_llm).execute(ctx)

    second_score = ctx.reasoning_output.suggested_score
    second_gaps = len(ctx.reasoning_output.gaps)

    # Revised reasoning should be different (lower score, more gaps)
    assert second_score < first_score, (
        f"Revised score ({second_score}) should be lower than initial ({first_score})"
    )
    assert second_gaps >= first_gaps, (
        f"Revised gaps ({second_gaps}) should be >= initial ({first_gaps})"
    )

    print(f"  [PASS] Reasoning revises: {first_score} -> {second_score}, gaps {first_gaps} -> {second_gaps}")


# ──────────────────────────────────────────────
# Test 3: Full orchestrator with reflection
# ──────────────────────────────────────────────

def test_orchestrator_with_reflection():
    """Full pipeline with reflection produces revised result."""
    mock_llm = MockLLMClient()
    orchestrator = Orchestrator(llm_client=mock_llm, enable_reflection=True)
    context = orchestrator.run(SAMPLE_CV, SAMPLE_JD, scenario="A")

    # All outputs should be populated
    assert context.cv_entities is not None
    assert context.similarity_scores is not None
    assert context.reasoning_output is not None
    assert context.reflection_output is not None
    assert context.final_decision is not None

    # Should have gone through at least one revision cycle
    assert context.revision_count >= 1, (
        f"Should have revised at least once, got {context.revision_count}"
    )

    # Final reflection should be consistent (approved after revision)
    assert context.reflection_output.is_consistent is True, (
        "Final reflection should approve the revised reasoning"
    )

    # Check that the orchestrator logged the revision cycle
    log_actions = [log.action for log in context.logs]
    assert "revision_cycle" in log_actions, "Should log revision cycles"

    # Check that ReflectionAgent appears in the logs
    log_agents = [log.agent_name for log in context.logs]
    assert "ReflectionAgent" in log_agents, "Should have reflection logs"

    print(f"  [PASS] Orchestrator with reflection")
    print(f"         Revisions: {context.revision_count}")
    print(f"         Final score: {context.final_decision.score}")
    print(f"         Reflection confidence: {context.reflection_output.confidence}")


def test_orchestrator_without_reflection():
    """Pipeline without reflection produces result with no reflection data."""
    mock_llm = MockLLMClient()
    orchestrator = Orchestrator(llm_client=mock_llm, enable_reflection=False)
    context = orchestrator.run(SAMPLE_CV, SAMPLE_JD, scenario="A")

    # Should have decision but NO reflection
    assert context.final_decision is not None
    assert context.reflection_output is None, "Should not have reflection when disabled"
    assert context.revision_count == 0, "No revisions without reflection"

    # Logs should NOT contain ReflectionAgent
    log_agents = [log.agent_name for log in context.logs]
    assert "ReflectionAgent" not in log_agents

    print(f"  [PASS] Orchestrator without reflection (score={context.final_decision.score})")


# ──────────────────────────────────────────────
# Test 4: Max revisions cap
# ──────────────────────────────────────────────

def test_max_revisions_respected():
    """Reflection loop stops at max_revisions even if still inconsistent."""
    ctx, mock_llm = _build_context_with_reasoning()
    ctx.max_revisions = 0  # No revisions allowed

    agent = ReflectionAgent(mock_llm)
    ctx = agent.execute(ctx)

    # Should find issues but NOT request revision (budget exhausted)
    assert ctx.reflection_output.is_consistent is False
    assert ctx.needs_revision is False, "Should not request revision when budget is 0"

    print(f"  [PASS] Max revisions respected (max=0, needs_revision={ctx.needs_revision})")


# ──────────────────────────────────────────────
# Test 5: Context serialization with reflection data
# ──────────────────────────────────────────────

def test_context_serialization_with_reflection():
    """Full context with reflection data can be serialized and restored."""
    mock_llm = MockLLMClient()
    orchestrator = Orchestrator(llm_client=mock_llm, enable_reflection=True)
    context = orchestrator.run(SAMPLE_CV, SAMPLE_JD, scenario="A")

    # Serialize and restore
    json_str = context.model_dump_json(indent=2)
    restored = SharedContext.model_validate_json(json_str)

    assert restored.reflection_output is not None
    assert restored.reflection_output.is_consistent == context.reflection_output.is_consistent
    assert restored.revision_count == context.revision_count
    assert restored.final_decision.score == context.final_decision.score

    print(f"  [PASS] Serialization with reflection data ({len(json_str)} chars)")


# ──────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Phase 3 Reflection Tests ===\n")

    print("ReflectionAgent isolation:")
    test_reflection_agent_first_call_finds_issues()
    test_reflection_agent_second_call_approves()

    print("\nRevision behavior:")
    test_reasoning_revises_after_reflection()

    print("\nOrchestrator integration:")
    test_orchestrator_with_reflection()
    test_orchestrator_without_reflection()

    print("\nEdge cases:")
    test_max_revisions_respected()
    test_context_serialization_with_reflection()

    print("\n=== All Phase 3 tests passed! ===\n")
