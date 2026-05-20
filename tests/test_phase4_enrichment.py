"""
Phase 4 tests: ContextEnrichmentAgent + ESCO taxonomy + Scenario B.

Tests cover:
1. Taxonomy loading and lookup
2. ContextEnrichmentAgent local resolution
3. ContextEnrichmentAgent LLM fallback
4. Scenario B end-to-end through orchestrator
5. Normalized skills used by downstream agents
6. Scenario A unchanged (no enrichment)
7. Serialization with enrichment data
"""

import json
from pathlib import Path

import pytest

from agents.enrichment import ContextEnrichmentAgent, load_taxonomy
from embeddings.similarity import EmbeddingSimilarity
from llm.client import MockLLMClient
from models.entities import ExtractedEntities
from models.shared_context import SharedContext
from orchestrator.orchestrator import Orchestrator


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.fixture
def taxonomy():
    return load_taxonomy()


@pytest.fixture
def context_with_entities():
    """Context with extracted entities, ready for enrichment."""
    ctx = SharedContext(cv_text="test cv", jd_text="test jd", scenario="B")
    ctx.cv_entities = ExtractedEntities(
        skills=["Python", "Machine Learning", "SQL", "Docker", "SomeUnknownSkill"],
        experience=["3 years dev"],
        education=["BSc CS"],
        raw_summary="Python developer",
    )
    ctx.jd_entities = ExtractedEntities(
        skills=["Python programming", "deep learning", "TensorFlow", "Kubernetes", "AnotherUnknown"],
        experience=["5 years ML"],
        education=["MSc preferred"],
        raw_summary="ML Engineer position",
    )
    return ctx


# ── Test 1: Taxonomy loading ─────────────────────────────────

def test_taxonomy_loads_correctly(taxonomy):
    """The ESCO taxonomy file loads and contains expected entries."""
    assert len(taxonomy) > 0, "Taxonomy should not be empty"

    # Check that entries have required fields
    for entry in taxonomy:
        assert "esco_code" in entry
        assert "label" in entry
        assert "synonyms" in entry
        assert isinstance(entry["synonyms"], list)

    # Check a known entry exists
    labels = [e["label"] for e in taxonomy]
    assert "Python programming" in labels
    assert "machine learning" in labels
    assert "deep learning" in labels


# ── Test 2: Local taxonomy lookup ─────────────────────────────

def test_local_lookup_resolves_known_skills(mock_llm):
    """Skills matching taxonomy entries are resolved locally without LLM."""
    agent = ContextEnrichmentAgent(mock_llm)

    # These should all match via synonym lookup
    resolved, unresolved = agent._local_lookup(["Python", "SQL", "Docker"])

    assert len(resolved) == 3
    assert len(unresolved) == 0

    # Check that Python resolved correctly
    python_skill = next(s for s in resolved if s.original == "Python")
    assert python_skill.normalized == "Python programming"
    assert python_skill.esco_code == "S1.1.1"
    assert "Python" in python_skill.synonyms


def test_local_lookup_marks_unknown_skills_as_unresolved(mock_llm):
    """Skills not in the taxonomy are returned as unresolved."""
    agent = ContextEnrichmentAgent(mock_llm)

    resolved, unresolved = agent._local_lookup(
        ["Python", "SomeInventedSkill", "AnotherRandom"]
    )

    assert len(resolved) == 1  # Only Python
    assert len(unresolved) == 2
    assert "SomeInventedSkill" in unresolved
    assert "AnotherRandom" in unresolved


# ── Test 3: Full enrichment with LLM fallback ────────────────

def test_enrichment_agent_processes_context(mock_llm, context_with_entities):
    """ContextEnrichmentAgent writes normalized_entities and enrichment_notes."""
    agent = ContextEnrichmentAgent(mock_llm)
    context = agent.execute(context_with_entities)

    # Enrichment data should be populated
    assert context.has_enrichment()
    assert context.normalized_entities is not None

    # All CV skills should be normalized (some via taxonomy, some via LLM)
    assert len(context.normalized_entities.cv_skills) == 5
    assert len(context.normalized_entities.jd_skills) == 5

    # Check that known skills got ESCO codes
    python_cv = next(
        s for s in context.normalized_entities.cv_skills if s.original == "Python"
    )
    assert python_cv.esco_code == "S1.1.1"
    assert python_cv.normalized == "Python programming"

    # Check that unknown skills were still normalized (via LLM fallback)
    unknown = next(
        s for s in context.normalized_entities.cv_skills
        if s.original == "SomeUnknownSkill"
    )
    assert unknown.esco_code is None  # LLM doesn't assign ESCO codes
    assert unknown.normalized  # Should have some normalized form

    # Enrichment notes should exist
    assert len(context.enrichment_notes) > 0


# ── Test 4: Scenario B end-to-end ─────────────────────────────

def test_scenario_b_includes_enrichment(mock_llm):
    """Scenario B runs enrichment agent, producing normalized entities."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    context = orchestrator.run("test cv", "test jd", scenario="B")

    # Enrichment should have run
    assert context.has_enrichment()
    assert context.normalized_entities is not None
    assert len(context.normalized_entities.cv_skills) > 0
    assert len(context.normalized_entities.jd_skills) > 0

    # All other phases should also have completed
    assert context.cv_entities is not None
    assert context.jd_entities is not None
    assert context.similarity_scores is not None
    assert context.reasoning_output is not None
    assert context.reflection_output is not None
    assert context.final_decision is not None

    # Check that ContextEnrichmentAgent appears in logs
    log_agents = [log.agent_name for log in context.logs]
    assert "ContextEnrichmentAgent" in log_agents


# ── Test 5: Normalized skills used by matching ────────────────

def test_matching_uses_normalized_skills(mock_llm, context_with_entities):
    """When enrichment exists, matching should use normalized skill names."""
    # Run enrichment first
    enrichment_agent = ContextEnrichmentAgent(mock_llm)
    context = enrichment_agent.execute(context_with_entities)

    # get_skills_for_matching should return normalized skills
    cv_skills, jd_skills = context.get_skills_for_matching()

    # Skills should come from normalized_entities, not raw entities
    assert context.has_enrichment()

    # Normalized skills should contain the ESCO labels, not raw strings
    # "Python" should have been normalized to "Python programming"
    assert any("Python programming" in s for s in cv_skills)


# ── Test 6: Scenario A is unchanged (no enrichment) ──────────

def test_scenario_a_has_no_enrichment(mock_llm):
    """Scenario A should NOT run enrichment agent."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    context = orchestrator.run("test cv", "test jd", scenario="A")

    # No enrichment
    assert not context.has_enrichment()
    assert context.normalized_entities is None
    assert len(context.enrichment_notes) == 0

    # But everything else should work
    assert context.cv_entities is not None
    assert context.similarity_scores is not None
    assert context.final_decision is not None

    # ContextEnrichmentAgent should NOT appear in logs
    log_agents = [log.agent_name for log in context.logs]
    assert "ContextEnrichmentAgent" not in log_agents


# ── Test 7: Serialization with enrichment data ────────────────

def test_serialization_with_enrichment(mock_llm):
    """Full context with enrichment can be serialized to JSON and back."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    context = orchestrator.run("test cv", "test jd", scenario="B")

    # Serialize
    json_str = context.model_dump_json(indent=2)
    data = json.loads(json_str)

    # Check enrichment data is in the JSON
    assert "normalized_entities" in data
    assert data["normalized_entities"] is not None
    assert len(data["normalized_entities"]["cv_skills"]) > 0
    assert len(data["normalized_entities"]["jd_skills"]) > 0

    # Check a specific normalized skill
    cv_skills = data["normalized_entities"]["cv_skills"]
    has_esco = any(s.get("esco_code") is not None for s in cv_skills)
    assert has_esco, "At least one CV skill should have an ESCO code"

    # Check enrichment_notes
    assert "enrichment_notes" in data
    assert len(data["enrichment_notes"]) > 0

    # Reconstruct from JSON
    reconstructed = SharedContext.model_validate_json(json_str)
    assert reconstructed.has_enrichment()
    assert len(reconstructed.normalized_entities.cv_skills) == len(
        context.normalized_entities.cv_skills
    )


# ── Test 8: Agent chain for Scenario B ────────────────────────

def test_scenario_b_agent_chain_order(mock_llm):
    """Scenario B agent chain should include enrichment between extraction and matching."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    context = orchestrator.run("test cv", "test jd", scenario="B")

    # Extract agent execution order from logs
    agent_starts = [
        log.agent_name
        for log in context.logs
        if log.action == "started"
    ]

    # Verify the order: Extraction -> Enrichment -> Matching -> Reasoning -> Decision
    assert "ExtractionAgent" in agent_starts
    assert "ContextEnrichmentAgent" in agent_starts
    assert "SemanticMatchingAgent" in agent_starts

    ext_idx = agent_starts.index("ExtractionAgent")
    enr_idx = agent_starts.index("ContextEnrichmentAgent")
    mat_idx = agent_starts.index("SemanticMatchingAgent")

    assert ext_idx < enr_idx < mat_idx, (
        f"Expected Extraction({ext_idx}) < Enrichment({enr_idx}) < Matching({mat_idx})"
    )
