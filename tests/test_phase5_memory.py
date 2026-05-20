"""
Phase 5 tests: MemoryStore + MemoryRetrievalAgent + Scenario C.

Tests cover:
1. MemoryStore: add, retrieve, save, load, clear
2. MemoryRetrievalAgent: retrieval from store, empty store handling
3. Scenario C end-to-end through orchestrator
4. Memory accumulation across multiple runs
5. Scenario A and B remain unchanged
6. Serialization with memory data
"""

import json
import shutil
from pathlib import Path

import pytest

from agents.memory_retrieval import MemoryRetrievalAgent
from embeddings.similarity import EmbeddingSimilarity
from llm.client import MockLLMClient
from memory.store import MemoryStore
from models.entities import MemoryEntry
from models.shared_context import SharedContext
from orchestrator.orchestrator import Orchestrator


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    return MockLLMClient()


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Temporary directory for memory storage."""
    d = tmp_path / "test_memory"
    d.mkdir()
    return str(d)


@pytest.fixture
def memory_store(tmp_memory_dir):
    """Fresh memory store in a temp directory."""
    return MemoryStore(memory_dir=tmp_memory_dir)


@pytest.fixture
def sample_memory():
    """A sample MemoryEntry for testing."""
    return MemoryEntry(
        cv_summary="Python developer with 3 years ML experience",
        jd_summary="ML Engineer position requiring deep learning",
        decision_score=65.0,
        reasoning_summary="Good Python skills but lacks deep learning architecture experience",
    )


@pytest.fixture
def sample_memory_2():
    """A second, different MemoryEntry for testing."""
    return MemoryEntry(
        cv_summary="Java backend developer with Spring Boot",
        jd_summary="Senior Java developer for microservices",
        decision_score=82.0,
        reasoning_summary="Strong Java match, good microservices experience",
    )


# ── Test 1: MemoryStore basic operations ─────────────────────

def test_memory_store_starts_empty(memory_store):
    """New memory store should have zero memories."""
    assert memory_store.count == 0


def test_memory_store_add(memory_store, sample_memory):
    """Adding a memory increases the count."""
    memory_store.add(sample_memory)
    assert memory_store.count == 1


def test_memory_store_add_multiple(memory_store, sample_memory, sample_memory_2):
    """Can add multiple memories."""
    memory_store.add(sample_memory)
    memory_store.add(sample_memory_2)
    assert memory_store.count == 2


# ── Test 2: MemoryStore retrieval ─────────────────────────────

def test_memory_store_retrieve_similar(memory_store, sample_memory, sample_memory_2):
    """Retrieval returns memories sorted by similarity."""
    memory_store.add(sample_memory)
    memory_store.add(sample_memory_2)

    # Query about Python ML — should match sample_memory better
    results = memory_store.retrieve("Python ML developer deep learning")

    assert len(results) > 0
    # Each result should have similarity_to_current populated
    for r in results:
        assert r.similarity_to_current > 0.0

    # The ML-related memory should rank higher than the Java one
    if len(results) >= 2:
        assert results[0].decision_score == 65.0  # ML memory


def test_memory_store_retrieve_empty(memory_store):
    """Retrieval from empty store returns empty list."""
    results = memory_store.retrieve("anything")
    assert results == []


def test_memory_store_retrieve_top_k(memory_store, sample_memory, sample_memory_2):
    """Retrieval respects top_k limit."""
    memory_store.add(sample_memory)
    memory_store.add(sample_memory_2)

    results = memory_store.retrieve("Python ML", top_k=1)
    assert len(results) <= 1


# ── Test 3: MemoryStore persistence ───────────────────────────

def test_memory_store_save_and_load(tmp_memory_dir, sample_memory, sample_memory_2):
    """Memories persist across store instances via save/load."""
    # Save
    store1 = MemoryStore(memory_dir=tmp_memory_dir)
    store1.add(sample_memory)
    store1.add(sample_memory_2)
    store1.save()

    # Verify files exist
    assert (Path(tmp_memory_dir) / "memories.json").exists()
    assert (Path(tmp_memory_dir) / "embeddings.npy").exists()

    # Load in a new instance
    store2 = MemoryStore(memory_dir=tmp_memory_dir)
    assert store2.count == 2

    # Retrieval should still work
    results = store2.retrieve("Python ML developer")
    assert len(results) > 0


def test_memory_store_clear(memory_store, sample_memory):
    """Clear removes all memories in-memory."""
    memory_store.add(sample_memory)
    assert memory_store.count == 1

    memory_store.clear()
    assert memory_store.count == 0


# ── Test 4: MemoryRetrievalAgent ──────────────────────────────

def test_memory_retrieval_agent_with_memories(memory_store, sample_memory):
    """Agent retrieves memories and writes them to context."""
    memory_store.add(sample_memory)

    agent = MemoryRetrievalAgent(memory_store)
    context = SharedContext(
        cv_text="Python developer with ML skills and data analysis",
        jd_text="Looking for ML engineer with deep learning",
        scenario="C",
    )

    context = agent.execute(context)

    assert context.has_memory()
    assert len(context.memory_entries) > 0
    assert context.memory_entries[0].similarity_to_current > 0


def test_memory_retrieval_agent_empty_store(memory_store):
    """Agent handles empty memory store gracefully."""
    agent = MemoryRetrievalAgent(memory_store)
    context = SharedContext(
        cv_text="any cv text",
        jd_text="any jd text",
        scenario="C",
    )

    context = agent.execute(context)

    assert not context.has_memory()
    assert len(context.memory_entries) == 0

    # Should have logged that memory is empty
    log_actions = [log.action for log in context.logs]
    assert "memory_empty" in log_actions


# ── Test 5: Scenario C end-to-end ─────────────────────────────

def test_scenario_c_first_run(mock_llm, tmp_memory_dir):
    """First Scenario C run works with empty memory, then saves result."""
    memory_store = MemoryStore(memory_dir=tmp_memory_dir)
    orchestrator = Orchestrator(
        llm_client=mock_llm,
        memory_store=memory_store,
    )

    context = orchestrator.run("test cv", "test jd", scenario="C")

    # All phases should complete
    assert context.cv_entities is not None
    assert context.has_enrichment()  # Scenario C includes enrichment
    assert context.similarity_scores is not None
    assert context.reasoning_output is not None
    assert context.reflection_output is not None
    assert context.final_decision is not None

    # Memory should have been saved (1 entry now)
    assert memory_store.count == 1

    # Check that MemoryRetrievalAgent appears in logs
    log_agents = [log.agent_name for log in context.logs]
    assert "MemoryRetrievalAgent" in log_agents


def test_scenario_c_second_run_retrieves_memory(mock_llm, tmp_memory_dir):
    """Second Scenario C run retrieves the first run's result from memory."""
    memory_store = MemoryStore(memory_dir=tmp_memory_dir)
    orchestrator = Orchestrator(
        llm_client=mock_llm,
        memory_store=memory_store,
    )

    # First run — populates memory
    orchestrator.run("test cv", "test jd", scenario="C")
    assert memory_store.count == 1

    # Need fresh mock LLM (reset call counters)
    mock_llm2 = MockLLMClient()
    orchestrator2 = Orchestrator(
        llm_client=mock_llm2,
        memory_store=memory_store,
    )

    # Second run — should retrieve the first run's result
    context2 = orchestrator2.run("similar cv", "similar jd", scenario="C")

    # Memory should have been retrieved
    assert context2.has_memory()
    assert len(context2.memory_entries) > 0

    # And a new memory should have been saved (now 2 total)
    assert memory_store.count == 2


# ── Test 6: Scenario A and B unchanged ────────────────────────

def test_scenario_a_no_memory(mock_llm):
    """Scenario A should not use memory, even if store exists."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    context = orchestrator.run("test cv", "test jd", scenario="A")

    assert not context.has_memory()
    log_agents = [log.agent_name for log in context.logs]
    assert "MemoryRetrievalAgent" not in log_agents


def test_scenario_b_no_memory(mock_llm):
    """Scenario B should not use memory, even if store exists."""
    orchestrator = Orchestrator(llm_client=mock_llm)
    context = orchestrator.run("test cv", "test jd", scenario="B")

    assert not context.has_memory()
    log_agents = [log.agent_name for log in context.logs]
    assert "MemoryRetrievalAgent" not in log_agents


# ── Test 7: Agent chain order for Scenario C ──────────────────

def test_scenario_c_agent_chain_order(mock_llm, tmp_memory_dir):
    """Scenario C agent chain: Memory -> Extraction -> Enrichment -> Matching -> Reasoning -> Decision."""
    memory_store = MemoryStore(memory_dir=tmp_memory_dir)
    # Seed memory so retrieval actually runs
    memory_store.add(MemoryEntry(
        cv_summary="existing cv",
        jd_summary="existing jd",
        decision_score=50.0,
        reasoning_summary="previous reasoning",
    ))

    orchestrator = Orchestrator(
        llm_client=mock_llm,
        memory_store=memory_store,
    )
    context = orchestrator.run("test cv", "test jd", scenario="C")

    # Extract agent execution order from logs
    agent_starts = [
        log.agent_name
        for log in context.logs
        if log.action == "started"
    ]

    assert "MemoryRetrievalAgent" in agent_starts
    assert "ExtractionAgent" in agent_starts
    assert "ContextEnrichmentAgent" in agent_starts
    assert "SemanticMatchingAgent" in agent_starts

    mem_idx = agent_starts.index("MemoryRetrievalAgent")
    ext_idx = agent_starts.index("ExtractionAgent")
    enr_idx = agent_starts.index("ContextEnrichmentAgent")
    mat_idx = agent_starts.index("SemanticMatchingAgent")

    assert mem_idx < ext_idx < enr_idx < mat_idx, (
        f"Expected Memory({mem_idx}) < Extraction({ext_idx}) "
        f"< Enrichment({enr_idx}) < Matching({mat_idx})"
    )


# ── Test 8: Serialization with memory ─────────────────────────

def test_serialization_with_memory(mock_llm, tmp_memory_dir):
    """Context with memory data can be serialized and deserialized."""
    memory_store = MemoryStore(memory_dir=tmp_memory_dir)
    # Seed one memory
    memory_store.add(MemoryEntry(
        cv_summary="seed cv",
        jd_summary="seed jd",
        decision_score=55.0,
        reasoning_summary="seed reasoning",
    ))

    orchestrator = Orchestrator(
        llm_client=mock_llm,
        memory_store=memory_store,
    )
    context = orchestrator.run("test cv", "test jd", scenario="C")

    # Serialize
    json_str = context.model_dump_json(indent=2)
    data = json.loads(json_str)

    # Check memory data is in JSON
    assert "memory_entries" in data
    assert len(data["memory_entries"]) > 0
    assert data["memory_entries"][0]["similarity_to_current"] > 0

    # Reconstruct
    reconstructed = SharedContext.model_validate_json(json_str)
    assert reconstructed.has_memory()
    assert len(reconstructed.memory_entries) == len(context.memory_entries)
