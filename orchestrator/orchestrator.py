"""
Orchestrator: drives agent execution based on scenario.

The orchestrator is NOT a framework — it is a simple coordinator that:
1. Selects which agents to run based on the scenario
2. Passes the shared context through each agent
3. Handles the reflection loop (iterative self-improvement)
4. Manages memory persistence (save after run, load before)
5. Logs the overall execution for evaluation

This is the component that makes the system more than a pipeline:
- Scenario-based agent composition
- Conditional execution paths
- Reflection-driven re-execution (ReflectionAgent can trigger
  ReasoningAgent to re-run with feedback)
- Memory-augmented reasoning (Scenario C)
"""

import time
from typing import Literal

from agents.base import BaseAgent
from agents.decision import DecisionAgent
from agents.enrichment import ContextEnrichmentAgent
from agents.extraction import ExtractionAgent
from agents.matching import SemanticMatchingAgent
from agents.memory_retrieval import MemoryRetrievalAgent
from agents.reasoning import ReasoningAgent
from agents.reflection import ReflectionAgent
from embeddings.similarity import EmbeddingSimilarity
from llm.client import BaseLLMClient
from memory.store import MemoryStore
from models.entities import MemoryEntry
from models.shared_context import SharedContext


class Orchestrator:
    """
    Coordinates agent execution for CV-JD matching.

    The orchestrator builds an agent chain based on the scenario,
    then executes agents sequentially, passing the shared context
    through each one.

    Scenario A (basic):
      Extraction -> Matching -> Reasoning -> [Reflection loop] -> Decision

    Scenario B (+enrichment):
      Extraction -> Enrichment -> Matching -> Reasoning -> [Reflection] -> Decision

    Scenario C (+enrichment +memory):
      MemoryRetrieval -> Extraction -> Enrichment -> Matching ->
      Reasoning -> [Reflection] -> Decision -> [save to memory]

    The reflection loop is the key feature that makes this system
    agentic: an agent (ReflectionAgent) autonomously decides whether
    another agent's work (ReasoningAgent) is good enough, and can
    force a revision. This is NOT possible in a simple pipeline.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_similarity: EmbeddingSimilarity | None = None,
        memory_store: MemoryStore | None = None,
        labeled_memory_store=None,  # LabeledMemoryStore (Tier 2). Optional.
        enable_reflection: bool = True,
        architecture: str = "tier1",  # "tier1" (default, legacy) or "tier2" (new)
    ):
        """
        Args:
            architecture: "tier1" uses the legacy chain (ExtractionAgent + EnrichmentAgent +
                MemoryRetrievalAgent + Reasoning + Reflection + Decision). "tier2" uses the
                new chain (CV/JDProfilingAgent + Reasoning + Reflection + DecisionAgent ->
                LabeledMemoryRetrievalAgent + CalibrationAgent for Scenario C).
            memory_store: legacy unlabeled MemoryStore. Used by Tier 1 Scenario C.
            labeled_memory_store: new LabeledMemoryStore. Used by Tier 2 Scenario C.
        """
        self._llm = llm_client
        self._embedding_sim = embedding_similarity or EmbeddingSimilarity()
        self._memory_store = memory_store
        self._labeled_memory_store = labeled_memory_store
        self._enable_reflection = enable_reflection
        if architecture not in ("tier1", "tier2"):
            raise ValueError(f"architecture must be 'tier1' or 'tier2', got {architecture!r}")
        self._architecture = architecture

    def run(
        self,
        cv_text: str,
        jd_text: str,
        scenario: Literal["A", "B", "C"] = "A",
    ) -> SharedContext:
        """
        Run the full matching pipeline for a given scenario.

        Args:
            cv_text: The CV text to analyze
            jd_text: The job description text to analyze
            scenario: Which scenario to run (A, B, or C)

        Returns:
            The completed SharedContext with all agent outputs
        """
        # Create the shared context (blackboard)
        context = SharedContext(
            cv_text=cv_text,
            jd_text=jd_text,
            scenario=scenario,
        )

        context.add_log(
            "Orchestrator", "run_started",
            f"Scenario {scenario}, architecture={self._architecture}",
        )
        start_time = time.time()

        # Build the agent chain for this scenario + architecture
        if self._architecture == "tier2":
            agents = self._build_tier2_chain(scenario)
        else:
            agents = self._build_agent_chain(scenario)

        context.add_log(
            "Orchestrator",
            "agents_selected",
            f"Agent chain: {' -> '.join(a.name for a in agents)}",
        )

        # Tier 2 optimization: CVProfilingAgent and JDProfilingAgent are
        # independent (read disjoint inputs, write disjoint context fields).
        # Run them concurrently to roughly halve per-pair profiling latency.
        # Each agent gets a child LLM client so the BaseAgent token-snapshot
        # logic doesn't race; counters are rolled up into the parent client
        # after both finish, so the runner's per-pair token totals stay correct.
        if (
            self._architecture == "tier2"
            and len(agents) >= 2
            and self._can_parallelize_profiling(agents[0], agents[1])
        ):
            context = self._run_parallel_profiling(agents[:2], context)
            remaining_agents = agents[2:]
        else:
            remaining_agents = agents

        # Execute remaining agents sequentially
        for agent in remaining_agents:
            context = agent.execute(context)

            # After reasoning, run the reflection loop if enabled
            if agent.name == "ReasoningAgent" and self._should_reflect(context):
                context = self._reflection_loop(context)

        duration = time.time() - start_time
        context.add_log(
            "Orchestrator",
            "run_completed",
            f"Total duration: {duration:.2f}s, "
            f"Revisions: {context.revision_count}",
            duration=duration,
        )

        # Tier 1: legacy memory save after Scenario C
        if self._architecture == "tier1" and scenario == "C" and self._memory_store is not None:
            self._save_to_memory(context)

        # Tier 2: labeled memory save is handled by ExperimentRunner (which has the
        # ground truth label needed to build a LabeledMemoryEntry). The Orchestrator
        # only produces the prediction here.

        return context

    def _build_agent_chain(self, scenario: str) -> list[BaseAgent]:
        """
        Build the list of agents to execute for the given scenario.

        This is where scenario-based composition happens. Different
        scenarios use different agent chains, demonstrating that the
        system is not a fixed pipeline.

        Note: ReflectionAgent is NOT in the chain — it is handled
        by _reflection_loop() because it triggers conditional
        re-execution that a flat list cannot express.
        """
        agents: list[BaseAgent] = []

        # Phase: Memory retrieval (Scenario C only)
        # Retrieves similar past decisions before any analysis begins
        if scenario == "C" and self._memory_store is not None:
            agents.append(MemoryRetrievalAgent(self._memory_store))

        # Phase: Extraction (all scenarios)
        agents.append(ExtractionAgent(self._llm))

        # Phase: Enrichment (Scenario B and C)
        # Normalizes skills via ESCO taxonomy + LLM, giving downstream
        # agents standardized skill names for better matching
        if scenario in ("B", "C"):
            agents.append(ContextEnrichmentAgent(self._llm))

        # Phase: Matching (all scenarios)
        agents.append(SemanticMatchingAgent(self._embedding_sim))

        # Phase: Reasoning (all scenarios)
        agents.append(ReasoningAgent(self._llm))

        # Phase: Decision (all scenarios)
        agents.append(DecisionAgent(self._llm))

        return agents

    def _build_tier2_chain(self, scenario: str) -> list[BaseAgent]:
        """
        Tier 2 agent chain.

        Scenario A: CVProfiling + JDProfiling(no ESCO) + Matching + Reasoning + Reflection + Decision
        Scenario B: CVProfiling + JDProfiling(with ESCO) + Matching + Reasoning + Reflection + Decision
        Scenario C: same as B + LabeledMemoryRetrieval + CalibrationAgent (Pass 2)

        For Scenario C the DecisionAgent is configured to commit to initial_decision
        (Pass 1). CalibrationAgent then commits the calibrated final_decision (Pass 2).
        """
        # Lazy imports — keep tier1 paths working even if tier2 modules are missing
        from agents.calibration import CalibrationAgent
        from agents.cv_profiling import CVProfilingAgent
        from agents.jd_profiling import JDProfilingAgent
        from agents.labeled_memory_retrieval import LabeledMemoryRetrievalAgent

        agents: list[BaseAgent] = []

        # Phase 1: Profiling (CV and JD in parallel-ready order)
        # Scenarios B and C use ESCO role context in JDProfilingAgent.
        # Scenario A skips ESCO entirely — JDProfilingAgent uses fallback profiling.
        agents.append(CVProfilingAgent(self._llm))
        agents.append(JDProfilingAgent(
            self._llm,
            use_esco_context=(scenario in ("B", "C")),
        ))

        # Phase 2: Matching (deterministic, sentence-BERT, no LLM)
        agents.append(SemanticMatchingAgent(self._embedding_sim))

        # Phase 3: Reasoning + Reflection loop
        agents.append(ReasoningAgent(self._llm))

        # Phase 4: Decision (Pass 1)
        # In Scenario C, this commits to initial_decision (CalibrationAgent will
        # commit final_decision in Pass 2). In A/B, it commits to final_decision directly.
        commits_to = "initial" if scenario == "C" else "final"
        agents.append(DecisionAgent(self._llm, commits_to=commits_to))

        # Phase 5 (Scenario C only): Pass 2 — labeled memory + calibration
        if scenario == "C" and self._labeled_memory_store is not None:
            agents.append(LabeledMemoryRetrievalAgent(self._labeled_memory_store))
            agents.append(CalibrationAgent(self._llm))

        return agents

    def _can_parallelize_profiling(self, agent_a: BaseAgent, agent_b: BaseAgent) -> bool:
        """Return True when the first two Tier 2 agents are CV+JD profiling.

        Both agents must be the expected types and use the same parent LLM
        client (so we can swap in child clients safely).
        """
        from agents.cv_profiling import CVProfilingAgent
        from agents.jd_profiling import JDProfilingAgent
        if not isinstance(agent_a, CVProfilingAgent):
            return False
        if not isinstance(agent_b, JDProfilingAgent):
            return False
        return getattr(agent_a, "_llm", None) is getattr(agent_b, "_llm", None)

    def _run_parallel_profiling(
        self, agents: list[BaseAgent], context: SharedContext,
    ) -> SharedContext:
        """Run two independent profiling agents concurrently.

        Each agent gets a fresh child LLMClient so BaseAgent's per-agent
        token-snapshot logic doesn't race. The agents write to disjoint
        SharedContext fields (cv_profile vs jd_profile), so concurrent
        attribute writes are safe under CPython's GIL.

        After both finish, child token counters are rolled up into the parent
        LLMClient so the runner's per-pair totals match the sequential version
        modulo expected nondeterminism (LLM stochasticity, log ordering).

        For MockLLMClient, the swap is skipped — mock is fast enough that
        sequential mock execution remains the simpler test path.
        """
        from concurrent.futures import ThreadPoolExecutor
        from llm.client import LLMClient

        parent_llm = self._llm
        swapped: list[tuple[BaseAgent, LLMClient]] = []

        # Only swap clients when the parent is a real LLMClient — we don't have
        # a meaningful way to "fork" a MockLLMClient's call counters and the
        # mock is single-threaded-friendly anyway.
        if isinstance(parent_llm, LLMClient):
            for agent in agents:
                if getattr(agent, "_llm", None) is parent_llm:
                    child = LLMClient(
                        api_key=parent_llm.api_key,
                        model=parent_llm.model,
                        temperature=parent_llm.temperature,
                        max_tokens=parent_llm.max_tokens,
                        base_url=parent_llm.base_url,
                    )
                    agent._llm = child  # type: ignore[attr-defined]
                    swapped.append((agent, child))

        try:
            with ThreadPoolExecutor(max_workers=len(agents)) as executor:
                futures = [executor.submit(a.execute, context) for a in agents]
                # Surface the first exception (others would be lost otherwise)
                for f in futures:
                    f.result()
        finally:
            # Restore parent client and roll up token counters
            for agent, child in swapped:
                agent._llm = parent_llm  # type: ignore[attr-defined]
                if isinstance(parent_llm, LLMClient) and isinstance(child, LLMClient):
                    parent_llm._total_prompt_tokens += child._total_prompt_tokens
                    parent_llm._total_completion_tokens += child._total_completion_tokens
                    parent_llm._total_calls += child._total_calls

        return context

    def _should_reflect(self, context: SharedContext) -> bool:
        """
        Decide whether to run the reflection loop after reasoning.

        Reflection is enabled by default. It can be disabled via the
        constructor for comparison experiments (e.g., comparing
        results with and without reflection).
        """
        return self._enable_reflection

    def _reflection_loop(self, context: SharedContext) -> SharedContext:
        """
        Run the reflection-revision loop.

        This is the key mechanism that makes the system truly agentic:

        1. ReflectionAgent reviews the reasoning output
        2. If is_consistent=True: done, proceed to DecisionAgent
        3. If is_consistent=False AND revision budget remains:
           - Increment revision_count
           - Re-run ReasoningAgent (which sees the reflection feedback)
           - Re-run ReflectionAgent to review the revised reasoning
        4. Repeat until consistent or max_revisions reached

        The ReasoningAgent's _build_prompt() method automatically
        includes the reflection feedback when revision_count > 0,
        creating a genuine feedback loop between agents.
        """
        reflection_agent = ReflectionAgent(self._llm)
        reasoning_agent = ReasoningAgent(self._llm)

        # First reflection pass
        context = reflection_agent.execute(context)

        # Revision loop: re-reason and re-reflect if needed
        while context.needs_revision and context.revision_count < context.max_revisions:
            context.revision_count += 1
            context.needs_revision = False  # Reset before re-reasoning

            context.add_log(
                "Orchestrator",
                "revision_cycle",
                f"Starting revision {context.revision_count}/{context.max_revisions}",
            )

            # Re-run reasoning with reflection feedback in prompt
            context = reasoning_agent.execute(context)

            # Re-run reflection to check the revised reasoning
            context = reflection_agent.execute(context)

        return context

    def _save_to_memory(self, context: SharedContext) -> None:
        """
        Save the completed matching result to memory for future retrieval.

        Creates a MemoryEntry from the current context and adds it to
        the memory store. This is how the system accumulates knowledge
        across runs.
        """
        if not context.final_decision:
            return  # Nothing to save if decision wasn't reached

        # Build summaries from context
        cv_summary = ""
        if context.cv_entities:
            cv_summary = context.cv_entities.raw_summary or (
                f"Skills: {', '.join(context.cv_entities.skills[:5])}"
            )

        jd_summary = ""
        if context.jd_entities:
            jd_summary = context.jd_entities.raw_summary or (
                f"Requirements: {', '.join(context.jd_entities.skills[:5])}"
            )

        reasoning_summary = ""
        if context.reasoning_output:
            reasoning_summary = context.reasoning_output.overall_assessment

        # Context Engineering 2.0: record which past memories influenced
        # this decision (logical dependency chain). Enables traceability
        # of how the agent's memory-augmented reasoning evolved over time.
        influenced_by = [m.memory_id for m in context.memory_entries if m.memory_id]

        memory = MemoryEntry(
            cv_summary=cv_summary,
            jd_summary=jd_summary,
            decision_score=context.final_decision.score,
            reasoning_summary=reasoning_summary,
            influenced_by=influenced_by,
        )

        added = self._memory_store.add(memory)
        self._memory_store.save()

        if added:
            deps_str = f", influenced_by={len(influenced_by)}" if influenced_by else ""
            context.add_log(
                "Orchestrator",
                "memory_saved",
                f"Saved result to memory (score: {context.final_decision.score}, "
                f"total memories: {self._memory_store.count}{deps_str})",
            )
        else:
            context.add_log(
                "Orchestrator",
                "memory_skipped_duplicate",
                f"Skipped saving — near-duplicate of existing memory. "
                f"Total memories: {self._memory_store.count}",
            )
