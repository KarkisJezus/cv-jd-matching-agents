"""
ExperimentRunner: executes scenarios across multiple CV-JD pairs.

This is the main evaluation tool for the thesis. It:
1. Loads a dataset of CV-JD pairs (with optional ground-truth scores)
2. Runs each pair through Scenarios A, B, and C
3. Collects metrics for each run
4. Produces a ComparisonReport with per-pair and aggregate results
5. Saves raw results as JSON for later analysis

Usage:
    runner = ExperimentRunner(llm_client)
    report = runner.run_all(dataset)
    print(format_comparison_table(report))

The runner handles MockLLMClient state resets between runs,
so each CV-JD pair starts with a fresh mock state.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from embeddings.similarity import EmbeddingSimilarity
from evaluation.metrics import (
    ComparisonReport,
    ScenarioMetrics,
    compute_metrics,
    format_comparison_table,
)
from llm.client import BaseLLMClient, LLMClient, MockLLMClient
from memory.store import MemoryStore
from models.entities import LabeledMemoryEntry
from orchestrator.orchestrator import Orchestrator


@dataclass
class CVJDPair:
    """A single CV-JD pair for evaluation."""

    pair_id: str
    cv_text: str
    jd_text: str
    ground_truth_score: Optional[float] = None  # Human-assigned score 0-100
    ground_truth_label: Optional[bool] = None    # True=match, False=no-match
    description: str = ""  # Brief description for the report
    reference_reason: str = ""  # Reason text from the dataset
    # True if reference_reason is a curator-written, content-grounded note that
    # should be passed into Tier 2's labeled-memory calibration. Defaults False
    # because the synthetic source dataset's reasons are templated noise (see
    # methodology). Set by load_dataset_from_json when the dataset's _meta
    # contains 'has_curator_reasons': true.
    has_curator_reason: bool = False


class _TraceShim:
    """Minimal attribute-access wrapper around a serialized trace dict.

    Used only on resume: when reloading traces from disk, we need the same
    attribute access pattern that TraceRecord provides (`t.scenario`,
    `t.pair_id`, `t.agent_durations`, `t.to_dict()`), but reconstructing
    full TraceRecord objects is brittle as the schema evolves. The shim
    proxies attribute access into the original dict and re-emits the same
    dict in to_dict(), so downstream consumers don't see the difference.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name: str):
        # Top-level fields: pair_id, scenario, agent_durations, etc.
        if name in self._d:
            return self._d[name]
        # Nested fields: e.g. t.suggested_score from {"reasoning": {...}}
        for nested_key in ("reasoning", "reflection", "decision", "tier2", "memory"):
            section = self._d.get(nested_key)
            if isinstance(section, dict) and name in section:
                return section[name]
        # Fallback: empty list / dict / string — mimics dataclass defaults
        if name.startswith("agent_"):
            return {} if name.endswith("_durations") or name.endswith("_token_usage") else []
        return ""

    def to_dict(self) -> dict:
        # Already a dict — return a shallow copy so callers can't mutate cache.
        return dict(self._d)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    scenarios: list[str] = field(default_factory=lambda: ["A", "B", "C"])
    enable_reflection: bool = True
    output_dir: str = "results"
    experiment_name: str = "experiment"
    threshold: float = 50.0           # Score >= threshold → "match"
    memory_mode: str = "isolated"     # "isolated" (default) or "shared"
    run_baseline: bool = False        # Include embedding-only baseline
    model_name: str = "gpt-4o-mini"   # For reproducibility metadata
    memory_dir: Optional[str] = None  # Persistent memory dir for shared mode.
                                      # If None + shared mode: uses tempfile (per-run).
                                      # If set + shared mode: persists across runs.

    # Tier 2 architecture options (ignored when architecture="tier1")
    architecture: str = "tier1"        # "tier1" (legacy) or "tier2" (new)
    streaming_memory_mode: str = "cold-start"  # "cold-start" | "continue-stream" | "fresh-build"

    # Concurrency: number of worker threads for pair-level parallelism.
    # 1 = sequential (default, preserves the original execution model).
    # >1 = parallel via ThreadPoolExecutor with chunk_size = parallel_workers.
    # For Tier 2 Scenario C, labeled-memory commits batch at chunk boundaries
    # to preserve the streaming-protocol invariant while allowing concurrent
    # execution within a chunk.
    parallel_workers: int = 1

    # Checkpointing: save partial results every N completed pairs.
    # 0 disables checkpointing (only final save). >0 writes the same output
    # files (metrics JSON, traces, details) at this interval — so a crash or
    # rate-limit kill leaves usable partial data on disk.
    # In parallel mode, the actual checkpoint cadence rounds up to the nearest
    # chunk boundary (chunks of size parallel_workers).
    checkpoint_every: int = 0

    # Resume: skip pair_ids whose results already exist in the output JSON.
    # When True, the runner reads the existing metrics file from output_dir,
    # collects pair_ids from pair_results, and excludes them from this run.
    # Combined with checkpoint_every, lets you safely interrupt and continue.
    resume: bool = False


@dataclass
class TraceRecord:
    """
    Full reasoning/decision trace for one scenario run on one CV-JD pair.

    Preserved for qualitative analysis in the thesis — lets you inspect
    exactly what the system "thought" for each pair, and compare traces
    across scenarios A/B/C side by side.
    """

    pair_id: str = ""
    scenario: str = ""

    # Reasoning trace
    strengths: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    overall_assessment: str = ""
    suggested_score: float = 0.0

    # Reflection trace
    reflection_is_consistent: Optional[bool] = None
    reflection_issues: list[str] = field(default_factory=list)
    reflection_suggestions: list[str] = field(default_factory=list)
    revision_reason: str = ""

    # Decision trace
    final_score: float = 0.0
    confidence: float = 0.0
    recommendation: str = ""
    explanation: str = ""
    key_factors: list[str] = field(default_factory=list)

    # Per-agent efficiency (populated from context.logs and context.agent_token_usage)
    agent_durations: dict[str, float] = field(default_factory=dict)
    agent_token_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    # Memory impact (Scenario C only)
    memories_retrieved: int = 0
    best_memory_similarity: float = 0.0

    # Tier 2 — profiles
    cv_profile: Optional[dict] = None  # CandidateProfile.model_dump() if Tier 2
    jd_profile: Optional[dict] = None  # IdealCandidateProfile.model_dump() if Tier 2

    # Tier 2 — two-pass decision
    initial_score: float = 0.0          # Pass 1 score (Tier 2 Scenario C)
    initial_recommendation: str = ""    # Pass 1 recommendation
    calibration_decision: str = ""      # 'lower' | 'raise' | 'keep' | ''
    calibration_adjustment: float = 0.0 # final_score - initial_score (signed)
    calibration_rationale: str = ""
    calibration_pattern: str = ""
    calibration_n_supporting: int = 0
    calibration_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "pair_id": self.pair_id,
            "scenario": self.scenario,
            "reasoning": {
                "strengths": self.strengths,
                "gaps": self.gaps,
                "concerns": self.concerns,
                "overall_assessment": self.overall_assessment,
                "suggested_score": self.suggested_score,
            },
            "reflection": {
                "is_consistent": self.reflection_is_consistent,
                "issues": self.reflection_issues,
                "suggestions": self.reflection_suggestions,
                "revision_reason": self.revision_reason,
            },
            "decision": {
                "final_score": self.final_score,
                "confidence": self.confidence,
                "recommendation": self.recommendation,
                "explanation": self.explanation,
                "key_factors": self.key_factors,
            },
            "efficiency": {
                "agent_durations": self.agent_durations,
                "agent_token_usage": self.agent_token_usage,
            },
            "memory": {
                "memories_retrieved": self.memories_retrieved,
                "best_memory_similarity": self.best_memory_similarity,
            },
            "tier2": {
                "cv_profile": self.cv_profile,
                "jd_profile": self.jd_profile,
                "initial_score": self.initial_score,
                "initial_recommendation": self.initial_recommendation,
                "calibration_decision": self.calibration_decision,
                "calibration_adjustment": self.calibration_adjustment,
                "calibration_rationale": self.calibration_rationale,
                "calibration_pattern": self.calibration_pattern,
                "calibration_n_supporting": self.calibration_n_supporting,
                "calibration_confidence": self.calibration_confidence,
            },
        }


def _extract_trace(context, pair_id: str) -> TraceRecord:
    """Extract a TraceRecord from a completed SharedContext."""
    from models.shared_context import SharedContext
    ctx: SharedContext = context

    trace = TraceRecord(pair_id=pair_id, scenario=ctx.scenario)

    if ctx.reasoning_output:
        r = ctx.reasoning_output
        trace.strengths = r.strengths
        trace.gaps = r.gaps
        trace.concerns = r.concerns
        trace.overall_assessment = r.overall_assessment
        trace.suggested_score = r.suggested_score

    if ctx.reflection_output:
        ref = ctx.reflection_output
        trace.reflection_is_consistent = ref.is_consistent
        trace.reflection_issues = ref.issues_found
        trace.reflection_suggestions = ref.suggestions
        trace.revision_reason = ref.revision_reason

    if ctx.final_decision:
        d = ctx.final_decision
        trace.final_score = d.score
        trace.confidence = d.confidence
        trace.recommendation = d.recommendation
        trace.explanation = d.explanation
        trace.key_factors = d.key_factors

    # Per-agent durations: aggregate from the execution log.
    # Each agent's "completed" log entry carries its duration_seconds.
    # Agents that revised (ReasoningAgent, ReflectionAgent) may appear
    # multiple times; sum their durations.
    for log in ctx.logs:
        if log.action == "completed" and log.agent_name != "Orchestrator":
            trace.agent_durations[log.agent_name] = (
                trace.agent_durations.get(log.agent_name, 0.0) + log.duration_seconds
            )

    # Per-agent token usage (populated by BaseAgent.execute for LLM agents)
    trace.agent_token_usage = dict(ctx.agent_token_usage)

    # Memory metrics (Scenario C)
    if ctx.memory_entries:
        trace.memories_retrieved = len(ctx.memory_entries)
        trace.best_memory_similarity = max(
            m.similarity_to_current for m in ctx.memory_entries
        )
    # Tier 2 also populates labeled_memory_entries
    if ctx.labeled_memory_entries:
        trace.memories_retrieved = len(ctx.labeled_memory_entries)
        trace.best_memory_similarity = max(
            m.similarity_to_current for m in ctx.labeled_memory_entries
        )

    # Tier 2 — profiles
    if ctx.cv_profile:
        trace.cv_profile = ctx.cv_profile.model_dump()
    if ctx.jd_profile:
        trace.jd_profile = ctx.jd_profile.model_dump()

    # Tier 2 — two-pass decision
    if ctx.initial_decision:
        trace.initial_score = ctx.initial_decision.score
        trace.initial_recommendation = ctx.initial_decision.recommendation
    if ctx.calibration_output:
        cal = ctx.calibration_output
        trace.calibration_decision = cal.calibration_decision
        trace.calibration_rationale = cal.rationale
        trace.calibration_pattern = cal.pattern_observed
        trace.calibration_n_supporting = cal.n_supporting_memories
        trace.calibration_confidence = cal.confidence
        # Adjustment = final - initial (positive means raised, negative means lowered)
        if ctx.initial_decision and ctx.final_decision:
            trace.calibration_adjustment = round(
                ctx.final_decision.score - ctx.initial_decision.score, 2,
            )

    return trace


class ExperimentRunner:
    """
    Runs evaluation experiments across multiple CV-JD pairs and scenarios.

    The runner coordinates:
    - Orchestrator setup for each scenario
    - MockLLMClient state resets between runs
    - Metric collection and aggregation
    - Full trace preservation for qualitative analysis
    - Result persistence

    For Scenario C, memory handling depends on memory_mode:
    - "isolated" (default): each pair gets a fresh temporary memory store.
      This is the correct mode for baseline evaluation (no cross-contamination).
    - "shared": one MemoryStore persists across all pairs. Each completed
      pair's result is saved, so later pairs can retrieve earlier decisions.
      This is an experimental mode for studying memory's impact.

    IMPORTANT: Memory never stores ground truth labels or scores.
    Memory stores only: cv_summary, jd_summary, decision_score, reasoning_summary.
    This is a contextual augmentation mechanism that reuses prior reasoning
    outputs — not machine learning. No model weights are updated.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        config: Optional[ExperimentConfig] = None,
    ):
        self._llm_client = llm_client
        self._config = config or ExperimentConfig()
        self._traces: list[TraceRecord] = []
        self._pair_details: list[dict] = []  # Per-pair JSONL rows
        self._token_usage: list[dict] = []   # Per-scenario token usage

        # Create ONE EmbeddingSimilarity instance shared across all pairs,
        # scenarios, and components. Without this, every new Orchestrator /
        # MemoryStore / BaselineEvaluator created per pair would load the
        # sentence-BERT model fresh from disk — a measurable performance hit
        # (the "Loading weights 103/103" message would print once per pair).
        # The model is lazy-loaded on first use and then cached in the instance.
        self._embedding_sim = EmbeddingSimilarity()

        # Shared memory store for Scenario C (only when memory_mode="shared").
        # If config.memory_dir is set, use it as a PERSISTENT directory — the
        # store auto-loads any existing memories on init, and we save at the
        # end of run_all(). This enables memory to accumulate across separate
        # program invocations (thesis train/test workflow).
        # If not set, fall back to a per-run tempfile (original behavior).
        self._shared_memory: Optional[MemoryStore] = None
        if self._config.memory_mode == "shared":
            if self._config.memory_dir:
                self._shared_memory = MemoryStore(
                    memory_dir=self._config.memory_dir,
                    embedding_similarity=self._embedding_sim,
                )
                existing = self._shared_memory.count
                if existing > 0:
                    print(f"Loaded {existing} existing memories from {self._config.memory_dir}")
            else:
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="memory_shared_")
                self._shared_memory = MemoryStore(
                    memory_dir=temp_dir,
                    embedding_similarity=self._embedding_sim,
                )

        # Tier 2 labeled memory store. Initialized lazily in run_all() once we
        # know the input pair_ids (needed for continue-stream overlap detection).
        self._labeled_memory_store = None  # set in run_all() for tier2 + Scenario C

    @property
    def traces(self) -> list[TraceRecord]:
        """All collected traces from the last run_all() call."""
        return self._traces

    def run_all(self, dataset: list[CVJDPair]) -> ComparisonReport:
        """
        Run all scenarios on all CV-JD pairs and produce a comparison report.

        Args:
            dataset: List of CVJDPair objects to evaluate

        Returns:
            ComparisonReport with per-pair and aggregate results
        """
        report = ComparisonReport()
        self._traces = []
        self._pair_details = []
        self._token_usage = []

        # Generate run ID for reproducibility
        run_id = time.strftime("%Y-%m-%d_%H%M%S")

        # Resume support: load existing partial results if resume=True and the
        # output file exists. We re-seed the ComparisonReport, traces, details
        # and token_usage from the saved state, then filter the input dataset
        # to skip already-completed pair_ids.
        completed_pair_ids: set[str] = set()
        if self._config.resume:
            completed_pair_ids = self._load_resume_state(report)
            if completed_pair_ids:
                pre_count = len(dataset)
                dataset = [p for p in dataset if p.pair_id not in completed_pair_ids]
                print(f"Resume: {len(completed_pair_ids)} pairs already complete; "
                      f"{len(dataset)} of {pre_count} remaining.")
            else:
                print("Resume requested but no prior partial results found; starting fresh.")

        print(f"\nRunning experiment: {self._config.experiment_name}")
        print(f"Run ID: {run_id}")
        print(f"Dataset: {len(dataset)} pairs")
        print(f"Architecture: {self._config.architecture}")
        print(f"Scenarios: {', '.join(self._config.scenarios)}")
        print(f"Reflection: {'enabled' if self._config.enable_reflection else 'disabled'}")
        print(f"Memory mode: {self._config.memory_mode}")
        print(f"Threshold: {self._config.threshold}")
        if self._config.run_baseline:
            print("Baseline: embedding-only (enabled)")
        print("-" * 60)

        # Tier 2: initialize the labeled memory store via the streaming protocol.
        # Only needed if Scenario C is in the run.
        if self._config.architecture == "tier2" and "C" in self._config.scenarios:
            from evaluation.streaming_protocol import prepare_memory_store
            memory_dir_for_streaming = (
                self._config.memory_dir
                or f"{self._config.output_dir}/{self._config.experiment_name}_labeled_memory"
            )
            input_pair_ids = [p.pair_id for p in dataset]
            self._labeled_memory_store = prepare_memory_store(
                memory_dir=memory_dir_for_streaming,
                mode=self._config.streaming_memory_mode,
                input_pair_ids=input_pair_ids,
                embedding_similarity=self._embedding_sim,
            )
            print(f"Streaming memory mode: {self._config.streaming_memory_mode}")
            print(f"Labeled memory dir: {memory_dir_for_streaming}")
            print("-" * 60)

        # Run baseline if enabled
        baseline_results = {}
        if self._config.run_baseline:
            from evaluation.baseline import BaselineEvaluator
            baseline_eval = BaselineEvaluator(
                threshold=self._config.threshold,
                embedding_similarity=self._embedding_sim,
            )
            print("\nRunning baseline (embedding-only)...")
            for pair in dataset:
                result = baseline_eval.evaluate(pair.pair_id, pair.cv_text, pair.jd_text)
                baseline_results[pair.pair_id] = result
                print(f"  {pair.pair_id}: score={result.predicted_score:.1f}")

        # --- Pair execution: sequential or chunked-parallel ---
        workers = max(1, int(self._config.parallel_workers))

        if workers == 1:
            # Sequential: original execution model. Commits memory immediately
            # after each pair's Scenario C, preserving the strict streaming
            # invariant (pair N+1 sees memory of pairs 1..N).
            for i, pair in enumerate(dataset, 1):
                print(f"\n[{i}/{len(dataset)}] Pair: {pair.pair_id}")
                if pair.description:
                    print(f"  Description: {pair.description}")
                pair_outcome = self._run_one_pair(pair, run_id)
                # Commit Tier 2 memory for this pair before moving on
                for entry in pair_outcome["memory_entries"]:
                    self._labeled_memory_store.add_labeled(entry)
                if pair_outcome["memory_entries"] and self._labeled_memory_store is not None:
                    self._labeled_memory_store.save()
                self._aggregate_pair_outcome(report, pair, pair_outcome, baseline_results)

                # Periodic checkpoint: save partial results every N pairs.
                if self._config.checkpoint_every and i % self._config.checkpoint_every == 0:
                    self._save_checkpoint(report, completed=i, total=len(dataset))
        else:
            # Parallel: chunked execution.
            # - Each chunk of `workers` pairs runs concurrently in a thread pool.
            # - Within a chunk, all pairs see the labeled-memory state from
            #   PREVIOUS chunks; no within-chunk dependency.
            # - At chunk boundary, all of this chunk's memory entries are committed
            #   atomically, and the next chunk starts.
            # - This is "chunked streaming" — preserves the streaming invariant
            #   across chunks while allowing concurrent execution within each chunk.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            print(
                f"\nParallel mode: workers={workers}, chunk_size={workers}. "
                "Tier 2 labeled memory commits batch per chunk."
            )

            for chunk_start in range(0, len(dataset), workers):
                chunk = dataset[chunk_start:chunk_start + workers]
                chunk_end = chunk_start + len(chunk)
                print(
                    f"\n=== Chunk pairs [{chunk_start + 1}..{chunk_end}] of {len(dataset)} ==="
                )

                results_by_pid: dict[str, dict] = {}
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_pair = {
                        executor.submit(self._run_one_pair, pair, run_id): pair
                        for pair in chunk
                    }
                    for future in as_completed(future_to_pair):
                        pair = future_to_pair[future]
                        try:
                            results_by_pid[pair.pair_id] = future.result()
                        except Exception as e:
                            # The worker should have caught its own exceptions, but
                            # belt-and-braces: build an empty outcome so aggregation
                            # doesn't choke on a missing pair_id.
                            print(f"FATAL pair {pair.pair_id}: {type(e).__name__}: {e}")
                            results_by_pid[pair.pair_id] = {
                                "pair_id": pair.pair_id,
                                "metrics": {},
                                "traces": [],
                                "details": [],
                                "usages": [],
                                "memory_entries": [],
                            }

                # Commit memory for the entire chunk at once (atomic save)
                committed = 0
                for pair in chunk:
                    outcome = results_by_pid.get(pair.pair_id)
                    if outcome is None:
                        continue
                    for entry in outcome["memory_entries"]:
                        self._labeled_memory_store.add_labeled(entry)
                        committed += 1
                if committed and self._labeled_memory_store is not None:
                    self._labeled_memory_store.save()
                    print(f"  Committed {committed} labeled-memory entries from this chunk.")

                # Aggregate in dataset order so report rows are deterministic
                for pair in chunk:
                    outcome = results_by_pid.get(pair.pair_id)
                    if outcome is None:
                        continue
                    self._aggregate_pair_outcome(report, pair, outcome, baseline_results)

                # Periodic checkpoint at chunk boundaries. The chunk-end is the
                # natural checkpoint point in parallel mode: memory has just been
                # committed atomically, and report is up-to-date through chunk_end.
                completed_so_far = chunk_end
                if (
                    self._config.checkpoint_every
                    and completed_so_far % self._config.checkpoint_every < workers
                    and completed_so_far > 0
                ):
                    self._save_checkpoint(
                        report, completed=completed_so_far, total=len(dataset),
                    )

        report.compute_summary(threshold=self._config.threshold)

        # Persist accumulated memory to disk if using a custom memory_dir.
        # This enables the train/test workflow: the store keeps growing
        # across separate program invocations. Tempfile stores don't need
        # this since they're ephemeral anyway.
        if (
            self._config.memory_mode == "shared"
            and self._config.memory_dir
            and self._shared_memory is not None
        ):
            self._shared_memory.save()
            print(
                f"\nPersisted {self._shared_memory.count} memories to "
                f"{self._config.memory_dir}"
            )

        return report

    def _run_one_pair(self, pair: CVJDPair, run_id: str) -> dict:
        """
        Run all configured scenarios for one pair.

        Returns an outcome dict with everything the caller needs to commit
        memory and aggregate into the ComparisonReport. This method is the
        unit of parallelism: in parallel mode, it runs in a worker thread;
        in sequential mode, it runs inline. It does NOT commit memory or
        mutate the runner's shared state (other than via short, append-only
        operations to instance lists which are GIL-safe in CPython for list
        appends, and which we additionally serialize per-chunk).

        Side effects: prints per-scenario progress lines. In parallel mode
        these will interleave across pairs — that's expected.
        """
        outcome = {
            "pair_id": pair.pair_id,
            "metrics": {},        # scenario -> ScenarioMetrics
            "traces": [],         # list[TraceRecord]
            "details": [],        # list[dict] (JSONL rows)
            "usages": [],         # list[dict] per scenario
            "memory_entries": [], # list[LabeledMemoryEntry] (Tier 2 Scenario C only)
        }

        for scenario in self._config.scenarios:
            start = time.time()
            try:
                m, trace, memory_entry, usage = self._run_single(pair, scenario)
            except Exception as e:
                duration = time.time() - start
                print(f"[{pair.pair_id}/{scenario}] ERROR ({duration:.1f}s): {e}")
                m = ScenarioMetrics(
                    scenario=scenario,
                    pair_id=pair.pair_id,
                    final_score=0.0,
                    confidence=0.0,
                    recommendation="error",
                )
                trace = TraceRecord(
                    pair_id=pair.pair_id,
                    scenario=scenario,
                    overall_assessment=f"Error: {e}",
                )
                outcome["metrics"][scenario] = m
                outcome["traces"].append(trace)
                continue

            duration = time.time() - start
            outcome["metrics"][scenario] = m
            outcome["traces"].append(trace)
            if memory_entry is not None:
                outcome["memory_entries"].append(memory_entry)

            # Per-scenario usage + cost
            usage_row = {"pair_id": pair.pair_id, "scenario": scenario}
            usage_row.update(usage)
            outcome["usages"].append(usage_row)

            tokens_str = f", tokens={usage.get('total_tokens', 0)}" if usage else ""
            print(
                f"[{pair.pair_id}/{scenario}] "
                f"score={m.final_score:.1f}, "
                f"confidence={m.confidence:.0%}, "
                f"revisions={m.revision_count} "
                f"({duration:.1f}s{tokens_str})"
            )

            # Per-pair detail row (JSONL)
            predicted_label = m.final_score >= self._config.threshold
            detail = {
                "run_id": run_id,
                "model": self._config.model_name,
                "threshold": self._config.threshold,
                "pair_id": pair.pair_id,
                "scenario": scenario,
                "memory_mode": self._config.memory_mode,
                "predicted_score": m.final_score,
                "predicted_label": predicted_label,
                "ground_truth_label": pair.ground_truth_label,
                "correct": (
                    predicted_label == pair.ground_truth_label
                    if pair.ground_truth_label is not None
                    else None
                ),
                "confidence": m.confidence,
                "recommendation": m.recommendation,
                "reasoning_summary": trace.overall_assessment,
                "revision_count": m.revision_count,
            }
            if usage:
                from config.pricing import compute_cost
                detail["token_usage"] = usage
                detail["cost_usd"] = round(
                    compute_cost(
                        self._config.model_name,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    ),
                    6,
                )
            outcome["details"].append(detail)

        return outcome

    def _aggregate_pair_outcome(
        self,
        report: ComparisonReport,
        pair: CVJDPair,
        outcome: dict,
        baseline_results: dict,
    ) -> None:
        """
        Merge a pair's outcome into the report and the runner's shared lists.
        Called from both sequential and parallel paths AFTER memory is committed.
        Caller is responsible for ensuring chunk-level ordering when in parallel
        mode so the report rows match dataset order.
        """
        # Append to runner-level lists (single-threaded at this point)
        self._traces.extend(outcome["traces"])
        self._pair_details.extend(outcome["details"])
        self._token_usage.extend(outcome["usages"])

        metrics = outcome["metrics"]
        report.add_pair(
            pair_id=pair.pair_id,
            metrics_a=metrics.get("A"),
            metrics_b=metrics.get("B"),
            metrics_c=metrics.get("C"),
            ground_truth_score=pair.ground_truth_score,
            ground_truth_label=pair.ground_truth_label,
            baseline_score=(
                baseline_results[pair.pair_id].predicted_score
                if pair.pair_id in baseline_results else None
            ),
        )

    def _run_single(
        self,
        pair: CVJDPair,
        scenario: Literal["A", "B", "C"],
    ) -> tuple[ScenarioMetrics, TraceRecord, Optional[LabeledMemoryEntry], dict]:
        """
        Run a single CV-JD pair through one scenario.

        Handles:
        - Fresh thread-local LLM client (counters not shared across threads)
        - Temporary memory store for Scenario C
        - Metric computation from completed context
        - Trace extraction for qualitative analysis
        - Build (but do NOT commit) the Tier 2 labeled-memory entry — caller
          decides when to commit. Sequential path commits immediately; the
          parallel path batches commits at chunk boundaries to preserve the
          streaming protocol while allowing concurrent execution.

        Returns:
            (ScenarioMetrics, TraceRecord, Optional[LabeledMemoryEntry], usage_dict)
        """
        llm = self._get_fresh_llm()

        # Set up memory store for Scenario C
        memory_store = None
        if scenario == "C":
            if self._config.memory_mode == "shared" and self._shared_memory is not None:
                # Shared mode: reuse the same store across all pairs.
                # Each pair's result accumulates, so later pairs
                # can retrieve earlier decisions.
                memory_store = self._shared_memory
            else:
                # Isolated mode (default): fresh temp store per pair.
                # No cross-pair contamination — correct for baseline eval.
                # Pass the shared embedding instance so we don't reload
                # the sentence-BERT model for each pair.
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="memory_eval_")
                memory_store = MemoryStore(
                    memory_dir=temp_dir,
                    embedding_similarity=self._embedding_sim,
                )

        # Pass the shared embedding instance to Orchestrator. Without this,
        # every new Orchestrator would create its own EmbeddingSimilarity and
        # re-load the sentence-BERT model from disk on first use.
        orchestrator = Orchestrator(
            llm_client=llm,
            embedding_similarity=self._embedding_sim,
            memory_store=memory_store,
            labeled_memory_store=self._labeled_memory_store,  # Tier 2 only; harmless in Tier 1
            enable_reflection=self._config.enable_reflection,
            architecture=self._config.architecture,
        )

        context = orchestrator.run(
            cv_text=pair.cv_text,
            jd_text=pair.jd_text,
            scenario=scenario,
        )

        # Tier 2: build the labeled-memory entry but DO NOT commit. The caller
        # (sequential or parallel path) is responsible for committing at the
        # right moment. Sequential commits immediately; parallel commits at
        # chunk boundaries to keep the labeled store thread-safe and to preserve
        # the streaming-protocol invariant within a chunk.
        memory_entry = None
        if (
            self._config.architecture == "tier2"
            and scenario == "C"
            and self._labeled_memory_store is not None
            and context.final_decision is not None
            and pair.ground_truth_label is not None
        ):
            from evaluation.streaming_protocol import build_labeled_entry
            cv_summary = (
                context.cv_profile.raw_summary if context.cv_profile else ""
            )
            jd_summary = (
                context.jd_profile.raw_summary if context.jd_profile else ""
            )
            detected_role = (
                context.jd_profile.detected_role if context.jd_profile else ""
            )
            initial_score = (
                context.initial_decision.score
                if context.initial_decision else context.final_decision.score
            )
            reasoning_summary = (
                context.reasoning_output.overall_assessment
                if context.reasoning_output else ""
            )
            influenced_by = [
                m.memory_id for m in context.labeled_memory_entries if m.memory_id
            ]
            memory_entry = build_labeled_entry(
                pair_id=pair.pair_id,
                cv_profile_summary=cv_summary,
                jd_profile_summary=jd_summary,
                detected_role=detected_role,
                system_score=context.final_decision.score,
                system_recommendation=context.final_decision.recommendation,
                system_initial_score=initial_score,
                system_reasoning_summary=reasoning_summary,
                ground_truth_label=pair.ground_truth_label,
                # Reasons from the synthetic source dataset are templated noise
                # ("Lacks cloud experience" on a JD that doesn't mention cloud)
                # and are stripped by default. But curator-curated gold sets
                # carry hand-written, content-grounded notes that the calibration
                # agent SHOULD see — those are flagged via has_curator_reason
                # set by the dataset loader. See methodology section.
                ground_truth_reason=(
                    pair.reference_reason if pair.has_curator_reason else ""
                ),
                threshold=self._config.threshold,
                influenced_by=influenced_by,
            )

        metrics = compute_metrics(context, pair_id=pair.pair_id)
        trace = _extract_trace(context, pair_id=pair.pair_id)

        # Capture token usage from the local llm (thread-safe — caller will not
        # see usage from another thread because the client is local to this call).
        usage = llm.usage if hasattr(llm, "usage") else {}

        return metrics, trace, memory_entry, usage

    def _get_fresh_llm(self) -> BaseLLMClient:
        """
        Return a thread-local LLM client.

        Concurrency requirement: each pair-task runs in its own worker thread
        and accumulates token usage on its local client. Sharing one LLMClient
        across threads would cause counter races (the usage property reads the
        same _total_prompt_tokens that another thread is writing).

        For LLMClient: builds a fresh instance with the same config.
        For MockLLMClient: builds a fresh mock to reset call counters.
        """
        if isinstance(self._llm_client, MockLLMClient):
            return MockLLMClient()
        if isinstance(self._llm_client, LLMClient):
            return LLMClient(
                api_key=self._llm_client.api_key,
                model=self._llm_client.model,
                temperature=self._llm_client.temperature,
                max_tokens=self._llm_client.max_tokens,
                base_url=self._llm_client.base_url,
            )
        return self._llm_client

    def _save_checkpoint(
        self, report: ComparisonReport, completed: int, total: int,
    ) -> None:
        """Save partial results mid-run.

        Writes the same files as save_results — same format — so the dashboard
        can load mid-run output without any special handling. Run only emits a
        single-line progress message (full save details would spam logs every
        N pairs).
        """
        # Recompute summary so the checkpointed file has up-to-date metrics.
        # Cheap: the full report state is in memory.
        report.compute_summary(threshold=self._config.threshold)
        try:
            self.save_results(report)
            msg = f"  [checkpoint saved: {completed}/{total} pairs]"
        except Exception as e:
            msg = f"  [checkpoint save FAILED: {type(e).__name__}: {e}]"
        # Use ASCII-only glyphs in the message above; Windows console
        # defaults to cp1252 and can't render ✓/⚠. Wrap the print itself
        # in a try/except too so a transient print error never kills the
        # main run.
        try:
            print(msg)
        except Exception:
            pass

    def _load_resume_state(self, report: ComparisonReport) -> set[str]:
        """Load partial results from a prior run for resume support.

        Reads {output_dir}/{experiment_name}.json (and matching traces +
        details), repopulates report.pair_results, self._traces,
        self._pair_details, and self._token_usage so the resumed run continues
        seamlessly.

        Returns the set of pair_ids that are already complete.
        """
        out_dir = Path(self._config.output_dir)
        metrics_file = out_dir / f"{self._config.experiment_name}.json"
        traces_file = out_dir / f"{self._config.experiment_name}_traces.json"
        details_file = out_dir / f"{self._config.experiment_name}_details.jsonl"

        if not metrics_file.exists():
            return set()

        try:
            data = json.loads(metrics_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ⚠ resume: could not parse {metrics_file}: {e}; starting fresh.")
            return set()

        # Repopulate pair_results — these are plain dicts, ComparisonReport
        # consumes them as-is for its summary / classification logic.
        prior_pair_results = data.get("pair_results", []) or []
        for pr in prior_pair_results:
            report.pair_results.append(pr)

        # Repopulate traces. The save format flattens TraceRecord into nested
        # dicts; we need to reconstruct the lightweight attribute access used
        # later in save_results (t.scenario, t.pair_id, t.agent_durations).
        # A simple namespace shim is enough — we don't need a true TraceRecord
        # because the only use sites are scenario/pair_id filtering and a
        # final to_dict() pass.
        if traces_file.exists():
            try:
                raw_traces = json.loads(traces_file.read_text(encoding="utf-8"))
                self._traces = [_TraceShim(d) for d in raw_traces]
            except (OSError, json.JSONDecodeError):
                self._traces = []

        # Repopulate per-pair details (JSONL).
        if details_file.exists():
            try:
                rows = []
                for line in details_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        rows.append(json.loads(line))
                self._pair_details = rows
            except (OSError, json.JSONDecodeError):
                self._pair_details = []

        # Token usage was aggregated per-scenario in the saved file, not
        # per-pair, so we can't perfectly reconstruct self._token_usage.
        # That's fine — token totals will simply be undercounted for the
        # already-completed pairs in any post-resume save. Cost reports will
        # reflect only the resumed segment, which the user can document.

        return {pr["pair_id"] for pr in prior_pair_results if "pair_id" in pr}

    def save_results(
        self,
        report: ComparisonReport,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Save the experiment results to JSON.

        Saves two files:
        - {experiment_name}.json  — metrics, summary, classification
        - {experiment_name}_traces.json — full reasoning/decision traces

        Args:
            report: The ComparisonReport to save
            output_dir: Directory to save results (defaults to config)

        Returns:
            Path to the saved metrics JSON file
        """
        out_dir = Path(output_dir or self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics + classification
        metrics_file = out_dir / f"{self._config.experiment_name}.json"

        classification_data = {}
        for label, cm in report.classification.items():
            classification_data[label] = {
                "accuracy": round(cm.accuracy, 4),
                "precision": round(cm.precision, 4),
                "recall": round(cm.recall, 4),
                "f1": round(cm.f1, 4),
                "true_positives": cm.true_positives,
                "false_positives": cm.false_positives,
                "true_negatives": cm.true_negatives,
                "false_negatives": cm.false_negatives,
                "threshold": cm.threshold,
            }

        # Aggregate token usage + dollar cost per scenario
        from config.pricing import compute_cost, get_price_per_token
        model_name = self._config.model_name
        price_per_tok = get_price_per_token(model_name)

        token_summary = {}
        if self._token_usage:
            for label in self._config.scenarios:
                scenario_usage = [u for u in self._token_usage if u["scenario"] == label]
                if scenario_usage:
                    total_prompt = sum(u.get("prompt_tokens", 0) for u in scenario_usage)
                    total_completion = sum(u.get("completion_tokens", 0) for u in scenario_usage)
                    total_calls = sum(u.get("total_calls", 0) for u in scenario_usage)
                    n = len(scenario_usage)
                    total_cost = compute_cost(model_name, total_prompt, total_completion)
                    token_summary[label] = {
                        "total_prompt_tokens": total_prompt,
                        "total_completion_tokens": total_completion,
                        "total_tokens": total_prompt + total_completion,
                        "total_llm_calls": total_calls,
                        "mean_tokens_per_pair": (total_prompt + total_completion) / n,
                        "mean_calls_per_pair": total_calls / n,
                        "n_pairs": n,
                        "total_cost_usd": round(total_cost, 6),
                        "mean_cost_per_pair_usd": round(total_cost / n, 6),
                    }

        # Aggregate per-agent durations and tokens across all pairs per scenario.
        # Uses the per-trace data populated in _extract_trace.
        agent_efficiency = {}
        for label in self._config.scenarios:
            scenario_traces = [t for t in self._traces if t.scenario == label]
            if not scenario_traces:
                continue
            agent_stats: dict[str, dict] = {}
            for trace in scenario_traces:
                for agent_name, dur in trace.agent_durations.items():
                    stats = agent_stats.setdefault(
                        agent_name, {"total_duration_s": 0.0, "calls": 0,
                                     "prompt_tokens": 0, "completion_tokens": 0}
                    )
                    stats["total_duration_s"] += dur
                    stats["calls"] += 1
                for agent_name, usage in trace.agent_token_usage.items():
                    stats = agent_stats.setdefault(
                        agent_name, {"total_duration_s": 0.0, "calls": 0,
                                     "prompt_tokens": 0, "completion_tokens": 0}
                    )
                    stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    stats["completion_tokens"] += usage.get("completion_tokens", 0)
            # Compute means and cost per agent
            n = len(scenario_traces)
            for agent_name, stats in agent_stats.items():
                stats["mean_duration_s"] = stats["total_duration_s"] / n
                stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]
                stats["cost_usd"] = round(
                    compute_cost(model_name, stats["prompt_tokens"], stats["completion_tokens"]),
                    6,
                )
            agent_efficiency[label] = agent_stats

        # Compute memory impact per pair (Scenario C only).
        # For each pair where C retrieved memories, check if C's final score
        # differs meaningfully from A's (which has no memory). A non-trivial
        # difference suggests memory influenced the outcome.
        memory_impact_analysis = None
        if "C" in self._config.scenarios and "A" in self._config.scenarios:
            impact_rows = []
            for pair_res in report.pair_results:
                # Find matching trace for Scenario C to get retrieval info
                pair_id = pair_res.get("pair_id")
                c_trace = next(
                    (t for t in self._traces if t.pair_id == pair_id and t.scenario == "C"),
                    None,
                )
                if c_trace is None:
                    continue
                score_a = pair_res.get("score_A")
                score_c = pair_res.get("score_C")
                if score_a is None or score_c is None:
                    continue
                mem_used = c_trace.memories_retrieved > 0
                score_diff = abs(score_c - score_a)
                impact_rows.append({
                    "pair_id": pair_id,
                    "memory_used": mem_used,
                    "memories_retrieved": c_trace.memories_retrieved,
                    "best_memory_similarity": c_trace.best_memory_similarity,
                    "score_A": score_a,
                    "score_C": score_c,
                    "score_diff": round(score_diff, 2),
                    # Heuristic: memory "impacted" the outcome if it was used
                    # AND scores differ by more than noise (default >2 points).
                    "memory_impacted": mem_used and score_diff > 2.0,
                })
            if impact_rows:
                n_with_memory = sum(1 for r in impact_rows if r["memory_used"])
                n_impacted = sum(1 for r in impact_rows if r["memory_impacted"])
                memory_impact_analysis = {
                    "total_pairs": len(impact_rows),
                    "pairs_with_memory_retrieval": n_with_memory,
                    "pairs_where_memory_impacted": n_impacted,
                    "memory_utilization_rate": (
                        n_with_memory / len(impact_rows) if impact_rows else 0.0
                    ),
                    "memory_impact_rate": (
                        n_impacted / n_with_memory if n_with_memory > 0 else 0.0
                    ),
                    "per_pair": impact_rows,
                }

        # Aggregate reflection statistics per scenario
        reflection_stats = {}
        for label in self._config.scenarios:
            scenario_details = [d for d in self._pair_details if d["scenario"] == label]
            if scenario_details:
                revisions = [d.get("revision_count", 0) for d in scenario_details]
                revised = [r for r in revisions if r > 0]
                reflection_stats[label] = {
                    "total_pairs": len(scenario_details),
                    "pairs_revised": len(revised),
                    "revision_rate": len(revised) / len(scenario_details),
                    "mean_revisions": sum(revisions) / len(revisions),
                    "max_revisions": max(revisions),
                }

        data = {
            "experiment": self._config.experiment_name,
            "config": {
                "scenarios": self._config.scenarios,
                "enable_reflection": self._config.enable_reflection,
                "threshold": self._config.threshold,
                "memory_mode": self._config.memory_mode,
                "model": self._config.model_name,
                "run_baseline": self._config.run_baseline,
                "architecture": self._config.architecture,
                "streaming_memory_mode": self._config.streaming_memory_mode,
            },
            "pair_results": report.pair_results,
            "summary": report.summary,
            "classification": classification_data,
            "token_usage": token_summary,
            "reflection_statistics": reflection_stats,
            "agent_efficiency": agent_efficiency,
            "memory_impact": memory_impact_analysis,
            "pricing": {
                "model": model_name,
                "input_usd_per_token": price_per_tok["input"],
                "output_usd_per_token": price_per_tok["output"],
            },
        }

        metrics_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Save traces for qualitative analysis
        if self._traces:
            traces_file = out_dir / f"{self._config.experiment_name}_traces.json"
            traces_data = [t.to_dict() for t in self._traces]
            traces_file.write_text(
                json.dumps(traces_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Save per-pair detail rows as JSONL (one JSON object per line)
        if self._pair_details:
            details_file = out_dir / f"{self._config.experiment_name}_details.jsonl"
            lines = [json.dumps(row, ensure_ascii=False) for row in self._pair_details]
            details_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        return str(metrics_file)


def load_dataset_from_json(path: str) -> list[CVJDPair]:
    """
    Load a CV-JD evaluation dataset from a JSON file.

    Expected format:
    {
      "pairs": [
        {
          "pair_id": "pair_1",
          "cv_text": "...",
          "jd_text": "...",
          "ground_truth_score": 75.0,   // optional
          "description": "Strong ML match"  // optional
        },
        ...
      ]
    }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    has_curator_reasons = bool(data.get("_meta", {}).get("has_curator_reasons", False))
    pairs = []
    for p in data.get("pairs", []):
        pairs.append(CVJDPair(
            pair_id=p["pair_id"],
            cv_text=p["cv_text"],
            jd_text=p["jd_text"],
            ground_truth_score=p.get("ground_truth_score"),
            ground_truth_label=p.get("ground_truth_label"),
            description=p.get("description", ""),
            reference_reason=p.get("reference_reason", "") or "",
            has_curator_reason=has_curator_reasons,
        ))
    return pairs


def load_dataset_from_files(
    pairs: list[dict],
) -> list[CVJDPair]:
    """
    Load CV-JD pairs from separate text files.

    Each dict in pairs should have:
      - pair_id: str
      - cv_path: str (path to CV text file)
      - jd_path: str (path to JD text file)
      - ground_truth_score: float (optional)
      - description: str (optional)
    """
    result = []
    for p in pairs:
        cv_text = Path(p["cv_path"]).read_text(encoding="utf-8")
        jd_text = Path(p["jd_path"]).read_text(encoding="utf-8")
        result.append(CVJDPair(
            pair_id=p["pair_id"],
            cv_text=cv_text,
            jd_text=jd_text,
            ground_truth_score=p.get("ground_truth_score"),
            description=p.get("description", ""),
        ))
    return result
