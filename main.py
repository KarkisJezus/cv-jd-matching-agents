"""
Main entry point for the CV-JD matching system.

Usage:
    # Run with mock LLM (no API key needed, for development):
    python main.py

    # Run with real LLM (requires OPENAI_API_KEY in .env):
    python main.py --real

    # Run a specific scenario:
    python main.py --scenario B

    # Run Scenario C (with memory):
    python main.py --scenario C
    python main.py --scenario C  # second run will retrieve first run's memory

    # Use custom data files:
    python main.py --cv path/to/cv.txt --jd path/to/jd.txt

    # Run evaluation across multiple CV-JD pairs:
    python main.py --evaluate data/eval_dataset.json
    python main.py --evaluate data/eval_dataset.json --results-dir results/

    # Download HuggingFace dataset and evaluate:
    python main.py --real --hf-dataset AzharAli05/Resume-Screening-Dataset --sample 50

    # Evaluation with baseline and custom threshold:
    python main.py --real --evaluate data/hf_eval_dataset.json --baseline --threshold 60

    # Shared memory experiment (Scenario C cross-pair memory):
    python main.py --real --evaluate data/hf_eval_dataset.json --memory-mode shared

    # Train/test workflow for testing memory accumulation (thesis-rigorous):
    # 1. Download full dataset once
    python main.py --hf-dataset AzharAli05/Resume-Screening-Dataset --sample-all
    # 2. Split 70/30 into train + test (seeded for reproducibility)
    python data/split_dataset.py --input data/hf_full.json --train 0.7 --seed 42
    # 3. Build memory on training set (Scenario C only, persists to disk)
    python main.py --real --evaluate data/hf_full_train.json \
        --scenario C --memory-mode shared --memory-dir data/memory_trained
    # 4. Evaluate on held-out test set with the accumulated memory
    python main.py --real --evaluate data/hf_full_test.json --baseline \
        --memory-mode shared --memory-dir data/memory_trained

    # Memory store management:
    python main.py --memory-stats --memory-dir data/memory_trained
    python main.py --memory-clear --memory-dir data/memory_trained --yes
"""

import argparse
import json
import sys
from pathlib import Path

from config.settings import settings
from embeddings.similarity import EmbeddingSimilarity
from llm.client import LLMClient, MockLLMClient
from memory.store import MemoryStore
from orchestrator.orchestrator import Orchestrator


def load_text(path: str) -> str:
    """Load text from a file."""
    return Path(path).read_text(encoding="utf-8")


def print_result(context) -> None:
    """Print a human-readable summary of the matching result."""
    print("\n" + "=" * 60)
    print("  CV-JD MATCHING RESULT")
    print("=" * 60)

    # Final decision
    if context.final_decision:
        d = context.final_decision
        print(f"\n  Score:          {d.score}/100")
        print(f"  Confidence:     {d.confidence:.0%}")
        print(f"  Recommendation: {d.recommendation}")
        print(f"\n  Explanation:")
        print(f"    {d.explanation}")
        if d.key_factors:
            print(f"\n  Key factors:")
            for f in d.key_factors:
                print(f"    - {f}")

    # Retrieved memories (Scenario C)
    if context.has_memory():
        print(f"\n{'-' * 60}")
        print(f"  Retrieved memories ({len(context.memory_entries)} past decisions):")
        for i, mem in enumerate(context.memory_entries, 1):
            print(f"    [{i}] Score: {mem.decision_score}, "
                  f"Similarity: {mem.similarity_to_current:.3f}")
            if mem.cv_summary:
                print(f"        CV: {mem.cv_summary[:80]}...")
            if mem.reasoning_summary:
                print(f"        Reasoning: {mem.reasoning_summary[:80]}...")

    # Extracted skills comparison
    if context.cv_entities and context.jd_entities:
        print(f"\n{'-' * 60}")
        print(f"  CV skills:  {', '.join(context.cv_entities.skills)}")
        print(f"  JD skills:  {', '.join(context.jd_entities.skills)}")

    # Enrichment results (Scenario B+)
    if context.has_enrichment() and context.normalized_entities:
        ne = context.normalized_entities
        print(f"\n{'-' * 60}")
        print(f"  Enrichment (ESCO taxonomy mapping):")
        print(f"    CV normalized skills:")
        for s in ne.cv_skills:
            code = s.esco_code or "---"
            print(f"      [{code}] {s.original} -> {s.normalized}")
        print(f"    JD normalized skills:")
        for s in ne.jd_skills:
            code = s.esco_code or "---"
            print(f"      [{code}] {s.original} -> {s.normalized}")
        if context.enrichment_notes:
            print(f"    Notes:")
            for note in context.enrichment_notes:
                print(f"      * {note}")

    # Similarity scores
    if context.similarity_scores:
        s = context.similarity_scores
        print(f"\n{'-' * 60}")
        print(f"  Semantic similarity: {s.overall_score:.3f}")
        print(f"  Skill coverage:     {s.coverage_ratio:.0%} ({s.matched_skills_count}/{s.total_jd_skills})")
        print(f"\n  Skill matches:")
        for m in s.skill_matches:
            marker = "[+]" if m.match_type != "below_threshold" else "[-]"
            print(f"    {marker} {m.cv_skill:30s} <-> {m.jd_skill:30s} ({m.similarity:.3f})")

    # Reasoning summary
    if context.reasoning_output:
        r = context.reasoning_output
        print(f"\n{'-' * 60}")
        print(f"  Reasoning (suggested score: {r.suggested_score}):")
        if r.strengths:
            print("    Strengths:")
            for item in r.strengths:
                print(f"      + {item}")
        if r.gaps:
            print("    Gaps:")
            for g in r.gaps:
                print(f"      - {g}")

    # Reflection summary
    if context.reflection_output:
        ref = context.reflection_output
        print(f"\n{'-' * 60}")
        status = "CONSISTENT" if ref.is_consistent else "INCONSISTENT"
        print(f"  Reflection: {status} (confidence: {ref.confidence:.0%})")
        if context.revision_count > 0:
            print(f"  Revision cycles: {context.revision_count}")
        if ref.issues_found:
            print("    Issues found:")
            for issue in ref.issues_found:
                print(f"      ! {issue}")
        if ref.suggestions:
            print("    Suggestions:")
            for sug in ref.suggestions:
                print(f"      > {sug}")
        if ref.revision_reason:
            print(f"    Revision reason: {ref.revision_reason}")

    print(f"\n{'=' * 60}")
    print(f"  Scenario: {context.scenario} | Agents executed: {len(context.logs)} log entries")
    print(f"{'=' * 60}\n")


def run_single(args) -> None:
    """Run a single CV-JD matching (original behavior)."""
    # Select LLM client
    if args.real:
        issues = settings.validate()
        if issues:
            print("Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        llm_client = LLMClient()
        print(f"Using real LLM: {settings.llm_model}")
    else:
        llm_client = MockLLMClient()
        print("Using mock LLM (no API calls)")

    # Set up memory store for Scenario C
    memory_store = None
    if args.scenario == "C":
        memory_dir = args.memory_dir or settings.memory_dir
        memory_store = MemoryStore(memory_dir=memory_dir)
        print(f"Memory store: {memory_dir} ({memory_store.count} existing memories)")

    # Load input texts
    print(f"Loading CV: {args.cv}")
    print(f"Loading JD: {args.jd}")
    cv_text = load_text(args.cv)
    jd_text = load_text(args.jd)

    # Run the orchestrator
    print(f"Running Scenario {args.scenario}...")
    orchestrator = Orchestrator(
        llm_client=llm_client,
        memory_store=memory_store,
    )
    context = orchestrator.run(cv_text, jd_text, scenario=args.scenario)

    # Display results
    print_result(context)

    # Optionally save full context
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            context.model_dump_json(indent=2),
            encoding="utf-8",
        )
        print(f"Full context saved to: {args.output}")


def run_evaluation(args) -> None:
    """Run evaluation across multiple CV-JD pairs."""
    from evaluation.metrics import format_comparison_table
    from evaluation.runner import (
        ExperimentConfig,
        ExperimentRunner,
        load_dataset_from_json,
    )

    # Handle --hf-dataset: download and convert first
    evaluate_path = args.evaluate
    if args.hf_dataset:
        sample_count = None if args.sample_all else args.sample
        evaluate_path = _download_hf_dataset(args.hf_dataset, sample_count)

    # Select LLM client
    model_name = settings.llm_model if args.real else "mock"
    if args.real:
        issues = settings.validate()
        if issues:
            print("Configuration issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        llm_client = LLMClient()
    else:
        llm_client = MockLLMClient()

    # Load dataset
    dataset = load_dataset_from_json(evaluate_path)
    print(f"Loaded {len(dataset)} CV-JD pairs from {evaluate_path}")

    # Configure experiment
    scenarios = [args.scenario] if args.scenario else ["A", "B", "C"]
    config = ExperimentConfig(
        scenarios=scenarios,
        enable_reflection=True,
        output_dir=args.results_dir,
        experiment_name=Path(evaluate_path).stem,
        threshold=args.threshold,
        memory_mode=args.memory_mode,
        run_baseline=args.baseline,
        model_name=model_name,
        memory_dir=args.memory_dir,  # None = tempfile (default), path = persistent
        architecture=args.architecture,
        streaming_memory_mode=args.streaming_memory_mode,
        parallel_workers=args.parallel_workers,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    # Run experiment
    runner = ExperimentRunner(llm_client, config)
    report = runner.run_all(dataset)

    # Display results
    print(format_comparison_table(report))

    # Save results
    results_path = runner.save_results(report, args.results_dir)
    print(f"\nResults saved to: {results_path}")


def run_memory_stats(args) -> None:
    """Print summary of the persistent memory store."""
    memory_dir = args.memory_dir or settings.memory_dir
    store = MemoryStore(memory_dir=memory_dir)

    print(f"Memory store: {memory_dir}")
    print(f"  Count: {store.count}")

    if store.count == 0:
        print("  Empty.")
        return

    # Collect timestamps + recent entries
    memories = store._memories  # list[MemoryEntry]
    timestamps = [m.timestamp for m in memories if m.timestamp]
    if timestamps:
        print(f"  Earliest: {min(timestamps)}")
        print(f"  Latest:   {max(timestamps)}")

    # Show the most recent 5 entries
    recent = sorted(memories, key=lambda m: m.timestamp, reverse=True)[:5]
    print()
    print(f"  Most recent {len(recent)} memories:")
    for m in recent:
        cv_preview = (m.cv_summary or "")[:60].replace("\n", " ")
        reasoning_preview = (m.reasoning_summary or "")[:60].replace("\n", " ")
        deps = len(m.influenced_by)
        print(
            f"    [{m.memory_id[:8]}] score={m.decision_score:.0f} "
            f"deps={deps} | {cv_preview}..."
        )
        if reasoning_preview:
            print(f"              reasoning: {reasoning_preview}...")


def run_memory_clear(args) -> None:
    """Wipe the persistent memory store."""
    memory_dir = args.memory_dir or settings.memory_dir
    store = MemoryStore(memory_dir=memory_dir)
    count = store.count

    print(f"Memory store: {memory_dir}")
    print(f"  Currently holds {count} memories.")

    if count == 0:
        print("  Nothing to clear.")
        return

    if not args.yes:
        resp = input(f"  Wipe all {count} memories? [y/N]: ").strip().lower()
        if resp != "y":
            print("  Aborted.")
            return

    # Remove the JSON + NPY files rather than just clearing in-memory state
    for fname in ("memories.json", "embeddings.npy"):
        path = Path(memory_dir) / fname
        if path.exists():
            path.unlink()
            print(f"  Removed {path}")
    print(f"  Memory store cleared.")


def _download_hf_dataset(
    dataset_name: str,
    sample: int | None,
    output_path: str | None = None,
) -> str:
    """
    Download a HuggingFace dataset and convert to internal format.

    Args:
        dataset_name: HF dataset identifier (e.g., "AzharAli05/Resume-Screening-Dataset")
        sample: Number of pairs to sample. If None, returns the entire dataset.
        output_path: Where to save the converted JSON. Defaults to
            data/hf_full.json for full-dataset downloads, else
            data/hf_eval_dataset.json.
    """
    from data.load_hf_dataset import (
        convert_to_internal_format,
        load_hf_dataset,
        sample_balanced,
    )

    ds = load_hf_dataset()
    rows = sample_balanced(ds, sample)
    data = convert_to_internal_format(rows)

    if output_path is None:
        output_path = (
            "data/hf_full.json" if sample is None else "data/hf_eval_dataset.json"
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {len(data['pairs'])} pairs to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Agent-based CV-JD semantic matching system",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real OpenAI API instead of mock (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--scenario",
        choices=["A", "B", "C"],
        default=None,
        help="Which scenario to run. In single mode defaults to A. "
             "In evaluation mode, runs all scenarios unless specified.",
    )
    parser.add_argument(
        "--cv",
        default="data/sample_cv.txt",
        help="Path to CV text file",
    )
    parser.add_argument(
        "--jd",
        default="data/sample_jd.txt",
        help="Path to job description text file",
    )
    parser.add_argument(
        "--output",
        help="Save full context as JSON to this path",
    )
    parser.add_argument(
        "--memory-dir",
        default=None,
        help="Directory for memory storage (Scenario C). Defaults to data/memory/",
    )
    parser.add_argument(
        "--evaluate",
        metavar="DATASET",
        help="Run evaluation mode: path to a JSON dataset file with CV-JD pairs",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save evaluation results (default: results/)",
    )

    # HuggingFace dataset integration
    parser.add_argument(
        "--hf-dataset",
        metavar="NAME",
        help="Download and evaluate a HuggingFace dataset "
             "(e.g., AzharAli05/Resume-Screening-Dataset)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of pairs to sample from HF dataset (default: 50). "
             "Ignored when --sample-all is set.",
    )
    parser.add_argument(
        "--sample-all",
        action="store_true",
        help="Download the ENTIRE HF dataset (ignores --sample). "
             "Use this once to get a full reference file for train/test splitting.",
    )

    # Evaluation options
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Classification threshold: score >= threshold is 'match' (default: 50)",
    )
    parser.add_argument(
        "--memory-mode",
        choices=["isolated", "shared"],
        default="isolated",
        help="Memory mode for Scenario C: 'isolated' (default, no cross-pair memory) "
             "or 'shared' (experimental, memory persists across pairs)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include embedding-only baseline for comparison (no agents, no LLM)",
    )

    # Tier 2 architecture options
    parser.add_argument(
        "--architecture",
        choices=["tier1", "tier2"],
        default="tier1",
        help="Agent architecture. 'tier1' (default) is the legacy chain "
             "(ExtractionAgent + ContextEnrichmentAgent + reasoning + reflection + decision). "
             "'tier2' uses the new chain (CV/JD profiling + ESCO role context + two-pass "
             "decision with labeled-memory calibration in Scenario C).",
    )
    parser.add_argument(
        "--streaming-memory-mode",
        choices=["cold-start", "continue-stream", "fresh-build"],
        default="cold-start",
        help="Tier 2 streaming protocol mode (Scenario C only). "
             "'cold-start' (default): clear labeled memory at start of each run. "
             "'continue-stream': preserve memory across runs; refuses to run if any input pair_id "
             "already exists in memory (prevents test-set leakage). "
             "'fresh-build': same as cold-start but warns if memory existed.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of worker threads for pair-level parallelism. "
             "1 (default) = sequential. >1 = run that many pairs concurrently. "
             "For Tier 2 Scenario C, labeled-memory commits batch at chunk boundaries "
             "(chunk_size = parallel_workers) to preserve the streaming protocol. "
             "Watch OpenAI rate limits: 8 workers on gpt-4o-mini Tier-1 stays under 200K TPM, "
             "but bigger pools can throttle.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save partial results every N completed pairs. 0 (default) disables "
             "checkpointing. Useful for long local runs (Qwen on Ollama, multi-day "
             "GPU jobs) where a crash would otherwise lose all progress. The same "
             "files are written as the final save (metrics JSON, traces, details), "
             "so the dashboard can load mid-run output. In parallel mode, the "
             "actual cadence rounds up to the next chunk boundary.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run by loading partial results from "
             "{results_dir}/{dataset_stem}.json (and matching _traces.json + "
             "_details.jsonl). Pair_ids already in pair_results are skipped. "
             "Combined with --checkpoint-every for safe long-running jobs. "
             "Note: token-usage totals after resume reflect only the resumed "
             "segment, since the original run's per-pair usage is not stored.",
    )

    # Memory store management (Scenario C)
    parser.add_argument(
        "--memory-stats",
        action="store_true",
        help="Print statistics about the persistent memory store and exit. "
             "Reads from --memory-dir (or default data/memory/).",
    )
    parser.add_argument(
        "--memory-clear",
        action="store_true",
        help="Wipe the persistent memory store and exit. "
             "Prompts for confirmation unless --yes is passed.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (used with --memory-clear).",
    )

    args = parser.parse_args()

    # Memory management commands short-circuit before any evaluation
    if args.memory_stats:
        run_memory_stats(args)
        return
    if args.memory_clear:
        run_memory_clear(args)
        return

    if args.evaluate or args.hf_dataset:
        run_evaluation(args)
    else:
        # Single-run mode: default scenario to A if not specified
        if args.scenario is None:
            args.scenario = "A"
        run_single(args)


if __name__ == "__main__":
    main()
