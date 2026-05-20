"""
audit.py — run the LLM-as-Judge auditor over a CV-JD dataset.

The judge is independent of the matching system: it receives only CV + JD +
source label, and produces an independent verdict on whether the source label
is correct.

Default judge provider: DeepSeek (cross-family from gpt-4o-mini, cheap).
Switch to Gemini by setting:
    JUDGE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    JUDGE_MODEL=gemini-2.5-pro
    JUDGE_API_KEY=<your gemini key>

Usage:
    python audit.py --evaluate data/hf_test_150.json --workers 8
    python audit.py --evaluate data/hf_full_test.json --workers 8 --output results/judge_full_test.json
    python audit.py --evaluate data/hf_test_150.json --limit 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

# Make package imports work whether run as `python audit.py` or `python -m audit`
sys.path.insert(0, str(Path(__file__).parent))

from agents.judge import JudgeAgent, JudgeResult, get_prompt_version_hash
from config.settings import settings
from llm.client import LLMClient


# Per-1M-token costs for known judge providers, used to estimate run cost.
# Update when prices change.
JUDGE_PRICING = {
    "deepseek-chat":      {"in": 0.28, "out": 1.10},
    "deepseek-reasoner":  {"in": 0.55, "out": 2.19},
    "gemini-2.5-pro":     {"in": 1.25, "out": 10.00},
    "gemini-2.5-flash":   {"in": 0.10, "out": 0.40},
    "gpt-4o":             {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":        {"in": 0.15, "out": 0.60},
}


def make_judge_client() -> LLMClient:
    """Build the judge LLM client from environment settings.

    Reuses the existing LLMClient (OpenAI-compatible) — both DeepSeek and
    Gemini expose OpenAI-compatible endpoints, so no provider-specific code.
    """
    if not settings.judge_api_key:
        raise SystemExit(
            "JUDGE_API_KEY is not set in .env.\n"
            "For DeepSeek (default): get a key at platform.deepseek.com and add\n"
            "    JUDGE_API_KEY=<your-deepseek-key>\n"
            "to .env. For Gemini, also set JUDGE_BASE_URL and JUDGE_MODEL."
        )
    return LLMClient(
        api_key=settings.judge_api_key,
        model=settings.judge_model,
        temperature=settings.judge_temperature,
        max_tokens=settings.judge_max_tokens,
        base_url=settings.judge_base_url,
    )


def make_judge_client_per_thread() -> LLMClient:
    """Per-thread client. Equivalent to make_judge_client but documents intent.

    Each worker thread gets its own client so token counters don't race —
    same pattern as the runner's _get_fresh_llm.
    """
    return LLMClient(
        api_key=settings.judge_api_key,
        model=settings.judge_model,
        temperature=settings.judge_temperature,
        max_tokens=settings.judge_max_tokens,
        base_url=settings.judge_base_url,
    )


def _judge_one_pair(pair: dict, temperature_override: float | None = None) -> dict:
    """Worker function for ThreadPoolExecutor. Each call gets a fresh client.

    `temperature_override` lets the self-consistency check re-run a sample at
    a different temperature without disturbing the main run's deterministic
    settings.
    """
    if temperature_override is None:
        client = make_judge_client_per_thread()
    else:
        client = LLMClient(
            api_key=settings.judge_api_key,
            model=settings.judge_model,
            temperature=temperature_override,
            max_tokens=settings.judge_max_tokens,
            base_url=settings.judge_base_url,
        )
    judge = JudgeAgent(client)
    result = judge.judge(
        pair_id=pair["pair_id"],
        cv_text=pair.get("cv_text", ""),
        jd_text=pair.get("jd_text", ""),
        source_label=pair.get("ground_truth_label"),
    )
    return asdict(result)


def _aggregate(judgments: list[dict]) -> dict:
    """Compute headline stats from a list of judgment dicts."""
    n = len(judgments)
    if n == 0:
        return {}
    n_correct = sum(1 for j in judgments if j["source_assessment"] == "correct")
    n_incorrect = sum(1 for j in judgments if j["source_assessment"] == "incorrect")
    n_ambiguous = sum(1 for j in judgments if j["source_assessment"] == "ambiguous")
    n_errors = sum(1 for j in judgments if j.get("error"))
    confidences = [j["judge_confidence"] for j in judgments if not j.get("error")]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    total_in = sum(j.get("tokens_in", 0) for j in judgments)
    total_out = sum(j.get("tokens_out", 0) for j in judgments)
    return {
        "n_pairs": n,
        "n_source_correct": n_correct,
        "n_source_incorrect": n_incorrect,
        "n_ambiguous": n_ambiguous,
        "n_errors": n_errors,
        "fraction_source_correct": n_correct / n,
        "fraction_source_incorrect": n_incorrect / n,
        "fraction_ambiguous": n_ambiguous / n,
        "mean_confidence": mean_conf,
        "tokens_in_total": total_in,
        "tokens_out_total": total_out,
    }


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = JUDGE_PRICING.get(model)
    if not pricing:
        return 0.0
    return (tokens_in / 1_000_000) * pricing["in"] + (tokens_out / 1_000_000) * pricing["out"]


def _hash_file(path: Path) -> str:
    """SHA256 of the file contents, truncated to 12 hex chars."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _self_consistency_check(
    pairs: list[dict],
    deterministic_judgments: dict[str, dict],
    sample_size: int,
    workers: int,
    temperature: float = 0.5,
) -> dict:
    """Re-run judge on a random sample of pairs at higher temperature, compare verdicts.

    Returns a dict with agreement statistics. The deterministic verdict is
    treated as the reference; the higher-temperature re-run is the variance test.
    """
    import random
    if not pairs:
        return {"sample_size": 0, "agreement_rate": None}
    rng = random.Random(42)
    sample = rng.sample(pairs, min(sample_size, len(pairs)))
    sample_ids = [p["pair_id"] for p in sample]

    print(f"\n=== Self-consistency check: re-running {len(sample)} pairs at "
          f"temperature={temperature} ===")
    rerun: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pair = {
            executor.submit(_judge_one_pair, p, temperature): p for p in sample
        }
        for i, future in enumerate(as_completed(future_to_pair), 1):
            pair = future_to_pair[future]
            try:
                judgment = future.result()
            except Exception as e:
                judgment = {"raw_verdict": "?", "error": str(e), "pair_id": pair["pair_id"]}
            rerun[pair["pair_id"]] = judgment
            print(f"  [{i}/{len(sample)}] {pair['pair_id']}: rerun verdict={judgment.get('raw_verdict','?')}")

    # Compute verdict agreement
    verdict_agree = source_assess_agree = 0
    valid = 0
    for pid in sample_ids:
        det = deterministic_judgments.get(pid)
        rerun_j = rerun.get(pid)
        if not det or not rerun_j or rerun_j.get("error") or det.get("error"):
            continue
        valid += 1
        if det.get("raw_verdict") == rerun_j.get("raw_verdict"):
            verdict_agree += 1
        if det.get("source_assessment") == rerun_j.get("source_assessment"):
            source_assess_agree += 1

    return {
        "sample_size": valid,
        "temperature_rerun": temperature,
        "verdict_agreement_rate": verdict_agree / valid if valid else None,
        "source_assessment_agreement_rate": source_assess_agree / valid if valid else None,
        "sample_pair_ids": sample_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LLM-as-Judge auditor.")
    parser.add_argument("--evaluate", required=True, help="Path to dataset JSON file.")
    parser.add_argument("--output", default=None, help="Output path; defaults to results/judge_<dataset>.json.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent judge calls.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N pairs (for smoke tests).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume by skipping pair_ids already in --output (idempotent reruns).")
    parser.add_argument("--self-consistency-sample", type=int, default=0,
                        help="After the main run, re-run N random pairs at higher temperature "
                             "to measure verdict stability. 0 disables (default).")
    args = parser.parse_args()

    dataset_path = Path(args.evaluate)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    pairs = raw if isinstance(raw, list) else raw.get("pairs", [])
    if args.limit:
        pairs = pairs[:args.limit]

    output_path = Path(args.output) if args.output else (
        Path("results") / f"judge_{dataset_path.stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resumability: load existing judgments, skip those already done
    existing: dict[str, dict] = {}
    if args.resume and output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        existing = prior.get("judgments", {})
        pairs = [p for p in pairs if p["pair_id"] not in existing]
        print(f"Resume: skipping {len(existing)} already-judged pairs; "
              f"{len(pairs)} remain.")

    if not pairs:
        print("No pairs to judge (everything already done?).")
        return 0

    # Verify auth + connectivity with one synchronous call before parallelism
    print(f"Judge config: provider={settings.judge_base_url}, model={settings.judge_model}, "
          f"workers={args.workers}")
    print(f"Dataset: {dataset_path.name} ({len(pairs)} pairs to judge)")

    start_time = time.time()
    judgments: dict[str, dict] = dict(existing)  # carry forward resume state
    completed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pair = {executor.submit(_judge_one_pair, p): p for p in pairs}
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                judgment = future.result()
            except Exception as e:
                judgment = {
                    "pair_id": pair["pair_id"],
                    "judge_label": None,
                    "judge_confidence": 0.0,
                    "source_assessment": "ambiguous",
                    "rationale": f"Worker exception: {type(e).__name__}: {e}",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
            judgments[pair["pair_id"]] = judgment
            completed += 1

            verdict = judgment.get("raw_verdict") or "?"
            conf = judgment.get("judge_confidence", 0)
            assessment = judgment.get("source_assessment", "?")
            err = " ERROR" if judgment.get("error") else ""
            print(f"  [{completed}/{len(pairs)}] {pair['pair_id']}: "
                  f"verdict={verdict} conf={conf:.2f} source={assessment}{err}")

            # Periodic checkpoint every 25 pairs so a crash is recoverable
            if completed % 25 == 0:
                _save(output_path, dataset_path, judgments)

    # Self-consistency check (if requested)
    self_consistency = None
    if args.self_consistency_sample and args.self_consistency_sample > 0:
        new_pair_data = [p for p in pairs if p["pair_id"] in judgments]
        # if --resume picked up no new pairs, fall back to all available pairs in the dataset
        all_pairs_in_dataset = (
            raw if isinstance(raw, list) else raw.get("pairs", [])
        )
        if args.limit:
            all_pairs_in_dataset = all_pairs_in_dataset[:args.limit]
        sample_pool = new_pair_data or all_pairs_in_dataset
        self_consistency = _self_consistency_check(
            sample_pool, judgments,
            sample_size=args.self_consistency_sample,
            workers=args.workers,
        )

    elapsed = time.time() - start_time
    summary = _aggregate(list(judgments.values()))
    cost = _estimate_cost(
        settings.judge_model, summary.get("tokens_in_total", 0),
        summary.get("tokens_out_total", 0),
    )

    print()
    print(f"Done in {elapsed/60:.1f} min.")
    print(f"Pairs judged: {len(judgments)}")
    print(f"  Source correct:   {summary.get('n_source_correct', 0)} ({summary.get('fraction_source_correct', 0):.1%})")
    print(f"  Source incorrect: {summary.get('n_source_incorrect', 0)} ({summary.get('fraction_source_incorrect', 0):.1%})")
    print(f"  Ambiguous:        {summary.get('n_ambiguous', 0)} ({summary.get('fraction_ambiguous', 0):.1%})")
    print(f"  Errors:           {summary.get('n_errors', 0)}")
    print(f"  Mean confidence:  {summary.get('mean_confidence', 0):.2f}")
    print(f"  Tokens: in={summary.get('tokens_in_total', 0):,}, "
          f"out={summary.get('tokens_out_total', 0):,}")
    print(f"  Estimated cost:   ${cost:.4f}")
    if self_consistency:
        print(f"  Self-consistency: verdict={self_consistency.get('verdict_agreement_rate'):.0%} "
              f"source_assess={self_consistency.get('source_assessment_agreement_rate'):.0%} "
              f"(n={self_consistency.get('sample_size')})")

    _save(
        output_path, dataset_path, judgments,
        summary=summary, elapsed=elapsed, cost=cost,
        self_consistency=self_consistency,
    )
    print(f"Saved: {output_path}")
    return 0


def _save(
    path: Path, dataset_path: Path, judgments: dict[str, dict],
    summary: dict | None = None, elapsed: float | None = None,
    cost: float | None = None, self_consistency: dict | None = None,
) -> None:
    """Write the audit results JSON with full reproducibility metadata."""
    from datetime import datetime
    dataset_hash = _hash_file(dataset_path) if dataset_path.exists() else "unknown"
    payload = {
        "_meta": {
            # Reproducibility — examiners can reconstruct exactly what produced this file
            "schema_version": 2,                       # bump if JudgeResult fields change
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "judge_model": settings.judge_model,
            "judge_base_url": settings.judge_base_url,
            "judge_temperature": settings.judge_temperature,
            "judge_max_tokens": settings.judge_max_tokens,
            "judge_prompt_version": get_prompt_version_hash(),
            "dataset": str(dataset_path),
            "dataset_sha256_short": dataset_hash,
            "ambiguous_confidence_threshold": 0.7,     # from agents/judge.py
            "blind": True,                             # judge never saw system predictions
            "n_pairs": len(judgments),
            "duration_s": elapsed,
            "cost_usd_estimated": cost,
            "methodology_reference": "Gu et al. 2025, A Survey on LLM-as-a-Judge (arXiv:2411.15594)",
            **(summary or {}),
            "self_consistency": self_consistency,
        },
        "judgments": judgments,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
