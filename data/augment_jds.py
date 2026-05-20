#!/usr/bin/env python3
"""
Augment short JDs in the CV-JD dataset using Claude Haiku 4.5.

This script:
1. Loads hf_test_rest.json
2. Samples 1000 pairs with seed=42, preserving ~50/50 select/reject balance
3. Identifies JDs < 30 words (blind augmentation candidates)
4. Calls Claude Haiku 4.5 to expand each short JD
5. Saves output to hf_test_1000_augmented.json with metadata

BLIND PROMPT CONSTRAINT: The LLM sees ONLY jd_text, never cv_text/label/reason.
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import Optional
from anthropic import Anthropic

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SEED = 42
SAMPLE_SIZE = 1000
CHECKPOINT_PATH = Path("data/augment_jds_checkpoint.json")
OUTPUT_PATH = Path("data/hf_test_1000_augmented.json")
SOURCE_PATH = Path("data/hf_test_rest.json")

HAIKU_MODEL = "claude-haiku-4-5-20251001"
WORD_COUNT_THRESHOLD = 30

SYSTEM_PROMPT = """You are an expert technical recruiter. Your task is to expand a short job description into a realistic, professional 2-3 paragraph job posting (150-250 words).

CRITICAL RULES:
1. Preserve the EXACT role title and any domain mentions from the original JD
2. Do NOT invent company names, specific dates, salary ranges, or recruiting platitudes
3. Focus on realistic job responsibilities, required skills, and why someone would want the role
4. Write naturally—this is a real job posting, not a template
5. If the original JD mentions unusual role/domain combinations (e.g., "Game Developer for data science"), PRESERVE that signal

Output ONLY the expanded JD text. No preamble, no markdown, no explanation."""

AUGMENTATION_PROMPT_TEMPLATE = """Original JD (short): {jd_text}

Expand this into a realistic 2-3 paragraph job description (150-250 words). Remember: preserve the exact role and domain, no invented details."""

# ─────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def load_data(path: Path) -> dict:
    """Load JSON dataset."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(path: Path, data: dict):
    """Save JSON dataset."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_checkpoint() -> Optional[dict]:
    """Load checkpoint if it exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint: dict):
    """Save checkpoint."""
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

def augment_jd_with_claude(client: Anthropic, jd_text: str) -> tuple[str, int, int]:
    """
    Call Claude Haiku to augment a short JD.
    Returns: (augmented_text, input_tokens, output_tokens)
    """
    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": AUGMENTATION_PROMPT_TEMPLATE.format(jd_text=jd_text)
            }
        ]
    )

    augmented_text = message.content[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    return augmented_text, input_tokens, output_tokens

# ─────────────────────────────────────────────────────────────
# Main Augmentation Logic
# ─────────────────────────────────────────────────────────────

def sample_dataset(data: dict, size: int, seed: int) -> list:
    """
    Sample pairs with fixed seed, preserving ~50/50 select/reject balance.
    """
    random.seed(seed)
    pairs = data["pairs"]

    # Separate by label
    selected = [p for p in pairs if p["ground_truth_label"]]
    rejected = [p for p in pairs if not p["ground_truth_label"]]

    # Sample proportionally to maintain balance
    n_selected = int(size * len(selected) / len(pairs))
    n_rejected = size - n_selected

    sampled_selected = random.sample(selected, min(n_selected, len(selected)))
    sampled_rejected = random.sample(rejected, min(n_rejected, len(rejected)))

    sampled = sampled_selected + sampled_rejected
    random.shuffle(sampled)

    return sampled

def augment_batch(
    client: Anthropic,
    sampled_pairs: list,
    start_idx: int = 0,
    checkpoint_interval: int = 50
) -> dict:
    """
    Augment all short JDs in sampled pairs.
    Saves checkpoint every checkpoint_interval pairs.
    """
    augmented_pairs = []
    total_input_tokens = 0
    total_output_tokens = 0
    augmentation_count = 0

    for i, pair in enumerate(sampled_pairs[start_idx:], start=start_idx):
        pair_idx = start_idx + i
        jd_word_count = count_words(pair["jd_text"])

        if jd_word_count < WORD_COUNT_THRESHOLD:
            # Augment this JD
            print(f"[{pair_idx + 1}/{len(sampled_pairs)}] Augmenting (JD: {jd_word_count} words)...", end=" ", flush=True)

            try:
                augmented_jd, in_tokens, out_tokens = augment_jd_with_claude(client, pair["jd_text"])
                total_input_tokens += in_tokens
                total_output_tokens += out_tokens
                augmentation_count += 1

                augmented_word_count = count_words(augmented_jd)
                cost = (in_tokens / 1e6) * 0.08 + (out_tokens / 1e6) * 0.24  # Haiku pricing (approx)

                augmented_pair = {
                    **pair,
                    "jd_text": augmented_jd,
                    "jd_augmented": True,
                    "original_jd_text": pair["jd_text"]
                }
                augmented_pairs.append(augmented_pair)

                print(f"✓ ({augmented_word_count} words, ${cost:.4f})")
            except Exception as e:
                print(f"✗ Error: {e}")
                raise
        else:
            # Keep JD as-is
            augmented_pair = {
                **pair,
                "jd_augmented": False
            }
            augmented_pairs.append(augmented_pair)
            print(f"[{pair_idx + 1}/{len(sampled_pairs)}] Skipping (JD: {jd_word_count} words)")

        # Save checkpoint
        if (pair_idx + 1) % checkpoint_interval == 0:
            checkpoint = {
                "completed_pairs": pair_idx + 1,
                "total_pairs": len(sampled_pairs),
                "augmented_pairs": augmented_pairs,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "augmentation_count": augmentation_count
            }
            save_checkpoint(checkpoint)
            print(f"\n[CHECKPOINT] Saved progress at {pair_idx + 1} pairs")
            print(f"  Augmentations so far: {augmentation_count}")
            print(f"  Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out")
            cost_so_far = (total_input_tokens / 1e6) * 0.08 + (total_output_tokens / 1e6) * 0.24
            print(f"  Cost so far: ${cost_so_far:.2f}\n")

    return {
        "augmented_pairs": augmented_pairs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "augmentation_count": augmentation_count
    }

def main():
    """Main entry point."""
    print("=" * 70)
    print("JD AUGMENTATION PIPELINE - Claude Haiku 4.5")
    print("=" * 70)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not found in environment.")
        print("Please set it in .env or as an environment variable.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Load source data
    print("\n[1] Loading source data...")
    data = load_data(SOURCE_PATH)
    print(f"  ✓ Loaded {len(data['pairs'])} pairs from {SOURCE_PATH}")

    # Check balance
    selected = sum(1 for p in data["pairs"] if p["ground_truth_label"])
    rejected = len(data["pairs"]) - selected
    print(f"  ✓ Balance: {selected} selected, {rejected} rejected")

    # Sample with seed=42
    print(f"\n[2] Sampling {SAMPLE_SIZE} pairs (seed={SEED})...")
    sampled_pairs = sample_dataset(data, SAMPLE_SIZE, SEED)

    sampled_selected = sum(1 for p in sampled_pairs if p["ground_truth_label"])
    sampled_rejected = len(sampled_pairs) - sampled_selected
    print(f"  ✓ Sampled {SAMPLE_SIZE} pairs")
    print(f"  ✓ Balance: {sampled_selected} selected, {sampled_rejected} rejected")

    # Analyze word counts
    print(f"\n[3] Analyzing JD word counts...")
    word_counts = [count_words(p["jd_text"]) for p in sampled_pairs]
    short_jds = sum(1 for w in word_counts if w < WORD_COUNT_THRESHOLD)
    print(f"  ✓ JDs < {WORD_COUNT_THRESHOLD} words: {short_jds} ({100*short_jds/len(word_counts):.1f}%)")
    print(f"  ✓ Word count range: {min(word_counts)} - {max(word_counts)}")

    # Check checkpoint
    checkpoint = load_checkpoint()
    start_idx = 0
    if checkpoint:
        start_idx = checkpoint["completed_pairs"]
        print(f"\n[4] Found checkpoint at {start_idx} pairs. Resuming...")
        augmented_pairs = checkpoint["augmented_pairs"]
        total_input_tokens = checkpoint["total_input_tokens"]
        total_output_tokens = checkpoint["total_output_tokens"]
        augmentation_count = checkpoint["augmentation_count"]
    else:
        print(f"\n[4] Starting augmentation from the beginning...")
        augmented_pairs = []
        total_input_tokens = 0
        total_output_tokens = 0
        augmentation_count = 0

    # Run augmentation
    print(f"\n[5] Augmenting JDs (starting from pair {start_idx + 1})...\n")

    try:
        result = augment_batch(client, sampled_pairs, start_idx=start_idx)
        augmented_pairs = result["augmented_pairs"]
        total_input_tokens += result["total_input_tokens"]
        total_output_tokens += result["total_output_tokens"]
        augmentation_count += result["augmentation_count"]
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Checkpoint saved.")
        sys.exit(0)

    # Save output
    print(f"\n[6] Saving augmented dataset...")
    output_data = {
        "metadata": {
            "source": "hf_test_rest.json",
            "sample_size": len(augmented_pairs),
            "augmentations_count": augmentation_count,
            "augmentation_model": HAIKU_MODEL,
            "word_count_threshold": WORD_COUNT_THRESHOLD,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "seed": SEED
        },
        "pairs": augmented_pairs
    }

    save_data(OUTPUT_PATH, output_data)
    print(f"  ✓ Saved {len(augmented_pairs)} pairs to {OUTPUT_PATH}")

    # Summary
    cost_total = (total_input_tokens / 1e6) * 0.08 + (total_output_tokens / 1e6) * 0.24
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total pairs augmented: {augmentation_count}")
    print(f"Total input tokens:   {total_input_tokens:,}")
    print(f"Total output tokens:  {total_output_tokens:,}")
    print(f"Total cost:           ${cost_total:.2f}")
    print(f"Output file:          {OUTPUT_PATH}")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
