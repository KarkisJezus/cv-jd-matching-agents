#!/usr/bin/env python3
"""
Augment short JDs using direct Claude API calls.
This script loads sampled pairs and generates augmentations via Claude.
"""

import json
import sys
from pathlib import Path
from typing import Tuple
import anthropic

WORD_COUNT_THRESHOLD = 30
HAIKU_MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are an expert technical recruiter. Your task is to expand a short job description into a realistic, professional 2-3 paragraph job posting (150-250 words).

CRITICAL RULES:
1. Preserve the EXACT role title and any domain mentions from the original JD
2. Do NOT invent company names, specific dates, salary ranges, or recruiting platitudes
3. Focus on realistic job responsibilities, required skills, and why someone would want the role
4. Write naturally—this is a real job posting, not a template
5. If the original JD mentions unusual role/domain combinations (e.g., "Game Developer for data science"), PRESERVE that signal

Output ONLY the expanded JD text. No preamble, no markdown, no explanation."""

def count_words(text: str) -> int:
    return len(text.split())

def load_sampled_pairs() -> list:
    """Load sampled pairs from prepare_augmentation output."""
    with open("data/sampled_1000_pairs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pairs"]

def augment_jd(client: anthropic.Anthropic, jd_text: str) -> Tuple[str, int, int]:
    """Call Claude Haiku to augment a JD. Returns: (augmented_text, input_tokens, output_tokens)"""
    prompt = f"""Original JD (short): {jd_text}

Expand this into a realistic 2-3 paragraph job description (150-250 words). Remember: preserve the exact role and domain, no invented details."""

    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    augmented_text = message.content[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    return augmented_text, input_tokens, output_tokens

def augment_batch(client: anthropic.Anthropic, pairs: list, batch_size: int = 10) -> dict:
    """Augment a batch of pairs."""
    augmented_pairs = []
    total_input_tokens = 0
    total_output_tokens = 0
    augmentation_count = 0

    short_pairs = [p for p in pairs if count_words(p["jd_text"]) < WORD_COUNT_THRESHOLD]
    print(f"Processing {batch_size} pairs (out of {len(short_pairs)} short JDs)...\n")

    for i, pair in enumerate(short_pairs[:batch_size]):
        jd_word_count = count_words(pair["jd_text"])
        print(f"[{i+1}/{batch_size}] Augmenting pair_id={pair['pair_id']} ({jd_word_count} words)...", flush=True)

        try:
            augmented_jd, in_tokens, out_tokens = augment_jd(client, pair["jd_text"])
            total_input_tokens += in_tokens
            total_output_tokens += out_tokens
            augmentation_count += 1

            augmented_word_count = count_words(augmented_jd)
            cost = (in_tokens / 1e6) * 0.08 + (out_tokens / 1e6) * 0.24

            augmented_pair = {
                **pair,
                "jd_text": augmented_jd,
                "jd_augmented": True,
                "original_jd_text": pair["jd_text"]
            }
            augmented_pairs.append(augmented_pair)

            print(f"  --> Augmented ({augmented_word_count} words, ${cost:.4f})\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            raise

    # Cost summary
    cost_total = (total_input_tokens / 1e6) * 0.08 + (total_output_tokens / 1e6) * 0.24

    return {
        "augmented_pairs": augmented_pairs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "augmentation_count": augmentation_count,
        "total_cost": cost_total
    }

def main():
    print("=" * 70)
    print("DRY RUN: Augment 10 short JDs with Claude Haiku 4.5")
    print("=" * 70)

    # Initialize client (will use ANTHROPIC_API_KEY from environment)
    try:
        client = anthropic.Anthropic()
    except Exception as e:
        print(f"\nERROR: Could not initialize Anthropic client.")
        print(f"Make sure ANTHROPIC_API_KEY is set in environment.")
        print(f"Details: {e}")
        sys.exit(1)

    # Load pairs
    print("\nLoading sampled pairs...")
    pairs = load_sampled_pairs()
    short_pairs = [p for p in pairs if count_words(p["jd_text"]) < WORD_COUNT_THRESHOLD]
    print(f"[OK] Loaded {len(pairs)} pairs ({len(short_pairs)} short)")

    # Augment batch
    print(f"\n{'=' * 70}")
    result = augment_batch(client, pairs, batch_size=10)
    augmented_pairs = result["augmented_pairs"]

    # Print results
    print(f"{'=' * 70}")
    print("\nRESULTS:\n")
    print(f"Augmented:        {result['augmentation_count']} pairs")
    print(f"Input tokens:     {result['total_input_tokens']:,}")
    print(f"Output tokens:    {result['total_output_tokens']:,}")
    print(f"Total cost:       ${result['total_cost']:.2f}")

    # Show 3 before/after examples
    print(f"\n{'=' * 70}")
    print("BEFORE/AFTER EXAMPLES:\n")

    for i, aug_pair in enumerate(augmented_pairs[:3], 1):
        orig_jd = aug_pair["original_jd_text"]
        new_jd = aug_pair["jd_text"]
        print(f"--- Example {i}: {aug_pair['pair_id']} ---")
        print(f"\nORIGINAL JD ({count_words(orig_jd)} words):")
        print(f'"{orig_jd}"')
        print(f"\nAUGMENTED JD ({count_words(new_jd)} words):")
        print(f'"{new_jd}"')
        print()

    print(f"{'=' * 70}")
    print("\nDRY RUN COMPLETE")
    print(f"\nTo proceed with full augmentation of all 905 pairs:")
    print(f"  python data/augment_all.py")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
