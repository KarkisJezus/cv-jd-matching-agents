#!/usr/bin/env python3
"""
Prepare data for augmentation: load, sample, and identify short JDs.
"""

import json
import random
from pathlib import Path
from typing import List

SEED = 42
SAMPLE_SIZE = 1000
WORD_COUNT_THRESHOLD = 30

def count_words(text: str) -> int:
    return len(text.split())

def load_data(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sample_dataset(data: dict, size: int, seed: int) -> list:
    """Sample pairs with fixed seed, preserving ~50/50 select/reject balance."""
    random.seed(seed)
    pairs = data["pairs"]

    selected = [p for p in pairs if p["ground_truth_label"]]
    rejected = [p for p in pairs if not p["ground_truth_label"]]

    n_selected = int(size * len(selected) / len(pairs))
    n_rejected = size - n_selected

    sampled_selected = random.sample(selected, min(n_selected, len(selected)))
    sampled_rejected = random.sample(rejected, min(n_rejected, len(rejected)))

    sampled = sampled_selected + sampled_rejected
    random.shuffle(sampled)

    return sampled

def main():
    print("=" * 70)
    print("PREPARATION: Data Loading & Sampling")
    print("=" * 70)

    # Load
    print("\nLoading data...")
    data = load_data(Path("data/hf_test_rest.json"))
    print(f"[OK] Loaded {len(data['pairs'])} pairs")

    # Check balance
    selected = sum(1 for p in data["pairs"] if p["ground_truth_label"])
    rejected = len(data["pairs"]) - selected
    print(f"[OK] Balance: {selected} selected, {rejected} rejected")

    # Sample
    print(f"\nSampling {SAMPLE_SIZE} pairs (seed={SEED})...")
    sampled = sample_dataset(data, SAMPLE_SIZE, SEED)
    sampled_selected = sum(1 for p in sampled if p["ground_truth_label"])
    sampled_rejected = len(sampled) - sampled_selected
    print(f"[OK] Sampled {SAMPLE_SIZE} pairs")
    print(f"[OK] Balance: {sampled_selected} selected, {sampled_rejected} rejected")

    # Word count analysis
    print(f"\nAnalyzing JD word counts...")
    word_counts = [count_words(p["jd_text"]) for p in sampled]
    short_jds = [p for p in sampled if count_words(p["jd_text"]) < WORD_COUNT_THRESHOLD]
    print(f"[OK] Short JDs (< {WORD_COUNT_THRESHOLD} words): {len(short_jds)} ({100*len(short_jds)/len(sampled):.1f}%)")
    print(f"[OK] Word count range: {min(word_counts)} - {max(word_counts)}")

    # Save sampled data
    output_file = Path("data/sampled_1000_pairs.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"pairs": sampled, "short_jds_count": len(short_jds)}, f, indent=2)
    print(f"\n[OK] Saved sampled pairs to {output_file}")

    print(f"\n{'=' * 70}\n")

    # Print first 5 short JDs as examples
    print("Example short JDs to augment:\n")
    for i, pair in enumerate(short_jds[:5]):
        print(f"{i+1}. pair_id: {pair['pair_id']}")
        print(f"   original JD ({count_words(pair['jd_text'])} words):")
        print(f"   \"{pair['jd_text']}\"")
        print()

if __name__ == "__main__":
    main()
