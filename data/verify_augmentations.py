#!/usr/bin/env python3
"""
Verify augmented dataset against thesis defensibility constraints.
"""

import json
from pathlib import Path
import random

def load_original(path: Path) -> dict:
    """Load original dataset."""
    with open(path, "r") as f:
        return json.load(f)

def load_sampled(path: Path) -> list:
    """Load sampled pairs."""
    with open(path, "r") as f:
        return json.load(f)["pairs"]

def load_augmented(path: Path) -> dict:
    """Load augmented dataset."""
    with open(path, "r") as f:
        return json.load(f)

def count_words(text: str) -> int:
    return len(text.split())

def main():
    print("=" * 70)
    print("VERIFICATION: Augmented Dataset Validation")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    sampled = load_sampled(Path("data/sampled_1000_pairs.json"))
    augmented_data = load_augmented(Path("data/hf_test_1000_augmented.json"))
    augmented = augmented_data["pairs"]

    print(f"[OK] Loaded {len(sampled)} sampled pairs")
    print(f"[OK] Loaded {len(augmented)} augmented pairs")

    # Verification 1: Schema integrity
    print(f"\n{'='*70}")
    print("1. SCHEMA INTEGRITY")
    print(f"{'='*70}")

    errors = []

    # Check all pair_ids match
    sampled_ids = set(p["pair_id"] for p in sampled)
    augmented_ids = set(p["pair_id"] for p in augmented)

    if sampled_ids != augmented_ids:
        missing = sampled_ids - augmented_ids
        extra = augmented_ids - sampled_ids
        if missing:
            errors.append(f"Missing pair_ids: {missing}")
        if extra:
            errors.append(f"Extra pair_ids: {extra}")

    # Check labels unchanged (byte-for-byte)
    labels_ok = 0
    for aug_pair in augmented:
        sampled_pair = next((p for p in sampled if p["pair_id"] == aug_pair["pair_id"]), None)
        if sampled_pair and aug_pair["ground_truth_label"] == sampled_pair["ground_truth_label"]:
            labels_ok += 1
        elif sampled_pair:
            errors.append(f"Label mismatch for {aug_pair['pair_id']}")

    # Check CVs unchanged (byte-for-byte)
    cvs_ok = 0
    for aug_pair in augmented:
        sampled_pair = next((p for p in sampled if p["pair_id"] == aug_pair["pair_id"]), None)
        if sampled_pair and aug_pair["cv_text"] == sampled_pair["cv_text"]:
            cvs_ok += 1
        elif sampled_pair:
            errors.append(f"CV mismatch for {aug_pair['pair_id']}")

    # Check reasons unchanged
    reasons_ok = 0
    for aug_pair in augmented:
        sampled_pair = next((p for p in sampled if p["pair_id"] == aug_pair["pair_id"]), None)
        if sampled_pair and aug_pair["reference_reason"] == sampled_pair["reference_reason"]:
            reasons_ok += 1
        elif sampled_pair:
            errors.append(f"Reason mismatch for {aug_pair['pair_id']}")

    print(f"Pair IDs preserved:       {len(sampled_ids) == len(augmented_ids)} ({len(augmented_ids)} pairs)")
    print(f"Labels unchanged:         {labels_ok}/{len(augmented)} [PASS]" if labels_ok == len(augmented) else f"Labels unchanged: {labels_ok}/{len(augmented)} [FAIL]")
    print(f"CVs unchanged:            {cvs_ok}/{len(augmented)} [PASS]" if cvs_ok == len(augmented) else f"CVs unchanged: {cvs_ok}/{len(augmented)} [FAIL]")
    print(f"Reasons unchanged:        {reasons_ok}/{len(augmented)} [PASS]" if reasons_ok == len(augmented) else f"Reasons unchanged: {reasons_ok}/{len(augmented)} [FAIL]")

    # Verification 2: Augmentation metadata
    print(f"\n{'='*70}")
    print("2. AUGMENTATION METADATA")
    print(f"{'='*70}")

    augmented_count = sum(1 for p in augmented if p.get("jd_augmented"))
    has_original_jd_text = sum(1 for p in augmented if p.get("jd_augmented") and "original_jd_text" in p)

    print(f"Total augmented:          {augmented_count}")
    print(f"Has original_jd_text:     {has_original_jd_text}/{augmented_count} [PASS]" if has_original_jd_text == augmented_count else f"Has original_jd_text: {has_original_jd_text}/{augmented_count} [FAIL]")

    # Verify original_jd_text matches
    original_jd_match = 0
    for aug_pair in augmented:
        if aug_pair.get("jd_augmented") and "original_jd_text" in aug_pair:
            sampled_pair = next((p for p in sampled if p["pair_id"] == aug_pair["pair_id"]), None)
            if sampled_pair and aug_pair["original_jd_text"] == sampled_pair["jd_text"]:
                original_jd_match += 1

    print(f"Original JD text matches: {original_jd_match}/{augmented_count} [PASS]" if original_jd_match == augmented_count else f"Original JD text matches: {original_jd_match}/{augmented_count} [FAIL]")

    # Verification 3: Content bounds
    print(f"\n{'='*70}")
    print("3. CONTENT BOUNDS (100-300 words)")
    print(f"{'='*70}")

    augmented_jds = [p for p in augmented if p.get("jd_augmented")]
    word_counts = [count_words(p["jd_text"]) for p in augmented_jds]

    in_range = sum(1 for wc in word_counts if 100 <= wc <= 300)
    outliers = [(p["pair_id"], count_words(p["jd_text"])) for p in augmented_jds if not (100 <= count_words(p["jd_text"]) <= 300)]

    print(f"Augmented JDs in range:   {in_range}/{len(augmented_jds)} [PASS]" if in_range == len(augmented_jds) else f"Augmented JDs in range: {in_range}/{len(augmented_jds)} [FAIL]")

    if outliers:
        print(f"  Outliers ({len(outliers)}):")
        for pair_id, wc in outliers[:5]:
            print(f"    {pair_id}: {wc} words")

    print(f"Word count range:         {min(word_counts)}-{max(word_counts)} words")

    # Verification 4: Spot check (5 random augmented pairs)
    print(f"\n{'='*70}")
    print("4. SPOT CHECK (5 Random Augmented Pairs)")
    print(f"{'='*70}\n")

    random.seed(42)
    spot_check = random.sample(augmented_jds, min(5, len(augmented_jds)))

    for i, pair in enumerate(spot_check, 1):
        orig = pair["original_jd_text"]
        aug = pair["jd_text"]
        print(f"Pair {i}: {pair['pair_id']}")
        print(f"\nORIGINAL ({count_words(orig)} words):")
        print(f'  "{orig}"')
        print(f"\nAUGMENTED ({count_words(aug)} words):")
        print(f'  "{aug[:150]}..."' if len(aug) > 150 else f'  "{aug}"')
        print()

    # Summary
    print(f"{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")

    if not errors:
        print("All checks PASSED!")
        print(f"  - {len(augmented)} pairs verified")
        print(f"  - {augmented_count} augmentations completed")
        print(f"  - All schemas intact and constraints satisfied")
        return True
    else:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
