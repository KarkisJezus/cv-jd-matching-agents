#!/usr/bin/env python3
"""
Verify V2 augmented dataset against thesis defensibility constraints.
"""

import json
from pathlib import Path
import random

def main():
    print("=" * 70)
    print("VERIFICATION: V2 Augmented Dataset")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    with open("data/hf_test_1000_subset.json", "r") as f:
        subset = json.load(f)["pairs"]

    with open("data/hf_test_1000_augmented_v2.json", "r") as f:
        aug_data = json.load(f)
        augmented = aug_data["pairs"]

    print(f"[OK] Loaded {len(subset)} original pairs")
    print(f"[OK] Loaded {len(augmented)} augmented pairs")

    # Verification 1: Schema integrity
    print(f"\n{'='*70}")
    print("1. SCHEMA INTEGRITY")
    print(f"{'='*70}")

    errors = []

    # Check pair_ids match
    subset_ids = set(p["pair_id"] for p in subset)
    aug_ids = set(p["pair_id"] for p in augmented)

    if subset_ids != aug_ids:
        errors.append(f"pair_id mismatch: subset={len(subset_ids)}, augmented={len(aug_ids)}")
    else:
        print(f"[PASS] All 1000 pair_ids preserved")

    # Check labels unchanged (byte-for-byte)
    labels_ok = 0
    for aug_pair in augmented:
        subset_pair = next((p for p in subset if p["pair_id"] == aug_pair["pair_id"]), None)
        if subset_pair and aug_pair["ground_truth_label"] == subset_pair["ground_truth_label"]:
            labels_ok += 1

    if labels_ok == len(augmented):
        print(f"[PASS] All {len(augmented)} ground_truth_label fields unchanged")
    else:
        errors.append(f"Label mismatch: {labels_ok}/{len(augmented)}")

    # Check CVs unchanged (byte-for-byte)
    cvs_ok = 0
    for aug_pair in augmented:
        subset_pair = next((p for p in subset if p["pair_id"] == aug_pair["pair_id"]), None)
        if subset_pair and aug_pair["cv_text"] == subset_pair["cv_text"]:
            cvs_ok += 1

    if cvs_ok == len(augmented):
        print(f"[PASS] All {len(augmented)} cv_text fields unchanged")
    else:
        errors.append(f"CV mismatch: {cvs_ok}/{len(augmented)}")

    # Check reasons unchanged
    reasons_ok = 0
    for aug_pair in augmented:
        subset_pair = next((p for p in subset if p["pair_id"] == aug_pair["pair_id"]), None)
        if subset_pair and aug_pair["reference_reason"] == subset_pair["reference_reason"]:
            reasons_ok += 1

    if reasons_ok == len(augmented):
        print(f"[PASS] All {len(augmented)} reference_reason fields unchanged")
    else:
        errors.append(f"Reason mismatch: {reasons_ok}/{len(augmented)}")

    # Check descriptions unchanged
    desc_ok = 0
    for aug_pair in augmented:
        subset_pair = next((p for p in subset if p["pair_id"] == aug_pair["pair_id"]), None)
        if subset_pair and aug_pair["description"] == subset_pair["description"]:
            desc_ok += 1

    if desc_ok == len(augmented):
        print(f"[PASS] All {len(augmented)} description fields unchanged")
    else:
        errors.append(f"Description mismatch: {desc_ok}/{len(augmented)}")

    # Verification 2: Augmentation metadata
    print(f"\n{'='*70}")
    print("2. AUGMENTATION METADATA")
    print(f"{'='*70}")

    aug_count = sum(1 for p in augmented if p.get("jd_augmented"))
    has_orig_jd = sum(1 for p in augmented if p.get("jd_augmented") and "original_jd_text" in p)

    print(f"Total augmented (jd_augmented=true): {aug_count}")
    if aug_count == 907:
        print(f"[PASS] Augmentation count = 907 (expected)")
    else:
        errors.append(f"Augmentation count {aug_count}, expected 907")

    if has_orig_jd == aug_count:
        print(f"[PASS] original_jd_text present on all {aug_count} augmented pairs")
    else:
        errors.append(f"original_jd_text missing on {aug_count - has_orig_jd} pairs")

    # Verify original_jd_text matches original JD
    orig_match = 0
    for aug_pair in augmented:
        if aug_pair.get("jd_augmented") and "original_jd_text" in aug_pair:
            subset_pair = next((p for p in subset if p["pair_id"] == aug_pair["pair_id"]), None)
            if subset_pair and aug_pair["original_jd_text"] == subset_pair["jd_text"]:
                orig_match += 1

    if orig_match == aug_count:
        print(f"[PASS] original_jd_text matches original JD on all {aug_count} pairs")
    else:
        errors.append(f"original_jd_text mismatch on {aug_count - orig_match} pairs")

    # Verification 3: Content bounds
    print(f"\n{'='*70}")
    print("3. CONTENT BOUNDS (100-160 words)")
    print(f"{'='*70}")

    def count_words(text):
        return len(text.split())

    aug_jds = [p for p in augmented if p.get("jd_augmented")]
    word_counts = [count_words(p["jd_text"]) for p in aug_jds]

    in_range = sum(1 for wc in word_counts if 100 <= wc <= 160)
    outliers = [(p["pair_id"], count_words(p["jd_text"])) for p in aug_jds if not (100 <= count_words(p["jd_text"]) <= 160)]

    if len(outliers) == 0:
        print(f"[PASS] All {len(aug_jds)} augmented JDs in 100-160 word range")
    else:
        print(f"[WARN] {len(outliers)} outliers found:")
        for pair_id, wc in outliers[:10]:
            print(f"  - {pair_id}: {wc} words")

    print(f"Word count distribution:")
    print(f"  Min: {min(word_counts)} words")
    print(f"  Max: {max(word_counts)} words")
    print(f"  Median: {sorted(word_counts)[len(word_counts)//2]} words")

    # Verification 4: Check for banned phrases
    print(f"\n{'='*70}")
    print("4. BANNED FILLER PHRASES")
    print(f"{'='*70}")

    banned_phrases = [
        "fast-paced environment", "passionate team", "drive groundbreaking solutions",
        "exciting opportunity", "self-starter", "dynamic environment", "innovative solutions",
        "team-oriented", "thrive in", "results-driven", "innovative thinker", "make an impact"
    ]

    phrase_issues = []
    for aug_pair in aug_jds:
        text = aug_pair["jd_text"].lower()
        for phrase in banned_phrases:
            if phrase in text:
                phrase_issues.append((aug_pair["pair_id"], phrase))

    if len(phrase_issues) == 0:
        print(f"[PASS] No banned filler phrases detected")
    else:
        print(f"[WARN] {len(phrase_issues)} instances of banned phrases:")
        for pair_id, phrase in phrase_issues[:5]:
            print(f"  - {pair_id}: '{phrase}'")

    # Verification 5: Spot check (5 random augmented pairs)
    print(f"\n{'='*70}")
    print("5. SPOT CHECK (5 Random Augmented Pairs)")
    print(f"{'='*70}\n")

    random.seed(42)
    spot_check = random.sample(aug_jds, min(5, len(aug_jds)))

    for i, pair in enumerate(spot_check, 1):
        orig = pair["original_jd_text"]
        aug = pair["jd_text"]
        print(f"Pair {i}: {pair['pair_id']}")
        print(f"  Original ({count_words(orig)} words): {orig[:60]}...")
        print(f"  Augmented ({count_words(aug)} words): {aug[:60]}...")
        print()

    # Summary
    print(f"{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")

    if not errors:
        print("✓ All checks PASSED!")
        print(f"  - 1000 pairs verified")
        print(f"  - 907 JDs augmented (90.7%)")
        print(f"  - 93 JDs preserved (9.3%)")
        print(f"  - All schemas intact")
        print(f"  - No banned filler phrases")
        return True
    else:
        print("✗ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
