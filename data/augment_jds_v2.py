#!/usr/bin/env python3
"""
Re-augment 1000-pair CV-JD subset using quality-first V2 prompt.
Uses Claude Haiku 4.5 with blind augmentation (JD + role prefix only, no CV/label/reason).
"""

import json
import sys
import os
from pathlib import Path
from anthropic import Anthropic

HAIKU_MODEL = "claude-haiku-4-5-20251001"
WORD_THRESHOLD = 30

SYSTEM_PROMPT = """You are expanding a job description from a recruitment dataset.
The original JD is too short (often 1-2 sentences) to evaluate a candidate fairly. Expand it into a realistic, substantive JD while preserving the original meaning and any specific role/domain mentions.

CRITICAL CONSTRAINTS:
1. Preserve role + domain signal. If the original says "Game Developer driving solutions in data science", your output must still describe a Game Developer role that mentions data science. Do NOT change it to a pure data science JD or a pure game development JD.
2. No invented specifics. Do NOT invent company names, salary figures, specific years of experience numbers, location names, or interview steps.
3. No filler phrases. Avoid: "fast-paced environment", "passionate team", "drive groundbreaking solutions", "exciting opportunity", "self-starter", "dynamic environment", "innovative solutions", "team-oriented", "thrive in", "results-driven", "innovative thinker", "make an impact". These add no information.
4. Concrete responsibilities. Include 3-5 specific, role-realistic responsibilities. Pull from the role's domain (e.g., a data engineer JD might mention ETL pipelines, query optimization, schema design — not "innovation").
5. Required skills. Include a brief list of role-relevant technical skills. Make them realistic for the role and the domain mentioned in the input.
6. Length: 100-150 words. Quality > quantity. Better short and substantive than long and padded.
7. Output plain prose. No JSON, no markdown headings, no bullet lists in the final text — just clean paragraph(s) suitable as the JD body of a real posting.
8. Maintain the role-domain mash-up if present. The "weird" pairing in the input (e.g., UX Designer + cybersecurity) is the experimental variable. Preserve it explicitly in the output.

WORKED EXAMPLES:

Example 1:
INPUT: "We are looking for a Data Engineer to drive solutions in machine learning."
ROLE: data_engineer
OUTPUT:
We are seeking a Data Engineer to design and maintain data pipelines that support machine learning workflows. Responsibilities include building ETL processes, optimizing query performance on large datasets, and collaborating with ML engineers to ensure clean, accessible training data. The ideal candidate has experience with SQL, Python, and at least one workflow orchestration tool such as Airflow. Familiarity with cloud data platforms (BigQuery, Snowflake, or Redshift) and data modeling principles is required. Knowledge of feature engineering for ML pipelines is a plus.

Example 2 (weird role/domain):
INPUT: "Looking for a UX Designer to help drive groundbreaking solutions in cybersecurity."
ROLE: ux_designer
OUTPUT:
We are hiring a UX Designer to work on user-facing interfaces in cybersecurity products. The role involves user research with security professionals (analysts, IT teams, end users), wireframing and prototyping for security dashboards and threat-monitoring tools, and collaborating with product managers and security engineers to translate complex security workflows into accessible designs. Candidates should have experience with Figma or Sketch, user research methods, and information architecture. Familiarity with security domain terminology is helpful but not required.

Example 3:
INPUT: "Senior Software Engineer wanted."
ROLE: software_engineer
OUTPUT:
We are looking for a Senior Software Engineer to lead development of production services. Responsibilities include designing scalable backend systems, leading code reviews, mentoring mid-level engineers, and collaborating with product on feature scoping. The candidate should have strong experience in at least one general-purpose language (Python, Java, Go, or similar), familiarity with distributed systems concepts, and a track record of delivering production software. Knowledge of cloud infrastructure, CI/CD pipelines, and test-driven development is required.

You will receive the original short JD and the role prefix from the pair_id. You do NOT receive the CV, the label, or the rejection reason. This blind setup prevents your augmentation from biasing toward making the answer obvious.

Output: the expanded JD as plain prose. No preamble, no explanation, no markdown — just the expanded JD ready to drop in."""

def count_words(text: str) -> int:
    return len(text.split())

def extract_role_from_pair_id(pair_id: str) -> str:
    """Extract role prefix from pair_id (e.g., 'data_engineer' from 'data_engineer_615')."""
    parts = pair_id.rsplit("_", 1)
    return parts[0] if parts else "professional"

def augment_jd_with_haiku(client: Anthropic, original_jd: str, role_prefix: str) -> tuple[str, int, int]:
    """
    Call Claude Haiku to augment a JD.
    Returns: (augmented_text, input_tokens, output_tokens)
    """
    user_message = f"""ORIGINAL JD:
{original_jd}

ROLE PREFIX (from pair_id, for context — do NOT mention the candidate or CV):
{role_prefix}

EXPECTED LENGTH: 100-150 words.
Output: the expanded JD as plain prose. No preamble, no explanation, no markdown — just the expanded JD ready to drop in."""

    message = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    augmented_text = message.content[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    return augmented_text, input_tokens, output_tokens

def main():
    print("=" * 70)
    print("V2 AUGMENTATION: Quality-First JD Expansion")
    print("=" * 70)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY not found in environment.")
        print("Please set it and try again.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Load subset
    print("\nLoading subset...")
    with open("data/hf_test_1000_subset.json", "r") as f:
        data = json.load(f)

    pairs = data["pairs"]
    print(f"[OK] Loaded {len(pairs)} pairs")

    # Identify short JDs
    short_pairs = [(i, p) for i, p in enumerate(pairs) if count_words(p["jd_text"]) < WORD_THRESHOLD]
    print(f"[OK] Found {len(short_pairs)} short JDs (< {WORD_THRESHOLD} words)")

    # Dry run on first 10 short JDs
    print(f"\n{'='*70}")
    print("DRY RUN: 10 Short JDs")
    print(f"{'='*70}\n")

    dry_run_results = []
    total_input = 0
    total_output = 0

    for idx, (pair_idx, pair) in enumerate(short_pairs[:10], 1):
        original_jd = pair["jd_text"]
        role_prefix = extract_role_from_pair_id(pair["pair_id"])
        word_count = count_words(original_jd)

        print(f"[{idx}/10] {pair['pair_id']} ({word_count} words)...", end=" ", flush=True)

        try:
            augmented_jd, in_tokens, out_tokens = augment_jd_with_haiku(client, original_jd, role_prefix)
            total_input += in_tokens
            total_output += out_tokens

            aug_word_count = count_words(augmented_jd)
            cost = (in_tokens / 1e6) * 1.0 + (out_tokens / 1e6) * 5.0

            dry_run_results.append({
                "pair_id": pair["pair_id"],
                "original_jd": original_jd,
                "augmented_jd": augmented_jd,
                "original_words": word_count,
                "augmented_words": aug_word_count,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "cost": cost
            })

            print(f"OK ({aug_word_count} words, ${cost:.4f})")

        except Exception as e:
            print(f"ERROR: {e}")
            raise

    # Print summary
    print(f"\n{'='*70}")
    print("DRY RUN RESULTS")
    print(f"{'='*70}")
    print(f"Augmentations:       10")
    print(f"Input tokens:        {total_input:,}")
    print(f"Output tokens:       {total_output:,}")
    total_cost = (total_input / 1e6) * 1.0 + (total_output / 1e6) * 5.0
    print(f"Total cost:          ${total_cost:.2f}")

    # Show 3 examples
    print(f"\n{'='*70}")
    print("3 BEFORE/AFTER EXAMPLES")
    print(f"{'='*70}\n")

    for i, result in enumerate(dry_run_results[:3], 1):
        print(f"Example {i}: {result['pair_id']}")
        print(f"\nORIGINAL ({result['original_words']} words):")
        print(f'"{result["original_jd"]}"')
        print(f"\nAUGMENTED ({result['augmented_words']} words):")
        print(f'"{result["augmented_jd"]}"')
        print()

    # Sanity checks
    print(f"{'='*70}")
    print("SANITY CHECKS")
    print(f"{'='*70}")

    word_counts = [r["augmented_words"] for r in dry_run_results]
    in_range = sum(1 for wc in word_counts if 100 <= wc <= 150)
    print(f"Augmented JDs in 100-150 word range: {in_range}/{len(dry_run_results)}")

    banned_phrases = [
        "fast-paced environment", "passionate team", "drive groundbreaking solutions",
        "exciting opportunity", "self-starter", "dynamic environment", "innovative solutions",
        "team-oriented", "thrive in", "results-driven", "innovative thinker", "make an impact"
    ]

    for result in dry_run_results:
        text = result["augmented_jd"].lower()
        for phrase in banned_phrases:
            if phrase in text:
                print(f"WARNING: '{phrase}' found in {result['pair_id']}")

    print(f"\n{'='*70}")
    print("PROCEED TO FULL RUN?")
    print("Review the examples above. If satisfied, re-run with argument 'full' to continue.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
