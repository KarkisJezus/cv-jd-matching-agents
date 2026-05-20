"""
Download and convert the HuggingFace Resume-Screening-Dataset
to the internal eval_dataset.json format.

Source: https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset
Fields: Resume, Job_Description, Decision (select/reject), Role, Reason_for_decision

IMPORTANT: This script is for dataset INGESTION only. Ground truth labels
are stored for post-hoc evaluation — they are never exposed to agents
during execution.

Usage:
    python data/load_hf_dataset.py --sample 50
    python data/load_hf_dataset.py --sample 5 --output data/hf_debug.json
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DATASET_QUALITY_DISCLAIMER = (
    "Source labels may contain noise or inconsistencies. "
    "Reason_for_decision may not always reflect true semantic matching quality. "
    "Results must be interpreted with caution. "
    "Qualitative analysis complements quantitative metrics."
)


def load_hf_dataset():
    """Download the dataset from HuggingFace."""
    from datasets import load_dataset

    print("Downloading AzharAli05/Resume-Screening-Dataset from HuggingFace...")
    ds = load_dataset("AzharAli05/Resume-Screening-Dataset", split="train")
    print(f"Downloaded {len(ds)} rows.")
    return ds


def sample_balanced(ds, sample_size: int | None, seed: int = 42) -> list[dict]:
    """
    Sample pairs with class balance and role diversity.

    Strategy:
    - Split into select/reject groups
    - Within each group, spread across different Role values
    - Take half select, half reject (balanced)

    If sample_size is None or >= len(ds), returns ALL rows (shuffled,
    class balance preserved as-is from the source). This is used when
    --sample-all is specified to download the entire dataset.
    """
    rng = random.Random(seed)

    # Group by decision
    select_rows = []
    reject_rows = []
    for row in ds:
        if row["Decision"].strip().lower() == "select":
            select_rows.append(row)
        else:
            reject_rows.append(row)

    print(f"Dataset composition: {len(select_rows)} select, {len(reject_rows)} reject")

    # Full-dataset mode: return everything, just shuffle
    total = len(select_rows) + len(reject_rows)
    if sample_size is None or sample_size >= total:
        combined = select_rows + reject_rows
        rng.shuffle(combined)
        print(f"Returning all {len(combined)} rows (full dataset)")
        return combined

    half = sample_size // 2
    select_half = half
    reject_half = sample_size - select_half  # handles odd numbers

    # Sample with role diversity
    select_sample = _diverse_sample(select_rows, select_half, rng)
    reject_sample = _diverse_sample(reject_rows, reject_half, rng)

    combined = select_sample + reject_sample
    rng.shuffle(combined)
    return combined


def _diverse_sample(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n rows, trying to pick from diverse Role values."""
    if n >= len(rows):
        return rows[:n]

    # Group by role
    by_role = defaultdict(list)
    for row in rows:
        by_role[row["Role"]].append(row)

    # Round-robin across roles
    sampled = []
    roles = list(by_role.keys())
    rng.shuffle(roles)

    role_idx = 0
    while len(sampled) < n:
        role = roles[role_idx % len(roles)]
        if by_role[role]:
            sampled.append(by_role[role].pop(rng.randrange(len(by_role[role]))))
        role_idx += 1
        # Safety: if all role buckets are empty, break
        if all(len(v) == 0 for v in by_role.values()):
            break

    return sampled[:n]


def convert_to_internal_format(rows: list[dict]) -> dict:
    """
    Convert HF dataset rows to internal eval_dataset.json format.

    Ground truth is stored for post-hoc evaluation only.
    It is never exposed to agents during execution.
    """
    # Track role counts for unique pair_ids
    role_counts = defaultdict(int)

    pairs = []
    for row in rows:
        role = row["Role"].strip()
        role_slug = role.lower().replace(" ", "_").replace("-", "_")
        role_counts[role_slug] += 1
        pair_id = f"{role_slug}_{role_counts[role_slug]:03d}"

        decision = row["Decision"].strip().lower()
        is_match = decision == "select"

        pairs.append({
            "pair_id": pair_id,
            "cv_text": row["Resume"].strip(),
            "jd_text": row["Job_Description"].strip(),
            "ground_truth_label": is_match,
            "description": f"{role} — {decision}",
            "reference_reason": row["Reason_for_decision"].strip(),
        })

    return {
        "_meta": {
            "source": "AzharAli05/Resume-Screening-Dataset",
            "source_url": "https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset",
            "sample_size": len(pairs),
            "class_balance": {
                "select": sum(1 for p in pairs if p["ground_truth_label"]),
                "reject": sum(1 for p in pairs if not p["ground_truth_label"]),
            },
            "roles": sorted(set(
                p["description"].rsplit(" — ", 1)[0] for p in pairs
            )),
            "disclaimer": DATASET_QUALITY_DISCLAIMER,
            "labels": (
                "ground_truth_label: true=select (candidate should be considered), "
                "false=reject. reference_reason is the dataset's original justification "
                "and is stored for qualitative comparison only — never shown to agents."
            ),
        },
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert HuggingFace Resume-Screening-Dataset"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="Number of CV-JD pairs to sample (default: 50). "
             "Balanced between select/reject.",
    )
    parser.add_argument(
        "--output",
        default="data/hf_eval_dataset.json",
        help="Output JSON file path (default: data/hf_eval_dataset.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    # Download
    ds = load_hf_dataset()

    # Sample
    print(f"Sampling {args.sample} pairs (balanced, seed={args.seed})...")
    rows = sample_balanced(ds, args.sample, seed=args.seed)

    # Convert
    data = convert_to_internal_format(rows)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Report
    meta = data["_meta"]
    print(f"\nSaved {meta['sample_size']} pairs to {args.output}")
    print(f"  Select: {meta['class_balance']['select']}")
    print(f"  Reject: {meta['class_balance']['reject']}")
    print(f"  Roles:  {len(meta['roles'])} unique")
    print(f"\nDisclaimer: {DATASET_QUALITY_DISCLAIMER}")


if __name__ == "__main__":
    main()
