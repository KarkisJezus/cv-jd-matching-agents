"""
Stratified train/test split for evaluation datasets.

Takes a dataset JSON produced by load_hf_dataset.py (or any file with the
same schema: {"_meta": {...}, "pairs": [...]}) and splits it into two files
preserving the class balance of ground_truth_label.

Why stratified: the source dataset is class-balanced 50/50. A naive random
split risks skewing one side (e.g., 60/40 train, 40/60 test). Stratified
sampling guarantees the ratio is preserved in both outputs, so accuracy
metrics are comparable between train and test.

Usage:
    python data/split_dataset.py --input data/hf_full.json --train 0.7 --seed 42
    # Produces data/hf_full_train.json and data/hf_full_test.json

    python data/split_dataset.py --input data/hf_eval_400.json --train 0.8
    # Produces data/hf_eval_400_train.json and data/hf_eval_400_test.json
"""

import argparse
import json
import random
from pathlib import Path


def stratified_split(
    pairs: list[dict],
    train_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """
    Split pairs by ground_truth_label preserving class ratio.

    Positives (label=True) and negatives (label=False) are independently
    shuffled and split at train_fraction. The two train halves are combined
    and shuffled; the two test halves likewise.

    Returns (train_pairs, test_pairs).
    """
    rng = random.Random(seed)

    pos = [p for p in pairs if p.get("ground_truth_label") is True]
    neg = [p for p in pairs if p.get("ground_truth_label") is False]
    unknown = [p for p in pairs if p.get("ground_truth_label") is None]

    if unknown:
        print(f"WARNING: {len(unknown)} pairs have no ground_truth_label; dropped.")

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_train = int(len(pos) * train_fraction)
    n_neg_train = int(len(neg) * train_fraction)

    train = pos[:n_pos_train] + neg[:n_neg_train]
    test = pos[n_pos_train:] + neg[n_neg_train:]

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test


def main():
    parser = argparse.ArgumentParser(
        description="Stratified train/test split for job-match evaluation datasets"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset JSON (e.g., data/hf_full.json)",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Fraction of pairs to put in the training set (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    parser.add_argument(
        "--train-output",
        default=None,
        help="Path for training output. Defaults to <input>_train.json",
    )
    parser.add_argument(
        "--test-output",
        default=None,
        help="Path for test output. Defaults to <input>_test.json",
    )
    args = parser.parse_args()

    if not (0.0 < args.train < 1.0):
        raise SystemExit("--train must be between 0 and 1 (exclusive)")

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    data = json.loads(input_path.read_text(encoding="utf-8"))
    pairs = data.get("pairs", [])
    if not pairs:
        raise SystemExit(f"No 'pairs' array in {input_path}")

    print(f"Loaded {len(pairs)} pairs from {input_path}")

    # Split
    train_pairs, test_pairs = stratified_split(pairs, args.train, args.seed)

    # Build output files with metadata
    def build_output(split_pairs: list[dict], split_name: str) -> dict:
        meta = dict(data.get("_meta", {}))
        meta.update({
            "split": split_name,
            "split_fraction": args.train if split_name == "train" else 1 - args.train,
            "split_seed": args.seed,
            "split_parent": str(input_path),
            "sample_size": len(split_pairs),
            "class_balance": {
                "select": sum(1 for p in split_pairs if p.get("ground_truth_label")),
                "reject": sum(1 for p in split_pairs if p.get("ground_truth_label") is False),
            },
        })
        return {"_meta": meta, "pairs": split_pairs}

    train_data = build_output(train_pairs, "train")
    test_data = build_output(test_pairs, "test")

    # Determine output paths
    stem = input_path.stem
    train_out = Path(args.train_output or input_path.with_name(f"{stem}_train.json"))
    test_out = Path(args.test_output or input_path.with_name(f"{stem}_test.json"))

    train_out.write_text(json.dumps(train_data, indent=2, ensure_ascii=False), encoding="utf-8")
    test_out.write_text(json.dumps(test_data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Report
    print()
    print(f"Train: {len(train_pairs)} pairs -> {train_out}")
    print(f"  Select: {train_data['_meta']['class_balance']['select']}")
    print(f"  Reject: {train_data['_meta']['class_balance']['reject']}")
    print(f"Test:  {len(test_pairs)} pairs -> {test_out}")
    print(f"  Select: {test_data['_meta']['class_balance']['select']}")
    print(f"  Reject: {test_data['_meta']['class_balance']['reject']}")
    print(f"\nSeed: {args.seed} (re-running with the same seed produces the same split)")


if __name__ == "__main__":
    main()
