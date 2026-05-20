"""
Streaming evaluation protocol — Tier 2 architecture.

Implements the StreamBench-style streaming protocol (Yehudai et al. 2025 §3)
for the new labeled-memory architecture:

1. Pair N is fetched from the dataset.
2. Pair N is fed to the agent chain. The system commits a final_decision
   WITHOUT seeing pair N's own ground_truth_label.
3. After the decision is committed, pair N's ground_truth_label and
   ground_truth_reason are attached. A LabeledMemoryEntry is built and
   added to the LabeledMemoryStore.
4. The next pair (N+1) may retrieve LabeledMemoryEntries from prior pairs
   (1..N) when it runs Pass 2 (CalibrationAgent).

Three memory modes are supported:
- cold-start (default): clear memory dir at the start of every run
- continue-stream: load existing memory; refuse to run if any input
  pair_id already exists in memory (would cause label leakage)
- fresh-build: same as cold-start but with an explicit warning if memory existed

The cold-start default ensures every primary thesis evaluation is reproducible
from a known empty starting state. continue-stream is the opt-in mode for
extending streaming evaluation across multiple sessions.
"""

from typing import Literal

from memory.labeled_store import LabeledMemoryStore
from models.entities import LabeledMemoryEntry


MemoryMode = Literal["cold-start", "continue-stream", "fresh-build"]


class StreamingProtocolError(Exception):
    """Raised when the streaming protocol's invariants would be violated."""


def prepare_memory_store(
    memory_dir: str,
    mode: MemoryMode,
    input_pair_ids: list[str],
    embedding_similarity=None,
) -> LabeledMemoryStore:
    """
    Initialize a LabeledMemoryStore according to the requested memory mode.

    Args:
        memory_dir: Path to the memory directory.
        mode: One of 'cold-start', 'continue-stream', 'fresh-build'.
        input_pair_ids: All pair_ids in the input dataset (for overlap detection).
        embedding_similarity: Optional shared EmbeddingSimilarity instance.

    Returns:
        A ready-to-use LabeledMemoryStore.

    Raises:
        StreamingProtocolError: in continue-stream mode if any input pair_id
            already exists in memory (would cause label leakage).
        ValueError: if mode is invalid.
    """
    if mode not in ("cold-start", "continue-stream", "fresh-build"):
        raise ValueError(
            f"Invalid memory mode: {mode!r}. "
            "Must be one of: cold-start, continue-stream, fresh-build."
        )

    # Construct store (auto-loads existing files if present)
    store = LabeledMemoryStore(
        memory_dir=memory_dir,
        embedding_similarity=embedding_similarity,
    )
    existing_count = store.count

    if mode == "cold-start":
        # Always start empty. Clear any existing files silently.
        if existing_count > 0:
            store.clear_files()
            print(
                f"[memory mode: cold-start] Cleared {existing_count} prior "
                f"memories from {memory_dir}. Starting fresh."
            )

    elif mode == "fresh-build":
        # Same effect as cold-start, but warn loudly if memory existed.
        if existing_count > 0:
            print(
                f"[memory mode: fresh-build] WARNING: "
                f"{existing_count} memories existed in {memory_dir}; wiping them."
            )
            store.clear_files()
        else:
            print(f"[memory mode: fresh-build] Memory dir empty — same as cold-start.")

    elif mode == "continue-stream":
        # Preserve existing memory but enforce no-overlap with input.
        existing_pair_ids = store.pair_ids()
        overlap = set(input_pair_ids) & existing_pair_ids
        if overlap:
            raise StreamingProtocolError(
                f"Cannot run continue-stream evaluation: {len(overlap)} input "
                f"pair_ids already exist in memory at {memory_dir}. "
                f"Re-evaluating these pairs would leak their known labels into the "
                f"system's prediction (test-set contamination).\n"
                f"Overlapping pair_ids (first 10 shown): "
                f"{sorted(list(overlap))[:10]}\n\n"
                f"Resolutions:\n"
                f"  1. Use --memory-mode cold-start to wipe memory and start fresh.\n"
                f"  2. Use a different input dataset that doesn't overlap.\n"
                f"  3. Remove the overlapping entries manually from "
                f"{memory_dir}/labeled_memories.json."
            )
        if existing_count > 0:
            print(
                f"[memory mode: continue-stream] Loaded {existing_count} prior "
                f"memories from {memory_dir}. Continuing the stream with "
                f"{len(input_pair_ids)} new pairs."
            )
        else:
            print(
                f"[memory mode: continue-stream] Memory dir empty. "
                f"Behaving identically to cold-start for this run."
            )

    return store


def build_labeled_entry(
    pair_id: str,
    cv_profile_summary: str,
    jd_profile_summary: str,
    detected_role: str,
    system_score: float,
    system_recommendation: str,
    system_initial_score: float,
    system_reasoning_summary: str,
    ground_truth_label: bool,
    ground_truth_reason: str,
    threshold: float,
    influenced_by: list[str] | None = None,
) -> LabeledMemoryEntry:
    """
    Construct a LabeledMemoryEntry with derived fields filled in.

    Called by the runner AFTER the system's final_decision is committed and
    AFTER the ground_truth_label has been attached from the dataset.

    Computes:
    - was_correct: did the system's predicted_label match ground_truth_label?
    - error_direction: TP / FP / TN / FN
    """
    predicted_match = system_score >= threshold

    if predicted_match and ground_truth_label:
        error_direction = "TP"
    elif predicted_match and not ground_truth_label:
        error_direction = "FP"
    elif not predicted_match and not ground_truth_label:
        error_direction = "TN"
    else:
        error_direction = "FN"

    was_correct = error_direction in ("TP", "TN")

    return LabeledMemoryEntry(
        pair_id=pair_id,
        cv_profile_summary=cv_profile_summary,
        jd_profile_summary=jd_profile_summary,
        detected_role=detected_role,
        system_score=system_score,
        system_recommendation=system_recommendation,
        system_initial_score=system_initial_score,
        system_reasoning_summary=system_reasoning_summary,
        ground_truth_label=ground_truth_label,
        ground_truth_reason=ground_truth_reason,
        was_correct=was_correct,
        error_direction=error_direction,
        influenced_by=influenced_by or [],
    )
