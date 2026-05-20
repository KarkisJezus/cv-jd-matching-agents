"""4.5 poskyrio metrikos: JD augmentacijos eksperimentas (1000 porų).

Lygina sistemos veikimą su:
  - Originaliais JD (1000-porų pogrupis iš 5000 pagrindinio eksperimento)
  - Augmentuotais JD (tie patys 1000 pair_ids, bet su pratęstais skelbimais)

Trys etalonai (kaip 4.3):
  1. Šaltinio etiketė
  2. Teisėjo etiketė (atfiltruojant ambiguous)
  3. Pataisytas etalonas

Naudojimas:
    python scripts/compute_augmentation_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ORIG_PATH = Path("results/tier2_5000_combined/hf_test_5000.json")
AUG_PATH = Path("results/tier2_1000_augmented_v2_gpt/hf_test_1000_augmented_v2.json")
ORIG_JUDGE_PATH = Path("results/judge_hf_test_5000.json")
AUG_JUDGE_PATH = Path("results/judge_hf_test_1000_augmented_v2.json")

SCENARIOS = ["baseline", "A", "B", "C"]
THRESHOLD = 60.0


def get_score_field(s: str) -> str:
    return "score_baseline" if s == "baseline" else f"score_{s}"


def compute_metrics(predictions, labels):
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    orig = json.loads(ORIG_PATH.read_text(encoding="utf-8"))
    aug = json.loads(AUG_PATH.read_text(encoding="utf-8"))
    orig_judge = json.loads(ORIG_JUDGE_PATH.read_text(encoding="utf-8"))
    aug_judge = json.loads(AUG_JUDGE_PATH.read_text(encoding="utf-8"))

    orig_by_id = {p["pair_id"]: p for p in orig["pair_results"]}
    aug_by_id = {p["pair_id"]: p for p in aug["pair_results"]}
    orig_judge_by_id = orig_judge.get("judgments", {})
    aug_judge_by_id = aug_judge.get("judgments", {})

    # Bendri pair_ids: tos pačios poros, kurios buvo augmentuotos
    common_ids = set(orig_by_id.keys()) & set(aug_by_id.keys())
    print(f"Bendri pair_ids: {len(common_ids)} (orig: {len(orig_by_id)}, aug: {len(aug_by_id)})\n")

    # Apjungtos poros (turi originalūs, augmentuoti, abu auditai)
    joined = []
    for pid in sorted(common_ids):
        op = orig_by_id[pid]
        ap = aug_by_id[pid]
        oj = orig_judge_by_id.get(pid)
        aj = aug_judge_by_id.get(pid)
        joined.append((pid, op, ap, oj, aj))

    # ──────────────────────────────────────────────────────────────
    # 1 LENTELĖ: prieš ŠALTINIO etiketes (atskirai orig vs aug)
    # ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("1 LENTELĖ: F1 prieš ŠALTINIO etiketes (n=" + str(len(joined)) + ")")
    print("=" * 78)
    print(f"{'Metodas':<12} {'F1 orig':>10} {'F1 aug':>10} {'Δ aug-orig':>14}")
    for s in SCENARIOS:
        field = get_score_field(s)
        # Orig
        preds_o = [op[field] >= THRESHOLD for _, op, _, _, _ in joined]
        labels_o = [bool(op["ground_truth_label"]) for _, op, _, _, _ in joined]
        m_o = compute_metrics(preds_o, labels_o)
        # Aug
        preds_a = [ap[field] >= THRESHOLD for _, _, ap, _, _ in joined]
        labels_a = [bool(ap["ground_truth_label"]) for _, _, ap, _, _ in joined]
        m_a = compute_metrics(preds_a, labels_a)
        delta = m_a["f1"] - m_o["f1"]
        print(f"{s:<12} {m_o['f1']:>10.4f} {m_a['f1']:>10.4f} {delta:>+13.4f}")
    print()

    # ──────────────────────────────────────────────────────────────
    # 2 LENTELĖ: prieš TEISĖJO etiketes (atfiltruojant ambiguous)
    # ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("2 LENTELĖ: F1 prieš TEISĖJO etiketes (atfiltruojant ambiguous)")
    print("=" * 78)
    # Iš teisėjo audito atfiltruojame ambiguous
    print(f"{'Metodas':<12} {'F1 orig':>10} {'n_orig':>8} {'F1 aug':>10} {'n_aug':>8} {'Δ aug-orig':>14}")
    for s in SCENARIOS:
        field = get_score_field(s)
        # Orig
        preds_o, labels_o = [], []
        for pid, op, _, oj, _ in joined:
            if oj is None or oj.get("error") or oj.get("source_assessment") == "ambiguous":
                continue
            jl = oj.get("judge_label")
            if jl is None:
                continue
            preds_o.append(op[field] >= THRESHOLD)
            labels_o.append(bool(jl))
        m_o = compute_metrics(preds_o, labels_o)
        # Aug
        preds_a, labels_a = [], []
        for pid, _, ap, _, aj in joined:
            if aj is None or aj.get("error") or aj.get("source_assessment") == "ambiguous":
                continue
            jl = aj.get("judge_label")
            if jl is None:
                continue
            preds_a.append(ap[field] >= THRESHOLD)
            labels_a.append(bool(jl))
        m_a = compute_metrics(preds_a, labels_a)
        delta = m_a["f1"] - m_o["f1"]
        print(f"{s:<12} {m_o['f1']:>10.4f} {m_o['n']:>8} {m_a['f1']:>10.4f} {m_a['n']:>8} {delta:>+13.4f}")
    print()

    # ──────────────────────────────────────────────────────────────
    # 3 LENTELĖ: detalios metrikos su augmentuotais JD
    # ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("3 LENTELĖ: detalios augmentuoto eksperimento metrikos prieš ŠALTINĮ")
    print("=" * 78)
    print(f"{'Metodas':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    for s in SCENARIOS:
        field = get_score_field(s)
        preds = [ap[field] >= THRESHOLD for _, _, ap, _, _ in joined]
        labels = [bool(ap["ground_truth_label"]) for _, _, ap, _, _ in joined]
        m = compute_metrics(preds, labels)
        print(f"{s:<12} {m['accuracy']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}  {m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}")
    print()

    # ──────────────────────────────────────────────────────────────
    # 4 LENTELĖ: teisėjo audito statistika augmentuotam eksperimentui
    # ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("4 LENTELĖ: teisėjo audito statistika augmentuotam eksperimentui")
    print("=" * 78)
    aug_meta = aug_judge.get("_meta", {})
    print(f"  source_correct:   {aug_meta.get('n_source_correct')}  ({aug_meta.get('fraction_source_correct'):.4f})")
    print(f"  source_incorrect: {aug_meta.get('n_source_incorrect')}  ({aug_meta.get('fraction_source_incorrect'):.4f})")
    print(f"  ambiguous:        {aug_meta.get('n_ambiguous')}  ({aug_meta.get('fraction_ambiguous'):.4f})")
    print(f"  iš viso:          {aug_meta.get('n_pairs')}")
    print()

    # Palyginti su pagrindiniu (orig) auditu
    orig_meta = orig_judge.get("_meta", {})
    print(f"  ORIGINALIAI (5000): source_incorrect = {orig_meta.get('fraction_source_incorrect'):.4f}")
    print(f"  AUGMENTUOTI (1000): source_incorrect = {aug_meta.get('fraction_source_incorrect'):.4f}")
    print()

    # ──────────────────────────────────────────────────────────────
    # 5 LENTELĖ: vidutiniai sistemos balai (orig vs aug)
    # ──────────────────────────────────────────────────────────────
    print("=" * 78)
    print("5 LENTELĖ: vidutiniai sistemos balai")
    print("=" * 78)
    print(f"{'Metodas':<12} {'mean orig':>12} {'mean aug':>12} {'Δ':>10}")
    for s in SCENARIOS:
        field = get_score_field(s)
        scores_o = [op[field] for _, op, _, _, _ in joined]
        scores_a = [ap[field] for _, _, ap, _, _ in joined]
        mean_o = sum(scores_o) / len(scores_o)
        mean_a = sum(scores_a) / len(scores_a)
        print(f"{s:<12} {mean_o:>12.2f} {mean_a:>12.2f} {mean_a - mean_o:>+10.2f}")
    print()


if __name__ == "__main__":
    main()
