"""4.2/4.3 poskyrių metrikos: trijų etalonų lyginamoji analizė.

Apjungia sistemos rezultatus (tier2_5000_combined) su teisėjo auditu (judge_hf_test_5000)
per pair_id ir paskaičiuoja sistemos metrikas prieš tris skirtingus etalonus:

  1. Šaltinio etiketė (rinkinio originali; turi etikečių triukšmo)
  2. Teisėjo etiketė (nepriklausomo LLM-teisėjo verdiktas; atfiltruoja ambiguous)
  3. Pataisytas etalonas (jei teisėjas pažymi šaltinį kaip „correct" — paliekam šaltinio
     etiketę; jei „incorrect" — pakeičiam į teisėjo etiketę; ambiguous atfiltruoja)

Skaičiuoja:
  - Audito statistikas (kiek source_correct / source_incorrect / ambiguous)
  - Sistemos metrikas (acc, prec, rec, F1) prieš tris etalonus, visiems scenarijams
  - Klaidų taksonomijos pasiskirstymą
  - Cohen kappa tarp porų (šaltinis, teisėjas) ir (sistema, teisėjas)

Naudojimas:
    python scripts/compute_audit_metrics.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PATH = Path("results/tier2_5000_combined/hf_test_5000.json")
JUDGE_PATH = Path("results/judge_hf_test_5000.json")
SCENARIOS = ["baseline", "A", "B", "C"]
THRESHOLD = 60.0


def load_data():
    sys_data = json.loads(SYSTEM_PATH.read_text(encoding="utf-8"))
    judge_data = json.loads(JUDGE_PATH.read_text(encoding="utf-8"))
    return sys_data, judge_data


def get_score_field(scenario: str) -> str:
    return "score_baseline" if scenario == "baseline" else f"score_{scenario}"


def compute_metrics(predictions: list[bool], labels: list[bool]) -> dict:
    """Standard binary classification metrics."""
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def cohen_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Cohen's kappa for binary agreement."""
    n = len(labels_a)
    if n == 0:
        return 0.0
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_o = agree / n
    a_pos = sum(labels_a) / n
    b_pos = sum(labels_b) / n
    p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0.0


def main():
    sys_data, judge_data = load_data()
    pairs = sys_data["pair_results"]
    judgments = judge_data["judgments"]

    # Index judgments by pair_id
    judge_by_id = {pid: j for pid, j in judgments.items()}

    # Join system results with judgments
    joined = []
    for p in pairs:
        pid = p.get("pair_id")
        if pid not in judge_by_id:
            continue
        j = judge_by_id[pid]
        if j.get("error"):
            continue
        joined.append((p, j))

    print(f"Apjungtos poros: {len(joined)} (iš {len(pairs)} sistemos + {len(judgments)} teisėjo)\n")

    # ── 1. Audito statistika ────────────────────────────────────────
    print("=" * 78)
    print("1 LENTELĖ: teisėjo audito statistika (n=5000)")
    print("=" * 78)
    sa_counts = Counter(j.get("source_assessment", "?") for _, j in joined)
    n_total = len(joined)
    for cat in ["correct", "incorrect", "ambiguous"]:
        n = sa_counts.get(cat, 0)
        print(f"  source_assessment = {cat:<12}: {n:>5} ({100 * n / n_total:.2f}\\,\\%)")
    print(f"  iš viso: {n_total}")
    print()

    # Klaidų taksonomijos pasiskirstymas
    print("=" * 78)
    print("2 LENTELĖ: klaidų taksonomijos pasiskirstymas")
    print("=" * 78)
    fm_counts = Counter(j.get("failure_mode", "?") for _, j in joined)
    for cat, n in fm_counts.most_common():
        print(f"  {cat:<22}: {n:>5} ({100 * n / n_total:.2f}\\,\\%)")
    print()

    # ── 2. Sistemos metrikos prieš tris etalonus ───────────────────
    print("=" * 78)
    print("3 LENTELĖ: sistemos metrikos prieš tris etalonus (slenkstis 60)")
    print("=" * 78)

    # Etalonas 1: šaltinis (visos poros)
    print("\n--- Prieš ŠALTINIO etiketes (n=visos pateiktos) ---")
    print(f"{'Metodas':<12} {'n':>5} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    src_metrics = {}
    for s in SCENARIOS:
        field = get_score_field(s)
        preds = [p[field] >= THRESHOLD for p, _ in joined]
        labels = [bool(p["ground_truth_label"]) for p, _ in joined]
        m = compute_metrics(preds, labels)
        src_metrics[s] = m
        print(f"{s:<12} {m['n']:>5} {m['accuracy']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}")

    # Etalonas 2: teisėjas (atfiltruojant ambiguous)
    print("\n--- Prieš TEISĖJO etiketes (atfiltruojant ambiguous) ---")
    print(f"{'Metodas':<12} {'n':>5} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    judge_metrics = {}
    for s in SCENARIOS:
        field = get_score_field(s)
        preds = []
        labels = []
        for p, j in joined:
            if j.get("source_assessment") == "ambiguous":
                continue
            if j.get("judge_label") is None:
                continue
            preds.append(p[field] >= THRESHOLD)
            labels.append(bool(j["judge_label"]))
        m = compute_metrics(preds, labels)
        judge_metrics[s] = m
        print(f"{s:<12} {m['n']:>5} {m['accuracy']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}")

    # Etalonas 3: pataisytas etalonas (šaltinis kur correct, teisėjas kur incorrect, atfiltruojant ambiguous)
    print("\n--- Prieš PATAISYTĄ etaloną (šaltinis + teisėjo korekcija) ---")
    print(f"{'Metodas':<12} {'n':>5} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    corr_metrics = {}
    for s in SCENARIOS:
        field = get_score_field(s)
        preds = []
        labels = []
        for p, j in joined:
            sa = j.get("source_assessment")
            if sa == "ambiguous":
                continue
            if sa == "correct":
                lbl = bool(p["ground_truth_label"])
            elif sa == "incorrect":
                if j.get("judge_label") is None:
                    continue
                lbl = bool(j["judge_label"])
            else:
                continue
            preds.append(p[field] >= THRESHOLD)
            labels.append(lbl)
        m = compute_metrics(preds, labels)
        corr_metrics[s] = m
        print(f"{s:<12} {m['n']:>5} {m['accuracy']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}")

    # ── 3. F1 pokyčiai trims etalonams ─────────────────────────────
    print()
    print("=" * 78)
    print("4 LENTELĖ: F1 pagal etaloną — paryškina šaltinio triukšmo poveikį")
    print("=" * 78)
    print(f"{'Metodas':<12} {'F1 šaltinis':>12} {'F1 teisėjas':>12} {'F1 pataisytas':>14}  {'Δ pataisytas-šaltinis':>22}")
    for s in SCENARIOS:
        f_src = src_metrics[s]["f1"]
        f_jud = judge_metrics[s]["f1"]
        f_cor = corr_metrics[s]["f1"]
        delta = f_cor - f_src
        print(f"{s:<12} {f_src:>12.4f} {f_jud:>12.4f} {f_cor:>14.4f}  {delta:>+21.4f}")
    print()

    # ── 4. Cohen kappa: source vs judge, system vs judge ───────────
    print("=" * 78)
    print("5 LENTELĖ: Cohen kappa (susitarimo koeficientai)")
    print("=" * 78)
    # Filtruojam poras be ambiguous
    non_amb = [(p, j) for p, j in joined if j.get("source_assessment") != "ambiguous" and j.get("judge_label") is not None]
    src_lbls = [bool(p["ground_truth_label"]) for p, _ in non_amb]
    jud_lbls = [bool(j["judge_label"]) for _, j in non_amb]
    kappa_sj = cohen_kappa(src_lbls, jud_lbls)
    print(f"  Šaltinis vs Teisėjas:  kappa = {kappa_sj:.4f}  (n={len(non_amb)})")
    print()
    for s in SCENARIOS:
        field = get_score_field(s)
        sys_preds = [p[field] >= THRESHOLD for p, _ in non_amb]
        kappa = cohen_kappa(sys_preds, jud_lbls)
        print(f"  Sistema ({s}) vs Teisėjas: kappa = {kappa:.4f}")
    print()

    # ── 5. Asimetrijos analizė: wrong-rejects vs wrong-accepts ─────
    print("=" * 78)
    print("6 SANTRAUKA: šaltinio etikečių klaidų asimetrija")
    print("=" * 78)
    wrong_rejects = 0  # source=False, judge=True (rinkinys atmetė, bet teisėjas sako match)
    wrong_accepts = 0  # source=True, judge=False (rinkinys priėmė, bet teisėjas sako no-match)
    for p, j in joined:
        if j.get("source_assessment") != "incorrect":
            continue
        src = bool(p["ground_truth_label"])
        jud = j.get("judge_label")
        if jud is None:
            continue
        if not src and jud:
            wrong_rejects += 1
        elif src and not jud:
            wrong_accepts += 1
    total_incorrect = wrong_rejects + wrong_accepts
    if total_incorrect > 0:
        print(f"  wrong-rejects (atmesti, bet turi būti priimti): {wrong_rejects} ({100*wrong_rejects/total_incorrect:.1f}\\,\\%)")
        print(f"  wrong-accepts (priimti, bet turi būti atmesti): {wrong_accepts} ({100*wrong_accepts/total_incorrect:.1f}\\,\\%)")
        print(f"  iš viso neteisingai paženklintų: {total_incorrect}")
    print()


if __name__ == "__main__":
    main()
