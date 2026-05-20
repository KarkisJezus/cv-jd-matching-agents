"""Vienkartinis skriptas: 4.1 poskyrio metrikos pagrindiniam 5000-porų eksperimentui.

Skaičiuoja:
  - Pagrindines klasifikacijos metrikas (jau yra `classification` JSON lauke,
    bet pakartoju, kad būtų aiškios ataskaitos forma);
  - Slenksčio jautrumą (F1 prie 50, 55, 60, 65, 70);
  - McNemar testą poriniams scenarijų palyginimams (A--B, B--C, A--C, baseline--A).

Naudojimas:
    python scripts/compute_4_1_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_PATH = Path("results/tier2_5000_combined/hf_test_5000.json")
SCENARIOS = ["baseline", "A", "B", "C"]
THRESHOLDS = [50, 55, 60, 65, 70]


def confusion_at_threshold(pairs: list[dict], score_field: str, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for p in pairs:
        score = p.get(score_field)
        gt = p.get("ground_truth_label")
        if score is None or gt is None:
            continue
        pred = score >= threshold
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and not gt:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def predictions_at_threshold(pairs: list[dict], score_field: str, threshold: float) -> list[bool | None]:
    out = []
    for p in pairs:
        score = p.get(score_field)
        if score is None:
            out.append(None)
        else:
            out.append(score >= threshold)
    return out


def mcnemar(pairs: list[dict], field_a: str, field_b: str, threshold: float) -> dict:
    """McNemar testas tarp dviejų scenarijų prognozių, palygintų su ground_truth_label.

    Skaičiuoja porų skaičių, kuriose:
      - n01: scenarijus A teisus, B neteisus
      - n10: scenarijus A neteisus, B teisus
    Statistika: chi^2 = (|n01 - n10| - 1)^2 / (n01 + n10)  (su tęstinumo pataisymu)
    """
    n01 = n10 = 0
    for p in pairs:
        gt = p.get("ground_truth_label")
        if gt is None:
            continue
        sa = p.get(field_a)
        sb = p.get(field_b)
        if sa is None or sb is None:
            continue
        pa = sa >= threshold
        pb = sb >= threshold
        a_correct = pa == gt
        b_correct = pb == gt
        if a_correct and not b_correct:
            n01 += 1
        elif not a_correct and b_correct:
            n10 += 1
    discordant = n01 + n10
    chi2 = ((abs(n01 - n10) - 1) ** 2) / discordant if discordant else 0.0
    # Approximate two-sided p-value from chi^2 with 1 dof using normal approximation
    # p = exp(-chi2/2) for chi^2 with 1 dof, but better is via scipy.stats.chi2.sf
    # Without scipy, use a simple lookup for common thresholds:
    p_value = "p<0.001" if chi2 > 10.83 else (
        "p<0.01" if chi2 > 6.63 else (
            "p<0.05" if chi2 > 3.84 else (
                "p<0.1" if chi2 > 2.71 else "n.s."
            )
        )
    )
    return {"n01": n01, "n10": n10, "chi2": chi2, "p_value": p_value}


def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}\\,\\%"


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"Failas nerastas: {RESULTS_PATH}")
        return
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    pairs = data["pair_results"]
    n = len(pairs)
    pos = sum(1 for p in pairs if p.get("ground_truth_label") is True)
    neg = sum(1 for p in pairs if p.get("ground_truth_label") is False)
    threshold = data["config"]["threshold"]

    print(f"Eksperimentas: {data['experiment']}")
    print(f"Modelis: {data['config']['model']}, slenkstis: {threshold}")
    print(f"Iš viso porų: {n} (priimta: {pos}, atmesta: {neg})\n")

    # ── 1. Pagrindinė metrikų lentelė (slenkstis = 60) ─────────────
    print("=" * 78)
    print(f"1 LENTELĖ: pagrindinės klasifikavimo metrikos (slenkstis {threshold})")
    print("=" * 78)
    print(f"{'Metodas':<20} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    for s in SCENARIOS:
        field = f"score_{s}" if s != "baseline" else "score_baseline"
        m = confusion_at_threshold(pairs, field, threshold)
        print(
            f"{s:<20} {m['accuracy']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}  "
            f"{m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}"
        )
    print()

    # ── 2. Slenksčio jautrumas: F1 vs threshold ─────────────────────
    print("=" * 78)
    print("2 LENTELĖ: F1 priklausomybė nuo slenksčio")
    print("=" * 78)
    header = f"{'Metodas':<20}" + "".join(f"{f'F1@{t}':>10}" for t in THRESHOLDS)
    print(header)
    for s in SCENARIOS:
        field = f"score_{s}" if s != "baseline" else "score_baseline"
        row = f"{s:<20}"
        for t in THRESHOLDS:
            m = confusion_at_threshold(pairs, field, t)
            row += f"{m['f1']:>9.4f} "
        print(row)
    print()

    # ── 3. McNemar testas poriniams palyginimams ────────────────────
    print("=" * 78)
    print(f"3 LENTELĖ: McNemar testas (slenkstis {threshold})")
    print("=" * 78)
    pairs_to_test = [
        ("baseline", "A"),
        ("A", "B"),
        ("B", "C"),
        ("A", "C"),
    ]
    print(f"{'Palyginimas':<20} {'n01':>6} {'n10':>6} {'chi^2':>10} {'p_value':>12}")
    for a, b in pairs_to_test:
        fa = f"score_{a}" if a != "baseline" else "score_baseline"
        fb = f"score_{b}" if b != "baseline" else "score_baseline"
        r = mcnemar(pairs, fa, fb, threshold)
        print(f"{a + ' vs ' + b:<20} {r['n01']:>6} {r['n10']:>6} {r['chi2']:>10.4f} {r['p_value']:>12}")
    print()

    # ── 4. F1 deltai (B-A, C-B) ─────────────────────────────────────
    print("=" * 78)
    print("4 SANTRAUKA: F1 progresijos pokyčiai (slenkstis 60)")
    print("=" * 78)
    metrics_at_60 = {}
    for s in SCENARIOS:
        field = f"score_{s}" if s != "baseline" else "score_baseline"
        metrics_at_60[s] = confusion_at_threshold(pairs, field, threshold)
    f1_base = metrics_at_60["baseline"]["f1"]
    f1_a = metrics_at_60["A"]["f1"]
    f1_b = metrics_at_60["B"]["f1"]
    f1_c = metrics_at_60["C"]["f1"]
    print(f"  baseline: {f1_base:.4f}")
    print(f"  A:        {f1_a:.4f}  (Δ vs baseline: {f1_a - f1_base:+.4f})")
    print(f"  B:        {f1_b:.4f}  (Δ vs A:        {f1_b - f1_a:+.4f})")
    print(f"  C:        {f1_c:.4f}  (Δ vs B:        {f1_c - f1_b:+.4f})")
    print()
    if f1_a < f1_b < f1_c:
        print("  → H2 patvirtinta (A < B < C F1 progresija prieš šaltinio etiketes)")
    elif f1_a > f1_b > f1_c:
        print("  → H2 NEPATVIRTINTA: F1 progresija ATVIRKŠTINĖ (A > B > C)")
    else:
        print("  → H2 dalinai patvirtinta arba neaiški (ne monotoniška)")
    print()


if __name__ == "__main__":
    main()
