"""Generate all thesis plots from existing result files.

Outputs PNG files to thesis/images/ for inclusion via \\includegraphics{} in
Sablonas.tex. Lithuanian labels match the thesis text.

Run from repo root:
    python scripts/generate_thesis_plots.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DATA = REPO / "data"
IMAGES = REPO / "thesis" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

THRESHOLD = 60
JUDGE_CONF_THRESHOLD = 0.7
SCENARIOS = ("A", "B", "C")

# Consistent palette
COL_CORRECT = "#2E7D32"   # green
COL_WRONG = "#C62828"     # red
COL_REHAB = "#81C784"     # light green (rehabilitated)
COL_AMBIG = "#9E9E9E"     # gray
COL_BASELINE = "#90A4AE"  # gray-blue
COL_SCEN = {"A": "#1565C0", "B": "#6A1B9A", "C": "#EF6C00"}
COL_MODEL = {"gpt-4o-mini": "#1B5E20", "Qwen 7B": "#7B1FA2", "Embedding pagrindas": "#546E7A"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


# ---------- data loading ----------

def load_main_5000() -> list[dict]:
    with open(RESULTS / "tier2_5000_combined" / "hf_test_5000.json", encoding="utf-8") as f:
        return json.load(f)["pair_results"]


def load_tfidf_5000() -> dict[str, dict]:
    """Įkelia TF-IDF baseline'ų rezultatus, indeksuotus pagal pair_id."""
    path = RESULTS / "tier2_5000_combined" / "hf_test_5000_tfidf.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {p["pair_id"]: p for p in json.load(f)["pair_results"]}


def load_judge_5000() -> dict[str, dict]:
    with open(RESULTS / "judge_hf_test_5000.json", encoding="utf-8") as f:
        return json.load(f)["judgments"]


def load_qwen_1000() -> list[dict]:
    with open(RESULTS / "qwen_1000_A" / "hf_test_1000_qwen.json", encoding="utf-8") as f:
        return json.load(f)["pair_results"]


def load_qwen_dataset() -> dict:
    with open(DATA / "hf_test_1000_qwen.json", encoding="utf-8") as f:
        return json.load(f)


# ---------- metrics ----------

def predict(score: float | None, threshold: float = THRESHOLD) -> bool:
    return score is not None and score >= threshold


def confusion(predictions: list[bool], labels: list[bool]) -> dict[str, int]:
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def f1_from_cm(cm: dict[str, int]) -> float:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def accuracy_from_cm(cm: dict[str, int]) -> float:
    n = sum(cm.values())
    return (cm["tp"] + cm["tn"]) / n if n else 0.0


def judge_label(verdict: dict) -> bool | None:
    """Return True/False for confident judge verdict, None for ambiguous."""
    if verdict.get("error"):
        return None
    if verdict.get("judge_confidence", 0) < JUDGE_CONF_THRESHOLD:
        return None
    return verdict.get("judge_label")


# ---------- Plot 1: Decisions before/after audit ----------

def plot_decisions_before_after_audit(pairs: list[dict], judge: dict[str, dict]) -> None:
    """The main thesis plot — system correctness against source vs against judge,
    plus decomposition of 'wrong against source' into 'truly wrong' vs 'rehabilitated'."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.2))

    for ax, scen in zip(axes, SCENARIOS):
        # gather data
        correct_src, wrong_src = 0, 0
        correct_jud, wrong_jud, ambig = 0, 0, 0
        # subset: "wrong against source"
        rehab, truly_wrong, rehab_ambig = 0, 0, 0

        for p in pairs:
            pid = p["pair_id"]
            score = p.get(f"score_{scen}")
            label_src = p.get("ground_truth_label")
            if score is None or label_src is None:
                continue
            pred = predict(score)
            against_source_correct = (pred == bool(label_src))

            if against_source_correct:
                correct_src += 1
            else:
                wrong_src += 1

            v = judge.get(pid)
            label_jud = judge_label(v) if v else None
            if label_jud is None:
                ambig += 1
                if not against_source_correct:
                    rehab_ambig += 1
            else:
                against_judge_correct = (pred == bool(label_jud))
                if against_judge_correct:
                    correct_jud += 1
                else:
                    wrong_jud += 1
                if not against_source_correct:
                    if against_judge_correct:
                        rehab += 1
                    else:
                        truly_wrong += 1

        total = correct_src + wrong_src

        # Three bars: Source / Judge (excluding ambiguous) / Decomposition of "wrong vs source"
        bar_w = 0.55
        x = np.arange(3)

        # Bar 1 — vs source
        ax.bar(x[0], correct_src, bar_w, color=COL_CORRECT, label="Teisingi")
        ax.bar(x[0], wrong_src, bar_w, bottom=correct_src, color=COL_WRONG, label="Klaidingi")

        # Bar 2 — vs judge (ambiguous shown gray on top)
        ax.bar(x[1], correct_jud, bar_w, color=COL_CORRECT)
        ax.bar(x[1], wrong_jud, bar_w, bottom=correct_jud, color=COL_WRONG)
        ax.bar(x[1], ambig, bar_w, bottom=correct_jud + wrong_jud, color=COL_AMBIG, label="Neaiškūs")

        # Bar 3 — decomposition of "wrong vs source"
        ax.bar(x[2], rehab, bar_w, color=COL_REHAB, label="Rehabilituoti (teisėjas sutinka su sistema)")
        ax.bar(x[2], truly_wrong, bar_w, bottom=rehab, color=COL_WRONG)
        ax.bar(x[2], rehab_ambig, bar_w, bottom=rehab + truly_wrong, color=COL_AMBIG)

        # Annotations
        ax.text(x[0], total + total*0.02, f"{correct_src}\n({100*correct_src/total:.1f}%)",
                ha="center", va="bottom", fontsize=8, color=COL_CORRECT, fontweight="bold")
        ax.text(x[1], total + total*0.02, f"{correct_jud}\n({100*correct_jud/total:.1f}%)",
                ha="center", va="bottom", fontsize=8, color=COL_CORRECT, fontweight="bold")
        if wrong_src:
            ax.text(x[2], total + total*0.02, f"{rehab}/{wrong_src}\n({100*rehab/wrong_src:.1f}%)",
                    ha="center", va="bottom", fontsize=8, color=COL_REHAB, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["Prieš auditą\n(etalonas: šaltinis)",
                            "Po audito\n(etalonas: teisėjas)",
                            "Klaidingų prieš šaltinį\nišskaidymas"], fontsize=8)
        ax.set_ylabel("Porų skaičius")
        ax.set_title(f"Scenarijus {scen}", fontsize=11, fontweight="bold")
        ax.set_ylim(0, total * 1.18)
        ax.grid(axis="y", alpha=0.3)

    # Shared legend below the three panels
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COL_CORRECT, label="Teisingi"),
        Patch(facecolor=COL_WRONG, label="Klaidingi"),
        Patch(facecolor=COL_AMBIG, label="Neaiškūs (teisėjo verdiktas)"),
        Patch(facecolor=COL_REHAB, label="Rehabilituoti (teisėjas sutinka su sistema)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)

    fig.suptitle("Sistemos sprendimų teisingumas prieš ir po LLM-teisėjo audito (n=5000)",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(IMAGES / "sprendimai-pries-po-audito.png")
    plt.close(fig)
    print(f"  ✓ sprendimai-pries-po-audito.png")


# ---------- Plot 2: F1 by scenario × etalon ----------

def plot_f1_by_scenario(
    pairs: list[dict],
    judge: dict[str, dict],
    tfidf: dict[str, dict] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5))

    # --- Embedding (SBERT) baseline ---
    valid_main = [p for p in pairs if p.get("ground_truth_label") is not None]
    base_pred = [predict(p.get("score_baseline")) for p in valid_main]
    labels_src = [bool(p["ground_truth_label"]) for p in valid_main]
    baseline_f1_src = f1_from_cm(confusion(base_pred, labels_src))

    base_preds_jud, base_labs_jud = [], []
    for p in valid_main:
        v = judge.get(p["pair_id"])
        if v is None or p.get("score_baseline") is None:
            continue
        jl = judge_label(v)
        if jl is None:
            continue
        base_preds_jud.append(predict(p.get("score_baseline")))
        base_labs_jud.append(bool(jl))
    baseline_f1_jud = f1_from_cm(confusion(base_preds_jud, base_labs_jud))

    # --- TF-IDF + Logistic Regression baseline (jei prieinamas) ---
    tfidf_f1_src = tfidf_f1_jud = None
    if tfidf:
        tfidf_pred_src, tfidf_lab_src = [], []
        tfidf_pred_jud, tfidf_lab_jud = [], []
        for p in valid_main:
            t = tfidf.get(p["pair_id"])
            if t is None or t.get("score_tfidf_lr") is None:
                continue
            pred = predict(t["score_tfidf_lr"])
            tfidf_pred_src.append(pred)
            tfidf_lab_src.append(bool(p["ground_truth_label"]))
            v = judge.get(p["pair_id"])
            if v is None:
                continue
            jl = judge_label(v)
            if jl is None:
                continue
            tfidf_pred_jud.append(pred)
            tfidf_lab_jud.append(bool(jl))
        tfidf_f1_src = f1_from_cm(confusion(tfidf_pred_src, tfidf_lab_src))
        tfidf_f1_jud = f1_from_cm(confusion(tfidf_pred_jud, tfidf_lab_jud))

    # --- Scenarijų F1 prieš šaltinį ir teisėją ---
    f1_src = {}
    f1_jud = {}
    for scen in SCENARIOS:
        valid = [(p, judge.get(p["pair_id"])) for p in pairs
                 if p.get(f"score_{scen}") is not None and p.get("ground_truth_label") is not None]
        preds_src = [predict(p.get(f"score_{scen}")) for p, _ in valid]
        labs_src = [bool(p["ground_truth_label"]) for p, _ in valid]
        f1_src[scen] = f1_from_cm(confusion(preds_src, labs_src))

        preds_jud, labs_jud = [], []
        for p, v in valid:
            if v is None:
                continue
            jl = judge_label(v)
            if jl is None:
                continue
            preds_jud.append(predict(p.get(f"score_{scen}")))
            labs_jud.append(bool(jl))
        f1_jud[scen] = f1_from_cm(confusion(preds_jud, labs_jud))

    # --- Grupuotos juostos ---
    labels: list[str] = []
    src_vals: list[float] = []
    jud_vals: list[float] = []

    if tfidf_f1_src is not None:
        labels.append("TF-IDF +\nLogReg")
        src_vals.append(tfidf_f1_src)
        jud_vals.append(tfidf_f1_jud if tfidf_f1_jud is not None else 0.0)

    labels.append("Embedding\npagrindas")
    src_vals.append(baseline_f1_src)
    jud_vals.append(baseline_f1_jud)

    for s in SCENARIOS:
        labels.append(f"Scenarijus {s}")
        src_vals.append(f1_src[s])
        jud_vals.append(f1_jud[s])

    x = np.arange(len(labels))
    w = 0.38

    b1 = ax.bar(x - w/2, src_vals, w, label="Prieš šaltinio etiketę", color="#1976D2")
    b2 = ax.bar(x + w/2, jud_vals, w, label="Prieš teisėjo verdiktą", color="#F57C00")

    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.01,
                    f"{r.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 įvertis")
    ax.set_title("F1 įvertis pagal metodą ir atskaitos etaloną (n=5000, riba 60)", fontweight="bold")
    ax.set_ylim(0, max(max(src_vals), max(jud_vals)) * 1.18)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(IMAGES / "scenarijai-f1.png")
    plt.close(fig)
    print(f"  ✓ scenarijai-f1.png")
    return f1_src, f1_jud


# ---------- Plot 3: Confusion matrices panel ----------

def plot_confusion_panel(pairs: list[dict], judge: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    for col, scen in enumerate(SCENARIOS):
        # Row 0 — vs source
        preds_src, labs_src = [], []
        for p in pairs:
            if p.get(f"score_{scen}") is None or p.get("ground_truth_label") is None:
                continue
            preds_src.append(predict(p[f"score_{scen}"]))
            labs_src.append(bool(p["ground_truth_label"]))
        cm = confusion(preds_src, labs_src)
        _draw_cm(axes[0, col], cm, f"Scenarijus {scen}\nprieš šaltinio etaloną", cmap="Blues")

        # Row 1 — vs judge
        preds_jud, labs_jud = [], []
        for p in pairs:
            v = judge.get(p["pair_id"])
            if v is None or p.get(f"score_{scen}") is None:
                continue
            jl = judge_label(v)
            if jl is None:
                continue
            preds_jud.append(predict(p[f"score_{scen}"]))
            labs_jud.append(bool(jl))
        cm = confusion(preds_jud, labs_jud)
        _draw_cm(axes[1, col], cm, f"Scenarijus {scen}\nprieš teisėjo etaloną", cmap="Oranges")

    fig.suptitle("Sumaišties matricos: scenarijai × atskaitos etalonai (n=5000)",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout()
    fig.savefig(IMAGES / "sumaisties-matricos.png")
    plt.close(fig)
    print(f"  ✓ sumaisties-matricos.png")


def _draw_cm(ax, cm: dict[str, int], title: str, cmap: str = "Blues") -> None:
    mat = np.array([[cm["tn"], cm["fp"]],
                    [cm["fn"], cm["tp"]]])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max())
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: neatitinka", "Pred: atitinka"], fontsize=8)
    ax.set_yticklabels(["Tikra: neatitinka", "Tikra: atitinka"], fontsize=8)
    ax.set_title(title, fontsize=10)
    # Annotate
    thresh = mat.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if mat[i, j] > thresh else "black"
            ax.text(j, i, f"{mat[i, j]}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")


# ---------- Plot 4: Threshold sensitivity ----------

def plot_threshold_sensitivity(pairs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    thresholds = list(range(20, 95, 5))
    for scen in SCENARIOS:
        f1s = []
        for thr in thresholds:
            preds, labs = [], []
            for p in pairs:
                if p.get(f"score_{scen}") is None or p.get("ground_truth_label") is None:
                    continue
                preds.append(p[f"score_{scen}"] >= thr)
                labs.append(bool(p["ground_truth_label"]))
            f1s.append(f1_from_cm(confusion(preds, labs)))
        ax.plot(thresholds, f1s, marker="o", label=f"Scenarijus {scen}",
                color=COL_SCEN[scen], linewidth=2)

    # Highlight 60 threshold
    ax.axvline(60, color="black", linestyle="--", alpha=0.4, label="Naudota riba (60)")

    ax.set_xlabel("Sprendimo riba")
    ax.set_ylabel("F1 įvertis (prieš šaltinio etiketę)")
    ax.set_title("Ribos jautrumas: F1 priklausomybė nuo sprendimo ribos (n=5000)", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.7)

    plt.tight_layout()
    fig.savefig(IMAGES / "ribos-jautrumas.png")
    plt.close(fig)
    print(f"  ✓ ribos-jautrumas.png")


# ---------- Plot 5: Error taxonomy ----------

def plot_error_taxonomy(judge: dict[str, dict]) -> None:
    from collections import Counter
    fm_counts = Counter()
    for v in judge.values():
        fm = v.get("failure_mode")
        if fm:
            fm_counts[fm] += 1
    if not fm_counts:
        return

    fm_lt = {
        "good_match": "Sėkmingi sutapimai",
        "domain_mismatch": "Sritinis nesutapimas",
        "templated_reject": "Šabloninis atmetimas",
        "skill_mismatch": "Įgūdžių nesutapimas",
        "categorical_mismatch": "Vaidmens kategorijos nesutapimas",
        "ambiguous_jd": "Neaiškus skelbimas",
        "experience_gap": "Patirties spraga",
        "seniority_mismatch": "Vyresniškumo nesutapimas",
        "underqualified": "Per žema kvalifikacija",
        "other": "Kita",
    }
    items = fm_counts.most_common()
    labels = [fm_lt.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars = ax.barh(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + total*0.005, b.get_y() + b.get_height()/2,
                f"{v} ({100*v/total:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Porų skaičius")
    ax.set_title(f"Klaidų taksonomijos pasiskirstymas (LLM-teisėjas, n={total})", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(values) * 1.18)

    plt.tight_layout()
    fig.savefig(IMAGES / "klaidu-taksonomija.png")
    plt.close(fig)
    print(f"  ✓ klaidu-taksonomija.png")


# ---------- Plot 6: Cross-model F1 ----------

def plot_cross_model(pairs_gpt: list[dict], pairs_qwen: list[dict]) -> None:
    qwen_ids = {p["pair_id"] for p in pairs_qwen}
    gpt_by_id = {p["pair_id"]: p for p in pairs_gpt if p["pair_id"] in qwen_ids}

    labels, gpt_preds, qwen_preds, base_preds = [], [], [], []
    for p in pairs_qwen:
        pid = p["pair_id"]
        if pid not in gpt_by_id:
            continue
        g = gpt_by_id[pid]
        lab = p.get("ground_truth_label")
        if lab is None or g.get("score_A") is None or p.get("score_A") is None:
            continue
        labels.append(bool(lab))
        gpt_preds.append(predict(g["score_A"]))
        qwen_preds.append(predict(p["score_A"]))
        base_preds.append(predict(p.get("score_baseline")))

    base_cm = confusion(base_preds, labels)
    gpt_cm = confusion(gpt_preds, labels)
    qwen_cm = confusion(qwen_preds, labels)

    names = ["Embedding\npagrindas", "gpt-4o-mini\nScen. A", "Qwen 7B\nScen. A"]
    f1s = [f1_from_cm(base_cm), f1_from_cm(gpt_cm), f1_from_cm(qwen_cm)]
    accs = [accuracy_from_cm(base_cm), accuracy_from_cm(gpt_cm), accuracy_from_cm(qwen_cm)]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    w = 0.38
    b1 = ax.bar(x - w/2, f1s, w, label="F1 įvertis", color="#1976D2")
    b2 = ax.bar(x + w/2, accs, w, label="Tikslumas", color="#388E3C")
    for bars, vals in [(b1, f1s), (b2, accs)]:
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Įvertis")
    ax.set_title(f"Tarp-modelinis palyginimas, scenarijus A (n={len(labels)})", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.75)

    plt.tight_layout()
    fig.savefig(IMAGES / "tarp-modelinis.png")
    plt.close(fig)
    print(f"  ✓ tarp-modelinis.png")


# ---------- Plot 7: JD length histogram ----------

def plot_jd_length() -> None:
    data = load_qwen_dataset()
    lens = [len(p.get("jd_text", "").split()) for p in data["pairs"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(lens, bins=60, color="#1976D2", edgecolor="black", alpha=0.8)
    median = sorted(lens)[len(lens)//2]
    ax.axvline(median, color="red", linestyle="--", linewidth=1.5,
               label=f"Mediana: {median} žodžių")
    ax.axvspan(100, 300, alpha=0.1, color="green",
               label="Realių industrinių skelbimų ilgis")
    ax.set_xlabel("JD ilgis (žodžiais)")
    ax.set_ylabel("Porų skaičius")
    ax.set_title(f"Darbo skelbimų ilgio pasiskirstymas (n={len(lens)})", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(IMAGES / "jd-ilgio-pasiskirstymas.png")
    plt.close(fig)
    print(f"  ✓ jd-ilgio-pasiskirstymas.png")


# ---------- Plot 8a: Model agreement matrix (Qwen vs gpt-4o-mini) ----------

def plot_model_agreement_matrix(pairs_gpt: list[dict], pairs_qwen: list[dict]) -> None:
    """2x2 agreement matrix between Qwen and gpt-4o-mini predictions (Scenario A)."""
    qwen_by_id = {p["pair_id"]: p for p in pairs_qwen}

    both_pos = 0  # both predict atitinka
    both_neg = 0  # both predict neatitinka
    only_qwen_pos = 0  # only Qwen positive
    only_gpt_pos = 0  # only gpt positive

    for g in pairs_gpt:
        q = qwen_by_id.get(g["pair_id"])
        if q is None or g.get("score_A") is None or q.get("score_A") is None:
            continue
        gpt_pos = g["score_A"] >= THRESHOLD
        qwen_pos = q["score_A"] >= THRESHOLD
        if gpt_pos and qwen_pos:
            both_pos += 1
        elif gpt_pos and not qwen_pos:
            only_gpt_pos += 1
        elif not gpt_pos and qwen_pos:
            only_qwen_pos += 1
        else:
            both_neg += 1

    total = both_pos + both_neg + only_qwen_pos + only_gpt_pos
    agree = both_pos + both_neg

    # Matrix layout:
    # rows: gpt (atitinka, neatitinka)
    # cols: Qwen (atitinka, neatitinka)
    mat = np.array([
        [both_pos, only_gpt_pos],
        [only_qwen_pos, both_neg],
    ])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.05)

    # Axis ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Atitinka", "Neatitinka"], fontsize=11)
    ax.set_yticklabels(["Atitinka", "Neatitinka"], fontsize=11)
    ax.set_xlabel("Qwen 7B prognozė", fontsize=12, fontweight="bold")
    ax.set_ylabel("gpt-4o-mini prognozė", fontsize=12, fontweight="bold")

    # Cell annotations
    for i in range(2):
        for j in range(2):
            count = mat[i, j]
            pct = 100 * count / total
            colour = "white" if count > mat.max() * 0.55 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f} %)",
                    ha="center", va="center", color=colour,
                    fontsize=14, fontweight="bold")

    # Title with key headline
    ax.set_title(
        f"Modelių sutarimo matrica, scenarijus A (n={total})\n"
        f"Sutaria {agree}/{total} porose ({100*agree/total:.1f} %)",
        fontsize=11, fontweight="bold", pad=12,
    )

    # Visual hints for agreement vs disagreement quadrants
    # Diagonal cells get a thin border to highlight "agreement"
    for i, j in [(0, 0), (1, 1)]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="#2E7D32", linewidth=2.5))
    for i, j in [(0, 1), (1, 0)]:
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="#C62828", linewidth=2.5))

    # Legend explaining the colored frames
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(edgecolor="#2E7D32", facecolor="none", linewidth=2.5, label="Sutaria"),
        Patch(edgecolor="#C62828", facecolor="none", linewidth=2.5, label="Nesutaria"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=10, frameon=False)

    plt.tight_layout()
    fig.savefig(IMAGES / "modeliu-sutarimo-matrica.png")
    plt.close(fig)
    print(f"  ✓ modeliu-sutarimo-matrica.png")


# ---------- Plot 9: Cohen kappa heatmap ----------

def plot_kappa_heatmap(pairs: list[dict], judge: dict[str, dict]) -> None:
    from itertools import combinations

    # Build matched arrays: source, judge (excluding ambiguous), system A/B/C
    rows = []
    for p in pairs:
        lab_src = p.get("ground_truth_label")
        if lab_src is None:
            continue
        v = judge.get(p["pair_id"])
        if v is None:
            continue
        lab_jud = judge_label(v)
        if lab_jud is None:
            continue
        if any(p.get(f"score_{s}") is None for s in SCENARIOS):
            continue
        rows.append({
            "source": bool(lab_src),
            "judge": bool(lab_jud),
            "A": predict(p["score_A"]),
            "B": predict(p["score_B"]),
            "C": predict(p["score_C"]),
        })

    def kappa(xs, ys):
        n = len(xs)
        po = sum(1 for a, b in zip(xs, ys) if a == b) / n
        pa = sum(xs) / n; pb = sum(ys) / n
        pe = pa*pb + (1-pa)*(1-pb)
        return (po - pe) / (1 - pe) if pe < 1 else 0

    keys = ["source", "judge", "A", "B", "C"]
    labels = ["Šaltinis", "Teisėjas", "Sist. A", "Sist. B", "Sist. C"]
    mat = np.zeros((5, 5))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i == j:
                mat[i, j] = 1.0
            else:
                mat[i, j] = kappa([r[k1] for r in rows], [r[k2] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            color = "black" if abs(mat[i,j]) < 0.6 else "white"
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
    ax.set_title(f"Cohen κ susitarimo koeficientai (n={len(rows)})", fontweight="bold")
    plt.colorbar(im, ax=ax, label="κ", shrink=0.8)
    plt.tight_layout()
    fig.savefig(IMAGES / "kappa-heatmap.png")
    plt.close(fig)
    print(f"  ✓ kappa-heatmap.png")


# ---------- main ----------

def main() -> None:
    print("Generuoju thesis grafikus į", IMAGES)
    print()
    print("Įkeliami duomenys…")
    pairs = load_main_5000()
    judge = load_judge_5000()
    pairs_qwen = load_qwen_1000()
    tfidf = load_tfidf_5000()
    print(f"  • Pagrindinis 5000 ({len(pairs)} porų)")
    print(f"  • Teisėjo audito verdiktai ({len(judge)})")
    print(f"  • Qwen 1000 ({len(pairs_qwen)})")
    if tfidf:
        print(f"  • TF-IDF baseline ({len(tfidf)} porų)")
    else:
        print("  • TF-IDF baseline — nerasta (paleiskit scripts/run_tfidf_baselines.py)")
    print()
    print("Generuojami grafikai…")
    plot_decisions_before_after_audit(pairs, judge)
    plot_f1_by_scenario(pairs, judge, tfidf=tfidf)
    plot_confusion_panel(pairs, judge)
    plot_threshold_sensitivity(pairs)
    plot_error_taxonomy(judge)
    plot_cross_model(pairs, pairs_qwen)
    # plot_jd_length()  # išjungta — figūra pašalinta iš tezės kaip neinformatyvi
    plot_model_agreement_matrix(pairs, pairs_qwen)
    plot_kappa_heatmap(pairs, judge)
    print()
    print("✓ Visi grafikai sugeneruoti į", IMAGES)


if __name__ == "__main__":
    main()
