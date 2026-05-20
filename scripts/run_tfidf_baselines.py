"""Standalone TF-IDF baseline runner pagrindiniam 5000-porų eksperimentui.

Du klasikiniai baseline'ai, lyginami su Sentence-BERT embedding pagrindu ir
agentine architektūra (scenarijai A, B, C):

  1. **TF-IDF panašumas** — kiekvienai porai CV ir JD vektorizuojami atskirai
     per `TfidfVectorizer`, skaičiuojamas kosinusinis panašumas tarp jų,
     pakeliamas į skalę 0-100. Treniravimo nereikia; analogiškas SBERT
     baseline'ui, tik su klasikiniu leksiniu reprezentavimu.

  2. **TF-IDF + Logistic Regression** — sukonkatenuoti CV ir JD vektorizuojami
     kaip vienas dokumentas, paleidžiami per `LogisticRegression` klasifikatorių
     su 5-fold stratified kryžminiu validavimu. Klasifikatoriaus
     `predict_proba(teigiama klasė)` pakeliama į skalę 0-100. Tai stipresnis
     baseline'as: jis turi prieigą prie etikečių (kryžminis validavimas),
     skirtingai nuo embedding pagrindo, kuris yra zero-shot.

Įvedimas:
    `data/hf_test_5000.json` su pair_id, cv_text, jd_text, ground_truth_label.

Išvedimas:
    `results/tier2_5000_combined/hf_test_5000_tfidf.json` — JSON su pair_results
    sąrašu: `pair_id`, `score_tfidf_sim`, `score_tfidf_lr`, `ground_truth_label`.
    Tas pats schema, kaip pagrindinis rezultatų failas, todėl
    `compute_4_1_metrics.py` gali nuskaityti abu lygiagrečiai.

    Be to, atspausdinami accuracy / precision / recall / F1 prie ribos 60.

Naudojimas:
    PYTHONIOENCODING=utf-8 python scripts/run_tfidf_baselines.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path("data/hf_test_5000.json")
OUTPUT_PATH = Path("results/tier2_5000_combined/hf_test_5000_tfidf.json")
THRESHOLD = 60.0
N_FOLDS = 5
RANDOM_SEED = 42


def load_pairs() -> list[dict]:
    """Įkelia 5000 porų iš pagrindinio duomenų failo."""
    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    pairs = raw["pairs"]
    print(f"Įkelta {len(pairs)} porų iš {DATA_PATH}")
    return pairs


def compute_tfidf_similarity(pairs: list[dict]) -> list[float]:
    """1 baseline: kosinusinis panašumas tarp CV ir JD TF-IDF vektorių.

    Vektorizatorius mokomas ant viso CV + JD korpuso (be etikečių), kad TF
    statistika atitiktų visus dokumentus. Tai standartinis informacijos
    paieškos baseline'as: nei kandidatui, nei rezultatui jokio mokymosi.
    """
    print("\n[1/2] TF-IDF panašumas (per porą)...")
    t0 = time.time()

    cv_texts = [p["cv_text"] for p in pairs]
    jd_texts = [p["jd_text"] for p in pairs]
    corpus = cv_texts + jd_texts

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
    )
    vectorizer.fit(corpus)
    cv_mat = vectorizer.transform(cv_texts)
    jd_mat = vectorizer.transform(jd_texts)

    # Element-wise kosinusinis panašumas (kiekvienos eilutės su atitinkama
    # JD eilute, ne kryžminė matrica)
    scores = []
    for i in range(cv_mat.shape[0]):
        sim = float(cosine_similarity(cv_mat[i], jd_mat[i])[0, 0])
        # Skalė 0-100, kaip ir SBERT baseline'as
        scores.append(round(sim * 100, 2))

    print(f"  Vidutinis panašumas: {np.mean(scores):.2f}")
    print(f"  Trukmė: {time.time() - t0:.1f} s")
    return scores


def compute_tfidf_logreg(pairs: list[dict]) -> list[float]:
    """2 baseline: TF-IDF + LogisticRegression su 5-fold stratified CV.

    Kiekviena pora vektorizuojama kaip vienas dokumentas `[CV] [SEP] [JD]`.
    LR klasifikatorius mokomas ant 4 folds, prognozuoja 5-tą — taip kiekvienai
    porai gauname „nepamatyto" verdikto tikimybę. predict_proba(teigiama klasė)
    pakeliamas į skalę 0-100, kad būtų suderinta su agentinės sistemos balais.
    """
    print(f"\n[2/2] TF-IDF + LogisticRegression ({N_FOLDS}-fold CV)...")
    t0 = time.time()

    docs = [f"{p['cv_text']} [SEP] {p['jd_text']}" for p in pairs]
    labels = np.array([1 if p["ground_truth_label"] else 0 for p in pairs])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = np.zeros(len(pairs), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(docs, labels), 1):
        train_docs = [docs[i] for i in train_idx]
        test_docs = [docs[i] for i in test_idx]
        train_y = labels[train_idx]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50_000,
        )
        train_X = vectorizer.fit_transform(train_docs)
        test_X = vectorizer.transform(test_docs)

        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
        clf.fit(train_X, train_y)
        probs = clf.predict_proba(test_X)[:, 1]
        scores[test_idx] = np.round(probs * 100, 2)
        print(f"  Fold {fold_idx}/{N_FOLDS}: n_test={len(test_idx)}, train_pos_frac={train_y.mean():.3f}")

    print(f"  Vidutinis balas: {scores.mean():.2f}")
    print(f"  Trukmė: {time.time() - t0:.1f} s")
    return scores.tolist()


def confusion_at_threshold(
    pair_results: list[dict], score_field: str, threshold: float
) -> dict:
    """Skaičiuoja TP/FP/TN/FN ir accuracy/precision/recall/F1."""
    tp = fp = tn = fn = 0
    for p in pair_results:
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


def main() -> None:
    if not DATA_PATH.exists():
        print(f"FAIL: nerastas duomenų failas {DATA_PATH}")
        sys.exit(1)

    pairs = load_pairs()
    tfidf_sim_scores = compute_tfidf_similarity(pairs)
    tfidf_lr_scores = compute_tfidf_logreg(pairs)

    # Sukurti pair_results JSON tokiu pat formatu, kaip pagrindinis failas
    pair_results = []
    for pair, sim_score, lr_score in zip(pairs, tfidf_sim_scores, tfidf_lr_scores):
        pair_results.append(
            {
                "pair_id": pair["pair_id"],
                "ground_truth_label": pair["ground_truth_label"],
                "score_tfidf_sim": sim_score,
                "score_tfidf_lr": lr_score,
            }
        )

    output_data = {
        "experiment": "tfidf_baselines_5000",
        "config": {
            "n_folds": N_FOLDS,
            "threshold": THRESHOLD,
            "random_seed": RANDOM_SEED,
            "vectorizer": "TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)",
            "classifier": "LogisticRegression(C=1.0, class_weight=balanced, max_iter=1000)",
        },
        "pair_results": pair_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nIšsaugota: {OUTPUT_PATH}")

    # Atspausdinti metrikas
    print("\n" + "=" * 78)
    print(f"METRIKOS prie ribos {THRESHOLD}")
    print("=" * 78)
    print(f"{'Baseline':<25} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    for label, field in [("TF-IDF panašumas", "score_tfidf_sim"), ("TF-IDF + LogReg", "score_tfidf_lr")]:
        m = confusion_at_threshold(pair_results, field, THRESHOLD)
        print(
            f"{label:<25} {m['accuracy']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}  "
            f"{m['tp']:>5} {m['fp']:>5} {m['tn']:>5} {m['fn']:>5}"
        )


if __name__ == "__main__":
    main()
