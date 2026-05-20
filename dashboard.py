"""
Streamlit dashboard for the CV-JD Matching System.

Six tabs: Overview, Classification, Errors, Architecture, Pair Inspector,
Compare Runs. Each tab has a single purpose; helpers at the top of the
file are shared across tabs to keep individual sections small.

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Import pandas before plotly: plotly express uses pandas internally and a
# circular-import error surfaces if plotly initializes first under streamlit's
# hot-reload.
import pandas as pd  # noqa: F401
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Page config + constants
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CV-JD Match Dashboard", page_icon="🎯", layout="wide")

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

SCENARIO_COLORS = {
    "A": "#3498db",
    "B": "#2ecc71",
    "C": "#9b59b6",
    "baseline": "#95a5a6",
}
RECOMMENDATION_COLORS = {
    "strong_match": "#2ecc71",
    "good_match": "#27ae60",
    "partial_match": "#f39c12",
    "weak_match": "#e67e22",
    "no_match": "#e74c3c",
    "error": "#7f8c8d",
}

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_judge_results(dataset_stem: str) -> dict:
    """Load LLM-as-Judge audit results for a dataset, if available.

    Convention: audit.py writes to results/judge_<dataset_stem>.json.
    Returns {} if no judge file exists for this dataset.
    """
    candidates = [dataset_stem]
    parts = dataset_stem.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidates.append("_".join(parts[:i]))
    seen = set()
    for c in candidates:
        if c in seen: continue
        seen.add(c)
        path = RESULTS_DIR / f"judge_{c}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def find_judge_files() -> list[Path]:
    """Find all results/judge_*.json files for the Compare-Audits feature."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        [f for f in RESULTS_DIR.glob("judge_*.json")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )


@st.cache_data(show_spinner=False)
def load_tfidf_baselines(selected_file_stem: str, parent_dir: Path) -> dict[str, dict]:
    """Užkrauna TF-IDF baseline'ų rezultatus (panašumas + LogReg) jeigu jie yra.

    Konvencija: scripts/run_tfidf_baselines.py rašo į tą patį katalogą,
    kuriame yra pagrindinis rezultatų failas, su priesaga `_tfidf.json`.
    Grąžina dict, indeksuotą pagal pair_id.
    """
    path = parent_dir / f"{selected_file_stem}_tfidf.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {p["pair_id"]: p for p in data.get("pair_results", [])}


@st.cache_data(show_spinner=False)
def load_details(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def load_source_dataset(stem: str) -> dict[str, dict]:
    """Map pair_id → full source pair record (for CV/JD text in Pair Inspector).

    Tries the literal stem first, then progressively strips common suffixes
    that result files often have on top of their underlying dataset name
    (e.g. "hf_test_150_augmented_partial" → "hf_test_150_augmented").
    """
    candidates = [stem]
    # Progressive suffix stripping: peel one trailing _word at a time
    parts = stem.split("_")
    for i in range(len(parts) - 1, 0, -1):
        candidates.append("_".join(parts[:i]))
    seen = set()
    for c in candidates:
        if c in seen: continue
        seen.add(c)
        path = DATA_DIR / f"{c}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            pairs = raw if isinstance(raw, list) else raw.get("pairs", [])
            return {p["pair_id"]: p for p in pairs if "pair_id" in p}
    return {}


def find_result_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    excluded = {"memories.json", "labeled_memories.json"}
    return sorted(
        [
            f for f in RESULTS_DIR.rglob("*.json")
            if "_traces" not in f.name
            and "_details" not in f.name
            and not f.name.startswith("judge_")  # judge files have a different schema
            and f.name not in excluded
            and not any("memory" in part.lower() for part in f.relative_to(RESULTS_DIR).parts[:-1])
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def format_result_file_label(path: Path) -> str:
    try:
        rel = path.relative_to(RESULTS_DIR)
    except ValueError:
        return path.name
    parts = rel.parts
    if len(parts) == 1:
        return parts[0]
    return f"[{' / '.join(parts[:-1])}] {parts[-1]}"


def detect_provider(model: str) -> tuple[str, str]:
    if not model:
        return ("unknown", "#7f8c8d")
    m = model.lower()
    if any(m.startswith(p) for p in ("qwen", "llama", "mistral", "gemma", "phi", "deepseek")):
        return (f"Local ({model})", "#9b59b6")
    if m.startswith("gpt"):
        return (f"OpenAI ({model})", "#27ae60")
    if m.startswith("claude"):
        return (f"Anthropic ({model})", "#e67e22")
    if m == "mock":
        return ("Mock LLM (development)", "#95a5a6")
    return (model, "#3498db")


# ---------------------------------------------------------------------------
# Computations (pure functions over loaded data)
# ---------------------------------------------------------------------------

def scenarios_present(pair_results: list[dict]) -> list[str]:
    return [s for s in ("A", "B", "C") if any(f"score_{s}" in p for p in pair_results)]


def predictions_for(pair_results: list[dict], scenario: str, threshold: float) -> dict[str, dict]:
    """For each pair, compute (score, predicted_label, correct?) for one scenario."""
    out = {}
    for p in pair_results:
        score = p.get(f"score_{scenario}") if scenario != "baseline" else p.get("score_baseline")
        gt = p.get("ground_truth_label")
        if score is None or gt is None:
            continue
        pred = score >= threshold
        out[p["pair_id"]] = {
            "score": score,
            "predicted": pred,
            "ground_truth": gt,
            "correct": pred == gt,
            "error_dir": _error_direction(pred, gt),
        }
    return out


def _error_direction(pred: bool, gt: bool) -> str:
    if pred and gt: return "TP"
    if pred and not gt: return "FP"
    if not pred and gt: return "FN"
    return "TN"


def threshold_sweep(pair_results: list[dict], scenario: str, lo: int = 0, hi: int = 100, step: int = 5) -> list[dict]:
    """Compute accuracy / precision / recall / F1 across a threshold range."""
    scores_labels = []
    for p in pair_results:
        s = p.get(f"score_{scenario}") if scenario != "baseline" else p.get("score_baseline")
        gt = p.get("ground_truth_label")
        if s is None or gt is None:
            continue
        scores_labels.append((s, gt))
    rows = []
    for t in range(lo, hi + 1, step):
        tp = fp = tn = fn = 0
        for s, gt in scores_labels:
            pred = s >= t
            if pred and gt: tp += 1
            elif pred and not gt: fp += 1
            elif not pred and gt: fn += 1
            else: tn += 1
        n = tp + fp + tn + fn
        if n == 0: continue
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"threshold": t, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    return rows


def disagreement_matrix(pair_results: list[dict], scenarios: list[str], threshold: float) -> dict:
    """For each pair of scenarios, count agreements and disagreements."""
    preds = {sc: predictions_for(pair_results, sc, threshold) for sc in scenarios}
    matrix = {}
    for i, sa in enumerate(scenarios):
        for sb in scenarios[i + 1:]:
            common = set(preds[sa]) & set(preds[sb])
            agree = sum(1 for pid in common if preds[sa][pid]["predicted"] == preds[sb][pid]["predicted"])
            matrix[f"{sa} vs {sb}"] = {
                "n": len(common),
                "agree": agree,
                "disagree": len(common) - agree,
                "agreement_rate": agree / len(common) if common else 0.0,
            }
    return matrix


def hard_easy_pairs(pair_results: list[dict], scenarios: list[str], threshold: float) -> tuple[list[str], list[str]]:
    """Pairs all scenarios got wrong (hard) vs all right (easy)."""
    if not scenarios:
        return [], []
    preds = {sc: predictions_for(pair_results, sc, threshold) for sc in scenarios}
    hard, easy = [], []
    common = set.intersection(*(set(preds[sc]) for sc in scenarios)) if preds else set()
    for pid in common:
        results = [preds[sc][pid]["correct"] for sc in scenarios]
        if not any(results):
            hard.append(pid)
        elif all(results):
            easy.append(pid)
    return hard, easy


def role_accuracy(pair_results: list[dict], scenarios: list[str], threshold: float) -> dict:
    """Per-role accuracy for each scenario, derived from pair_id prefix."""
    role_results = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for p in pair_results:
        pid = p["pair_id"]
        # pair_id is "<role>_<n>" — split on last underscore
        role = "_".join(pid.split("_")[:-1]) or "unknown"
        gt = p.get("ground_truth_label")
        if gt is None: continue
        for sc in scenarios:
            score = p.get(f"score_{sc}")
            if score is None: continue
            pred = score >= threshold
            role_results[role][sc]["total"] += 1
            if pred == gt:
                role_results[role][sc]["correct"] += 1
    return {r: {sc: (v[sc]["correct"] / v[sc]["total"] if v[sc]["total"] else None, v[sc]["total"])
                for sc in scenarios}
            for r, v in role_results.items()}


def calibration_impact(traces: list[dict]) -> dict:
    """Aggregate Tier 2 Pass-1 vs Pass-2 outcomes (Scenario C only)."""
    decisions = Counter()
    pairs = []
    for tr in traces:
        if tr.get("scenario") != "C":
            continue
        t2 = tr.get("tier2") or {}
        decision = t2.get("calibration_decision") or ""
        if not decision:
            continue
        decisions[decision] += 1
        pairs.append({
            "pair_id": tr.get("pair_id"),
            "initial": t2.get("initial_score"),
            "calibrated": tr.get("decision", {}).get("final_score"),
            "decision": decision,
            "adjustment": t2.get("calibration_adjustment", 0),
            "n_supporting": t2.get("calibration_n_supporting", 0),
            "rationale": t2.get("calibration_rationale", ""),
            "pattern": t2.get("calibration_pattern", ""),
        })
    return {"decisions": decisions, "pairs": pairs}


def esco_classification_stats(traces: list[dict]) -> dict:
    """Tier 2 JDProfilingAgent role-classification breakdown."""
    role_counts = Counter()
    confidences = []
    fallback = 0
    total = 0
    for tr in traces:
        # only need one scenario per pair to avoid triple counting; pick B if present, else A
        if tr.get("scenario") not in ("B", "C"):
            continue
        t2 = tr.get("tier2") or {}
        jd = t2.get("jd_profile") or {}
        role = jd.get("detected_role") or ""
        if not role:
            continue
        total += 1
        role_counts[role] += 1
        conf = jd.get("role_confidence")
        if conf is not None:
            confidences.append(float(conf))
        if role == "generic_professional":
            fallback += 1
    return {
        "total": total,
        "fallback": fallback,
        "fallback_rate": fallback / total if total else 0.0,
        "role_counts": role_counts,
        "confidences": confidences,
    }


def headline_text(data: dict, threshold: float) -> str:
    """One-sentence narrative about the run's strongest signal."""
    cls = data.get("classification", {})
    pair_results = data.get("pair_results", [])
    scenarios = scenarios_present(pair_results)
    if not cls or not scenarios:
        return f"{len(pair_results)} pairs evaluated. No classification metrics available."
    accs = [(sc, cls.get(sc, {}).get("accuracy", 0.0)) for sc in scenarios]
    best_sc, best_acc = max(accs, key=lambda x: x[1])
    worst_sc, worst_acc = min(accs, key=lambda x: x[1])
    spread = (best_acc - worst_acc) * 100
    base_acc = cls.get("baseline", {}).get("accuracy")
    base_str = f" (baseline: {base_acc:.0%})" if base_acc is not None else ""
    if len(scenarios) > 1 and spread > 1:
        return (
            f"**Scenario {best_sc}** leads at {best_acc:.0%} accuracy on {len(pair_results)} pairs"
            f"{base_str}; spread across A/B/C is {spread:.1f}pp (worst: {worst_sc} at {worst_acc:.0%})."
        )
    return (
        f"Scenario {best_sc} reached {best_acc:.0%} accuracy on {len(pair_results)} pairs{base_str}."
    )


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def render_header_strip(config: dict, n_pairs: int) -> None:
    """Top-of-page run-context strip used on every tab."""
    model = config.get("model", "unknown")
    provider, color = detect_provider(model)
    arch = config.get("architecture", "tier1")
    threshold = config.get("threshold", 60)
    smm = config.get("streaming_memory_mode") or config.get("memory_mode") or "n/a"
    cols = st.columns([3, 1, 1, 1, 1])
    cols[0].markdown(
        f"<div style='padding:8px 12px;border-radius:6px;background:{color};color:white;font-weight:bold'>"
        f"Model: {provider}</div>",
        unsafe_allow_html=True,
    )
    cols[1].metric("Architecture", arch)
    cols[2].metric("Threshold", f"{threshold}")
    cols[3].metric("Pairs", f"{n_pairs}")
    cols[4].metric("Memory", smm)


def confusion_matrix_figure(cm: dict, title: str, color: str = "Blues") -> go.Figure:
    matrix = [
        [cm.get("true_positives", 0), cm.get("false_negatives", 0)],
        [cm.get("false_positives", 0), cm.get("true_negatives", 0)],
    ]
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=["Pred. Match", "Pred. No-Match"],
        y=["Actual Match", "Actual No-Match"],
        text=[[str(v) for v in row] for row in matrix],
        texttemplate="%{text}",
        colorscale=color,
        showscale=False,
    ))
    fig.update_layout(title=title, height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def score_scatter_chart(pair_results: list[dict], scenarios: list[str], threshold: float) -> go.Figure:
    """Scatter: pair index x score, colored by scenario, threshold horizontal line."""
    rows = []
    for idx, p in enumerate(pair_results, 1):
        for sc in scenarios:
            s = p.get(f"score_{sc}")
            if s is None: continue
            rows.append({
                "Pair #": idx,
                "Pair": p["pair_id"],
                "Scenario": f"Scenario {sc}",
                "Score": s,
                "GT": "Match" if p.get("ground_truth_label") else "No-Match",
            })
    fig = px.scatter(
        rows, x="Pair #", y="Score", color="Scenario", symbol="GT",
        hover_data=["Pair"],
        color_discrete_map={f"Scenario {s}": SCENARIO_COLORS[s] for s in scenarios},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"threshold={threshold}", annotation_position="right")
    fig.update_layout(yaxis_range=[0, 100], height=420, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def score_distribution_chart(pair_results: list[dict], scenarios: list[str]) -> go.Figure:
    """Box plot of score by scenario, faceted by ground-truth label."""
    rows = []
    for p in pair_results:
        gt_label = p.get("ground_truth_label")
        if gt_label is None: continue
        gt_str = "Match" if gt_label else "No-Match"
        for sc in scenarios:
            s = p.get(f"score_{sc}")
            if s is None: continue
            rows.append({"Scenario": f"Scenario {sc}", "Score": s, "Ground Truth": gt_str})
    fig = px.box(
        rows, x="Ground Truth", y="Score", color="Scenario", points="all",
        color_discrete_map={f"Scenario {s}": SCENARIO_COLORS[s] for s in scenarios},
    )
    fig.update_layout(yaxis_range=[0, 100], height=400, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def threshold_sweep_chart(pair_results: list[dict], scenarios: list[str]) -> go.Figure:
    rows = []
    for sc in scenarios:
        for r in threshold_sweep(pair_results, sc, lo=0, hi=100, step=5):
            rows.append({"Scenario": f"Scenario {sc}", **r})
    fig = px.line(
        rows, x="threshold", y="f1", color="Scenario", markers=True,
        color_discrete_map={f"Scenario {s}": SCENARIO_COLORS[s] for s in scenarios},
        title="F1 vs threshold (each line peaks at the optimal threshold for that scenario)",
    )
    fig.update_layout(yaxis_range=[0, 1.05], yaxis_tickformat=".0%", height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def reliability_diagram(pair_results: list[dict], scenarios: list[str], threshold: float) -> go.Figure:
    """Bin score range, plot empirical match rate per bin per scenario."""
    bins = [(i, i + 10) for i in range(0, 100, 10)]
    rows = []
    for sc in scenarios:
        for lo, hi in bins:
            in_bin = []
            for p in pair_results:
                s = p.get(f"score_{sc}")
                gt = p.get("ground_truth_label")
                if s is None or gt is None: continue
                if lo <= s < hi or (hi == 100 and s == 100):
                    in_bin.append(int(gt))
            if not in_bin: continue
            rows.append({
                "Scenario": f"Scenario {sc}",
                "Score bin": f"{lo}-{hi}",
                "Bin midpoint": (lo + hi) / 2,
                "Empirical match rate": sum(in_bin) / len(in_bin),
                "n": len(in_bin),
            })
    fig = px.line(
        rows, x="Bin midpoint", y="Empirical match rate", color="Scenario", markers=True,
        hover_data=["n", "Score bin"],
        color_discrete_map={f"Scenario {s}": SCENARIO_COLORS[s] for s in scenarios},
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=1.0,
                  line=dict(dash="dash", color="gray"))
    fig.add_annotation(x=80, y=0.85, text="perfect calibration", showarrow=False, font=dict(color="gray"))
    fig.update_layout(yaxis_range=[0, 1.05], yaxis_tickformat=".0%", height=400, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def ece_per_scenario(pair_results: list[dict], scenarios: list[str]) -> dict[str, float]:
    """Expected Calibration Error: weighted mean |confidence - accuracy| per bin."""
    out = {}
    for sc in scenarios:
        bins = defaultdict(lambda: {"n": 0, "correct": 0, "conf_sum": 0.0})
        threshold_local = 60
        for p in pair_results:
            score = p.get(f"score_{sc}")
            conf = p.get(f"confidence_{sc}")
            gt = p.get("ground_truth_label")
            if score is None or conf is None or gt is None: continue
            pred = score >= threshold_local
            bin_idx = min(int(conf * 10), 9)
            bins[bin_idx]["n"] += 1
            bins[bin_idx]["conf_sum"] += conf
            if pred == gt:
                bins[bin_idx]["correct"] += 1
        total = sum(b["n"] for b in bins.values())
        if total == 0:
            out[sc] = 0.0
            continue
        ece = 0.0
        for b in bins.values():
            if b["n"] == 0: continue
            mean_conf = b["conf_sum"] / b["n"]
            acc = b["correct"] / b["n"]
            ece += (b["n"] / total) * abs(mean_conf - acc)
        out[sc] = ece
    return out


# ---------------------------------------------------------------------------
# Judge rendering helpers (used by Pair Inspector)
# ---------------------------------------------------------------------------

FAILURE_MODE_COLORS = {
    "good_match": "#27ae60",
    "templated_reject": "#e67e22",
    "domain_mismatch": "#e74c3c",
    "skill_mismatch": "#c0392b",
    "experience_gap": "#d35400",
    "ambiguous_jd": "#95a5a6",
    "categorical_mismatch": "#8e44ad",
    "seniority_mismatch": "#9b59b6",
    "other": "#7f8c8d",
}

SOURCE_ASSESSMENT_COLORS = {
    "correct": "#27ae60",
    "incorrect": "#e74c3c",
    "ambiguous": "#95a5a6",
}


def _render_judge_block_for_pair(
    judgment: dict, pair: dict, pair_id: str, threshold: float, scenarios: list[str],
) -> None:
    """Render the per-pair Auditor judgment block in Pair Inspector."""
    if judgment.get("error"):
        st.error(
            f"**Auditor judgment failed for this pair:** "
            f"{judgment.get('error', 'unknown error')}"
        )
        return

    verdict = judgment.get("raw_verdict", "?")
    confidence = float(judgment.get("judge_confidence", 0))
    source_assessment = judgment.get("source_assessment", "?")
    failure_mode = judgment.get("failure_mode", "other")
    rationale = judgment.get("rationale", "")

    # Header strip with badges
    st.markdown("### 🧑‍⚖️ Auditor judgment")
    cols = st.columns([1, 1, 1, 1])
    verdict_color = {
        "match": "#27ae60", "no-match": "#e74c3c", "ambiguous": "#95a5a6",
    }.get(verdict, "#7f8c8d")
    cols[0].markdown(
        f"<div style='padding:8px 12px;border-radius:6px;background:{verdict_color};"
        f"color:white;font-weight:bold;text-align:center'>"
        f"Judge verdict: {verdict.upper()}</div>",
        unsafe_allow_html=True,
    )
    src_assess_color = SOURCE_ASSESSMENT_COLORS.get(source_assessment, "#7f8c8d")
    cols[1].markdown(
        f"<div style='padding:8px 12px;border-radius:6px;background:{src_assess_color};"
        f"color:white;font-weight:bold;text-align:center'>"
        f"Source label: {source_assessment.upper()}</div>",
        unsafe_allow_html=True,
    )
    cols[2].metric("Confidence", f"{confidence:.0%}")
    fm_color = FAILURE_MODE_COLORS.get(failure_mode, "#7f8c8d")
    cols[3].markdown(
        f"<div style='padding:8px 12px;border-radius:6px;background:{fm_color};"
        f"color:white;font-weight:bold;text-align:center;font-size:0.9em'>"
        f"Failure mode<br>{failure_mode}</div>",
        unsafe_allow_html=True,
    )

    # 4-way agreement table: source / each scenario / judge
    src_lbl = pair.get("ground_truth_label")
    rows = [{"Reference": "Source label",
             "Verdict": "Match" if src_lbl else "No-Match" if src_lbl is False else "—"}]
    judge_lbl = judgment.get("judge_label")
    rows.append({"Reference": "Judge",
                 "Verdict": "Match" if judge_lbl is True else "No-Match" if judge_lbl is False else "Ambiguous"})
    for sc in scenarios:
        sc_score = pair.get(f"score_{sc}")
        if sc_score is None: continue
        sc_pred = sc_score >= threshold
        rows.append({
            "Reference": f"Scenario {sc} (score={sc_score:.0f})",
            "Verdict": "Match" if sc_pred else "No-Match",
        })

    # Add agreement column with judge as the oracle
    judge_str = "Match" if judge_lbl is True else "No-Match" if judge_lbl is False else "Ambiguous"
    for r in rows:
        if r["Verdict"] == judge_str:
            r["Agrees with judge"] = "✓"
        elif r["Verdict"] == "—" or judge_str == "Ambiguous":
            r["Agrees with judge"] = "—"
        else:
            r["Agrees with judge"] = "✗"
    st.table(rows)

    # Rationale (prominent)
    if rationale:
        st.markdown(f"**Rationale:** {rationale}")

    # Criterion scores as horizontal bars
    cs = judgment.get("criterion_scores") or {}
    if cs:
        with st.expander("Criterion scores"):
            crit_rows = [{"criterion": k, "score": v} for k, v in cs.items()]
            fig = px.bar(
                crit_rows, x="score", y="criterion", orientation="h",
                color="score", color_continuous_scale=[(0, "#e74c3c"), (0.5, "#f39c12"), (1, "#27ae60")],
                range_color=(0, 1),
            )
            fig.update_layout(xaxis_range=[0, 1.0], xaxis_tickformat=".0%",
                              height=240, coloraxis_showscale=False,
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Step-by-step audit trail
    step_1 = judgment.get("step_1_jd_requirements") or []
    step_2 = judgment.get("step_2_cv_qualifications") or []
    step_3 = judgment.get("step_3_coverage") or []
    step_4 = judgment.get("step_4_severity") or ""
    if step_1 or step_2 or step_3 or step_4:
        with st.expander("Step-by-step audit trail"):
            steps = st.columns(2)
            with steps[0]:
                if step_1:
                    st.markdown("**Step 1 — JD requirements**")
                    for it in step_1:
                        st.markdown(f"- {it}")
            with steps[1]:
                if step_2:
                    st.markdown("**Step 2 — CV qualifications**")
                    for it in step_2:
                        st.markdown(f"- {it}")
            if step_3:
                st.markdown("**Step 3 — Coverage check**")
                cov_rows = []
                for c in step_3:
                    if not isinstance(c, dict): continue
                    cov_rows.append({
                        "Requirement": c.get("requirement", ""),
                        "Status": c.get("status", ""),
                        "Evidence": c.get("evidence", ""),
                    })
                if cov_rows:
                    st.dataframe(cov_rows, use_container_width=True)
            if step_4:
                st.markdown(f"**Step 4 — Severity:** {step_4}")

    st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar + data load
# ---------------------------------------------------------------------------

st.sidebar.title("CV-JD Match Dashboard")
mode = st.sidebar.radio(
    "Mode",
    ["View Results", "Browse Dataset", "Curate Gold Set", "Live Match"],
    index=0,
)

if mode == "View Results":
    result_files = find_result_files()
    if not result_files:
        st.sidebar.warning("No result files found in `results/`. Run an evaluation first.")
        st.stop()

    selected_file = st.sidebar.selectbox(
        "Result file",
        result_files,
        format_func=format_result_file_label,
    )

    data = load_results(selected_file)
    traces = load_traces(selected_file.parent / f"{selected_file.stem}_traces.json")
    details = load_details(selected_file.parent / f"{selected_file.stem}_details.jsonl")
    source_pairs = load_source_dataset(selected_file.stem)
    judge_data = load_judge_results(selected_file.stem)
    judge_judgments = judge_data.get("judgments", {}) if judge_data else {}

    config = data.get("config", {})
    pair_results = data.get("pair_results", [])
    summary = data.get("summary", {})
    classification = data.get("classification", {})
    threshold = float(config.get("threshold", 60))
    scenarios = scenarios_present(pair_results)

    # Sidebar: model badge + config
    _model = config.get("model", "unknown")
    _provider, _color = detect_provider(_model)
    st.sidebar.markdown(
        f"<div style='padding:6px 10px;border-radius:6px;background:{_color};"
        f"color:white;font-size:0.85em;text-align:center;margin-top:6px'>"
        f"Model: <b>{_provider}</b></div>",
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("Experiment config"):
        st.json(config)

    # Header strip + headline narrative — visible at top of every page
    render_header_strip(config, len(pair_results))
    st.info(headline_text(data, threshold))

    tabs = st.tabs([
        "Overview",
        "Classification",
        "Errors & Disagreements",
        "Architecture",
        "Auditor",
        "Pair Inspector",
        "Compare Runs",
        "Tezės grafikai",
    ])

    # =====================================================================
    # TAB 1: Overview
    # =====================================================================
    with tabs[0]:
        if not scenarios:
            st.warning("No scenario results in this file.")
        else:
            # KPI cards: per-scenario accuracy + best scenario delta
            kpi_cols = st.columns(len(scenarios) + 1)
            kpi_cols[0].metric("Pairs evaluated", len(pair_results))
            for i, sc in enumerate(scenarios):
                cm = classification.get(sc, {})
                kpi_cols[i + 1].metric(
                    f"Scenario {sc} accuracy",
                    f"{cm.get('accuracy', 0):.1%}" if cm else "—",
                    delta=f"F1 {cm.get('f1', 0):.2f}" if cm else None,
                    delta_color="off",
                )

            st.markdown("### Score by pair")
            st.caption("Each point is one pair. Symbol = ground-truth label. Threshold line shows the match cutoff.")
            st.plotly_chart(score_scatter_chart(pair_results, scenarios, threshold), use_container_width=True)

            st.markdown("### Score distribution by ground-truth class")
            st.caption("Box plots show how well each scenario separates real matches from non-matches. Wider gap between Match/No-Match boxes = stronger discrimination.")
            st.plotly_chart(score_distribution_chart(pair_results, scenarios), use_container_width=True)

            st.markdown("### Summary table")
            token_usage = data.get("token_usage", {})
            rows = []
            for sc in scenarios:
                s = summary.get(sc, {})
                cm = classification.get(sc, {})
                tu = token_usage.get(sc, {})
                rows.append({
                    "Scenario": sc,
                    "Accuracy": f"{cm.get('accuracy', 0):.1%}" if cm else "—",
                    "Precision": f"{cm.get('precision', 0):.1%}" if cm else "—",
                    "Recall": f"{cm.get('recall', 0):.1%}" if cm else "—",
                    "F1": f"{cm.get('f1', 0):.2f}" if cm else "—",
                    "Mean score": f"{s.get('mean_score', 0):.1f}",
                    "Std": f"{s.get('std_score', 0):.1f}",
                    "Mean conf": f"{s.get('mean_confidence', 0):.0%}",
                    "Mean revisions": f"{s.get('mean_revisions', 0):.1f}",
                    "Mean dur (s)": f"{s.get('mean_duration_s', 0):.1f}",
                    "Cost/pair": f"${tu.get('mean_cost_per_pair_usd', 0):.5f}" if tu else "—",
                })
            if classification.get("baseline"):
                bcm = classification["baseline"]
                rows.append({
                    "Scenario": "baseline",
                    "Accuracy": f"{bcm.get('accuracy', 0):.1%}",
                    "Precision": f"{bcm.get('precision', 0):.1%}",
                    "Recall": f"{bcm.get('recall', 0):.1%}",
                    "F1": f"{bcm.get('f1', 0):.2f}",
                    "Mean score": "—", "Std": "—", "Mean conf": "—",
                    "Mean revisions": "—", "Mean dur (s)": "—", "Cost/pair": "—",
                })
            st.table(rows)

            # Cost summary as expander
            if token_usage:
                with st.expander("Cost & efficiency breakdown"):
                    pricing = data.get("pricing", {})
                    if pricing:
                        in_p = pricing.get("input_usd_per_token", 0) * 1_000_000
                        out_p = pricing.get("output_usd_per_token", 0) * 1_000_000
                        st.caption(f"Pricing: ${in_p:.2f}/M in, ${out_p:.2f}/M out")
                    cost_rows = []
                    for sc, stats in sorted(token_usage.items()):
                        cost_rows.append({
                            "Scenario": sc,
                            "Total cost": f"${stats.get('total_cost_usd', 0):.4f}",
                            "Total tokens": f"{stats.get('total_tokens', 0):,}",
                            "LLM calls": f"{stats.get('total_llm_calls', 0):,}",
                            "Cost/pair": f"${stats.get('mean_cost_per_pair_usd', 0):.5f}",
                            "Tokens/pair": f"{stats.get('mean_tokens_per_pair', 0):.0f}",
                            "Calls/pair": f"{stats.get('mean_calls_per_pair', 0):.1f}",
                        })
                    st.table(cost_rows)

    # =====================================================================
    # TAB 2: Classification
    # =====================================================================
    with tabs[1]:
        if not classification:
            st.info("No classification metrics in this file. Ensure your dataset has `ground_truth_label` fields.")
        else:
            st.markdown(f"**Threshold:** score ≥ {int(threshold)} → match")

            # Metric bars
            metric_rows = []
            for sc in scenarios + (["baseline"] if "baseline" in classification else []):
                cm = classification.get(sc, {})
                if not cm: continue
                disp = "Baseline" if sc == "baseline" else f"Scenario {sc}"
                for m in ("accuracy", "precision", "recall", "f1"):
                    metric_rows.append({"Run": disp, "Metric": m.upper(), "Value": cm.get(m, 0)})
            fig_metrics = px.bar(
                metric_rows, x="Run", y="Value", color="Metric", barmode="group",
                color_discrete_sequence=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"],
            )
            fig_metrics.update_layout(yaxis_range=[0, 1.05], yaxis_tickformat=".0%", height=380,
                                      margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_metrics, use_container_width=True)

            # Confusion matrices
            st.markdown("### Confusion matrices")
            cm_keys = scenarios + (["baseline"] if "baseline" in classification else [])
            cm_cols = st.columns(len(cm_keys))
            for i, sc in enumerate(cm_keys):
                cm = classification.get(sc, {})
                disp = "Baseline" if sc == "baseline" else f"Scenario {sc}"
                color = "Greys" if sc == "baseline" else "Blues"
                cm_cols[i].plotly_chart(confusion_matrix_figure(cm, disp, color), use_container_width=True)

            # Threshold sweep
            st.markdown("### Threshold sensitivity")
            st.caption("F1 across thresholds 0–100. Use this to find each scenario's best operating point.")
            st.plotly_chart(threshold_sweep_chart(pair_results, scenarios), use_container_width=True)

            # Optimal thresholds table
            opt_rows = []
            for sc in scenarios:
                sweep = threshold_sweep(pair_results, sc)
                if sweep:
                    best = max(sweep, key=lambda r: r["f1"])
                    opt_rows.append({
                        "Scenario": sc,
                        "Optimal threshold": int(best["threshold"]),
                        "F1": f"{best['f1']:.3f}",
                        "Accuracy": f"{best['accuracy']:.1%}",
                        "Precision": f"{best['precision']:.1%}",
                        "Recall": f"{best['recall']:.1%}",
                    })
            if opt_rows:
                st.markdown("**Optimal threshold per scenario** (by F1)")
                st.table(opt_rows)

            # Calibration
            st.markdown("### Calibration: are confident predictions correct?")
            st.caption("If score X means the system is X% confident this is a match, the line should track the diagonal. Above the diagonal = under-confident; below = over-confident.")
            st.plotly_chart(reliability_diagram(pair_results, scenarios, threshold), use_container_width=True)

            ece = ece_per_scenario(pair_results, scenarios)
            if ece:
                ece_cols = st.columns(len(ece))
                for i, (sc, val) in enumerate(ece.items()):
                    ece_cols[i].metric(f"ECE Scenario {sc}", f"{val:.3f}", delta="lower is better", delta_color="off")

    # =====================================================================
    # TAB 3: Errors & Disagreements
    # =====================================================================
    with tabs[2]:
        if not scenarios:
            st.info("No scenarios available.")
        else:
            # Per-scenario error breakdown
            st.markdown("### Where do scenarios fail?")
            err_rows = []
            for sc in scenarios:
                preds = predictions_for(pair_results, sc, threshold)
                counts = Counter(p["error_dir"] for p in preds.values())
                for dir_ in ("TP", "TN", "FP", "FN"):
                    err_rows.append({"Scenario": f"Scenario {sc}", "Outcome": dir_, "Count": counts.get(dir_, 0)})
            fig_err = px.bar(
                err_rows, x="Scenario", y="Count", color="Outcome", barmode="group",
                color_discrete_map={"TP": "#27ae60", "TN": "#3498db", "FP": "#e74c3c", "FN": "#e67e22"},
            )
            fig_err.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_err, use_container_width=True)

            st.markdown("### Scenario disagreements")
            disagree = disagreement_matrix(pair_results, scenarios, threshold)
            if disagree:
                disag_rows = [
                    {"Pair": k, "n pairs": v["n"], "Agree": v["agree"], "Disagree": v["disagree"],
                     "Agreement rate": f"{v['agreement_rate']:.1%}"}
                    for k, v in disagree.items()
                ]
                st.table(disag_rows)

            # Hard / Easy pairs
            hard, easy = hard_easy_pairs(pair_results, scenarios, threshold)
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"### Hard pairs ({len(hard)} — every scenario wrong)")
                if hard:
                    hard_table = []
                    for pid in hard[:30]:
                        p = next((x for x in pair_results if x["pair_id"] == pid), {})
                        row = {"Pair": pid, "GT": "Match" if p.get("ground_truth_label") else "No-Match"}
                        for sc in scenarios:
                            row[f"Score {sc}"] = f"{p.get(f'score_{sc}', 0):.0f}"
                        hard_table.append(row)
                    st.dataframe(hard_table, use_container_width=True)
                else:
                    st.success("No pairs failed across all scenarios.")
            with cols[1]:
                st.markdown(f"### Easy pairs ({len(easy)} — every scenario correct)")
                if easy:
                    st.write(f"{len(easy)} pairs solved by every scenario. Sample IDs:")
                    st.code(", ".join(easy[:20]) + ("…" if len(easy) > 20 else ""))

            # Accuracy by role
            st.markdown("### Accuracy by role")
            roles = role_accuracy(pair_results, scenarios, threshold)
            role_rows = []
            for role, scs in roles.items():
                row = {"Role": role, "n": next(iter(scs.values()))[1] if scs else 0}
                for sc in scenarios:
                    acc, n = scs.get(sc, (None, 0))
                    row[f"Scenario {sc}"] = f"{acc:.0%}" if acc is not None else "—"
                role_rows.append(row)
            role_rows.sort(key=lambda r: r["n"], reverse=True)
            if role_rows:
                st.dataframe(role_rows[:40], use_container_width=True)
                if len(role_rows) > 40:
                    st.caption(f"Showing top 40 of {len(role_rows)} roles by sample count.")

    # =====================================================================
    # TAB 4: Architecture (agents, memory, Tier 2)
    # =====================================================================
    with tabs[3]:
        is_tier2 = config.get("architecture") == "tier2"

        # Reflection revisions
        st.markdown("### Reflection revisions")
        st.caption("How often the reflection loop forced a revision before committing a score.")
        rs = data.get("reflection_statistics", {})
        if rs:
            rs_rows = [
                {"Scenario": sc, "Pairs revised": v.get("pairs_revised", 0),
                 "Revision rate": f"{v.get('revision_rate', 0):.0%}",
                 "Mean revisions": f"{v.get('mean_revisions', 0):.2f}",
                 "Max": v.get("max_revisions", 0)}
                for sc, v in rs.items()
            ]
            st.table(rs_rows)

        # Tier 2 calibration
        if is_tier2 and traces:
            st.markdown("### Tier 2: Pass 1 vs Pass 2 (Scenario C calibration)")
            cal = calibration_impact(traces)
            if cal["pairs"]:
                ca_cols = st.columns(4)
                d = cal["decisions"]
                ca_cols[0].metric("Lowered", d.get("lower", 0))
                ca_cols[1].metric("Raised", d.get("raise", 0))
                ca_cols[2].metric("Kept", d.get("keep", 0))
                with_history = sum(1 for p in cal["pairs"] if p["n_supporting"] > 0)
                ca_cols[3].metric("With memory", with_history)

                # Pass1 vs Pass2 scatter
                cal_rows = [
                    {"Pair": p["pair_id"], "Pass 1": p["initial"], "Pass 2": p["calibrated"],
                     "Decision": p["decision"], "Δ": (p["calibrated"] or 0) - (p["initial"] or 0)}
                    for p in cal["pairs"] if p["initial"] is not None and p["calibrated"] is not None
                ]
                if cal_rows:
                    fig_cal = px.scatter(
                        cal_rows, x="Pass 1", y="Pass 2", color="Decision",
                        hover_data=["Pair", "Δ"],
                        color_discrete_map={"keep": "#3498db", "lower": "#e74c3c", "raise": "#2ecc71"},
                    )
                    fig_cal.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                                      line=dict(dash="dash", color="gray"))
                    fig_cal.update_layout(xaxis_range=[0, 100], yaxis_range=[0, 100],
                                          height=420, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_cal, use_container_width=True)

                with st.expander("Calibration adjustments table"):
                    st.dataframe(cal_rows, use_container_width=True)
            else:
                st.info("No Tier 2 calibration traces in this run.")

        # ESCO classification
        if is_tier2 and traces:
            st.markdown("### Tier 2: ESCO role classification")
            esco = esco_classification_stats(traces)
            if esco["total"]:
                esco_cols = st.columns(3)
                esco_cols[0].metric("Pairs classified", esco["total"])
                esco_cols[1].metric("Fallback to generic", esco["fallback"],
                                    delta=f"{esco['fallback_rate']:.0%}", delta_color="inverse")
                if esco["confidences"]:
                    esco_cols[2].metric("Mean confidence", f"{statistics.mean(esco['confidences']):.2f}")
                top_roles = esco["role_counts"].most_common(20)
                fig_roles = px.bar(
                    {"Role": [r for r, _ in top_roles], "Count": [c for _, c in top_roles]},
                    x="Role", y="Count",
                )
                fig_roles.update_layout(height=350, xaxis_tickangle=-45, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_roles, use_container_width=True)
            else:
                st.info("No ESCO classification data in traces (Tier 2 requires Scenario B or C).")

        # Memory learning curve
        st.markdown("### Memory learning curve (Scenario C)")
        memory_mode = config.get("memory_mode", "isolated")
        smm = config.get("streaming_memory_mode")
        if "C" in scenarios and (memory_mode == "shared" or smm in ("cold-start", "continue-stream")):
            running = []
            for sc_label, sc_key in (("Scenario A", "score_A"), ("Scenario B", "score_B"), ("Scenario C", "score_C")):
                if sc_key.split("_")[1] not in scenarios: continue
                correct = total = 0
                for idx, p in enumerate(pair_results, 1):
                    score = p.get(sc_key)
                    gt = p.get("ground_truth_label")
                    if score is None or gt is None: continue
                    total += 1
                    if (score >= threshold) == gt:
                        correct += 1
                    running.append({"Pair #": idx, "Scenario": sc_label, "Running accuracy": correct / total})
            if running:
                fig_run = px.line(
                    running, x="Pair #", y="Running accuracy", color="Scenario", markers=True,
                    color_discrete_map={f"Scenario {s}": SCENARIO_COLORS[s] for s in scenarios},
                )
                fig_run.update_layout(yaxis_range=[0, 1.05], yaxis_tickformat=".0%",
                                      height=380, margin=dict(l=20, r=20, t=20, b=20))
                fig_run.add_hline(y=0.5, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_run, use_container_width=True)
                st.caption("If Scenario C improves over A/B as more pairs accumulate, the labeled-memory calibration is helping.")
        else:
            st.info("Memory learning curve only available for Scenario C with shared/streaming memory.")

        # Per-agent breakdown
        agent_eff = data.get("agent_efficiency", {})
        if agent_eff:
            with st.expander("Per-agent token & duration breakdown"):
                agent_rows = []
                for sc, agents in agent_eff.items():
                    for name, stats in agents.items():
                        agent_rows.append({
                            "Scenario": sc,
                            "Agent": name,
                            "Calls": stats.get("calls", 0),
                            "Total dur (s)": f"{stats.get('total_duration_s', 0):.1f}",
                            "Mean dur (s)": f"{stats.get('mean_duration_s', 0):.2f}",
                            "Tokens": f"{stats.get('total_tokens', 0):,}",
                            "Cost": f"${stats.get('cost_usd', 0):.4f}",
                        })
                st.dataframe(agent_rows, use_container_width=True)

    # =====================================================================
    # TAB 5: Auditor (LLM-as-Judge analysis)
    # =====================================================================
    with tabs[4]:
        if not judge_data:
            st.info(
                "No auditor (LLM-as-Judge) data for this dataset.\n\n"
                f"Run: `python audit.py --evaluate data/{selected_file.stem}.json --workers 8 --self-consistency-sample 30`\n\n"
                "After it finishes, the file `results/judge_<dataset>.json` will be picked up here automatically."
            )
        else:
            jmeta = judge_data.get("_meta", {})

            # Reproducibility metadata strip
            rep_cols = st.columns([2, 1, 1, 1, 1])
            rep_cols[0].markdown(
                f"**Judge:** `{jmeta.get('judge_model', '?')}` "
                f"@ `{jmeta.get('judge_base_url', '?')}`"
            )
            rep_cols[1].metric("Prompt v.", jmeta.get("judge_prompt_version", "?")[:8])
            rep_cols[2].metric("Temperature", jmeta.get("judge_temperature", "?"))
            rep_cols[3].metric("n pairs", jmeta.get("n_pairs", len(judge_judgments)))
            sc = jmeta.get("self_consistency") or {}
            sc_rate = sc.get("verdict_agreement_rate")
            rep_cols[4].metric(
                "Self-consistency",
                f"{sc_rate:.0%}" if isinstance(sc_rate, (int, float)) else "—",
                delta=f"n={sc.get('sample_size', 0)}" if sc else None,
                delta_color="off",
            )
            st.caption(
                f"Methodology: Gu et al. 2025, *A Survey on LLM-as-a-Judge* "
                f"([arXiv:2411.15594](https://arxiv.org/abs/2411.15594)). "
                f"Cross-family judge, blind to system predictions; ambiguous "
                f"verdicts triggered when judge confidence < "
                f"{jmeta.get('ambiguous_confidence_threshold', 0.7)}."
            )

            # ===== KPI strip =====
            n_correct = sum(1 for j in judge_judgments.values() if j.get("source_assessment") == "correct")
            n_incorrect = sum(1 for j in judge_judgments.values() if j.get("source_assessment") == "incorrect")
            n_ambig = sum(1 for j in judge_judgments.values() if j.get("source_assessment") == "ambiguous")
            n_errors = sum(1 for j in judge_judgments.values() if j.get("error"))
            n_total = len(judge_judgments)
            mean_conf = (
                sum(j.get("judge_confidence", 0) for j in judge_judgments.values() if not j.get("error"))
                / max(1, n_total - n_errors)
            )

            st.markdown("### Source label quality")
            kpi = st.columns(5)
            kpi[0].metric("Source correct", f"{n_correct}", delta=f"{n_correct / max(1, n_total):.1%}", delta_color="off")
            kpi[1].metric("Source incorrect", f"{n_incorrect}", delta=f"{n_incorrect / max(1, n_total):.1%}", delta_color="inverse")
            kpi[2].metric("Ambiguous", f"{n_ambig}", delta=f"{n_ambig / max(1, n_total):.1%}", delta_color="off")
            kpi[3].metric("Errors", f"{n_errors}", delta_color="inverse" if n_errors else "off")
            kpi[4].metric("Mean confidence", f"{mean_conf:.2f}")

            st.info(
                f"**Headline:** the judge classified {n_incorrect / max(1, n_total):.0%} of source labels "
                f"as **incorrect** and {n_ambig / max(1, n_total):.0%} as **ambiguous** — "
                f"only {n_correct / max(1, n_total):.0%} of source labels are reliable. "
                f"This is the dataset noise floor."
            )

            # ===== Verdict distribution + Source vs Judge confusion =====
            row1 = st.columns(2)
            with row1[0]:
                st.markdown("#### Judge verdict distribution")
                verdict_counts = Counter(
                    j.get("raw_verdict", "?") for j in judge_judgments.values() if not j.get("error")
                )
                vd_rows = [{"verdict": k, "count": v} for k, v in verdict_counts.items()]
                fig_v = px.pie(vd_rows, names="verdict", values="count",
                               color="verdict",
                               color_discrete_map={
                                   "match": "#27ae60", "no-match": "#e74c3c",
                                   "ambiguous": "#95a5a6", "?": "#7f8c8d",
                               })
                fig_v.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_v, use_container_width=True)

            with row1[1]:
                st.markdown("#### Source label vs Judge verdict")
                # Build cross-tab
                src_pairs_local = source_pairs
                xtab_counts = defaultdict(int)
                for pid, j in judge_judgments.items():
                    if j.get("error"): continue
                    src_lbl = src_pairs_local.get(pid, {}).get("ground_truth_label")
                    src_str = "Match" if src_lbl is True else "No-Match" if src_lbl is False else "—"
                    j_str = "Match" if j.get("judge_label") is True else "No-Match" if j.get("judge_label") is False else "Ambiguous"
                    xtab_counts[(src_str, j_str)] += 1
                src_order = ["Match", "No-Match", "—"]
                jdg_order = ["Match", "No-Match", "Ambiguous"]
                z = [[xtab_counts[(s, j)] for j in jdg_order] for s in src_order]
                fig_xt = go.Figure(data=go.Heatmap(
                    z=z, x=jdg_order, y=src_order,
                    text=[[str(v) for v in row] for row in z],
                    texttemplate="%{text}", colorscale="Blues", showscale=False,
                ))
                fig_xt.update_layout(
                    height=320, xaxis_title="Judge verdict", yaxis_title="Source label",
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_xt, use_container_width=True)

            # ===== Source-label asymmetry =====
            st.markdown("#### Source-label asymmetry (which direction is the source wrong?)")
            wrong_rejects = sum(
                1 for pid, j in judge_judgments.items()
                if j.get("source_assessment") == "incorrect"
                and source_pairs.get(pid, {}).get("ground_truth_label") is False
            )
            wrong_accepts = sum(
                1 for pid, j in judge_judgments.items()
                if j.get("source_assessment") == "incorrect"
                and source_pairs.get(pid, {}).get("ground_truth_label") is True
            )
            asym_rows = [
                {"Direction": "Wrong rejects (source said no, judge says match)", "Count": wrong_rejects},
                {"Direction": "Wrong accepts (source said yes, judge says no-match)", "Count": wrong_accepts},
            ]
            fig_asym = px.bar(asym_rows, x="Count", y="Direction", orientation="h",
                              color="Direction",
                              color_discrete_map={
                                  asym_rows[0]["Direction"]: "#e67e22",
                                  asym_rows[1]["Direction"]: "#3498db",
                              })
            fig_asym.update_layout(height=180, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_asym, use_container_width=True)
            if wrong_rejects + wrong_accepts > 0:
                st.caption(
                    f"Of incorrect labels: **{wrong_rejects / (wrong_rejects + wrong_accepts):.0%}** are wrong-rejects, "
                    f"**{wrong_accepts / (wrong_rejects + wrong_accepts):.0%}** are wrong-accepts. "
                    f"If wrong-rejects dominate, the dataset's *rejection* mechanism is the broken part."
                )

            # ===== Failure mode taxonomy =====
            st.markdown("### Failure mode taxonomy")
            mode_counts = Counter(
                j.get("failure_mode", "other")
                for j in judge_judgments.values() if not j.get("error")
            )
            mode_rows = [{"failure_mode": m, "count": c} for m, c in mode_counts.most_common()]
            fig_m = px.bar(mode_rows, x="failure_mode", y="count", color="failure_mode",
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig_m.update_layout(height=350, showlegend=False, xaxis_tickangle=-25,
                                margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_m, use_container_width=True)

            templated = mode_counts.get("templated_reject", 0)
            if templated:
                st.warning(
                    f"**{templated} pairs ({templated / max(1, n_total):.0%}) are templated rejects** — "
                    f"the JD trivially asks for the candidate's exact role yet the source labeled "
                    f"the pair 'rejected'. Every templated_reject pair maps to source_assessment=incorrect, "
                    f"which is the strongest single signal of dataset noise we can produce."
                )

            # ===== Per-scenario accuracy: source vs judge =====
            if scenarios:
                st.markdown("### Per-scenario accuracy: against source labels vs against judge labels")
                acc_rows = []
                for sc in scenarios:
                    cm = classification.get(sc, {})
                    if cm:
                        acc_rows.append({"Scenario": f"Scenario {sc}", "Reference": "Source label",
                                         "Accuracy": cm.get("accuracy", 0)})
                    # vs judge (excluding ambiguous)
                    correct = total_eval = 0
                    for p in pair_results:
                        pid = p["pair_id"]
                        j = judge_judgments.get(pid)
                        if not j or j.get("error") or j.get("judge_label") is None:
                            continue
                        score = p.get(f"score_{sc}")
                        if score is None: continue
                        sys_pred = score >= threshold
                        total_eval += 1
                        if sys_pred == j["judge_label"]:
                            correct += 1
                    if total_eval:
                        acc_rows.append({"Scenario": f"Scenario {sc}", "Reference": "Judge label",
                                         "Accuracy": correct / total_eval})
                fig_acc = px.bar(acc_rows, x="Scenario", y="Accuracy", color="Reference",
                                 barmode="group",
                                 color_discrete_map={"Source label": "#95a5a6", "Judge label": "#9b59b6"})
                fig_acc.update_layout(yaxis_range=[0, 1.05], yaxis_tickformat=".0%",
                                      height=380, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_acc, use_container_width=True)
                st.caption(
                    "When measured against an independent expert auditor (judge), accuracy is markedly "
                    "higher than against the noisy source labels — the system was being penalized "
                    "for being right about pairs the source mislabeled."
                )

            # ===== Disputed-pair adjudication =====
            if scenarios:
                st.markdown("### Disputed-pair adjudication")
                st.caption(
                    "When the system disagreed with the source label, who did the independent "
                    "judge side with? Higher 'sided with system' rate = the system was right and "
                    "the source was wrong on the disputed pairs."
                )
                disp_rows = []
                for sc in scenarios:
                    sided_sys = sided_src = total_disp = 0
                    for p in pair_results:
                        pid = p["pair_id"]
                        j = judge_judgments.get(pid)
                        if not j or j.get("error") or j.get("judge_label") is None:
                            continue
                        src_lbl = p.get("ground_truth_label")
                        if src_lbl is None: continue
                        score = p.get(f"score_{sc}")
                        if score is None: continue
                        sys_pred = score >= threshold
                        if sys_pred == src_lbl:
                            continue  # not disputed
                        total_disp += 1
                        if j["judge_label"] == sys_pred:
                            sided_sys += 1
                        else:
                            sided_src += 1
                    if total_disp:
                        disp_rows.append({
                            "Scenario": f"Scenario {sc}",
                            "Disputed pairs": total_disp,
                            "Judge sided with system": sided_sys,
                            "Judge sided with source": sided_src,
                            "% sided with system": f"{sided_sys / total_disp:.0%}",
                        })
                if disp_rows:
                    st.table(disp_rows)

            # ===== Confidence distribution =====
            st.markdown("### Judge confidence distribution")
            confs = [j.get("judge_confidence", 0) for j in judge_judgments.values() if not j.get("error")]
            fig_conf = px.histogram(
                {"confidence": confs}, x="confidence", nbins=20,
                color_discrete_sequence=["#3498db"],
            )
            fig_conf.update_layout(
                height=260, margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="pairs", xaxis_title="judge confidence",
            )
            fig_conf.add_vline(
                x=jmeta.get("ambiguous_confidence_threshold", 0.7),
                line_dash="dash", line_color="red",
                annotation_text="ambiguous threshold", annotation_position="top right",
            )
            st.plotly_chart(fig_conf, use_container_width=True)

            # ===== Per-criterion score distributions =====
            crit_rows = []
            for j in judge_judgments.values():
                if j.get("error"): continue
                cs = j.get("criterion_scores", {}) or {}
                assess = j.get("source_assessment", "?")
                for crit, score in cs.items():
                    crit_rows.append({"criterion": crit, "score": score, "source assessment": assess})
            if crit_rows:
                st.markdown("### Per-criterion score distribution by source assessment")
                fig_crit = px.box(
                    crit_rows, x="criterion", y="score", color="source assessment", points=False,
                    color_discrete_map={
                        "correct": "#27ae60", "incorrect": "#e74c3c",
                        "ambiguous": "#95a5a6",
                    },
                )
                fig_crit.update_layout(yaxis_range=[0, 1.05], height=380,
                                       margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_crit, use_container_width=True)
                st.caption(
                    "If 'incorrect' source assessments cluster at high criterion scores, that means "
                    "the judge consistently saw strong matches on pairs the source labeled as rejects — "
                    "again, dataset noise rather than system error."
                )

            # ===== Per-role accuracy =====
            st.markdown("### Per-role breakdown")
            role_stats = defaultdict(lambda: {"n": 0, "src_correct": 0, "src_incorrect": 0, "src_ambig": 0})
            for pid, j in judge_judgments.items():
                if j.get("error"): continue
                role = "_".join(pid.split("_")[:-1]) or "unknown"
                role_stats[role]["n"] += 1
                if j.get("source_assessment") == "correct":
                    role_stats[role]["src_correct"] += 1
                elif j.get("source_assessment") == "incorrect":
                    role_stats[role]["src_incorrect"] += 1
                elif j.get("source_assessment") == "ambiguous":
                    role_stats[role]["src_ambig"] += 1
            role_rows = []
            for role, st_ in sorted(role_stats.items(), key=lambda x: -x[1]["n"]):
                if st_["n"] < 2: continue  # too few to be meaningful
                role_rows.append({
                    "Role": role,
                    "n": st_["n"],
                    "Source correct": f"{st_['src_correct']} ({st_['src_correct'] / st_['n']:.0%})",
                    "Source incorrect": f"{st_['src_incorrect']} ({st_['src_incorrect'] / st_['n']:.0%})",
                    "Ambiguous": f"{st_['src_ambig']} ({st_['src_ambig'] / st_['n']:.0%})",
                })
            if role_rows:
                st.dataframe(role_rows, use_container_width=True)
            else:
                st.caption("Not enough per-role samples for a breakdown (need ≥2 pairs per role).")

            # ===== Errors block =====
            errs = {pid: j for pid, j in judge_judgments.items() if j.get("error")}
            if errs:
                with st.expander(f"Judge errors ({len(errs)})"):
                    for pid, j in errs.items():
                        st.code(f"{pid}: {j.get('error', '')}")

    # =====================================================================
    # TAB 6: Pair Inspector
    # =====================================================================
    with tabs[5]:
        st.markdown("Pick a pair and see all scenarios side-by-side, including reasoning, reflection, and (if Tier 2) calibration.")

        if not pair_results:
            st.info("No pairs to inspect.")
        else:
            pid_options = [p["pair_id"] for p in pair_results]

            # Per-result-file session-state key so switching files doesn't carry stale selection
            pi_state_key = f"pi_pid::{selected_file}"
            if pi_state_key not in st.session_state or st.session_state[pi_state_key] not in pid_options:
                st.session_state[pi_state_key] = pid_options[0]

            idx = pid_options.index(st.session_state[pi_state_key])

            def _pi_prev():
                cur = pid_options.index(st.session_state[pi_state_key])
                st.session_state[pi_state_key] = pid_options[max(0, cur - 1)]

            def _pi_next():
                cur = pid_options.index(st.session_state[pi_state_key])
                st.session_state[pi_state_key] = pid_options[min(len(pid_options) - 1, cur + 1)]

            nav_cols = st.columns([1, 1, 6, 2])
            nav_cols[0].button("◀ Prev", on_click=_pi_prev, disabled=idx == 0, key="pi_prev_btn", use_container_width=True)
            nav_cols[1].button("Next ▶", on_click=_pi_next, disabled=idx >= len(pid_options) - 1, key="pi_next_btn", use_container_width=True)
            sel_pid = nav_cols[2].selectbox(
                f"Pair ({idx + 1} of {len(pid_options)})",
                pid_options,
                index=idx,
                key=pi_state_key,
            )
            nav_cols[3].markdown(
                f"<div style='padding-top:30px;text-align:right;color:#666'>"
                f"{idx + 1} / {len(pid_options)}</div>",
                unsafe_allow_html=True,
            )

            pair = next(p for p in pair_results if p["pair_id"] == sel_pid)
            src = source_pairs.get(sel_pid, {})

            # Top summary card
            top_cols = st.columns([1, 1, 1, 2])
            gt_label = pair.get("ground_truth_label")
            top_cols[0].metric("Ground truth", "Match" if gt_label else "No-Match" if gt_label is False else "—")
            top_cols[1].metric("Baseline score", f"{pair.get('score_baseline', 0):.0f}" if "score_baseline" in pair else "—")
            top_cols[2].metric("# scenarios run", len(scenarios))
            ref_reason = src.get("reference_reason") or "—"
            top_cols[3].markdown(
                f"<div style='font-size:0.85em;line-height:1.3'>"
                f"<b>Reference reason:</b> {ref_reason}<br>"
                f"<span style='color:#999;font-size:0.85em'>"
                f"<i>Templated synthetic reasons; suppressed at the labeled-memory boundary "
                f"unless the source is a curator-curated gold set.</i></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # Side-by-side scenario columns
            sc_cols = st.columns(len(scenarios))
            for i, sc in enumerate(scenarios):
                with sc_cols[i]:
                    score = pair.get(f"score_{sc}", 0)
                    rec = pair.get(f"recommendation_{sc}", "—")
                    conf = pair.get(f"confidence_{sc}", 0)
                    rev = pair.get(f"revisions_{sc}", 0)
                    pred_match = score >= threshold
                    correct = (gt_label is not None) and (pred_match == gt_label)

                    rec_color = RECOMMENDATION_COLORS.get(rec, "#666")
                    correct_color = "#27ae60" if correct else "#e74c3c"
                    correct_label = "✓ correct" if correct else "✗ wrong"

                    st.markdown(
                        f"<div style='padding:8px;border-radius:6px;background:{SCENARIO_COLORS[sc]};color:white;font-weight:bold;text-align:center'>"
                        f"Scenario {sc}</div>",
                        unsafe_allow_html=True,
                    )
                    st.metric(f"Score", f"{score:.0f}",
                              delta=f"{correct_label}" if gt_label is not None else None,
                              delta_color="off")
                    st.markdown(
                        f"<span style='color:{rec_color};font-weight:bold'>{rec}</span><br>"
                        f"Confidence: {conf:.0%} · Revisions: {rev}",
                        unsafe_allow_html=True,
                    )

                    # Trace details for this pair + scenario
                    trace = next((t for t in traces if t.get("pair_id") == sel_pid and t.get("scenario") == sc), None)
                    if trace:
                        with st.expander("Reasoning"):
                            r = trace.get("reasoning", {})
                            if r.get("strengths"):
                                st.markdown("**Strengths**")
                                for s in r["strengths"]:
                                    st.markdown(f"- {s}")
                            if r.get("gaps"):
                                st.markdown("**Gaps**")
                                for g in r["gaps"]:
                                    st.markdown(f"- {g}")
                            if r.get("concerns"):
                                st.markdown("**Concerns**")
                                for c in r["concerns"]:
                                    st.markdown(f"- {c}")
                            if r.get("overall_assessment"):
                                st.markdown(f"**Assessment:** {r['overall_assessment']}")
                            if r.get("suggested_score") is not None:
                                st.caption(f"Suggested score: {r['suggested_score']}")

                        refl = trace.get("reflection") or {}
                        if refl:
                            with st.expander("Reflection"):
                                st.markdown(f"**Consistent:** {refl.get('is_consistent', '—')}")
                                if refl.get("issues"):
                                    st.markdown("**Issues**")
                                    for it in refl["issues"]:
                                        st.markdown(f"- {it}")
                                if refl.get("suggestions"):
                                    st.markdown("**Suggestions**")
                                    for s in refl["suggestions"]:
                                        st.markdown(f"- {s}")
                                if refl.get("revision_reason"):
                                    st.markdown(f"**Why revised:** {refl['revision_reason']}")

                        dec = trace.get("decision") or {}
                        if dec:
                            with st.expander("Decision"):
                                st.markdown(f"**Explanation:** {dec.get('explanation', '—')}")
                                if dec.get("key_factors"):
                                    st.markdown("**Key factors**")
                                    for kf in dec["key_factors"]:
                                        st.markdown(f"- {kf}")

                        # Tier 2 calibration (Scenario C only)
                        t2 = trace.get("tier2") or {}
                        if sc == "C" and t2.get("calibration_decision"):
                            with st.expander("Pass 2: calibration"):
                                st.markdown(f"**Pass 1 score:** {t2.get('initial_score', '—')}")
                                st.markdown(f"**Pass 2 score:** {dec.get('final_score', '—')}")
                                st.markdown(f"**Decision:** {t2.get('calibration_decision', '—')} "
                                            f"(adjustment: {t2.get('calibration_adjustment', 0):+.1f})")
                                st.markdown(f"**Pattern:** {t2.get('calibration_pattern', '—')}")
                                st.markdown(f"**Rationale:** {t2.get('calibration_rationale', '—')}")
                                st.caption(f"Supporting memories: {t2.get('calibration_n_supporting', 0)}")

            st.markdown("---")

            # Judge (LLM-as-Judge) data for this pair, if available
            judge_for_pair = judge_judgments.get(sel_pid)
            if judge_for_pair:
                _render_judge_block_for_pair(judge_for_pair, pair, sel_pid, threshold, scenarios)

            # Tier 2 profiles (if available)
            sample_trace = next((t for t in traces if t.get("pair_id") == sel_pid), None)
            t2 = (sample_trace or {}).get("tier2") or {}
            if t2.get("cv_profile") or t2.get("jd_profile"):
                with st.expander("Tier 2 profiles (CV ↔ JD)"):
                    pcols = st.columns(2)
                    with pcols[0]:
                        st.markdown("#### Candidate profile")
                        cvp = t2.get("cv_profile") or {}
                        st.markdown(f"**Archetype:** {cvp.get('candidate_archetype', '—')}")
                        st.markdown(f"**Seniority:** {cvp.get('seniority_level', '—')}")
                        st.markdown(f"**Likely role fit:** `{cvp.get('likely_role_fit', '—')}`")
                        if cvp.get("domain_expertise"):
                            st.markdown("**Domain expertise:** " + ", ".join(cvp["domain_expertise"]))
                        if cvp.get("skills"):
                            st.markdown("**Skills:** " + ", ".join(cvp["skills"][:20])
                                        + ("…" if len(cvp["skills"]) > 20 else ""))
                    with pcols[1]:
                        st.markdown("#### Ideal-candidate profile (from JD)")
                        jdp = t2.get("jd_profile") or {}
                        st.markdown(f"**Detected role:** `{jdp.get('detected_role', '—')}` "
                                    f"(confidence {jdp.get('role_confidence', 0):.2f})")
                        st.markdown(f"**Seniority required:** {jdp.get('seniority_required', '—')}")
                        st.markdown(f"**Required exp (years):** {jdp.get('required_experience_years', 0)}")
                        if jdp.get("required_skills"):
                            st.markdown("**Required skills:** " + ", ".join(jdp["required_skills"]))
                        if jdp.get("typical_role_skills"):
                            st.markdown("**ESCO-typical skills:** " + ", ".join(jdp["typical_role_skills"][:15])
                                        + ("…" if len(jdp["typical_role_skills"]) > 15 else ""))

            # CV / JD source text
            with st.expander("CV text"):
                st.text(src.get("cv_text", "Source dataset not available — load the matching dataset into data/."))
            with st.expander("Job description text"):
                jd_text = src.get("jd_text", "")
                if src.get("jd_augmented"):
                    st.caption("This JD was augmented from a shorter original.")
                    if src.get("original_jd_text"):
                        with st.expander("Original (pre-augmentation) JD"):
                            st.text(src["original_jd_text"])
                st.text(jd_text or "Source dataset not available.")

    # =====================================================================
    # TAB 7: Compare Runs
    # =====================================================================
    with tabs[6]:
        other_files = [f for f in result_files if f != selected_file]
        if not other_files:
            st.info("Only one result file present. Run another evaluation to enable comparison.")
        else:
            other_file = st.selectbox(
                "Compare against",
                other_files,
                format_func=format_result_file_label,
                key="cmp_other",
            )
            other = load_results(other_file)
            other_pair_results = other.get("pair_results", [])
            other_classification = other.get("classification", {})
            other_config = other.get("config", {})
            other_threshold = float(other_config.get("threshold", 60))
            other_scenarios = scenarios_present(other_pair_results)
            common_sc = [s for s in scenarios if s in other_scenarios]

            # Cross-model warning
            this_model = config.get("model")
            other_model = other_config.get("model")
            if this_model and other_model and this_model != other_model:
                st.warning(
                    f"**Cross-model comparison.** Differences may reflect both architectural changes "
                    f"AND model differences ({this_model} vs {other_model})."
                )

            # Headlines
            cols = st.columns(2)
            with cols[0]:
                st.markdown(f"**A:** {format_result_file_label(selected_file)}")
                st.caption(headline_text(data, threshold))
            with cols[1]:
                st.markdown(f"**B:** {format_result_file_label(other_file)}")
                st.caption(headline_text(other, other_threshold))

            # Config diff
            with st.expander("Config diff"):
                all_keys = sorted(set(config) | set(other_config))
                diff_rows = []
                for k in all_keys:
                    a, b = config.get(k), other_config.get(k)
                    if a != b:
                        diff_rows.append({"Key": k, "A": str(a), "B": str(b)})
                if diff_rows:
                    st.table(diff_rows)
                else:
                    st.success("Configs are identical.")

            # Classification deltas
            if classification and other_classification and common_sc:
                st.markdown("### Classification deltas (B − A)")
                cmp_rows = []
                for sc in common_sc:
                    a = classification.get(sc, {})
                    b = other_classification.get(sc, {})
                    cmp_rows.append({
                        "Scenario": sc,
                        "Acc A": f"{a.get('accuracy', 0):.1%}",
                        "Acc B": f"{b.get('accuracy', 0):.1%}",
                        "Δ Acc": f"{(b.get('accuracy', 0) - a.get('accuracy', 0)) * 100:+.1f}pp",
                        "F1 A": f"{a.get('f1', 0):.3f}",
                        "F1 B": f"{b.get('f1', 0):.3f}",
                        "Δ F1": f"{b.get('f1', 0) - a.get('f1', 0):+.3f}",
                    })
                st.table(cmp_rows)

            # Per-pair flips
            st.markdown("### Per-pair decision flips")
            st.caption("Pairs where A and B disagreed on the match decision. Highlights what actually changed between runs.")
            other_by_pid = {p["pair_id"]: p for p in other_pair_results}
            flips_correct, flips_wrong, both_correct, both_wrong = 0, 0, 0, 0
            flip_rows = []
            for sc in common_sc:
                for p in pair_results:
                    pid = p["pair_id"]
                    other_p = other_by_pid.get(pid)
                    if not other_p: continue
                    sa = p.get(f"score_{sc}")
                    sb = other_p.get(f"score_{sc}")
                    gt = p.get("ground_truth_label")
                    if sa is None or sb is None or gt is None: continue
                    pred_a = sa >= threshold
                    pred_b = sb >= other_threshold
                    correct_a = pred_a == gt
                    correct_b = pred_b == gt
                    if pred_a == pred_b:
                        if correct_a: both_correct += 1
                        else: both_wrong += 1
                        continue
                    if (not correct_a) and correct_b:
                        flips_correct += 1
                        flip_type = "wrong → correct"
                    elif correct_a and (not correct_b):
                        flips_wrong += 1
                        flip_type = "correct → wrong"
                    else:
                        flip_type = "—"  # shouldn't happen
                    flip_rows.append({
                        "Scenario": sc, "Pair": pid,
                        "Score A": f"{sa:.0f}", "Score B": f"{sb:.0f}",
                        "GT": "Match" if gt else "No-Match",
                        "Flip": flip_type,
                    })

            flip_cols = st.columns(4)
            flip_cols[0].metric("Wrong → correct", flips_correct, delta="B improvement", delta_color="off")
            flip_cols[1].metric("Correct → wrong", flips_wrong, delta="B regression", delta_color="off")
            flip_cols[2].metric("Both correct", both_correct)
            flip_cols[3].metric("Both wrong", both_wrong)
            net = flips_correct - flips_wrong
            if net > 0:
                st.success(f"Net improvement: **+{net} pairs** correct in B that were wrong in A.")
            elif net < 0:
                st.error(f"Net regression: **{net} pairs** wrong in B that were correct in A.")
            else:
                st.info("Net flips even — same accuracy, different specific pairs.")

            if flip_rows:
                with st.expander(f"Show all {len(flip_rows)} flips"):
                    st.dataframe(flip_rows, use_container_width=True)

            # Cost / quality
            tu_a = data.get("token_usage", {})
            tu_b = other.get("token_usage", {})
            if tu_a and tu_b and common_sc:
                st.markdown("### Cost vs accuracy")
                pts = []
                for sc in common_sc:
                    pts.append({"Run": "A", "Scenario": sc,
                                "Cost/pair": tu_a.get(sc, {}).get("mean_cost_per_pair_usd", 0),
                                "Accuracy": classification.get(sc, {}).get("accuracy", 0)})
                    pts.append({"Run": "B", "Scenario": sc,
                                "Cost/pair": tu_b.get(sc, {}).get("mean_cost_per_pair_usd", 0),
                                "Accuracy": other_classification.get(sc, {}).get("accuracy", 0)})
                fig_cost = px.scatter(
                    pts, x="Cost/pair", y="Accuracy", color="Scenario", symbol="Run", size_max=20,
                    hover_data=["Run", "Scenario"],
                    color_discrete_map={s: SCENARIO_COLORS[s] for s in common_sc},
                )
                fig_cost.update_traces(marker=dict(size=14, line=dict(width=1, color="black")))
                fig_cost.update_layout(yaxis_tickformat=".0%", height=380, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_cost, use_container_width=True)

    # =====================================================================
    # TAB 8: Tezės grafikai — all thesis figures rendered interactively
    # =====================================================================
    with tabs[7]:
        st.header("Tezės grafikai")
        st.caption(
            "Šiame skyriuje surinkti visi tezės darbui naudingi grafikai. "
            "Kiekvienas yra interaktyvi versija, sukurta iš dabar pasirinkto rezultatų failo. "
            "Statinės PNG versijos automatiškai gaminamos į `thesis/images/` paleidus "
            "`python scripts/generate_thesis_plots.py`."
        )

        if not scenarios:
            st.warning("No scenario results in this file — pasirink kitą failą.")
        else:
            judge_dataset_stem = selected_file.stem
            judge_data = load_judge_results(judge_dataset_stem)
            judge = (judge_data or {}).get("judgments", {})
            has_judge = bool(judge)

            CONF_THR = 0.7

            def _judge_label(v: dict):
                if not v or v.get("error"):
                    return None
                if v.get("judge_confidence", 0) < CONF_THR:
                    return None
                return v.get("judge_label")

            # ---------------- Grafikas 1: Sprendimai prieš/po audito ----------------
            st.subheader("1. Sistemos sprendimų teisingumas prieš ir po LLM-teisėjo audito")
            if not has_judge:
                st.info(
                    "Šiam rezultatų failui dar nėra teisėjo audito (`results/judge_<dataset>.json`). "
                    "Paleisti: `python audit.py --evaluate data/<dataset>.json`."
                )
            else:
                COL_OK = "#2E7D32"; COL_WRONG = "#C62828"
                COL_AMB = "#9E9E9E"; COL_REHAB = "#81C784"
                fig = make_subplots(
                    rows=1, cols=len(scenarios),
                    subplot_titles=[f"Scenarijus {s}" for s in scenarios],
                    shared_yaxes=True,
                )
                for col_i, scen in enumerate(scenarios, start=1):
                    correct_src, wrong_src = 0, 0
                    correct_jud, wrong_jud, ambig = 0, 0, 0
                    rehab, truly_wrong, rehab_amb = 0, 0, 0
                    for p in pair_results:
                        sc = p.get(f"score_{scen}")
                        lab = p.get("ground_truth_label")
                        if sc is None or lab is None:
                            continue
                        pred = sc >= threshold
                        ok_src = (pred == bool(lab))
                        if ok_src:
                            correct_src += 1
                        else:
                            wrong_src += 1
                        v = judge.get(p["pair_id"])
                        jl = _judge_label(v) if v else None
                        if jl is None:
                            ambig += 1
                            if not ok_src:
                                rehab_amb += 1
                        else:
                            ok_jud = (pred == bool(jl))
                            if ok_jud:
                                correct_jud += 1
                            else:
                                wrong_jud += 1
                            if not ok_src:
                                if ok_jud:
                                    rehab += 1
                                else:
                                    truly_wrong += 1

                    x_labels = ["Prieš auditą", "Po audito", "Klaidingų išskaidymas"]
                    fig.add_trace(go.Bar(x=x_labels, y=[correct_src, correct_jud, rehab],
                                         name="Teisingi", marker_color=COL_OK,
                                         showlegend=(col_i == 1), legendgroup="ok"),
                                  row=1, col=col_i)
                    fig.add_trace(go.Bar(x=x_labels, y=[wrong_src, wrong_jud, truly_wrong],
                                         name="Klaidingi", marker_color=COL_WRONG,
                                         showlegend=(col_i == 1), legendgroup="bad"),
                                  row=1, col=col_i)
                    fig.add_trace(go.Bar(x=x_labels, y=[0, ambig, rehab_amb],
                                         name="Neaiškūs (teisėjo verdiktas)", marker_color=COL_AMB,
                                         showlegend=(col_i == 1), legendgroup="amb"),
                                  row=1, col=col_i)
                fig.update_layout(
                    barmode="stack", height=520,
                    title=f"Sistemos sprendimų teisingumas prieš ir po LLM-teisėjo audito (n={len(pair_results)})",
                    margin=dict(l=20, r=20, t=80, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.18, x=0.5, xanchor="center"),
                )
                st.plotly_chart(fig, use_container_width=True)

            # ---------------- Grafikas 2: F1 × scenarijus × etalonas ----------------
            st.subheader("2. F1 įvertis pagal scenarijų ir atskaitos etaloną")
            rows = []
            labels_full = [bool(p["ground_truth_label"]) for p in pair_results
                           if p.get("ground_truth_label") is not None]
            for p in pair_results:
                if p.get("ground_truth_label") is None:
                    continue
            # Baseline F1
            def _f1(preds, labels):
                tp = sum(1 for x, y in zip(preds, labels) if x and y)
                fp = sum(1 for x, y in zip(preds, labels) if x and not y)
                fn = sum(1 for x, y in zip(preds, labels) if not x and y)
                if not tp:
                    return 0.0
                pr = tp / (tp + fp); rc = tp / (tp + fn)
                return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0

            base_preds = [(p.get("score_baseline") or 0) >= threshold for p in pair_results
                          if p.get("ground_truth_label") is not None]
            f1_src_base = _f1(base_preds, labels_full)
            f1_src = {s: _f1([(p.get(f"score_{s}") or 0) >= threshold for p in pair_results
                              if p.get("ground_truth_label") is not None], labels_full) for s in scenarios}

            # Pasitelkti TF-IDF baseline'ų rezultatus, jei jie sugeneruoti
            tfidf_results = load_tfidf_baselines(selected_file.stem, selected_file.parent)
            f1_src_tfidf = None
            f1_jud_tfidf = None
            if tfidf_results:
                tfidf_preds_src, tfidf_labs_src = [], []
                for p in pair_results:
                    if p.get("ground_truth_label") is None:
                        continue
                    t = tfidf_results.get(p["pair_id"])
                    if t is None or t.get("score_tfidf_lr") is None:
                        continue
                    tfidf_preds_src.append(t["score_tfidf_lr"] >= threshold)
                    tfidf_labs_src.append(bool(p["ground_truth_label"]))
                f1_src_tfidf = _f1(tfidf_preds_src, tfidf_labs_src)

            f1_jud = {}
            f1_jud_base = 0.0
            if has_judge:
                preds_b, labs_b = [], []
                for p in pair_results:
                    v = judge.get(p["pair_id"]); jl = _judge_label(v) if v else None
                    if jl is None or p.get("score_baseline") is None:
                        continue
                    preds_b.append(p["score_baseline"] >= threshold); labs_b.append(bool(jl))
                f1_jud_base = _f1(preds_b, labs_b)
                for s in scenarios:
                    pr, lb = [], []
                    for p in pair_results:
                        v = judge.get(p["pair_id"]); jl = _judge_label(v) if v else None
                        if jl is None or p.get(f"score_{s}") is None:
                            continue
                        pr.append(p[f"score_{s}"] >= threshold); lb.append(bool(jl))
                    f1_jud[s] = _f1(pr, lb)
                if tfidf_results:
                    pr_t, lb_t = [], []
                    for p in pair_results:
                        v = judge.get(p["pair_id"]); jl = _judge_label(v) if v else None
                        t = tfidf_results.get(p["pair_id"])
                        if jl is None or t is None or t.get("score_tfidf_lr") is None:
                            continue
                        pr_t.append(t["score_tfidf_lr"] >= threshold); lb_t.append(bool(jl))
                    f1_jud_tfidf = _f1(pr_t, lb_t)

            # Stulpelių tvarka: TF-IDF + LR (jei yra) → Embedding → Scenarijai
            cat_names: list[str] = []
            src_values: list[float] = []
            jud_values: list[float] = []
            if f1_src_tfidf is not None:
                cat_names.append("TF-IDF + LogReg")
                src_values.append(f1_src_tfidf)
                jud_values.append(f1_jud_tfidf if f1_jud_tfidf is not None else 0.0)
            cat_names.append("Embedding pagrindas")
            src_values.append(f1_src_base)
            jud_values.append(f1_jud_base)
            for s in scenarios:
                cat_names.append(f"Scenarijus {s}")
                src_values.append(f1_src[s])
                jud_values.append(f1_jud.get(s, 0.0))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cat_names, y=src_values,
                name="Prieš šaltinio etiketę", marker_color="#1976D2",
                text=[f"{v:.3f}" for v in src_values],
                textposition="outside",
            ))
            if has_judge:
                fig.add_trace(go.Bar(
                    x=cat_names, y=jud_values,
                    name="Prieš teisėjo verdiktą", marker_color="#F57C00",
                    text=[f"{v:.3f}" for v in jud_values],
                    textposition="outside",
                ))
            fig.update_layout(
                barmode="group", height=460,
                yaxis_title="F1 įvertis", yaxis_range=[0, 0.85],
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- Grafikas 3: Sumaišties matricos ----------------
            st.subheader("3. Sumaišties matricos: scenarijai × atskaitos etalonai")
            cm_cols = st.columns(len(scenarios))
            for i, s in enumerate(scenarios):
                preds = [(p.get(f"score_{s}") or 0) >= threshold for p in pair_results
                         if p.get("ground_truth_label") is not None]
                cm_src = {
                    "tp": sum(1 for x, y in zip(preds, labels_full) if x and y),
                    "fp": sum(1 for x, y in zip(preds, labels_full) if x and not y),
                    "tn": sum(1 for x, y in zip(preds, labels_full) if not x and not y),
                    "fn": sum(1 for x, y in zip(preds, labels_full) if not x and y),
                }
                with cm_cols[i]:
                    st.plotly_chart(
                        confusion_matrix_figure(cm_src, f"Scen. {s} — šaltinis", color="Blues"),
                        use_container_width=True,
                    )
            if has_judge:
                cm_cols = st.columns(len(scenarios))
                for i, s in enumerate(scenarios):
                    preds, labs = [], []
                    for p in pair_results:
                        v = judge.get(p["pair_id"]); jl = _judge_label(v) if v else None
                        if jl is None or p.get(f"score_{s}") is None:
                            continue
                        preds.append(p[f"score_{s}"] >= threshold); labs.append(bool(jl))
                    cm_jud = {
                        "tp": sum(1 for x, y in zip(preds, labs) if x and y),
                        "fp": sum(1 for x, y in zip(preds, labs) if x and not y),
                        "tn": sum(1 for x, y in zip(preds, labs) if not x and not y),
                        "fn": sum(1 for x, y in zip(preds, labs) if not x and y),
                    }
                    with cm_cols[i]:
                        st.plotly_chart(
                            confusion_matrix_figure(cm_jud, f"Scen. {s} — teisėjas", color="Oranges"),
                            use_container_width=True,
                        )

            # ---------------- Grafikas 4: Ribos jautrumas ----------------
            st.subheader("4. Ribos jautrumas — F1 priklausomybė nuo sprendimo ribos")
            fig = threshold_sweep_chart(pair_results, scenarios)
            fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                          annotation_text=f"Naudota riba ({threshold:.0f})", annotation_position="top")
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- Grafikas 5: Klaidų taksonomija ----------------
            if has_judge:
                st.subheader("5. Klaidų taksonomijos pasiskirstymas (LLM-teisėjas)")
                from collections import Counter
                fm_counts = Counter()
                for v in judge.values():
                    fm = v.get("failure_mode")
                    if fm:
                        fm_counts[fm] += 1
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
                if items:
                    total_fm = sum(c for _, c in items)
                    fig = go.Figure(go.Bar(
                        x=[c for _, c in items],
                        y=[fm_lt.get(k, k) for k, _ in items],
                        orientation="h",
                        text=[f"{c} ({100*c/total_fm:.1f}%)" for _, c in items],
                        textposition="outside",
                        marker_color="#1976D2",
                    ))
                    fig.update_layout(
                        height=430,
                        xaxis_title="Porų skaičius",
                        margin=dict(l=20, r=20, t=20, b=20),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ---------------- Grafikas 6: Per-poros balų sklaida (tik kai yra dviejų scenarijų) ----------------
            if len(scenarios) >= 2:
                st.subheader("6. Per-poros balų sklaida")
                st.caption(
                    "Lyginami pirmieji du esami scenarijai. Kai dviejų scenarijų balų taškai "
                    "neguli ties įstrižaine — sistema nesutampa per-poros lygyje, net jei agreguotos metrikos panašios."
                )
                fig = score_scatter_chart(pair_results, scenarios[:2], threshold)
                st.plotly_chart(fig, use_container_width=True)

            # ---------------- Grafikas 7: Cohen kappa šilumos žemėlapis ----------------
            if has_judge:
                st.subheader("7. Cohen κ susitarimo koeficientai")
                rows_kappa = []
                for p in pair_results:
                    lab = p.get("ground_truth_label")
                    if lab is None:
                        continue
                    v = judge.get(p["pair_id"]); jl = _judge_label(v) if v else None
                    if jl is None:
                        continue
                    if any(p.get(f"score_{s}") is None for s in scenarios):
                        continue
                    rows_kappa.append({
                        "Šaltinis": bool(lab), "Teisėjas": bool(jl),
                        **{f"Sist. {s}": p[f"score_{s}"] >= threshold for s in scenarios},
                    })
                if rows_kappa:
                    keys = ["Šaltinis", "Teisėjas"] + [f"Sist. {s}" for s in scenarios]

                    def _kappa(xs, ys):
                        n = len(xs)
                        po = sum(1 for a, b in zip(xs, ys) if a == b) / n
                        pa = sum(xs) / n; pb = sum(ys) / n
                        pe = pa * pb + (1 - pa) * (1 - pb)
                        return (po - pe) / (1 - pe) if pe < 1 else 0.0

                    mat = [[1.0 if a == b else _kappa([r[a] for r in rows_kappa], [r[b] for r in rows_kappa])
                            for b in keys] for a in keys]
                    fig = go.Figure(go.Heatmap(
                        z=mat, x=keys, y=keys, colorscale="RdYlGn", zmin=-0.1, zmax=1.0,
                        text=[[f"{v:.3f}" for v in row] for row in mat],
                        texttemplate="%{text}", textfont={"size": 11},
                        colorbar=dict(title="κ"),
                    ))
                    fig.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

            # ---------------- Footer ----------------
            st.markdown("---")
            st.caption(
                "💡 Visos statinės PNG versijos su lietuviškomis etiketėmis (thesis/images/) gaminamos paleidus: "
                "`python scripts/generate_thesis_plots.py`. Tas pats skriptas naudoja "
                "`results/tier2_5000_combined/`, `results/judge_hf_test_5000.json` ir "
                "`results/qwen_1000_A/` rezultatų failus, todėl jie turi būti vietoje."
            )

# ---------------------------------------------------------------------------
# Browse Dataset mode — inspect any data/*.json without needing a result file
# ---------------------------------------------------------------------------
elif mode == "Browse Dataset":
    st.header("Browse Dataset")

    dataset_files = sorted(
        [f for f in DATA_DIR.glob("*.json") if not f.name.startswith("esco")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not dataset_files:
        st.warning("No dataset JSON files found in `data/`.")
        st.stop()

    selected = st.sidebar.selectbox("Dataset file", dataset_files, format_func=lambda p: p.name)
    raw = json.loads(selected.read_text(encoding="utf-8"))
    pairs = raw if isinstance(raw, list) else raw.get("pairs", [])

    if not pairs:
        st.warning("Selected file has no pairs.")
        st.stop()

    # KPI strip
    n = len(pairs)
    sel = sum(1 for p in pairs if p.get("ground_truth_label") is True)
    rej = sum(1 for p in pairs if p.get("ground_truth_label") is False)
    augmented = sum(1 for p in pairs if p.get("jd_augmented"))
    jd_lens = [len((p.get("jd_text") or "").split()) for p in pairs]
    cv_lens = [len((p.get("cv_text") or "").split()) for p in pairs]

    cols = st.columns(5)
    cols[0].metric("Pairs", n)
    cols[1].metric("Select", sel)
    cols[2].metric("Reject", rej)
    cols[3].metric("Augmented JDs", f"{augmented}", delta=f"{augmented / n:.0%}" if n else "")
    cols[4].metric("Median JD words", int(statistics.median(jd_lens)) if jd_lens else 0)

    st.markdown("### JD length distribution")
    fig_lens = px.histogram(
        {"JD word count": jd_lens, "CV word count": cv_lens},
        nbins=40, title=None,
    )
    fig_lens.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20),
                           xaxis_title="word count", yaxis_title="pairs")
    st.plotly_chart(fig_lens, use_container_width=True)

    st.markdown("### Pair browser")
    # Filter
    filt_cols = st.columns([2, 1, 1])
    role_filter = filt_cols[0].text_input("Filter by pair_id substring (e.g. 'qa_engineer')", "")
    label_filter = filt_cols[1].selectbox("Label", ["all", "select only", "reject only"], index=0)
    aug_filter = filt_cols[2].selectbox("Augmented", ["all", "augmented only", "original only"], index=0)

    filtered = pairs
    if role_filter:
        filtered = [p for p in filtered if role_filter.lower() in p.get("pair_id", "").lower()]
    if label_filter == "select only":
        filtered = [p for p in filtered if p.get("ground_truth_label") is True]
    elif label_filter == "reject only":
        filtered = [p for p in filtered if p.get("ground_truth_label") is False]
    if aug_filter == "augmented only":
        filtered = [p for p in filtered if p.get("jd_augmented")]
    elif aug_filter == "original only":
        filtered = [p for p in filtered if not p.get("jd_augmented")]

    st.caption(f"{len(filtered)} of {n} pairs match your filter")

    if filtered:
        sel_pid = st.selectbox(
            "Pick a pair",
            [p["pair_id"] for p in filtered],
            key="ds_pid",
        )
        pair = next(p for p in filtered if p["pair_id"] == sel_pid)

        # Top metadata
        m_cols = st.columns(4)
        gt = pair.get("ground_truth_label")
        m_cols[0].metric("Ground truth", "Match" if gt else "No-Match" if gt is False else "—")
        m_cols[1].metric("JD words", len((pair.get("jd_text") or "").split()))
        m_cols[2].metric("CV words", len((pair.get("cv_text") or "").split()))
        m_cols[3].metric("Augmented?", "Yes" if pair.get("jd_augmented") else "No")

        ref = pair.get("reference_reason")
        if ref:
            st.caption(
                f"Source-dataset reference reason (templated, not used for evaluation): {ref}"
            )

        # Side-by-side JD: original vs augmented (if applicable)
        if pair.get("jd_augmented") and pair.get("original_jd_text"):
            st.markdown("### JD: original vs augmented")
            jd_cols = st.columns(2)
            jd_cols[0].markdown("**Original (pre-augmentation)**")
            jd_cols[0].text(pair["original_jd_text"])
            jd_cols[1].markdown("**Augmented**")
            jd_cols[1].text(pair["jd_text"])
        else:
            st.markdown("### Job description")
            st.text(pair.get("jd_text", "—"))

        with st.expander("CV text"):
            st.text(pair.get("cv_text", "—"))


# ---------------------------------------------------------------------------
# Curate Gold Set mode — hand-pick CV-JD pairs into a thesis-grade gold dataset
# ---------------------------------------------------------------------------
elif mode == "Curate Gold Set":
    st.header("Curate Gold Set")
    st.markdown(
        "Hand-pick CV-JD pairs you trust into a separate gold evaluation set. "
        "**Keep** = pair belongs in your gold set. **Reject** = explicitly mark "
        "as not-trustworthy (won't show again as pending). **Skip** = decide later. "
        "Curation state persists in the target file across sessions."
    )

    # ----- Source dataset selector -----
    sources = sorted(
        [f for f in DATA_DIR.glob("*.json") if not f.name.startswith("esco")
         and not f.name.startswith("gold_")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not sources:
        st.warning("No source datasets in `data/`.")
        st.stop()

    src_path = st.sidebar.selectbox(
        "Source dataset", sources, format_func=lambda p: p.name, key="curate_src"
    )

    # ----- Target gold file (created on first save) -----
    default_target = "gold_curated.json"
    target_name = st.sidebar.text_input(
        "Gold set filename", default_target, key="curate_target_name",
    )
    if not target_name.endswith(".json"):
        target_name += ".json"
    if not target_name.startswith("gold_"):
        st.sidebar.caption("Filenames are conventionally prefixed `gold_` so they're easy to spot.")
    target_path = DATA_DIR / target_name

    # ----- Load source pairs -----
    raw = json.loads(src_path.read_text(encoding="utf-8"))
    src_pairs = raw if isinstance(raw, list) else raw.get("pairs", [])

    # ----- Load or initialize gold file -----
    def _load_or_init_gold(p: Path) -> dict:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {
            "_meta": {
                "source_dataset": str(src_path.name),
                "rejected_pair_ids": [],
                "curator_notes_per_pair": {},
            },
            "pairs": [],
        }

    gold = _load_or_init_gold(target_path)
    gold.setdefault("_meta", {})
    gold["_meta"].setdefault("rejected_pair_ids", [])
    gold["_meta"].setdefault("curator_notes_per_pair", {})
    gold.setdefault("pairs", [])

    kept_ids = {p["pair_id"] for p in gold["pairs"]}
    rejected_ids = set(gold["_meta"]["rejected_pair_ids"])

    # ----- Apply pending action from previous interaction -----
    pending = st.session_state.get("curate_pending_action")
    if pending:
        action, pid, notes, label_override = pending
        # Find the source pair (so we save full data, not just the id)
        src_lookup = {p["pair_id"]: p for p in src_pairs}
        pair_data = src_lookup.get(pid)

        if action == "keep" and pair_data is not None:
            if pid in rejected_ids:
                gold["_meta"]["rejected_pair_ids"].remove(pid)
                rejected_ids.discard(pid)
            # Remove any prior copy if re-curating with a different override
            gold["pairs"] = [p for p in gold["pairs"] if p["pair_id"] != pid]
            kept_ids.discard(pid)

            pair_copy = dict(pair_data)
            pair_copy["_curated_at"] = "added"
            # Apply label override if set
            if label_override is not None:
                pair_copy["_source_ground_truth_label"] = pair_data.get("ground_truth_label")
                pair_copy["ground_truth_label"] = label_override
                pair_copy["_label_overridden_by_curator"] = True
                gold["_meta"].setdefault("label_overrides", {})
                gold["_meta"]["label_overrides"][pid] = {
                    "source": pair_data.get("ground_truth_label"),
                    "curator": label_override,
                }
            # If curator wrote a note, replace the templated source reason
            # with the note. The runner detects this via _meta.has_curator_reasons
            # and passes the note into Tier 2 labeled memory (otherwise reasons
            # are stripped because the source's are templated noise).
            if notes:
                pair_copy["_source_reference_reason"] = pair_data.get("reference_reason", "")
                pair_copy["reference_reason"] = notes
                gold["_meta"]["has_curator_reasons"] = True
                gold["_meta"]["curator_notes_per_pair"][pid] = notes
            gold["pairs"].append(pair_copy)
            kept_ids.add(pid)

        elif action == "reject":
            if pid in kept_ids:
                gold["pairs"] = [p for p in gold["pairs"] if p["pair_id"] != pid]
                kept_ids.discard(pid)
            if pid not in rejected_ids:
                gold["_meta"]["rejected_pair_ids"].append(pid)
                rejected_ids.add(pid)
            gold["_meta"].get("label_overrides", {}).pop(pid, None)

        elif action == "undecide":
            if pid in kept_ids:
                gold["pairs"] = [p for p in gold["pairs"] if p["pair_id"] != pid]
                kept_ids.discard(pid)
            if pid in rejected_ids:
                gold["_meta"]["rejected_pair_ids"].remove(pid)
                rejected_ids.discard(pid)
            gold["_meta"]["curator_notes_per_pair"].pop(pid, None)
            gold["_meta"].get("label_overrides", {}).pop(pid, None)

        target_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
        st.session_state["curate_pending_action"] = None
        # don't rerun — the natural next render shows the updated state

    # ----- KPIs -----
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Source pairs", len(src_pairs))
    kpi_cols[1].metric("Kept", len(kept_ids))
    kpi_cols[2].metric("Rejected", len(rejected_ids))
    kpi_cols[3].metric("Pending", len(src_pairs) - len(kept_ids) - len(rejected_ids))

    if target_path.exists():
        st.caption(f"Saving to: `{target_path.relative_to(DATA_DIR.parent)}`")
    else:
        st.caption(f"Will create: `{target_path.relative_to(DATA_DIR.parent)}` on first Keep/Reject")

    # ----- Filters -----
    st.markdown("### Filters")
    f_cols = st.columns([2, 1, 1, 1])
    role_filter = f_cols[0].text_input("pair_id contains", "", key="curate_role_filter")
    label_filter = f_cols[1].selectbox("Label", ["all", "select only", "reject only"], key="curate_label_filter")
    show_filter = f_cols[2].selectbox("Show", ["pending", "kept", "rejected", "all"], key="curate_show_filter")
    aug_filter = f_cols[3].selectbox("Augmented", ["all", "yes", "no"], key="curate_aug_filter")

    filtered = src_pairs
    if role_filter:
        filtered = [p for p in filtered if role_filter.lower() in p.get("pair_id", "").lower()]
    if label_filter == "select only":
        filtered = [p for p in filtered if p.get("ground_truth_label") is True]
    elif label_filter == "reject only":
        filtered = [p for p in filtered if p.get("ground_truth_label") is False]
    if show_filter == "pending":
        filtered = [p for p in filtered if p["pair_id"] not in kept_ids and p["pair_id"] not in rejected_ids]
    elif show_filter == "kept":
        filtered = [p for p in filtered if p["pair_id"] in kept_ids]
    elif show_filter == "rejected":
        filtered = [p for p in filtered if p["pair_id"] in rejected_ids]
    if aug_filter == "yes":
        filtered = [p for p in filtered if p.get("jd_augmented")]
    elif aug_filter == "no":
        filtered = [p for p in filtered if not p.get("jd_augmented")]

    st.caption(f"{len(filtered)} pair(s) match your filter")

    if not filtered:
        st.success("Nothing left to curate matching this filter.")
    else:
        # ----- Index navigation -----
        nav_state_key = f"curate_idx::{src_path.name}::{target_name}::{label_filter}::{show_filter}::{aug_filter}::{role_filter}"
        if nav_state_key not in st.session_state:
            st.session_state[nav_state_key] = 0
        st.session_state[nav_state_key] = max(0, min(st.session_state[nav_state_key], len(filtered) - 1))
        idx = st.session_state[nav_state_key]
        pair = filtered[idx]
        pid = pair["pair_id"]

        def _curate_prev():
            cur = st.session_state[nav_state_key]
            st.session_state[nav_state_key] = max(0, cur - 1)

        def _curate_next():
            cur = st.session_state[nav_state_key]
            st.session_state[nav_state_key] = min(len(filtered) - 1, cur + 1)

        nav_cols = st.columns([1, 1, 4, 2])
        nav_cols[0].button("◀ Prev", on_click=_curate_prev, disabled=idx == 0,
                           key="curate_prev_btn", use_container_width=True)
        nav_cols[1].button("Skip ▶", on_click=_curate_next, disabled=idx >= len(filtered) - 1,
                           key="curate_skip_btn", use_container_width=True)
        nav_cols[2].markdown(f"**Pair {idx + 1} of {len(filtered)}** · `{pid}`")
        # Status badge
        if pid in kept_ids:
            badge_color, badge_label = "#27ae60", "✓ KEPT"
        elif pid in rejected_ids:
            badge_color, badge_label = "#e74c3c", "✗ REJECTED"
        else:
            badge_color, badge_label = "#95a5a6", "Pending"
        nav_cols[3].markdown(
            f"<div style='background:{badge_color};color:white;padding:8px 12px;"
            f"border-radius:4px;text-align:center;font-weight:bold'>{badge_label}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ----- Pair info -----
        info_cols = st.columns(4)
        gt = pair.get("ground_truth_label")
        info_cols[0].metric("Source label", "Match" if gt else "No-Match" if gt is False else "—")
        info_cols[1].metric("JD words", len((pair.get("jd_text") or "").split()))
        info_cols[2].metric("CV words", len((pair.get("cv_text") or "").split()))
        info_cols[3].metric("Augmented?", "Yes" if pair.get("jd_augmented") else "No")

        if pair.get("reference_reason"):
            st.caption(
                f"*Source reference reason (templated, not used for evaluation):* "
                f"{pair['reference_reason']}"
            )

        # Existing notes for this pair (if any)
        existing_notes = gold["_meta"]["curator_notes_per_pair"].get(pid, "")

        # ----- CV / JD side-by-side -----
        txt_cols = st.columns(2)
        txt_cols[0].markdown("#### CV")
        txt_cols[0].text_area(
            "cv-text", pair.get("cv_text", ""), height=420,
            key=f"cv_view::{pid}", label_visibility="collapsed",
        )
        txt_cols[1].markdown("#### Job Description")
        txt_cols[1].text_area(
            "jd-text", pair.get("jd_text", ""), height=420,
            key=f"jd_view::{pid}", label_visibility="collapsed",
        )
        if pair.get("jd_augmented") and pair.get("original_jd_text"):
            with txt_cols[1].expander("Show original (pre-augmentation) JD"):
                st.text(pair["original_jd_text"])

        # ----- Curator notes -----
        notes_input = st.text_input(
            "Curator notes (optional, saved with kept pair)",
            value=existing_notes, key=f"curate_notes::{pid}",
        )

        # ----- Label override -----
        # If you disagree with the source label (e.g., a clear match labeled
        # 'reject' because of nonsense reason), you can override it here.
        # Saved with the pair so the gold set carries YOUR judgment, with the
        # source label preserved in '_source_ground_truth_label' for audit.
        existing_override = gold["_meta"].get("label_overrides", {}).get(pid, {}).get("curator")
        override_default = "use source"
        if existing_override is True:
            override_default = "force MATCH"
        elif existing_override is False:
            override_default = "force NO-MATCH"
        override_choice = st.radio(
            "Override the source label?",
            ["use source", "force MATCH", "force NO-MATCH"],
            index=["use source", "force MATCH", "force NO-MATCH"].index(override_default),
            horizontal=True, key=f"curate_override::{pid}",
            help="Override only when the source label is clearly wrong (e.g. coherent CV+JD with templated rejection reason). Records both source and your override in the gold file.",
        )
        # Visually flag if override differs from source
        src_label = pair.get("ground_truth_label")
        if override_choice == "force MATCH" and src_label is not True:
            st.warning(f"⚠ You will save this pair with label **MATCH** (source label was: {('Match' if src_label else 'No-Match' if src_label is False else '—')}).")
        elif override_choice == "force NO-MATCH" and src_label is not False:
            st.warning(f"⚠ You will save this pair with label **NO-MATCH** (source label was: {('Match' if src_label else 'No-Match' if src_label is False else '—')}).")

        # ----- Action buttons -----
        st.markdown("### Decision")
        btn_cols = st.columns([1, 1, 1, 3])

        def _override_value():
            choice = st.session_state.get(f"curate_override::{pid}", "use source")
            if choice == "force MATCH":
                return True
            if choice == "force NO-MATCH":
                return False
            return None

        def _do_keep():
            st.session_state["curate_pending_action"] = (
                "keep", pid,
                st.session_state.get(f"curate_notes::{pid}", ""),
                _override_value(),
            )
            cur = st.session_state[nav_state_key]
            st.session_state[nav_state_key] = min(len(filtered) - 1, cur + 1)

        def _do_reject():
            st.session_state["curate_pending_action"] = ("reject", pid, "", None)
            cur = st.session_state[nav_state_key]
            st.session_state[nav_state_key] = min(len(filtered) - 1, cur + 1)

        def _do_undecide():
            st.session_state["curate_pending_action"] = ("undecide", pid, "", None)

        btn_cols[0].button(
            "✓ Keep", on_click=_do_keep, type="primary",
            key="curate_keep_btn", use_container_width=True,
        )
        btn_cols[1].button(
            "✗ Reject", on_click=_do_reject,
            key="curate_reject_btn", use_container_width=True,
        )
        if pid in kept_ids or pid in rejected_ids:
            btn_cols[2].button(
                "↺ Undecide", on_click=_do_undecide,
                key="curate_undecide_btn", use_container_width=True,
            )

    # ----- Kept-pairs review expander -----
    with st.expander(f"Review kept pairs ({len(kept_ids)})"):
        if not gold["pairs"]:
            st.write("No kept pairs yet.")
        else:
            overrides = gold["_meta"].get("label_overrides", {})
            review_rows = []
            for p in gold["pairs"]:
                pid_r = p["pair_id"]
                final_label = "Match" if p.get("ground_truth_label") else "No-Match"
                override_marker = ""
                if pid_r in overrides:
                    src_l = overrides[pid_r].get("source")
                    src_str = "Match" if src_l else "No-Match" if src_l is False else "—"
                    override_marker = f" (overridden, source was {src_str})"
                review_rows.append({
                    "pair_id": pid_r,
                    "label": final_label + override_marker,
                    "jd_words": len((p.get("jd_text") or "").split()),
                    "augmented": "Y" if p.get("jd_augmented") else "N",
                    "notes": gold["_meta"]["curator_notes_per_pair"].get(pid_r, ""),
                })
            st.dataframe(review_rows, use_container_width=True)
            n_overrides = sum(1 for p in gold["pairs"] if p.get("_label_overridden_by_curator"))
            if n_overrides:
                st.caption(f"{n_overrides} of {len(gold['pairs'])} kept pairs have curator-overridden labels.")


# ---------------------------------------------------------------------------
# Live Match mode
# ---------------------------------------------------------------------------
elif mode == "Live Match":
    st.header("Run a Live Match")
    st.markdown("Paste a CV and a JD; run a single match through the pipeline.")

    col_cv, col_jd = st.columns(2)
    with col_cv:
        cv_input = st.text_area("CV Text", height=300, placeholder="Paste CV…", key="live_cv")
    with col_jd:
        jd_input = st.text_area("Job Description", height=300, placeholder="Paste JD…", key="live_jd")

    opt_cols = st.columns(3)
    with opt_cols[0]:
        live_scenario = st.selectbox("Scenario", ["A", "B", "C"], index=0, key="live_sc")
    with opt_cols[1]:
        use_real = st.checkbox("Use real LLM (requires API key)", value=False, key="live_real")
    with opt_cols[2]:
        st.write("")

    if st.button("Run Match", type="primary", disabled=not (cv_input and jd_input), key="live_btn"):
        with st.spinner("Running matching pipeline…"):
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from llm.client import LLMClient, MockLLMClient
                from memory.store import MemoryStore
                from orchestrator.orchestrator import Orchestrator

                if use_real:
                    from config.settings import settings
                    issues = settings.validate()
                    if issues:
                        st.error(f"Config issues: {', '.join(issues)}")
                        st.stop()
                    llm_client = LLMClient()
                else:
                    llm_client = MockLLMClient()

                memory_store = None
                if live_scenario == "C":
                    import tempfile
                    memory_store = MemoryStore(memory_dir=tempfile.mkdtemp(prefix="dashboard_mem_"))

                orchestrator = Orchestrator(llm_client=llm_client, memory_store=memory_store)
                context = orchestrator.run(cv_input, jd_input, scenario=live_scenario)

                st.success("Match complete.")

                if context.final_decision:
                    d = context.final_decision
                    res_cols = st.columns(4)
                    res_cols[0].metric("Score", f"{d.score}/100")
                    res_cols[1].metric("Confidence", f"{d.confidence:.0%}")
                    rec_color = RECOMMENDATION_COLORS.get(d.recommendation, "#666")
                    res_cols[2].markdown(
                        f"**Recommendation**<br>"
                        f"<span style='color:{rec_color};font-size:1.5em;font-weight:bold'>"
                        f"{d.recommendation}</span>",
                        unsafe_allow_html=True,
                    )
                    res_cols[3].metric("Revisions", context.revision_count)

                    st.markdown(f"**Explanation:** {d.explanation}")
                    if d.key_factors:
                        st.markdown("**Key factors:**")
                        for f in d.key_factors:
                            st.markdown(f"- {f}")

                if context.cv_entities and context.jd_entities:
                    st.markdown("---")
                    sk_cols = st.columns(2)
                    with sk_cols[0]:
                        st.markdown("**CV Skills**")
                        st.write(", ".join(context.cv_entities.skills))
                    with sk_cols[1]:
                        st.markdown("**JD Skills**")
                        st.write(", ".join(context.jd_entities.skills))

                if context.similarity_scores:
                    st.markdown("---")
                    s = context.similarity_scores
                    sim_cols = st.columns(3)
                    sim_cols[0].metric("Semantic Similarity", f"{s.overall_score:.3f}")
                    sim_cols[1].metric("Skill Coverage", f"{s.coverage_ratio:.0%}")
                    sim_cols[2].metric("Matched Skills", f"{s.matched_skills_count}/{s.total_jd_skills}")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
