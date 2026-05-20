"""
Evaluation metrics for comparing scenario quality.

These metrics are designed for the thesis evaluation chapter.
They measure different aspects of the agent system's performance
across Scenarios A, B, and C.

Metric categories:
1. Score consistency  — How stable are scores across runs?
2. Execution profile  — Which agents ran, how long, how many revisions?
3. Enrichment impact  — Did ESCO normalization change matching outcomes?
4. Memory impact      — Did historical context change reasoning?
5. Reflection quality — Did the reflection loop improve the analysis?

All metrics operate on SharedContext objects produced by the orchestrator.
They do NOT require ground-truth labels (this is unsupervised evaluation).
For supervised evaluation with human labels, see ExperimentRunner.
"""

from dataclasses import dataclass, field
from typing import Optional

from models.shared_context import SharedContext


# ── Classification metrics ───────────────────────────────────

DEFAULT_MATCH_THRESHOLD = 50.0  # score >= 50 → "match", < 50 → "no_match"


@dataclass
class ClassificationMetrics:
    """
    Standard classification metrics computed from score-based predictions
    vs binary ground-truth labels.

    The system produces continuous scores (0-100), but for thesis evaluation
    we also need binary classification metrics. We convert scores to
    match/no-match using a threshold (default: 50), then compute standard
    metrics against human-assigned labels.

    This is standard practice in IR/NLP evaluation when the system output
    is continuous but the evaluation criteria are binary.
    """

    threshold: float = DEFAULT_MATCH_THRESHOLD
    total: int = 0
    true_positives: int = 0   # predicted match, actually match
    false_positives: int = 0  # predicted match, actually no-match
    true_negatives: int = 0   # predicted no-match, actually no-match
    false_negatives: int = 0  # predicted no-match, actually match

    @property
    def accuracy(self) -> float:
        """Fraction of correct predictions."""
        if self.total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / self.total

    @property
    def precision(self) -> float:
        """Of all predicted matches, how many were correct."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Of all actual matches, how many were predicted correctly."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_classification_metrics(
    predictions: list[tuple[float, bool]],
    threshold: float = DEFAULT_MATCH_THRESHOLD,
) -> ClassificationMetrics:
    """
    Compute classification metrics from (predicted_score, ground_truth_is_match) pairs.

    Args:
        predictions: List of (system_score, human_label_is_match) tuples.
                     system_score is 0-100, human_label_is_match is True/False.
        threshold:   Score >= threshold is predicted as "match".

    Returns:
        ClassificationMetrics with all counts and derived metrics.
    """
    m = ClassificationMetrics(threshold=threshold, total=len(predictions))

    for score, is_match in predictions:
        predicted_match = score >= threshold

        if predicted_match and is_match:
            m.true_positives += 1
        elif predicted_match and not is_match:
            m.false_positives += 1
        elif not predicted_match and not is_match:
            m.true_negatives += 1
        else:  # not predicted_match and is_match
            m.false_negatives += 1

    return m


@dataclass
class ScenarioMetrics:
    """Metrics computed from a single scenario run."""

    # Identification
    scenario: str = ""
    pair_id: str = ""

    # Score metrics
    final_score: float = 0.0
    confidence: float = 0.0
    recommendation: str = ""
    reasoning_suggested_score: float = 0.0
    score_delta: float = 0.0  # |final_score - reasoning_suggested_score|

    # Execution profile
    total_agents_executed: int = 0
    revision_count: int = 0
    total_log_entries: int = 0
    execution_duration_seconds: float = 0.0

    # Matching metrics
    similarity_overall: float = 0.0
    coverage_ratio: float = 0.0
    matched_skills: int = 0
    total_jd_skills: int = 0

    # Enrichment metrics (Scenario B+)
    has_enrichment: bool = False
    cv_skills_normalized: int = 0
    jd_skills_normalized: int = 0
    taxonomy_mapped_count: int = 0  # Skills with ESCO codes

    # Memory metrics (Scenario C)
    has_memory: bool = False
    memories_retrieved: int = 0
    best_memory_similarity: float = 0.0

    # Reflection metrics
    reflection_triggered: bool = False
    reflection_confidence: float = 0.0
    reflection_issues_count: int = 0
    was_revised: bool = False

    # Calibrated confidence: decision_confidence × reflection_confidence
    # Based on LLM-as-a-Judge dual-score method (Gu et al., 2024)
    calibrated_confidence: float = 0.0


def compute_metrics(
    context: SharedContext,
    pair_id: str = "",
) -> ScenarioMetrics:
    """
    Compute all metrics from a completed SharedContext.

    Args:
        context: The completed shared context from an orchestrator run
        pair_id: An identifier for the CV-JD pair (for cross-pair comparison)

    Returns:
        ScenarioMetrics with all fields populated
    """
    m = ScenarioMetrics(
        scenario=context.scenario,
        pair_id=pair_id,
        total_log_entries=len(context.logs),
        revision_count=context.revision_count,
    )

    # Score metrics
    if context.final_decision:
        d = context.final_decision
        m.final_score = d.score
        m.confidence = d.confidence
        m.recommendation = d.recommendation

    if context.reasoning_output:
        m.reasoning_suggested_score = context.reasoning_output.suggested_score

    if context.final_decision and context.reasoning_output:
        m.score_delta = abs(
            context.final_decision.score
            - context.reasoning_output.suggested_score
        )

    # Execution profile
    agent_starts = [
        log for log in context.logs if log.action == "started"
    ]
    m.total_agents_executed = len(agent_starts)

    # Extract total duration from the orchestrator's run_completed log
    for log in context.logs:
        if log.agent_name == "Orchestrator" and log.action == "run_completed":
            m.execution_duration_seconds = log.duration_seconds
            break

    # Matching metrics
    if context.similarity_scores:
        s = context.similarity_scores
        m.similarity_overall = s.overall_score
        m.coverage_ratio = s.coverage_ratio
        m.matched_skills = s.matched_skills_count
        m.total_jd_skills = s.total_jd_skills

    # Enrichment metrics
    m.has_enrichment = context.has_enrichment()
    if context.normalized_entities:
        ne = context.normalized_entities
        m.cv_skills_normalized = len(ne.cv_skills)
        m.jd_skills_normalized = len(ne.jd_skills)
        m.taxonomy_mapped_count = sum(
            1 for s in ne.cv_skills + ne.jd_skills
            if s.esco_code is not None
        )

    # Memory metrics
    m.has_memory = context.has_memory()
    if context.memory_entries:
        m.memories_retrieved = len(context.memory_entries)
        m.best_memory_similarity = max(
            mem.similarity_to_current for mem in context.memory_entries
        )

    # Reflection metrics
    if context.reflection_output:
        ref = context.reflection_output
        m.reflection_triggered = True
        m.reflection_confidence = ref.confidence
        m.reflection_issues_count = len(ref.issues_found)
        m.was_revised = context.revision_count > 0

    # Calibrated confidence (LLM-as-a-Judge dual-score approach):
    # Combines decision agent's self-assessed confidence with the
    # reflection agent's independent quality assessment.
    if context.final_decision and context.reflection_output:
        m.calibrated_confidence = m.confidence * m.reflection_confidence

    return m


@dataclass
class ComparisonReport:
    """
    Comparison of metrics across scenarios for one or more CV-JD pairs.

    This is the primary output for the thesis evaluation chapter.
    """

    pair_results: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)  # per-scenario ClassificationMetrics

    def add_pair(
        self,
        pair_id: str,
        metrics_a: Optional[ScenarioMetrics] = None,
        metrics_b: Optional[ScenarioMetrics] = None,
        metrics_c: Optional[ScenarioMetrics] = None,
        ground_truth_score: Optional[float] = None,
        ground_truth_label: Optional[bool] = None,
        baseline_score: Optional[float] = None,
    ) -> None:
        """
        Add a CV-JD pair's results across all scenarios.

        Args:
            ground_truth_score: Human-assigned score 0-100 (for error metrics)
            ground_truth_label: Human-assigned binary label True=match, False=no-match
                                (for precision/recall/F1). If not provided but
                                ground_truth_score is, label is derived from score >= 50.
            baseline_score: Score from embedding-only baseline (0-100), if computed.
        """
        entry = {
            "pair_id": pair_id,
            "ground_truth": ground_truth_score,
        }

        # Store baseline score if available
        if baseline_score is not None:
            entry["score_baseline"] = baseline_score

        # Derive binary label from score if not explicitly provided
        if ground_truth_label is not None:
            entry["ground_truth_label"] = ground_truth_label
        elif ground_truth_score is not None:
            entry["ground_truth_label"] = ground_truth_score >= DEFAULT_MATCH_THRESHOLD

        for label, m in [("A", metrics_a), ("B", metrics_b), ("C", metrics_c)]:
            if m is not None:
                entry[f"score_{label}"] = m.final_score
                entry[f"confidence_{label}"] = m.confidence
                entry[f"recommendation_{label}"] = m.recommendation
                entry[f"suggested_{label}"] = m.reasoning_suggested_score
                entry[f"delta_{label}"] = m.score_delta
                entry[f"similarity_{label}"] = m.similarity_overall
                entry[f"coverage_{label}"] = m.coverage_ratio
                entry[f"revisions_{label}"] = m.revision_count
                entry[f"duration_{label}"] = m.execution_duration_seconds
                entry[f"agents_{label}"] = m.total_agents_executed
                entry[f"reflection_confidence_{label}"] = m.reflection_confidence
                entry[f"calibrated_confidence_{label}"] = m.calibrated_confidence

        # Compute accuracy vs ground truth if available
        if ground_truth_score is not None:
            for label, m in [("A", metrics_a), ("B", metrics_b), ("C", metrics_c)]:
                if m is not None:
                    entry[f"error_{label}"] = abs(m.final_score - ground_truth_score)

        self.pair_results.append(entry)

    def compute_summary(self, threshold: float = DEFAULT_MATCH_THRESHOLD) -> dict:
        """
        Compute aggregate statistics across all pairs.

        Calculates mean/std for scores, errors, durations, etc.
        This is what goes into the thesis results tables.

        Args:
            threshold: Score >= threshold is classified as "match" for
                       binary classification metrics. Fixed across all
                       scenarios for fair comparison.
        """
        if not self.pair_results:
            self.summary = {}
            return self.summary

        scenarios = ["A", "B", "C"]
        summary = {}

        for label in scenarios:
            scores = [
                r[f"score_{label}"] for r in self.pair_results
                if f"score_{label}" in r
            ]
            if not scores:
                continue

            confidences = [
                r.get(f"confidence_{label}", 0) for r in self.pair_results
                if f"confidence_{label}" in r
            ]
            calibrated_confs = [
                r.get(f"calibrated_confidence_{label}", 0) for r in self.pair_results
                if f"calibrated_confidence_{label}" in r
            ]
            durations = [
                r.get(f"duration_{label}", 0) for r in self.pair_results
                if f"duration_{label}" in r
            ]
            deltas = [
                r.get(f"delta_{label}", 0) for r in self.pair_results
                if f"delta_{label}" in r
            ]
            errors = [
                r[f"error_{label}"] for r in self.pair_results
                if f"error_{label}" in r
            ]
            revisions = [
                r.get(f"revisions_{label}", 0) for r in self.pair_results
                if f"revisions_{label}" in r
            ]

            revised_count = sum(1 for r in revisions if r > 0)

            summary[label] = {
                "n_pairs": len(scores),
                "mean_score": _mean(scores),
                "std_score": _std(scores),
                "mean_confidence": _mean(confidences),
                "mean_duration_s": _mean(durations),
                "mean_score_delta": _mean(deltas),
                "mean_revisions": _mean(revisions),
                "revision_rate": revised_count / len(scores) if scores else 0.0,
                "max_revisions": max(revisions) if revisions else 0,
                "mean_calibrated_confidence": _mean(calibrated_confs),
            }

            if errors:
                summary[label]["mean_error"] = _mean(errors)
                summary[label]["std_error"] = _std(errors)

        self.summary = summary

        # Compute classification metrics per scenario (and baseline if present)
        self.classification = {}
        all_labels = scenarios.copy()

        # Include baseline if any pair has a baseline score
        if any("score_baseline" in r for r in self.pair_results):
            all_labels.append("baseline")

        for label in all_labels:
            predictions = []
            for r in self.pair_results:
                if f"score_{label}" in r and "ground_truth_label" in r:
                    predictions.append(
                        (r[f"score_{label}"], r["ground_truth_label"])
                    )
            if predictions:
                self.classification[label] = compute_classification_metrics(
                    predictions, threshold=threshold,
                )

        return summary


def format_comparison_table(report: ComparisonReport) -> str:
    """
    Format the comparison report as an ASCII table for terminal output.

    Produces a table suitable for the thesis and for quick review
    in the terminal.
    """
    lines = []
    lines.append("")
    lines.append("=" * 90)
    lines.append("  SCENARIO COMPARISON REPORT")
    lines.append("=" * 90)

    # Per-pair results
    if report.pair_results:
        lines.append("")
        header = (
            f"  {'Pair':<12s} | {'Sc.':<3s} | {'Score':>6s} | {'Conf.':>5s} | "
            f"{'Sugg.':>6s} | {'Delta':>5s} | {'Cov.':>5s} | "
            f"{'Rev.':>4s} | {'Time':>6s} | {'Recommendation':<16s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 86)

        for pair in report.pair_results:
            pair_id = pair["pair_id"]
            gt = pair.get("ground_truth")
            gt_str = f" (GT: {gt:.0f})" if gt is not None else ""

            for label in ["A", "B", "C"]:
                if f"score_{label}" not in pair:
                    continue

                row = (
                    f"  {pair_id:<12s} | {label:>3s} | "
                    f"{pair[f'score_{label}']:>6.1f} | "
                    f"{pair.get(f'confidence_{label}', 0):>5.0%} | "
                    f"{pair.get(f'suggested_{label}', 0):>6.1f} | "
                    f"{pair.get(f'delta_{label}', 0):>5.1f} | "
                    f"{pair.get(f'coverage_{label}', 0):>5.0%} | "
                    f"{pair.get(f'revisions_{label}', 0):>4d} | "
                    f"{pair.get(f'duration_{label}', 0):>5.2f}s | "
                    f"{pair.get(f'recommendation_{label}', ''):16s}"
                )
                lines.append(row)

            if gt is not None:
                errors = []
                for label in ["A", "B", "C"]:
                    if f"error_{label}" in pair:
                        errors.append(f"{label}={pair[f'error_{label}']:.1f}")
                if errors:
                    lines.append(f"  {'':12s}   GT={gt:.0f}, errors: {', '.join(errors)}")

            lines.append("  " + "-" * 86)

    # Summary statistics
    if report.summary:
        lines.append("")
        lines.append("  SUMMARY STATISTICS")
        lines.append("  " + "-" * 60)

        header = (
            f"  {'Scenario':<10s} | {'Mean Score':>10s} | {'Std':>5s} | "
            f"{'Conf.':>5s} | {'Cal.Conf.':>9s} | "
            f"{'Mean Delta':>10s} | {'Mean Rev.':>9s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 70)

        for label in ["A", "B", "C"]:
            if label not in report.summary:
                continue
            s = report.summary[label]
            row = (
                f"  {label:<10s} | "
                f"{s['mean_score']:>10.1f} | "
                f"{s['std_score']:>5.1f} | "
                f"{s['mean_confidence']:>5.0%} | "
                f"{s['mean_calibrated_confidence']:>9.0%} | "
                f"{s['mean_score_delta']:>10.1f} | "
                f"{s['mean_revisions']:>9.1f}"
            )
            lines.append(row)

        # Error vs ground truth (if available)
        has_errors = any("mean_error" in report.summary.get(l, {}) for l in ["A", "B", "C"])
        if has_errors:
            lines.append("")
            lines.append("  ACCURACY vs GROUND TRUTH")
            lines.append("  " + "-" * 40)
            for label in ["A", "B", "C"]:
                if label in report.summary and "mean_error" in report.summary[label]:
                    s = report.summary[label]
                    lines.append(
                        f"  {label:<10s} | Mean Error: {s['mean_error']:>5.1f} "
                        f"(std: {s['std_error']:>.1f})"
                    )

    # Classification metrics (precision, recall, F1)
    if report.classification:
        # Determine threshold from first available ClassificationMetrics
        first_cm = next(iter(report.classification.values()))
        thr = first_cm.threshold

        lines.append("")
        lines.append(f"  CLASSIFICATION METRICS (threshold: score >= {thr:.0f} = match)")
        lines.append("  " + "-" * 70)
        header = (
            f"  {'Scenario':<10s} | {'Acc.':>6s} | {'Prec.':>6s} | "
            f"{'Recall':>6s} | {'F1':>6s} | "
            f"{'TP':>3s} | {'FP':>3s} | {'TN':>3s} | {'FN':>3s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 70)

        for label in ["baseline", "A", "B", "C"]:
            if label not in report.classification:
                continue
            c = report.classification[label]
            display_name = "Baseline" if label == "baseline" else label
            row = (
                f"  {display_name:<10s} | "
                f"{c.accuracy:>6.0%} | "
                f"{c.precision:>6.0%} | "
                f"{c.recall:>6.0%} | "
                f"{c.f1:>6.0%} | "
                f"{c.true_positives:>3d} | "
                f"{c.false_positives:>3d} | "
                f"{c.true_negatives:>3d} | "
                f"{c.false_negatives:>3d}"
            )
            lines.append(row)

    # Reflection loop statistics
    lines.append("")
    lines.append("  REFLECTION LOOP STATISTICS")
    lines.append("  " + "-" * 60)
    for label in ["A", "B", "C"]:
        if label not in report.summary:
            continue
        s = report.summary[label]
        n = s["n_pairs"]
        rev_rate = s.get("revision_rate", 0.0)
        mean_rev = s.get("mean_revisions", 0.0)
        lines.append(
            f"  {label:<10s} | "
            f"Revision rate: {rev_rate:>5.0%} | "
            f"Mean revisions: {mean_rev:.2f} | "
            f"Pairs: {n}"
        )

    lines.append("")
    lines.append("=" * 90)
    return "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5
